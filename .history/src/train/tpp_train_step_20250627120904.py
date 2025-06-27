import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import  log_metrics
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer, scheduler, device, accumulation_steps=2):
    """Epoch operation in training phase."""
    import numpy as np
    from tqdm import tqdm

    model.train()

    total_loss = 0  # cumulative event log-likelihood
    total_time_se = 0   # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct type predictions
    total_num_event = 0  # number of total non-pad events
    total_num_pred = 0   # total number of predictions for time RMSE

    optimizer.zero_grad()  # Only call once at the beginning, to initialize gradients
    step_count = 0  # 跟踪已处理的步骤数
    for batch in tqdm(data_loader, desc='Training'):
        batch = batch.to(device)  # Move the entire batch to device
        label_dtime = batch[:, 1:].inter_times.to(device)
        label_type = batch[:, 1:].type_seq.to(device)
        pad_mask = label_type != model.pad_token_id  # assume model has pad_token_id

        # Forward
        pred_dtime, pred_type = model.predict_one_step_at_every_event(batch)

        loss, num_event = model.log_likelihood(batch)
        loss = loss/accumulation_steps  
        loss.backward()  # Accumulate gradients

        # === Type prediction accuracy ===
        if pred_type is not None:
            correct = (pred_type == label_type) & pad_mask
            total_event_rate += correct.sum().item()

        # === Time prediction RMSE ===
        if pred_dtime is not None:
            # Ensure pred_dtime shape matches label_dtime
            time_se = ((pred_dtime - label_dtime) ** 2)[pad_mask]
            total_time_se += time_se.sum().item()
            total_num_pred += pad_mask.sum().item()

        # Accumulate loss
        total_loss += loss.item() * accumulation_steps
        total_num_event += num_event

        # If accumulation_steps have been completed, update the model
        step_count += 1
        if  step_count % accumulation_steps == 0 or  step_count  == len(data_loader):  # Update after every 'accumulation_steps' batches
            optimizer.step()  # Perform parameter update
            optimizer.zero_grad()  # Clear gradients for the next accumulation
            step_scheduler(scheduler, event='batch')  # Update scheduler

    avg_loss = total_loss / total_num_event if total_num_event > 0 else 0
    type_acc = total_event_rate / total_num_event if total_num_event > 0 else 0
    rmse = np.sqrt(total_time_se / total_num_pred) if total_num_pred > 0 else 0
    metrics = {
        'avg_event_ll': -avg_loss,
        'type_acc': type_acc,
        'rmse': rmse
    }
    log_metrics(metrics, prefix="Training")

    return avg_loss, metrics


def validate(data_loader, model, criterion, device):
    model.eval()

    total_loss = 0
    total_time_se = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating'):
            batch = batch.to(device)
            label_dtime = batch[:, 1:].inter_times.to(device)
            label_type = batch[:, 1:].type_seq.to(device)
            pad_mask = label_type != model.pad_token_id
            
            pred_dtime, pred_type = None, None
            pred_dtime, pred_type = model.predict_one_step_at_every_event(batch)
            loss, num_event = model.log_likelihood(batch)

            total_loss += loss.item()
            total_num_event += num_event

            if pred_type is not None:
                correct = (pred_type == label_type) & pad_mask
                total_event_rate += correct.sum().item()

            if pred_dtime is not None:
                time_se = ((pred_dtime - label_dtime) ** 2)[pad_mask]
                total_time_se += time_se.sum().item()
                total_num_pred += pad_mask.sum().item()


    avg_loss = total_loss / total_num_event if total_num_event > 0 else 0
    type_acc = total_event_rate / total_num_event if total_num_event > 0 else 0
    rmse = np.sqrt(total_time_se / total_num_pred) if total_num_pred > 0 else 0
    metrics = {
        'avg_event_ll': -avg_loss,
        'type_acc': type_acc,
        'rmse': rmse
    }
    log_metrics(metrics, prefix="Validation")
    return avg_loss, metrics





def test(data_loader, model, criterion, device, save_dir=None):   
    from collections import defaultdict

    model.eval()

    total_loss = 0
    total_time_se = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    all_label_dtime = []
    all_pred_dtime = []

    true_type_count = defaultdict(int)
    pred_type_count = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            batch = batch.to(device)

            label_dtime = batch[:, 1:].inter_times.to(device)
            label_type = batch[:, 1:].type_seq.to(device)
            pad_mask = label_type != model.pad_token_id

            pred_dtime, pred_type = model.predict_one_step_at_every_event(batch)
            loss, num_event = model.log_likelihood(batch)

            total_loss += loss.item()
            total_num_event += num_event

            if pred_type is not None:
                correct = (pred_type == label_type) & pad_mask
                total_event_rate += correct.sum().item()

                # 收集类别统计
                true_type_ids = label_type[pad_mask].view(-1).tolist()
                pred_type_ids = pred_type[pad_mask].view(-1).tolist()

                for t in true_type_ids:
                    true_type_count[t] += 1
                for t in pred_type_ids:
                    pred_type_count[t] += 1

            if pred_dtime is not None:
                time_se = ((pred_dtime - label_dtime) ** 2)[pad_mask]
                total_time_se += time_se.sum().item()
                total_num_pred += pad_mask.sum().item()

                all_label_dtime.append(label_dtime[pad_mask].cpu())
                all_pred_dtime.append(pred_dtime[pad_mask].cpu())

    # 汇总结果
    all_label_dtime = torch.cat(all_label_dtime).numpy()
    all_pred_dtime = torch.cat(all_pred_dtime).numpy()

    avg_loss = total_loss / total_num_event if total_num_event > 0 else 0
    type_acc = total_event_rate / total_num_event if total_num_event > 0 else 0
    rmse = np.sqrt(total_time_se / total_num_pred) if total_num_pred > 0 else 0

    metrics = {
        'avg_event_ll': -avg_loss,
        'type_acc': type_acc,
        'rmse': rmse
    }
    log_metrics(metrics, prefix="Testing")
    
    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 1. 实际 vs 预测散点图
        plt.figure()
        plt.scatter(all_label_dtime, all_pred_dtime, alpha=0.5, s=10)
        plt.xlabel("True Delta Time")
        plt.ylabel("Predicted Delta Time")
        plt.title("Predicted vs True Delta Time")
        plt.plot([min(all_label_dtime), max(all_label_dtime)],
                 [min(all_label_dtime), max(all_label_dtime)], 'r--')
        plt.savefig(os.path.join(save_dir, "scatter_pred_vs_true.png"))
        plt.close()

        # 2. 误差直方图
        error = all_pred_dtime - all_label_dtime
        plt.figure()
        plt.hist(error, bins=50, alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title("Prediction Error Distribution")
        plt.savefig(os.path.join(save_dir, "error_histogram.png"))
        plt.close()

        # 3. 实际类别与预测类别分布图
        all_types = sorted(set(true_type_count.keys()) | set(pred_type_count.keys()))
        true_counts = [true_type_count[t] for t in all_types]
        pred_counts = [pred_type_count[t] for t in all_types]

        x = np.arange(len(all_types))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, true_counts, width, label='True')
        plt.bar(x + width/2, pred_counts, width, label='Predicted')
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        plt.title("True vs Predicted Event Type Distribution")
        plt.xticks(x, [str(t) for t in all_types])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "event_type_distribution.png"))
        plt.close()

        # 4. 保存预测与标签为 .npz 文件
        np.savez(os.path.join(save_dir, "pred_results.npz"),
                 label_dtime=all_label_dtime,
                 pred_dtime=all_pred_dtime)

    return avg_loss, metrics

