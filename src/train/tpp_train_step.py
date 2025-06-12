import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import  log_metrics
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer,scheduler, device):
    """Epoch operation in training phase."""
    import numpy as np
    from tqdm import tqdm

    model.train()

    total_loss = 0  # cumulative event log-likelihood
    total_time_se = 0   # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct type predictions
    total_num_event = 0  # number of total non-pad events
    total_num_pred = 0   # total number of predictions for time RMSE

    for batch in tqdm(data_loader, desc='Training'):
        batch = batch.to(device)  # Move the entire batch to device
        # Move data to device
       
        label_dtime = batch[:,1:].inter_times.to(device)
        label_type = batch[:,1:].type_seq.to(device)
        pad_mask = label_type != model.pad_token_id  # assume model has pad_token_id

        # Forward
        optimizer.zero_grad()
        pred_dtime, pred_type = None, None
        pred_dtime, pred_type = model.predict_one_step_at_every_event(batch)

        loss, num_event = model.log_likelihood(batch)
        loss.backward()
        optimizer.step()
        step_scheduler(scheduler, event='batch')

        # Logging
        total_loss += loss.item()
        total_num_event += num_event

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




def test(data_loader, model, criterion, device):   
    """Epoch operation in testing phase."""
    import numpy as np
    from tqdm import tqdm

    model.eval()

    total_loss = 0
    total_time_se = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            batch = batch.to(device)

            # 标签截断，避免对最后一个 event 预测下一个（不存在）
            label_dtime = batch[:, 1:].inter_times.to(device)
            label_type = batch[:, 1:].type_seq.to(device)
            pad_mask = label_type != model.pad_token_id

            # 预测下一事件时间和类型
            pred_dtime, pred_type = model.predict_one_step_at_every_event(batch)
            loss, num_event = model.log_likelihood(batch)

            total_loss += loss.item()
            total_num_event += num_event

            # 类型预测准确率
            if pred_type is not None:
                correct = (pred_type == label_type) & pad_mask
                total_event_rate += correct.sum().item()

            # 时间预测 RMSE
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
    log_metrics(metrics, prefix="Testing")

    return avg_loss, metrics
