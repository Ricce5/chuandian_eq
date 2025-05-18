import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import binom
from src.utils.metrics import compute_metrics, log_metrics, plot_and_save_roc_curve
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer,scheduler, device, threshold=0.5):
    model.train()
    total_loss = 0
    all_node_preds = []  # 存储所有预测值
    all_node_targets = []  # 存储所有目标值

    for batch, (x, y) in enumerate(tqdm(data_loader, desc="Training")):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        step_scheduler(scheduler, event='batch')
        total_loss += loss.item()

        # 收集所有预测和目标值
        all_node_preds.append(pred.cpu().detach().numpy())
        all_node_targets.append(y.cpu().detach().numpy())

        if batch % 100 == 0:
            tqdm.write(f"Batch {batch:>5d}/{len(data_loader):>5d} | Loss: {loss.item():.6f}")

  
    all_node_preds = np.concatenate(all_node_preds, axis=0)
    all_node_targets = np.concatenate(all_node_targets, axis=0)


    metrics = compute_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Training")
    avg_train_loss = total_loss / len(data_loader)
    
    return avg_train_loss, metrics

def validate(data_loader, model, criterion, device, threshold=0.5):
    model.eval()
    val_loss = 0

    all_node_preds = []  # 存储所有区域的预测值
    all_node_targets = []  # 存储所有区域的真实值

    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(data_loader, desc="Validating")):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

            # 将所有预测值和真实值添加到相应的列表中
            all_node_preds.extend(pred.cpu().detach().numpy())
            all_node_targets.extend(y.cpu().detach().numpy())

    # 将所有预测值和真实值合并为一个大的数组
    all_node_preds = np.array(all_node_preds)
    all_node_targets = np.array(all_node_targets)

    metrics = compute_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Validation")

    avg_val_loss = val_loss / len(data_loader)
  
    return avg_val_loss, metrics

def test(data_loader, model, criterion, device, threshold=0.5, save_dir=None):
    
    model.eval()
    test_loss = 0

    all_node_preds = []  # Store all predicted values
    all_node_targets = []  # Store all target values

    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(data_loader, desc="Testing")):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()

            # Collect all predictions and targets
            all_node_preds.extend(pred.cpu().detach().numpy())
            all_node_targets.extend(y.cpu().detach().numpy())

    # Convert predictions and targets to numpy arrays
    all_node_preds = np.array(all_node_preds)
    all_node_targets = np.array(all_node_targets)


    # Calculate evaluation metrics
    metrics = compute_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Test")
    # ==== 🔽 绘制 ROC 曲线 ====
    plot_and_save_roc_curve(all_node_targets, all_node_preds,metrics['auc'], save_dir, filename="roc_curve.png")
    # ===========================
    avg_test_loss = test_loss / len(data_loader)

    return avg_test_loss, metrics