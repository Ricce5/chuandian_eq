import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import regression_metrics, log_metrics, plot_regression_scatter, plot_regression_series
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer,scheduler, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    all_node_preds = []  # 存储所有预测值
    all_node_targets = []  # 存储所有目标值
    optimizer.zero_grad()
    for batch, (x, y) in enumerate(tqdm(data_loader, desc="Training")):
        x, y = x.to(device), y.to(device)
        
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        print(accumulation_steps)
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(data_loader):  # 达到累积批次后更新
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度
            step_scheduler(scheduler, event='batch')  # 更新调度器

        total_loss += loss.item()

        # 收集所有预测和目标值
        all_node_preds.append(pred.cpu().detach().numpy())
        all_node_targets.append(y.cpu().detach().numpy())

        if batch % 100 == 0:
            tqdm.write(f"Batch {batch:>5d}/{len(data_loader):>5d} | Loss: {loss.item():.6f}")

  
    all_node_preds = np.concatenate(all_node_preds, axis=0)
    all_node_targets = np.concatenate(all_node_targets, axis=0)


    metrics = regression_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Training")
    avg_train_loss = total_loss / len(data_loader)
    
    return avg_train_loss, metrics

def validate(data_loader, model, criterion, device):
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

    metrics = regression_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Validation")

    avg_val_loss = val_loss / len(data_loader)
  
    return avg_val_loss, metrics

def test(data_loader, model, criterion, device, save_dir=None):
    
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
    metrics = regression_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Test")
    avg_test_loss = test_loss / len(data_loader)

    return avg_test_loss, metrics


def visualize_results(model, train_loader, val_loader, test_loader, device, save_dir):
    """
    Visualize regression model predictions across train, validation, and test sets.
    Applies inverse normalization if dataset provides it.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    def get_root_dataset(loader):
        dataset = loader.dataset
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    def collect_predictions(loader):
        y_true, y_pred = [], []
        dataset = get_root_dataset(loader)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                preds = model(x).cpu().numpy()
                labels = y.cpu().numpy()

                # 如果 Dataset 有 inverse_normalize_label 方法
                if hasattr(dataset, "inverse_normalize_label"):
                    preds = dataset.inverse_normalize_label(preds)
                    labels = dataset.inverse_normalize_label(labels)

                y_true.extend(labels)
                y_pred.extend(preds)

        return np.array(y_true), np.array(y_pred)

    # 收集（已反归一化的）数据
    train_true, train_pred = collect_predictions(train_loader)
    val_true, val_pred = collect_predictions(val_loader)
    test_true, test_pred = collect_predictions(test_loader)

    data_dict = {
        "Train": (train_true, train_pred),
        "Validation": (val_true, val_pred),
        "Test": (test_true, test_pred),
    }

    # 可视化
    plot_regression_scatter(data_dict, save_path=os.path.join(save_dir, "regression_scatter.png"))
    plot_regression_series(data_dict, save_path=os.path.join(save_dir, "regression_series.png"))

