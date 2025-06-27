import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import classification_metrics, log_metrics, plot_and_save_roc_curve, plot_classification_distribution
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer, scheduler, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    all_node_preds = []  # 存储所有预测值
    all_node_targets = []  # 存储所有目标值
    optimizer.zero_grad()  # 初始化梯度

    for batch, (x, y) in enumerate(tqdm(data_loader, desc="Training")):
        x, y = x.to(device), y.to(device)

        # 计算预测值
        pred = model(x)
        loss = criterion(pred, y)

        # 反向传播
        loss = loss/accumulation_steps  
        loss.backward()
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(data_loader):  # 达到累积批次后更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度
            step_scheduler(scheduler, event='batch')  # 更新调度器

        # 累加损失
        total_loss += loss.item()*accumulation_steps

        # 收集所有预测和目标值
        all_node_preds.append(pred.cpu().detach().numpy())
        all_node_targets.append(y.cpu().detach().numpy())

        if batch % 100 == 0:
            tqdm.write(f"Batch {batch:>5d}/{len(data_loader):>5d} | Loss: {loss.item():.6f}")

    # 合并所有的预测和目标
    all_node_preds = np.concatenate(all_node_preds, axis=0)
    all_node_targets = np.concatenate(all_node_targets, axis=0)

    # 计算指标
    metrics = classification_metrics(all_node_targets, all_node_preds)
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

    metrics = classification_metrics(all_node_targets, all_node_preds)
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
    metrics = classification_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Test")
    # ==== 🔽 绘制 ROC 曲线 ====
    plot_and_save_roc_curve(all_node_targets, all_node_preds,metrics['auc'], save_dir, filename="roc_curve.png")
    # ===========================
    avg_test_loss = test_loss / len(data_loader)

    return avg_test_loss, metrics


def visualize_results(model,train_loader, val_loader, test_loader, device, save_dir):
    visualize_predictions(model, train_loader, device, save_dir, title="Train Set Prediction Distribution", filename="train_pred_distribution.png")
    visualize_predictions(model, val_loader, device, save_dir, title="Validation Set Prediction Distribution", filename="val_pred_distribution.png")
    visualize_predictions(model, test_loader, device, save_dir, title="Test Set Prediction Distribution", filename="test_pred_distribution.png")



def visualize_predictions(model, data_loader, device, save_dir, title="Prediction Probability Distribution", filename="pred_distribution.png"):
    """
    收集任意数据集预测结果并可视化分布图
    :param model: 已加载的模型
    :param data_loader: DataLoader（可以是train_loader、val_loader、test_loader等）
    :param device: 当前设备
    :param save_dir: 保存图像的目录
    :param title: 图像标题
    :param filename: 保存图像的文件名
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 画图
    save_path = os.path.join(save_dir, filename)
    plot_classification_distribution(
        preds=all_preds,
        labels=all_labels,
        title=title,
        save_path=save_path
    )

    print(f"[✔] 预测分布图已保存至: {save_path}")
