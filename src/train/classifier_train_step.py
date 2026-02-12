import logging
import torch
import os
import numpy as np
from tqdm import tqdm
from src.utils.metrics import (
    classification_metrics,
    log_metrics,
    plot_and_save_roc_curve,
    plot_and_save_pr_curve,
    plot_and_save_confusion_matrix,
    plot_classification_distribution,
)
from .trainer import step_scheduler

logger = logging.getLogger(__name__)

def train(data_loader, model, criterion, optimizer, scheduler, device, accumulation_steps=2, ema_model=None):
    model.train()
    total_loss = 0
    all_node_preds = []  
    all_node_targets = []
    optimizer.zero_grad()
    for batch, (x, y) in enumerate(tqdm(data_loader, desc="Training")):
        x, y = x.to(device), y.to(device)

        # Magnitude noise augmentation during training (optional)
        noise_std = float(getattr(model, "mag_noise_std", 0.0) or 0.0)
        if noise_std > 0.0:
            noise_type = getattr(model, "mag_noise_type", "gaussian")
            non_pad_mask = (x[:, :, 0] != 0).to(x.dtype)
            if noise_type == "uniform":
                noise = (torch.rand_like(x[:, :, 2]) - 0.5) * 2.0 * noise_std
            else:
                noise = torch.randn_like(x[:, :, 2]) * noise_std
            x[:, :, 2] = x[:, :, 2] + noise * non_pad_mask

        pred = model(x)
        loss = criterion(pred, y)

        loss = loss/accumulation_steps  
        loss.backward()
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(data_loader): 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step() 
            optimizer.zero_grad()
            step_scheduler(scheduler, event='step')
            if ema_model is not None:
                ema_model.update_parameters(model)

        total_loss += loss.item()*accumulation_steps

        all_node_preds.append(pred.cpu().detach().numpy())
        all_node_targets.append(y.cpu().detach().numpy())

        if batch % 100 == 0:
            tqdm.write(f"Batch {batch:>5d}/{len(data_loader):>5d} | Loss: {loss.item():.6f}")

    all_node_preds = np.concatenate(all_node_preds, axis=0)
    all_node_targets = np.concatenate(all_node_targets, axis=0)

    metrics = classification_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Training")

    avg_train_loss = total_loss / len(data_loader)
    
    return avg_train_loss, metrics


def validate(data_loader, model, criterion, device):
    model.eval()
    val_loss = 0

    all_node_preds = []
    all_node_targets = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(data_loader, desc="Validating")):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

            all_node_preds.extend(pred.cpu().detach().numpy())
            all_node_targets.extend(y.cpu().detach().numpy())

    all_node_preds = np.array(all_node_preds)
    all_node_targets = np.array(all_node_targets)

    metrics = classification_metrics(all_node_targets, all_node_preds)
    log_metrics(metrics, prefix="Validation")

    avg_val_loss = val_loss / len(data_loader)
  
    return avg_val_loss, metrics

def test(data_loader, model, criterion, device, save_dir=None,threshold=None):
    
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
    metrics = classification_metrics(all_node_targets, all_node_preds, threshold=threshold)
    log_metrics(metrics, prefix="Test")
    plot_and_save_roc_curve(all_node_targets, all_node_preds,metrics['auc'], save_dir, filename="roc_curve.png")
    plot_and_save_pr_curve(all_node_targets, all_node_preds, metrics["pr_auc"], save_dir, filename="pr_curve.png")
    plot_and_save_confusion_matrix(
        all_node_targets,
        all_node_preds,
        threshold=metrics["threshold"],
        save_dir=save_dir,
        filename="confusion_matrix.png",
        apply_sigmoid=True,
    )
    avg_test_loss = test_loss / len(data_loader)

    return avg_test_loss, metrics


def visualize_results(model,train_loader, val_loader, test_loader, device, save_dir):
    visualize_predictions(model, train_loader, device, save_dir, title="Train Set Prediction Distribution", filename="train_pred_distribution.png")
    visualize_predictions(model, val_loader, device, save_dir, title="Validation Set Prediction Distribution", filename="val_pred_distribution.png")
    visualize_predictions(model, test_loader, device, save_dir, title="Test Set Prediction Distribution", filename="test_pred_distribution.png")



def visualize_predictions(model, data_loader, device, save_dir, title="Prediction Probability Distribution", filename="pred_distribution.png"):
    """
    Visualize the distribution of predicted probabilities for positive and negative classes.
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

    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    plot_classification_distribution(
        preds=all_preds,
        labels=all_labels,
        title=title,
        save_path=save_path
    )

    logger.info("Prediction distribution plot has been saved to: %s", save_path)
