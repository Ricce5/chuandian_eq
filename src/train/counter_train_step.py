import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import count_metrics, log_metrics,plot_count_scatter, plot_count_series
from .trainer import step_scheduler

def train(data_loader, model, criterion, optimizer,scheduler, device):
    model.train()
    total_loss = 0
    all_node_preds = []  
    all_node_targets = []  

    for batch, (x, y) in enumerate(tqdm(data_loader, desc="Training")):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        optimizer.step()
        step_scheduler(scheduler, event='step')
        total_loss += loss.item()

        all_node_preds.append(pred.cpu().detach().numpy())
        all_node_targets.append(y.cpu().detach().numpy())

        if batch % 100 == 0:
            tqdm.write(f"Batch {batch:>5d}/{len(data_loader):>5d} | Loss: {loss.item():.6f}")

  
    all_node_preds = np.concatenate(all_node_preds, axis=0)
    all_node_targets = np.concatenate(all_node_targets, axis=0)


    metrics = count_metrics(all_node_targets, all_node_preds)
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

    metrics = count_metrics(all_node_targets, all_node_preds)
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
    metrics = count_metrics(all_node_targets, all_node_preds)
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

                y_true.extend(labels)
                y_pred.extend(preds)

        return np.array(y_true), np.array(y_pred)


    train_true, train_pred = collect_predictions(train_loader)
    val_true, val_pred = collect_predictions(val_loader)
    test_true, test_pred = collect_predictions(test_loader)

    data_dict = {
        "Train": (train_true, train_pred),
        "Validation": (val_true, val_pred),
        "Test": (test_true, test_pred),
    }

    plot_count_scatter(data_dict, save_path=os.path.join(save_dir, "regression_scatter.png"))
    plot_count_series(data_dict, save_path=os.path.join(save_dir, "regression_series.png"))

