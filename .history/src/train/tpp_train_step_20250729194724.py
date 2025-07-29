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

    optimizer.zero_grad()  # Only call once at the beginning, to initialize gradients
    step_count = 0  # 跟踪已处理的步骤数
    for batch in tqdm(data_loader, desc='Training'):
        batch = batch.to(device)  # Move the entire batch to device

        loss  = model.nll_loss(batch).mean()  
        loss = loss/accumulation_steps  
        loss.backward()  # Accumulate gradients

        # Accumulate loss
        total_loss += loss.item() * accumulation_steps
    
        # If accumulation_steps have been completed, update the model
        step_count += 1
        if  step_count % accumulation_steps == 0 or  step_count  == len(data_loader):  # Update after every 'accumulation_steps' batches
            optimizer.step()  # Perform parameter update
            optimizer.zero_grad()  # Clear gradients for the next accumulation
            step_scheduler(scheduler, event='batch')  # Update scheduler

    metrics = {
        'total_nll': total_loss,
    }
    log_metrics(metrics, prefix="Training")

    return total_loss, metrics


def validate(data_loader, model, criterion, device, accumulation_steps=2):
    """Epoch operation in validation phase (only nll_loss)."""
    import numpy as np
    from tqdm import tqdm

    model.eval()

    total_loss = 0  # 累积的损失

    step_count = 0  # 跟踪已处理的步骤数
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating'):
            batch = batch.to(device)
            loss= model.nll_loss(batch).mean()

            # 累积损失和事件数量
            total_loss += loss.item()

            step_count += 1

    metrics = {
        'total_nll': total_loss,  # 事件的对数似然
    }
    
    log_metrics(metrics, prefix="Validation")
    return total_loss, metrics





def test(train_loader=None, val_loader=None, test_loader=None, model=None, criterion=None, device=None, save_dir=None):
    """
    Evaluate model on any subset of [train, val, test].
    Returns:
        total_loss_dict: {'train': float, 'val': float, 'test': float}
        metrics: {'nll_train': float, 'nll_val': float, 'nll_test': float}
    """
    import os
    import torch
    from tqdm import tqdm

    def compute_nll(loader, name):
        if loader is None:
            return None, None
        model.eval()
        total_loss = 0.0
        step_count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {name}"):
                batch = batch.to(device)
                loss = model.nll_loss(batch).mean()
                total_loss += loss.item()
                step_count += 1
        avg_loss = total_loss / step_count if step_count > 0 else float("nan")
        return total_loss, avg_loss

    results = {}
    metrics = {}

    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        total_loss, avg_loss = compute_nll(loader, split_name)
        if avg_loss is not None:
            results[f'nll_{split_name}_total'] = total_loss
            metrics[f'nll_{split_name}'] = avg_loss

    log_metrics(metrics, prefix="Evaluation")
    return results, metrics

