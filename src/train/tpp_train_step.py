import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.metrics import  log_metrics
from .trainer import step_scheduler
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from torch import amp

def train(data_loader, model, criterion, optimizer, scheduler, device,
          accumulation_steps=2, ema_model=None, use_amp=False, max_grad_norm=3.0, pbar_desc='Training'):
    """
    兼容“batch是整体对象”的训练循环：
      - 计算方式：loss = model.nll_loss(batch).mean()
      - 支持 AMP（autocast + GradScaler）
      - 支持梯度累积与最后一个尾步
      - 每个优化步后可选 EMA 同步与 scheduler 的 batch 级步进
    """

    model.train()
    total_loss = 0.0
    step_in_accum = 0
    backend = device.type  # “cuda” 或 “cpu”
    scaler = amp.GradScaler(backend, enabled=(backend == 'cuda' and use_amp))
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (device.type == 'cuda' and use_amp) else nullcontext())

    optimizer.zero_grad(set_to_none=True)  # 更省内存/更快；与官方建议一致。:contentReference[oaicite:2]{index=2}

    for i, batch in enumerate(tqdm(data_loader, desc=pbar_desc)):
        batch = batch.to(device)

        with amp_ctx:
            loss = model.nll_loss(batch).mean()

        loss_to_backward = loss / accumulation_steps
        scaler.scale(loss_to_backward).backward()
        total_loss += loss.item()

        step_in_accum += 1
        is_update_step = (step_in_accum % accumulation_steps == 0) or (i == len(data_loader) - 1)

        if is_update_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 下一轮前清梯度。:contentReference[oaicite:5]{index=5}
            step_scheduler(scheduler, event='batch')

            if ema_model is not None:
                ema_model.update_parameters(model)

            if device.type == 'cuda':
                torch.cuda.synchronize()

    metrics = {
        'avg_nll': total_loss / len(data_loader)
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
        'avg_nll': total_loss/len(data_loader),  # Average event log-likelihood
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

