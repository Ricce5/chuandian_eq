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

def _mean_if_tensor(x):
    # 支持标量或 (B,) 张量
    return x.mean().item() if torch.is_tensor(x) else float(x)

def _avg_from_out_dict(out_dict):
    return {key: _mean_if_tensor(value) for key, value in out_dict.items()}

def train(
    data_loader, model, criterion, optimizer, scheduler, device,
    accumulation_steps=2, ema_model=None, use_amp=False, max_grad_norm=3.0,
    pbar_desc='Training', loss_key='total',  # 反传的目标：'total'|'time'|'mag' 或其他任意 key
    nll_kwargs: dict | None = None           # 透传给 model.nll_loss 的参数（例如 predict_b / mag_weight / reduction）
):
    """
    兼容“model.nll_loss 返回任意键的字典”的训练循环。
    - 反传目标由 loss_key 决定；同时记录所有项的 batch 平均指标。
    - 建议 nll_kwargs 至少包含 {'reduction': 'none'}，便于逐序列取 mean。
    """
    if nll_kwargs is None:
        nll_kwargs = {}
    nll_kwargs.setdefault('reduction', 'per_time')

    model.train()
    step_in_accum = 0
    backend = device.type  # 'cuda' or 'cpu'
    scaler = amp.GradScaler(backend, enabled=(backend == 'cuda' and use_amp))
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (device.type == 'cuda' and use_amp) else nullcontext())

    optimizer.zero_grad(set_to_none=True)

    # 统计指标（epoch 平均）
    sum_metrics = {}
    num_steps = 0

    for i, batch in enumerate(tqdm(data_loader, desc=pbar_desc)):
        batch = batch.to(device)

        with amp_ctx:
            out = model.nll_loss(batch, **nll_kwargs)
            if loss_key not in out:
                raise KeyError(f"loss_key='{loss_key}' 不在 nll 输出中，可选项：{list(out.keys())}")
            loss = out[loss_key].mean()


        loss_to_backward = loss / accumulation_steps
        scaler.scale(loss_to_backward).backward()


        metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
        for key, value in metrics_now.items():
            if key not in sum_metrics:
                sum_metrics[key] = 0.0
            sum_metrics[key] += value
        num_steps += 1

        step_in_accum += 1
        is_update_step = (step_in_accum % accumulation_steps == 0) or (i == len(data_loader) - 1)

        if is_update_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_scheduler(scheduler, event='batch')

            if ema_model is not None:
                ema_model.update_parameters(model)

            if device.type == 'cuda':
                torch.cuda.synchronize()

    # epoch 平均指标
    metrics = {f'avg_{key}': (sum_value / max(1, num_steps)) for key, sum_value in sum_metrics.items()}
    log_metrics(metrics, prefix="Training")
    return metrics.get(f'avg_{loss_key}', 0.0), metrics



def validate(
    data_loader, model, criterion, device,
    loss_key='total', nll_kwargs: dict | None = None
):
    """
    验证阶段（无反传），返回所有键的平均 NLL。
    """
    if data_loader is None:
        return float('nan'), {}
    if nll_kwargs is None:
        nll_kwargs = {}
    nll_kwargs.setdefault('reduction', 'per_time')

    model.eval()
    sum_metrics = {}
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating'):
            batch = batch.to(device)
            out = model.nll_loss(batch, **nll_kwargs)  # 任意字典
            metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
            for key, value in metrics_now.items():
                if key not in sum_metrics:
                    sum_metrics[key] = 0.0
                sum_metrics[key] += value
            num_steps += 1

    metrics = {f'avg_{key}': (sum_value / max(1, num_steps)) for key, sum_value in sum_metrics.items()}
    log_metrics(metrics, prefix="Validation")
    return metrics.get(f'avg_{loss_key}', 0.0), metrics


def test(
    train_loader=None, val_loader=None, test_loader=None,
    model=None, criterion=None, device=None, save_dir=None,
    nll_kwargs: dict | None = None
):
    """
    在任意子集 {train, val, test} 上评估，分别返回所有键的平均 NLL。
    """
    if nll_kwargs is None:
        nll_kwargs = {}
        nll_kwargs.setdefault('reduction', 'per_time')
        nll_kwargs.setdefault('predict_b', None)
        nll_kwargs.setdefault('mag_weight', 1)

    def compute(loader, name):
        if loader is None:
            return None, None
        model.eval()
        sum_metrics = {}
        num_steps = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {name}"):
                batch = batch.to(device)
                out = model.nll_loss(batch, **nll_kwargs)
                metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
                for key, value in metrics_now.items():
                    if key not in sum_metrics:
                        sum_metrics[key] = 0.0
                    sum_metrics[key] += value
                num_steps += 1
        # 计算所有键的平均值
        avg = {key: sum_value / max(1, num_steps) for key, sum_value in sum_metrics.items()}
        return avg, sum_metrics

    results = {}
    metrics = {}

    avg, sums = compute(train_loader, 'train')
    if avg is not None:
        for key, value in sums.items():
            results[f'nll_train_{key}'] = value
            metrics[f'nll_train_{key}'] = avg.get(key, 0.0)

    avg, sums = compute(val_loader, 'val')
    if avg is not None:
        for key, value in sums.items():
            results[f'nll_val_{key}'] = value
            metrics[f'nll_val_{key}'] = avg.get(key, 0.0)

    avg, sums = compute(test_loader, 'test')
    if avg is not None:
        for key, value in sums.items():
            results[f'nll_test_{key}'] = value
            metrics[f'nll_test_{key}'] = avg.get(key, 0.0)

    log_metrics(metrics, prefix="Evaluation")
    return results, metrics
