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
    # out_dict 里应有 'time'/'mag'/'total' 三项
    return {
        'nll_time':  _mean_if_tensor(out_dict['time']),
        'nll_mag':   _mean_if_tensor(out_dict['mag']),
        'nll_total': _mean_if_tensor(out_dict['total']),
        'smooth':    _mean_if_tensor(out_dict['smooth']),
    }

def train(
    data_loader, model, criterion, optimizer, scheduler, device,
    accumulation_steps=2, ema_model=None, use_amp=False, max_grad_norm=3.0,
    pbar_desc='Training', loss_key='total',  # 反传的目标：'total'|'time'|'mag'
    nll_kwargs: dict | None = None           # 透传给 model.nll_loss 的参数（例如 predict_b / mag_weight / reduction）
):
    """
    兼容“model.nll_loss 返回 dict(time, mag, total)”的训练循环。
    - 反传目标由 loss_key 决定；同时记录三项的 batch 平均指标。
    - 建议 nll_kwargs 至少包含 {'reduction': 'none'}，便于逐序列取 mean。
    """
    if nll_kwargs is None:
        nll_kwargs = {}
    # 确保不在 model 内提前做聚合，方便我们在这里做 .mean()
    nll_kwargs.setdefault('reduction', 'none')

    model.train()
    step_in_accum = 0
    backend = device.type  # 'cuda' or 'cpu'
    scaler = amp.GradScaler(backend, enabled=(backend == 'cuda' and use_amp))
    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (device.type == 'cuda' and use_amp) else nullcontext())

    optimizer.zero_grad(set_to_none=True)

    # 统计指标（epoch 平均）
    sum_time = 0.0
    sum_mag = 0.0
    sum_total = 0.0
    sum_smooth = 0.0
    num_steps = 0

    for i, batch in enumerate(tqdm(data_loader, desc=pbar_desc)):
        batch = batch.to(device)

        with amp_ctx:
            # out: {'time': (B,), 'mag': (B,), 'total': (B,)}
            out = model.nll_loss(batch, **nll_kwargs)
            # 反传目标
            if loss_key not in out:
                raise KeyError(f"loss_key='{loss_key}' 不在 nll 输出中，可选项：{list(out.keys())}")
            loss = out[loss_key].mean()

        # 梯度累积
        loss_to_backward = loss / accumulation_steps
        scaler.scale(loss_to_backward).backward()

        # 累计日志（不参与反传）
        metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
        sum_time += metrics_now['nll_time']
        sum_mag += metrics_now['nll_mag']
        sum_total += metrics_now['nll_total']
        sum_smooth += metrics_now['smooth']
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
    metrics = {
        'avg_nll_time':  (sum_time  / max(1, num_steps)),
        'avg_nll_mag':   (sum_mag   / max(1, num_steps)),
        'avg_nll_total': (sum_total / max(1, num_steps)),
        'avg_smooth':    (sum_smooth / max(1, num_steps)),
    }
    log_metrics(metrics, prefix="Training")
    return metrics['avg_nll_total'], metrics  # 返回一个主指标和字典


def validate(
    data_loader, model, criterion, device,
    loss_key='total', nll_kwargs: dict | None = None
):
    """
    验证阶段（无反传），返回 time/mag/total 三项的平均 NLL。
    """
    if data_loader is None:
        return float('nan'), {}
    if nll_kwargs is None:
        nll_kwargs = {}
    nll_kwargs.setdefault('reduction', 'none')

    model.eval()
    sum_time = 0.0
    sum_mag = 0.0
    sum_total = 0.0
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating'):
            batch = batch.to(device)
            out = model.nll_loss(batch, **nll_kwargs)  # dict
            metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
            sum_time += metrics_now['nll_time']
            sum_mag += metrics_now['nll_mag']
            sum_total += metrics_now['nll_total']
            num_steps += 1

    metrics = {
        'avg_nll_time':  (sum_time  / max(1, num_steps)),
        'avg_nll_mag':   (sum_mag   / max(1, num_steps)),
        'avg_nll_total': (sum_total / max(1, num_steps)),
    }
    log_metrics(metrics, prefix="Validation")
    return metrics['avg_nll_total'], metrics


def test(
    train_loader=None, val_loader=None, test_loader=None,
    model=None, criterion=None, device=None, save_dir=None,
    nll_kwargs: dict | None = None
):
    """
    在任意子集 {train, val, test} 上评估，分别返回 time/mag/total 的平均 NLL。
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
        sum_time = 0.0
        sum_mag = 0.0
        sum_total = 0.0
        num_steps = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {name}"):
                batch = batch.to(device)
                out = model.nll_loss(batch, **nll_kwargs)
                metrics_now = _avg_from_out_dict({k: v.mean() for k, v in out.items()})
                sum_time += metrics_now['nll_time']
                sum_mag += metrics_now['nll_mag']
                sum_total += metrics_now['nll_total']
                num_steps += 1
        avg = {
            'time':  sum_time  / max(1, num_steps),
            'mag':   sum_mag   / max(1, num_steps),
            'total': sum_total / max(1, num_steps),
        }
        return avg, (sum_time, sum_mag, sum_total)

    results = {}
    metrics = {}

    avg, sums = compute(train_loader, 'train')
    if avg is not None:
        results['nll_train_total'], results['nll_train_time'], results['nll_train_mag'] = sums[2], sums[0], sums[1]
        metrics['nll_train_total'] = avg['total']
        metrics['nll_train_time']  = avg['time']
        metrics['nll_train_mag']   = avg['mag']

    avg, sums = compute(val_loader, 'val')
    if avg is not None:
        results['nll_val_total'], results['nll_val_time'], results['nll_val_mag'] = sums[2], sums[0], sums[1]
        metrics['nll_val_total'] = avg['total']
        metrics['nll_val_time']  = avg['time']
        metrics['nll_val_mag']   = avg['mag']

    avg, sums = compute(test_loader, 'test')
    if avg is not None:
        results['nll_test_total'], results['nll_test_time'], results['nll_test_mag'] = sums[2], sums[0], sums[1]
        metrics['nll_test_total'] = avg['total']
        metrics['nll_test_time']  = avg['time']
        metrics['nll_test_mag']   = avg['mag']

    log_metrics(metrics, prefix="Evaluation")
    return results, metrics
