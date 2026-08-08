"""Shared sequence sampling and compensator evaluation helpers.

These utilities centralize repeated post-processing patterns (sampled batch
construction, compensator grid evaluation) used across multiple TPP models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import src


def _pad_sequence_like(
    tensor: Optional[torch.Tensor],
    *,
    pad_cols: int,
    pad_value: float = 0.0,
) -> Optional[torch.Tensor]:
    if tensor is None or pad_cols <= 0:
        return tensor
    if tensor.ndim < 2:
        raise ValueError(
            f"Expected a sequence-like tensor with ndim >= 2, got shape={tuple(tensor.shape)}."
        )
    pad_shape = list(tensor.shape)
    pad_shape[1] = pad_cols
    pad_tensor = tensor.new_full(pad_shape, pad_value)
    return torch.cat([tensor, pad_tensor], dim=1)


def build_sample_batch(
    *,
    inter_times: torch.Tensor,
    t_start: float,
    t_end: float,
    device: torch.device,
    magnitudes: Optional[torch.Tensor] = None,
    time_dtype: torch.dtype = torch.float32,
    epsilon: float = 1e-5,
    clamp_last_surv_time: bool = False,
    check_last_surv_nonnegative: bool = False,
) -> src.data.Batch:
    """Build a padded `Batch` from sampled inter-event times."""
    batch_size = inter_times.shape[0]
    duration = float(t_end - t_start)

    unclipped_arrival_times = inter_times.cumsum(-1)
    zero_padding = (inter_times == 0.0).to(torch.int64).cummax(dim=-1).values.bool()
    padding_mask = (unclipped_arrival_times > duration - epsilon) | zero_padding
    inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
    end_idx = (1 - padding_mask.long()).sum(-1)

    last_surv_time = duration - inter_times.sum(-1)
    if clamp_last_surv_time:
        last_surv_time = last_surv_time.clamp_min(0.0)
    if check_last_surv_nonnegative and (last_surv_time < 0).any():
        min_value = float(last_surv_time.min().item())
        raise ValueError(f"last_surv_time < 0 detected (min={min_value})")

    required_len = int(end_idx.max().item()) + 1 if batch_size > 0 else inter_times.shape[1]
    pad_cols = max(required_len - inter_times.shape[1], 0)
    if pad_cols > 0:
        inter_times = _pad_sequence_like(inter_times, pad_cols=pad_cols, pad_value=0.0)
        padding_pad = torch.ones(
            batch_size,
            pad_cols,
            device=padding_mask.device,
            dtype=padding_mask.dtype,
        )
        padding_mask = torch.cat([padding_mask, padding_pad], dim=1)
        magnitudes = _pad_sequence_like(magnitudes, pad_cols=pad_cols, pad_value=0.0)

    arange = torch.arange(batch_size, device=device)
    inter_times[arange, end_idx] = last_surv_time

    return src.data.Batch(
        inter_times=inter_times,
        arrival_times=inter_times.cumsum(-1) + float(t_start),
        t_start=torch.full([batch_size], t_start, device=device, dtype=time_dtype),
        t_end=torch.full([batch_size], t_end, device=device, dtype=time_dtype),
        t_nll_start=torch.full([batch_size], t_start, device=device, dtype=time_dtype),
        mask=padding_mask.float(),
        start_idx=torch.zeros(batch_size, device=device).long(),
        end_idx=end_idx,
        mag=magnitudes,
    )


def evaluate_compensator_from_model(
    *,
    model,
    sequence: src.data.Sequence,
    num_grid_points: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the total compensator on an inter-event grid.

    For a recurrent TPP with an optional exogenous background, this returns
    the compensator of the same total intensity used by training,
    ``h(t | history) + f(t)``.  The triggering contribution is obtained from
    the inter-time survival function and the background contribution is
    integrated over absolute time.
    """
    batch = src.data.Batch.from_list([sequence]).to(model.device)
    context = model.get_context(batch).squeeze(0)
    time_context_fn = getattr(model, "_get_time_context", None)
    if callable(time_context_fn):
        context = time_context_fn(context.unsqueeze(0), batch=batch).squeeze(0)
    inter_time_dist = model.get_inter_time_dist(context)

    x = batch.inter_times * torch.linspace(
        1e-4,
        1,
        num_grid_points,
        device=batch.inter_times.device,
        dtype=batch.inter_times.dtype,
    )[:, None]
    log_surv = inter_time_dist.log_survival(x)
    surv_offsets = torch.cat(
        [
            torch.tensor([0.0], device=x.device, dtype=x.dtype),
            log_surv[-1].cumsum(dim=-1)[:-1],
        ]
    )
    compensator = -(log_surv + surv_offsets).T.reshape(-1)

    offsets = torch.cat(
        [
            batch.t_start.to(device=x.device, dtype=x.dtype),
            batch.arrival_times.squeeze(0)[:-1].to(dtype=x.dtype),
        ]
    )
    grid_2d = x + offsets

    bg_model = getattr(model, "bg_model", None)
    if bg_model is not None:
        bg_batch = batch
        if not hasattr(bg_batch, "time_series"):
            bg_batch = getattr(bg_model, "ts_batch_cache", None)
        if bg_batch is None:
            raise ValueError(
                "Background compensator requires sequence time-series fields "
                "or bg_model.ts_batch_cache."
            )
        integrate_between = getattr(bg_model, "intensity_integral_between", None)
        if not callable(integrate_between):
            raise ValueError(
                "Background model must implement intensity_integral_between() "
                "to evaluate the total compensator."
            )

        grid_flat = grid_2d.T.reshape(1, -1)
        bg_start = batch.t_start.to(device=grid_flat.device, dtype=grid_flat.dtype)
        bg_start = bg_start.expand_as(grid_flat)
        bg_compensator = integrate_between(
            bg_batch,
            t_start=bg_start,
            t_end=grid_flat,
        ).reshape(-1)
        compensator = compensator + bg_compensator

    grid = grid_2d.T.reshape(-1)
    return grid, compensator


__all__ = [
    "build_sample_batch",
    "evaluate_compensator_from_model",
]
