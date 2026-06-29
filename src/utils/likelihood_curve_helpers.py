import contextlib
import inspect
import io

import numpy as np
import torch

from src.data import Batch


def _resolve_device(model, device=None):
    """Resolve execution device from explicit input or model parameters."""
    if device is not None:
        return device
    return next(model.parameters()).device


def _to_numpy_float(values):
    """Convert tensor/list/array values to a float NumPy array."""
    if torch.is_tensor(values):
        return values.detach().cpu().to(torch.float64).numpy()
    return np.asarray(values, dtype=float)


def _get_inter_time_dist_silent(model, context):
    """Call ``model.get_inter_time_dist`` while suppressing stdout noise."""
    with contextlib.redirect_stdout(io.StringIO()):
        return model.get_inter_time_dist(context)


def _get_recurrent_time_context(model, context, batch):
    """Return the context used by recurrent time likelihood heads."""
    get_time_context = getattr(model, "_get_time_context", None)
    if get_time_context is None:
        return context
    return get_time_context(context, batch=batch)


def _prefix_cumsum_at_queries(event_times, cum_values, query_times):
    """Gather cumulative values at right-closed prefix indices."""
    prefix = torch.zeros_like(query_times)
    if event_times.numel() == 0:
        return prefix, None, None
    idx = torch.searchsorted(event_times, query_times, right=True) - 1
    has_event = idx >= 0
    prefix[has_event] = cum_values[idx[has_event]]
    return prefix, idx, has_event


def _build_time_curve_output(times, cum_log_likelihood):
    """Build standard output dict for time-only cumulative curve."""
    out = {
        "time": _to_numpy_float(times),
        "cum_log_likelihood": _to_numpy_float(cum_log_likelihood),
    }
    out["cum_nll"] = -out["cum_log_likelihood"]
    return out


def _build_full_curve_output(
    times,
    cum_log_likelihood_time,
    cum_log_likelihood_total,
    cum_log_likelihood_trigger=None,
):
    """Build standard output dict for time and total cumulative curves."""
    ll_time = _to_numpy_float(cum_log_likelihood_time)
    ll_total = _to_numpy_float(cum_log_likelihood_total)
    out = {
        "time": _to_numpy_float(times),
        "cum_log_likelihood": ll_time.copy(),
        "cum_log_likelihood_time": ll_time,
        "cum_log_likelihood_total": ll_total,
    }
    out["cum_nll"] = -out["cum_log_likelihood"]
    out["cum_nll_time"] = -out["cum_log_likelihood_time"]
    out["cum_nll_total"] = -out["cum_log_likelihood_total"]
    if cum_log_likelihood_trigger is not None:
        ll_trigger = _to_numpy_float(cum_log_likelihood_trigger)
        out["cum_log_likelihood_trigger"] = ll_trigger
        out["cum_nll_trigger"] = -ll_trigger
    return out


def _build_curve_meta(curve_method, num_events_in_nll, final_log_likelihood, final_log_likelihood_total=None, **extra):
    """Build metadata dict shared by cumulative curve methods."""
    meta = {
        "num_events_in_nll": int(num_events_in_nll),
        "final_log_likelihood": float(final_log_likelihood),
        "curve_method": curve_method,
    }
    if final_log_likelihood_total is not None:
        meta["final_log_likelihood_total"] = float(final_log_likelihood_total)
    meta.update(extra)
    return meta


def _build_nll_kwargs(model):
    """Build optional kwargs supported by ``model.nll_loss``.

    The helper inspects ``model.nll_loss`` at runtime and only forwards flags
    that exist in the target signature.
    """
    sig = inspect.signature(model.nll_loss).parameters
    kwargs = {}
    if "reduction" in sig:
        kwargs["reduction"] = "none"
    if "return_dict" in sig:
        kwargs["return_dict"] = True
    if "use_b_updater" in sig:
        kwargs["use_b_updater"] = False
    return kwargs


def _extract_time_total_from_out(out):
    """Normalize ``nll_loss`` outputs into ``(time, total)`` values."""
    if isinstance(out, dict):
        time_val = out.get("time", out.get("total"))
        total_val = out.get("total", time_val)
        return time_val, total_val
    return out, out


def _model_nll_values_batch(model, sequences, device=None):
    """Compute per-sequence ``(nll_time, nll_total)`` for a sequence batch.

    Returns Python lists aligned with ``sequences`` and validates that the
    model returns exactly one value per input sequence.
    """
    if len(sequences) == 0:
        return [], []

    device = _resolve_device(model, device)

    batch = Batch.from_list(sequences).to(device)
    kwargs = _build_nll_kwargs(model)

    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        out = model.nll_loss(batch, **kwargs)

    time_val, total_val = _extract_time_total_from_out(out)
    time_tensor = torch.as_tensor(time_val).reshape(-1)
    total_tensor = torch.as_tensor(total_val).reshape(-1)

    expected = len(sequences)
    if time_tensor.numel() != expected or total_tensor.numel() != expected:
        raise RuntimeError(
            f"Unexpected nll shape for batched prefixes: "
            f"time={tuple(time_tensor.shape)}, total={tuple(total_tensor.shape)}, expected={expected}"
        )

    time_list = time_tensor.detach().cpu().to(torch.float64).tolist()
    total_list = total_tensor.detach().cpu().to(torch.float64).tolist()
    return time_list, total_list


def _model_nll_values(model, sequence, device=None):
    """Compute scalar ``(nll_time, nll_total)`` for one sequence."""
    time_list, total_list = _model_nll_values_batch(model, [sequence], device=device)
    return float(time_list[0]), float(total_list[0])


def _build_eval_and_end_values(sequence):
    """Build prefix evaluation times and clamped subsequence end values.

    Returns:
        tuple[list[float], list[float], torch.Tensor]:
            ``(eval_times, end_eval_values, nll_event_times)`` where
            ``end_eval_values`` are adjusted to satisfy minimum prefix span
            constraints.
    """
    event_times = sequence.arrival_times
    nll_event_times = event_times[(event_times > sequence.t_nll_start) & (event_times <= sequence.t_end)]

    eval_times = [float(sequence.t_nll_start)]
    eval_times.extend(float(t) for t in nll_event_times.detach().cpu().tolist())

    t_end = float(sequence.t_end)
    if eval_times[-1] < t_end:
        eval_times.append(t_end)

    seq_t_start = float(sequence.t_start)
    seq_t_nll_start = float(sequence.t_nll_start)
    seq_t_end = float(sequence.t_end)

    min_prefix_span = 1e-6
    min_end_required = seq_t_nll_start + min_prefix_span
    if hasattr(sequence, "time_series_times") and sequence.time_series_times is not None:
        ts_times = torch.as_tensor(sequence.time_series_times)
        if ts_times.numel() >= 2:
            min_end_required = max(min_end_required, float(ts_times[1].detach().cpu().item()) + 1e-12)

    end_eval_values = []
    for t_cur in eval_times:
        end_eval = float(t_cur)
        if end_eval <= seq_t_nll_start:
            end_eval = min(seq_t_end, min_end_required)
        end_eval = max(end_eval, min_end_required)
        end_eval = max(end_eval, seq_t_start)
        end_eval = min(end_eval, seq_t_end)
        end_eval_values.append(end_eval)

    return eval_times, end_eval_values, nll_event_times


def _bg_integral_prefix(bg_model, batch, end_eval_values):
    """Integrate background intensity from ``t_nll_start`` to each endpoint."""
    if not hasattr(bg_model, "intensity_integral_between"):
        raise ValueError("Fast BG integral requires bg_model.intensity_integral_between.")
    if batch.batch_size != 1:
        raise ValueError("Fast BG integral currently supports batch_size == 1 only.")

    end_eval = torch.as_tensor(
        end_eval_values,
        device=batch.arrival_times.device,
        dtype=batch.arrival_times.dtype,
    )
    t_start = batch.t_nll_start.view(1, 1).expand(1, end_eval.numel())
    t_end = end_eval.view(1, -1)
    out = bg_model.intensity_integral_between(
        ts_batch=batch,
        t_start=t_start,
        t_end=t_end,
    )  # (1, N) or (N,)
    return out.reshape(-1)


def _cumulative_curve_recurrent_fast(model, sequence, device=None, eps=1e-10):
    """Fast cumulative time log-likelihood curve for recurrent models."""
    device = _resolve_device(model, device)

    batch = Batch.from_list([sequence]).to(device)

    with torch.inference_mode():
        context = model.get_context(batch)
        context = _get_recurrent_time_context(model, context, batch)

        inter_time_dist = _get_inter_time_dist_silent(model, context)

        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(eps))[0]
        nll_mask = batch.nll_event_mask[0].bool()

        event_times = batch.arrival_times[0][nll_mask]
        event_log_like = log_pdf[nll_mask]
        cum_event_log_like = torch.cumsum(event_log_like, dim=0)

        arange = torch.arange(batch.batch_size, device=device)

        last_surv_context = context[arange, batch.end_idx, :]
        last_surv_dist = _get_inter_time_dist_silent(model, last_surv_context)
        last_surv_val = last_surv_dist.log_survival(batch.inter_times[arange, batch.end_idx].clamp_min(eps))
        last_log_surv = last_surv_val.reshape(-1)[0]

        offset_log_like = torch.tensor(0.0, device=device)
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = _get_inter_time_dist_silent(model, prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_surv_val = prev_surv_dist.log_survival(prev_surv_time.clamp_min(eps))
            prev_log_surv = prev_surv_val.reshape(-1)[0]
            offset_log_like = -prev_log_surv

        total_log_like = offset_log_like + event_log_like.sum() + last_log_surv

    t0 = float(batch.t_nll_start[0].detach().cpu().item())
    t_end = float(batch.t_end[0].detach().cpu().item())

    times = [t0]
    cum_ll = [float(offset_log_like.detach().cpu().item())]

    event_times_np = event_times.detach().cpu().numpy()
    cum_event_np = (offset_log_like + cum_event_log_like).detach().cpu().numpy()

    for t, val in zip(event_times_np, cum_event_np):
        times.append(float(t))
        cum_ll.append(float(val))

    times.append(t_end)
    cum_ll.append(float(total_log_like.detach().cpu().item()))

    out = _build_time_curve_output(times, cum_ll)
    meta = _build_curve_meta(
        curve_method="recurrent_fast",
        num_events_in_nll=nll_mask.sum().item(),
        final_log_likelihood=total_log_like.detach().cpu().item(),
        offset_log_likelihood=offset_log_like.detach().cpu().item(),
        terminal_log_survival=last_log_surv.detach().cpu().item(),
    )

    return out, meta


def _cumulative_curve_recurrent_total_fast(model, sequence, device=None, eps=1e-10):
    """Fast cumulative time/total log-likelihood for recurrent models with BG."""
    device = _resolve_device(model, device)
    if getattr(model, "bg_model", None) is None:
        raise ValueError("Recurrent total fast path requires model.bg_model.")

    eval_times, end_eval_values, nll_event_times = _build_eval_and_end_values(sequence)
    batch = Batch.from_list([sequence]).to(device)
    end_eval = torch.as_tensor(end_eval_values, device=device, dtype=batch.arrival_times.dtype)

    with torch.inference_mode():
        context = model.get_context(batch)
        context = _get_recurrent_time_context(model, context, batch)
        inter_time_dist = _get_inter_time_dist_silent(model, context)

        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(eps))[0]
        nll_mask = batch.nll_event_mask[0].bool()
        event_times = batch.arrival_times[0][nll_mask]
        event_log_like = log_pdf[nll_mask]
        cum_event_log_like = torch.cumsum(event_log_like, dim=0) if event_log_like.numel() > 0 else event_log_like

        event_prefix, idx_nll, has_nll_event = _prefix_cumsum_at_queries(
            event_times=event_times,
            cum_values=cum_event_log_like,
            query_times=end_eval,
        )

        # Survival contribution at each endpoint, matching per-prefix nll semantics.
        end_idx = int(batch.end_idx[0].item())
        real_event_times = batch.arrival_times[0, :end_idx]
        prefix_end_idx = torch.searchsorted(real_event_times, end_eval, right=True)
        prev_time = torch.full_like(end_eval, float(batch.t_start[0].item()))
        has_prev = prefix_end_idx > 0
        prev_time[has_prev] = real_event_times[prefix_end_idx[has_prev] - 1]
        surv_dt = (end_eval - prev_time).clamp_min(eps)

        surv_context = context[0, prefix_end_idx, :]
        surv_dist = _get_inter_time_dist_silent(model, surv_context)
        log_surv_prefix = surv_dist.log_survival(surv_dt).reshape(-1)

        offset_log_like = torch.tensor(0.0, device=device, dtype=event_prefix.dtype)
        if torch.any(batch.t_nll_start != batch.t_start):
            arange = torch.arange(batch.batch_size, device=device)
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = _get_inter_time_dist_silent(model, prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_surv_val = prev_surv_dist.log_survival(prev_surv_time.clamp_min(eps))
            prev_log_surv = prev_surv_val.reshape(-1)[0]
            offset_log_like = -prev_log_surv

        cum_ll_trigger = offset_log_like + event_prefix + log_surv_prefix

        # BG change term: sum log(1 + f/h) over events - integral f.
        log_h = inter_time_dist.log_hazard(batch.inter_times.clamp_min(eps))[0]
        h_events = torch.exp(log_h[nll_mask]).clamp_min(eps)
        if event_times.numel() > 0:
            f_events = model.bg_model.intensity(batch, t_query=event_times.unsqueeze(0))[0]
            bg_event = torch.log1p(f_events / h_events)
            cum_bg_event = torch.cumsum(bg_event, dim=0)
            bg_event_prefix = torch.zeros_like(end_eval)
            bg_event_prefix[has_nll_event] = cum_bg_event[idx_nll[has_nll_event]]
        else:
            bg_event_prefix = torch.zeros_like(end_eval)

        bg_int_prefix = _bg_integral_prefix(model.bg_model, batch, end_eval_values)
        cum_ll_time = cum_ll_trigger + bg_event_prefix - bg_int_prefix
        cum_ll_total = cum_ll_time.clone()

    out = _build_full_curve_output(
        eval_times,
        cum_ll_time,
        cum_ll_total,
        cum_log_likelihood_trigger=cum_ll_trigger,
    )
    meta = _build_curve_meta(
        curve_method="recurrent_total_fast",
        num_events_in_nll=nll_event_times.numel(),
        final_log_likelihood=out["cum_log_likelihood_time"][-1],
        final_log_likelihood_total=out["cum_log_likelihood_total"][-1],
        num_prefix_evals=len(eval_times),
    )
    return out, meta


def _is_etas_like(model):
    """Return whether model exposes ETAS fast-curve interface."""
    return hasattr(model, "h_intensity") and hasattr(model, "prefix_h_integral")


def _is_recurrent_curve_model(model):
    """Return whether model exposes recurrent fast-curve interface."""
    return hasattr(model, "get_context") and hasattr(model, "get_inter_time_dist")


def _cumulative_curve_etas_total_fast(model, sequence, device=None, eps=1e-10, query_block_size=256):
    """Fast cumulative total log-likelihood curve for ETAS-like models.

    Uses block-wise intensity queries to control peak memory for long
    sequences.
    """
    device = _resolve_device(model, device)
    if not _is_etas_like(model):
        raise ValueError("ETAS total fast path requires an ETAS-like model.")

    eval_times, end_eval_values, nll_event_times = _build_eval_and_end_values(sequence)
    batch = Batch.from_list([sequence]).to(device)
    end_eval = torch.as_tensor(end_eval_values, device=device, dtype=batch.arrival_times.dtype)

    with torch.inference_mode():
        end_idx = int(batch.end_idx[0].item())
        t_all = batch.arrival_times[0, :end_idx]  # real events only
        nll_mask = batch.nll_event_mask[0, :end_idx].bool()
        t_nll_events = t_all[nll_mask]

        # Event log-intensity terms, queried in blocks to cap peak memory.
        h_chunks = []
        for st in range(0, int(t_nll_events.numel()), int(query_block_size)):
            t_query = t_nll_events[st : st + int(query_block_size)].unsqueeze(0)
            h_chunk = model.h_intensity(batch, t_query=t_query)[0]
            h_chunks.append(h_chunk)
        h_events = torch.cat(h_chunks, dim=0) if h_chunks else torch.empty(0, device=device, dtype=t_all.dtype)

        if getattr(model, "bg_model", None) is not None and t_nll_events.numel() > 0:
            f_events = model.bg_model.intensity(batch, t_query=t_nll_events.unsqueeze(0))[0]
        else:
            f_events = torch.zeros_like(h_events)

        event_log_like = torch.log((h_events + f_events).clamp_min(eps))
        cum_event_log_like = (
            torch.cumsum(event_log_like, dim=0) if event_log_like.numel() > 0 else event_log_like
        )

        event_prefix, idx_nll, has_nll_event = _prefix_cumsum_at_queries(
            event_times=t_nll_events,
            cum_values=cum_event_log_like,
            query_times=end_eval,
        )

        mag_all = batch.mag[0, :end_idx]
        t0 = batch.t_nll_start[0]
        int_h_prefix = model.prefix_h_integral(
            batch=batch,
            t_all=t_all,
            mag_all=mag_all,
            t0=t0,
            t_query=end_eval,
            query_block_size=int(query_block_size),
        )

        if getattr(model, "bg_model", None) is not None:
            int_f_prefix = _bg_integral_prefix(model.bg_model, batch, end_eval_values)
        else:
            int_f_prefix = torch.zeros_like(end_eval)

        cum_ll_time = event_prefix - (int_h_prefix + int_f_prefix)
        cum_ll_total = cum_ll_time.clone()

    out = _build_full_curve_output(eval_times, cum_ll_time, cum_ll_total)
    meta = _build_curve_meta(
        curve_method="etas_total_fast",
        num_events_in_nll=nll_event_times.numel(),
        final_log_likelihood=out["cum_log_likelihood_time"][-1],
        final_log_likelihood_total=out["cum_log_likelihood_total"][-1],
        num_prefix_evals=len(eval_times),
    )
    return out, meta


def _cumulative_curve_prefix_generic(
    model,
    sequence,
    device=None,
    prefix_chunk_size=32,
    show_progress=False,
    progress_desc=None,
):
    """Generic prefix-recompute fallback for cumulative log-likelihood curves.

    Prefixes are processed in chunks, with adaptive chunk shrinking when CUDA
    OOM is encountered.
    """
    device = _resolve_device(model, device)

    chunk_size = int(prefix_chunk_size)
    chunk_size = max(1, chunk_size)

    eval_times, end_eval_values, nll_event_times = _build_eval_and_end_values(sequence)

    seq_t_start = float(sequence.t_start)

    cum_ll_time = []
    cum_ll_total = []

    pbar = None
    if show_progress:
        from tqdm.auto import tqdm

        pbar = tqdm(
            total=len(end_eval_values),
            desc=(progress_desc or "Prefix NLL"),
            unit="prefix",
            leave=False,
        )

    idx = 0
    try:
        while idx < len(end_eval_values):
            cur_chunk = min(chunk_size, len(end_eval_values) - idx)
            end_chunk = end_eval_values[idx : idx + cur_chunk]
            seq_chunk = [
                sequence.get_subsequence(start=seq_t_start, end=end_eval) for end_eval in end_chunk
            ]

            try:
                nll_time_chunk, nll_total_chunk = _model_nll_values_batch(model, seq_chunk, device=device)
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and cur_chunk > 1:
                    chunk_size = max(1, cur_chunk // 2)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise

            cum_ll_time.extend([-float(v) for v in nll_time_chunk])
            cum_ll_total.extend([-float(v) for v in nll_total_chunk])
            idx += cur_chunk

            if pbar is not None:
                pbar.update(cur_chunk)
    finally:
        if pbar is not None:
            pbar.close()

    out = _build_full_curve_output(eval_times, cum_ll_time, cum_ll_total)
    meta = _build_curve_meta(
        curve_method="prefix_nll",
        num_events_in_nll=nll_event_times.numel(),
        final_log_likelihood=cum_ll_time[-1],
        final_log_likelihood_total=cum_ll_total[-1],
        prefix_chunk_size=chunk_size,
        num_prefix_evals=len(eval_times),
    )
    return out, meta


def _promote_time_curve_to_total(out, meta):
    """Populate total-curve fields when total equals time."""
    ll = np.asarray(out["cum_log_likelihood"], dtype=float)
    out["cum_log_likelihood_time"] = ll.copy()
    out["cum_log_likelihood_total"] = ll.copy()
    out["cum_nll_time"] = -out["cum_log_likelihood_time"]
    out["cum_nll_total"] = -out["cum_log_likelihood_total"]
    meta["final_log_likelihood_total"] = float(out["cum_log_likelihood_total"][-1])
    return out, meta


def _run_prefix_curve_fallback(
    model,
    sequence,
    device,
    prefix_chunk_size,
    show_progress,
    progress_desc,
):
    """Run generic prefix-recompute curve path."""
    return _cumulative_curve_prefix_generic(
        model,
        sequence,
        device=device,
        prefix_chunk_size=prefix_chunk_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def cumulative_log_likelihood_curve(
    model,
    sequence,
    device=None,
    eps=1e-10,
    component="time",
    prefix_chunk_size=32,
    show_progress=False,
    progress_desc=None,
):
    """Compute cumulative log-likelihood curves for one event sequence.

    Args:
        model: Point-process model providing one or more supported interfaces.
        sequence: Single sequence object used to build prefix evaluations.
        device: Torch device for model execution. Defaults to model device.
        eps: Small positive constant used for numerical clamping.
        component: Which component to return, ``"time"`` or ``"total"``.
        prefix_chunk_size: Prefix batch size for the generic fallback path.
        show_progress: Whether to show a progress bar in fallback mode.
        progress_desc: Optional progress-bar description.

    Returns:
        tuple[dict, dict]:
            Curve outputs and metadata. The output dict always includes
            ``time``, ``cum_log_likelihood``, ``cum_nll``,
            ``cum_log_likelihood_time``, ``cum_nll_time``,
            ``cum_log_likelihood_total``, and ``cum_nll_total``.
    """
    if component not in {"time", "total"}:
        raise ValueError("component must be one of {'time', 'total'}.")
    device = _resolve_device(model, device)

    if component == "total":
        # Exact fast paths:
        # - ETAS family: O(N^2) block-wise vectorization instead of prefix recomputation.
        # - Recurrent+BG family: single forward + cumulative decomposition.
        if _is_etas_like(model):
            return _cumulative_curve_etas_total_fast(
                model,
                sequence,
                device=device,
                eps=eps,
            )
        if _is_recurrent_curve_model(model) and getattr(model, "bg_model", None) is not None:
            return _cumulative_curve_recurrent_total_fast(
                model,
                sequence,
                device=device,
                eps=eps,
            )

        # For recurrent models without BG terms, total == time.
        if _is_recurrent_curve_model(model) and getattr(model, "bg_model", None) is None:
            out, meta = _cumulative_curve_recurrent_fast(model, sequence, device=device, eps=eps)
            return _promote_time_curve_to_total(out, meta)

        return _run_prefix_curve_fallback(
            model,
            sequence,
            device=device,
            prefix_chunk_size=prefix_chunk_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    if _is_etas_like(model):
        return _cumulative_curve_etas_total_fast(
            model,
            sequence,
            device=device,
            eps=eps,
        )
    if _is_recurrent_curve_model(model) and getattr(model, "bg_model", None) is not None:
        return _cumulative_curve_recurrent_total_fast(
            model,
            sequence,
            device=device,
            eps=eps,
        )
    if _is_recurrent_curve_model(model):
        return _cumulative_curve_recurrent_fast(model, sequence, device=device, eps=eps)
    return _run_prefix_curve_fallback(
        model,
        sequence,
        device=device,
        prefix_chunk_size=prefix_chunk_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
