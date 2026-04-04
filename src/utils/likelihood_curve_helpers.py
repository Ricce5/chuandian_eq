import contextlib
import inspect
import io

import numpy as np
import torch

from src.data import Batch


def _build_nll_kwargs(model):
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
    if isinstance(out, dict):
        time_val = out.get("time", out.get("total"))
        total_val = out.get("total", time_val)
        return time_val, total_val
    return out, out


def _model_nll_values_batch(model, sequences, device=None):
    if len(sequences) == 0:
        return [], []

    if device is None:
        device = next(model.parameters()).device

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
    time_list, total_list = _model_nll_values_batch(model, [sequence], device=device)
    return float(time_list[0]), float(total_list[0])


def _cumulative_curve_recurrent_fast(model, sequence, device=None, eps=1e-10):
    if device is None:
        device = next(model.parameters()).device

    batch = Batch.from_list([sequence]).to(device)

    with torch.inference_mode():
        context = model.get_context(batch)

        with contextlib.redirect_stdout(io.StringIO()):
            inter_time_dist = model.get_inter_time_dist(context)

        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(eps))[0]
        nll_mask = batch.nll_event_mask[0].bool()

        event_times = batch.arrival_times[0][nll_mask]
        event_log_like = log_pdf[nll_mask]
        cum_event_log_like = torch.cumsum(event_log_like, dim=0)

        arange = torch.arange(batch.batch_size, device=device)

        last_surv_context = context[arange, batch.end_idx, :]
        with contextlib.redirect_stdout(io.StringIO()):
            last_surv_dist = model.get_inter_time_dist(last_surv_context)
        last_surv_val = last_surv_dist.log_survival(batch.inter_times[arange, batch.end_idx].clamp_min(eps))
        last_log_surv = last_surv_val.reshape(-1)[0]

        offset_log_like = torch.tensor(0.0, device=device)
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            with contextlib.redirect_stdout(io.StringIO()):
                prev_surv_dist = model.get_inter_time_dist(prev_surv_context)
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

    out = {
        "time": np.asarray(times, dtype=float),
        "cum_log_likelihood": np.asarray(cum_ll, dtype=float),
    }
    out["cum_nll"] = -out["cum_log_likelihood"]

    meta = {
        "num_events_in_nll": int(nll_mask.sum().item()),
        "offset_log_likelihood": float(offset_log_like.detach().cpu().item()),
        "terminal_log_survival": float(last_log_surv.detach().cpu().item()),
        "final_log_likelihood": float(total_log_like.detach().cpu().item()),
        "curve_method": "recurrent_fast",
    }

    return out, meta


def _cumulative_curve_prefix_generic(
    model,
    sequence,
    device=None,
    prefix_chunk_size=32,
    show_progress=False,
    progress_desc=None,
):
    if device is None:
        device = next(model.parameters()).device

    chunk_size = int(prefix_chunk_size)
    chunk_size = max(1, chunk_size)

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

    cum_ll_time = []
    cum_ll_total = []

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=len(end_eval_values),
                desc=(progress_desc or "Prefix NLL"),
                unit="prefix",
                leave=False,
            )
        except Exception:
            pbar = None

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

    out = {
        "time": np.asarray(eval_times, dtype=float),
        "cum_log_likelihood": np.asarray(cum_ll_time, dtype=float),
        "cum_log_likelihood_time": np.asarray(cum_ll_time, dtype=float),
        "cum_log_likelihood_total": np.asarray(cum_ll_total, dtype=float),
    }
    out["cum_nll"] = -out["cum_log_likelihood"]
    out["cum_nll_time"] = -out["cum_log_likelihood_time"]
    out["cum_nll_total"] = -out["cum_log_likelihood_total"]

    meta = {
        "num_events_in_nll": int(nll_event_times.numel()),
        "final_log_likelihood": float(cum_ll_time[-1]),
        "final_log_likelihood_total": float(cum_ll_total[-1]),
        "curve_method": "prefix_nll",
        "prefix_chunk_size": int(chunk_size),
        "num_prefix_evals": int(len(eval_times)),
    }
    return out, meta


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
    if component not in {"time", "total"}:
        raise ValueError("component must be one of {'time', 'total'}.")

    if component == "total":
        return _cumulative_curve_prefix_generic(
            model,
            sequence,
            device=device,
            prefix_chunk_size=prefix_chunk_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    if hasattr(model, "get_context") and hasattr(model, "get_inter_time_dist"):
        return _cumulative_curve_recurrent_fast(model, sequence, device=device, eps=eps)
    return _cumulative_curve_prefix_generic(
        model,
        sequence,
        device=device,
        prefix_chunk_size=prefix_chunk_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def reference_nll_from_model(model, sequence, device=None):
    return _model_nll_values(model, sequence, device=device)
