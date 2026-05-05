from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .plot_style import apply_publication_style, format_time_axis, save_pub_figure, style_axes


DEFAULT_PLOT_COLORS: dict[str, str] = {
    "true": "#1f77b4",
    "mean": "#ff7f0e",
    "pi": "#4c78a8",
    "err": "#d62728",
    "covered": "#2ca02c",
    "below": "#e45756",
    "above": "#9467bd",
}


def style_current_figure(*, title: str | None = None):
    fig = plt.gcf()
    if title is not None and fig.axes:
        fig.axes[0].set_title(title)
    for ax in fig.axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    fig.tight_layout()
    return fig


def run_sliding_window_forecast(
    model,
    seq,
    device,
    *,
    duration: float = 12,
    slide_step: float = 12,
    start_time: float | None = None,
    end_time: float | None = None,
    quantiles: tuple[float, float] = (2.5, 97.5),
    samples_per_batch: int = 1000,
    max_sample_len: int | None = None,
    return_sim_count_matrix: bool = False,
):
    start = float(seq.arrival_times[0].item()) if start_time is None else float(start_time)
    end = float(seq.arrival_times[-1].item()) if end_time is None else float(end_time)
    if end <= start:
        raise ValueError(f"Invalid sliding range: end ({end}) must be greater than start ({start}).")
    t_forecast_list = np.arange(start + duration, end - duration, slide_step)

    counts_list: list[int] = []
    q_list: list[np.ndarray] = []
    mean_list: list[float] = []
    sim_count_rows: list[np.ndarray] = []

    model.eval()
    try:
        sample_sig = inspect.signature(model.sample).parameters
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sample_sig.values()
        )
    except Exception:
        sample_sig = {}
        accepts_var_kwargs = False

    def _supports(name: str) -> bool:
        return accepts_var_kwargs or name in sample_sig

    resolved_max_sample_len = None if max_sample_len is None else int(max_sample_len)

    pbar = tqdm(t_forecast_list, desc="Sliding window forecast")
    for t_forecast in pbar:
        pbar.set_postfix(t_forecast=f"{float(t_forecast):.3f}")
        t_end = min(t_forecast + duration, end)
        past_seq = seq.get_subsequence(seq.t_start, t_forecast).to(device)
        observed_seq = seq.get_subsequence(
            t_forecast,
            t_end,
        ).to(device)

        sample_kwargs: dict[str, Any] = {
            "batch_size": samples_per_batch,
            "duration": (t_end - t_forecast),
            "past_seq": past_seq,
            "return_sequences": True,
        }
        if resolved_max_sample_len is not None:
            if _supports("max_sample_len"):
                sample_kwargs["max_sample_len"] = resolved_max_sample_len
            if _supports("max_length"):
                sample_kwargs["max_length"] = resolved_max_sample_len

        try:
            forecasts = model.sample(**sample_kwargs)
        except RuntimeError as exc:
            if "Exceeded max_sample_len" in str(exc):
                raise RuntimeError(
                    "Exceeded max_sample_len during sliding-window sampling. "
                    f"window_start={float(t_forecast):.6f}, window_end={float(t_end):.6f}, "
                    f"configured_max_sample_len={resolved_max_sample_len}. "
                    "Increase sliding_max_sample_len."
                ) from exc
            raise
        fc_counts = np.fromiter((len(fc) for fc in forecasts), dtype=np.int32)
        q = np.percentile(fc_counts, quantiles)
        mean = float(fc_counts.mean())

        q_list.append(q)
        mean_list.append(mean)
        counts_list.append(len(observed_seq))
        if return_sim_count_matrix:
            sim_count_rows.append(fc_counts)

    t_forecast_arr = np.array(t_forecast_list)
    counts_arr = np.array(counts_list)
    q_arr = np.array(q_list)
    mean_arr = np.array(mean_list)

    if return_sim_count_matrix:
        if sim_count_rows:
            sim_count_matrix = np.stack(sim_count_rows, axis=0)
        else:
            sim_count_matrix = np.empty((0, int(samples_per_batch)), dtype=np.int32)
        return t_forecast_arr, counts_arr, q_arr, mean_arr, sim_count_matrix

    return t_forecast_arr, counts_arr, q_arr, mean_arr


def compute_mean_nb_log_prob(obs_counts, sim_count_matrix, *, eps: float = 1e-10):
    """
    Fit a per-bin negative binomial (method of moments) and compute
    LP_NB = mean_i log P(N_obs^i | theta_hat_i).
    """
    obs = np.asarray(obs_counts, dtype=np.int64).reshape(-1)
    sim = np.asarray(sim_count_matrix, dtype=np.float64)

    if sim.ndim != 2:
        raise ValueError("sim_count_matrix must be a 2D array with shape (n_bins, n_samples).")
    if sim.shape[0] != obs.shape[0]:
        raise ValueError("obs_counts and sim_count_matrix must have the same number of bins.")
    if sim.shape[1] == 0:
        raise ValueError("sim_count_matrix must contain at least one simulated sample per bin.")

    mu = np.mean(sim, axis=1)
    var = np.var(sim, axis=1, ddof=1) if sim.shape[1] > 1 else mu.copy()

    log_prob_bins = np.full(obs.shape[0], np.nan, dtype=np.float64)

    zero_mu = mu <= eps
    obs_zero = obs == 0
    log_prob_bins[zero_mu & obs_zero] = 0.0
    log_prob_bins[zero_mu & (~obs_zero)] = -np.inf

    active = ~zero_mu
    if np.any(active):
        mu_active = np.clip(mu[active], eps, None)
        var_active = np.maximum(var[active], mu_active + eps)
        total_count = np.clip((mu_active * mu_active) / (var_active - mu_active), eps, 1e12)

        k_t = torch.as_tensor(obs[active], dtype=torch.float64)
        mu_t = torch.as_tensor(mu_active, dtype=torch.float64)
        total_count_t = torch.as_tensor(total_count, dtype=torch.float64)

        # NB parameterized by mean mu and total_count r:
        # log PMF = lgamma(k+r)-lgamma(r)-lgamma(k+1)
        #           + r*log(r/(r+mu)) + k*log(mu/(r+mu))
        log_prob_active = (
            torch.lgamma(k_t + total_count_t)
            - torch.lgamma(total_count_t)
            - torch.lgamma(k_t + 1.0)
            + total_count_t * (torch.log(total_count_t) - torch.log(total_count_t + mu_t))
            + k_t * (torch.log(mu_t) - torch.log(total_count_t + mu_t))
        )
        log_prob_bins[active] = log_prob_active.cpu().numpy()

    lp_nb = float(np.mean(log_prob_bins))
    return lp_nb, log_prob_bins


def to_absolute_time_axis(rel_times, base_ts, freq_td):
    rel = np.asarray(rel_times, dtype=float).reshape(-1)
    delta = pd.to_timedelta(rel * freq_td.total_seconds(), unit="s")
    return pd.DatetimeIndex(pd.Timestamp(base_ts) + delta)


def resolve_catalog_time_reference(catalog_ds):
    md = getattr(catalog_ds, "metadata", {})
    if isinstance(md, dict):
        start_val = md.get("start_ts")
        freq_val = md.get("freq")
    else:
        start_val = getattr(md, "start_ts", None)
        freq_val = getattr(md, "freq", None)

    if start_val is None or freq_val is None:
        raise KeyError("catalog_ds.metadata must contain 'start_ts' and 'freq'.")

    return pd.to_datetime(start_val), pd.to_timedelta(freq_val)


def compute_display_upper_cap(q_low, q_high, counts, mean, pct: float = 80):
    q_ref = np.asarray(q_high, dtype=float).reshape(-1)
    robust_cap = float(np.nanpercentile(q_ref, pct))
    anchor = float(np.nanmax(np.concatenate([q_low.reshape(-1), counts.reshape(-1), mean.reshape(-1)])))
    return max(robust_cap, anchor * 1.10, 1.0)


def infer_step_timedelta(ts_index, fallback_td):
    ts_index = pd.DatetimeIndex(ts_index)
    fallback_td = pd.to_timedelta(fallback_td)
    if len(ts_index) >= 2:
        deltas = pd.Series(ts_index).diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        if not deltas.empty:
            return deltas.median()
    return fallback_td if fallback_td > pd.Timedelta(0) else pd.Timedelta(days=1)


def resolve_time_bounds(ts_index, fallback_td):
    ts_index = pd.DatetimeIndex(ts_index)
    step_td = infer_step_timedelta(ts_index, fallback_td)

    if len(ts_index) == 0:
        now = pd.Timestamp.today().normalize()
        return now, now + step_td, step_td

    left = ts_index.min() - step_td * 0.35
    right = ts_index.max() + step_td * 0.65
    if right <= left:
        right = left + step_td
    return left, right, step_td


def build_post_step(ts_index, values, step_td):
    ts_index = pd.DatetimeIndex(ts_index)
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(ts_index) == 0:
        return ts_index, values
    if len(ts_index) == 1:
        ts_step = pd.DatetimeIndex([ts_index[0], ts_index[0] + step_td])
        v_step = np.array([values[0], values[0]], dtype=float)
        return ts_step, v_step

    ts_step = ts_index.append(pd.DatetimeIndex([ts_index[-1] + step_td]))
    v_step = np.r_[values, values[-1]]
    return ts_step, v_step


def resolve_view_mode(config_value, *, auto_use_zoom: bool):
    mode = str(config_value).strip().lower()
    if mode not in {"zoomed", "full", "auto"}:
        mode = "auto"
    if mode == "auto":
        return "zoomed" if auto_use_zoom else "full"
    return mode


SLIDING_CACHE_VERSION = 1
DEFAULT_SLIDING_CACHE_FILENAME = "sliding_window_cache.npz"


def build_sliding_cache_metadata(
    seq,
    *,
    duration: float,
    slide_step: float,
    sliding_start: float,
    sliding_end: float,
    quantiles: tuple[float, float],
    samples_per_batch: int,
    max_sample_len: int | None,
):
    return {
        "cache_version": int(SLIDING_CACHE_VERSION),
        "duration": float(duration),
        "slide_step": float(slide_step),
        "quantile_low": float(quantiles[0]),
        "quantile_high": float(quantiles[1]),
        "samples_per_batch": int(samples_per_batch),
        "max_sample_len": int(max_sample_len) if max_sample_len is not None else -1,
        "seq_start": float(sliding_start),
        "seq_end": float(sliding_end),
    }


def save_sliding_window_cache(
    cache_path,
    *,
    metadata,
    t_forecast_list,
    counts_list,
    q_list,
    mean_list,
    sim_count_matrix,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        cache_version=np.int64(metadata["cache_version"]),
        duration=np.float64(metadata["duration"]),
        slide_step=np.float64(metadata["slide_step"]),
        quantile_low=np.float64(metadata["quantile_low"]),
        quantile_high=np.float64(metadata["quantile_high"]),
        samples_per_batch=np.int64(metadata["samples_per_batch"]),
        max_sample_len=np.int64(metadata["max_sample_len"]),
        seq_start=np.float64(metadata["seq_start"]),
        seq_end=np.float64(metadata["seq_end"]),
        t_forecast_list=np.asarray(t_forecast_list, dtype=np.float64),
        counts_list=np.asarray(counts_list, dtype=np.int64),
        q_list=np.asarray(q_list, dtype=np.float64),
        mean_list=np.asarray(mean_list, dtype=np.float64),
        sim_count_matrix=np.asarray(sim_count_matrix, dtype=np.int32),
    )
    return cache_path


def load_sliding_window_cache_if_compatible(cache_path, *, metadata, atol: float = 1e-9):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    required_keys = {
        "cache_version",
        "duration",
        "slide_step",
        "quantile_low",
        "quantile_high",
        "samples_per_batch",
        "max_sample_len",
        "seq_start",
        "seq_end",
        "t_forecast_list",
        "counts_list",
        "q_list",
        "mean_list",
        "sim_count_matrix",
    }

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if not required_keys.issubset(set(data.files)):
                return None

            if int(np.asarray(data["cache_version"]).item()) != int(metadata["cache_version"]):
                return None
            if int(np.asarray(data["samples_per_batch"]).item()) != int(metadata["samples_per_batch"]):
                return None
            if int(np.asarray(data["max_sample_len"]).item()) != int(metadata["max_sample_len"]):
                return None

            float_keys = ("duration", "slide_step", "quantile_low", "quantile_high", "seq_start", "seq_end")
            for k in float_keys:
                cache_val = float(np.asarray(data[k]).item())
                if not np.isclose(cache_val, float(metadata[k]), atol=atol, rtol=0.0):
                    return None

            t_forecast_list = np.asarray(data["t_forecast_list"])
            counts_list = np.asarray(data["counts_list"])
            q_list = np.asarray(data["q_list"])
            mean_list = np.asarray(data["mean_list"])
            sim_count_matrix = np.asarray(data["sim_count_matrix"])
    except Exception:
        return None

    n_bins = int(counts_list.shape[0])
    if t_forecast_list.shape[0] != n_bins:
        return None
    if mean_list.shape[0] != n_bins:
        return None
    if q_list.ndim != 2 or q_list.shape[0] != n_bins or q_list.shape[1] < 2:
        return None
    if sim_count_matrix.ndim != 2 or sim_count_matrix.shape[0] != n_bins:
        return None

    return t_forecast_list, counts_list, q_list, mean_list, sim_count_matrix


def evaluate_sliding_window_forecast_plots(
    *,
    model,
    seq,
    device,
    catalog_ds,
    checkpoint_dir,
    sliding_duration: float,
    sliding_step: float,
    sliding_start: float | None = None,
    sliding_end: float | None = None,
    sliding_quantiles: tuple[float, float],
    samples_per_batch: int,
    sliding_max_sample_len: int | None = None,
    sliding_view_mode: str = "auto",
    load_sliding_cache: bool = True,
    force_recompute_sliding: bool = False,
    sliding_cache_filename: str = DEFAULT_SLIDING_CACHE_FILENAME,
    plot_colors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    colors = dict(DEFAULT_PLOT_COLORS)
    if plot_colors is not None:
        colors.update(plot_colors)

    checkpoint_dir = Path(checkpoint_dir)
    resolved_sliding_start = (
        float(seq.arrival_times[0].item())
        if sliding_start is None
        else float(sliding_start)
    )
    resolved_sliding_end = (
        float(seq.arrival_times[-1].item())
        if sliding_end is None
        else float(sliding_end)
    )
    if resolved_sliding_end <= resolved_sliding_start:
        raise ValueError(
            "sliding_end must be greater than sliding_start, got "
            f"{resolved_sliding_end} <= {resolved_sliding_start}."
        )

    sliding_cache_path = checkpoint_dir / str(sliding_cache_filename)
    cache_meta = build_sliding_cache_metadata(
        seq,
        duration=sliding_duration,
        slide_step=sliding_step,
        sliding_start=resolved_sliding_start,
        sliding_end=resolved_sliding_end,
        quantiles=sliding_quantiles,
        samples_per_batch=samples_per_batch,
        max_sample_len=sliding_max_sample_len,
    )

    loaded_from_cache = False
    cached_payload = None
    if load_sliding_cache and not force_recompute_sliding:
        cached_payload = load_sliding_window_cache_if_compatible(
            sliding_cache_path,
            metadata=cache_meta,
        )

    if cached_payload is not None:
        t_forecast_list, counts_list, q_list, mean_list, sim_count_matrix = cached_payload
        loaded_from_cache = True
    else:
        t_forecast_list, counts_list, q_list, mean_list, sim_count_matrix = run_sliding_window_forecast(
            model=model,
            seq=seq,
            device=device,
            duration=sliding_duration,
            slide_step=sliding_step,
            start_time=resolved_sliding_start,
            end_time=resolved_sliding_end,
            quantiles=sliding_quantiles,
            samples_per_batch=samples_per_batch,
            max_sample_len=sliding_max_sample_len,
            return_sim_count_matrix=True,
        )
        if load_sliding_cache:
            save_sliding_window_cache(
                sliding_cache_path,
                metadata=cache_meta,
                t_forecast_list=t_forecast_list,
                counts_list=counts_list,
                q_list=q_list,
                mean_list=mean_list,
                sim_count_matrix=sim_count_matrix,
            )

    if len(t_forecast_list) == 0:
        return {
            "status": "empty",
            "message": "No sliding windows available with current duration/step settings.",
            "sliding_cache_path": str(sliding_cache_path),
            "sliding_loaded_from_cache": loaded_from_cache,
        }

    q_low, q_high = q_list[:, 0], q_list[:, 1]
    covered = (counts_list >= q_low) & (counts_list <= q_high)
    coverage = float(covered.mean())
    mae = float(np.mean(np.abs(counts_list - mean_list)))
    rmse = float(np.sqrt(np.mean((counts_list - mean_list) ** 2)))
    lp_nb, _ = compute_mean_nb_log_prob(counts_list, sim_count_matrix)

    base_start_ts, freq_td_local = resolve_catalog_time_reference(catalog_ds)
    seq_start_rel = float(getattr(seq, "t_start", 0.0))
    seq_start_ts = pd.Timestamp(base_start_ts) + pd.to_timedelta(
        seq_start_rel * freq_td_local.total_seconds(),
        unit="s",
    )
    t_forecast_ts = to_absolute_time_axis(t_forecast_list, seq_start_ts, freq_td_local)

    default_step_td = pd.to_timedelta(
        max(float(sliding_step), 1e-6) * freq_td_local.total_seconds(),
        unit="s",
    )
    plot_x_left, plot_x_right, inferred_step_td = resolve_time_bounds(
        t_forecast_ts,
        default_step_td,
    )

    # 1) Counts over time
    display_cap = compute_display_upper_cap(q_low, q_high, counts_list, mean_list, pct=80)
    base_bottom = min(0.0, float(np.nanmin(q_low)) * 1.05)
    zoom_top = display_cap * 1.05
    full_top = max(float(np.nanmax(q_high)) * 1.05, zoom_top)
    counts_mode = resolve_view_mode(sliding_view_mode, auto_use_zoom=(full_top > zoom_top * 1.8))

    if counts_mode == "zoomed":
        q_high_plot = np.minimum(q_high, display_cap)
        clipped_mask = q_high > display_cap
        counts_ylim_top = zoom_top
        counts_title = "Counts: true vs forecast (zoomed)"
        cap_note = f" | clipped={int(np.count_nonzero(clipped_mask))}" if np.any(clipped_mask) else ""
    else:
        q_high_plot = q_high
        clipped_mask = np.zeros_like(q_high, dtype=bool)
        counts_ylim_top = full_top
        counts_title = "Counts: true vs forecast (full range)"
        cap_note = ""

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(
        t_forecast_ts,
        counts_list,
        label="true counts",
        color=colors["true"],
        linewidth=1.1,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    ax.plot(
        t_forecast_ts,
        mean_list,
        label="forecast mean",
        color=colors["mean"],
        linewidth=1.1,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    ax.fill_between(
        t_forecast_ts,
        q_low,
        q_high_plot,
        alpha=0.14,
        color=colors["pi"],
        label="forecast PI (2.5-97.5)",
    )
    if np.any(clipped_mask):
        ax.scatter(
            t_forecast_ts[clipped_mask],
            np.full(int(clipped_mask.sum()), display_cap),
            marker="^",
            s=20,
            color=colors["pi"],
            label="PI upper clipped (display)",
            zorder=4,
        )
    style_axes(ax, ylabel="count in next window", integer_y=True)
    format_time_axis(ax)
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_ylim(base_bottom, counts_ylim_top)
    ax.set_title(counts_title)
    ax.text(
        0.01,
        0.02,
        f"Coverage: {coverage:.2%} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | LP_NB: {lp_nb:.4f}{cap_note}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    save_pub_figure(fig, checkpoint_dir / "forecast_counts_over_time.png")
    plt.show()

    # 2) Error over time
    err = counts_list - mean_list
    bias = float(np.mean(err))
    err_abs = np.abs(err)
    err_zoom_lim = max(float(np.nanpercentile(err_abs, 80)) * 1.2, 1.0)
    err_full_lim = max(float(np.max(err_abs)) * 1.05, err_zoom_lim)
    error_mode = resolve_view_mode(sliding_view_mode, auto_use_zoom=(err_full_lim > err_zoom_lim * 1.8))

    clip_lim = err_zoom_lim if error_mode == "zoomed" else None
    if clip_lim is None:
        err_plot = err
        clip_hi = np.zeros(err.shape, dtype=bool)
        clip_lo = np.zeros(err.shape, dtype=bool)
        err_title = "Forecast error over time (full range)"
        err_ylim = err_full_lim
    else:
        err_plot = np.clip(err, -clip_lim, clip_lim)
        clip_hi = err > clip_lim
        clip_lo = err < -clip_lim
        err_title = "Forecast error over time (zoomed)"
        err_ylim = clip_lim

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(
        t_forecast_ts,
        err_plot,
        label="true - mean",
        color=colors["err"],
        linewidth=1.0,
        marker="o",
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    if clip_lim is not None and np.any(clip_hi):
        ax.scatter(
            t_forecast_ts[clip_hi],
            np.full(int(clip_hi.sum()), clip_lim),
            marker="^",
            s=20,
            color=colors["err"],
            zorder=4,
        )
    if clip_lim is not None and np.any(clip_lo):
        ax.scatter(
            t_forecast_ts[clip_lo],
            np.full(int(clip_lo.sum()), -clip_lim),
            marker="v",
            s=20,
            color=colors["err"],
            zorder=4,
        )

    clipped_err_points = int(np.count_nonzero(clip_hi | clip_lo))
    clip_note = f" | clipped={clipped_err_points}" if clipped_err_points > 0 else ""
    ax.axhline(0, linewidth=0.9, color="gray", linestyle="--")
    style_axes(ax, ylabel="error")
    format_time_axis(ax)
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_ylim(-err_ylim, err_ylim)
    ax.set_title(err_title)
    ax.text(
        0.01,
        0.02,
        f"Bias: {bias:.2f} | RMSE: {rmse:.2f}{clip_note}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_pub_figure(fig, checkpoint_dir / "forecast_error_over_time.png")
    plt.show()

    # 3) Coverage over time
    covered_i = covered.astype(int)
    coverage_t_step, coverage_v_step = build_post_step(t_forecast_ts, covered_i, inferred_step_td)
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.step(
        coverage_t_step,
        coverage_v_step,
        where="post",
        label=f"covered (avg={coverage:.3f})",
        color=colors["covered"],
        linewidth=1.0,
    )
    ax.scatter(t_forecast_ts, covered_i, s=16, color=colors["covered"], zorder=3)
    ax.fill_between(
        coverage_t_step,
        np.zeros_like(coverage_v_step),
        coverage_v_step,
        step="post",
        alpha=0.12,
        color=colors["covered"],
    )
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_title("Coverage over time")
    style_axes(ax, ylabel="in PI")
    format_time_axis(ax)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_pub_figure(fig, checkpoint_dir / "forecast_pi_coverage_over_time.png")
    plt.show()

    # 4) Observed vs forecast scatter
    out_of_lower = counts_list < q_low
    out_of_upper = counts_list > q_high
    scatter_min = min(float(counts_list.min()), float(mean_list.min()))
    scatter_max = max(float(counts_list.max()), float(mean_list.max()))
    scatter_zoom_max = float(np.nanpercentile(np.concatenate([counts_list, mean_list]), 85)) * 1.10
    scatter_zoom_max = max(scatter_zoom_max, 1.0)
    scatter_mode = resolve_view_mode(
        sliding_view_mode,
        auto_use_zoom=(scatter_max > scatter_zoom_max * 1.8),
    )

    def draw_scatter(ax, title, xlim=None, ylim=None):
        ax.scatter(
            counts_list,
            mean_list,
            color=colors["true"],
            alpha=0.7,
            s=28,
            label="true vs mean",
        )
        ax.scatter(
            counts_list[out_of_lower],
            mean_list[out_of_lower],
            color=colors["below"],
            marker="v",
            s=40,
            label="below PI",
        )
        ax.scatter(
            counts_list[out_of_upper],
            mean_list[out_of_upper],
            color=colors["above"],
            marker="^",
            s=40,
            label="above PI",
        )
        min_v = scatter_min if xlim is None else min(xlim[0], scatter_min)
        max_v = scatter_max if xlim is None else max(xlim[1], scatter_max)
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray", linewidth=1.0, label="ideal y=x")
        style_axes(ax, xlabel="Observed count", ylabel="Forecast mean", integer_y=True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    if scatter_mode == "zoomed":
        zoom_low = scatter_min - max(0.03 * scatter_zoom_max, 0.5)
        zoom_high = scatter_zoom_max
        draw_scatter(
            ax,
            f"Observed vs Forecast (zoomed) | Coverage: {coverage:.2%}",
            xlim=(zoom_low, zoom_high),
            ylim=(zoom_low, zoom_high),
        )
    else:
        draw_scatter(ax, f"Observed vs Forecast Counts (full range) | Coverage: {coverage:.2%}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_pub_figure(fig, checkpoint_dir / "obs_vs_forecast_scatter.png")
    plt.show()

    return {
        "status": "ok",
        "t_forecast_list": t_forecast_list,
        "counts_list": counts_list,
        "q_list": q_list,
        "mean_list": mean_list,
        "coverage": coverage,
        "mae": mae,
        "rmse": rmse,
        "lp_nb": lp_nb,
        "sliding_cache_path": str(sliding_cache_path),
        "sliding_loaded_from_cache": loaded_from_cache,
    }
