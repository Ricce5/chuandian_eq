import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import ticker as mticker


def run_sliding_window_forecast(
    model,
    seq,
    device,
    duration=12,
    slide_step=12,
    quantiles=(2.5, 97.5),
    samples_per_batch=1000,
):
    start = float(seq.arrival_times[0].item())
    end = float(seq.arrival_times[-1].item())

    t_forecast_list = np.arange(start + duration, end - duration, slide_step)
    print(f"t_forecast_list: {t_forecast_list}")

    counts_list = []
    q_list = []
    mean_list = []

    model.eval()

    for t_forecast in t_forecast_list:
        t_end = min(t_forecast + duration, end)

        past_seq = seq.get_subsequence(0, t_forecast, reset_t_nll_to_end=True).to(device)
        observed_seq = seq.get_subsequence(t_forecast, t_end, reset_t_nll_to_end=True).to(device)

        forecasts = model.sample(
            batch_size=samples_per_batch,
            duration=(t_end - t_forecast),
            past_seq=past_seq,
            return_sequences=True,
        )
        fc_counts = np.fromiter((len(fc) for fc in forecasts), dtype=np.int32)

        q = np.percentile(fc_counts, quantiles)
        mean = float(fc_counts.mean())

        q_list.append(q)
        mean_list.append(mean)
        counts_list.append(len(observed_seq))

    return np.array(t_forecast_list), np.array(counts_list), np.array(q_list), np.array(mean_list)


def to_absolute_time_axis(rel_times, base_ts, freq_td):
    rel = np.asarray(rel_times, dtype=float).reshape(-1)
    delta = pd.to_timedelta(rel * freq_td.total_seconds(), unit="s")
    return pd.DatetimeIndex(pd.Timestamp(base_ts) + delta)


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=0)


def style_axes(ax, xlabel="Forecast start date", ylabel=None, integer_y=False):
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if integer_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


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


def compute_display_upper_cap(q_low, q_high, counts, mean, pct=80):
    q_ref = np.asarray(q_high, dtype=float).reshape(-1)
    robust_cap = float(np.nanpercentile(q_ref, pct))
    anchor = float(
        np.nanmax(np.concatenate([q_low.reshape(-1), counts.reshape(-1), mean.reshape(-1)]))
    )
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


def resolve_view_mode(config_value, *, auto_use_zoom):
    mode = str(config_value).strip().lower()
    if mode not in {"zoomed", "full", "auto"}:
        print(f"Invalid SLIDING_VIEW_MODE={config_value!r}; fallback to 'auto'.")
        mode = "auto"
    if mode == "auto":
        return "zoomed" if auto_use_zoom else "full"
    return mode
