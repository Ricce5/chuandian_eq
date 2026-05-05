# Features for Small Earthquakes Can Help Predict Large Earthquakes: A Machine Learning Perspective
# Reference: https://www.mdpi.com/2076-3417/13/11/6424
import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG10_E = np.log10(np.exp(1.0))
_EMPTY_NUM_MAG = np.empty(0, dtype=float)

_DEFAULT_MAG_ELAPS = [6.0, 6.5]
_DEFAULT_MAG_ELAPS_ELAPSED = [6.0, 6.5, 7.0, 7.5]


@dataclass(frozen=True)
class FeatureConfig:
    Mc: float
    Mf: float
    Twindow: float
    Tfore: float
    dt: float
    dMag: float
    Mag_elaps: list[float]
    L_max: int
    context_len: int


@dataclass(frozen=True)
class CatalogData:
    jd: np.ndarray
    mag: np.ndarray


def _coerce_scalar_window(value, name: str) -> float:
    """Accept a scalar or a single-value sequence and return float."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a scalar or a single-value sequence, got {value}")
        value = value[0]
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _build_feature_config(
    Mc,
    Mf,
    Twindow,
    Tfore,
    dt,
    dMag,
    Mag_elaps,
    L_max,
    context_len,
) -> FeatureConfig:
    twindow = _coerce_scalar_window(Twindow, "Twindow")
    tfore = _coerce_scalar_window(Tfore, "Tfore")
    step = _coerce_scalar_window(dt, "dt")
    dmag = _coerce_scalar_window(dMag, "dMag")
    if L_max <= 0:
        raise ValueError(f"L_max must be > 0, got {L_max}")
    mag_elaps = _DEFAULT_MAG_ELAPS if Mag_elaps is None else list(Mag_elaps)
    return FeatureConfig(
        Mc=float(Mc),
        Mf=float(Mf),
        Twindow=twindow,
        Tfore=tfore,
        dt=step,
        dMag=dmag,
        Mag_elaps=mag_elaps,
        L_max=int(L_max),
        context_len=int(context_len),
    )


def _prepare_catalog_data(data_input, Mc: float) -> CatalogData:
    data = np.asarray(data_input)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data_input must be a 2D array with at least two columns [JD, mag, ...]")

    data1 = data[data[:, 1] >= Mc]
    if len(data1) == 0:
        raise ValueError("No earthquake events meet the Mc condition")

    jd = data1[:, 0].astype(float)
    mag = data1[:, 1].astype(float)

    order = np.argsort(jd)
    return CatalogData(jd=jd[order], mag=mag[order])


def _build_time_array(jd: np.ndarray, twindow: float, tfore: float, dt: float, t_arrary=None) -> np.ndarray:
    if t_arrary is not None:
        t_arrary = np.asarray(t_arrary, dtype=float)
        if t_arrary.size == 0:
            raise ValueError("t_arrary cannot be empty")
        valid = (t_arrary >= jd[0] + twindow) & (t_arrary <= jd[-1])
        if not np.all(valid):
            logger.warning("t_arrary contains values outside valid range; those values are ignored.")
        t_array_final = t_arrary[valid]
        if t_array_final.size == 0:
            raise ValueError("No valid times remained in t_arrary after range filtering.")
        return t_array_final

    span = jd[-1] - jd[0] - twindow - tfore
    if span < 0:
        raise ValueError(
            "Catalog time span is shorter than Twindow + Tfore; cannot build any forecasting samples."
        )
    n_loop = int(np.ceil(span / dt))
    if n_loop <= 0:
        raise ValueError("No forecasting samples were generated; check Twindow, Tfore and dt.")
    return twindow + jd[0] + np.arange(n_loop) * dt


def _init_feature_arrays(n_loop: int, mag_elaps: list[float], t_array_final: np.ndarray) -> dict[str, np.ndarray]:
    features = {
        "t": t_array_final.copy(),
        "Num": np.full(n_loop, np.nan),
        "Mag_max": np.full(n_loop, np.nan),
        "Mag_max_obs": np.full(n_loop, np.nan),
        "Mag_mean": np.full(n_loop, np.nan),
        "b_lsq": np.full(n_loop, np.nan),
        "a_lsq": np.full(n_loop, np.nan),
        "b_std_lsq": np.full(n_loop, np.nan),
        "std_gr_lsq": np.full(n_loop, np.nan),
        "b_mlk": np.full(n_loop, np.nan),
        "a_mlk": np.full(n_loop, np.nan),
        "b_std_mlk": np.full(n_loop, np.nan),
        "std_gr_mlk": np.full(n_loop, np.nan),
        "dM_lsq": np.full(n_loop, np.nan),
        "dM_mlk": np.full(n_loop, np.nan),
        "Energy_sqrt": np.full(n_loop, np.nan),
        "prob_x7_lsq": np.full(n_loop, np.nan),
        "prob_x7_mlk": np.full(n_loop, np.nan),
        "zvalue": np.full(n_loop, np.nan),
        "beta": np.full(n_loop, np.nan),
        "shock_len": np.zeros(n_loop),
        "Negative": np.zeros(n_loop),
    }
    for mag_threshold in mag_elaps:
        features[f"T_elaps{mag_threshold}"] = np.full(n_loop, np.nan)
    return features


def _window_range(jd: np.ndarray, start: float, end: float) -> tuple[int, int]:
    left = int(np.searchsorted(jd, start, side="left"))
    right = int(np.searchsorted(jd, end, side="left"))
    return left, right


def _compute_negative_label(jd: np.ndarray, mag: np.ndarray, t_now: float, cfg: FeatureConfig) -> float:
    shock_start, shock_end = _window_range(jd, t_now, t_now + cfg.Tfore)
    shock_len = int(np.sum(mag[shock_start:shock_end] >= cfg.Mf))

    near_start, near_end = _window_range(
        jd,
        t_now - cfg.context_len * cfg.Tfore,
        t_now + cfg.context_len * cfg.Tfore,
    )
    near_mag = mag[near_start:near_end]
    negative = (near_mag.size == 0) or (np.max(near_mag) < cfg.Mf)
    return float(shock_len), float(shock_len == 0) if cfg.context_len == 0 else float(negative)


def _fill_window_features(
    features: dict[str, np.ndarray],
    num_mag: np.ndarray,
    i: int,
    t_now: float,
    jd: np.ndarray,
    mag: np.ndarray,
    cfg: FeatureConfig,
) -> None:
    history_start, history_end = _window_range(jd, t_now - cfg.Twindow, t_now)
    sub_jd = jd[history_start:history_end]
    sub_mag = mag[history_start:history_end]

    shock_len, negative = _compute_negative_label(jd, mag, t_now, cfg)
    features["shock_len"][i] = shock_len
    features["Negative"][i] = negative

    if sub_jd.size == 0:
        return

    (
        b_lsq,
        a_lsq,
        std_gr_lsq,
        b_mlk,
        a_mlk,
        std_gr_mlk,
        dM_lsq,
        dM_mlk,
        b_std_lsq,
        b_std_mlk,
        num_mag_int,
    ) = calculate_magnitudes_and_features(sub_mag, cfg.Mc, cfg.dMag)
    t_elaps = calculate_elapsed_times(jd, t_now, mag, cfg.Mag_elaps)
    beta, zvalue = calculate_seismic_change_rate(sub_jd, cfg.Twindow, t_now)
    mag_max_obs = get_max_magnitude_in_forecast(jd, mag, t_now, cfg.Tfore)

    features["Num"][i] = sub_mag.size
    features["Mag_max"][i] = np.max(sub_mag)
    features["Mag_mean"][i] = np.mean(sub_mag)
    features["b_lsq"][i] = b_lsq
    features["a_lsq"][i] = a_lsq
    features["std_gr_lsq"][i] = std_gr_lsq
    features["b_mlk"][i] = b_mlk
    features["a_mlk"][i] = a_mlk
    features["std_gr_mlk"][i] = std_gr_mlk
    features["dM_lsq"][i] = dM_lsq
    features["dM_mlk"][i] = dM_mlk
    features["b_std_lsq"][i] = b_std_lsq
    features["b_std_mlk"][i] = b_std_mlk
    features["prob_x7_lsq"][i] = np.exp(-3 * b_lsq / LOG10_E) if np.isfinite(b_lsq) else np.nan
    features["prob_x7_mlk"][i] = np.exp(-3 * b_mlk / LOG10_E) if np.isfinite(b_mlk) else np.nan
    features["Energy_sqrt"][i] = np.sqrt(np.sum(10 ** (12 + 1.8 * sub_mag)))
    features["beta"][i] = beta
    features["zvalue"][i] = zvalue
    features["Mag_max_obs"][i] = mag_max_obs

    for j, mag_threshold in enumerate(cfg.Mag_elaps):
        features[f"T_elaps{mag_threshold}"][i] = t_elaps[j]

    max_bins = min(len(num_mag_int), cfg.L_max)
    num_mag[i, :max_bins] = num_mag_int[:max_bins]


def cal2jd(date):
    """Convert calendar date to Julian date."""
    if isinstance(date, (pd.Timestamp, datetime)):
        year = date.year
        month = date.month
        day = date.day
        hour = date.hour
        minute = date.minute
        second = date.second + date.microsecond / 1e6
    else:
        year, month, day, hour, minute, second = date

    if month <= 2:
        year -= 1
        month += 12

    a_term = math.floor(year / 100)
    b_term = 2 - a_term + math.floor(a_term / 4)
    jdn = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b_term - 1524.5
    jd = jdn + (hour - 12) / 24 + minute / 1440 + second / 86400
    return jd


def calculate_magnitudes_and_features(Mag, Mc, dMag):
    """
    Calculate b-value/a-value related features for one time window.

    Returns
    -------
    tuple
        b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk,
        dM_lsq, dM_mlk, b_std_lsq, b_std_mlk, num_mag_int
    """
    mag = np.asarray(Mag, dtype=float)
    if mag.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, _EMPTY_NUM_MAG

    mag_max = float(np.max(mag))
    mag_mean = float(np.mean(mag))

    mag_int = np.round(np.arange(Mc, mag_max + dMag, dMag), 1)
    num_mag_int = np.array([(mag >= m).sum() for m in mag_int], dtype=float)
    log_num_m = np.log10(np.maximum(num_mag_int, 1e-10))

    n = len(mag_int)
    sum_x = np.sum(mag_int)
    sum_xx = np.sum(mag_int * mag_int)
    sum_y = np.sum(log_num_m)
    sum_xy = np.sum(mag_int * log_num_m)

    denominator_b_lsq = (sum_x * sum_x) - (n * sum_xx)
    if np.isclose(denominator_b_lsq, 0.0):
        b_lsq = np.nan
        a_lsq = np.nan
    else:
        b_lsq = (n * sum_xy - sum_x * sum_y) / denominator_b_lsq
        a_lsq = (sum_y - b_lsq * sum_x) / n

    if n > 1 and np.isfinite(b_lsq) and np.isfinite(a_lsq):
        std_gr_lsq = np.sum((log_num_m - a_lsq - b_lsq * mag_int) ** 2) / (n - 1)
    else:
        std_gr_lsq = np.nan

    if mag_mean > Mc:
        b_mlk = LOG10_E / (mag_mean - Mc)
        a_mlk = np.log10(len(mag)) + b_mlk * Mc
    else:
        b_mlk = np.nan
        a_mlk = np.nan

    if n > 1 and np.isfinite(b_mlk) and np.isfinite(a_mlk):
        std_gr_mlk = np.sum((log_num_m - a_mlk - b_mlk * mag_int) ** 2) / (n - 1)
    else:
        std_gr_mlk = np.nan

    if np.isfinite(b_lsq) and not np.isclose(b_lsq, 0.0):
        dM_lsq = mag_max - (a_lsq / b_lsq)
    else:
        dM_lsq = np.nan

    if np.isfinite(b_mlk) and not np.isclose(b_mlk, 0.0):
        dM_mlk = mag_max - (a_mlk / b_mlk)
    else:
        dM_mlk = np.nan

    if n > 1:
        denominator_std = np.sum((mag_int - mag_mean) ** 2) / n / (n - 1)
        if not np.isclose(denominator_std, 0.0) and np.isfinite(denominator_std):
            b_std_lsq = 2.3 * (b_lsq ** 2) * np.sqrt(denominator_std) if np.isfinite(b_lsq) else np.nan
            b_std_mlk = 2.3 * (b_mlk ** 2) * np.sqrt(denominator_std) if np.isfinite(b_mlk) else np.nan
        else:
            b_std_lsq = np.nan
            b_std_mlk = np.nan
    else:
        b_std_lsq = np.nan
        b_std_mlk = np.nan

    return b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk, num_mag_int


def calculate_elapsed_times(jd, t, mag, Mag_elaps=None):
    """
    Calculate elapsed times since the last event above each magnitude threshold.
    """
    if Mag_elaps is None:
        Mag_elaps = _DEFAULT_MAG_ELAPS_ELAPSED

    jd = np.asarray(jd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    t_elaps = np.full(len(Mag_elaps), np.nan, dtype=float)

    index_t_elaps = np.where(jd < t)[0]
    if index_t_elaps.size == 0:
        return t_elaps

    for i, mag_threshold in enumerate(Mag_elaps):
        idx_in_window = np.where(mag[index_t_elaps] >= mag_threshold)[0]
        if idx_in_window.size == 0:
            continue
        last_idx = index_t_elaps[idx_in_window[-1]]
        t_elaps[i] = t - jd[last_idx]
    return t_elaps


def calculate_seismic_change_rate(sub_jd, Twindow, t):
    """
    Calculate seismic change rates, including z-value and beta.
    """
    Twindow = _coerce_scalar_window(Twindow, "Twindow")
    sub_jd = np.asarray(sub_jd, dtype=float)
    if sub_jd.size == 0:
        return np.nan, np.nan

    f_t_start = t - Twindow
    f_t_mid = t - 0.5 * Twindow
    f_tw = 0.5 * Twindow
    t_bin = 0.05 * Twindow

    index_r1 = np.where((sub_jd >= f_t_start) & (sub_jd < f_t_mid))[0]
    index_r2 = np.where((sub_jd >= f_t_mid) & (sub_jd < (f_t_mid + f_tw)))[0]
    n1 = len(index_r1)
    n2 = len(index_r2)

    r1 = n1 / (f_t_mid - f_t_start)
    r2 = n2 / f_tw

    n_r1, _ = np.histogram(sub_jd[index_r1], bins=np.arange(f_t_start, f_t_mid + 2 * t_bin, t_bin))
    n_r2, _ = np.histogram(sub_jd[index_r2], bins=np.arange(f_t_mid, f_t_mid + f_tw + 2 * t_bin, t_bin))

    s1 = np.var(n_r1, ddof=1) if len(n_r1) > 1 else np.nan
    s2 = np.var(n_r2, ddof=1) if len(n_r2) > 1 else np.nan

    if n1 == 0 or n2 == 0 or not np.isfinite(s1) or not np.isfinite(s2):
        zvalue = np.nan
    else:
        denominator = (s1 / n1) + (s2 / n2)
        zvalue = (r1 - r2) / np.sqrt(denominator) if denominator > 0 and np.isfinite(denominator) else np.nan

    v_r1, _ = np.histogram(sub_jd, bins=np.arange(f_t_start, f_t_mid + f_tw + 2 * t_bin, t_bin))
    n_eq1 = np.sum(v_r1)
    n_bin1 = len(v_r1)
    if n_bin1 == 0:
        return np.nan, zvalue

    win_len_days = f_tw / t_bin
    f_norm_inval_length = win_len_days / n_bin1
    beta_denom = n_eq1 * f_norm_inval_length * (1 - f_norm_inval_length)
    beta = (n2 - n_eq1 * f_norm_inval_length) / np.sqrt(beta_denom) if beta_denom > 0 else np.nan

    return beta, zvalue


def get_max_magnitude_in_forecast(jd, mag, t, Tfore):
    """
    Get the maximum magnitude within [t, t + Tfore).
    """
    index_max_mag_obs = np.where((jd >= t) & (jd < (t + Tfore)))[0]
    if len(index_max_mag_obs) > 0:
        return np.max(mag[index_max_mag_obs])
    return np.nan


def calculate_seismic_features(
    data_input,
    Mc=4.7,
    Mf=5.5,
    Twindow=20,
    Tfore=30,
    dt=30,
    dMag=0.1,
    Mag_elaps=None,
    t_arrary=None,
    L_max=60,
    context_len=2,
):
    """
    Calculate seismic features in sliding windows.

    Notes
    -----
    - Keeps legacy argument name `t_arrary` for compatibility.
    - Accepts legacy `Twindow=[20]` format by coercing a single-value sequence.
    """
    cfg = _build_feature_config(
        Mc=Mc,
        Mf=Mf,
        Twindow=Twindow,
        Tfore=Tfore,
        dt=dt,
        dMag=dMag,
        Mag_elaps=Mag_elaps,
        L_max=L_max,
        context_len=context_len,
    )
    catalog = _prepare_catalog_data(data_input, cfg.Mc)
    t_array_final = _build_time_array(catalog.jd, cfg.Twindow, cfg.Tfore, cfg.dt, t_arrary=t_arrary)

    n_loop = len(t_array_final)
    features = _init_feature_arrays(n_loop, cfg.Mag_elaps, t_array_final)
    num_mag = np.zeros((n_loop, cfg.L_max))

    for i, t_now in enumerate(t_array_final):
        _fill_window_features(
            features=features,
            num_mag=num_mag,
            i=i,
            t_now=t_now,
            jd=catalog.jd,
            mag=catalog.mag,
            cfg=cfg,
        )

    features_df = pd.DataFrame(features)
    return features_df, num_mag


def calculate_seismic_features_n(
    data_input,
    Mc=4.7,
    Mf=5.5,
    Twindow_list=None,
    Tfore=30,
    dt=30,
    dMag=0.1,
    Mag_elaps=None,
    t_arrary=None,
):
    """
    Calculate seismic features for multiple window lengths.
    """
    if Twindow_list is None:
        Twindow_list = [200]
    if Mag_elaps is None:
        Mag_elaps = _DEFAULT_MAG_ELAPS

    Twindow_list = sorted(Twindow_list, reverse=True)
    results = {}
    num_mag_all = {}
    for Twindow in Twindow_list:
        out, num_mag = calculate_seismic_features(
            data_input,
            Mc=Mc,
            Mf=Mf,
            Twindow=Twindow,
            Tfore=Tfore,
            dt=dt,
            dMag=dMag,
            Mag_elaps=Mag_elaps,
            t_arrary=t_arrary,
        )
        results[Twindow] = out
        num_mag_all[Twindow] = num_mag
        t_arrary = out["t"].values
    return results, num_mag_all
