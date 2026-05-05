from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.data.event_loader as event_loader
import src.features.seismic_features as seismic_features
from src.data.preprocessing import load_and_filter_catalog
from src.utils.file_utils import save_or_load_data

TASK_PREFIX_BY_TASK_TYPE = {
    "classification": "classifier",
    "regression": "regressor",
    "count": "counter",
}

FEATURE_CONSTRUCTION_FULL_WINDOW = "full_window"
FEATURE_CONSTRUCTION_INNER_SLIDING = "inner_sliding"
FEATURE_CONSTRUCTION_EXTERNAL_SLIDING = "external_sliding"
T_ELAPS_MODE_WINDOW = "window"
T_ELAPS_MODE_GLOBAL = "global"


@dataclass(frozen=True)
class EventWindowBundle:
    df: pd.DataFrame
    samples_list: list[dict[str, Any]]
    array_dict: dict[str, dict[str, list[np.ndarray]]]


def resolve_task_prefix(task_type: str) -> str:
    if task_type not in TASK_PREFIX_BY_TASK_TYPE:
        supported = ", ".join(sorted(TASK_PREFIX_BY_TASK_TYPE))
        raise ValueError(f"Unsupported task_type='{task_type}', supported: {supported}")
    return TASK_PREFIX_BY_TASK_TYPE[task_type]


def load_event_windows_with_cache(
    base_dir: str | Path,
    *,
    Mc: float,
    Mf: float,
    Twindow: float,
    Tfore: float,
    dt: float,
    context_len: int,
    task_prefix: str,
) -> EventWindowBundle:
    base_dir = str(base_dir)
    df = load_and_filter_catalog(base_dir, Mc=Mc)
    df_nl, _ = event_loader.normalize_df(df)

    def generate_data():
        samples_list, array_dict = event_loader.construct_samples_list(
            df,
            df_nl,
            Mc=Mc,
            Mf=Mf,
            Twindow=Twindow,
            Tfore=Tfore,
            dt=dt,
            context_len=context_len,
        )
        return {
            "samples_list": samples_list,
            "array_dict": array_dict,
        }

    cache_key_kwargs = {
        "Mc": Mc,
        "Mf": Mf,
        "Twindow": Twindow,
        "Tfore": Tfore,
        "dt": dt,
        "context_len": context_len,
    }

    cached = save_or_load_data(
        base_path=base_dir,
        generate_fn=generate_data,
        sub_dir=task_prefix,
        prefix=task_prefix,
        **cache_key_kwargs,
    )
    return EventWindowBundle(df=df, samples_list=cached["samples_list"], array_dict=cached["array_dict"])


def load_event_windows_for_task(
    base_dir: str | Path,
    *,
    task_type: str,
    Mc: float,
    Mf: float,
    Twindow: float,
    Tfore: float,
    dt: float,
    context_len: int,
) -> EventWindowBundle:
    task_prefix = resolve_task_prefix(task_type)
    return load_event_windows_with_cache(
        base_dir=base_dir,
        Mc=Mc,
        Mf=Mf,
        Twindow=Twindow,
        Tfore=Tfore,
        dt=dt,
        context_len=context_len,
        task_prefix=task_prefix,
    )


def build_classification_targets(array_dict, Mf: float):
    return event_loader.build_classification_labels(array_dict, Mf)


def _resolve_t_elaps_mode(t_elaps_mode: str) -> str:
    mode = str(t_elaps_mode).strip().lower()
    if mode not in {T_ELAPS_MODE_WINDOW, T_ELAPS_MODE_GLOBAL}:
        raise ValueError(
            f"Unsupported t_elaps_mode={t_elaps_mode!r}. "
            f"Supported: {T_ELAPS_MODE_WINDOW!r}, {T_ELAPS_MODE_GLOBAL!r}."
        )
    return mode


def _extract_elapsed_thresholds(feature_cols: list[str]) -> dict[str, float]:
    thresholds = {}
    for col in feature_cols:
        if not col.startswith("T_elaps"):
            continue
        try:
            thresholds[col] = float(col.replace("T_elaps", ""))
        except ValueError:
            continue
    return thresholds


def _compute_window_feature_map(
    history_t: np.ndarray,
    history_mag: np.ndarray,
    *,
    feature_cols: list[str],
    Mc: float,
    dMag: float,
    t_reference: float,
    elapsed_thresholds: dict[str, float],
    t_elaps_mode: str,
    global_t: np.ndarray | None = None,
    global_mag: np.ndarray | None = None,
) -> dict[str, float]:
    feature_map = {col: np.nan for col in feature_cols}
    if history_mag.size == 0:
        return feature_map

    if "Num" in feature_map:
        feature_map["Num"] = history_mag.size
    if "Mag_max" in feature_map:
        feature_map["Mag_max"] = np.max(history_mag)
    if "Mag_mean" in feature_map:
        feature_map["Mag_mean"] = np.mean(history_mag)

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
        _num_mag_int,
    ) = seismic_features.calculate_magnitudes_and_features(history_mag, Mc=float(Mc), dMag=float(dMag))

    scalar_map = {
        "b_lsq": b_lsq,
        "a_lsq": a_lsq,
        "std_gr_lsq": std_gr_lsq,
        "b_mlk": b_mlk,
        "a_mlk": a_mlk,
        "std_gr_mlk": std_gr_mlk,
        "dM_lsq": dM_lsq,
        "dM_mlk": dM_mlk,
        "b_std_lsq": b_std_lsq,
        "b_std_mlk": b_std_mlk,
        "prob_x7_lsq": np.exp(-3 * b_lsq / seismic_features.LOG10_E) if np.isfinite(b_lsq) else np.nan,
        "prob_x7_mlk": np.exp(-3 * b_mlk / seismic_features.LOG10_E) if np.isfinite(b_mlk) else np.nan,
        "Energy_sqrt": np.sqrt(np.sum(10 ** (12 + 1.8 * history_mag))),
    }
    for key, value in scalar_map.items():
        if key in feature_map:
            feature_map[key] = value

    if "beta" in feature_map or "zvalue" in feature_map:
        twindow_eff = float(t_reference - history_t[0]) if history_t.size > 0 else np.nan
        if np.isfinite(twindow_eff) and twindow_eff > 0:
            beta, zvalue = seismic_features.calculate_seismic_change_rate(
                history_t,
                Twindow=twindow_eff,
                t=t_reference,
            )
        else:
            beta, zvalue = np.nan, np.nan
        if "beta" in feature_map:
            feature_map["beta"] = beta
        if "zvalue" in feature_map:
            feature_map["zvalue"] = zvalue

    if elapsed_thresholds:
        mode = _resolve_t_elaps_mode(t_elaps_mode)
        if mode == T_ELAPS_MODE_GLOBAL:
            if global_t is None or global_mag is None:
                raise ValueError(
                    "global_t/global_mag are required when t_elaps_mode='global'."
                )
            elapsed_t = np.asarray(global_t, dtype=float)
            elapsed_mag = np.asarray(global_mag, dtype=float)
        else:
            elapsed_t = history_t
            elapsed_mag = history_mag

    for col, threshold in elapsed_thresholds.items():
        feature_map[col] = seismic_features.calculate_elapsed_times(
            elapsed_t,
            t_reference,
            elapsed_mag,
            Mag_elaps=[threshold],
        )[0]

    return feature_map


def _compute_mag_max_obs_targets(array_dict) -> np.ndarray:
    targets = np.full(len(array_dict["future"]["Magnitude"]), np.nan, dtype=float)
    for i, future_mag in enumerate(array_dict["future"]["Magnitude"]):
        future_mag = np.asarray(future_mag, dtype=float)
        if future_mag.size > 0:
            targets[i] = np.max(future_mag)
    return targets


def build_full_window_feature_frame(
    array_dict,
    samples_list,
    feature_cols,
    *,
    Mc: float,
    dMag: float,
    t_elaps_mode: str = T_ELAPS_MODE_WINDOW,
    global_t: np.ndarray | None = None,
    global_mag: np.ndarray | None = None,
):
    num_samples = len(samples_list)
    feature_cols = list(feature_cols)
    elapsed_thresholds = _extract_elapsed_thresholds(feature_cols)
    t_elaps_mode_resolved = _resolve_t_elaps_mode(t_elaps_mode)

    features = {"t": np.array([sample["t"] for sample in samples_list], dtype=float)}
    for col in feature_cols:
        if col not in features:
            features[col] = np.full(num_samples, np.nan)

    mag_max_obs = _compute_mag_max_obs_targets(array_dict)
    if "Mag_max_obs" in features:
        features["Mag_max_obs"] = mag_max_obs.copy()

    for i in range(num_samples):
        history_t = np.asarray(array_dict["history"]["t"][i], dtype=float)
        history_mag = np.asarray(array_dict["history"]["Magnitude"][i], dtype=float)
        t_now = float(features["t"][i])
        feature_map = _compute_window_feature_map(
            history_t,
            history_mag,
            feature_cols=feature_cols,
            Mc=Mc,
            dMag=dMag,
            t_reference=t_now,
            elapsed_thresholds=elapsed_thresholds,
            t_elaps_mode=t_elaps_mode_resolved,
            global_t=global_t,
            global_mag=global_mag,
        )
        for col in feature_cols:
            features[col][i] = feature_map[col]

    return pd.DataFrame(features)


def build_rf_feature_frame(
    array_dict,
    samples_list,
    feature_cols,
    *,
    Mc: float,
    dMag: float,
    t_elaps_mode: str = T_ELAPS_MODE_WINDOW,
    global_t: np.ndarray | None = None,
    global_mag: np.ndarray | None = None,
):
    return build_full_window_feature_frame(
        array_dict,
        samples_list,
        feature_cols,
        Mc=Mc,
        dMag=dMag,
        t_elaps_mode=t_elaps_mode,
        global_t=global_t,
        global_mag=global_mag,
    )


def resolve_inner_sliding_params(
    *,
    Twindow: float,
    dt: float | None = None,
    time_step: int | None = None,
    feature_window: float | None = None,
    feature_step: float | None = None,
) -> tuple[float, float, int]:
    twindow = float(Twindow)
    if twindow <= 0:
        raise ValueError(f"Twindow must be > 0, got {Twindow}")

    if feature_window is None:
        if time_step is not None and int(time_step) > 0:
            feature_window = twindow / int(time_step)
        else:
            feature_window = twindow
    feature_window = float(feature_window)
    if feature_window <= 0:
        raise ValueError(f"feature_window must be > 0, got {feature_window}")
    if feature_window > twindow:
        raise ValueError(f"feature_window ({feature_window}) cannot exceed Twindow ({twindow})")

    if feature_step is None:
        feature_step = dt if dt is not None else feature_window
    feature_step = float(feature_step)
    if feature_step <= 0:
        raise ValueError(f"feature_step must be > 0, got {feature_step}")

    seq_len = int(np.floor((twindow - feature_window) / feature_step + 1e-12)) + 1
    if seq_len <= 0:
        raise ValueError(
            f"Invalid sliding params: Twindow={twindow}, feature_window={feature_window}, feature_step={feature_step}"
        )
    return feature_window, feature_step, seq_len


def build_inner_sliding_feature_tensor(
    array_dict,
    samples_list,
    feature_cols,
    *,
    Mc: float,
    dMag: float,
    t_elaps_mode: str = T_ELAPS_MODE_WINDOW,
    global_t: np.ndarray | None = None,
    global_mag: np.ndarray | None = None,
    Twindow: float,
    dt: float | None = None,
    time_step: int | None = None,
    feature_window: float | None = None,
    feature_step: float | None = None,
):
    feature_cols = list(feature_cols)
    elapsed_thresholds = _extract_elapsed_thresholds(feature_cols)
    t_elaps_mode_resolved = _resolve_t_elaps_mode(t_elaps_mode)
    feature_window, feature_step, seq_len = resolve_inner_sliding_params(
        Twindow=Twindow,
        dt=dt,
        time_step=time_step,
        feature_window=feature_window,
        feature_step=feature_step,
    )

    num_samples = len(samples_list)
    num_features = len(feature_cols)
    timestamps = np.array([sample["t"] for sample in samples_list], dtype=float)
    targets = _compute_mag_max_obs_targets(array_dict)
    X = np.full((num_samples, seq_len, num_features), np.nan, dtype=float)

    for i in range(num_samples):
        history_t = np.asarray(array_dict["history"]["t"][i], dtype=float)
        history_mag = np.asarray(array_dict["history"]["Magnitude"][i], dtype=float)
        t_now = timestamps[i]
        history_start = t_now - float(Twindow)

        for step_idx in range(seq_len):
            sub_start = history_start + step_idx * feature_step
            sub_end = sub_start + feature_window
            if history_t.size == 0:
                sub_t = np.array([], dtype=float)
                sub_mag = np.array([], dtype=float)
            else:
                left = np.searchsorted(history_t, sub_start, side="left")
                right = np.searchsorted(history_t, sub_end, side="left")
                sub_t = history_t[left:right]
                sub_mag = history_mag[left:right]

            feature_map = _compute_window_feature_map(
                sub_t,
                sub_mag,
                feature_cols=feature_cols,
                Mc=Mc,
                dMag=dMag,
                t_reference=sub_end,
                elapsed_thresholds=elapsed_thresholds,
                t_elaps_mode=t_elaps_mode_resolved,
                global_t=global_t,
                global_mag=global_mag,
            )
            for col_idx, col in enumerate(feature_cols):
                X[i, step_idx, col_idx] = feature_map[col]

    feature_meta = pd.DataFrame(
        {
            "t": timestamps,
            "Mag_max_obs": targets,
        }
    )
    return X, targets, feature_meta


def resolve_external_sliding_params(
    *,
    Twindow: float,
    dt: float | None = None,
    time_step: int | None = None,
    feature_window: float | None = None,
    feature_step: float | None = None,
) -> tuple[int, float]:
    """Resolve sequence length for external contiguous sliding windows.

    For external sliding, one sequence step corresponds to one outer window.
    Sequence length is explicitly controlled by ``time_step``.
    """
    if feature_window is not None:
        raise ValueError(
            "feature_window is not supported for external_sliding mode. "
            "Use time_step to control sequence length."
        )
    if feature_step is not None:
        raise ValueError(
            "feature_step is not supported for external_sliding mode. "
            "Use time_step to control sequence length."
        )

    if time_step is None:
        raise ValueError(
            "external_sliding requires a positive integer time_step."
        )

    seq_len_raw = float(time_step)
    if not seq_len_raw.is_integer():
        raise ValueError(f"time_step must be an integer, got {time_step}")
    seq_len = int(seq_len_raw)
    if seq_len <= 0:
        raise ValueError(f"time_step must be a positive integer, got {seq_len_raw}")

    feature_step_eff = float(dt) if dt is not None else np.nan
    return seq_len, feature_step_eff


def build_external_sliding_feature_tensor(
    array_dict,
    samples_list,
    feature_cols,
    *,
    Mc: float,
    dMag: float,
    t_elaps_mode: str = T_ELAPS_MODE_WINDOW,
    global_t: np.ndarray | None = None,
    global_mag: np.ndarray | None = None,
    Twindow: float,
    dt: float | None = None,
    time_step: int | None = None,
    feature_window: float | None = None,
    feature_step: float | None = None,
):
    """Build LSTM inputs from contiguous *outer* windows instead of inner windows.

    Each step in the sequence corresponds to one already-constructed outer sample
    (i.e., one full history window ending at that sample's ``t``).
    """
    feature_cols = list(feature_cols)
    seq_len, feature_step_eff = resolve_external_sliding_params(
        Twindow=Twindow,
        dt=dt,
        time_step=time_step,
        feature_window=feature_window,
        feature_step=feature_step,
    )

    feature_frame = build_full_window_feature_frame(
        array_dict,
        samples_list,
        feature_cols,
        Mc=Mc,
        dMag=dMag,
        t_elaps_mode=t_elaps_mode,
        global_t=global_t,
        global_mag=global_mag,
    )
    X_full = feature_frame[feature_cols].to_numpy(dtype=float)
    timestamps = feature_frame["t"].to_numpy(dtype=float)
    targets_full = _compute_mag_max_obs_targets(array_dict)

    n_samples = X_full.shape[0]
    if n_samples < seq_len:
        raise ValueError(
            f"Not enough outer windows to build external sliding features: "
            f"n_samples={n_samples}, required_seq_len={seq_len}."
        )

    out_count = n_samples - seq_len + 1
    X = np.full((out_count, seq_len, len(feature_cols)), np.nan, dtype=float)
    y = np.full(out_count, np.nan, dtype=float)
    out_t = np.full(out_count, np.nan, dtype=float)

    for out_idx in range(out_count):
        end_idx = out_idx + seq_len - 1
        start_idx = end_idx - seq_len + 1
        X[out_idx] = X_full[start_idx : end_idx + 1]
        y[out_idx] = targets_full[end_idx]
        out_t[out_idx] = timestamps[end_idx]

    feature_meta = pd.DataFrame(
        {
            "t": out_t,
            "Mag_max_obs": y,
            "feature_step": np.full(out_count, feature_step_eff, dtype=float),
            "sequence_mode": np.full(out_count, FEATURE_CONSTRUCTION_EXTERNAL_SLIDING),
        }
    )
    return X, y, feature_meta
