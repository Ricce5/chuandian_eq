import logging

import numpy as np
import pandas as pd

import src.data.event_pipeline as event_pipeline
from src.data.event_pipeline import (
    EventWindowBundle,
    FEATURE_CONSTRUCTION_EXTERNAL_SLIDING,
    FEATURE_CONSTRUCTION_INNER_SLIDING,
    build_external_sliding_feature_tensor,
    build_inner_sliding_feature_tensor,
    build_rf_feature_frame,
)
from src.data.normalization import normalize_magnitude_range, resolve_magnitude_bounds

from .common import (
    _build_outer_split_indices_for_lstm,
    _ensure_args_defaults,
    _maybe_apply_train_subset,
    _resolve_t_elaps_mode,
    _split_lstm_indices_aligned_to_outer_windows,
)

logger = logging.getLogger(__name__)


def _normalize_lstm_inputs(features_3d, feature_cols, lstm_loader_module):
    X_flat = features_3d.reshape(-1, features_3d.shape[-1])
    X_flat_df = pd.DataFrame(X_flat, columns=list(feature_cols))
    X_flat_nl, feature_scalars = lstm_loader_module.normalize_df(X_flat_df)
    X_nl = X_flat_nl.to_numpy().reshape(features_3d.shape)

    return X_nl, feature_scalars


def _normalize_lstm_labels_to_fixed_range(targets_1d, *, mag_min: float, mag_max: float):
    return normalize_magnitude_range(targets_1d, mag_min=mag_min, mag_max=mag_max)


def _load_lstm_event_windows_with_cache(args, base_dir) -> EventWindowBundle:
    import src.data.event_loader as event_loader

    _ensure_args_defaults(args, {"Mag_elaps": []})
    base_dir = str(base_dir)
    df = event_pipeline.load_and_filter_catalog(base_dir, Mc=args.Mc)
    df_nl, _ = event_loader.normalize_df(df)

    context_len = getattr(args, "context_len", 0)
    feature_mode = str(getattr(args, "lstm_feature_mode", FEATURE_CONSTRUCTION_INNER_SLIDING)).strip().lower()
    time_step = getattr(args, "time_step", None)
    feature_window = getattr(args, "feature_window", None)
    feature_step = getattr(args, "feature_step", None)
    mag_elaps = list(getattr(args, "Mag_elaps", []))

    def generate_data():
        samples_list, array_dict = event_loader.construct_samples_list(
            df,
            df_nl,
            Mc=args.Mc,
            Mf=args.Mf,
            Twindow=args.Twindow,
            Tfore=args.Tfore,
            dt=args.dt,
            context_len=context_len,
        )
        return {
            "samples_list": samples_list,
            "array_dict": array_dict,
        }

    cached = event_pipeline.save_or_load_data(
        base_path=base_dir,
        generate_fn=generate_data,
        sub_dir="lstm_regressor",
        prefix="lstm_regressor",
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=context_len,
        time_step=time_step,
        feature_window=feature_window,
        feature_step=feature_step,
        lstm_feature_mode=feature_mode,
        Mag_elaps=mag_elaps,
    )
    return EventWindowBundle(df=df, samples_list=cached["samples_list"], array_dict=cached["array_dict"])


def _impute_telaps_nan_in_tensor(features_3d, feature_cols, *, fallback_value: float, epsilon: float = 1e-6):
    X = np.asarray(features_3d, dtype=float).copy()
    stats = []
    for feature_idx, col in enumerate(feature_cols):
        if not str(col).startswith("T_elaps"):
            continue

        channel = X[:, :, feature_idx]
        nan_mask = np.isnan(channel)
        nan_count = int(nan_mask.sum())
        if nan_count == 0:
            continue

        finite = channel[~nan_mask]
        if finite.size > 0:
            fill_value = float(np.nanmax(finite) + float(epsilon))
        else:
            fill_value = float(fallback_value)
        channel[nan_mask] = fill_value
        X[:, :, feature_idx] = channel
        stats.append({"feature": str(col), "nan_count": nan_count, "fill_value": fill_value})
    return X, stats


def _impute_telaps_nan_in_feature_frame(feature_frame: pd.DataFrame, *, fallback_value: float, epsilon: float = 1e-6):
    if feature_frame is None:
        return None, []

    out = feature_frame.copy()
    stats = []
    for col in out.columns:
        if not str(col).startswith("T_elaps"):
            continue

        col_values = pd.to_numeric(out[col], errors="coerce")
        nan_mask = col_values.isna()
        nan_count = int(nan_mask.sum())
        if nan_count == 0:
            continue

        finite = col_values[~nan_mask].to_numpy(dtype=float)
        if finite.size > 0:
            fill_value = float(np.nanmax(finite) + float(epsilon))
        else:
            fill_value = float(fallback_value)
        out[col] = col_values.fillna(fill_value)
        stats.append({"feature": str(col), "nan_count": nan_count, "fill_value": fill_value})
    return out, stats


def _resolve_lstm_feature_mode(args) -> str:
    feature_mode = str(getattr(args, "lstm_feature_mode", FEATURE_CONSTRUCTION_INNER_SLIDING)).strip().lower()
    if feature_mode not in {FEATURE_CONSTRUCTION_INNER_SLIDING, FEATURE_CONSTRUCTION_EXTERNAL_SLIDING}:
        raise ValueError(
            f"Unsupported lstm_feature_mode={feature_mode!r}. "
            f"Supported: {FEATURE_CONSTRUCTION_INNER_SLIDING!r}, {FEATURE_CONSTRUCTION_EXTERNAL_SLIDING!r}."
        )
    return feature_mode


def _resolve_lstm_windowing_args(args, feature_mode: str):
    time_step = getattr(args, "time_step", None)
    feature_window = getattr(args, "feature_window", None)
    feature_step = getattr(args, "feature_step", None)
    if feature_mode == FEATURE_CONSTRUCTION_EXTERNAL_SLIDING:
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
    return time_step, feature_window, feature_step


def _resolve_change_rate_twindow(args):
    mode = str(getattr(args, "change_rate_twindow_mode", "effective")).strip().lower()
    if mode not in {"effective", "fixed"}:
        raise ValueError(
            f"Unsupported change_rate_twindow_mode={mode!r}. "
            "Supported: 'effective', 'fixed'."
        )
    if mode == "fixed":
        return float(args.Twindow)
    return None


def _resolve_external_target_mode(args):
    mode = str(getattr(args, "external_target_mode", "sequence_end")).strip().lower()
    if mode not in {"sequence_end", "next_step"}:
        raise ValueError(
            f"Unsupported external_target_mode={mode!r}. "
            "Supported: 'sequence_end', 'next_step'."
        )
    return mode


def _resolve_lstm_split_mode(args):
    mode = str(getattr(args, "lstm_split_mode", "aligned_outer")).strip().lower()
    if mode not in {"aligned_outer", "sequence"}:
        raise ValueError(
            f"Unsupported lstm_split_mode={mode!r}. "
            "Supported: 'aligned_outer', 'sequence'."
        )
    return mode


def _build_lstm_raw_tensor(
    *,
    args,
    feature_mode: str,
    array_dict,
    samples_list,
    feature_builder_kwargs,
):
    builder_kwargs = dict(feature_builder_kwargs)
    if feature_mode == FEATURE_CONSTRUCTION_EXTERNAL_SLIDING:
        X, y, features_meta = build_external_sliding_feature_tensor(
            array_dict,
            samples_list,
            args.feature_cols,
            **builder_kwargs,
        )
        seq_len = int(X.shape[1])
        if "target_outer_idx" in features_meta.columns:
            endpoint_outer_indices = features_meta["target_outer_idx"].to_numpy(dtype=int)
        else:
            endpoint_outer_indices = np.arange(seq_len - 1, seq_len - 1 + X.shape[0], dtype=int)
    else:
        builder_kwargs.pop("external_target_mode", None)
        X, y, features_meta = build_inner_sliding_feature_tensor(
            array_dict,
            samples_list,
            args.feature_cols,
            **builder_kwargs,
        )
        endpoint_outer_indices = np.arange(X.shape[0], dtype=int)
    return X, y, features_meta, endpoint_outer_indices


def _validate_lstm_seq_len(X, feature_mode: str):
    if X.shape[1] < 2:
        if feature_mode == FEATURE_CONSTRUCTION_EXTERNAL_SLIDING:
            raise ValueError(
                "LSTM needs at least 2 sequence steps for external_sliding. "
                f"Got seq_len={X.shape[1]}. Increase time_step to >= 2."
            )
        raise ValueError(
            "LSTM needs at least 2 sequence steps. "
            f"Got seq_len={X.shape[1]}. Increase Twindow, reduce feature_window, or reduce feature_step."
        )


def _build_optional_feature_frame(
    *,
    return_feature_frame: bool,
    array_dict,
    samples_list,
    feature_cols,
    Mc,
    dMag,
    t_elaps_mode,
    change_rate_twindow,
    global_t,
    global_mag,
    endpoint_outer_indices,
    y_arr,
    valid_mask,
):
    if not return_feature_frame:
        return None

    feature_frame_full = build_rf_feature_frame(
        array_dict,
        samples_list,
        feature_cols,
        Mc=float(Mc),
        dMag=float(dMag),
        t_elaps_mode=t_elaps_mode,
        change_rate_twindow=change_rate_twindow,
        global_t=global_t,
        global_mag=global_mag,
    )
    feature_frame = feature_frame_full.iloc[endpoint_outer_indices].reset_index(drop=True)
    if "Mag_max_obs" not in feature_frame.columns:
        feature_frame = feature_frame.copy()
        feature_frame["Mag_max_obs"] = y_arr[valid_mask]
    return feature_frame


def _apply_telaps_nan_policy(*, args, X, feature_frame, return_feature_frame):
    telaps_nan_strategy = str(getattr(args, "telaps_nan_strategy", "high_sentinel")).strip().lower()
    if telaps_nan_strategy not in {"high_sentinel", "zero"}:
        raise ValueError(
            f"Unsupported telaps_nan_strategy={telaps_nan_strategy!r}. "
            "Supported: 'high_sentinel', 'zero'."
        )

    if telaps_nan_strategy == "zero":
        logger.info("Using legacy T_elaps NaN strategy: zero fill via clean_data.")
        return X, feature_frame

    telaps_fallback = float(args.Twindow)
    telaps_eps = 1e-6
    X, telaps_impute_stats = _impute_telaps_nan_in_tensor(
        X,
        args.feature_cols,
        fallback_value=telaps_fallback,
        epsilon=telaps_eps,
    )
    if return_feature_frame and feature_frame is not None:
        feature_frame, _ = _impute_telaps_nan_in_feature_frame(
            feature_frame,
            fallback_value=telaps_fallback,
            epsilon=telaps_eps,
        )
    if telaps_impute_stats:
        logger.info("Applied T_elaps NaN high-sentinel imputation: %s", telaps_impute_stats)
    return X, feature_frame


def _make_lstm_loaders(*, args, dataset, endpoint_outer_indices, total_outer_windows):
    from torch.utils.data import DataLoader, Subset
    import src.data.event_loader as event_loader
    from src.data.utils import get_split_indices

    split_mode = _resolve_lstm_split_mode(args)
    if split_mode == "sequence":
        train_idx, val_idx, test_idx = get_split_indices(
            total_length=len(dataset),
            train_ratio=0.8,
            val_ratio=0.1,
            seed=int(getattr(args, "seed", 0)),
            by_time=bool(getattr(args, "split_by_time", True)),
            time_order=getattr(args, "time_order", ("train", "val", "test")),
        )
    else:
        train_idx, val_idx, test_idx = _split_lstm_indices_aligned_to_outer_windows(
            endpoint_outer_indices=endpoint_outer_indices,
            total_outer_windows=total_outer_windows,
            loader_module=event_loader,
            args=args,
        )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    train_set = _maybe_apply_train_subset(train_set, args, "regression")
    return {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=False),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
    }


def prepare_data_lstm(args, base_dir, return_feature_frame=False):
    import torch
    import src.data.lstm_loader as loader
    from sklearn.preprocessing import MinMaxScaler

    _ensure_args_defaults(args, {"Mag_elaps": []})
    t_elaps_mode = _resolve_t_elaps_mode(args)
    mag_min, mag_max = resolve_magnitude_bounds(args)

    feature_mode = _resolve_lstm_feature_mode(args)
    time_step, feature_window, feature_step = _resolve_lstm_windowing_args(args, feature_mode)
    change_rate_twindow = _resolve_change_rate_twindow(args)
    external_target_mode = _resolve_external_target_mode(args)

    event_bundle = _load_lstm_event_windows_with_cache(args, base_dir)
    df = event_bundle.df
    samples_list = event_bundle.samples_list
    array_dict = event_bundle.array_dict
    global_t = df["t"].to_numpy(dtype=float) if t_elaps_mode == "global" else None
    global_mag = df["Magnitude"].to_numpy(dtype=float) if t_elaps_mode == "global" else None

    if not hasattr(args, "feature_cols"):
        raise AttributeError("args.feature_cols is required for prepare_data_lstm")

    feature_builder_kwargs = dict(
        Mc=float(args.Mc),
        dMag=float(getattr(args, "dMag", 0.1)),
        t_elaps_mode=t_elaps_mode,
        change_rate_twindow=change_rate_twindow,
        global_t=global_t,
        global_mag=global_mag,
        Twindow=args.Twindow,
        dt=getattr(args, "dt", None),
        time_step=time_step,
        feature_window=feature_window,
        feature_step=feature_step,
        external_target_mode=external_target_mode,
    )
    X, y, features_meta, endpoint_outer_indices = _build_lstm_raw_tensor(
        args=args,
        feature_mode=feature_mode,
        array_dict=array_dict,
        samples_list=samples_list,
        feature_builder_kwargs=feature_builder_kwargs,
    )
    _validate_lstm_seq_len(X, feature_mode)

    y_arr = np.asarray(y).reshape(-1)
    valid_mask = ~np.isnan(y_arr)
    endpoint_outer_indices = endpoint_outer_indices[valid_mask]

    feature_frame = _build_optional_feature_frame(
        return_feature_frame=return_feature_frame,
        array_dict=array_dict,
        samples_list=samples_list,
        feature_cols=args.feature_cols,
        Mc=args.Mc,
        dMag=getattr(args, "dMag", 0.1),
        t_elaps_mode=t_elaps_mode,
        change_rate_twindow=change_rate_twindow,
        global_t=global_t,
        global_mag=global_mag,
        endpoint_outer_indices=endpoint_outer_indices,
        y_arr=y_arr,
        valid_mask=valid_mask,
    )

    if len(features_meta) != valid_mask.shape[0]:
        raise ValueError("Inconsistent number of features and valid samples.")
    features_meta = features_meta.iloc[valid_mask].reset_index(drop=True)

    X, feature_frame = _apply_telaps_nan_policy(
        args=args,
        X=X,
        feature_frame=feature_frame,
        return_feature_frame=return_feature_frame,
    )

    X, y = loader.clean_data(X, y)
    X_nl, feature_scalars = _normalize_lstm_inputs(X, args.feature_cols, loader)
    y_norm_mode = str(getattr(args, "label_norm_mode", "fixed_range")).strip().lower()
    if y_norm_mode == "fixed_range":
        y_nl = _normalize_lstm_labels_to_fixed_range(y, mag_min=mag_min, mag_max=mag_max)
        label_norm_cfg = {
            "type": "fixed_range",
            "mag_min": mag_min,
            "mag_max": mag_max,
        }
    elif y_norm_mode == "legacy_minmax":
        targets_full = event_pipeline._compute_mag_max_obs_targets(array_dict)
        scaler = MinMaxScaler()
        scaler.fit(np.asarray(targets_full, dtype=float).reshape(-1, 1))
        y_nl = scaler.transform(np.asarray(y, dtype=float).reshape(-1, 1)).reshape(-1)
        feature_scalars = dict(feature_scalars)
        feature_scalars["Mag_max_obs"] = scaler
        label_norm_cfg = {"type": "scaler_mag_max_obs"}
    else:
        raise ValueError(
            f"Unsupported label_norm_mode={y_norm_mode!r}. "
            "Supported: 'fixed_range', 'legacy_minmax'."
        )
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    print(f'X_nl shape: {X_nl.shape}, y_nl shape: {y_nl.shape}')

    dataset = loader.LSTMDataset(
        torch.tensor(X_nl, dtype=torch.float32),
        torch.tensor(y_nl, dtype=torch.float32),
        scalars=feature_scalars,
        label_norm_cfg=label_norm_cfg,
    )

    data_loaders = _make_lstm_loaders(
        args=args,
        dataset=dataset,
        endpoint_outer_indices=endpoint_outer_indices,
        total_outer_windows=len(samples_list),
    )

    if return_feature_frame:
        return (
            features_meta,
            data_loaders['train'],
            data_loaders['val'],
            data_loaders['test'],
            dataset,
            feature_frame,
        )

    return features_meta, data_loaders['train'], data_loaders['val'], data_loaders['test'], dataset
