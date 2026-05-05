import math
import logging
from functools import partial
import pandas as pd
import numpy as np
from src.utils.catalog_utils import split_minibatches, split_sequence
from src.data.preprocessing import load_and_filter_catalog, calculate_catalog_statistics 
from src.data.normalization import normalize_magnitude_range, resolve_magnitude_bounds
from src.data.event_pipeline import (
    FEATURE_CONSTRUCTION_EXTERNAL_SLIDING,
    FEATURE_CONSTRUCTION_INNER_SLIDING,
    build_classification_targets,
    build_external_sliding_feature_tensor,
    build_inner_sliding_feature_tensor,
    build_rf_feature_frame,
    load_event_windows_for_task,
    load_event_windows_with_cache,
)
from  omegaconf import OmegaConf

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1


def _split_config(args):
    return {
        "by_time": getattr(args, "split_by_time", True),
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "val_ratio": DEFAULT_VAL_RATIO,
        "time_order": getattr(args, "time_order", ("train", "val", "test")),
    }


def _split_with_event_loader(dataset, *, task_type: str, loader_module, args):
    split_cfg = _split_config(args)
    return loader_module.split_dataset(
        dataset,
        by_time=split_cfg["by_time"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        time_order=split_cfg["time_order"],
        task_type=task_type,
    )


def _build_outer_split_indices_for_lstm(*, total_outer_windows: int, loader_module, args):
    class _IndexOnlyDataset:
        def __init__(self, n: int):
            self._n = int(n)

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            return idx

    if total_outer_windows <= 0:
        raise ValueError(f"total_outer_windows must be positive, got {total_outer_windows}")

    proxy_dataset = _IndexOnlyDataset(total_outer_windows)
    train_set, val_set, test_set = _split_with_event_loader(
        proxy_dataset,
        task_type="regression",
        loader_module=loader_module,
        args=args,
    )
    return (
        np.asarray(train_set.indices, dtype=int),
        np.asarray(val_set.indices, dtype=int),
        np.asarray(test_set.indices, dtype=int),
    )


def _split_lstm_indices_aligned_to_outer_windows(
    *,
    endpoint_outer_indices: np.ndarray,
    total_outer_windows: int,
    loader_module,
    args,
):
    train_outer_idx, val_outer_idx, test_outer_idx = _build_outer_split_indices_for_lstm(
        total_outer_windows=total_outer_windows,
        loader_module=loader_module,
        args=args,
    )

    split_bucket = np.full(total_outer_windows, -1, dtype=np.int8)
    split_bucket[train_outer_idx] = 0
    split_bucket[val_outer_idx] = 1
    split_bucket[test_outer_idx] = 2

    endpoint_outer_indices = np.asarray(endpoint_outer_indices, dtype=int).reshape(-1)
    if endpoint_outer_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    if np.any(endpoint_outer_indices < 0) or np.any(endpoint_outer_indices >= total_outer_windows):
        raise ValueError(
            "endpoint_outer_indices contains out-of-range values. "
            f"Expected [0, {total_outer_windows - 1}], got "
            f"min={int(endpoint_outer_indices.min())}, max={int(endpoint_outer_indices.max())}."
        )

    sample_bucket = split_bucket[endpoint_outer_indices]
    train_idx = np.flatnonzero(sample_bucket == 0).astype(int)
    val_idx = np.flatnonzero(sample_bucket == 1).astype(int)
    test_idx = np.flatnonzero(sample_bucket == 2).astype(int)
    return train_idx, val_idx, test_idx


def _ensure_args_defaults(args, defaults: dict):
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _resolve_t_elaps_mode(args) -> str:
    mode = str(getattr(args, "t_elaps_mode", "window")).strip().lower()
    if mode not in {"window", "global"}:
        raise ValueError(
            f"Unsupported t_elaps_mode={mode!r}. Supported: 'window', 'global'."
        )
    return mode


def _normalize_lstm_inputs(features_3d, feature_cols, lstm_loader_module):
    X_flat = features_3d.reshape(-1, features_3d.shape[-1])
    X_flat_df = pd.DataFrame(X_flat, columns=list(feature_cols))
    X_flat_nl, feature_scalars = lstm_loader_module.normalize_df(X_flat_df)
    X_nl = X_flat_nl.to_numpy().reshape(features_3d.shape)

    return X_nl, feature_scalars


def _normalize_lstm_labels_to_fixed_range(targets_1d, *, mag_min: float, mag_max: float):
    return normalize_magnitude_range(targets_1d, mag_min=mag_min, mag_max=mag_max)


def _drop_tpp_sequence_events(seq, drop_prob: float, min_nll_events: int):
    if not hasattr(seq, "drop_events") or drop_prob <= 0:
        return seq
    return seq.drop_events(drop_prob=drop_prob, min_nll_events=min_nll_events)


def _maybe_wrap_tpp_train_dataset(dataset, args):
    drop_prob = float(getattr(args, "event_drop_prob", 0.0) or 0.0)
    if drop_prob <= 0.0:
        return dataset

    min_nll_events = int(getattr(args, "event_drop_min_nll_events", 1))
    logger.info(
        "Applying train-time TPP event dropping with prob=%.4f and min_nll_events=%s",
        drop_prob,
        min_nll_events,
    )
    from src.data.tpp_dataset import TppDataset

    return TppDataset(
        dataset.sequences,
        sequence_transform=partial(
            _drop_tpp_sequence_events,
            drop_prob=drop_prob,
            min_nll_events=min_nll_events,
        ),
    )

def prepare_data(args, base_dir):
    import src.data.event_loader as loader
    args.task_type = getattr(args, "task_type", "classification")
    mag_min, mag_max = resolve_magnitude_bounds(args)
    event_bundle = load_event_windows_for_task(
        base_dir=base_dir,
        task_type=args.task_type,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=getattr(args, 'context_len', 0),
    )
    df = event_bundle.df
    statistics = calculate_catalog_statistics(df)
    args.stats = OmegaConf.create(statistics)
    logger.info("stats %s", args.stats)
    array_dict = event_bundle.array_dict

    dataset = loader.EventDataset(
        array_dict,
        args.Mf,
        task_type=args.task_type,
        mag_min=mag_min,
        mag_max=mag_max,
    )
    train_set, val_set, test_set = _split_with_event_loader(
        dataset,
        task_type=args.task_type,
        loader_module=loader,
        args=args,
    )
    if  getattr(args, 'use_sampler', False) and args.task_type == "classification":
        sampler = loader.get_balanced_sampler(train_set)
    else:
        sampler = None
    train_loader = loader.get_dataloader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler,task_type=args.task_type)
    val_loader = loader.get_dataloader(val_set, batch_size=args.batch_size, shuffle=False, task_type=args.task_type)
    test_loader = loader.get_dataloader(test_set, batch_size=args.batch_size, shuffle=False,task_type=args.task_type)
    return df, train_loader, val_loader, test_loader,dataset


def print_sample_distribution(y, dataset_name):
    import numpy as np

    positive_samples = int(np.sum(y == 1))
    negative_samples = int(np.sum(y == 0))
    total_samples = len(y)
    positive_ratio = positive_samples / total_samples if total_samples > 0 else 0.0
    negative_ratio = negative_samples / total_samples if total_samples > 0 else 0.0

    print(f'[{dataset_name}] positive: {positive_samples} | negative: {negative_samples} | total: {total_samples}')
    print(f'[{dataset_name}] positive ratio: {positive_ratio:.3f} | negative ratio: {negative_ratio:.3f}')


def prepare_data_rf(args, base_dir):
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import src.data.event_loader as loader

    context_len = getattr(args, 'context_len', 0)
    _ensure_args_defaults(args, {"dMag": 0.1, "Mag_elaps": []})
    t_elaps_mode = _resolve_t_elaps_mode(args)

    event_bundle = load_event_windows_with_cache(
        base_dir=base_dir,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=context_len,
        task_prefix='classifier',
    )
    df = event_bundle.df
    samples_list = event_bundle.samples_list
    array_dict = event_bundle.array_dict
    global_t = df["t"].to_numpy(dtype=float) if t_elaps_mode == "global" else None
    global_mag = df["Magnitude"].to_numpy(dtype=float) if t_elaps_mode == "global" else None
    if not hasattr(args, "feature_cols"):
        raise AttributeError("args.feature_cols is required for prepare_data_rf")
    features_df = build_rf_feature_frame(
        array_dict,
        samples_list,
        args.feature_cols,
        Mc=float(args.Mc),
        dMag=float(args.dMag),
        t_elaps_mode=t_elaps_mode,
        global_t=global_t,
        global_mag=global_mag,
    )
    np.save("features_df.npy", features_df.to_numpy())
    labels_full, valid_mask = build_classification_targets(array_dict, args.Mf)

    missing_cols = [col for col in args.feature_cols if col not in features_df.columns]
    if missing_cols:
        raise KeyError(f'Missing feature columns: {missing_cols}')

    X = features_df[args.feature_cols].to_numpy()
    X = X[valid_mask]
    y = labels_full[valid_mask].astype(int)
    features_df = features_df.loc[valid_mask].reset_index(drop=True)

    class _RFIndexDataset(torch.utils.data.Dataset):
        def __init__(self, labels):
            self.labels = np.asarray(labels, dtype=int)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return None, int(self.labels[idx])

        @property
        def pos_count(self):
            return int(np.sum(self.labels == 1))

        @property
        def neg_count(self):
            return int(np.sum(self.labels == 0))

    rf_split_dataset = _RFIndexDataset(y)
    train_set, val_set, test_set = _split_with_event_loader(
        rf_split_dataset,
        task_type='classification',
        loader_module=loader,
        args=args,
    )
    train_idx = np.asarray(train_set.indices, dtype=int)
    val_idx = np.asarray(val_set.indices, dtype=int)
    test_idx = np.asarray(test_set.indices, dtype=int)

    x_train_raw = X[train_idx]
    x_val_raw = X[val_idx]
    x_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    print_sample_distribution(y_train, 'Train')
    print_sample_distribution(y_val, 'Validation')
    print_sample_distribution(y_test, 'Test')

    return df, features_df, x_train, x_val, x_test, y_train, y_val, y_test


def prepare_data_lstm(args, base_dir):
    import src.data.lstm_loader as loader
    import torch
    from torch.utils.data import DataLoader, Subset
    import src.data.event_loader as event_loader

    _ensure_args_defaults(args, {"Mag_elaps": []})
    t_elaps_mode = _resolve_t_elaps_mode(args)
    mag_min, mag_max = resolve_magnitude_bounds(args)
    feature_mode = str(getattr(args, "lstm_feature_mode", FEATURE_CONSTRUCTION_INNER_SLIDING)).strip().lower()
    if feature_mode not in {FEATURE_CONSTRUCTION_INNER_SLIDING, FEATURE_CONSTRUCTION_EXTERNAL_SLIDING}:
        raise ValueError(
            f"Unsupported lstm_feature_mode={feature_mode!r}. "
            f"Supported: {FEATURE_CONSTRUCTION_INNER_SLIDING!r}, {FEATURE_CONSTRUCTION_EXTERNAL_SLIDING!r}."
        )
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

    event_bundle = load_event_windows_with_cache(
        base_dir=base_dir,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=getattr(args, "context_len", 0),
        task_prefix='regressor',
    )
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
        global_t=global_t,
        global_mag=global_mag,
        Twindow=args.Twindow,
        dt=getattr(args, "dt", None),
        time_step=time_step,
        feature_window=feature_window,
        feature_step=feature_step,
    )
    if feature_mode == FEATURE_CONSTRUCTION_EXTERNAL_SLIDING:
        X, y, features_meta = build_external_sliding_feature_tensor(
            array_dict,
            samples_list,
            args.feature_cols,
            **feature_builder_kwargs,
        )
        seq_len = int(X.shape[1])
        endpoint_outer_indices = np.arange(seq_len - 1, seq_len - 1 + X.shape[0], dtype=int)
    else:
        X, y, features_meta = build_inner_sliding_feature_tensor(
            array_dict,
            samples_list,
            args.feature_cols,
            **feature_builder_kwargs,
        )
        endpoint_outer_indices = np.arange(X.shape[0], dtype=int)

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

    y_arr = np.asarray(y).reshape(-1)
    valid_mask = ~np.isnan(y_arr)
    endpoint_outer_indices = endpoint_outer_indices[valid_mask]
    if len(features_meta) != valid_mask.shape[0]:
        raise ValueError("Inconsistent number of features and valid samples.")
    features_meta = features_meta.iloc[valid_mask].reset_index(drop=True)


    X, y = loader.clean_data(X, y)
    X_nl, feature_scalars = _normalize_lstm_inputs(X, args.feature_cols, loader)
    y_nl = _normalize_lstm_labels_to_fixed_range(y, mag_min=mag_min, mag_max=mag_max)
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    print(f'X_nl shape: {X_nl.shape}, y_nl shape: {y_nl.shape}')

    dataset = loader.LSTMDataset(
        torch.tensor(X_nl, dtype=torch.float32),
        torch.tensor(y_nl, dtype=torch.float32),
        scalars=feature_scalars,
        label_norm_cfg={
            "type": "fixed_range",
            "mag_min": mag_min,
            "mag_max": mag_max,
        },
    )
    train_idx, val_idx, test_idx = _split_lstm_indices_aligned_to_outer_windows(
        endpoint_outer_indices=endpoint_outer_indices,
        total_outer_windows=len(samples_list),
        loader_module=event_loader,
        args=args,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    data_loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=False),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
    }

    return features_meta, data_loaders['train'], data_loaders['val'], data_loaders['test'], dataset


def _auto_configure_global_bg_time_bounds(args, stats_source):
    """Populate bg_model_cfg global time bounds when global normalization is enabled."""
    bg_cfg = getattr(args, "bg_model_cfg", None)
    if bg_cfg is None:
        return

    time_normalization = str(getattr(bg_cfg, "time_normalization", "per_sequence")).strip().lower()
    if time_normalization != "global":
        return

    has_min = getattr(bg_cfg, "global_time_min", None) is not None
    has_max = getattr(bg_cfg, "global_time_max", None) is not None
    if has_min and has_max:
        return
    if has_min != has_max:
        raise ValueError(
            "bg_model_cfg.global_time_min/global_time_max must be both set or both unset "
            "when time_normalization='global'."
        )

    time_min = min(float(seq.t_start) for seq in stats_source)
    time_max = max(float(seq.t_end) for seq in stats_source)
    bg_cfg.global_time_min = float(time_min)
    bg_cfg.global_time_max = float(time_max)
    logger.info(
        "Auto-configured bg_model_cfg global time bounds: "
        "global_time_min=%.6f, global_time_max=%.6f",
        bg_cfg.global_time_min,
        bg_cfg.global_time_max,
    )


def prepare_data_tpp(args, base_dir):
    import torch
    from src.data.tpp_dataset import TppDataset
    import src.data.catalog as catalog
    import src.catalogs as catalogs # ensure catalogs are registered
    from src.catalogs.pathing import build_tpp_catalog_init_kwargs

    catalog_ds_class = catalog.Catalog.by_name(f"{args.dataset}-Standard")
    catalog_cfg = getattr(args, 'catalog_cfg', {})
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=catalog_ds_class,
        base_dir=base_dir,
        catalog_cfg=catalog_cfg,
    )

    catalog_ds = catalog_ds_class(**init_kwargs)
        
    use_all_for_train =  getattr(args, "use_all_data", False)

    # Use either the original train split or the union of all splits for statistics.
    stats_source = catalog_ds.train
    if use_all_for_train:
        stats_source = TppDataset(
            catalog_ds.train.sequences
            + catalog_ds.val.sequences
            + catalog_ds.test.sequences
        )
    _auto_configure_global_bg_time_bounds(args, stats_source)

    args.tau_mean = torch.cat([seq.inter_times[:-1] for seq in stats_source]).mean().item()
    args.time_min = torch.min(torch.tensor([seq.t_start for seq in stats_source])).item()
    args.tau_min = torch.cat([seq.inter_times[:-1] for seq in stats_source]).min().item()
    args.tau_max = torch.cat([seq.inter_times[:-1] for seq in stats_source]).max().item()
    args.tau_q05 = torch.cat([seq.inter_times[:-1] for seq in stats_source]).quantile(0.5).item()
    args.tau_q025 = torch.cat([seq.inter_times[:-1] for seq in stats_source]).quantile(0.025).item()
    args.mag_mean = torch.cat([seq.mag for seq in stats_source]).mean().item()
    args.time_max = torch.max(torch.tensor([seq.t_end for seq in stats_source])).item()
    args.time_mean = torch.cat([seq.arrival_times[:-1] for seq in stats_source]).mean().item()
    args.mag_completeness = catalog_ds.metadata["mag_completeness"]

    # Allow overriding b-value from config (richter_b or richter_b_mle)
    config_b = getattr(args, "richter_b", None)

    if config_b is not None:
        args.richter_b_mle = float(config_b)
    elif "richter_b" in catalog_ds.metadata:
        # Use ground truth value from catalog metadata when provided
        args.richter_b_mle = catalog_ds.metadata["richter_b"]
    else:
        mag_roundoff_error = catalog_ds.metadata.get("mag_roundoff_error", 0.0)
        args.richter_b_mle = math.log10(math.exp(1)) / (
            args.mag_mean - args.mag_completeness + 0.5 * mag_roundoff_error
        )
        
    if  getattr(args, 'use_b_updater', False):
        from src.data.bayesian_b_updater import BayesianGRBUpdater
        b_updater = BayesianGRBUpdater(
            Mc=args.mag_completeness,
            **args.b_updater_cfg,
            mag_key="mag",
            write_back=True,
        )
        catalog_ds.set_b_updater(b_updater)
        catalog_ds.estimate_gr_b()
        logger.info(
            "Using Bayesian GR b-value updater with delta=%s, a0=%s, init b=%.4f, mag_completeness=%s",
            b_updater.delta,
            b_updater.a0,
            b_updater.init_b_target,
            b_updater.Mc,
        )

    if getattr(args, 'use_double_precision:', False):
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.double()
        catalog_ds.full_sequence = catalog_ds.full_sequence.double()
        args.precision = 64
    else:
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.float()
        catalog_ds.full_sequence = catalog_ds.full_sequence.float()
        args.precision = 32

    if use_all_for_train:
        logger.info("TPP pretrain mode: using all splits as training, disabling val/test loaders.")
        if getattr(args, 'minibatch_training', True):
            max_events = getattr(args, 'max_seq_len', 2000)
            mean_nll_events = getattr(args, 'mean_nll_events', 300)
            train_dataset = split_sequence(catalog_ds.full_sequence, mean_nll_events, max_events)
        else:
            train_dataset = TppDataset([catalog_ds.full_sequence])
        args.num_events_train = sum(seq.num_nll_events for seq in train_dataset)
        train_dataset = _maybe_wrap_tpp_train_dataset(train_dataset, args)
        train_loader = train_dataset.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            pad_token_id=getattr(args, 'pad_token_id', None),
        )
        val_loader = None
        test_loader = None
        args.num_events_val = 0
    else:
        args.num_events_train = sum(seq.num_nll_events for seq in catalog_ds.train)
        args.num_events_val = sum(seq.num_nll_events for seq in catalog_ds.val)
        args.num_events_test = sum(seq.num_nll_events for seq in catalog_ds.test)
        logger.info("Number of training events: %s", args.num_events_train)
        logger.info("Number of validation events: %s", args.num_events_val)
        logger.info("Number of test events: %s", args.num_events_test)
        if getattr(args, 'minibatch_training', True):
            logger.info("Splitting into minibatches")
            max_events = getattr(args, 'max_seq_len', 2000)
            mean_nll_events = getattr(args, 'mean_nll_events', 300)
            catalog_ds = split_minibatches(catalog_ds, mean_nll_events, max_events)

        train_dataset = _maybe_wrap_tpp_train_dataset(catalog_ds.train, args)
        train_loader = train_dataset.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            pad_token_id=getattr(args, 'pad_token_id', None),
        )
        val_loader = catalog_ds.val.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            pad_token_id=getattr(args, 'pad_token_id', None),
        )
        test_loader = catalog_ds.test.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            pad_token_id=getattr(args, 'pad_token_id', None),
        )

    return catalog_ds.full_sequence, train_loader, val_loader, test_loader, catalog_ds
