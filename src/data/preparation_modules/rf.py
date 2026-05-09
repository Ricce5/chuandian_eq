import numpy as np

from src.data.event_pipeline import (
    build_classification_targets,
    build_rf_feature_frame,
    load_event_windows_with_cache,
)

from .common import (
    _ensure_args_defaults,
    _maybe_apply_train_subset,
    _resolve_subset_base_indices,
    _resolve_t_elaps_mode,
    _split_with_event_loader,
)


def print_sample_distribution(y, dataset_name):
    positive_samples = int(np.sum(y == 1))
    negative_samples = int(np.sum(y == 0))
    total_samples = len(y)
    positive_ratio = positive_samples / total_samples if total_samples > 0 else 0.0
    negative_ratio = negative_samples / total_samples if total_samples > 0 else 0.0

    print(f'[{dataset_name}] positive: {positive_samples} | negative: {negative_samples} | total: {total_samples}')
    print(f'[{dataset_name}] positive ratio: {positive_ratio:.3f} | negative ratio: {negative_ratio:.3f}')


def prepare_data_rf(args, base_dir):
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
    train_set = _maybe_apply_train_subset(
        train_set,
        args,
        "classification",
        loader_module=loader,
    )
    train_idx = _resolve_subset_base_indices(train_set)
    val_idx = _resolve_subset_base_indices(val_set)
    test_idx = _resolve_subset_base_indices(test_set)

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
