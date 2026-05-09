import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1


def _loader_runtime_kwargs(args):
    num_workers = int(getattr(args, "num_workers", 8))
    pin_memory = bool(getattr(args, "pin_memory", True))
    persistent_workers = bool(getattr(args, "persistent_workers", False))
    seed = int(getattr(args, "seed", 0))
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "seed": seed,
    }


def _resolve_train_subset_ratio(args) -> float:
    ratio_raw = getattr(args, "train_subset_ratio", 1.0)
    try:
        ratio = float(ratio_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"train_subset_ratio must be a float in (0, 1], got {ratio_raw!r}"
        ) from exc
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {ratio}.")
    return ratio


def _build_train_subset_local_indices(*, total: int, ratio: float, by_time: bool, seed: int):
    if total <= 0:
        return np.array([], dtype=int)
    keep = int(math.ceil(total * ratio))
    keep = max(1, min(total, keep))
    if keep == total:
        return np.arange(total, dtype=int)

    if by_time:
        return np.arange(keep, dtype=int)

    rng = np.random.default_rng(int(seed))
    sampled = np.sort(rng.choice(total, size=keep, replace=False))
    return sampled.astype(int, copy=False)


def _subset_dataset_by_local_indices(dataset, local_indices):
    from torch.utils.data import Subset

    local_indices = np.asarray(local_indices, dtype=int)
    if isinstance(dataset, Subset):
        base_indices = np.asarray(dataset.indices, dtype=int)
        mapped_indices = base_indices[local_indices]
        return Subset(dataset.dataset, mapped_indices.tolist())
    return Subset(dataset, local_indices.tolist())


def _resolve_subset_base_indices(dataset):
    from torch.utils.data import Subset

    if not isinstance(dataset, Subset):
        return np.arange(len(dataset), dtype=int)

    indices = np.asarray(dataset.indices, dtype=int)
    base = dataset.dataset
    while isinstance(base, Subset):
        parent_indices = np.asarray(base.indices, dtype=int)
        indices = parent_indices[indices]
        base = base.dataset
    return indices.astype(int, copy=False)


def _maybe_apply_train_subset(train_set, args, task_type: str, loader_module=None):
    ratio = _resolve_train_subset_ratio(args)
    total = len(train_set)
    if total == 0:
        logger.warning("Train set is empty; skip train_subset_ratio.")
        return train_set
    if ratio >= 1.0:
        return train_set

    by_time = bool(getattr(args, "split_by_time", True))
    seed = int(getattr(args, "seed", 0))
    selected = _build_train_subset_local_indices(
        total=total,
        ratio=ratio,
        by_time=by_time,
        seed=seed,
    )
    subset = _subset_dataset_by_local_indices(train_set, selected)
    logger.info(
        "Applied train_subset_ratio=%.4f: kept %s/%s training samples (by_time=%s, seed=%s).",
        ratio,
        len(subset),
        total,
        by_time,
        seed,
    )
    if task_type == "classification" and loader_module is not None:
        pos_count, neg_count = loader_module.count_pos_neg(subset)
        logger.info(
            "Subset training labels: pos=%s neg=%s total=%s",
            pos_count,
            neg_count,
            len(subset),
        )
    return subset


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
