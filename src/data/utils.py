import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


def get_split_indices(
    total_length,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=0,
    by_time=True,
    time_order=('train', 'val', 'test')
):
    assert set(time_order) == {'train', 'val', 'test'}, "time_order must be a permutation of ('train', 'val', 'test')"

    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio >= 0, "Sum of train_ratio and val_ratio must be <= 1"

    ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    lengths = {k: int(ratios[k] * total_length) for k in ratios}

    # Adjust to ensure the sum of lengths == total_length
    total_assigned = sum(lengths.values())
    if total_assigned < total_length:
        # Assign remaining to the last group in the time order
        lengths[time_order[-1]] += total_length - total_assigned

    indices = np.arange(total_length)

    if not by_time:
        np.random.seed(seed)
        indices = np.random.permutation(indices)
        train_end = lengths['train']
        val_end = train_end + lengths['val']

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

    else:
        np.random.seed(seed)
        idx_map = {}
        start = 0
        for group in time_order:
            end = start + lengths[group]
            idx_block = np.arange(start, end)
            idx_map[group] = idx_block
            start = end
        train_idx = idx_map['train']
        val_idx = idx_map['val']
        test_idx = idx_map['test']

    return train_idx, val_idx, test_idx






