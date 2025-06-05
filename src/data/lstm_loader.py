import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['axes.unicode_minus'] = False
from scipy.spatial.distance import cdist
from torch.utils.data import WeightedRandomSampler,Subset
from src.data.data_utils import get_split_indices


def normalize_df(df):
    scalars = {}
    df_nl = df.copy()
    for col in df_nl.columns:
        if col not in ['time']:
            scalar = MinMaxScaler()
            df_nl[col] = scalar.fit_transform(df_nl[col].values.reshape(-1, 1))
            scalars[col] = scalar
    return df_nl, scalars


def split_dataset(X, y, by_time=False, batch_size=64, seed=0, train_ratio=0.7, val_ratio=0.15, time_order=('train','val', 'test'),):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    total = len(dataset)
    train_idx, val_idx, test_idx = get_split_indices(
        total_length=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        by_time=by_time,
        time_order= time_order
    )


    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)


    print(f"Train set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Test set: {len(test_ds)} samples")

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
