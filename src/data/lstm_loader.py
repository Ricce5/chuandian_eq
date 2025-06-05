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
from torch.utils.data import WeightedRandomSampler,Subset, Dataset
from src.data.data_utils import get_split_indices

class LSTMDataset(Dataset):
    def __init__(self, X, y, scalars=None):
        """
        初始化自定义数据集
        :param X: 输入特征数据
        :param y: 标签数据
        :param scalars: 用于归一化的缩放器字典（例如 MinMaxScaler）
        """
        self.X = X
        self.y = y
        self.scalars = scalars  # 存储用于逆归一化的缩放器
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def inverse_normalize_label(self, norm_value):
        """
        使用提供的缩放器进行标签的逆归一化。
        :param norm_value: 归一化后的值
        :return: 逆归一化后的值
        """
        # 检查是否包含标签的归一化器（scalars 字典中是否有标签列）
        if 'Mag_max_obs' in self.scalars:
            scalar = self.scalars['Mag_max_obs']
            return scalar.inverse_transform(norm_value.reshape(-1, 1)).squeeze(1)
        else:
            raise ValueError("未找到标签列的归一化缩放器。")


def normalize_df(df):
    scalars = {}
    df_nl = df.copy()
    for col in df_nl.columns:
        if col not in ['time']:
            scalar = MinMaxScaler()
            df_nl[col] = scalar.fit_transform(df_nl[col].values.reshape(-1, 1))
            scalars[col] = scalar
    return df_nl, scalars


def split_dataset(X, y, by_time=False, batch_size=64, seed=0, train_ratio=0.7, val_ratio=0.15, time_order=('train','val', 'test'),scalars=None):
    dataset = LSTMDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        scalars=scalars
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
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
