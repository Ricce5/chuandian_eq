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
        支持 (B, F) 或 (B,) 输入，输出与输入形状相同。
        :param norm_value: 归一化后的值 (B, F) 或 (B,)
        :return: 逆归一化后的值，形状与 norm_value 相同
        """
        if 'Mag_max_obs' in self.scalars:
            scalar = self.scalars['Mag_max_obs']
            norm_value_np = np.array(norm_value)
            orig_shape = norm_value_np.shape
            norm_value_2d = norm_value_np.reshape(-1, 1) if norm_value_np.ndim == 1 else norm_value_np
            value = scalar.inverse_transform(norm_value_2d)
            value = value.reshape(orig_shape)
            return torch.tensor(value, dtype=torch.float32)
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



def clean_data(X, y, nan_value_for_x=0.0, verbose=True):
    """
    清洗输入特征 X 和标签 y：
    - 删除 y 为 NaN 的样本（以及对应的 X）
    - 将 X 中的 NaN 替换为指定值（默认是 0.0）
    
    参数：
        X (np.ndarray): 输入特征，二维数组 (n_samples, n_features)
        y (np.ndarray): 目标变量，一维或二维数组
        nan_value_for_x (float): 用于替换 X 中 NaN 的值，默认 0.0
        verbose (bool): 是否打印处理日志

    返回：
        X_clean (np.ndarray): 清洗后的 X
        y_clean (np.ndarray): 清洗后的 y
    """
    # 转为 numpy 数组（如果不是的话）
    X = np.array(X)
    y = np.array(y)

    # 1. 去除 y 为 NaN 的样本
    valid_mask = ~np.isnan(y).flatten()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    # 2. 替换 X 中的 NaN 为指定值
    X_clean = np.nan_to_num(X_clean, nan=nan_value_for_x)

    if verbose:
        print(f"   原始样本数: {len(y)}, 清洗后样本数: {len(y_clean)}")
        print(f"   替换了 X 中的 NaN 为 {nan_value_for_x}")

    return X_clean, y_clean


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


    data_loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }

    return dataset, data_loaders
