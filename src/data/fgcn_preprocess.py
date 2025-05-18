import numpy as np


def normalize_f(f):
    """
    对 (N, T, D) 的特征 f 沿 (N, T) 做 min-max 归一化
    """
    f_min = f.min(axis=(0, 1), keepdims=True)
    f_max = f.max(axis=(0, 1), keepdims=True)
    f_range = f_max - f_min
    f_normalized = np.divide(f - f_min, f_range, where=f_range != 0)
    return np.nan_to_num(f_normalized, nan=0)

def normalize_mag(mag):
    """
    对 mag 做全局 min-max 归一化
    """
    min_val = mag.min()
    max_val = mag.max()
    range_val = max_val - min_val
    normed = (mag - min_val) / range_val if range_val != 0 else mag * 0
    return np.nan_to_num(normed, nan=0)

def compute_mag_diff(mag):
    """
    对累积震级-频次分布转化为震级-频次分布
    """
    mag_dif = mag[:, :, 0:-1, :] - mag[:, :, 1:, :]
    return mag_dif