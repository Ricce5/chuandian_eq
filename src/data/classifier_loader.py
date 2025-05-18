import pandas as pd
import json
import glob
import os
import torch
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
    scaler = MinMaxScaler()

    magnitude_cols = [col for col in df.columns if 'Mag' in col]
    lat_cols = [col for col in df.columns if 'Lat' in col]
    lon_cols = [col for col in df.columns if 'Lon' in col]
    dep_cols = [col for col in df.columns if 'Dep' in col]

    df_nl = df.copy()

    for col_list in [lat_cols, lon_cols, dep_cols, magnitude_cols]:
        for col in col_list:
            df_nl[col] = scaler.fit_transform(df[[col]])

    return df_nl


def dict_to_array(dict_list, field):
    values = [sample[field] for sample in dict_list]
    return np.array(values) if values else np.array([])


def get_list(df, df_nl, Mc, Mf, Twindow=20, Tfore=2, dt=10, t_array=None, context_len=2):
    df = df[df['Magnitude'] >= Mc].copy()
    samples_list = []
    t = df["t"].values
    t.sort()
    print(f"地震事件数量（大于Mc）：{len(t)}")

    if t_array is not None:
        t_array = np.array(t_array)
        if len(t_array) == 0:
            raise ValueError("t_array 不能为空")
        if np.any(t_array < t[0] + Twindow) or np.any(t_array > t[-1]):
            print("Warning: t_array 有值超出数据时间范围")
        Nloop = len(t_array)
        t_array_final = t_array
    else:
        Nloop = int(np.ceil((t[-1] - t[0] - Twindow - Tfore) / dt))
        t_array_final = Twindow + t[0] + np.arange(Nloop) * dt

    for t_now in t_array_final:
        history = df[(df["t"] < t_now) & (df["t"] >= t_now - Twindow)]
        future_shocks = df[(df["t"] >= t_now) & (df["t"] < t_now + Tfore) & (df["Magnitude"] >= Mf)]
        context = df[(df["t"] >= t_now - context_len * Tfore) & (df["t"] < t_now + context_len * Tfore)]

        sample_structure = {
            "history_dict": [],
            "future_dict": [],
            "context_dict": [],
            "t": t_now,
        }

        for _, quake in future_shocks.iterrows():
            future_sample = {col: df_nl.loc[quake.name, col] for col in ["t", "Magnitude", "Latitude", "Longitude", "Depth"]}
            sample_structure["future_dict"].append(future_sample)

        for _, quake in history.iterrows():
            history_sample = {col: df_nl.loc[quake.name, col] for col in ["t", "Magnitude", "Latitude", "Longitude", "Depth"]}
            history_sample["t_nl"] = (quake["t"] - t_now) / Twindow + 1
            sample_structure["history_dict"].append(history_sample)

        for _, quake in context.iterrows():
            context_sample = {col: df.loc[quake.name, col] for col in ["t", "Magnitude", "Latitude", "Longitude", "Depth"]}
            sample_structure["context_dict"].append(context_sample)

        samples_list.append(sample_structure)

    array_dict = {
        "history": {
            field: [dict_to_array(samples_list[i]["history_dict"], field) for i in range(len(samples_list))]
            for field in ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]
        },
        "future": {
            field: [dict_to_array(samples_list[i]["future_dict"], field) for i in range(len(samples_list))]
            for field in ["t", "Magnitude", "Latitude", "Longitude", "Depth"]
        },
        "context": {
            field: [dict_to_array(samples_list[i]["context_dict"], field) for i in range(len(samples_list))]
            for field in ["t", "Magnitude", "Latitude", "Longitude", "Depth"]
        }
    }

    return samples_list, array_dict


def to_flag(arr_f, arr_c_mag, Mf):
    flag_f = np.count_nonzero(arr_f)
    flag = np.nan
    if flag_f > 0:
        flag = 1
    elif len(arr_c_mag) == 0 or np.max(arr_c_mag) < Mf:
        flag = 0
    return flag


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, array_dict, Mf):
        self.array_dict = array_dict
        self.data_fields = ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]

        self.samples = []
        self.targets = []
   

        future = array_dict["future"]
        context = array_dict["context"]
        num_samples = len(future["t"])

        for idx in range(num_samples):
            target_flag = to_flag(future["t"][idx], context["Magnitude"][idx], Mf)
            if not np.isnan(target_flag):
                history_data = [np.array(array_dict["history"][field][idx]) for field in self.data_fields]
                target_val = target_flag

                self.samples.append(history_data)
                self.targets.append(target_val)

               

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
    @property
    def pos_count(self):
        return sum(1 for t in self.targets if t == 1)
    @property
    def neg_count(self):
        return sum(1 for t in self.targets if t == 0)



def array_pad_t(insts, PAD=0):
    max_len = max(len(inst) for inst in insts)
    padded_batch = [np.pad(inst, (0, max_len - len(inst)), constant_values=PAD) for inst in insts]
    return np.stack(padded_batch)


def collate_fn(instances):
    sample_tuples, target_values = zip(*instances)
    padded_samples = [array_pad_t(sample, PAD=0) for sample in zip(*sample_tuples)]
    padded_samples = np.array(padded_samples)
    padded_samples = torch.tensor(padded_samples, dtype=torch.float32).permute(1, 2, 0)
    padded_targets = torch.tensor(target_values, dtype=torch.float32)
    return padded_samples, padded_targets



def count_pos_neg(subset):
    if isinstance(subset, torch.utils.data.Subset):
        base_dataset = subset.dataset
        indices = subset.indices
        pos = sum(1 for i in indices if base_dataset[i][1] == 1)
        neg = sum(1 for i in indices if base_dataset[i][1] == 0)
    else:
        # 默认是完整的数据集
        pos = sum(1 for _, target in subset if target == 1)
        neg = sum(1 for _, target in subset if target == 0)
    return pos, neg


def get_dataloader(dataset, batch_size,shuffle=True):
    ds = dataset
    pos_count, neg_count = count_pos_neg(ds)
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    print(f"Total samples: {len(ds)}, Positive ratio: {pos_count / len(ds):.2f}, Negative ratio: {neg_count / len(ds):.2f}")
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=8,
         pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl



def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=0, by_time=True):
    total = len(dataset)
    print(f"Number of positive samples: {dataset.pos_count}")
    print(f"Number of negative samples: {dataset.neg_count}")
    train_idx, val_idx, test_idx = get_split_indices(
        total_length=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        by_time=by_time
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set


def split_data(samples_list, array_dict, test_size=0.1, val_size=0.1, random_state=40, by_time=False):
    n = len(samples_list)
    indices = list(range(n))

    if by_time:
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    else:
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=random_state)

    def extract_subset(indices):
        subset_samples = [samples_list[i] for i in indices]
        subset_array_dict = {
            "history": {key: [array_dict["history"][key][i] for i in indices] for key in array_dict["history"]},
            "future": {key: [array_dict["future"][key][i] for i in indices] for key in array_dict["future"]},
            "context": {key: [array_dict["context"][key][i] for i in indices] for key in array_dict["context"]}
        }
        return subset_samples, subset_array_dict

    train_samples, train_dict = extract_subset(train_idx)
    val_samples, val_dict = extract_subset(val_idx)
    test_samples, test_dict = extract_subset(test_idx)

    return (train_samples, train_dict), (val_samples, val_dict), (test_samples, test_dict)
