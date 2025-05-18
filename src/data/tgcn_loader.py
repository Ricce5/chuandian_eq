import pandas as pd
import glob
import os
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler,Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['axes.unicode_minus'] = False   
# 设置保存路径
from scipy.spatial.distance import cdist
import numpy as np
from .data_utils import get_split_indices
from .preprocessing import map_region,process_df


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



regions = ['ANH', 'DLSH', 'XJ', 'ZMH']
depth_levels = list(range(7))  




def normalize_df(df):
    """
    对 DataFrame 中的列进行 Min-Max 缩放，并计算时间累积和。
    1. 按类别缩放列：'Lat'，'Lon'，'Dep'，'dt'，'Mag'。
    2. 计算 't' 列，作为 'dt' 列的累积和。
    
    参数：
    df (pd.DataFrame): 包含数据的 DataFrame。

    返回：
    pd.DataFrame: 处理后的 DataFrame，包含缩放后的列和新添加的 't' 列。
    """
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Define column categories
    magnitude_cols = [col for col in df.columns if 'Mag' in col]
    lat_cols = [col for col in df.columns if 'Lat' in col]
    lon_cols = [col for col in df.columns if 'Lon' in col]
    dep_cols = [col for col in df.columns if 'Dep' in col]
    dt_cols = [col for col in df.columns if 'dt' in col]

    # Create a copy of the original DataFrame to avoid modifying it directly
    df_nl = df.copy()

    # Apply Min-Max scaling to the relevant columns
    for col_list in [lat_cols, lon_cols, dep_cols, dt_cols, magnitude_cols]:
        df_nl[col_list] = scaler.fit_transform(df[col_list])

    # Add 't' column as the cumulative sum of 'dt' column
    df_nl['t'] = df_nl['dt'].cumsum()

    return df_nl


def pad_time(insts, PAD=0):
    """ Pad the instance to the max seq length in batch. """
    max_len = max(len(inst) for inst in insts)  # 找出 batch 中最长的序列长度
    batch_seq = np.array([
          inst+[PAD] * (max_len - len(inst))  # 填充零到最大长度
        for inst in insts
    ])
    return batch_seq



def dict_to_array(type_dict, categories, field):
    lists = []

    for i, type in enumerate(categories):
    # 循环结构决定了行与类的对应关系
        values = []
        if type in type_dict:
            values = [sample[field] for sample in type_dict[type]]
        lists.append(values)
        # print(f"第 {i} 类 {type}，提取 {len(values)} 个字段“{field}”的值")

    array = pad_time(lists)
    return array



def get_list(df, df_nl, Mc, Mf, Twindow=20, Tfore=2, dt=10, t_array=None):
    df = df[df['Magnitude'] >= Mc]
    regions = ['ANH', 'DLSH', 'XJ', 'ZMH']
    samples_list = []

    jd = df["Time"].values
    jd.sort()

    if t_array is not None:
        t_array = np.array(t_array)
        if len(t_array) == 0:
            raise ValueError("t_array 不能为空")
        if np.any(t_array < jd[0] + Twindow) or np.any(t_array > jd[-1]):
            print("Warning: t_array 有值超出数据时间范围，会导致空窗口")
        Nloop = len(t_array)
        t_array_final = t_array
    else:
        Nloop = int(np.ceil((jd[-1] - jd[0] - Twindow - Tfore) / dt))
        t_array_final = Twindow + jd[0] + np.arange(Nloop) * dt

    for t_now in t_array_final:
        history = df[(df["Time"] < t_now) & 
                     (df["Time"] >= t_now - Twindow)]

        future_shocks = df[(df["Time"] >= t_now) & 
                           (df["Time"] < t_now + Tfore) & 
                           (df["Magnitude"] >= Mf)]

        context = df[(df["Time"] >= t_now - 3 * Tfore) & 
                     (df["Time"] < t_now + 3 * Tfore)]

        sample_structure = {
            "history_dict": {},
            "future_dict": {},
            "context_dict": {}
        }

        for _, quake in future_shocks.iterrows():
            future_sample = {col: df_nl.loc[quake.name, col] for col in 
                             ["ID2", "Time", "Magnitude", "Latitude", "Longitude", "Depth_m", "region"]}
            quake_region = quake["region"]
            sample_structure["future_dict"].setdefault(quake_region, []).append(future_sample)

        for _, quake in history.iterrows():
            history_sample = {col: df_nl.loc[quake.name, col] for col in 
                              ["ID2", "Time", "Magnitude", "Latitude", "Longitude", "Depth_m", "region"]}
            history_sample["Time_norm"] = (quake["Time"] - t_now) / Twindow + 1
            quake_region = quake["region"]
            sample_structure["history_dict"].setdefault(quake_region, []).append(history_sample)

        for _, quake in context.iterrows():
            context_sample = {col: df.loc[quake.name, col] for col in 
                              ["ID2", "Time", "Magnitude", "Latitude", "Longitude", "Depth_m", "region"]}
            quake_region = quake["region"]
            sample_structure["context_dict"].setdefault(quake_region, []).append(context_sample)

        samples_list.append(sample_structure)

    array_dict = {
        "history": {
            field: [
                dict_to_array(samples_list[i]["history_dict"], regions, field)
                for i in range(len(samples_list))
            ]
            for field in ["ID2", "Time", "Time_norm", "Magnitude", "Latitude", "Longitude", "Depth_m"]
        },
        "future": {
            field: [
                dict_to_array(samples_list[i]["future_dict"], regions, field)
                for i in range(len(samples_list))
            ]
            for field in ["ID2", "Time", "Magnitude", "Latitude", "Longitude", "Depth_m"]
        },
        "context": {
            field: [
                dict_to_array(samples_list[i]["context_dict"], regions, field)
                for i in range(len(samples_list))
            ]
            for field in ["ID2", "Time", "Magnitude", "Latitude", "Longitude", "Depth_m"]
        }
    }

    return samples_list, array_dict




def to_flag(arr_f, arr_c,arr_c_mag,Mf):
    flag_f = np.count_nonzero(arr_f, axis=1)
    flag_c = np.count_nonzero(arr_c, axis=1)

    # 初始化为 nan
    flag = np.full_like(flag_f, np.nan, dtype=float)

    # 条件赋值
    flag[flag_f > 0] = 1
    flag[(flag_c == 0) | (np.max(arr_c_mag,axis=1) < Mf)] = 0
    return flag



class EventDataset(torch.utils.data.Dataset):
    def __init__(self, array_dict,Mf, i=0):
        self.array_dict = array_dict
        self.data_fields = ["Time", "Time_norm", "Magnitude", "Latitude", "Longitude", "Depth_m"]

        self.samples = []
        self.targets = []

        future = array_dict["future"]
        context = array_dict["context"]
        num_samples = len(future["Time"])

        for idx in range(num_samples):
            target_flag = to_flag(future["Time"][idx], context["Time"][idx],context["Magnitude"][idx],Mf)
            if i < len(target_flag) and not np.isnan(target_flag[i]):
                history_data = [array_dict["history"][field][idx] for field in self.data_fields]
                target_val = target_flag[i]

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




def array_pad_time(insts, PAD=0):
    """ Pad the 2D arrays to the max column length in batch by adding padding columns in front. """
    
    # 找到 batch 中最大列数
    max_cols = max(inst.shape[1] for inst in insts)  # max number of columns in the 2D arrays
    
    # 对每个实例进行填充，列数不足的地方补0，并将新的列添加到后面
    padded_batch = [
        # np.hstack([np.full((inst.shape[0], max_cols - inst.shape[1]), PAD), inst])  # 在前面添加列
        np.hstack([inst, np.full((inst.shape[0], max_cols - inst.shape[1]), PAD)])

        for inst in insts
    ]
    
    return padded_batch

def collate_fn(instances):
    """
    Collate function to pad and combine instances into a batch, handling scalar targets directly.
    """
    sample_tuples, target_values = zip(*instances)  # Unpack the samples and scalar targets
    
    # Pad the samples' data fields (history data)
    padded_samples = [
        array_pad_time(sample, PAD=0)  # Apply padding to each feature across the batch
        for sample in zip(*sample_tuples)
    ]
    
    # Convert the padded samples into PyTorch tensors
    padded_samples = np.array(padded_samples)  # shape: (features, batch_size, time_steps, region_count)
    padded_samples = torch.tensor(padded_samples, dtype=torch.float32)
    padded_samples = padded_samples.permute(1, 2,3, 0)  # (batch_size, features, time_steps, region_count)
    
    # Convert scalar targets into tensor
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



# split_dataset 函数用于将数据集划分为训练集、验证集和测试集,现在改用torch.utils.data.subset划分，不使用该函数
def split_data(samples_list, array_dict, test_size=0.1, val_size=0.1, random_state=40, by_time=False):
    n = len(samples_list)
    indices = list(range(n))

    if by_time:
        # 按时间顺序划分（假设原始顺序就是时间先后）
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    else:
        # 随机划分
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, random_state=random_state
        )

    def extract_subset(indices):
        subset_samples = [samples_list[i] for i in indices]

        subset_array_dict = {
            "history": {
                key: [array_dict["history"][key][i] for i in indices]
                for key in array_dict["history"]
            },
            "future": {
                key: [array_dict["future"][key][i] for i in indices]
                for key in array_dict["future"]
            },
            "context": {
                key: [array_dict["context"][key][i] for i in indices]
                for key in array_dict["context"]
            }
        }
        return subset_samples, subset_array_dict

    # 提取训练集、验证集和测试集
    train_samples, train_dict = extract_subset(train_idx)
    val_samples, val_dict = extract_subset(val_idx)
    test_samples, test_dict = extract_subset(test_idx)

    return (train_samples, train_dict), (val_samples, val_dict), (test_samples, test_dict)

