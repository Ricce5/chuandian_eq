import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset, WeightedRandomSampler

from .constants import PAD
from .normalization import (
    DEFAULT_MAG_MAX,
    DEFAULT_MAG_MIN,
    inverse_normalize_magnitude_range,
    normalize_magnitude_range,
    validate_magnitude_bounds,
)
from src.data.utils import get_split_indices

plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

FUTURE_CONTEXT_FIELDS = ["t", "Magnitude", "Latitude", "Longitude", "Depth"]
HISTORY_BASE_FIELDS = ["t", "Magnitude", "Latitude", "Longitude", "Depth", "dt"]
HISTORY_ARRAY_FIELDS = ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth", "dt"]


def compute_classification_label(arr_f_t, arr_c_mag, Mf):
    if np.count_nonzero(arr_f_t) > 0:
        return 1
    if len(arr_c_mag) == 0 or np.max(arr_c_mag) < Mf:
        return 0
    return np.nan


def build_classification_labels(array_dict, Mf):
    future_t = array_dict["future"]["t"]
    context_mag = array_dict["context"]["Magnitude"]
    labels = np.array(
        [compute_classification_label(arr_f_t, arr_c_mag, Mf) for arr_f_t, arr_c_mag in zip(future_t, context_mag)],
        dtype=float,
    )
    valid_mask = ~np.isnan(labels)
    return labels, valid_mask



def normalize_df(df):
    # magnitude_cols = [col for col in df.columns if 'Mag' in col]
    lat_cols = [col for col in df.columns if 'Lat' in col]
    lon_cols = [col for col in df.columns if 'Lon' in col]
    dep_cols = [col for col in df.columns if 'Dep' in col]

    df_nl = df.copy()
    scalers = {}

    # for col_list in [lat_cols, lon_cols, dep_cols, magnitude_cols]:
    for col_list in [lat_cols, lon_cols, dep_cols]:
        for col in col_list:
            scaler = MinMaxScaler()
            df_nl[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
    return df_nl, scalers



def extract_field_array(dict_list, field):
    values = [sample[field] for sample in dict_list]
    return np.array(values) if values else np.array([])


def generate_time_array(df, Twindow, Tfore, dt, t_array=None):
    t = df["t"].values
    t.sort()
    logger.info("Number of earthquake events (greater than Mc): %s", len(t))

    if t_array is not None:
        t_array = np.array(t_array)
        if len(t_array) == 0:
            raise ValueError("t_array cannot be empty")
        if np.any(t_array < t[0] + Twindow) or np.any(t_array > t[-1]):
            logger.warning("t_array contains values outside the data time range")
        return t_array
    else:
        Nloop = int(np.ceil((t[-1] - t[0] - Twindow - Tfore) / dt))
        return Twindow + t[0] + np.arange(Nloop) * dt


def generate_single_sample(df, df_nl, t_now, Twindow, Tfore, Mf, context_len):
    t_values = df["t"]
    history = df[(t_values < t_now) & (t_values >= t_now - Twindow)]
    future_shocks = df[(t_values >= t_now) & (t_values < t_now + Tfore) & (df["Magnitude"] >= Mf)]
    context = df[(t_values >= t_now - context_len * Tfore) & (t_values < t_now + context_len * Tfore)]

    history_dict = []
    if not history.empty:
        history_frame = df_nl.loc[history.index, HISTORY_BASE_FIELDS].copy()
        history_frame["t_nl"] = (history["t"].to_numpy() - t_now) / Twindow + 1
        history_dict = history_frame.to_dict(orient="records")

    return {
        "history_dict": history_dict,
        "future_dict": future_shocks[FUTURE_CONTEXT_FIELDS].to_dict(orient="records"),
        "context_dict": context[FUTURE_CONTEXT_FIELDS].to_dict(orient="records"),
        "t": t_now,
    }


def construct_samples_list(df, df_nl, Mc, Mf=None, Twindow=20, Tfore=2, dt=10, t_array=None, context_len=0):
    if Mf is None:
        Mf = Mc

    df_filtered = df[df['Magnitude'] >= Mc].copy()
    t_array_final = generate_time_array(df_filtered, Twindow, Tfore, dt, t_array)

    samples_list = [
        generate_single_sample(df_filtered, df_nl, t_now, Twindow, Tfore, Mf, context_len)
        for t_now in t_array_final
    ]

    array_dict = {
        "history": {
            field: [extract_field_array(sample["history_dict"], field) for sample in samples_list]
            for field in HISTORY_ARRAY_FIELDS
        },
        "future": {
            field: [extract_field_array(sample["future_dict"], field) for sample in samples_list]
            for field in FUTURE_CONTEXT_FIELDS
        },
        "context": {
            field: [extract_field_array(sample["context_dict"], field) for sample in samples_list]
            for field in FUTURE_CONTEXT_FIELDS
        }
    }

    return samples_list, array_dict


class EventDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        array_dict,
        Mf=None,
        task_type='classification',
        mag_min=DEFAULT_MAG_MIN,
        mag_max=DEFAULT_MAG_MAX,
    ):
        assert task_type in ["classification", "regression", "count"], "Unsupported task type"
        self.task_type = task_type
        self.array_dict = array_dict
        self.Mf = Mf
        self.mag_min, self.mag_max = validate_magnitude_bounds(mag_min, mag_max)
        self.data_fields = HISTORY_ARRAY_FIELDS

        self.samples = []
        self.labels = []            # labels normalized by fixed [mag_min, mag_max]
        self.lengths = []

        future = array_dict["future"]
        context = array_dict["context"]
        num_samples = len(future["t"])

        for idx in range(num_samples):
            if self.task_type == "classification":
                label = self.compute_flag_label(future["t"][idx], context["Magnitude"][idx])
            elif self.task_type == "regression":
                label = self.compute_max_magnitude_label(future["Magnitude"][idx])  # fixed-range normalization
            elif self.task_type == "count":
                label = self.compute_count_label(future["t"][idx])
            else:
                raise ValueError("unsupported task type. Supported types are: 'classification', 'regression', 'count'.")
            if not np.isnan(label):
                history_data = [np.array(array_dict["history"][field][idx]) for field in self.data_fields]
                self.samples.append(history_data)
                self.labels.append(label)
                self.lengths.append(len(history_data[0]))

        if self.lengths:
            logger.info(
                "lengths of samples: min=%s, max=%s, avg=%.2f",
                min(self.lengths),
                max(self.lengths),
                sum(self.lengths) / len(self.lengths),
            )
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def compute_flag_label(self, arr_f_t, arr_c_mag):
        return compute_classification_label(arr_f_t, arr_c_mag, self.Mf)
    
    def compute_count_label(self, arr_f_t):
        return np.count_nonzero(arr_f_t)

    def compute_max_magnitude_label(self, arr_f_mag):
        if len(arr_f_mag) > 0:
            max_mag = np.max(arr_f_mag)
            return float(normalize_magnitude_range(max_mag, mag_min=self.mag_min, mag_max=self.mag_max))
        return np.nan

    def inverse_normalize_label(self, norm_value):
        return inverse_normalize_magnitude_range(norm_value, mag_min=self.mag_min, mag_max=self.mag_max)

    
    @property
    def pos_count(self):
        return sum(1 for t in self.labels if t == 1)
    @property
    def neg_count(self):
        return sum(1 for t in self.labels if t == 0)



def pad_1d_sequences(insts, PAD):
    max_len = max(len(inst) for inst in insts)
    padded_batch = [np.pad(inst, (0, max_len - len(inst)), constant_values=PAD) for inst in insts]
    return np.stack(padded_batch)


def collate_fn(instances):
    sample_tuples, target_values = zip(*instances)
    padded_samples = [pad_1d_sequences(sample, PAD=PAD) for sample in zip(*sample_tuples)]
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
        pos = sum(1 for _, target in subset if target == 1)
        neg = sum(1 for _, target in subset if target == 0)
    return pos, neg





def get_balanced_sampler(dataset):
    labels = [label for _, label in dataset]  # assume dataset[i] = (data, label)
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    sampler=None,
    task_type='classification',
    full_batch: bool = False,
):
    if task_type == 'classification':
        pos_count, neg_count = count_pos_neg(dataset)
        total = len(dataset)
        if total > 0:
            logger.info("Positive samples: %s, Negative samples: %s", pos_count, neg_count)
            logger.info(
                "Total samples: %s, Positive ratio: %.2f, Negative ratio: %.2f",
                total,
                pos_count / total,
                neg_count / total,
            )
        else:
            logger.warning("Classification dataset is empty.")

    if full_batch:
        if len(dataset) == 0:
            raise ValueError("Cannot create full-batch DataLoader from an empty dataset.")
        batch_size = len(dataset)
        shuffle = False
        sampler = None

    if sampler is not None:
        shuffle = False  

    dl = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        sampler=sampler
    )
    return dl



def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=0, by_time=True, time_order=('train', 'val', 'test'),task_type='classification'):
    total = len(dataset)
    if task_type == 'classification':
        logger.info("Number of positive samples: %s", dataset.pos_count)
        logger.info("Number of negative samples: %s", dataset.neg_count)
    train_idx, val_idx, test_idx = get_split_indices(
        total_length=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        by_time=by_time,
        time_order= time_order
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set



def get_sequence_length_stats(array_dict):
    history_times = array_dict["history"]["t"]
    lengths = [len(seq) for seq in history_times]

    if not lengths:
        logger.warning("No sequences found.")
        return

    logger.info(
        "Sequence lengths - min: %s, max: %s, avg: %.2f",
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths),
    )
    return lengths


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
