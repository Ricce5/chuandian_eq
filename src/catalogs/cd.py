import os
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence, default_catalogs_dir
from src.catalogs.catalog_utils import train_val_test_split_sequence
from src.data.data_utils import get_split_indices


def trim(x_min, x_max, p=0.05):
    length = x_max - x_min
    return x_min + length * p, x_min + length * (1 - p)

class CDBase(Catalog):
    def __init__(self, root_dir: Union[str, Path], catalog_file: Union[str, Path] = None, mag_completeness: float = 3.0):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "name": "CD",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1970-01-01"),
            "end_ts": pd.Timestamp("2023-08-14"),
        }

        if not (self.root_dir / "metadata.pt").exists():
            assert catalog_file is not None, "Must provide catalog_file to generate dataset"
            self.generate_catalog(catalog_file)
            torch.save(self.metadata, self.root_dir / "metadata.pt")

        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self, catalog_file: Union[str, Path]):
        column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
                        'Latitude', 'Longitude', 'Depth', 'Magnitude']
        df = pd.read_csv(catalog_file, header=None, names=column_names, sep=r"\s+")
        df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce') + pd.to_timedelta(df['Second'], unit='s')
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        df.sort_values("time", inplace=True)
        df["time_diff"] = df["time"].diff().dt.total_seconds()
        df = df[df["time_diff"] > 0].copy()

        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1D")

        arrival_times = ((df["time"] - start_ts) / pd.Timedelta("1D")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(df["Magnitude"].values, dtype=torch.float32),
            latitude=torch.tensor(df["Latitude"].values, dtype=torch.float32),
            longitude=torch.tensor(df["Longitude"].values, dtype=torch.float32),
            depth=torch.tensor(df["Depth"].values, dtype=torch.float32),
        )
        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")


class CDStandard(CDBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 3.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("1975-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("1995-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2007-01-01"),
    ):
        super().__init__(root_dir, catalog_file, mag_completeness)  # 传给父类的初始化参数

        self.metadata["train_start_ts"] = train_start_ts
        self.metadata["val_start_ts"] = val_start_ts
        self.metadata["test_start_ts"] = test_start_ts

        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )

        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])



class CDSlidingWindow(CDBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 3.0,
        window_size_days: int = 365,
        step_size_days: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        super().__init__(root_dir, catalog_file, mag_completeness)

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        self.metadata["window_size_days"] = window_size_days
        self.metadata["step_size_days"] = step_size_days
        self.metadata["train_ratio"] = train_ratio
        self.metadata["val_ratio"] = val_ratio
        self.metadata["test_ratio"] = test_ratio

        self.train, self.val, self.test = self.generate_sliding_windows(
            window_size_days, step_size_days
        )

    def generate_sliding_windows(self, window_size_days=365, step_size_days=30):
        sequences = []
        arrival_times = self.full_sequence.arrival_times
        t_start = self.full_sequence.t_start
        t_end = self.full_sequence.t_end

        window_start = t_start
        while window_start + window_size_days <= t_end:
            window_end = window_start + window_size_days
            mask = (arrival_times >= window_start) & (arrival_times < window_end)
            indices = mask.nonzero().squeeze(-1).tolist()
            if len(indices) < 2:
                window_start += step_size_days
                continue  # 跳过事件太少的窗口
            seq = self.full_sequence.get_subsequence(
                start=window_start,
                end=window_end,
            )
            sequences.append(seq)
            window_start += step_size_days

        print(f"Generated {len(sequences)} sliding window sequences.")

        if len(sequences) < 3:
            raise ValueError("Too few sequences to split into train/val/test.")

        # 划分索引
        train_idx, val_idx, test_idx = get_split_indices(
            total_length=len(sequences),
            train_ratio=self.metadata['train_ratio'],
            val_ratio=self.metadata['val_ratio'],
        )

        train_dataset = TppDataset([sequences[i] for i in train_idx])
        val_dataset = TppDataset([sequences[i] for i in val_idx])
        test_dataset = TppDataset([sequences[i] for i in test_idx])

        return train_dataset, val_dataset, test_dataset

