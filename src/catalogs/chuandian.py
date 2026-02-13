from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence
from src.utils.catalog_utils import train_val_test_split_sequence
from src.data.utils import get_split_indices


@Catalog.register(name="ChuanDian-Base")
class ChuanDianBase(Catalog):
    def __init__(self, root_dir: Union[str, Path], catalog_file: Union[str, Path] = None, mag_completeness: float = 3.0, normalize: bool = True):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(catalog_file, (str, Path)):
            self.catalog_file = Path(catalog_file)
        elif catalog_file is None:
            self.catalog_file = self.root_dir / "chuandian_2021.dat"
        else:
            raise TypeError("catalog_file must be a str or Path")
        self.normalize = normalize
        self.metadata = {
            "name": "ChuanDian",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1970-01-01"),
            "end_ts": pd.Timestamp("2021-05-24"),  
        }
        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self):
        column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
                        'Latitude', 'Longitude', 'Depth', 'Magnitude']
        df = pd.read_csv(self.catalog_file, header=None, names=column_names, sep=r"\s+")
        df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce') + pd.to_timedelta(df['Second'], unit='s')
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        df.sort_values("time", inplace=True)
 
        duplicated_mask = df["time"].duplicated(keep=False)
        if duplicated_mask.any():
            df.loc[duplicated_mask, "time"] += pd.to_timedelta(
                np.random.uniform(1e-30, 1e-27, duplicated_mask.sum()), unit="D"  # 1e-6D = 86.4ms 
            )
            df.sort_values("time", inplace=True)
        #
        df["time_diff"] = df["time"].diff().dt.total_seconds()
        df = df[df["time_diff"] > 0].copy()

        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1D")

        arrival_times = ((df["time"] - start_ts) / pd.Timedelta("1D")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        

        fields = {
            # "magnitude": df["Magnitude"].values,
            "latitude": df["Latitude"].values,
            "longitude": df["Longitude"].values,
            "depth": df["Depth"].values,
        }
        if self.normalize:
            fields = self.normalize_fields(fields) 

        torch.save(self.norm_stats, self.root_dir / "norm_stats.pt")  # 可选

        fields["loc"] = np.stack(
            [fields["latitude"], fields["longitude"]], axis=-1
        )

        
        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(df["Magnitude"].values, dtype=torch.float32),
            loc = torch.tensor(fields["loc"], dtype=torch.float32),
            depth= torch.tensor(fields["depth"], dtype=torch.float32),
        )

        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")

@Catalog.register(name="ChuanDian-Standard")
class ChuanDianStandard(ChuanDianBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = None,
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 3.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2000-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2015-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2018-01-01"),
        b_updater: any = None,
    ):
        super().__init__(root_dir, catalog_file, mag_completeness)

        self.metadata.update({
            "train_start_ts": train_start_ts,
            "val_start_ts": val_start_ts,
            "test_start_ts": test_start_ts,
        })
        self.b_updater = b_updater
        self._split_datasets()

    def _split_datasets(self):
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])

   


@Catalog.register(name="ChuanDian-SlidingWindow")
class ChuanDianSlidingWindow(ChuanDianBase):
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
        use_event_sequence: bool = True,
    ):
        super().__init__(root_dir, catalog_file, mag_completeness)

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        self.metadata["window_size_days"] = window_size_days
        self.metadata["step_size_days"] = step_size_days
        self.metadata["train_ratio"] = train_ratio
        self.metadata["val_ratio"] = val_ratio
        self.metadata["test_ratio"] = test_ratio
        self.metadata["use_event_sequence"] = use_event_sequence

        self.train, self.val, self.test = self.generate_sliding_windows(
            window_size_days, step_size_days
        )

    def generate_sliding_windows(self, window_size_days=365, step_size_days=30):
        sequences = []
        arrival_times = self.full_sequence.arrival_times
        t_start = self.full_sequence.t_start
        t_end = self.full_sequence.t_end

        window_start = t_start
        while window_start + window_size_days <= arrival_times[-1]:
            window_end = window_start + window_size_days
            seq = self.full_sequence.get_subsequence(
                start=window_start,
                end=window_end,
            )
            if self.metadata["use_event_sequence"]:
                 seq = seq.to_event_sequence()
            sequences.append(seq)
            window_start += step_size_days

        if len(sequences) < 3:
            raise ValueError("Too few sequences to split into train/val/test.")

        train_idx, val_idx, test_idx = get_split_indices(
            total_length=len(sequences),
            train_ratio=self.metadata['train_ratio'],
            val_ratio=self.metadata['val_ratio'],
        )

        train_dataset = TppDataset([sequences[i] for i in train_idx])
        val_dataset = TppDataset([sequences[i] for i in val_idx])
        test_dataset = TppDataset([sequences[i] for i in test_idx])

        return train_dataset, val_dataset, test_dataset
