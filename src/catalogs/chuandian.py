from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence
from src.utils.catalog_utils import train_val_test_split_sequence
from src.utils.catalog_pathing import (
    build_hashed_catalog_root,
    refresh_cached_metadata,
    resolve_dataset_data_dir,
    resolve_source_file,
)


@Catalog.register(name="ChuanDian-Base")
class ChuanDianBase(Catalog):
    def __init__(self, root_dir: Union[str, Path], data_dir: Union[str, Path] = None, catalog_file: Union[str, Path] = None, mag_completeness: float = 3.0, normalize: bool = True):
        dataset_dir = resolve_dataset_data_dir(root_dir=root_dir, data_dir=data_dir)
        catalog_cfg = {
            "normalize": normalize,
            "mag_completeness": mag_completeness,
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=True)
        self.catalog_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename="chuandian_2021.dat",
            explicit_path=catalog_file,
        )
        self.normalize = normalize
        self.metadata = {
            "name": "ChuanDian",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1970-01-01"),
            "end_ts": pd.Timestamp("2021-05-24"),  
        }
        refresh_cached_metadata(
            root_dir=self.root_dir,
            metadata=self.metadata,
            required_files=("full_sequence.pt",),
        )
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
        data_dir: Union[str, Path] = None,
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 3.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2000-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2015-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2018-01-01"),
        b_updater: any = None,
    ):
        super().__init__(
            root_dir=root_dir,
            data_dir=data_dir,
            catalog_file=catalog_file,
            mag_completeness=mag_completeness,
        )

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

   
