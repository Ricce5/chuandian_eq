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


@Catalog.register(name="Geysers-Base")
class GeysersBase(Catalog):
    def __init__(self, root_dir: Union[str, Path], data_dir: Union[str, Path] = None, catalog_file: Union[str, Path] = None, mag_completeness: float = 2.2, normalize: bool = True):
        dataset_dir = resolve_dataset_data_dir(root_dir=root_dir, data_dir=data_dir)
        catalog_cfg = {
            "normalize": normalize,
            "mag_completeness": mag_completeness,
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=True)
        self.catalog_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename="geysers_catalog.csv",
            explicit_path=catalog_file,
        )
        self.time_series_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename="daily_injection_production.csv",
            explicit_path=None,
        )
        self.normalize = normalize
        self.metadata = {
            "name": "Geysers",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("2003-05-01"),
            "end_ts": pd.Timestamp("2015-01-31"),
            
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
        df = pd.read_csv(self.catalog_file, parse_dates=['ts'])
        df['time'] = df['ts']
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        df.sort_values("time", inplace=True)
        # 微小扰动重复时间戳，避免 inter_time = 0
        duplicated_mask = df["time"].duplicated(keep=False)
        if duplicated_mask.any():
            df.loc[duplicated_mask, "time"] += pd.to_timedelta(
                np.random.uniform(1e-8, 1e-6, duplicated_mask.sum()), unit="D"  # 1e-6D = 86.4ms 
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

        torch.save(self.norm_stats, self.root_dir / "norm_stats.pt")  

        fields["loc"] = np.stack(
            [fields["latitude"], fields["longitude"]], axis=-1
        )

        df_ts = pd.read_csv(self.time_series_file, parse_dates=['Date'])
        df_ts.set_index('Date', inplace=True)
        df_ts['t'] = (df_ts.index - start_ts) / pd.Timedelta("1D")

        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(df["Magnitude"].values, dtype=torch.float32),
            loc=fields["loc"],
            depth=fields["depth"],
            time_series=torch.tensor(df_ts[['InjVol_daily', 'ProdVol_daily']].values, dtype=torch.float32),
            time_series_times=torch.tensor(df_ts['t'].values, dtype=torch.float32),

        )

        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")

@Catalog.register(name="Geysers-Standard")
class GeysersStandard(GeysersBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 2.2,
        train_start_ts: pd.Timestamp = pd.Timestamp("2006-05-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2012-05-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2014-05-01"),
    ):
        super().__init__(
            root_dir=root_dir,
            data_dir=data_dir,
            catalog_file=catalog_file,
            mag_completeness=mag_completeness,
        )

        self.metadata["train_start_ts"] = train_start_ts
        self.metadata["val_start_ts"] = val_start_ts
        self.metadata["test_start_ts"] = test_start_ts
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
    
