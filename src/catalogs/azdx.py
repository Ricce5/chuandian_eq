from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence
from src.utils.catalog_utils import train_val_test_split_sequence_float
from src.utils.catalog_pathing import (
    build_hashed_catalog_root,
    refresh_cached_metadata,
    resolve_dataset_data_dir,
    resolve_source_file,
)


@Catalog.register(name="AZDX-Base")
class AZDXBase(Catalog):
    def __init__(self, root_dir: Union[str, Path], data_dir: Union[str, Path] = None, catalog_file: Union[str, Path] = None, mag_completeness: float = 4.5, normalize: bool = True):
        dataset_dir = resolve_dataset_data_dir(root_dir=root_dir, data_dir=data_dir)
        catalog_cfg = {
            "normalize": normalize,
            "mag_completeness": mag_completeness,
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=True)
        self.catalog_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename="size_3k_stress_0.6_dyn_0.8_mm_5.5_dm_0.5_b_0.4_yr_20k_eq.dat",
            explicit_path=catalog_file,
        )
        self.normalize = normalize
        self.metadata = {
            "name": "AZDX",
            "freq": "1Y",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": 115,
            "end_ts": 20001,
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
        column_names =["ID1", "ID2", "time", "Magnitude", "Depth", "Longitude", "Latitude"]
        df = pd.read_csv(self.catalog_file, header=None, names=column_names, sep=r"\s+")
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        df.sort_values("time", inplace=True)
        df["time_diff"] = df["time"].diff()
        df = df[df["time_diff"] > 0].copy()

        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        t_start = 0.0
        t_end = end_ts - start_ts

        arrival_times = (df["time"] - start_ts) 
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
            loc=fields["loc"],
            depth=fields["depth"],
        )

        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")

@Catalog.register(name="AZDX-Standard")
class AZDXStandard(AZDXBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 4.5,
        train_start_ts: float = 2000,
        val_start_ts:   float = 10000,
        test_start_ts: float = 15000,
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
        seq_train, seq_val, seq_test = train_val_test_split_sequence_float(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])
