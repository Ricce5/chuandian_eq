# Reference: https://zenodo.org/records/8161777 - Using Deep Learning for Flexible and Scalable Earthquake Forecasting.
import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch
from src.data import Catalog, TppDataset, Sequence, default_catalogs_dir
from ..utils.catalog_utils import train_val_test_split_sequence
from src.utils.catalog_pathing import (
    build_hashed_catalog_root,
    refresh_cached_metadata,
    to_serializable_ts,
)

COL_NAMES = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "SCSN_cuspid",
    "latitude",
    "longitude",
    "depth",
    "magnitude",
    "number_of_P_S_picks",
    "dist_to_nearest_station", "rms_residual",
    "day_night",
    "location_method",
    "cluster_id",
    "cluster_size",
    "nt_loc",
    "horz_std",
    "depth_std",
    "horz_clust_std",
    "depth_clust_std",
    "event_type",
    "relocation_method",
    "poly_ID",
]


@Catalog.register(name="Hauksson-Standard")
class Hauksson(Catalog):
    url = "https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/sc_1981_2019_1d_3d_gc_soda_noqb_v0.gc"

    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "Hauksson" / "catalogs",
        mag_completeness: float = 1.0,
        val_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2016-01-01"),
    ):
        val_start_ts = pd.Timestamp(val_start_ts)
        test_start_ts = pd.Timestamp(test_start_ts)
        catalog_cfg = {
            "mag_completeness": float(mag_completeness),
            "val_start_ts": to_serializable_ts(val_start_ts),
            "test_start_ts": to_serializable_ts(test_start_ts),
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=True)
        metadata = {
            "name": f"HaukssonEtAl",
            "freq": "1D",
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1981-01-01"),
            "end_ts": pd.Timestamp("2020-01-01"),
            "val_start_ts": val_start_ts,
            "test_start_ts": test_start_ts,
        }
        refresh_cached_metadata(
            root_dir=self.root_dir,
            metadata=metadata,
            required_files=("full_sequence.pt",),
        )
        super().__init__(root_dir=self.root_dir, metadata=metadata)

        # Load the full sequence
        self.full_sequence = TppDataset.load_from_disk(
            self.root_dir / "full_sequence.pt"
        )[0]

        # Split full sequence into train / val / test parts
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self):
        stream = requests.get(self.url).content
        raw_df = pd.read_csv(
            io.StringIO(stream.decode("utf-8")),
            sep=r"\s+",
            header=None,
            names=COL_NAMES,
            index_col=False,
        )
        raw_df["date"] = pd.to_datetime(
            raw_df[["year", "month", "day", "hour", "minute", "second"]]
        )
        raw_df.sort_values(by=["date"], inplace=True)
        subset_df = raw_df.loc[raw_df["magnitude"] > self.metadata["mag_completeness"]]

        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        assert subset_df.date.min() > start_ts
        assert subset_df.date.max() < end_ts

        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")
        arrival_times = ((subset_df.date - start_ts) / pd.Timedelta("1 day")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        mag = subset_df["magnitude"].values
        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            mag=torch.as_tensor(mag, dtype=torch.float32),
        )
        dataset = TppDataset(sequences=[seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")
