import os
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence, default_catalogs_dir
from src.utils.catalog_utils import train_val_test_split_sequence
from src.data.utils import get_split_indices


VALID_REGIONS = {"1z", "2"}
START_TS = {
    "1z": pd.Timestamp("2018-10-15 8:00:00"),
    "2": pd.Timestamp("2019-8-13 15:00:00"),
}
END_TS = {
    "1z": pd.Timestamp("2018-12-18 13:00:00"),
    "2": pd.Timestamp("2019-10-2 6:00:00"),
}


@Catalog.register(name="PNR-Base")
class PNRBase(Catalog):
    def __init__(
        self,
        root_dir: Union[str, Path],
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = None,
        region: str = "1z",
        normalize: bool = True,
        train_start_ts: pd.Timestamp = None,
        val_start_ts: pd.Timestamp = None,    
        test_start_ts: pd.Timestamp = None,
    ):
        if region not in VALID_REGIONS:
            raise ValueError(f"Unsupported PNR region '{region}'. Supported: {sorted(VALID_REGIONS)}")

        self.region = region
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(catalog_file, (str, Path)):
            self.catalog_file = Path(catalog_file)
        elif catalog_file is None:
            self.catalog_file = self.root_dir / f"PNR_{region}_catalog.csv"
            self.time_series_file = self.root_dir / f"PNR_{region}_injection_rate_per_min.csv"
        else:
            raise TypeError("catalog_file must be a str or Path")
        self.normalize = normalize
        self.metadata = {
            "name": f"PNR_{region}",
            "freq": "1h",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "region": region,
            "start_ts": START_TS[region],
            "end_ts": END_TS[region],
        }
        if train_start_ts is not None and val_start_ts is not None and test_start_ts is not None:
            self.metadata["train_start_ts"] = pd.Timestamp(train_start_ts)
            self.metadata["val_start_ts"] = pd.Timestamp(val_start_ts)
            self.metadata["test_start_ts"] = pd.Timestamp(test_start_ts)
        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self):
        df = pd.read_csv(self.catalog_file, parse_dates=['ts'])
        print(self.catalog_file)
        print(df)
        print(df.columns)
        df["time"] = pd.to_datetime(df["ts"])
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        print(f"Magnitude completeness threshold: {self.metadata['mag_completeness']}")
        print(f"Min magnitude after completeness filter: {df['Magnitude'].min()}")
        df.sort_values("time", inplace=True)
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
        t_end = (end_ts - start_ts) / pd.Timedelta("1h")
        arrival_times = ((df["time"] - start_ts) / pd.Timedelta("1h")).values
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

        df_ts = pd.read_csv(self.time_series_file, parse_dates=['ts'])
        df_ts.set_index('ts', inplace=True)
        df_ts['t'] = (df_ts.index - start_ts) / pd.Timedelta("1h")
        print(df_ts)
        time_series = torch.tensor(df_ts[['IR_h']].values, dtype=torch.float32)
        assert torch.isnan(time_series).sum().item() == 0, "Found NaN in time series data."
        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(df["Magnitude"].values, dtype=torch.float32),
            loc=fields["loc"],
            depth=fields["depth"],
            time_series=torch.tensor(df_ts[['IR_h']].values, dtype=torch.float32),
            time_series_times=torch.tensor(df_ts['t'].values, dtype=torch.float32),

        )

        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")



@Catalog.register(name="PNR_1z-Standard")
class PNR1zStandard(PNRBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = -1.8,
        train_start_ts: pd.Timestamp = pd.Timestamp("2018-10-22"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2018-11-22"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2018-12-14"),
    ):
        super().__init__(
            root_dir=root_dir,
            catalog_file=catalog_file,
            mag_completeness=mag_completeness,
            region="1z",
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
        self._split_datasets()
    def _split_datasets(self):
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
            freq=pd.Timedelta(self.metadata["freq"]),
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])


@Catalog.register(name="PNR_2-Standard")
class PNR2Standard(PNRBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = -1.8,
        train_start_ts: pd.Timestamp = pd.Timestamp("2019-8-20"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2019-9-20"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2019-9-25"),

    ):
        super().__init__(
            root_dir=root_dir,
            catalog_file=catalog_file,
            mag_completeness=mag_completeness,
            region="2",
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
        self._split_datasets()
    def _split_datasets(self):
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
            freq=pd.Timedelta(self.metadata["freq"]),
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])


    



    