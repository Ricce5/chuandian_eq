from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog, TppDataset, Sequence, default_catalogs_dir
from src.utils.catalog_utils import train_test_split_sequence, train_val_test_split_sequence
from src.utils.catalog_pathing import (
    build_hashed_catalog_root,
    refresh_cached_metadata,
    resolve_dataset_data_dir,
    resolve_source_file,
    to_serializable_ts,
)


VALID_REGIONS = {"1z", "2"}
START_TS = {
    "1z": pd.Timestamp("2018-10-15 8:00:00"),
    "2": pd.Timestamp("2019-8-13 15:00:00"),
}
END_TS = {
    "1z": pd.Timestamp("2018-12-18 13:00:00"),
    "2": pd.Timestamp("2019-10-2 6:00:00"),
}

MAG_COMPLETENESS = {
    "1z": -1.8,
    "2": -1.8,
    "all": -1.8,
}


@Catalog.register(name="PNR-Base")
class PNRBase(Catalog):
    def __init__(
        self,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        mag_completeness: float = None,
        region: str = "1z",
        normalize: bool = True,
        train_start_ts: pd.Timestamp = None,
        val_start_ts: pd.Timestamp = None,    
        test_start_ts: pd.Timestamp = None,
        freq: str = "1h",
    ):
        catalog_cfg = {
            "region": region,
            "normalize": normalize,
            "mag_completeness": mag_completeness,
            "freq": freq,
            "train_start_ts": to_serializable_ts(train_start_ts),
            "val_start_ts": to_serializable_ts(val_start_ts),
            "test_start_ts": to_serializable_ts(test_start_ts),
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=True)
        if region not in VALID_REGIONS:
            raise ValueError(f"Unsupported PNR region '{region}'. Supported: {sorted(VALID_REGIONS)}")

        if mag_completeness is None:
            mag_completeness = MAG_COMPLETENESS.get(region, MAG_COMPLETENESS["all"])

        self.region = region
        self.mag_completeness = mag_completeness
        dataset_dir = resolve_dataset_data_dir(root_dir=root_dir, data_dir=data_dir)
        self.catalog_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename=f"PNR_{region}_catalog.csv",
        )
        self.time_series_file = resolve_source_file(
            dataset_dir=dataset_dir,
            default_filename=f"PNR_{region}_injection_rate_per_min.csv",
        )
        self.normalize = normalize
        self.metadata = {
            "name": f"PNR_{region}",
            "freq": freq,
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
        df["time"] = pd.to_datetime(df["ts"])
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]
        df = df[df["Magnitude"] > self.mag_completeness].copy()
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
        t_end = (end_ts - start_ts) / pd.Timedelta(self.metadata["freq"])
        arrival_times = ((df["time"] - start_ts) / pd.Timedelta(self.metadata["freq"])).values
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
        df_ts['t'] = (df_ts.index - start_ts) / pd.Timedelta(self.metadata["freq"])
        time_series = torch.tensor(df_ts[['IR_h']].values, dtype=torch.float32)
        assert torch.isnan(time_series).sum().item() == 0, "Found NaN in time series data."
        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(df["Magnitude"].values, dtype=torch.float32),
            loc=fields["loc"],
            depth=fields["depth"],
            time_series=time_series,
            time_series_times=torch.tensor(df_ts['t'].values, dtype=torch.float32),

        )

        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")



@Catalog.register(name="PNR_1z-Standard")
class PNR1zStandard(PNRBase):
    def __init__(
        self,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        mag_completeness: float = MAG_COMPLETENESS["1z"],
        train_start_ts: pd.Timestamp = pd.Timestamp("2018-10-22"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2018-11-22"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2018-12-14"),
        freq: str = "1h",
    ):
        super().__init__(
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            region="1z",
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            freq=freq,
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
        data_dir: Union[str, Path] = None,
        mag_completeness: float = MAG_COMPLETENESS["2"],
        train_start_ts: pd.Timestamp = pd.Timestamp("2019-8-20"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2019-8-24"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2019-9-25"),
        freq: str = "1h",

    ):
        super().__init__(
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            region="2",
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            freq=freq,
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


@Catalog.register(name="PNR-Standard")
class PNRStandard(Catalog):
    def __init__(
        self,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        mag_completeness: float = MAG_COMPLETENESS["all"],
        freq: str = "1h",
        region_split: tuple = ("1z", "2", "2"),
        train_start_ts: pd.Timestamp = pd.Timestamp("2018-10-22"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2019-8-20"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2019-8-24"),
    ):
        catalog_cfg = {
            "region_split": region_split,
            "mag_completeness": mag_completeness,
            "freq": freq,
            "train_start_ts": to_serializable_ts(train_start_ts),
            "val_start_ts": to_serializable_ts(val_start_ts),
            "test_start_ts": to_serializable_ts(test_start_ts),
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=False)

        dataset_dir = resolve_dataset_data_dir(root_dir=root_dir, data_dir=data_dir)
        self.norm_stats = {}
        if dataset_dir.name == "PNR":
            data_root = dataset_dir.parent
        elif (dataset_dir / "PNR_1z").exists() and (dataset_dir / "PNR_2").exists():
            data_root = dataset_dir
        else:
            data_root = default_catalogs_dir

        pnr_1z_dir = data_root / "PNR_1z"
        pnr_2_dir = data_root / "PNR_2"
        data_dir_1z = pnr_1z_dir / "catalogs"
        data_dir_2 = pnr_2_dir / "catalogs"


        self.catalog_1z = PNR1zStandard(root_dir=data_dir_1z, 
                                        data_dir=pnr_1z_dir,
                                        mag_completeness=mag_completeness,freq=freq)
        self.catalog_2 = PNR2Standard(root_dir=data_dir_2, 
                                      data_dir=pnr_2_dir,
                                      mag_completeness=mag_completeness,freq=freq)   

        self.sequences  = [self.catalog_1z.full_sequence, self.catalog_2.full_sequence]
        self.full_sequence = self.catalog_2.full_sequence

        self.metadata = self.catalog_1z.metadata.copy()
        self.metadata["name"] = "PNR"
        self.metadata["train_region"] = region_split[0]
        self.metadata["val_region"] = region_split[1]
        self.metadata["test_region"] = region_split[2]
        self.metadata['end_ts'] = self.catalog_2.metadata['end_ts']
        self.metadata['train_start_ts'] = pd.Timestamp(train_start_ts)
        self.metadata["val_start_ts"] = pd.Timestamp(val_start_ts)
        self.metadata["test_start_ts"] = pd.Timestamp(test_start_ts)
        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self._split_datasets()

    def _split_datasets(self):
        train_region = self.metadata["train_region"]
        val_region = self.metadata["val_region"]
        test_region = self.metadata["test_region"]

        if (train_region, val_region, test_region) == ("1z", "1z", "2"):
            seq_train, seq_val = train_test_split_sequence(
                seq=self.catalog_1z.full_sequence,
                start_ts=self.catalog_1z.metadata["start_ts"],
                train_start_ts=self.metadata["train_start_ts"],
                test_start_ts=self.metadata["val_start_ts"],
                freq=pd.Timedelta(self.catalog_1z.metadata["freq"]),
            )
            seq_test = self.catalog_2.full_sequence
            seq_test.t_nll_start = self.metadata["test_start_ts"]

        elif (train_region, val_region, test_region) == ("1z", "2", "2"):
            seq_train, _ = train_test_split_sequence(
                seq=self.catalog_1z.full_sequence,
                start_ts=self.catalog_1z.metadata["start_ts"],
                train_start_ts=self.metadata["train_start_ts"],
                test_start_ts=self.catalog_1z.metadata["end_ts"],
                freq=pd.Timedelta(self.catalog_1z.metadata["freq"]),
            )

          
            seq_val, seq_test = train_test_split_sequence(
                seq=self.catalog_2.full_sequence,
                start_ts=self.catalog_2.metadata["start_ts"],
                train_start_ts=self.metadata["val_start_ts"],
                test_start_ts=self.metadata["test_start_ts"],
                freq=pd.Timedelta(self.catalog_2.metadata["freq"]),
            )

        else:
            raise ValueError(
                f"Unsupported region combination for PNRStandard: "
                f"train_region={train_region}, val_region={val_region}, test_region={test_region}"
            )

        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])

    def generate_catalog(self):
        # No standalone generation: this catalog is composed from region-specific catalogs.
        return None
    
    @property
    def required_files(self):
        return ["metadata.pt"]



    



    
