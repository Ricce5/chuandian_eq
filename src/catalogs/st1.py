from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="St1-2018-Standard")
class St12018Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "St1-2018" / "catalogs",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = 0.0,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = "2018-8-21T23:59:59", 
         # 后面数据没有注水但地震大量增加，应该是注水数据缺失了，所以截止到2018-8-21
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = "2018-7-14T00:00:00",
        test_start_ts: Optional[Union[pd.Timestamp, str]] = "2018-7-20T00:00:00",
    ):
        super().__init__(
            dataset_name="St1-2018",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            resample_freq_min=resample_freq_min,
            freq=freq,
            end_ts=end_ts,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )


@Catalog.register(name="St1-2020-Standard")
class St12020Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "St1-2020" / "catalogs",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = -1.3,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = "2020-6-16T01:00:00",
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
    ):
        super().__init__(
            dataset_name="St1-2020",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            resample_freq_min=resample_freq_min,
            freq=freq,
            end_ts=end_ts,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
