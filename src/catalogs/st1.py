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
        freq: str = "1h",
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
    ):
        super().__init__(
            dataset_name="St1-2018",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
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
        freq: str = "1h",
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
            freq=freq,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
