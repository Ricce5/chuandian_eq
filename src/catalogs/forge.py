from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="FORGE2022-Standard")
class FORGE2022Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "FORGE2022" / "catalogs",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = -1.3,
        normalize: bool = True,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = None,
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = '2022-04-21 20:00:00',
        test_start_ts: Optional[Union[pd.Timestamp, str]] = '2022-04-21 23:00:00',
    ):
        super().__init__(
            dataset_name="FORGE2022",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
            end_ts=end_ts,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
