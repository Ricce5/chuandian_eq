from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="Geysers-Standard")
class GeysersStandard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "Geysers" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        mag_completeness: Optional[float] = 2.2,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1D",
        end_ts: Optional[Union[pd.Timestamp, str]] = "2015-01-31T00:00:00",
        train_start_ts: Optional[Union[pd.Timestamp, str]] = "2006-05-01T00:00:00",
        val_start_ts: Optional[Union[pd.Timestamp, str]] = "2012-05-01T00:00:00",
        test_start_ts: Optional[Union[pd.Timestamp, str]] = "2014-05-01T00:00:00",
    ):
        super().__init__(
            dataset_name="Geysers",
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
