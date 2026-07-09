from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="CCL-Standard")
class CCLStandard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CCL" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = None,
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        event_feature_builder: Optional[str] = None,
        event_feature_cfg: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(
            dataset_name="CCL",
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
            event_feature_builder=event_feature_builder,
            event_feature_cfg=event_feature_cfg,
        )
