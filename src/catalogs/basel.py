from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="Basel-Standard")
class BaselStandard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "Basel",
        data_dir: Union[str, Path] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        use_clean_injection: bool = True,
    ):
        super().__init__(
            dataset_name="Basel",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            use_clean_injection=use_clean_injection,
        )
