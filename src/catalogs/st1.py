from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="St1-2018-Standard")
class St12018Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "St1-2018",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = 0.0,
        normalize: bool = True,
        train_start_ts: Optional[float] = None,
        val_start_ts: Optional[float] = None,
        test_start_ts: Optional[float] = None,
        use_clean_injection: bool = True,
    ):
        super().__init__(
            dataset_name="St1-2018",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            use_clean_injection=use_clean_injection,
        )


@Catalog.register(name="St1-2020-Standard")
class St12020Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "St1-2020",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = -1.3,
        normalize: bool = True,
        train_start_ts: Optional[float] = None,
        val_start_ts: Optional[float] = None,
        test_start_ts: Optional[float] = None,
        use_clean_injection: bool = True,
    ):
        super().__init__(
            dataset_name="St1-2020",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            use_clean_injection=use_clean_injection,
        )
