from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="FORGE2022-Standard")
class FORGE2022Standard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "FORGE2022",
        data_dir: Union[str, Path] = None,
        mag_completeness: float = -1.3,
        normalize: bool = True,
        train_start_ts: Optional[float] = None,
        val_start_ts: Optional[float] = None,
        test_start_ts: Optional[float] = None,
        use_clean_injection: bool = True,
    ):
        super().__init__(
            dataset_name="FORGE2022",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
            use_clean_injection=use_clean_injection,
        )
