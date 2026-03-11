from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union
import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_grouped import InducedTripletGroupedCatalog
from .induced_triplet_base import InducedTripletBase


SSFS_DATASETS = ("SSFS1993", "SSFS2000", "SSFS2003", "SSFS2004", "SSFS2005")
SSFS_DEFAULT_MC = {
    "SSFS1993": -1.5,
    "SSFS2000": -0.5,
    "SSFS2003": 0.1,
    "SSFS2004": -0.8,
    "SSFS2005": -0.2,
}
SSFS_DEFAULT_SPLIT_GROUPS = {
    "train": ("SSFS1993", "SSFS2000"),
    "val": ("SSFS2003",),
    "test": ("SSFS2004", "SSFS2005"),
}
SSFS_ALIASES = {
    "1993": "SSFS1993",
    "2000": "SSFS2000",
    "2003": "SSFS2003",
    "2004": "SSFS2004",
    "2005": "SSFS2005",
}


@Catalog.register(name="SSFS-Standard")
class SSFSStandard(InducedTripletGroupedCatalog):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS",
        data_dir: Union[str, Path, None] = None,
        split_groups: Optional[Mapping[str, Union[str, Sequence[str], Path]]] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
        use_clean_injection: bool = True,
    ):
        super().__init__(
            family_name="SSFS",
            valid_datasets=SSFS_DATASETS,
            default_split_groups=SSFS_DEFAULT_SPLIT_GROUPS,
            root_dir=root_dir,
            data_dir=data_dir,
            split_groups=split_groups,
            dataset_aliases=SSFS_ALIASES,
            mag_completeness_map=SSFS_DEFAULT_MC,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
            use_clean_injection=use_clean_injection,
        )


class SSFSBase(InducedTripletBase):
    def __init__(
        self,
        dataset_name: str,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        use_clean_injection: bool = True,
    ):
        if dataset_name not in SSFS_DATASETS:
            raise ValueError(
                f"Unsupported SSFS dataset '{dataset_name}'. Supported: {SSFS_DATASETS}"
            )
        if mag_completeness is None:
            mag_completeness = SSFS_DEFAULT_MC[dataset_name]

        super().__init__(
            dataset_name=dataset_name,
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


@Catalog.register(name="SSFS1993-Standard")
class SSFS1993Standard(SSFSBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS1993",
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
            dataset_name="SSFS1993",
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


@Catalog.register(name="SSFS2000-Standard")
class SSFS2000Standard(SSFSBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS2000",
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
            dataset_name="SSFS2000",
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


@Catalog.register(name="SSFS2003-Standard")
class SSFS2003Standard(SSFSBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS2003",
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
            dataset_name="SSFS2003",
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


@Catalog.register(name="SSFS2004-Standard")
class SSFS2004Standard(SSFSBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS2004",
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
            dataset_name="SSFS2004",
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


@Catalog.register(name="SSFS2005-Standard")
class SSFS2005Standard(SSFSBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SSFS2005",
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
            dataset_name="SSFS2005",
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
