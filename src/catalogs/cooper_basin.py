from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union
import pandas as pd

from src.data import Catalog, default_catalogs_dir

from .induced_triplet_base import InducedTripletBase
from .induced_triplet_grouped import InducedTripletGroupedCatalog


COOPER_BASIN_DATASETS = ("CB_HAB1a", "CB_HAB1b", "CB_HAB4")
COOPER_BASIN_DEFAULT_MC = {
    "CB_HAB1a": -0.4,
    "CB_HAB1b": -0.2,
    "CB_HAB4": 0.9,
}
COOPER_BASIN_DEFAULT_SPLIT_GROUPS = {
    "train": ("CB_HAB1a",),
    "val": ("CB_HAB1b",),
    "test": ("CB_HAB4",),
}
COOPER_BASIN_ALIASES = {
    "1a": "CB_HAB1a",
    "1b": "CB_HAB1b",
    "4": "CB_HAB4",
    "hab1a": "CB_HAB1a",
    "hab1b": "CB_HAB1b",
    "hab4": "CB_HAB4",
}


@Catalog.register(name="CooperBasin-Standard")
class CooperBasinStandard(InducedTripletGroupedCatalog):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CooperBasin" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        split_groups: Optional[Mapping[str, Union[str, Sequence[str], Path]]] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
        event_feature_builder: Optional[str] = None,
        event_feature_cfg: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(
            family_name="CooperBasin",
            valid_datasets=COOPER_BASIN_DATASETS,
            default_split_groups=COOPER_BASIN_DEFAULT_SPLIT_GROUPS,
            root_dir=root_dir,
            data_dir=data_dir,
            split_groups=split_groups,
            dataset_aliases=COOPER_BASIN_ALIASES,
            mag_completeness_map=COOPER_BASIN_DEFAULT_MC,
            mag_completeness=mag_completeness,
            normalize=normalize,
            resample_freq_min=resample_freq_min,
            freq=freq,
            event_feature_builder=event_feature_builder,
            event_feature_cfg=event_feature_cfg,
        )


class CooperBasinBase(InducedTripletBase):
    def __init__(
        self,
        dataset_name: str,
        root_dir: Union[str, Path],
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
        if dataset_name not in COOPER_BASIN_DATASETS:
            raise ValueError(
                f"Unsupported Cooper Basin dataset '{dataset_name}'. "
                f"Supported: {COOPER_BASIN_DATASETS}"
            )
        if mag_completeness is None:
            mag_completeness = COOPER_BASIN_DEFAULT_MC[dataset_name]

        super().__init__(
            dataset_name=dataset_name,
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


@Catalog.register(name="CB_HAB1a-Standard")
class CBHAB1aStandard(CooperBasinBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CB_HAB1a" / "catalogs",
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
            dataset_name="CB_HAB1a",
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


@Catalog.register(name="CB_HAB1b-Standard")
class CBHAB1bStandard(CooperBasinBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CB_HAB1b" / "catalogs",
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
            dataset_name="CB_HAB1b",
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


@Catalog.register(name="CB_HAB4-Standard")
class CBHAB4Standard(CooperBasinBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CB_HAB4" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = None,
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = '2012-11-28 12:00:00',
        test_start_ts: Optional[Union[pd.Timestamp, str]] = '2012-11-30 00:00:00',
        event_feature_builder: Optional[str] = None,
        event_feature_cfg: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(
            dataset_name="CB_HAB4",
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
