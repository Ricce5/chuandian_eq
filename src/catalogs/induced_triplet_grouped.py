from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from src.data import Catalog, Sequence as TppSequence, TppDataset, default_catalogs_dir
from src.utils.catalog_pathing import build_hashed_catalog_root

from .induced_triplet_base import InducedTripletBase

SplitGroupValue = Union[str, Sequence[str], Path]
SplitGroups = Mapping[str, SplitGroupValue]
SPLIT_NAMES = ("train", "val", "test")


def _normalize_group_items(value: SplitGroupValue) -> list[str]:
    if isinstance(value, Path):
        raw_items = [value.name]
    elif isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Sequence):
        raw_items = [str(item) for item in value]
    else:
        raise TypeError(f"Unsupported split group type: {type(value)}")

    tokens: list[str] = []
    for raw in raw_items:
        basename = Path(raw).name
        for token in re.split(r"[+,]", basename):
            stripped = token.strip()
            if stripped:
                tokens.append(stripped)
    if not tokens:
        raise ValueError(f"Split group {value!r} resolves to an empty dataset list.")
    return tokens


def _resolve_dataset_name(
    token: str,
    valid_datasets: Sequence[str],
    aliases: Mapping[str, str],
) -> str:
    if token in valid_datasets:
        return token
    token_l = token.lower()
    if token_l in aliases:
        resolved = aliases[token_l]
        if resolved in valid_datasets:
            return resolved
    raise ValueError(f"Unsupported dataset token '{token}'. Valid datasets: {tuple(valid_datasets)}")


def _normalize_split_groups(
    split_groups: SplitGroups,
    valid_datasets: Sequence[str],
    aliases: Mapping[str, str],
) -> Dict[str, list[str]]:
    missing = [name for name in SPLIT_NAMES if name not in split_groups]
    if missing:
        raise ValueError(f"split_groups must contain keys {SPLIT_NAMES}. Missing: {missing}.")

    out: Dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        tokens = _normalize_group_items(split_groups[split])
        resolved: list[str] = []
        for token in tokens:
            name = _resolve_dataset_name(token, valid_datasets, aliases)
            if name in resolved:
                continue
            if name in seen:
                raise ValueError(
                    f"Dataset '{name}' appears in multiple splits. "
                    "Grouped split datasets must be disjoint."
                )
            resolved.append(name)
            seen.add(name)
        if not resolved:
            raise ValueError(f"Split '{split}' cannot be empty.")
        out[split] = resolved
    return out


def _event_attr_keys(seq: TppSequence) -> set[str]:
    return {key for key in seq.keys() if key not in TppSequence.default_sequence_attrs}


def _has_time_series(seq: TppSequence) -> bool:
    return "time_series" in seq and "time_series_times" in seq


def _merge_sequences(sequences: Sequence[TppSequence]) -> TppSequence:
    if not sequences:
        raise ValueError("Cannot merge an empty sequence list.")

    attr_keys = sorted(_event_attr_keys(sequences[0]))
    has_time_series = _has_time_series(sequences[0])
    for seq in sequences[1:]:
        if _event_attr_keys(seq) != set(attr_keys):
            raise ValueError("All grouped datasets must share the same event attributes.")
        if _has_time_series(seq) != has_time_series:
            raise ValueError(
                "All grouped datasets must either all contain or all omit time-series data."
            )

    offset = 0.0
    merged_inter_times: Optional[np.ndarray] = None
    attr_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in attr_keys}
    ts_chunks: list[torch.Tensor] = []
    ts_time_chunks: list[torch.Tensor] = []
    prev_ts_end: Optional[float] = None

    for seq in sequences:
        duration = float(seq.t_end - seq.t_start)
        seq_inter = seq.inter_times.detach().cpu().numpy().astype(np.float64)
        if merged_inter_times is None:
            merged_inter_times = seq_inter.copy()
        else:
            merged_inter_times[-1] += seq_inter[0]
            if seq_inter.shape[0] > 1:
                merged_inter_times = np.concatenate([merged_inter_times, seq_inter[1:]], axis=0)

        for key in attr_keys:
            attr_chunks[key].append(seq[key])

        if has_time_series:
            shifted_ts_times = (
                seq.time_series_times.detach().cpu().numpy().astype(np.float64)
                - float(seq.t_start)
                + offset
            )
            if shifted_ts_times.size > 0 and prev_ts_end is not None and shifted_ts_times[0] <= prev_ts_end:
                local_dt = np.diff(shifted_ts_times)
                positive_dt = local_dt[local_dt > 0]
                step = float(positive_dt.min()) if positive_dt.size else 1e-6
                shifted_ts_times += (prev_ts_end - shifted_ts_times[0]) + step

            if shifted_ts_times.size > 0:
                prev_ts_end = float(shifted_ts_times[-1])
            ts_chunks.append(seq.time_series)
            ts_time_chunks.append(torch.tensor(shifted_ts_times, dtype=torch.float64))

        offset += duration

    if merged_inter_times is None:
        merged_inter_times = np.array([offset], dtype=np.float64)

    seq_kwargs: dict = {
        "inter_times": torch.tensor(merged_inter_times, dtype=torch.float32),
        "t_start": 0.0,
        "t_nll_start": 0.0,
    }
    for key, chunks in attr_chunks.items():
        seq_kwargs[key] = torch.cat(chunks, dim=0)

    if has_time_series:
        seq_kwargs["time_series"] = torch.cat(ts_chunks, dim=0)
        seq_kwargs["time_series_times"] = torch.cat(ts_time_chunks, dim=0)

    return TppSequence(**seq_kwargs)


class InducedTripletGroupedCatalog(Catalog):
    """Compose train/val/test from multiple `InducedTripletBase` datasets."""

    def __init__(
        self,
        family_name: str,
        valid_datasets: Sequence[str],
        default_split_groups: SplitGroups,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path, None] = None,
        split_groups: Optional[SplitGroups] = None,
        dataset_aliases: Optional[Mapping[str, str]] = None,
        mag_completeness_map: Optional[Mapping[str, float]] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        resample_freq_min: Optional[int] = None,
        freq: str = "1h",
    ):
        self.family_name = family_name
        self.valid_datasets = tuple(valid_datasets)
        self.normalize = normalize
        self.global_mag_completeness = mag_completeness
        self.resample_freq_min = resample_freq_min
        self.freq = str(freq)

        freq_td = pd.Timedelta(self.freq)
        if freq_td <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {freq!r}")

        aliases = {key.lower(): value for key, value in (dataset_aliases or {}).items()}
        for dataset in self.valid_datasets:
            aliases[dataset.lower()] = dataset

        split_cfg = split_groups if split_groups is not None else default_split_groups
        self.split_groups = _normalize_split_groups(split_cfg, self.valid_datasets, aliases)
        self.data_root = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else default_catalogs_dir.expanduser().resolve()
        )

        catalog_cfg = {
            "family": family_name,
            "split_groups": self.split_groups,
            "normalize": normalize,
            "mag_completeness": mag_completeness,
            "resample_freq_min": self.resample_freq_min,
            "freq": self.freq,
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=False)

        selected = self._selected_dataset_order
        self._components = self._load_components(
            selected=selected,
            mag_completeness_map=mag_completeness_map,
        )
        component_freqs = {
            name: int(component.metadata["freq_min"])
            for name, component in self._components.items()
        }
        self._validate_component_freqs(component_freqs)

        self.sequences = [self._components[name].full_sequence for name in selected]
        self.full_sequence = _merge_sequences(self.sequences)
        metadata = self._build_metadata(
            selected=selected,
            component_freqs=component_freqs,
        )
        super().__init__(root_dir=self.root_dir, metadata=metadata)
        self._split_datasets()

    @property
    def _selected_dataset_order(self) -> list[str]:
        return list(self.split_groups["train"] + self.split_groups["val"] + self.split_groups["test"])

    @property
    def _all_selected_datasets(self) -> set[str]:
        return set(self._selected_dataset_order)

    def _resolve_component_mc(
        self,
        *,
        dataset_name: str,
        mag_completeness_map: Optional[Mapping[str, float]],
    ) -> Optional[float]:
        if self.global_mag_completeness is not None:
            return self.global_mag_completeness
        if mag_completeness_map is None:
            return None
        if dataset_name not in mag_completeness_map:
            raise KeyError(f"Missing mag_completeness for grouped component '{dataset_name}'.")
        return mag_completeness_map[dataset_name]

    def _load_components(
        self,
        *,
        selected: Sequence[str],
        mag_completeness_map: Optional[Mapping[str, float]],
    ) -> dict[str, InducedTripletBase]:
        components: dict[str, InducedTripletBase] = {}
        selected_set = set(selected)
        for name in self.valid_datasets:
            if name not in selected_set:
                continue
            component_data_dir = self.data_root / name
            if not component_data_dir.exists():
                raise FileNotFoundError(
                    f"Grouped component dataset dir not found: {component_data_dir}. "
                    f"Expected '{name}' under data root '{self.data_root}'."
                )

            component_root_dir = component_data_dir / "catalogs"
            component_mc = self._resolve_component_mc(
                dataset_name=name,
                mag_completeness_map=mag_completeness_map,
            )
            components[name] = InducedTripletBase(
                dataset_name=name,
                root_dir=component_root_dir,
                data_dir=component_data_dir,
                mag_completeness=component_mc,
                normalize=self.normalize,
                resample_freq_min=self.resample_freq_min,
                freq=self.freq,
            )
        return components

    def _validate_component_freqs(self, component_freqs: Mapping[str, int]) -> None:
        if len(set(component_freqs.values())) != 1:
            raise ValueError(
                "All grouped datasets must use the same resample frequency. "
                f"Got: {dict(component_freqs)}"
            )

    def _build_metadata(
        self,
        *,
        selected: Sequence[str],
        component_freqs: Mapping[str, int],
    ) -> dict:
        component_mc_map = {
            name: float(component.metadata["mag_completeness"])
            for name, component in self._components.items()
        }
        merged_mc = (
            float(self.global_mag_completeness)
            if self.global_mag_completeness is not None
            else float(min(component_mc_map.values()))
        )
        freq_min = next(iter(component_freqs.values()))
        selected_sorted = sorted(set(selected))
        return {
            "name": self.family_name,
            "freq": self.freq,
            "freq_min": freq_min,
            "mag_roundoff_error": 0.01,
            "mag_completeness": merged_mc,
            "start_ts": float(self.full_sequence.t_start),
            "end_ts": float(self.full_sequence.t_end),
            "split_groups": self.split_groups,
            "component_datasets": selected_sorted,
            "component_mag_completeness": component_mc_map,
            "component_inj_fill_policy": {
                name: self._components[name].metadata["inj_fill_policy"]
                for name in selected_sorted
            },
            "component_is_upsample": {
                name: bool(self._components[name].metadata["is_upsample"])
                for name in selected_sorted
            },
        }

    @property
    def required_files(self):
        return ["metadata.pt"]

    def generate_catalog(self):
        return None

    def _split_datasets(self):
        self.train = TppDataset([self._components[name].full_sequence for name in self.split_groups["train"]])
        self.val = TppDataset([self._components[name].full_sequence for name in self.split_groups["val"]])
        self.test = TppDataset([self._components[name].full_sequence for name in self.split_groups["test"]])
