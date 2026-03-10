from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from src.data import Catalog, Sequence as TppSequence, TppDataset, default_catalogs_dir
from src.utils.file_utils import build_catalog_root_dir

from .induced_triplet_base import InducedTripletBase

SplitGroupValue = Union[str, Sequence[str], Path]
SplitGroups = Mapping[str, SplitGroupValue]


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
    raise ValueError(
        f"Unsupported dataset token '{token}'. "
        f"Valid datasets: {tuple(valid_datasets)}"
    )


def _normalize_split_groups(
    split_groups: SplitGroups,
    valid_datasets: Sequence[str],
    aliases: Mapping[str, str],
) -> Dict[str, list[str]]:
    required = ("train", "val", "test")
    missing = [k for k in required if k not in split_groups]
    if missing:
        raise ValueError(
            f"split_groups must contain keys {required}. Missing: {missing}."
        )

    out: Dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in required:
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
    return {k for k in seq.keys() if k not in TppSequence.default_sequence_attrs}


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
    arrival_chunks: list[torch.Tensor] = []
    attr_chunks: dict[str, list[torch.Tensor]] = {k: [] for k in attr_keys}
    ts_chunks: list[torch.Tensor] = []
    ts_time_chunks: list[torch.Tensor] = []

    for seq in sequences:
        duration = float(seq.t_end - seq.t_start)
        shifted_arrivals = seq.arrival_times - float(seq.t_start) + offset
        arrival_chunks.append(shifted_arrivals)

        for key in attr_keys:
            attr_chunks[key].append(seq[key])

        if has_time_series:
            shifted_ts_times = seq.time_series_times - float(seq.t_start) + offset
            ts_chunks.append(seq.time_series)
            ts_time_chunks.append(shifted_ts_times)

        offset += duration

    if arrival_chunks and sum(chunk.shape[0] for chunk in arrival_chunks) > 0:
        merged_arrivals = torch.cat(arrival_chunks, dim=0).detach().cpu().numpy().astype(np.float64)
    else:
        merged_arrivals = np.array([], dtype=np.float64)

    merged_inter_times = np.diff(merged_arrivals, prepend=[0.0], append=[offset])
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
    """Compose train/val/test from multiple InducedTripletBase datasets."""

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
        use_clean_injection: bool = True,
    ):
        self.family_name = family_name
        self.valid_datasets = tuple(valid_datasets)
        self.normalize = normalize
        self.use_clean_injection = use_clean_injection
        self.global_mag_completeness = mag_completeness

        aliases = {k.lower(): v for k, v in (dataset_aliases or {}).items()}
        for name in self.valid_datasets:
            aliases[name.lower()] = name

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
            "use_clean_injection": use_clean_injection,
        }
        sub_root_dir, _ = build_catalog_root_dir(root_dir, catalog_cfg)
        self.root_dir = Path(sub_root_dir).expanduser().resolve()

        self._components: dict[str, InducedTripletBase] = {}
        for name in self.valid_datasets:
            if name not in self._all_selected_datasets:
                continue
            component_data_dir = self.data_root / name
            if not component_data_dir.exists():
                raise FileNotFoundError(
                    f"Grouped component dataset dir not found: {component_data_dir}. "
                    f"Expected '{name}' under data root '{self.data_root}'."
                )

            if self.global_mag_completeness is None and mag_completeness_map is not None:
                component_mc = mag_completeness_map.get(name)
            else:
                component_mc = self.global_mag_completeness

            component_root_dir = component_data_dir / "catalogs"
            self._components[name] = InducedTripletBase(
                dataset_name=name,
                root_dir=component_root_dir,
                data_dir=component_data_dir,
                mag_completeness=component_mc,
                normalize=normalize,
                use_clean_injection=use_clean_injection,
            )

        component_freqs = {
            name: int(catalog.metadata["freq_min"])
            for name, catalog in self._components.items()
        }
        if len(set(component_freqs.values())) != 1:
            raise ValueError(
                "All grouped datasets must use the same resample frequency. "
                f"Got: {component_freqs}"
            )

        merged_sequences = [
            self._components[name].full_sequence
            for name in (
                self.split_groups["train"] + self.split_groups["val"] + self.split_groups["test"]
            )
        ]
        self.full_sequence = _merge_sequences(merged_sequences)

        component_mc_map = {
            name: float(catalog.metadata["mag_completeness"])
            for name, catalog in self._components.items()
        }
        if self.global_mag_completeness is not None:
            merged_mc = float(self.global_mag_completeness)
        else:
            merged_mc = float(min(component_mc_map.values()))

        freq_min = next(iter(component_freqs.values()))
        selected_sorted = sorted(self._all_selected_datasets)
        metadata = {
            "name": family_name,
            "freq": f"{freq_min}min",
            "freq_min": freq_min,
            "mag_roundoff_error": 0.01,
            "mag_completeness": merged_mc,
            "start_ts": float(self.full_sequence.t_start),
            "end_ts": float(self.full_sequence.t_end),
            "split_groups": self.split_groups,
            "component_datasets": selected_sorted,
            "component_mag_completeness": component_mc_map,
            "component_inj_fill_policy": {
                name: self._components[name].metadata.get("inj_fill_policy", "unknown")
                for name in selected_sorted
            },
            "component_is_upsample": {
                name: bool(self._components[name].metadata.get("is_upsample", False))
                for name in selected_sorted
            },
        }
        super().__init__(root_dir=self.root_dir, metadata=metadata)
        self._split_datasets()

    @property
    def _all_selected_datasets(self) -> set[str]:
        return set(self.split_groups["train"] + self.split_groups["val"] + self.split_groups["test"])

    @property
    def required_files(self):
        return ["metadata.pt"]

    def generate_catalog(self):
        # This catalog composes already-generated component catalogs.
        return None

    def _split_datasets(self):
        self.train = TppDataset([self._components[name].full_sequence for name in self.split_groups["train"]])
        self.val = TppDataset([self._components[name].full_sequence for name in self.split_groups["val"]])
        self.test = TppDataset([self._components[name].full_sequence for name in self.split_groups["test"]])
