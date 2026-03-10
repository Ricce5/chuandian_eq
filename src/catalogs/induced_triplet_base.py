from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.data import Catalog, Sequence, TppDataset
from src.utils.catalog_utils import train_val_test_split_sequence_float
from src.utils.file_utils import build_catalog_root_dir


def _to_serializable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _clip_and_sort_split_points(
    t_end: float,
    train_start_ts: Optional[float],
    val_start_ts: Optional[float],
    test_start_ts: Optional[float],
) -> tuple[float, float, float]:
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError(f"Invalid t_end={t_end}.")

    if train_start_ts is None:
        train_start_ts = 0.0
    if val_start_ts is None:
        val_start_ts = t_end * 0.70
    if test_start_ts is None:
        test_start_ts = t_end * 0.85

    train_start_ts = float(np.clip(train_start_ts, 0.0, t_end))
    val_start_ts = float(np.clip(val_start_ts, train_start_ts, t_end))
    test_start_ts = float(np.clip(test_start_ts, val_start_ts, t_end))
    return train_start_ts, val_start_ts, test_start_ts


def _prepare_numeric_field(values: pd.Series) -> tuple[Optional[np.ndarray], bool]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return None, False

    fill_value = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill_value)
    return arr, True


class InducedTripletBase(Catalog):
    """Base catalog class for processed induced-seismicity triplet datasets.

    Expected files under ``data_dir/processed``:
    - ``{dataset_name}_eq_processed.csv``
    - ``{dataset_name}_inj_{resample_freq_min}min_processed.csv``
    - ``{dataset_name}_summary.json``
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        train_start_ts: Optional[float] = None,
        val_start_ts: Optional[float] = None,
        test_start_ts: Optional[float] = None,
        use_clean_injection: bool = True,
    ):
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.use_clean_injection = use_clean_injection

        if data_dir is None:
            data_dir_path = Path(root_dir)
        elif isinstance(data_dir, (str, Path)):
            data_dir_path = Path(data_dir)
        else:
            raise TypeError("data_dir must be a str, Path or None")
        self.data_dir = data_dir_path.expanduser().resolve()

        summary_path = self.data_dir / "processed" / f"{dataset_name}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Summary file not found: {summary_path}\n"
                f"Run preprocessing notebook first."
            )
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        self.freq_min = int(self.summary.get("resample_freq_min", 1))
        if self.freq_min <= 0:
            raise ValueError(f"Invalid resample_freq_min={self.freq_min} for {dataset_name}.")

        if mag_completeness is None:                                                           
            mag_completeness = float(self.summary.get("mc", 0.0))
        self.mag_completeness = float(mag_completeness)

        start_min = float(self.summary.get("start_min", 0.0))
        end_min = float(self.summary.get("end_min", 0.0))
        if not np.isfinite(end_min) or end_min <= start_min:
            raise ValueError(
                f"Invalid [start_min, end_min]=[{start_min}, {end_min}] for {dataset_name}."
            )

        self.start_min = start_min
        self.end_min = end_min
        self.t_start = 0.0
        self.t_end = (self.end_min - self.start_min) / float(self.freq_min)

        train_start_ts, val_start_ts, test_start_ts = _clip_and_sort_split_points(
            self.t_end,
            train_start_ts,
            val_start_ts,
            test_start_ts,
        )

        catalog_cfg: Dict[str, object] = {
            "dataset_name": dataset_name,
            "normalize": normalize,
            "mag_completeness": self.mag_completeness,
            "freq_min": self.freq_min,
            "train_start_ts": _to_serializable(train_start_ts),
            "val_start_ts": _to_serializable(val_start_ts),
            "test_start_ts": _to_serializable(test_start_ts),
            "use_clean_injection": use_clean_injection,
        }
        sub_root_dir, _ = build_catalog_root_dir(root_dir, catalog_cfg)

        self.root_dir = Path(sub_root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.eq_file = self.data_dir / "processed" / f"{dataset_name}_eq_processed.csv"
        self.inj_file = self.data_dir / "processed" / f"{dataset_name}_inj_{self.freq_min}min_processed.csv"
        if not self.eq_file.exists():
            raise FileNotFoundError(f"EQ file not found: {self.eq_file}")
        if not self.inj_file.exists():
            raise FileNotFoundError(f"Injection file not found: {self.inj_file}")

        self.metadata = {
            "name": dataset_name,
            "freq": f"{self.freq_min}min",
            "freq_min": self.freq_min,
            "mag_roundoff_error": 0.01,
            "mag_completeness": self.mag_completeness,
            "start_ts": float(self.t_start),
            "end_ts": float(self.t_end),
            "start_min": float(self.start_min),
            "end_min": float(self.end_min),
            "train_start_ts": float(train_start_ts),
            "val_start_ts": float(val_start_ts),
            "test_start_ts": float(test_start_ts),
            "inj_fill_policy": self.summary.get("inj_fill_policy", "unknown"),
            "is_upsample": bool(self.summary.get("is_upsample", False)),
        }

        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]
        self._split_datasets()

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def _split_datasets(self):
        seq_train, seq_val, seq_test = train_val_test_split_sequence_float(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])

    def _build_event_fields(self, df: pd.DataFrame) -> dict:
        out: dict = {}

        lat, has_lat = _prepare_numeric_field(df["latitude"]) if "latitude" in df.columns else (None, False)
        lon, has_lon = _prepare_numeric_field(df["longitude"]) if "longitude" in df.columns else (None, False)
        dep, has_dep = _prepare_numeric_field(df["depth_m"]) if "depth_m" in df.columns else (None, False)

        field_dict = {}
        if has_lat:
            field_dict["latitude"] = lat
        if has_lon:
            field_dict["longitude"] = lon
        if has_dep:
            field_dict["depth"] = dep

        if field_dict:
            if self.normalize:
                norm = self.normalize_fields(field_dict)
            else:
                norm = {k: torch.tensor(v, dtype=torch.float32) for k, v in field_dict.items()}

            if "latitude" in norm and "longitude" in norm:
                out["loc"] = torch.stack([norm["latitude"], norm["longitude"]], dim=-1)
            if "depth" in norm:
                out["depth"] = norm["depth"]

        return out

    def generate_catalog(self):
        df_eq = pd.read_csv(self.eq_file)
        df_eq["time_min"] = pd.to_numeric(df_eq["time_min"], errors="coerce")
        df_eq["magnitude"] = pd.to_numeric(df_eq["magnitude"], errors="coerce")

        # Keep valid events and enforce completeness threshold.
        df_eq = df_eq[np.isfinite(df_eq["time_min"]) & np.isfinite(df_eq["magnitude"])].copy()
        df_eq = df_eq[df_eq["magnitude"] > self.mag_completeness].copy()
        df_eq.sort_values("time_min", inplace=True)
        df_eq["t"] = (df_eq["time_min"] - self.start_min) / float(self.freq_min)
        df_eq = df_eq[(df_eq["t"] >= self.t_start) & (df_eq["t"] <= self.t_end + 1e-9)].copy()

        duplicated_mask = df_eq["t"].duplicated(keep=False)
        if duplicated_mask.any():
            df_eq.loc[duplicated_mask, "t"] += np.random.uniform(1e-9, 1e-7, duplicated_mask.sum())
            df_eq.sort_values("t", inplace=True)

        df_eq["t_diff"] = df_eq["t"].diff()
        df_eq = df_eq[df_eq["t_diff"] > 0].copy()

        arrival_times = df_eq["t"].to_numpy(dtype=np.float64)
        inter_times = np.diff(arrival_times, prepend=[self.t_start], append=[self.t_end])

        seq_kwargs = {
            "inter_times": torch.tensor(inter_times, dtype=torch.float32),
            "t_start": self.t_start,
            "mag": torch.tensor(df_eq["magnitude"].to_numpy(dtype=np.float64), dtype=torch.float32),
        }
        seq_kwargs.update(self._build_event_fields(df_eq))

        df_ts = pd.read_csv(self.inj_file)
        df_ts["time_min"] = pd.to_numeric(df_ts["time_min"], errors="coerce")
        if self.use_clean_injection and "inj_rate_clean_m3_min" in df_ts.columns:
            rate_col = "inj_rate_clean_m3_min"
        elif "inj_rate_m3_min" in df_ts.columns:
            rate_col = "inj_rate_m3_min"
        elif "IR_h" in df_ts.columns:
            rate_col = "IR_h"
        else:
            raise KeyError(
                f"No supported injection-rate column found in {self.inj_file}. "
                "Expected one of: inj_rate_clean_m3_min, inj_rate_m3_min, IR_h."
            )

        df_ts[rate_col] = pd.to_numeric(df_ts[rate_col], errors="coerce")
        df_ts = df_ts[np.isfinite(df_ts["time_min"])].copy()
        df_ts["rate"] = df_ts[rate_col].fillna(0.0)
        if self.use_clean_injection:
            df_ts["rate"] = df_ts["rate"].clip(lower=0.0)

        df_ts["t"] = (df_ts["time_min"] - self.start_min) / float(self.freq_min)
        df_ts = df_ts[(df_ts["t"] >= self.t_start) & (df_ts["t"] <= self.t_end + 1e-9)].copy()
        df_ts = df_ts.groupby("t", as_index=False)["rate"].mean()
        df_ts.sort_values("t", inplace=True)

        seq_kwargs["time_series"] = torch.tensor(df_ts[["rate"]].to_numpy(dtype=np.float64), dtype=torch.float32)
        seq_kwargs["time_series_times"] = torch.tensor(df_ts["t"].to_numpy(dtype=np.float64), dtype=torch.float32)

        seq = Sequence(**seq_kwargs)
        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")
