from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.data import Catalog, Sequence, TppDataset
from src.utils.catalog_utils import train_val_test_split_sequence
from src.utils.catalog_pathing import build_hashed_catalog_root, to_serializable_ts


REQUIRED_SUMMARY_KEYS = (
    "resample_freq_min",
    "mc",
    "inj_fill_policy",
    "is_upsample",
    "start_time_iso",
    "end_time_iso",
)


def _coerce_timestamp(value: object) -> pd.Timestamp:
    """Coerce a value to a timezone-naive pd.Timestamp in UTC."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if ts.tz is not None: # tz: timezone-aware, convert to UTC and remove tz info
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _resolve_absolute_bounds(summary: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Extract and validate absolute time bounds from summary dict."""
    missing = [k for k in ("start_time_iso", "end_time_iso") if k not in summary]
    if missing:
        raise ValueError(
            "Missing required timestamp bounds in summary. "
            f"Expected keys: start_time_iso, end_time_iso. Missing: {missing}."
        )
    start_iso = summary["start_time_iso"]
    end_iso = summary["end_time_iso"]

    start_ts = _coerce_timestamp(start_iso)
    end_ts = _coerce_timestamp(end_iso)

    if end_ts <= start_ts:
        raise ValueError(
            f"Invalid absolute time bounds: start_ts={start_ts}, end_ts={end_ts}."
        )

    return start_ts, end_ts


def _clip_ts(ts: pd.Timestamp, low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
    if ts < low:
        return low
    if ts > high:
        return high
    return ts


def _resolve_split_timestamps(
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    train_start_ts: Optional[Union[str, pd.Timestamp]] = None,
    val_start_ts: Optional[Union[str, pd.Timestamp]] = None,
    test_start_ts: Optional[Union[str, pd.Timestamp]] = None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    duration = end_ts - start_ts

    default_train_ts = start_ts
    default_val_ts = start_ts + duration * 0.70
    default_test_ts = start_ts + duration * 0.85

    def _as_ts(
        value: Optional[Union[str, pd.Timestamp]],
        default_ts: pd.Timestamp,
    ) -> pd.Timestamp:
        if value is None:
            return default_ts

        return _coerce_timestamp(value)

    train_ts = _as_ts(train_start_ts, default_train_ts)
    val_ts = _as_ts(val_start_ts, default_val_ts)
    test_ts = _as_ts(test_start_ts, default_test_ts)

    train_ts = _clip_ts(train_ts, start_ts, end_ts)
    val_ts = _clip_ts(val_ts, train_ts, end_ts)
    test_ts = _clip_ts(test_ts, val_ts, end_ts)
    return train_ts, val_ts, test_ts


def _prepare_numeric_field(values: pd.Series) -> tuple[Optional[np.ndarray], bool]:
    arr = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
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
        freq: str = "1h",
        train_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        val_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        test_start_ts: Optional[Union[str, pd.Timestamp]] = None,
    ):
        self.dataset_name = dataset_name
        self.normalize = normalize

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
        missing_summary_keys = [k for k in REQUIRED_SUMMARY_KEYS if k not in self.summary]
        if missing_summary_keys:
            raise KeyError(
                f"Missing required summary keys for {dataset_name}: {missing_summary_keys}"
            )

        self.freq_min = int(self.summary["resample_freq_min"])
        if self.freq_min <= 0:
            raise ValueError(f"Invalid resample_freq_min={self.freq_min} for {dataset_name}.")
        self.resample_freq_td = pd.to_timedelta(self.freq_min, unit="m")
        self.freq = str(freq)
        self.unit_td = pd.Timedelta(self.freq)
        if self.unit_td <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {freq!r}")

        if mag_completeness is None:
            mag_completeness = float(self.summary["mc"])
        self.mag_completeness = float(mag_completeness)

        self.start_time = None
        self.end_time = None
        self.start_time, self.end_time = _resolve_absolute_bounds(self.summary)
        self.t_start = 0.0
        self.t_end = float((self.end_time - self.start_time) / self.unit_td)

        if not np.isfinite(self.t_end) or self.t_end <= 0:
            raise ValueError(
                f"Invalid t_end={self.t_end} for dataset {dataset_name}. "
                f"start_time={self.start_time}, end_time={self.end_time}, freq={self.unit_td}"
            )

        train_start_time, val_start_time, test_start_time = _resolve_split_timestamps(
            start_ts=self.start_time,
            end_ts=self.end_time,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
        train_start_t = float(np.clip((train_start_time - self.start_time) / self.unit_td, 0.0, self.t_end))
        val_start_t = float(np.clip((val_start_time - self.start_time) / self.unit_td, train_start_t, self.t_end))
        test_start_t = float(np.clip((test_start_time - self.start_time) / self.unit_td, val_start_t, self.t_end))

        catalog_cfg: Dict[str, object] = {
            "dataset_name": dataset_name,
            "normalize": normalize,
            "mag_completeness": self.mag_completeness,
            "freq": self.freq,
            "freq_min": self.freq_min,
            "train_start_ts": to_serializable_ts(train_start_time),
            "val_start_ts": to_serializable_ts(val_start_time),
            "test_start_ts": to_serializable_ts(test_start_time),
        }
        self.root_dir = build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=False)

        self.eq_file = self.data_dir / "processed" / f"{dataset_name}_eq_processed.csv"
        self.inj_file = self.data_dir / "processed" / f"{dataset_name}_inj_{self.freq_min}min_processed.csv"
        if not self.eq_file.exists():
            raise FileNotFoundError(f"EQ file not found: {self.eq_file}")
        if not self.inj_file.exists():
            raise FileNotFoundError(f"Injection file not found: {self.inj_file}")

        self.metadata = {
            "name": dataset_name,
            "freq": self.freq,
            "freq_min": self.freq_min,
            "mag_roundoff_error": 0.01,
            "mag_completeness": self.mag_completeness,
            "start_ts": self.start_time,
            "end_ts": self.end_time,
            "train_start_ts": train_start_time,
            "val_start_ts": val_start_time,
            "test_start_ts": test_start_time,
            "start_t": float(self.t_start),
            "end_t": float(self.t_end),
            "train_start_t": train_start_t,
            "val_start_t": val_start_t,
            "test_start_t": test_start_t,
            "inj_fill_policy": self.summary["inj_fill_policy"],
            "is_upsample": bool(self.summary["is_upsample"]),
        }

        super().__init__(root_dir=self.root_dir, metadata=self.metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]
        self._split_datasets()

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def _split_datasets(self):
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
            freq=self.unit_td,
        )
        self.train = TppDataset([seq_train])
        self.val = TppDataset([seq_val])
        self.test = TppDataset([seq_test])

    def _relative_time_from_columns(self, df: pd.DataFrame) -> pd.Series:
        if "time_iso" not in df.columns:
            raise KeyError(
                f"Missing 'time_iso' in {self.dataset_name} processed file. "
                "Timestamp-based processing requires time_iso."
            )
        ts = pd.to_datetime(df["time_iso"], format="mixed", errors="raise", utc=True).dt.tz_convert(None)
        t = (ts - self.start_time) / self.unit_td
        out = pd.to_numeric(t, errors="raise").astype(np.float64)
        if not np.isfinite(out).all():
            bad = np.where(~np.isfinite(out.to_numpy(dtype=np.float64)))[0][:10].tolist()
            raise ValueError(f"Found non-finite relative times at rows {bad}.")
        return out

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
        required_eq_cols = ("time_iso", "magnitude")
        missing_eq_cols = [c for c in required_eq_cols if c not in df_eq.columns]
        if missing_eq_cols:
            raise KeyError(f"Missing required EQ columns in {self.eq_file}: {missing_eq_cols}")

        df_eq["magnitude"] = pd.to_numeric(df_eq["magnitude"], errors="raise")
        df_eq["t"] = self._relative_time_from_columns(df_eq)
        df_eq["t"] = pd.to_numeric(df_eq["t"], errors="raise").round(9)

        invalid_t_mask = ~np.isfinite(df_eq["t"])
        invalid_mag_mask = ~np.isfinite(df_eq["magnitude"])
        if invalid_t_mask.any() or invalid_mag_mask.any():
            raise ValueError(
                f"Found invalid EQ rows: invalid_t={int(invalid_t_mask.sum())}, "
                f"invalid_magnitude={int(invalid_mag_mask.sum())}."
            )

        # Enforce completeness threshold.
        df_eq = df_eq[df_eq["magnitude"] > self.mag_completeness].copy()
        df_eq.sort_values("t", inplace=True)
        df_eq = df_eq[(df_eq["t"] >= self.t_start) & (df_eq["t"] <= self.t_end + 1e-9)].copy()

        duplicated_mask = df_eq["t"].duplicated(keep=False)
        if duplicated_mask.any():
            df_eq.loc[duplicated_mask, "t"] += np.random.uniform(1e-9, 1e-7, duplicated_mask.sum())
            df_eq.sort_values("t", inplace=True)

        df_eq["t_diff"] = df_eq["t"].diff()
        non_increasing_mask = df_eq["t_diff"] <= 0
        non_increasing_mask = non_increasing_mask.fillna(False)
        if non_increasing_mask.any():
            raise ValueError(
                "EQ arrival times are not strictly increasing after duplicate-time jittering. "
                f"invalid_rows={int(non_increasing_mask.sum())}."
            )

        arrival_times = df_eq["t"].to_numpy(dtype=np.float64)
        inter_times = np.diff(arrival_times, prepend=[self.t_start], append=[self.t_end])

        seq_kwargs = {
            "inter_times": torch.tensor(inter_times, dtype=torch.float32),
            "t_start": self.t_start,
            "mag": torch.tensor(df_eq["magnitude"].to_numpy(dtype=np.float64), dtype=torch.float32),
        }
        seq_kwargs.update(self._build_event_fields(df_eq))

        df_ts = pd.read_csv(self.inj_file)
        required_inj_cols = ("time_iso", "inj_rate_m3_min")
        missing_inj_cols = [c for c in required_inj_cols if c not in df_ts.columns]
        if missing_inj_cols:
            raise KeyError(
                f"Missing required injection columns in {self.inj_file}: {missing_inj_cols}"
            )
        df_ts["t"] = self._relative_time_from_columns(df_ts)
        df_ts["t"] = pd.to_numeric(df_ts["t"], errors="raise").round(9)
        df_ts["inj_rate_m3_min"] = pd.to_numeric(df_ts["inj_rate_m3_min"], errors="raise")
        invalid_ts_t_mask = ~np.isfinite(df_ts["t"])
        invalid_ts_rate_mask = ~np.isfinite(df_ts["inj_rate_m3_min"])
        if invalid_ts_t_mask.any() or invalid_ts_rate_mask.any():
            raise ValueError(
                f"Found invalid injection rows: invalid_t={int(invalid_ts_t_mask.sum())}, "
                f"invalid_rate={int(invalid_ts_rate_mask.sum())}."
            )
        df_ts["rate"] = df_ts["inj_rate_m3_min"]

        out_of_range_ts_mask = (df_ts["t"] < self.t_start) | (df_ts["t"] > self.t_end + 1e-9)
        if out_of_range_ts_mask.any():
            raise ValueError(
                f"Found injection rows outside [{self.t_start}, {self.t_end}]: "
                f"count={int(out_of_range_ts_mask.sum())}."
            )
        df_ts = df_ts.groupby("t", as_index=False)["rate"].mean()
        df_ts.sort_values("t", inplace=True)

        seq_kwargs["time_series"] = torch.tensor(df_ts[["rate"]].to_numpy(dtype=np.float64), dtype=torch.float32)
        seq_kwargs["time_series_times"] = torch.tensor(df_ts["t"].to_numpy(dtype=np.float64), dtype=torch.float32)

        seq = Sequence(**seq_kwargs)
        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")
