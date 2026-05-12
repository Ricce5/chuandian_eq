from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from src.data import Catalog, Sequence as TppSequence, TppDataset
from src.utils.catalog_pathing import build_hashed_catalog_root, to_serializable_ts
from src.utils.catalog_utils import train_val_test_split_sequence


REQUIRED_SUMMARY_KEYS = (
    "resample_freq_min",
    "mc",
    "inj_fill_policy",
    "is_upsample",
    "start_time_iso",
    "end_time_iso",
)

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "eq_time": (
        "time_iso",
        "ts",
        "time",
        "timestamp",
        "origin_time",
        "origin_timestamp",
        "Date",
    ),
    "eq_magnitude": (
        "magnitude",
        "mag",
        "Magnitude",
        "ML",
        "Mw",
    ),
    "eq_latitude": (
        "latitude",
        "Latitude",
        "lat",
    ),
    "eq_longitude": (
        "longitude",
        "Longitude",
        "lon",
        "lng",
    ),
    "eq_depth": (
        "depth_m",
        "depth",
        "Depth",
        "depth_km",
    ),
    "inj_time": (
        "time_iso",
        "ts",
        "time",
        "timestamp",
        "Date",
    ),
    "inj_rate": (
        "inj_rate_m3_min",
        "inj_rate",
    ),
}

TIME_UNIT_ALIASES: dict[str, str] = {
    "unit": "unit",
    "relative": "unit",
    "model": "unit",
    "freq": "unit",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "m": "m",
    "min": "m",
    "minute": "m",
    "minutes": "m",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
    "abs": "abs",
    "absolute": "abs",
    "datetime": "abs",
    "timestamp": "abs",
    "iso": "abs",
}

_TIME_COLUMN_BY_KIND = {
    "eq": "eq_time",
    "inj": "inj_time",
}


def _coerce_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _resolve_absolute_bounds(summary: Mapping[str, object]) -> tuple[pd.Timestamp, pd.Timestamp]:
    missing = [key for key in ("start_time_iso", "end_time_iso") if key not in summary]
    if missing:
        raise ValueError(
            "Missing required timestamp bounds in summary. "
            f"Expected keys: start_time_iso, end_time_iso. Missing: {missing}."
        )
    start_ts = _coerce_timestamp(summary["start_time_iso"])
    end_ts = _coerce_timestamp(summary["end_time_iso"])
    if end_ts <= start_ts:
        raise ValueError(f"Invalid absolute time bounds: start_ts={start_ts}, end_ts={end_ts}.")
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
    defaults = {
        "train": start_ts,
        "val": start_ts + duration * 0.70,
        "test": start_ts + duration * 0.85,
    }

    def _as_ts(value: Optional[Union[str, pd.Timestamp]], default_ts: pd.Timestamp) -> pd.Timestamp:
        if value is None:
            return default_ts
        return _coerce_timestamp(value)

    train_ts = _as_ts(train_start_ts, defaults["train"])
    val_ts = _as_ts(val_start_ts, defaults["val"])
    test_ts = _as_ts(test_start_ts, defaults["test"])

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
    """Base catalog class for processed induced-seismicity triplet datasets."""

    def __init__(
        self,
        dataset_name: str,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path, None] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
        end_ts: Optional[Union[str, pd.Timestamp]] = None,
        train_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        val_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        test_start_ts: Optional[Union[str, pd.Timestamp]] = None,
    ):
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.freq = str(freq)
        self.unit_td = pd.Timedelta(self.freq)
        if self.unit_td <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {freq!r}")

        self.data_dir = self._resolve_data_dir(root_dir=root_dir, data_dir=data_dir)
        self.summary = self._load_summary()
        self.column_aliases = self._build_column_aliases(self.summary.get("column_aliases"))
        self.time_column_units = self._build_time_column_units(self.summary.get("time_column_units"))
        self.freq_min = self._resolve_freq_min()
        self.resample_freq_td = pd.to_timedelta(self.freq_min, unit="m")
        self.mag_completeness = self._resolve_mag_completeness(mag_completeness)
        self.start_time, self.end_time = self._resolve_time_range(end_ts=end_ts)
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

        self.root_dir = self._build_catalog_root(
            root_dir=root_dir,
            train_start_time=train_start_time,
            val_start_time=val_start_time,
            test_start_time=test_start_time,
        )
        self.eq_file = self.data_dir / "processed" / f"{dataset_name}_eq_processed.csv"
        self.inj_file = self.data_dir / "processed" / f"{dataset_name}_inj_{self.freq_min}min_processed.csv"
        self._validate_source_files()

        metadata = {
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
            "start_t": self.t_start,
            "end_t": self.t_end,
            "train_start_t": train_start_t,
            "val_start_t": val_start_t,
            "test_start_t": test_start_t,
            "inj_fill_policy": self.summary["inj_fill_policy"],
            "is_upsample": bool(self.summary["is_upsample"]),
        }
        super().__init__(root_dir=self.root_dir, metadata=metadata)
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

    def _resolve_data_dir(
        self,
        *,
        root_dir: Union[str, Path],
        data_dir: Union[str, Path, None],
    ) -> Path:
        if data_dir is None:
            data_dir_path = Path(root_dir)
        elif isinstance(data_dir, (str, Path)):
            data_dir_path = Path(data_dir)
        else:
            raise TypeError("data_dir must be a str, Path or None")
        return data_dir_path.expanduser().resolve()

    def _load_summary(self) -> dict:
        summary_path = self.data_dir / "processed" / f"{self.dataset_name}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Summary file not found: {summary_path}\n"
                "Run preprocessing notebook first."
            )
        with open(summary_path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)
        missing_keys = [key for key in REQUIRED_SUMMARY_KEYS if key not in summary]
        if missing_keys:
            raise KeyError(
                f"Missing required summary keys for {self.dataset_name}: {missing_keys}"
            )
        return summary

    def _resolve_freq_min(self) -> int:
        freq_min = int(self.summary["resample_freq_min"])
        if freq_min <= 0:
            raise ValueError(f"Invalid resample_freq_min={freq_min} for {self.dataset_name}.")
        return freq_min

    def _resolve_mag_completeness(self, requested_mc: Optional[float]) -> float:
        if requested_mc is None:
            requested_mc = float(self.summary["mc"])
        return float(requested_mc)

    def _resolve_time_range(
        self,
        *,
        end_ts: Optional[Union[str, pd.Timestamp]],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_time, summary_end_time = _resolve_absolute_bounds(self.summary)
        resolved_end_time = summary_end_time
        if end_ts is not None:
            clipped_end = _clip_ts(_coerce_timestamp(end_ts), start_time, summary_end_time)
            if clipped_end <= start_time:
                raise ValueError(
                    f"Invalid end_ts={clipped_end}. end_ts must be after start_time={start_time}."
                )
            resolved_end_time = clipped_end
        return start_time, resolved_end_time

    def _build_catalog_root(
        self,
        *,
        root_dir: Union[str, Path],
        train_start_time: pd.Timestamp,
        val_start_time: pd.Timestamp,
        test_start_time: pd.Timestamp,
    ) -> Path:
        catalog_cfg = {
            "dataset_name": self.dataset_name,
            "normalize": self.normalize,
            "mag_completeness": self.mag_completeness,
            "freq": self.freq,
            "freq_min": self.freq_min,
            "end_ts": to_serializable_ts(self.end_time),
            "train_start_ts": to_serializable_ts(train_start_time),
            "val_start_ts": to_serializable_ts(val_start_time),
            "test_start_ts": to_serializable_ts(test_start_time),
        }
        return build_hashed_catalog_root(root_dir, catalog_cfg, migrate_legacy=False)

    def _validate_source_files(self) -> None:
        if not self.eq_file.exists():
            raise FileNotFoundError(f"EQ file not found: {self.eq_file}")
        if not self.inj_file.exists():
            raise FileNotFoundError(f"Injection file not found: {self.inj_file}")

    def _build_column_aliases(
        self,
        custom_aliases: Optional[Mapping[str, Sequence[str] | str]],
    ) -> dict[str, tuple[str, ...]]:
        aliases: dict[str, tuple[str, ...]] = {}
        cfg = custom_aliases or {}
        for key, defaults in DEFAULT_COLUMN_ALIASES.items():
            raw = cfg.get(key, defaults)
            if isinstance(raw, str):
                values = (raw,)
            elif isinstance(raw, Sequence):
                values = tuple(str(item) for item in raw)
            else:
                raise TypeError(
                    f"Invalid column alias config for '{key}': expected str or list[str], got {type(raw)}"
                )
            cleaned = tuple(item.strip() for item in values if str(item).strip())
            aliases[key] = cleaned or defaults
        return aliases

    def _build_time_column_units(
        self,
        custom_units: Optional[Mapping[str, str]],
    ) -> dict[str, str]:
        units: dict[str, str] = {}
        for raw_col, raw_unit in (custom_units or {}).items():
            col_name = str(raw_col).strip()
            if not col_name:
                continue
            unit = TIME_UNIT_ALIASES.get(str(raw_unit).strip().lower())
            if unit is None:
                raise ValueError(
                    f"Unsupported time unit '{raw_unit}' for column '{col_name}'. "
                    "Supported: unit/model/freq, s, m, h, d, abs."
                )
            units[col_name] = unit
        return units

    def _resolve_column(self, df: pd.DataFrame, key: str, *, required: bool = True) -> Optional[str]:
        candidates = self.column_aliases.get(key, ())
        for name in candidates:
            if name in df.columns:
                return name
        if required:
            raise KeyError(
                f"Missing required column group '{key}' in dataset {self.dataset_name}. "
                f"Tried aliases: {candidates}. Available columns: {tuple(df.columns)}"
            )
        return None

    def _infer_numeric_time_unit(self, column_name: str) -> str:
        explicit = self.time_column_units.get(column_name)
        if explicit is not None:
            return explicit

        col_l = column_name.lower()
        if col_l == "t" or col_l == "time":
            return "unit"
        if col_l.endswith("_min") or "minute" in col_l or col_l == "min":
            return "m"
        if col_l.endswith("_sec") or "second" in col_l or col_l == "sec":
            return "s"
        if col_l.endswith("_hour") or col_l.endswith("_hr") or col_l == "hour":
            return "h"
        if col_l.endswith("_day") or col_l == "day":
            return "d"
        return "unit"

    def _convert_numeric_relative_time(
        self,
        values: pd.Series,
        *,
        column_name: str,
        forced_unit: Optional[str] = None,
    ) -> pd.Series:
        numeric = pd.to_numeric(values, errors="raise")
        unit = forced_unit or self._infer_numeric_time_unit(column_name)
        if unit in {"unit", "abs"}:
            return numeric.astype(np.float64)
        delta = pd.to_timedelta(numeric.to_numpy(dtype=np.float64), unit=unit)
        return pd.Series(delta / self.unit_td, index=values.index, dtype=np.float64)

    def _relative_time_from_columns(self, df: pd.DataFrame, *, kind: str) -> pd.Series:
        time_key = _TIME_COLUMN_BY_KIND.get(kind)
        if time_key is None:
            raise KeyError(f"Unsupported time kind: {kind}")
        time_col = self._resolve_column(df, time_key, required=True)
        values = df[time_col]
        explicit_unit = self.time_column_units.get(time_col)

        if explicit_unit == "abs":
            ts = pd.to_datetime(values, format="mixed", errors="raise", utc=True).dt.tz_convert(None)
            out = pd.to_numeric((ts - self.start_time) / self.unit_td, errors="raise").astype(np.float64)
        else:
            all_numeric_like = pd.to_numeric(values, errors="coerce").notna().all()
            if pd.api.types.is_numeric_dtype(values) or all_numeric_like:
                out = self._convert_numeric_relative_time(
                    values,
                    column_name=time_col,
                    forced_unit=explicit_unit,
                )
            else:
                ts = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
                if ts.notna().all():
                    ts_naive = ts.dt.tz_convert(None)
                    out = pd.to_numeric((ts_naive - self.start_time) / self.unit_td, errors="raise").astype(np.float64)
                else:
                    out = self._convert_numeric_relative_time(
                        values,
                        column_name=time_col,
                        forced_unit=explicit_unit,
                    )

        if not np.isfinite(out).all():
            bad_rows = np.where(~np.isfinite(out.to_numpy(dtype=np.float64)))[0][:10].tolist()
            raise ValueError(
                f"Found non-finite relative times at rows {bad_rows} for column '{time_col}'."
            )
        return out

    def _build_event_fields(self, df: pd.DataFrame) -> dict:
        out: dict = {}
        lat_col = self._resolve_column(df, "eq_latitude", required=False)
        lon_col = self._resolve_column(df, "eq_longitude", required=False)
        dep_col = self._resolve_column(df, "eq_depth", required=False)

        lat, has_lat = _prepare_numeric_field(df[lat_col]) if lat_col is not None else (None, False)
        lon, has_lon = _prepare_numeric_field(df[lon_col]) if lon_col is not None else (None, False)
        dep, has_dep = _prepare_numeric_field(df[dep_col]) if dep_col is not None else (None, False)

        field_dict = {}
        if has_lat:
            field_dict["latitude"] = lat
        if has_lon:
            field_dict["longitude"] = lon
        if has_dep:
            field_dict["depth"] = dep
        if not field_dict:
            return out

        if self.normalize:
            norm = self.normalize_fields(field_dict)
        else:
            norm = {key: torch.tensor(val, dtype=torch.float32) for key, val in field_dict.items()}

        if "latitude" in norm and "longitude" in norm:
            out["loc"] = torch.stack([norm["latitude"], norm["longitude"]], dim=-1)
        if "depth" in norm:
            out["depth"] = norm["depth"]
        return out

    def _prepare_eq_dataframe(self) -> pd.DataFrame:
        df_eq = pd.read_csv(self.eq_file)
        mag_col = self._resolve_column(df_eq, "eq_magnitude", required=True)
        df_eq["magnitude"] = pd.to_numeric(df_eq[mag_col], errors="raise")
        df_eq["t"] = self._relative_time_from_columns(df_eq, kind="eq")
        df_eq["t"] = pd.to_numeric(df_eq["t"], errors="raise").round(9)

        invalid_t_mask = ~np.isfinite(df_eq["t"])
        invalid_mag_mask = ~np.isfinite(df_eq["magnitude"])
        if invalid_t_mask.any() or invalid_mag_mask.any():
            raise ValueError(
                f"Found invalid EQ rows: invalid_t={int(invalid_t_mask.sum())}, "
                f"invalid_magnitude={int(invalid_mag_mask.sum())}."
            )

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
        return df_eq

    def _prepare_inj_dataframe(self) -> pd.DataFrame:
        df_ts = pd.read_csv(self.inj_file)
        inj_rate_col = self._resolve_column(df_ts, "inj_rate", required=True)
        df_ts["t"] = self._relative_time_from_columns(df_ts, kind="inj")
        df_ts["t"] = pd.to_numeric(df_ts["t"], errors="raise").round(9)
        df_ts["inj_rate_m3_min"] = pd.to_numeric(df_ts[inj_rate_col], errors="raise")

        invalid_t_mask = ~np.isfinite(df_ts["t"])
        invalid_rate_mask = ~np.isfinite(df_ts["inj_rate_m3_min"])
        if invalid_t_mask.any() or invalid_rate_mask.any():
            raise ValueError(
                f"Found invalid injection rows: invalid_t={int(invalid_t_mask.sum())}, "
                f"invalid_rate={int(invalid_rate_mask.sum())}."
            )

        df_ts["rate"] = df_ts["inj_rate_m3_min"]
        df_ts = df_ts[(df_ts["t"] >= self.t_start) & (df_ts["t"] <= self.t_end + 1e-9)].copy()
        if df_ts.empty:
            raise ValueError(
                f"No injection rows left within [{self.t_start}, {self.t_end}] after filtering."
            )

        df_ts = df_ts.groupby("t", as_index=False)["rate"].mean()
        df_ts.sort_values("t", inplace=True)
        if len(df_ts) < 2:
            raise ValueError(
                "Injection time series must contain at least 2 timestamps after filtering. "
                f"Got {len(df_ts)}."
            )
        return df_ts

    def generate_catalog(self):
        df_eq = self._prepare_eq_dataframe()
        arrival_times = df_eq["t"].to_numpy(dtype=np.float64)
        inter_times = np.diff(arrival_times, prepend=[self.t_start], append=[self.t_end])

        seq_kwargs = {
            "inter_times": torch.tensor(inter_times, dtype=torch.float32),
            "t_start": self.t_start,
            "mag": torch.tensor(df_eq["magnitude"].to_numpy(dtype=np.float64), dtype=torch.float32),
        }
        seq_kwargs.update(self._build_event_fields(df_eq))

        df_ts = self._prepare_inj_dataframe()
        seq_kwargs["time_series"] = torch.tensor(
            df_ts[["rate"]].to_numpy(dtype=np.float64),
            dtype=torch.float32,
        )
        seq_kwargs["time_series_times"] = torch.tensor(
            df_ts["t"].to_numpy(dtype=np.float64),
            dtype=torch.float32,
        )

        seq = TppSequence(**seq_kwargs)
        TppDataset([seq]).save_to_disk(self.root_dir / "full_sequence.pt")
