#!/usr/bin/env python3
"""Build processed triplet files for the Geysers induced-seismicity dataset.

Input (raw):
  data/Geysers/raw/geysers_catalog.csv
  data/Geysers/raw/daily_injection_production.csv

Output (processed):
  data/Geysers/processed/Geysers_eq_processed.csv
  data/Geysers/processed/Geysers_inj_1440min_processed.csv
  data/Geysers/processed/Geysers_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _build_eq_processed(eq_raw_path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    eq = pd.read_csv(eq_raw_path)
    required_cols = ("ts", "Magnitude", "Latitude", "Longitude", "Depth")
    missing = [column for column in required_cols if column not in eq.columns]
    if missing:
        raise KeyError(f"Missing required EQ columns in {eq_raw_path}: {missing}")

    eq["ts"] = pd.to_datetime(eq["ts"], errors="raise")
    eq = eq.sort_values("ts").reset_index(drop=True)
    duplicated = eq["ts"].duplicated(keep=False)
    if duplicated.any():
        eq.loc[duplicated, "ts"] = eq.loc[duplicated, "ts"] + pd.to_timedelta(
            np.random.uniform(1e-8, 1e-6, duplicated.sum()),
            unit="D",
        )
        eq = eq.sort_values("ts").reset_index(drop=True)

    start_ts = eq["ts"].min()
    eq_out = pd.DataFrame(
        {
            "time_iso": eq["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip("."),
            "magnitude": pd.to_numeric(eq["Magnitude"], errors="raise"),
            "latitude": pd.to_numeric(eq["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(eq["Longitude"], errors="coerce"),
            "depth_m": pd.to_numeric(eq["Depth"], errors="coerce"),
        }
    )
    return eq_out, start_ts


def _build_inj_processed(
    inj_raw_path: Path,
    *,
    anchor_start_ts: pd.Timestamp,
    inj_end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    inj = pd.read_csv(inj_raw_path)
    required_cols = ("Date", "InjVol_daily")
    missing = [column for column in required_cols if column not in inj.columns]
    if missing:
        raise KeyError(f"Missing required INJ columns in {inj_raw_path}: {missing}")

    inj["Date"] = pd.to_datetime(inj["Date"], errors="raise")
    inj = inj.sort_values("Date").reset_index(drop=True)
    end_ts = inj["Date"].max()
    clipped_end = min(end_ts, inj_end_ts)
    inj = inj[(inj["Date"] >= anchor_start_ts) & (inj["Date"] <= clipped_end)].copy()

    inj_out = pd.DataFrame(
        {
            "time_min": ((inj["Date"] - anchor_start_ts) / pd.Timedelta("1min")).astype(float),
            "time_iso": inj["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "inj_rate_m3_min": pd.to_numeric(inj["InjVol_daily"], errors="raise"),
            "pressure_mpa": 0.0,
            "eq_count": 0,
            "eq_cum_count": 0,
        }
    )
    return inj_out, clipped_end


def _build_summary(*, dataset: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, mc: float) -> dict:
    return {
        "dataset": dataset,
        "resample_freq_min": 1440,
        "resample_rule": "1D",
        "native_step_min": 1440.0,
        "mc": float(mc),
        "inj_fill_policy": "as_is_daily",
        "is_upsample": False,
        "start_time_iso": start_ts.isoformat(),
        "end_time_iso": end_ts.isoformat(),
    }


def _write_outputs(
    *,
    output_dir: Path,
    dataset: str,
    eq_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    summary: dict,
    overwrite: bool,
) -> None:
    eq_path = output_dir / f"{dataset}_eq_processed.csv"
    inj_path = output_dir / f"{dataset}_inj_1440min_processed.csv"
    summary_path = output_dir / f"{dataset}_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        for path in (eq_path, inj_path, summary_path):
            if path.exists():
                raise FileExistsError(
                    f"Output already exists: {path}. "
                    "Use --overwrite to replace existing files."
                )

    eq_df.to_csv(eq_path, index=False)
    inj_df.to_csv(inj_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote EQ processed: {eq_path}")
    print(f"[OK] Wrote INJ processed: {inj_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    print(f"[INFO] EQ rows: {len(eq_df)}, INJ rows: {len(inj_df)}")
    print(f"[INFO] Time range: {summary['start_time_iso']} -> {summary['end_time_iso']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Geysers processed triplet files.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory (default: data)")
    parser.add_argument("--dataset", type=str, default="Geysers", help="Dataset name (default: Geysers)")
    parser.add_argument("--mc", type=float, default=2.2, help="Magnitude completeness in summary (default: 2.2)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.data_root / args.dataset
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"

    eq_raw_path = raw_dir / "geysers_catalog.csv"
    inj_raw_path = raw_dir / "daily_injection_production.csv"
    if not eq_raw_path.exists():
        raise FileNotFoundError(f"Missing raw EQ file: {eq_raw_path}")
    if not inj_raw_path.exists():
        raise FileNotFoundError(f"Missing raw INJ file: {inj_raw_path}")

    eq_df, eq_start_ts = _build_eq_processed(eq_raw_path)
    inj_dates = pd.read_csv(inj_raw_path, usecols=["Date"])
    inj_dates["Date"] = pd.to_datetime(inj_dates["Date"], errors="raise")
    inj_start_ts = inj_dates["Date"].min()
    inj_end_ts = inj_dates["Date"].max()
    anchor_start_ts = min(eq_start_ts.floor("D"), inj_start_ts)

    inj_df, end_ts = _build_inj_processed(
        inj_raw_path,
        anchor_start_ts=anchor_start_ts,
        inj_end_ts=inj_end_ts,
    )
    summary = _build_summary(
        dataset=args.dataset,
        start_ts=anchor_start_ts,
        end_ts=end_ts,
        mc=args.mc,
    )
    _write_outputs(
        output_dir=processed_dir,
        dataset=args.dataset,
        eq_df=eq_df,
        inj_df=inj_df,
        summary=summary,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
