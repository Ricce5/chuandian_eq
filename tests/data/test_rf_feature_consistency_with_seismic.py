import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data import event_loader
from src.data import event_pipeline
from src.features import seismic_features


def _build_notebook_rf_feature_cols(mag_elaps):
    # Keep this aligned with config/config_loader.py::load_rf_config
    base_cols = [
        "Num",
        "Mag_max",
        "Mag_mean",
        "beta",
        "b_lsq",
        "a_lsq",
        "b_std_lsq",
        "std_gr_lsq",
        "b_mlk",
        "a_mlk",
        "b_std_mlk",
        "std_gr_mlk",
        "dM_lsq",
        "dM_mlk",
        "Energy_sqrt",
        "prob_x7_lsq",
        "prob_x7_mlk",
        "zvalue",
    ]
    telaps_cols = [f"T_elaps{mag}" for mag in mag_elaps]
    return list(dict.fromkeys(base_cols + telaps_cols))


def _make_catalog_df(n: int = 420):
    t = np.arange(n, dtype=float) * 2.5
    mag = 3.1 + 3.9 * (0.5 + 0.5 * np.sin(np.arange(n) / 8.0)) + 0.12 * np.cos(np.arange(n) / 3.0)
    lat = np.linspace(20.0, 21.0, num=n)
    lon = np.linspace(100.0, 101.0, num=n)
    dep = np.linspace(4.0, 10.0, num=n)
    dt = np.diff(np.concatenate(([t[0]], t)))
    return pd.DataFrame(
        {
            "t": t,
            "Magnitude": mag,
            "Latitude": lat,
            "Longitude": lon,
            "Depth": dep,
            "dt": dt,
            "dt_unfiltered": dt,
        }
    )


def _compare_columns(left_df, right_df, cols, *, atol: float = 1e-6, rtol: float = 1e-5):
    diff_cols = []
    stats = {}
    for col in cols:
        left = left_df[col].to_numpy(dtype=float)
        right = right_df[col].to_numpy(dtype=float)
        if left.shape != right.shape:
            raise AssertionError(f"Column shape mismatch for {col}: {left.shape} vs {right.shape}")

        n = left.shape[0]
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        both_nan = left_nan & right_nan
        finite_both = np.isfinite(left) & np.isfinite(right)
        finite_close = np.zeros(n, dtype=bool)
        finite_close[finite_both] = np.isclose(left[finite_both], right[finite_both], atol=atol, rtol=rtol)
        equal_mask = both_nan | finite_close
        diff_mask = ~equal_mask

        finite_abs_diff = np.abs(left[finite_both] - right[finite_both])
        max_abs_diff = float(np.max(finite_abs_diff)) if finite_abs_diff.size > 0 else None
        diff_count = int(np.sum(diff_mask))
        nan_mismatch = int(np.sum(np.logical_xor(left_nan, right_nan)))
        if diff_count > 0:
            diff_cols.append(col)

        stats[col] = {
            "total": int(n),
            "different_count": diff_count,
            "nan_mismatch_count": nan_mismatch,
            "max_abs_diff": max_abs_diff,
        }
    return diff_cols, stats


def _resolve_output_dir(tmp_path):
    # Optional fixed output root for easier manual inspection:
    # RF_FEATURE_CONSISTENCY_OUTDIR=/root/autodl-tmp/em_eqf/tmp/rf_feature_consistency
    outdir_env = os.environ.get("RF_FEATURE_CONSISTENCY_OUTDIR")
    if outdir_env:
        return Path(outdir_env).expanduser().resolve()
    return (tmp_path / "rf_feature_consistency").resolve()


def test_rf_feature_frame_consistency_vs_seismic_features(tmp_path):
    """
    Compare RF pipeline feature frame vs legacy seismic feature pipeline with aligned settings:
    - same Twindow / Tfore / dt
    - same sliding anchor times t
    - same Mc / Mf / Mag_elaps

    Outputs:
    - rf_feature_frame.csv
    - seismic_feature_frame.csv
    """
    cfg = OmegaConf.load(str(Path("config") / "rf.yaml"))
    feature_cols = _build_notebook_rf_feature_cols(cfg.Mag_elaps)

    df = _make_catalog_df()
    df_nl, _ = event_loader.normalize_df(df)
    samples_list, array_dict = event_loader.construct_samples_list(
        df,
        df_nl,
        Mc=float(cfg.Mc),
        Mf=float(cfg.Mf),
        Twindow=float(cfg.Twindow),
        Tfore=float(cfg.Tfore),
        dt=float(cfg.dt),
        context_len=int(cfg.context_len),
    )

    rf_features = event_pipeline.build_rf_feature_frame(
        array_dict,
        samples_list,
        feature_cols,
        Mc=float(cfg.Mc),
        dMag=0.1,
        t_elaps_mode="global",
        global_t=df["t"].to_numpy(dtype=float),
        global_mag=df["Magnitude"].to_numpy(dtype=float),
    )
    t_aligned = np.array([sample["t"] for sample in samples_list], dtype=float)

    seismic_df, _ = seismic_features.calculate_seismic_features(
        df[["t", "Magnitude"]].to_numpy(dtype=float),
        Mc=float(cfg.Mc),
        Mf=float(cfg.Mf),
        Twindow=float(cfg.Twindow),
        Tfore=float(cfg.Tfore),
        dt=float(cfg.dt),
        dMag=0.1,
        Mag_elaps=list(cfg.Mag_elaps),
        t_arrary=t_aligned,
        context_len=int(cfg.context_len),
    )

    compare_cols = [col for col in feature_cols if col in rf_features.columns and col in seismic_df.columns]
    assert compare_cols, "No overlapping feature columns to compare."

    rf_aligned = rf_features[["t", *compare_cols]].sort_values("t").reset_index(drop=True)
    seismic_aligned = seismic_df[["t", *compare_cols]].sort_values("t").reset_index(drop=True)

    assert len(rf_aligned) == len(seismic_aligned), "Row count mismatch after alignment."
    assert np.allclose(
        rf_aligned["t"].to_numpy(dtype=float),
        seismic_aligned["t"].to_numpy(dtype=float),
        equal_nan=True,
    ), "Sliding anchor times 't' are not aligned."

    diff_cols, stats = _compare_columns(rf_aligned, seismic_aligned, compare_cols)

    out_dir = _resolve_output_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf_path = out_dir / "rf_feature_frame.csv"
    seismic_path = out_dir / "seismic_feature_frame.csv"
    report_path = out_dir / "feature_diff_report.json"

    rf_aligned.to_csv(rf_path, index=False)
    seismic_aligned.to_csv(seismic_path, index=False)

    report = {
        "rf_feature_file": str(rf_path),
        "seismic_feature_file": str(seismic_path),
        "compared_columns": compare_cols,
        "different_columns": diff_cols,
        "column_stats": stats,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RF feature file: {rf_path}")
    print(f"Seismic feature file: {seismic_path}")
    print(f"Diff report: {report_path}")
    print(f"Different columns ({len(diff_cols)}): {diff_cols}")

    assert rf_path.exists()
    assert seismic_path.exists()
