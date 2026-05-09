import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data import event_loader
from src.data import event_pipeline
from src.features import seismic_features
RF_FEATURE_CONSISTENCY_OUTDIR = Path(__file__).resolve().parents[2] / "tmp" / "rf_feature_consistency"


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
    # Synthetic catalog designed to make t_elaps window/global differences visible:
    # - early strong events (>= 7) only appear near catalog start
    # - later events are clipped below 7, so window mode may miss old >=7 shocks
    mag = 4.2 + 2.2 * (0.5 + 0.5 * np.sin(np.arange(n) / 14.0)) + 0.18 * np.cos(np.arange(n) / 4.5)
    mag = np.asarray(mag, dtype=float)
    mag[:6] = np.array([7.4, 7.2, 7.1, 6.9, 6.8, 6.7], dtype=float)
    mag[6:] = np.minimum(mag[6:], 6.95)
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
    return RF_FEATURE_CONSISTENCY_OUTDIR.resolve()


def test_rf_feature_frame_consistency_vs_seismic_features(tmp_path):
    """
    Compare RF pipeline feature frame vs legacy seismic feature pipeline with aligned settings,
    and compare RF t_elaps modes ("window" vs "global"):
    - same Twindow / Tfore / dt
    - same sliding anchor times t
    - same Mc / Mf / Mag_elaps

    Outputs:
    - rf_feature_frame_window.csv
    - rf_feature_frame_global.csv
    - seismic_feature_frame.csv
    - feature_diff_report.json
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

    rf_features_window = event_pipeline.build_rf_feature_frame(
        array_dict,
        samples_list,
        feature_cols,
        Mc=float(cfg.Mc),
        dMag=0.1,
        t_elaps_mode="window",
    )
    rf_features_global = event_pipeline.build_rf_feature_frame(
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

    compare_cols = [
        col
        for col in feature_cols
        if col in rf_features_window.columns and col in rf_features_global.columns and col in seismic_df.columns
    ]
    assert compare_cols, "No overlapping feature columns to compare."

    rf_window_aligned = rf_features_window[["t", *compare_cols]].sort_values("t").reset_index(drop=True)
    rf_global_aligned = rf_features_global[["t", *compare_cols]].sort_values("t").reset_index(drop=True)
    seismic_aligned = seismic_df[["t", *compare_cols]].sort_values("t").reset_index(drop=True)

    assert len(rf_window_aligned) == len(seismic_aligned), "Row count mismatch after alignment (window mode)."
    assert len(rf_global_aligned) == len(seismic_aligned), "Row count mismatch after alignment (global mode)."
    assert np.allclose(
        rf_window_aligned["t"].to_numpy(dtype=float),
        seismic_aligned["t"].to_numpy(dtype=float),
        equal_nan=True,
    ), "Sliding anchor times 't' are not aligned (window mode)."
    assert np.allclose(
        rf_global_aligned["t"].to_numpy(dtype=float),
        seismic_aligned["t"].to_numpy(dtype=float),
        equal_nan=True,
    ), "Sliding anchor times 't' are not aligned (global mode)."

    diff_cols_window_vs_seismic, stats_window_vs_seismic = _compare_columns(
        rf_window_aligned,
        seismic_aligned,
        compare_cols,
    )
    diff_cols_global_vs_seismic, stats_global_vs_seismic = _compare_columns(
        rf_global_aligned,
        seismic_aligned,
        compare_cols,
    )
    diff_cols_window_vs_global, stats_window_vs_global = _compare_columns(
        rf_window_aligned,
        rf_global_aligned,
        compare_cols,
    )

    t_elaps_cols = [col for col in compare_cols if col.startswith("T_elaps")]
    t_elaps_mode_diff_cols = [col for col in diff_cols_window_vs_global if col in t_elaps_cols]
    non_t_elaps_mode_diff_cols = [col for col in diff_cols_window_vs_global if col not in t_elaps_cols]

    assert t_elaps_mode_diff_cols, "Expected at least one T_elaps column to differ between window/global modes."
    assert not non_t_elaps_mode_diff_cols, (
        "Only T_elaps columns should differ between window/global modes, "
        f"but got: {non_t_elaps_mode_diff_cols}"
    )

    out_dir = _resolve_output_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf_window_path = out_dir / "rf_feature_frame_window.csv"
    rf_global_path = out_dir / "rf_feature_frame_global.csv"
    seismic_path = out_dir / "seismic_feature_frame.csv"
    report_path = out_dir / "feature_diff_report.json"

    rf_window_aligned.to_csv(rf_window_path, index=False)
    rf_global_aligned.to_csv(rf_global_path, index=False)
    seismic_aligned.to_csv(seismic_path, index=False)

    report = {
        "rf_window_feature_file": str(rf_window_path),
        "rf_global_feature_file": str(rf_global_path),
        "seismic_feature_file": str(seismic_path),
        "compared_columns": compare_cols,
        "t_elaps_columns": t_elaps_cols,
        "rf_window_vs_seismic": {
            "different_columns": diff_cols_window_vs_seismic,
            "column_stats": stats_window_vs_seismic,
        },
        "rf_global_vs_seismic": {
            "different_columns": diff_cols_global_vs_seismic,
            "column_stats": stats_global_vs_seismic,
        },
        "rf_window_vs_global": {
            "different_columns": diff_cols_window_vs_global,
            "t_elaps_mode_diff_columns": t_elaps_mode_diff_cols,
            "column_stats": stats_window_vs_global,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RF feature file (window): {rf_window_path}")
    print(f"RF feature file (global): {rf_global_path}")
    print(f"Seismic feature file: {seismic_path}")
    print(f"Diff report: {report_path}")
    print(
        f"RF window vs seismic different columns ({len(diff_cols_window_vs_seismic)}): "
        f"{diff_cols_window_vs_seismic}"
    )
    print(
        f"RF global vs seismic different columns ({len(diff_cols_global_vs_seismic)}): "
        f"{diff_cols_global_vs_seismic}"
    )
    print(
        f"RF window vs global different columns ({len(diff_cols_window_vs_global)}): "
        f"{diff_cols_window_vs_global}"
    )

    assert rf_window_path.exists()
    assert rf_global_path.exists()
    assert seismic_path.exists()
