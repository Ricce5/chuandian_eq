import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize import summarize_sliding_window_eval as summarize


def _write_metrics(path, *, rmse):
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "coverage": 0.8,
                "mae": 10.0,
                "rmse": rmse,
                "crps": 5.0,
                "w95": 20.0,
                "lp_nb": -2.0,
                "mag_max_mae": 0.2,
                "num_windows": 3,
                "settings": {
                    "duration": 1.0,
                    "step": 1.0,
                    "eval_range": "test",
                    "include_truncated_final_window": True,
                    "samples_per_batch": 1000,
                    "predict_b": False,
                },
                "evaluation_range": {
                    "name": "test",
                    "start": 10.0,
                    "end": 20.0,
                },
                "metadata": {
                    "dataset": "Demo",
                    "registry_name": "Demo-Standard",
                    "seed": 0,
                },
                "sliding_loaded_from_cache": False,
            }
        ),
        encoding="utf-8",
    )


def test_default_output_prefix_uses_metrics_filename_stem():
    assert (
        summarize.default_output_prefix("sliding_window_eval_metrics.json")
        == "sliding_window_eval_metrics"
    )
    assert (
        summarize.default_output_prefix("sliding_window_eval_metrics_test_trunc.json")
        == "sliding_window_eval_metrics_test_trunc"
    )


def test_different_metrics_files_get_non_conflicting_reports(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiment"
    run_dir = exp_dir / "runs" / "demo_seed_0"
    run_dir.mkdir(parents=True)
    _write_metrics(run_dir / "sliding_window_eval_metrics.json", rmse=2.0)
    _write_metrics(
        run_dir / "sliding_window_eval_metrics_test_trunc.json",
        rmse=3.0,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_sliding_window_eval.py",
            "--exp_dir",
            str(exp_dir),
        ],
    )
    summarize.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_sliding_window_eval.py",
            "--exp-dir",
            str(exp_dir),
            "--metrics-filename",
            "sliding_window_eval_metrics_test_trunc.json",
        ],
    )
    summarize.main()

    reports_dir = exp_dir / "reports"
    default_summary = json.loads(
        (reports_dir / "sliding_window_eval_metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    trunc_summary = json.loads(
        (
            reports_dir
            / "sliding_window_eval_metrics_test_trunc_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert default_summary["metrics_filename"] == "sliding_window_eval_metrics.json"
    assert trunc_summary["metrics_filename"] == "sliding_window_eval_metrics_test_trunc.json"
    assert default_summary["n_success"] == 1
    assert trunc_summary["n_success"] == 1
    assert (
        reports_dir / "sliding_window_eval_metrics_per_run.csv"
    ).exists()
    assert (
        reports_dir / "sliding_window_eval_metrics_test_trunc_per_run.csv"
    ).exists()


def test_exclude_truncated_final_window_recomputes_from_cache(monkeypatch, tmp_path):
    exp_dir = tmp_path / "experiment"
    run_dir = exp_dir / "runs" / "demo_seed_0"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "sliding_window_eval_metrics_test_trunc.json"
    cache_path = run_dir / "sliding_window_cache_model_model_test_demo.npz"

    np.savez_compressed(
        cache_path,
        duration=np.float64(1.0),
        evaluation_end=np.float64(2.5),
        t_forecast_list=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        counts_list=np.asarray([1, 2, 10], dtype=np.int64),
        q_list=np.asarray([[0.0, 2.0], [1.0, 3.0], [0.0, 20.0]], dtype=np.float64),
        mean_list=np.asarray([1.0, 4.0, 20.0], dtype=np.float64),
        sim_count_matrix=np.asarray(
            [[1, 1, 1, 1], [3, 4, 5, 6], [10, 11, 12, 13]],
            dtype=np.int32,
        ),
    )
    metrics_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "coverage": 1.0,
                "mae": 4.0,
                "rmse": 5.0,
                "crps": 6.0,
                "w95": 7.0,
                "lp_nb": -8.0,
                "num_windows": 3,
                "settings": {
                    "duration": 1.0,
                    "step": 1.0,
                    "eval_range": "test",
                    "include_truncated_final_window": True,
                    "samples_per_batch": 4,
                    "predict_b": False,
                    "cache_filename": cache_path.name,
                },
                "evaluation_range": {
                    "name": "test",
                    "start": 0.0,
                    "end": 2.5,
                },
                "metadata": {
                    "dataset": "Demo",
                    "registry_name": "Demo-Standard",
                    "seed": 0,
                },
                "sliding_cache_path": str(cache_path),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_sliding_window_eval.py",
            "--exp_dir",
            str(exp_dir),
            "--metrics_filename",
            metrics_path.name,
            "--exclude-truncated-final-window",
        ],
    )
    summarize.main()

    reports_dir = exp_dir / "reports"
    summary = json.loads(
        (
            reports_dir
            / "sliding_window_eval_metrics_test_trunc_no_truncated_final_window_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["exclude_truncated_final_window"] is True

    per_run_csv = (
        reports_dir
        / "sliding_window_eval_metrics_test_trunc_no_truncated_final_window_per_run.csv"
    )
    lines = per_run_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    values = lines[1].split(",")
    row = dict(zip(header, values))

    assert math.isclose(float(row["num_windows"]), 2.0)
    assert math.isclose(float(row["summary_filter.num_windows_excluded"]), 1.0)
    assert math.isclose(float(row["summary_filter.recomputed_from_cache"]), 1.0)
    assert math.isclose(float(row["mae"]), 1.0)
    assert math.isclose(float(row["rmse"]), math.sqrt(2.0))
