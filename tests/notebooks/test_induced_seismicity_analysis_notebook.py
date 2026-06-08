import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, peak_widths


NOTEBOOK_PATH = Path("notebooks/induced_seismicity_analysis.ipynb")


def _load_notebook_namespace() -> dict[str, Any]:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    target_src = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def plot_peak_window(" in src and "def export_triplet_plots(" in src:
            target_src = src
            break

    if target_src is None:
        raise AssertionError("Could not find plotting helpers cell in notebook.")

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "np": np,
        "pd": pd,
        "plt": plt,
        "mdates": mdates,
        "find_peaks": find_peaks,
        "peak_widths": peak_widths,
    }
    exec(target_src, namespace)
    return namespace


def _make_series() -> dict[str, Any]:
    n = 8
    return {
        "inj_time_mid_ts": pd.date_range("2020-01-01", periods=n, freq="h"),
        "inj_time_mid": np.arange(n, dtype=float),
        "injection_mid": np.array([0.0, 0.2, 0.9, 0.4, 0.1, 0.0, 0.0, 0.0], dtype=float),
        "bin_counts": np.array([0, 1, 3, 1, 0, 0, 0, 0], dtype=float),
        "arrival_times": np.array([1.2, 2.4, 2.8, 3.1], dtype=float),
    }


def _make_peaks() -> dict[str, np.ndarray]:
    return {
        "peaks_idx": np.array([2], dtype=int),
        "peak_heights": np.array([0.9], dtype=float),
        "widths": np.array([2.0], dtype=float),
        "width_heights": np.array([0.0], dtype=float),
        "left_bases": np.array([1], dtype=int),
        "right_bases": np.array([4], dtype=int),
    }


def test_plot_peak_window_clips_out_of_range_peak_rank(tmp_path, capsys):
    namespace = _load_notebook_namespace()
    plot_peak_window = namespace["plot_peak_window"]

    out_path = tmp_path / "peak.png"
    saved = plot_peak_window(
        "Demo",
        _make_series(),
        _make_peaks(),
        peak_rank=2,
        offset=2,
        save_path=out_path,
        show=False,
    )

    captured = capsys.readouterr().out
    assert "requested peak_rank=2" in captured
    assert "using peak_rank=0" in captured
    assert saved == out_path
    assert out_path.exists()


def test_plot_peak_window_returns_none_when_no_peaks(capsys):
    namespace = _load_notebook_namespace()
    plot_peak_window = namespace["plot_peak_window"]

    saved = plot_peak_window(
        "Demo",
        _make_series(),
        {
            "peaks_idx": np.array([], dtype=int),
            "left_bases": np.array([], dtype=int),
            "right_bases": np.array([], dtype=int),
        },
        peak_rank=0,
        show=False,
    )

    captured = capsys.readouterr().out
    assert "no peaks found with current thresholds" in captured
    assert saved is None


def test_export_triplet_plots_records_effective_peak_rank(tmp_path):
    namespace = _load_notebook_namespace()
    export_triplet_plots = namespace["export_triplet_plots"]

    series = _make_series()
    peaks = _make_peaks()

    class _FakeSequence:
        def __init__(self):
            self.mag = torch.tensor([2.0, 2.2, 2.4], dtype=torch.float32)

    class _FakeCatalog:
        def __init__(self):
            self.full_sequence = _FakeSequence()
            self.metadata = {
                "start_ts": "2020-01-01T00:00:00",
                "freq": "1h",
                "name": "Demo",
                "mag_completeness": 2.0,
            }

    namespace["PROJECT_ROOT"] = tmp_path
    namespace["normalize_dataset_name"] = lambda value: value
    namespace["load_catalog_flexible"] = lambda ds_name, catalog_cfg=None: _FakeCatalog()
    namespace["extract_injection_count_series"] = lambda seq, metadata=None: series
    namespace["detect_peaks"] = lambda injection_mid, peak_height=0.5, peak_min_distance=60: peaks

    df = export_triplet_plots(["Demo"], peak_rank=2, offset=2, dpi=80)

    assert df.loc[0, "requested_peak_rank"] == 2
    assert df.loc[0, "effective_peak_rank"] == 0
    assert "peak_rank0" in df.loc[0, "peak_plot"]
    assert (tmp_path / df.loc[0, "peak_plot"]).exists()
