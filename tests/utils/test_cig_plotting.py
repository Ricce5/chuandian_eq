from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")

from src.utils.cig_plotting import (
    CIGPlotConfig,
    cig_figure_stem,
    draw_split_markers,
    metadata_boundary_days,
    panel_label,
    prepare_label_curve,
    resolve_plot_start_days,
)
from src.utils.dataset_location_map import lonlat_from_columns
from src.utils.injection_data import extract_injection_count_series
from src.utils.injection_phases import visible_phase_intervals


def _curve_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["model"] * 4,
            "time_days": [0.0, 2.0, 5.0, 8.0],
            "value": [0.0, 10.0, 20.0, 30.0],
        }
    )


def test_plot_start_modes_use_strict_split_boundaries():
    curve = _curve_dataframe()
    metadata = {"val_start_t": 2.0, "test_start_t": 5.0, "freq": "1D"}

    assert resolve_plot_start_days(metadata, curve, "full") == 0.0
    assert resolve_plot_start_days(metadata, curve, "nll") == 0.0
    assert resolve_plot_start_days(metadata, curve, "val") == 2.0
    assert resolve_plot_start_days(metadata, curve, "test") == 5.0

    with pytest.raises(ValueError, match="Missing 'val' split boundary"):
        resolve_plot_start_days({}, curve, "val")
    assert resolve_plot_start_days({}, curve, "val", strict=False) == 0.0


def test_timestamp_boundary_is_converted_to_days():
    metadata = {
        "start_ts": "2020-01-01",
        "val_start_ts": "2020-01-03",
        "freq": "12h",
    }
    assert metadata_boundary_days(metadata, "val") == 2.0


def test_prepare_label_curve_inserts_crop_and_split_reset_points():
    prepared = prepare_label_curve(
        _curve_dataframe(),
        "model",
        1.0,
        value_column="value",
        reset_y=True,
        split_reset_items=(("test", 6.0),),
    )

    np.testing.assert_allclose(prepared["absolute_days"], [1.0, 2.0, 5.0, 6.0, 8.0])
    np.testing.assert_allclose(prepared["source_value"], [0.0, 10.0, 20.0, 20.0, 30.0])
    np.testing.assert_allclose(prepared["plot_value"], [0.0, 10.0, 20.0, 0.0, 10.0])
    np.testing.assert_allclose(prepared["segment_start_days"], [1.0, 1.0, 1.0, 6.0, 6.0])


def test_phase_intervals_are_clipped_and_shifted_to_plot_start():
    item = {"plot_start_days": 3.0, "curve_df": pd.DataFrame({"time_days": [0.0, 10.0]})}
    annotation = {
        "intervals": (
            {"label": "a", "start_absolute_days": 0.0, "end_absolute_days": 5.0},
            {"label": "b", "start_absolute_days": 5.0, "end_absolute_days": 12.0},
        )
    }
    visible = visible_phase_intervals(item, annotation)

    assert [interval["label"] for interval in visible] == ["a", "b"]
    assert visible[0]["left_plot_days"] == 0.0
    assert visible[0]["right_plot_days"] == 2.0
    assert visible[1]["left_plot_days"] == 2.0
    assert visible[1]["right_plot_days"] == 7.0


def test_injection_extraction_prefers_raw_series_and_counts_events():
    sequence = SimpleNamespace(
        raw_time_series=torch.tensor([[0.0], [2.0], [4.0]]),
        raw_time_series_times=torch.tensor([0.0, 1.0, 2.0]),
        time_series=torch.tensor([[99.0], [99.0], [99.0]]),
        time_series_times=torch.tensor([0.0, 1.0, 2.0]),
        arrival_times=torch.tensor([0.25, 0.75, 1.5]),
    )
    series = extract_injection_count_series(sequence, {"freq": "12h"})

    assert series["source"] == "raw_time_series"
    np.testing.assert_allclose(series["absolute_days"], [0.25, 0.75])
    np.testing.assert_allclose(series["injection"], [1.0, 3.0])
    np.testing.assert_allclose(series["counts"], [2.0, 1.0])


def test_location_columns_accept_common_degree_names():
    dataframe = pd.DataFrame({"Latitude (deg)": [10.0], "Longitude (deg)": [20.0]})
    points = lonlat_from_columns(dataframe, dataset="demo", source="input.csv")

    assert points.to_dict("records") == [{"lat": 10.0, "lon": 20.0, "source": "input.csv"}]


def test_configuration_normalizes_aliases_and_panel_labels():
    config = CIGPlotConfig(reset_boundary_splits=("validation", "test", "val"))
    assert config.reset_boundary_splits == ("val", "test")
    assert config.direct_labels is True
    assert config.show_legend is False
    assert panel_label(0) == "(a)"
    assert panel_label(26) == "(aa)"


def test_cig_figure_stem_is_concise_and_marks_only_nondefault_modes():
    assert cig_figure_stem(CIGPlotConfig(start_mode="val")) == "cig_from_val"
    assert (
        cig_figure_stem(
            CIGPlotConfig(
                start_mode="full",
                reset_y_at_plot_start=False,
                reset_at_split_boundaries=True,
            ),
        )
        == "cig_from_full_absolute_reset_val-test"
    )


def test_close_split_marker_labels_are_staggered():
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 100.0)
    draw_split_markers(
        axis,
        {"plot_start_days": 0.0, "val_start_days": 40.0, "test_start_days": 42.0},
    )

    assert [text.get_text() for text in axis.texts] == ["Val", "Test"]
    assert axis.texts[0].get_position()[1] != axis.texts[1].get_position()[1]
    plt.close(figure)
