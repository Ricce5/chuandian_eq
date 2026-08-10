import numpy as np
import pytest
from matplotlib import pyplot as plt

from src.utils.induced_bg_comparison import (
    BackgroundComparisonLayout,
    BackgroundComparisonStyle,
    build_model_style_map,
    comparison_output_stem,
    filename_token,
    focused_window_end,
    save_figure_bundle,
)


def test_filename_token_and_output_stem_preserve_spec_identity():
    assert filename_token("ETAS+conv") == "ETAS_conv"
    assert comparison_output_stem("background", "CB_HAB4") == "background_CB_HAB4"
    assert (
        comparison_output_stem(
            "window",
            "CB_HAB4",
            layout=BackgroundComparisonLayout(
                overlay_background=False,
                panel_b_only=True,
            ),
        )
        == "window_CB_HAB4_stacked_panel"
    )
    assert (
        comparison_output_stem("n_test", "CB_HAB4", primary_model="ETAS+conv")
        == "n_test_CB_HAB4_ETAS_conv"
    )


def test_model_style_map_is_stable_and_accepts_index_and_label_overrides():
    specs = [{"label": "a"}, {"label": "b"}]
    styles = build_model_style_map(
        specs,
        BackgroundComparisonStyle(),
        overrides={0: {"linewidth": 2.0}, "b": {"color": "black"}},
    )
    assert styles["a"]["linewidth"] == 2.0
    assert styles["b"]["color"] == "black"
    assert styles["a"]["color"] != styles["b"]["color"]


def test_focused_window_end_trims_inactive_tail():
    times = np.arange(0.0, 5.0, 1.0)
    result = focused_window_end(
        times,
        times,
        np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        np.zeros((1, 5)),
        window_start=0.0,
        window_end=5.0,
    )
    assert result == pytest.approx(1.25)


def test_focused_window_end_keeps_window_when_no_activity_exists():
    result = focused_window_end(
        np.arange(3.0),
        np.arange(3.0),
        np.zeros(3),
        np.zeros((2, 3)),
        window_start=0.0,
        window_end=3.0,
    )
    assert result == 3.0


def test_save_bundle_preserves_existing_axis_labels(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.set_xlabel("Time")
    axis.set_ylabel("Rate")

    paths = save_figure_bundle(
        figure,
        tmp_path,
        "figure",
        style=BackgroundComparisonStyle(figure_formats=("png",), dpi=72),
    )

    assert axis.get_xlabel() == "Time"
    assert axis.get_ylabel() == "Rate"
    assert paths["png"].is_file()
    plt.close(figure)
