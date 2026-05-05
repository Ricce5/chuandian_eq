import pytest

from src.utils.bootstrap_presets import PairedBootstrapPreset


def test_paired_bootstrap_preset_builds_configs():
    preset = PairedBootstrapPreset(
        sampling="block",
        ci=(5.0, 95.0),
        n_resamples_curve=120,
        n_resamples_metric=240,
        block_size=12,
        circular_block=False,
    )

    curve_cfg = preset.curve_config(seed=101)
    metric_cfg = preset.metric_config(seed=202)

    assert curve_cfg.n_resamples == 120
    assert metric_cfg.n_resamples == 240
    assert curve_cfg.sampling == "block"
    assert metric_cfg.sampling == "block"
    assert curve_cfg.block_size == 12
    assert metric_cfg.block_size == 12
    assert curve_cfg.ci == (5.0, 95.0)
    assert metric_cfg.ci == (5.0, 95.0)
    assert curve_cfg.seed == 101
    assert metric_cfg.seed == 202
    assert curve_cfg.circular_block is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples_curve": 0},
        {"n_resamples_metric": 0},
        {"ci": (97.5, 2.5)},
        {"ci": (-1.0, 95.0)},
        {"block_size": 0},
    ],
)
def test_paired_bootstrap_preset_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        PairedBootstrapPreset(**kwargs)
