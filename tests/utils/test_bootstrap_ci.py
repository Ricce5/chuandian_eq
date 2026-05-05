import numpy as np

from src.utils.bootstrap_ci import (
    BootstrapConfig,
    bootstrap_curve,
    bootstrap_curve_ci,
    bootstrap_metric,
    bootstrap_metric_ci,
    bootstrap_multi_model_curve_ci,
    bootstrap_multi_model_metric_ci,
)


def _make_binary_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    logits = (y_true * 1.2) + rng.normal(0.0, 1.0, size=n)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    return y_true, y_prob


def _make_reg_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y_true = 0.8 * x + 0.1 * rng.normal(size=n)
    y_pred = 0.75 * x + 0.2 * rng.normal(size=n)
    return y_true, y_pred


def test_bootstrap_curve_supports_block_sampling():
    y_true, y_prob = _make_binary_data()
    result = bootstrap_curve(
        y_true,
        y_prob,
        curve="roc",
        config=BootstrapConfig(
            n_resamples=64,
            seed=123,
            sampling="block",
            block_size=12,
            circular_block=True,
        ),
        grid_n=101,
    )

    assert result.grid.shape == (101,)
    assert result.ci_low.shape == (101,)
    assert result.ci_high.shape == (101,)
    assert result.valid_resamples > 0
    assert result.total_resamples == 64


def test_bootstrap_metric_supports_classification_and_regression():
    y_true_cls, y_prob = _make_binary_data()
    cls_result = bootstrap_metric(
        y_true_cls,
        y_prob,
        metric="auc",
        config=BootstrapConfig(n_resamples=64, seed=7),
    )
    assert cls_result.metric_name == "auc"
    assert np.isfinite(cls_result.point_estimate)
    assert cls_result.valid_resamples > 0

    y_true_reg, y_pred = _make_reg_data()
    reg_result = bootstrap_metric(
        y_true_reg,
        y_pred,
        metric="rmse",
        task="regression",
        config=BootstrapConfig(n_resamples=64, seed=9, sampling="block", block_size=10),
    )
    assert reg_result.metric_name == "rmse"
    assert np.isfinite(reg_result.point_estimate)
    assert reg_result.valid_resamples > 0


def test_bootstrap_multi_model_metric_ci_uses_paired_indices():
    y_true, y_prob_a = _make_binary_data(seed=11)
    rng = np.random.default_rng(12)
    y_prob_b = np.clip(y_prob_a - 0.1 + 0.05 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)

    result = bootstrap_multi_model_metric_ci(
        y_true,
        {"model_a": y_prob_a, "model_b": y_prob_b},
        metric="auc",
        config=BootstrapConfig(n_resamples=80, seed=13, sampling="block", block_size=8),
    )

    assert "model_a" in result.model_results
    assert "model_b" in result.model_results
    assert ("model_a", "model_b") in result.pairwise_deltas
    delta = result.pairwise_deltas[("model_a", "model_b")]
    assert delta.valid_resamples > 0
    assert delta.samples.shape[0] == delta.valid_resamples


def test_bootstrap_multi_model_curve_ci_supports_paired_block_sampling():
    y_true, y_prob_a = _make_binary_data(seed=31)
    rng = np.random.default_rng(32)
    y_prob_b = np.clip(y_prob_a - 0.08 + 0.04 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)

    result = bootstrap_multi_model_curve_ci(
        y_true,
        {"model_a": y_prob_a, "model_b": y_prob_b},
        curve="pr",
        config=BootstrapConfig(n_resamples=72, seed=33, sampling="block", block_size=9),
        grid_n=111,
    )

    assert result.curve == "pr"
    assert result.grid.shape == (111,)
    assert "model_a" in result.model_results
    assert "model_b" in result.model_results
    assert result.valid_resamples > 0

    curve_a = result.model_results["model_a"]
    curve_b = result.model_results["model_b"]
    assert curve_a.valid_resamples == curve_b.valid_resamples == result.valid_resamples
    assert curve_a.ci_low.shape == (111,)
    assert curve_b.ci_high.shape == (111,)


def test_backward_compatible_wrappers_still_work():
    y_true, y_prob = _make_binary_data(seed=21)
    grid, lo, hi = bootstrap_curve_ci(y_true, y_prob, n_boot=32, seed=22, grid_n=41)
    assert grid.shape == (41,)
    assert lo.shape == (41,)
    assert hi.shape == (41,)

    ci_low, ci_high = bootstrap_metric_ci(y_true, y_prob, metric="auc", n_boot=32, seed=23)
    assert np.isfinite(ci_low)
    assert np.isfinite(ci_high)
