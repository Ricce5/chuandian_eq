import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.bootstrap_ci import (
    BootstrapConfig,
    aggregate_seed_predictions,
    bootstrap_curve,
    bootstrap_curve_ci,
    bootstrap_metric,
    bootstrap_metric_ci,
    bootstrap_multi_model_curve_ci,
    bootstrap_multi_model_metric_ci,
    bootstrap_multi_model_metric_ci_hierarchical,
    resolve_checkpoint_path,
    resolve_seed_checkpoint_paths,
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

    dtw_result = bootstrap_metric(
        y_true_reg,
        y_pred,
        metric="dtw",
        task="regression",
        config=BootstrapConfig(n_resamples=32, seed=10, sampling="iid"),
    )
    assert dtw_result.metric_name == "dtw"
    assert np.isfinite(dtw_result.point_estimate)
    assert dtw_result.valid_resamples > 0


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


def test_bootstrap_multi_model_metric_ci_requires_baseline_for_three_models():
    y_true, y_prob_a = _make_binary_data(seed=101)
    rng = np.random.default_rng(102)
    y_prob_b = np.clip(y_prob_a - 0.06 + 0.04 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)
    y_prob_c = np.clip(y_prob_a - 0.12 + 0.05 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)

    try:
        bootstrap_multi_model_metric_ci(
            y_true,
            {"model_a": y_prob_a, "model_b": y_prob_b, "model_c": y_prob_c},
            metric="auc",
            config=BootstrapConfig(n_resamples=60, seed=103, sampling="block", block_size=8),
        )
    except ValueError as exc:
        assert "baseline_model must be provided" in str(exc)
    else:
        raise AssertionError("Expected ValueError when baseline_model is missing for >=3 models.")


def test_bootstrap_multi_model_metric_ci_uses_baseline_deltas_for_three_models():
    y_true, y_prob_a = _make_binary_data(seed=111)
    rng = np.random.default_rng(112)
    y_prob_b = np.clip(y_prob_a - 0.05 + 0.03 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)
    y_prob_c = np.clip(y_prob_a - 0.10 + 0.04 * rng.normal(size=y_prob_a.shape[0]), 1e-6, 1 - 1e-6)

    result = bootstrap_multi_model_metric_ci(
        y_true,
        {"model_a": y_prob_a, "model_b": y_prob_b, "model_c": y_prob_c},
        metric="auc",
        baseline_model="model_b",
        config=BootstrapConfig(n_resamples=70, seed=113, sampling="block", block_size=8),
    )

    assert ("model_a", "model_b") in result.pairwise_deltas
    assert ("model_c", "model_b") in result.pairwise_deltas
    assert ("model_a", "model_c") not in result.pairwise_deltas
    for delta in result.pairwise_deltas.values():
        assert delta.valid_resamples > 0


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


def test_resolve_checkpoint_path_supports_relative_run_dir(tmp_path):
    project_root = tmp_path
    run_dir = project_root / "experiments" / "clf_grid_r_2" / "runs" / "tf_10_mf_4p0_seed_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model_1.pth"
    ckpt_path.write_bytes(b"unit-test")

    resolved = resolve_checkpoint_path(
        "experiments/clf_grid_r_2/runs/tf_10_mf_4p0_seed_1",
        project_root=project_root,
    )
    assert resolved == ckpt_path.resolve()


def test_resolve_seed_checkpoint_paths_collects_sibling_seed_runs(tmp_path):
    project_root = tmp_path
    runs_root = project_root / "experiments" / "clf_grid_r_2" / "runs"
    for seed in [2, 0, 1]:
        run_dir = runs_root / f"tf_90_mf_5p5_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model_1.pth").write_bytes(f"seed-{seed}".encode("utf-8"))

    resolved_paths = resolve_seed_checkpoint_paths(
        "experiments/clf_grid_r_2/runs/tf_90_mf_5p5_seed_1",
        project_root=project_root,
    )
    resolved_names = [path.parent.name for path in resolved_paths]
    assert resolved_names == [
        "tf_90_mf_5p5_seed_0",
        "tf_90_mf_5p5_seed_1",
        "tf_90_mf_5p5_seed_2",
    ]


def test_resolve_seed_checkpoint_paths_applies_max_seed_count(tmp_path):
    project_root = tmp_path
    runs_root = project_root / "experiments" / "clf_grid_r_2" / "runs"
    for seed in [3, 1, 0, 2]:
        run_dir = runs_root / f"tf_90_mf_5p5_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model_1.pth").write_bytes(f"seed-{seed}".encode("utf-8"))

    resolved_paths = resolve_seed_checkpoint_paths(
        "experiments/clf_grid_r_2/runs/tf_90_mf_5p5_seed_1",
        project_root=project_root,
        max_seed_count=2,
    )
    resolved_names = [path.parent.name for path in resolved_paths]
    assert resolved_names == ["tf_90_mf_5p5_seed_0", "tf_90_mf_5p5_seed_1"]


def test_resolve_seed_checkpoint_paths_collects_prefix_seed_runs(tmp_path):
    project_root = tmp_path
    runs_root = project_root / "experiments" / "clf_grid_r_2" / "runs"
    for seed in [0, 1]:
        run_dir = runs_root / f"tf_60_mf_5p0_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model_1.pth").write_bytes(f"seed-{seed}".encode("utf-8"))

    resolved_paths = resolve_seed_checkpoint_paths(
        "experiments/clf_grid_r_2/runs/tf_60_mf_5p0",
        project_root=project_root,
    )
    resolved_names = [path.parent.name for path in resolved_paths]
    assert resolved_names == ["tf_60_mf_5p0_seed_0", "tf_60_mf_5p0_seed_1"]


def test_aggregate_seed_predictions_supports_mean_and_median():
    preds = [
        np.asarray([0.1, 0.2, 0.7], dtype=np.float64),
        np.asarray([0.3, 0.4, 0.5], dtype=np.float64),
        np.asarray([0.9, 0.8, 0.1], dtype=np.float64),
    ]
    mean_pred = aggregate_seed_predictions(preds, method="mean")
    median_pred = aggregate_seed_predictions(preds, method="median")

    assert np.allclose(mean_pred, np.asarray([0.43333333, 0.46666667, 0.43333333]))
    assert np.allclose(median_pred, np.asarray([0.3, 0.4, 0.5]))


def test_aggregate_seed_predictions_validates_shape():
    preds = [
        np.asarray([0.1, 0.2], dtype=np.float64),
        np.asarray([0.3], dtype=np.float64),
    ]
    try:
        aggregate_seed_predictions(preds, method="mean")
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError when seed prediction lengths mismatch.")


def test_bootstrap_multi_model_metric_ci_hierarchical_supports_seed_and_sample_resampling():
    y_true, y_prob_base = _make_binary_data(seed=321)
    rng = np.random.default_rng(322)
    n = y_prob_base.shape[0]
    n_seeds = 3

    seed_preds_a = []
    seed_preds_b = []
    for _ in range(n_seeds):
        noise_a = 0.03 * rng.normal(size=n)
        noise_b = 0.04 * rng.normal(size=n)
        seed_preds_a.append(np.clip(y_prob_base + noise_a, 1e-6, 1 - 1e-6))
        seed_preds_b.append(np.clip(y_prob_base - 0.07 + noise_b, 1e-6, 1 - 1e-6))

    result = bootstrap_multi_model_metric_ci_hierarchical(
        y_true,
        {
            "model_a": seed_preds_a,
            "model_b": seed_preds_b,
        },
        metric="auc",
        baseline_model="model_b",
        seed_aggregation="median",
        config=BootstrapConfig(n_resamples=90, seed=323, sampling="block", block_size=8),
    )

    assert "model_a" in result.model_results
    assert "model_b" in result.model_results
    assert ("model_a", "model_b") in result.pairwise_deltas
    delta = result.pairwise_deltas[("model_a", "model_b")]
    assert delta.valid_resamples > 0
    assert delta.samples.shape[0] == delta.valid_resamples


def test_bootstrap_multi_model_metric_ci_hierarchical_validates_seed_count_alignment():
    y_true, y_prob_base = _make_binary_data(seed=401)
    n = y_prob_base.shape[0]
    seed_preds_a = [
        np.clip(y_prob_base + 0.01, 1e-6, 1 - 1e-6),
        np.clip(y_prob_base - 0.01, 1e-6, 1 - 1e-6),
    ]
    seed_preds_b = [
        np.clip(y_prob_base + 0.02, 1e-6, 1 - 1e-6),
    ]

    try:
        bootstrap_multi_model_metric_ci_hierarchical(
            y_true,
            {"model_a": seed_preds_a, "model_b": seed_preds_b},
            metric="auc",
            config=BootstrapConfig(n_resamples=20, seed=402),
        )
    except ValueError as exc:
        assert "same number of seed predictions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched per-model seed counts.")


def test_bootstrap_multi_model_metric_ci_hierarchical_point_estimate_mode_mean_differs_from_ensemble():
    y_true, y_prob_base = _make_binary_data(seed=515)
    rng = np.random.default_rng(516)
    n = y_prob_base.shape[0]
    n_seeds = 3

    seed_preds_a = []
    seed_preds_b = []
    for _ in range(n_seeds):
        noise_a = 0.08 * rng.normal(size=n)
        noise_b = 0.10 * rng.normal(size=n)
        seed_preds_a.append(np.clip(y_prob_base + noise_a, 1e-6, 1 - 1e-6))
        seed_preds_b.append(np.clip(y_prob_base - 0.06 + noise_b, 1e-6, 1 - 1e-6))

    config = BootstrapConfig(n_resamples=64, seed=517, sampling="block", block_size=8)
    ensemble_result = bootstrap_multi_model_metric_ci_hierarchical(
        y_true,
        {"model_a": seed_preds_a, "model_b": seed_preds_b},
        metric="auc",
        baseline_model="model_b",
        seed_aggregation="mean",
        point_estimate_mode="ensemble",
        config=config,
    )
    mean_result = bootstrap_multi_model_metric_ci_hierarchical(
        y_true,
        {"model_a": seed_preds_a, "model_b": seed_preds_b},
        metric="auc",
        baseline_model="model_b",
        seed_aggregation="mean",
        point_estimate_mode="mean",
        config=config,
    )

    assert "model_a" in ensemble_result.model_results
    assert "model_a" in mean_result.model_results
    est_ensemble = ensemble_result.model_results["model_a"].point_estimate
    est_mean = mean_result.model_results["model_a"].point_estimate
    assert np.isfinite(est_ensemble)
    assert np.isfinite(est_mean)
    assert abs(float(est_ensemble) - float(est_mean)) > 1e-7
    assert ensemble_result.model_results["model_a"].point_estimate_std is None
    assert mean_result.model_results["model_a"].point_estimate_std is not None
    assert float(mean_result.model_results["model_a"].point_estimate_std) >= 0.0


def test_bootstrap_multi_model_metric_ci_hierarchical_point_std_uses_sample_std_ddof1():
    y_true, y_prob_base = _make_binary_data(seed=611)
    rng = np.random.default_rng(612)
    n = y_prob_base.shape[0]
    n_seeds = 4

    seed_preds = []
    for _ in range(n_seeds):
        noise = 0.05 * rng.normal(size=n)
        seed_preds.append(np.clip(y_prob_base + noise, 1e-6, 1 - 1e-6))

    result = bootstrap_multi_model_metric_ci_hierarchical(
        y_true,
        {"model_a": seed_preds, "model_b": seed_preds},
        metric="auc",
        baseline_model="model_b",
        seed_aggregation="mean",
        point_estimate_mode="mean",
        config=BootstrapConfig(n_resamples=24, seed=613, sampling="iid"),
    )

    metric_values = [
        float(roc_auc_score(np.asarray(y_true).astype(int).reshape(-1), np.asarray(pred).reshape(-1)))
        for pred in seed_preds
    ]
    expected_sample_std = float(np.std(np.asarray(metric_values, dtype=np.float64), ddof=1))
    got_std = float(result.model_results["model_a"].point_estimate_std)
    assert np.isfinite(got_std)
    assert abs(got_std - expected_sample_std) < 1e-12
