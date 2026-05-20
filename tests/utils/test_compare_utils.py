import numpy as np

from src.utils.bootstrap_presets import PairedBootstrapPreset
from src.utils.classifier_compare_utils import (
    build_classification_bootstrap_export_rows,
    build_rf_em_window_ci_cache,
    summarize_classification_metric_rows,
)
from src.utils.regression_compare_utils import (
    collect_regression_split_payload,
    compute_regression_paired_block_bootstrap_tables,
)
from src.utils.bootstrap_ci import BootstrapConfig


def _make_classifier_window_row(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 120
    y_true = rng.integers(0, 2, size=n).astype(int)

    logits_rf = 0.8 * y_true + rng.normal(0.0, 1.0, size=n)
    logits_em = 1.0 * y_true + rng.normal(0.0, 1.0, size=n)
    y_prob_rf = 1.0 / (1.0 + np.exp(-logits_rf))
    y_prob_em = 1.0 / (1.0 + np.exp(-logits_em))

    seed_probs_rf = [
        np.clip(y_prob_rf + 0.03 * rng.normal(size=n), 1e-6, 1 - 1e-6),
        np.clip(y_prob_rf + 0.03 * rng.normal(size=n), 1e-6, 1 - 1e-6),
    ]
    seed_probs_em = [
        np.clip(y_prob_em + 0.03 * rng.normal(size=n), 1e-6, 1 - 1e-6),
        np.clip(y_prob_em + 0.03 * rng.normal(size=n), 1e-6, 1 - 1e-6),
    ]

    def _metrics(prob):
        threshold = 0.5
        pred = (prob >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        acc = (tp + tn) / max(n, 1)
        auc_like = float(np.mean(prob[y_true == 1]) - np.mean(prob[y_true == 0]) + 0.5)
        auc_like = float(np.clip(auc_like, 0.0, 1.0))

        return {
            "threshold": threshold,
            "f1": float(f1),
            "recall": float(recall),
            "precision": float(precision),
            "R": float(acc),
            "auc": auc_like,
            "pr_auc": float((precision + recall) / 2.0),
        }

    return {
        "window_id": 0,
        "Mf": 4.0,
        "Tfore": 10,
        "models": {
            "RF": {
                "metrics": _metrics(y_prob_rf),
                "y_true": y_true,
                "y_prob": y_prob_rf,
                "seed_probs": seed_probs_rf,
                "seed_metrics": [_metrics(seed_probs_rf[0]), _metrics(seed_probs_rf[1])],
            },
            "EM-EQF": {
                "metrics": _metrics(y_prob_em),
                "y_true": y_true,
                "y_prob": y_prob_em,
                "seed_probs": seed_probs_em,
                "seed_metrics": [_metrics(seed_probs_em[0]), _metrics(seed_probs_em[1])],
            },
        },
    }


def test_build_classification_bootstrap_export_rows_returns_expected_fields():
    multi_results = [_make_classifier_window_row(seed=7), _make_classifier_window_row(seed=9)]
    preset = PairedBootstrapPreset(
        sampling="block",
        ci=(2.5, 97.5),
        n_resamples_curve=20,
        n_resamples_metric=24,
        block_size=8,
        circular_block=True,
    )

    def _r4(x):
        if x is None:
            return None
        return round(float(x), 4)

    metrics_rows, delta_rows = build_classification_bootstrap_export_rows(
        multi_results=multi_results,
        bootstrap_preset=preset,
        round_fn=_r4,
        threshold_optimize_metric="f1",
        use_hierarchical_seed_sample=True,
        seed_aggregation="mean",
        baseline_model="EM-EQF",
        window_model_rows_getter=lambda row: row["models"],
    )

    assert len(metrics_rows) == 4
    assert len(delta_rows) == 2
    row0 = metrics_rows[0]
    for key in ("model", "auc", "auc_ci_low", "auc_ci_high", "ap", "ap_ci_low", "ap_ci_high"):
        assert key in row0
    assert row0["bootstrap_sampling"] == "block"
    assert row0["bootstrap_n_resamples"] == 24
    assert row0["bootstrap_block_size"] == 8

    delta0 = delta_rows[0]
    assert "comparison" in delta0
    assert delta0["comparison"].endswith("_minus_EM-EQF")
    assert "auc_delta_ci_low" in delta0
    assert "ap_delta_ci_high" in delta0


def test_summarize_classification_metric_rows_handles_none_values():
    metrics_rows = [
        {"model": "A", "threshold": 0.5, "f1": 0.7, "recall": 0.8, "precision": 0.6, "r": 0.75, "auc": 0.72, "ap": 0.70},
        {"model": "A", "threshold": None, "f1": 0.75, "recall": 0.82, "precision": 0.68, "r": 0.77, "auc": 0.74, "ap": 0.72},
        {"model": "B", "threshold": 0.4, "f1": 0.65, "recall": 0.7, "precision": 0.62, "r": 0.69, "auc": 0.68, "ap": 0.66},
    ]

    summary = summarize_classification_metric_rows(metrics_rows, round_fn=lambda x: None if x is None else round(float(x), 4))
    assert sorted(summary.keys()) == ["A", "B"]
    assert summary["A"]["mean_threshold"] == 0.5
    assert summary["A"]["num_windows"] == 2
    assert summary["B"]["num_windows"] == 1
    assert summary["A"]["std_auc"] is not None


def test_build_rf_em_window_ci_cache_outputs_expected_keys():
    plot_results = [_make_classifier_window_row(seed=31), _make_classifier_window_row(seed=32)]
    preset = PairedBootstrapPreset(
        sampling="block",
        ci=(2.5, 97.5),
        n_resamples_curve=16,
        n_resamples_metric=20,
        block_size=8,
        circular_block=True,
    )

    ci_cache = build_rf_em_window_ci_cache(
        plot_results,
        bootstrap_preset=preset,
        use_hierarchical_seed_sample=True,
        seed_aggregation="mean",
    )
    assert len(ci_cache) == 2
    item = ci_cache[0]
    for key in ("roc_rf", "roc_em", "pr_rf", "pr_em", "auc_ci_rf", "auc_ci_em", "ap_ci_rf", "ap_ci_em"):
        assert key in item
    roc_grid, roc_lo, roc_hi = item["roc_rf"]
    assert roc_grid.shape == roc_lo.shape == roc_hi.shape
    auc_lo, auc_hi = item["auc_ci_rf"]
    assert np.isfinite(auc_lo)
    assert np.isfinite(auc_hi)


def _canon_split(name: str) -> str:
    name_l = str(name).strip().lower()
    if name_l in {"tr", "train"}:
        return "Train"
    if name_l in {"val", "validation", "valid"}:
        return "Validation"
    if name_l in {"te", "test"}:
        return "Test"
    return str(name)


def _make_reg_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 96
    x = rng.normal(size=n)
    y_true = 0.9 * x + 0.1 * rng.normal(size=n)
    y_pred_a = 0.85 * x + 0.15 * rng.normal(size=n)
    y_pred_b = 0.80 * x + 0.20 * rng.normal(size=n)
    return y_true, y_pred_a, y_pred_b


def test_collect_regression_split_payload_aligns_models():
    y_true, y_pred_a, y_pred_b = _make_reg_data(seed=5)
    data_dicts = [
        {"Test": (y_true, y_pred_a), "Train": (y_true[:40], y_pred_a[:40])},
        {"test": (y_true, y_pred_b), "tr": (y_true[:40], y_pred_b[:40])},
    ]
    payload = collect_regression_split_payload(
        data_dicts,
        titles=["A", "B"],
        split_order=("Test",),
        canonical_split_name_fn=_canon_split,
    )
    assert "Test" in payload
    assert sorted(payload["Test"]["model_predictions"].keys()) == ["A", "B"]
    assert payload["Test"]["y_true"].shape[0] == y_true.shape[0]


def test_compute_regression_paired_block_bootstrap_tables_non_hierarchical():
    y_true, y_pred_a, y_pred_b = _make_reg_data(seed=11)
    data_dicts = [
        {"Test": (y_true, y_pred_a)},
        {"test": (y_true, y_pred_b)},
    ]

    model_df, delta_df, common_n = compute_regression_paired_block_bootstrap_tables(
        data_dicts=data_dicts,
        titles=["A", "B"],
        metrics=("rmse", "mae"),
        config=BootstrapConfig(n_resamples=20, seed=3, sampling="block", block_size=6),
        split_order=("Test",),
        baseline_model="B",
        use_hierarchical_seed_sample=False,
        seed_aggregation="mean",
        experiments=None,
        device="cpu",
        project_root=None,
        canonical_split_name_fn=_canon_split,
        get_data_dicts_from_checkpoints_fn=lambda *_args, **_kwargs: ([], []),
        mode_label="paired_block",
    )

    assert common_n is None
    assert not model_df.empty
    assert not delta_df.empty
    assert set(model_df["metric"].unique()) == {"rmse", "mae"}
    assert "ci_95" in model_df.columns
    assert "ci_95" in delta_df.columns

