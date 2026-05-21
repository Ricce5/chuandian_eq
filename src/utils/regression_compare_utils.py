"""Reusable helpers for regression comparison notebooks.

This module extracts bootstrap-heavy notebook logic into testable helpers so
notebook cells stay concise and easier to read.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.classifier_compare_utils import paired_metric_bootstrap
from src.utils.bootstrap_ci import (
    BootstrapConfig,
    PointEstimateMode,
    SeedAggregationMethod,
    resolve_seed_checkpoint_paths,
)


def collect_regression_split_payload(
    data_dicts: Sequence[Mapping[str, tuple[Sequence[float], Sequence[float]]]],
    titles: Sequence[str],
    split_order: Sequence[str],
    *,
    canonical_split_name_fn: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    """Build aligned per-split payload for paired regression bootstrap."""

    split_payload: dict[str, dict[str, Any]] = {}
    for model_title, data_dict in zip(titles, data_dicts):
        for raw_split_name, (y_true_raw, y_pred_raw) in data_dict.items():
            split_name = canonical_split_name_fn(raw_split_name)
            if split_name not in split_order:
                continue

            y_true = np.asarray(y_true_raw).reshape(-1)
            y_pred = np.asarray(y_pred_raw).reshape(-1)
            if y_true.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    f"Length mismatch in {model_title}/{split_name}: "
                    f"y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}"
                )

            payload = split_payload.setdefault(
                split_name,
                {"y_true": None, "model_predictions": {}},
            )
            if payload["y_true"] is None:
                payload["y_true"] = y_true
            else:
                ref = payload["y_true"]
                if ref.shape[0] != y_true.shape[0] or not np.allclose(
                    ref,
                    y_true,
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"y_true mismatch across models for split={split_name}. "
                        "Expected aligned samples for paired bootstrap."
                    )

            payload["model_predictions"][str(model_title)] = y_pred

    return split_payload


def _split_pair_from_data_dict(
    data_dict: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    split_name: str,
    model_title: str,
    seed_idx: int,
    *,
    canonical_split_name_fn: Callable[[str], str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one canonical split ``(y_true, y_pred)`` from a seed data dict."""

    matches: list[tuple[str, tuple[Sequence[float], Sequence[float]]]] = []
    for raw_split_name, pair in data_dict.items():
        if canonical_split_name_fn(raw_split_name) == split_name:
            matches.append((raw_split_name, pair))

    if len(matches) == 0:
        raise KeyError(
            f"Split={split_name!r} not found for model={model_title!r}, seed_index={seed_idx}. "
            f"Available splits={list(data_dict.keys())}"
        )
    if len(matches) > 1:
        raw_names = [name for name, _ in matches]
        raise ValueError(
            f"Ambiguous split mapping for split={split_name!r}, model={model_title!r}, "
            f"seed_index={seed_idx}: raw names={raw_names}"
        )

    y_true_raw, y_pred_raw = matches[0][1]
    y_true = np.asarray(y_true_raw).reshape(-1)
    y_pred = np.asarray(y_pred_raw).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch in {model_title}/{split_name}/seed_{seed_idx}: "
            f"y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}"
        )
    return y_true, y_pred


def resolve_seed_checkpoint_map(
    experiments: Sequence[Mapping[str, Any]],
    titles: Sequence[str],
    *,
    project_root: Path,
    selected_seeds: Sequence[int] | None = None,
    max_seed_count: int | None = None,
) -> tuple[dict[str, list[Path]], int]:
    """Resolve and align per-model seed checkpoint paths from experiment specs."""

    exp_by_title = {str(exp["title"]): exp for exp in experiments}
    seed_ckpts_by_model: dict[str, list[Path]] = {}

    for model_title in titles:
        if model_title not in exp_by_title:
            raise KeyError(
                f"Model title={model_title!r} not found in experiments. "
                f"Available={list(exp_by_title.keys())}"
            )

        checkpoint = Path(exp_by_title[model_title]["checkpoint"])
        checkpoint_filename = (
            checkpoint.name if checkpoint.suffix.lower() == ".pth" else "best_model_1.pth"
        )
        path_like = checkpoint.parent if checkpoint.suffix.lower() == ".pth" else checkpoint

        seed_ckpts = resolve_seed_checkpoint_paths(
            path_like,
            project_root=project_root,
            checkpoint_filename=checkpoint_filename,
            selected_seeds=selected_seeds,
            max_seed_count=max_seed_count,
        )
        seed_ckpts_by_model[model_title] = [Path(path) for path in seed_ckpts]

    if len(seed_ckpts_by_model) == 0:
        raise ValueError("No seed checkpoints resolved from experiments.")

    seed_counts = {model: len(paths) for model, paths in seed_ckpts_by_model.items()}
    common_n_seeds = min(seed_counts.values())
    if common_n_seeds <= 0:
        raise ValueError(f"At least one model has no checkpoint seeds: {seed_counts}")

    if len(set(seed_counts.values())) > 1:
        print(
            "[Warn] Models have different numbers of resolved seeds; "
            f"truncating all models to common_n_seeds={common_n_seeds}. Seed counts={seed_counts}"
        )

    for model_title in seed_ckpts_by_model:
        seed_ckpts_by_model[model_title] = seed_ckpts_by_model[model_title][:common_n_seeds]

    return seed_ckpts_by_model, int(common_n_seeds)


def build_model_seed_data_dicts(
    *,
    experiments: Sequence[Mapping[str, Any]],
    titles: Sequence[str],
    device: Any,
    project_root: Path,
    get_data_dicts_from_checkpoints_fn: Callable[[Sequence[Mapping[str, Any]], Any], tuple[list[Any], list[str]]],
    selected_seeds: Sequence[int] | None = None,
    max_seed_count: int | None = None,
    seed_cache_path: Path | None = None,
    use_seed_cache: bool = True,
    save_seed_cache: bool = True,
) -> tuple[dict[str, list[Mapping[str, Any]]], int | None]:
    """Build ``{model_title: [seed_data_dict, ...]}`` for hierarchical bootstrap."""

    cache_payload: Any = None
    if use_seed_cache and seed_cache_path is not None and seed_cache_path.exists():
        with open(seed_cache_path, "rb") as f:
            cache_payload = pickle.load(f)

    if isinstance(cache_payload, dict):
        cached_titles = cache_payload.get("titles")
        cached_models = cache_payload.get("model_seed_data_dicts")
        cached_n_seeds = cache_payload.get("common_n_seeds")
        cached_selected_seeds = cache_payload.get("selected_seeds")
        cached_max_seed_count = cache_payload.get("max_seed_count")
        normalized_selected_seeds = (
            [int(seed) for seed in selected_seeds]
            if selected_seeds is not None
            else None
        )
        if (
            isinstance(cached_models, dict)
            and cached_titles == list(titles)
            and cached_selected_seeds == normalized_selected_seeds
            and cached_max_seed_count == max_seed_count
            and all(title in cached_models for title in titles)
        ):
            print(f"Loaded bootstrap multi-seed cache from: {seed_cache_path}")
            return cached_models, int(cached_n_seeds) if cached_n_seeds is not None else None

    exp_by_title = {str(exp["title"]): exp for exp in experiments}
    seed_ckpts_by_model, common_n_seeds = resolve_seed_checkpoint_map(
        experiments,
        titles,
        project_root=project_root,
        selected_seeds=selected_seeds,
        max_seed_count=max_seed_count,
    )

    seed_experiments: list[dict[str, Any]] = []
    seed_meta: list[tuple[str, int, Path]] = []
    for model_title in titles:
        exp = exp_by_title[model_title]
        for seed_idx, ckpt_path in enumerate(seed_ckpts_by_model[model_title]):
            seed_experiments.append(
                {
                    "title": f"{model_title}__seed_{seed_idx}",
                    "checkpoint": ckpt_path,
                    "prepare_fn": exp["prepare_fn"],
                }
            )
            seed_meta.append((str(model_title), int(seed_idx), ckpt_path))

    seed_data_dicts, _ = get_data_dicts_from_checkpoints_fn(seed_experiments, device)

    model_seed_data_dicts: dict[str, list[Mapping[str, Any]]] = {
        str(title): [None] * int(common_n_seeds) for title in titles
    }
    for (model_title, seed_idx, _ckpt_path), data_dict in zip(seed_meta, seed_data_dicts):
        model_seed_data_dicts[model_title][seed_idx] = data_dict

    for model_title in titles:
        if any(item is None for item in model_seed_data_dicts[model_title]):
            raise RuntimeError(f"Missing seed data for model={model_title!r}.")

    if save_seed_cache and seed_cache_path is not None:
        with open(seed_cache_path, "wb") as f:
            pickle.dump(
                {
                    "titles": list(titles),
                    "common_n_seeds": int(common_n_seeds),
                    "selected_seeds": [int(seed) for seed in selected_seeds]
                    if selected_seeds is not None
                    else None,
                    "max_seed_count": max_seed_count,
                    "model_seed_data_dicts": model_seed_data_dicts,
                },
                f,
            )
        print(f"Saved bootstrap multi-seed cache to: {seed_cache_path}")

    return model_seed_data_dicts, int(common_n_seeds)


def collect_regression_split_payload_from_seed_data(
    model_seed_data_dicts: Mapping[str, Sequence[Mapping[str, Any]]],
    titles: Sequence[str],
    split_order: Sequence[str],
    *,
    canonical_split_name_fn: Callable[[str], str],
    seed_aggregation: SeedAggregationMethod,
) -> dict[str, dict[str, Any]]:
    """Build split payload from per-seed prediction dicts."""

    if seed_aggregation not in ("mean", "median"):
        raise ValueError(f"Unsupported seed aggregation method: {seed_aggregation!r}")

    split_payload: dict[str, dict[str, Any]] = {}

    for model_title in titles:
        seed_data_dict_list = model_seed_data_dicts.get(model_title)
        if not seed_data_dict_list:
            raise ValueError(f"No seed data dicts found for model={model_title!r}.")

        for split_name in split_order:
            seed_preds: list[np.ndarray] = []
            y_true_ref: np.ndarray | None = None

            for seed_idx, seed_data_dict in enumerate(seed_data_dict_list):
                y_true_seed, y_pred_seed = _split_pair_from_data_dict(
                    seed_data_dict,
                    split_name,
                    model_title,
                    seed_idx,
                    canonical_split_name_fn=canonical_split_name_fn,
                )

                if y_true_ref is None:
                    y_true_ref = y_true_seed
                else:
                    if (
                        y_true_ref.shape[0] != y_true_seed.shape[0]
                        or not np.allclose(y_true_ref, y_true_seed, rtol=1e-6, atol=1e-8)
                    ):
                        raise ValueError(
                            f"y_true mismatch across seeds for model={model_title!r}, split={split_name!r}, "
                            f"seed_index={seed_idx}."
                        )

                seed_preds.append(y_pred_seed)

            if y_true_ref is None or len(seed_preds) == 0:
                continue

            seed_pred_matrix = np.stack(seed_preds, axis=0)
            if seed_aggregation == "mean":
                y_pred_agg = np.nanmean(seed_pred_matrix, axis=0)
            else:
                y_pred_agg = np.nanmedian(seed_pred_matrix, axis=0)

            payload = split_payload.setdefault(
                split_name,
                {
                    "y_true": None,
                    "model_predictions": {},
                    "model_seed_predictions": {},
                    "n_model_seeds": {},
                },
            )

            if payload["y_true"] is None:
                payload["y_true"] = y_true_ref
            else:
                ref = payload["y_true"]
                if ref.shape[0] != y_true_ref.shape[0] or not np.allclose(
                    ref,
                    y_true_ref,
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"y_true mismatch across models for split={split_name!r}. "
                        "Expected aligned samples for paired bootstrap."
                    )

            payload["model_predictions"][model_title] = y_pred_agg.reshape(-1)
            payload["model_seed_predictions"][model_title] = seed_pred_matrix
            payload["n_model_seeds"][model_title] = int(seed_pred_matrix.shape[0])

    return split_payload


def compute_regression_paired_block_bootstrap_tables(
    *,
    data_dicts: Sequence[Mapping[str, tuple[Sequence[float], Sequence[float]]]],
    titles: Sequence[str],
    metrics: Sequence[str],
    config: BootstrapConfig,
    split_order: Sequence[str],
    baseline_model: str | None,
    use_hierarchical_seed_sample: bool,
    seed_aggregation: SeedAggregationMethod,
    point_estimate_mode: PointEstimateMode = "ensemble",
    experiments: Sequence[Mapping[str, Any]] | None,
    device: Any,
    project_root: Path,
    canonical_split_name_fn: Callable[[str], str],
    get_data_dicts_from_checkpoints_fn: Callable[[Sequence[Mapping[str, Any]], Any], tuple[list[Any], list[str]]],
    mode_label: str,
    selected_seeds: Sequence[int] | None = None,
    max_seed_count: int | None = None,
    seed_cache_path: Path | None = None,
    use_seed_cache: bool = True,
    save_seed_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Compute regression metric table and pairwise delta CI table."""

    common_n_seeds: int | None = None
    if use_hierarchical_seed_sample:
        if experiments is None:
            raise ValueError("experiments is required when use_hierarchical_seed_sample=True.")
        model_seed_data_dicts, common_n_seeds = build_model_seed_data_dicts(
            experiments=experiments,
            titles=titles,
            device=device,
            project_root=project_root,
            get_data_dicts_from_checkpoints_fn=get_data_dicts_from_checkpoints_fn,
            selected_seeds=selected_seeds,
            max_seed_count=max_seed_count,
            seed_cache_path=seed_cache_path,
            use_seed_cache=use_seed_cache,
            save_seed_cache=save_seed_cache,
        )
        split_payload = collect_regression_split_payload_from_seed_data(
            model_seed_data_dicts,
            titles,
            split_order,
            canonical_split_name_fn=canonical_split_name_fn,
            seed_aggregation=seed_aggregation,
        )
    else:
        split_payload = collect_regression_split_payload(
            data_dicts,
            titles,
            split_order,
            canonical_split_name_fn=canonical_split_name_fn,
        )

    ordered_splits = [split for split in split_order if split in split_payload]
    effective_baseline_model = baseline_model or (titles[0] if len(titles) > 0 else None)
    if effective_baseline_model is None:
        raise ValueError("No models found; cannot determine baseline model.")

    model_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for split_name in ordered_splits:
        payload = split_payload[split_name]
        y_true = payload["y_true"]
        model_predictions = payload["model_predictions"]
        model_seed_predictions = payload.get("model_seed_predictions")

        for metric_name in metrics:
            result = paired_metric_bootstrap(
                y_true=y_true,
                metric_name=str(metric_name),
                metric_config=config,
                use_hierarchical_seed_sample=use_hierarchical_seed_sample,
                seed_aggregation=seed_aggregation,
                point_estimate_mode=point_estimate_mode,
                baseline_model=effective_baseline_model,
                model_predictions=model_predictions,
                model_seed_predictions=model_seed_predictions,
                task="regression",
            )

            for model_name, boot in result.model_results.items():
                model_rows.append(
                    {
                        "mode": mode_label,
                        "baseline_model": effective_baseline_model,
                        "model": str(model_name),
                        "split": str(split_name),
                        "metric": str(metric_name),
                        "n_obs": int(y_true.size),
                        "point_estimate": float(boot.point_estimate),
                        "point_estimate_std": (
                            float(boot.point_estimate_std)
                            if boot.point_estimate_std is not None
                            else None
                        ),
                        "ci_low": float(boot.ci_low),
                        "ci_high": float(boot.ci_high),
                        "valid_resamples": int(boot.valid_resamples),
                        "total_resamples": int(boot.total_resamples),
                        "n_model_seeds": int(payload.get("n_model_seeds", {}).get(model_name, 1)),
                    }
                )

            for (left_model, right_model), delta in result.pairwise_deltas.items():
                delta_rows.append(
                    {
                        "mode": mode_label,
                        "baseline_model": effective_baseline_model,
                        "split": str(split_name),
                        "metric": str(metric_name),
                        "left_model": str(left_model),
                        "right_model": str(right_model),
                        "delta_name": f"{left_model} - {right_model}",
                        "point_estimate": float(delta.point_estimate),
                        "ci_low": float(delta.ci_low),
                        "ci_high": float(delta.ci_high),
                        "valid_resamples": int(delta.valid_resamples),
                        "total_resamples": int(delta.total_resamples),
                        "left_model_n_seeds": int(payload.get("n_model_seeds", {}).get(left_model, 1)),
                        "right_model_n_seeds": int(payload.get("n_model_seeds", {}).get(right_model, 1)),
                    }
                )

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["ci_95"] = model_df.apply(
            lambda row: f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
            axis=1,
        )

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df["ci_95"] = delta_df.apply(
            lambda row: f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
            axis=1,
        )

    return model_df, delta_df, common_n_seeds
