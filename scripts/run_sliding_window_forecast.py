#!/usr/bin/env python3
"""Run the sliding-window forecast section from the triplet multi-bg notebook.

Defaults mirror:
`notebooks/forecasting_induced_eq_triplet.ipynb`

Example:
    python scripts/run_sliding_window_forecast.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")

PLOT_COLORS = {
    "true": "#1f77b4",
    "mean": "#ff7f0e",
    "pi": "#4c78a8",
    "err": "#d62728",
    "covered": "#2ca02c",
    "below": "#e45756",
    "above": "#9467bd",
}

FIGURE_STEMS = (
    "forecast_counts_over_time",
    "forecast_error_over_time",
    "forecast_pi_coverage_over_time",
    "obs_vs_forecast_scatter",
    "forecast_mag_max_over_time",
    "obs_vs_forecast_mag_max_scatter",
)

DEFAULT_METRICS_FILENAME = "sliding_window_eval_metrics.json"
DEFAULT_CHECKPOINT_DIR = "checkpoints/etas_20260624-120141"
DEFAULT_CHECKPOINT_FILE = "best_model_1.pth"
DEFAULT_FORECAST_B_SAMPLING = "model"
DEFAULT_UPDATER_NAME = "bayesian_gr"


def resolve_project_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing 'src' directory.")


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.catalogs as _catalogs  # noqa: E402,F401
from src.train.config_setup import load_and_prepare_model  # noqa: E402
from src.utils.forecast_eval import (  # noqa: E402
    apply_publication_style,
    evaluate_sliding_window_forecast_plots,
)
from src.utils.runtime_utils import unwrap_compiled_model  # noqa: E402
from src.utils.tpp_experiments import load_tpp_catalog  # noqa: E402
from src.utils.utils import set_seed  # noqa: E402


def parse_predict_b(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"auto", "none", "null"}:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "--predict-b must be one of: auto, true, false."
    )


def resolve_path(path_value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window earthquake forecasts and save plots/metrics."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Checkpoint directory. Relative paths are resolved from project root. "
            f"Default: {DEFAULT_CHECKPOINT_DIR}."
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help="Checkpoint filename inside --checkpoint-dir.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name. Defaults to the checkpoint args.dataset value.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for cache and forecast plots. Defaults to checkpoint dir.",
    )
    parser.add_argument(
        "--sequence-idx",
        type=int,
        default=0,
        help="Sequence index from catalog_ds.sequences.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Sliding forecast duration. Notebook default: 1.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Sliding step size. Notebook default: 1.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=2,
        default=(2.5, 97.5),
        metavar=("LOW", "HIGH"),
        help="Prediction interval percentiles. Notebook default: 2.5 97.5.",
    )
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=1000,
        help="Number of samples per sliding window. Notebook default: 1000.",
    )
    parser.add_argument(
        "--view-mode",
        default="zoomed",
        choices=("zoomed", "full", "auto"),
        help="Plot view mode. Notebook default: zoomed.",
    )
    parser.add_argument(
        "--cache-filename",
        default=None,
        help=(
            "Sliding-window cache filename. Defaults to the notebook-style "
            "name derived from --forecast-b-sampling and --updater-name."
        ),
    )
    parser.add_argument(
        "--forecast-b-sampling",
        default=DEFAULT_FORECAST_B_SAMPLING,
        choices=("model", "updater"),
        help=(
            "B-value sampling mode used only for the default cache filename. "
            f"Notebook default: {DEFAULT_FORECAST_B_SAMPLING}."
        ),
    )
    parser.add_argument(
        "--updater-name",
        default=DEFAULT_UPDATER_NAME,
        help=(
            "Updater name used only when --forecast-b-sampling=updater for "
            f"the default cache filename. Notebook default: {DEFAULT_UPDATER_NAME}."
        ),
    )
    parser.add_argument(
        "--metrics-filename",
        default=DEFAULT_METRICS_FILENAME,
        help=(
            "Metrics JSON filename under --output-dir. "
            f"Default: {DEFAULT_METRICS_FILENAME}."
        ),
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Do not save sliding-window evaluation metrics JSON.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save forecast plots. Use --no-save-plots for faster metrics-only runs.",
    )
    parser.add_argument(
        "--load-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load a compatible sliding cache when available.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing cache and recompute forecasts.",
    )
    parser.add_argument(
        "--predict-b",
        type=parse_predict_b,
        default=None,
        metavar="{auto,true,false}",
        help="Optional predict_b mode passed to model.sample. Default: auto.",
    )
    parser.add_argument(
        "--model-mm",
        type=float,
        default=None,
        help="Override model.M_m. Defaults to no override.",
    )
    parser.add_argument(
        "--no-model-mm",
        action="store_true",
        help="Do not apply --model-mm.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint loading compile mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Notebook default: 0.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Set CUDA_LAUNCH_BLOCKING=1. Defaults to off because it can make "
            "GPU sampling much slower; enable only for debugging CUDA errors."
        ),
    )
    return parser


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_primary_model(args: argparse.Namespace, device: torch.device):
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    checkpoint_file = args.checkpoint_file
    checkpoint_path = checkpoint_dir / checkpoint_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    use_torch_compile = False if args.use_torch_compile is None else bool(args.use_torch_compile)
    model, model_args = load_and_prepare_model(
        checkpoint_path=checkpoint_path,
        device=device,
        compile=use_torch_compile,
    )
    model = unwrap_compiled_model(model)

    if args.no_model_mm:
        model_mm = None
    elif args.model_mm is not None:
        model_mm = float(args.model_mm)
    else:
        model_mm = None

    if model_mm is not None and hasattr(model, "M_m"):
        model.M_m = torch.tensor(float(model_mm), device=device)

    return model, model_args, checkpoint_path


def resolve_predict_b(args: argparse.Namespace, model_args: Any) -> bool | None:
    if args.predict_b is not None:
        return args.predict_b
    return bool(getattr(model_args, "predict_b", False))


def load_catalog(args: argparse.Namespace, model_args: Any):
    dataset_name = args.dataset or getattr(model_args, "dataset", None)
    if dataset_name is None:
        raise ValueError("Dataset name is unavailable; pass --dataset explicitly.")

    catalog_cfg = getattr(model_args, "catalog_cfg", {})
    data_root = resolve_path(args.data_root)
    catalog_ds, registry_name, _ = load_tpp_catalog(
        str(dataset_name),
        base_dir=data_root / str(dataset_name),
        catalog_cfg=catalog_cfg,
    )
    return catalog_ds, registry_name, str(dataset_name)


def select_sequence(catalog_ds: Any, sequence_idx: int):
    full_seq = catalog_ds.full_sequence
    sequences = catalog_ds.sequences if hasattr(catalog_ds, "sequences") else [full_seq]
    if sequence_idx < 0 or sequence_idx >= len(sequences):
        raise IndexError(
            f"--sequence-idx {sequence_idx} out of range for {len(sequences)} sequences."
        )
    return sequences[sequence_idx]


def print_metrics(sliding_result: dict[str, Any]) -> None:
    print(f"Sliding cache path: {sliding_result.get('sliding_cache_path', 'N/A')}")
    print(f"Loaded from cache: {bool(sliding_result.get('sliding_loaded_from_cache', False))}")

    if sliding_result.get("status") == "empty":
        print(sliding_result["message"])
        return

    lp_nb = sliding_result.get("lp_nb")
    crps = sliding_result.get("crps")
    w95 = sliding_result.get("w95")
    mag_max_mae = sliding_result.get("mag_max_mae")
    lp_nb_display = "N/A" if lp_nb is None or not np.isfinite(lp_nb) else f"{lp_nb:.4f}"
    crps_display = "N/A" if crps is None or not np.isfinite(crps) else f"{crps:.4f}"
    mag_display = (
        "N/A"
        if mag_max_mae is None or not np.isfinite(mag_max_mae)
        else f"{mag_max_mae:.4f}"
    )

    print(f"acceptance rate: {sliding_result['coverage']:.2%}")
    print(
        "MAE: "
        f"{sliding_result['mae']:.4f} | "
        f"RMSE: {sliding_result['rmse']:.4f} | "
        f"CRPS: {crps_display} | "
        f"LP_NB: {lp_nb_display} | "
        f"W95: {w95:.4f} | "
        f"MagMax MAE: {mag_display}"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_metrics_payload(
    sliding_result: dict[str, Any],
    *,
    args: argparse.Namespace,
    cache_filename: str,
    predict_b: bool | None,
    checkpoint_path: Path,
    output_dir: Path,
    registry_name: str,
    dataset_name: str,
    device: torch.device,
) -> dict[str, Any]:
    status = sliding_result.get("status")
    payload: dict[str, Any] = {
        "status": status,
        "settings": {
            "duration": float(args.duration),
            "step": float(args.step),
            "quantiles": [float(item) for item in args.quantiles],
            "samples_per_batch": int(args.samples_per_batch),
            "predict_b": predict_b,
            "forecast_b_sampling": args.forecast_b_sampling,
            "updater_name": args.updater_name,
            "view_mode": args.view_mode,
            "cache_filename": cache_filename,
            "load_sliding_cache": bool(args.load_cache),
            "force_recompute_sliding": bool(args.force_recompute),
        },
        "metadata": {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "dataset": dataset_name,
            "registry_name": registry_name,
            "sequence_idx": int(args.sequence_idx),
            "device": str(device),
            "seed": int(args.seed),
        },
        "sliding_cache_path": sliding_result.get("sliding_cache_path"),
        "sliding_loaded_from_cache": bool(
            sliding_result.get("sliding_loaded_from_cache", False)
        ),
    }

    if status == "empty":
        payload["message"] = sliding_result.get("message")
        payload["num_windows"] = 0
        return json_safe(payload)

    metric_keys = (
        "coverage",
        "mae",
        "rmse",
        "crps",
        "w95",
        "lp_nb",
        "mag_max_mae",
    )
    for key in metric_keys:
        value = sliding_result.get(key)
        if value is not None:
            payload[key] = value

    t_forecast_list = np.asarray(sliding_result.get("t_forecast_list", []))
    counts_list = np.asarray(sliding_result.get("counts_list", []))
    q_list = np.asarray(sliding_result.get("q_list", []))
    mean_list = np.asarray(sliding_result.get("mean_list", []))

    payload["num_windows"] = int(len(t_forecast_list))
    payload["windows"] = []
    if (
        len(t_forecast_list)
        and counts_list.shape[0] == len(t_forecast_list)
        and mean_list.shape[0] == len(t_forecast_list)
        and q_list.ndim == 2
        and q_list.shape[0] == len(t_forecast_list)
        and q_list.shape[1] >= 2
    ):
        payload["windows"] = [
            {
                "t_forecast": float(t_forecast_list[index]),
                "count": int(counts_list[index]),
                "forecast_mean": float(mean_list[index]),
                "q_low": float(q_list[index, 0]),
                "q_high": float(q_list[index, 1]),
                "covered": bool(
                    q_list[index, 0] <= counts_list[index] <= q_list[index, 1]
                ),
                "error": float(counts_list[index] - mean_list[index]),
                "abs_error": float(abs(counts_list[index] - mean_list[index])),
            }
            for index in range(len(t_forecast_list))
        ]

    return json_safe(payload)


def save_metrics_payload(payload: dict[str, Any], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, allow_nan=True)
        file_obj.write("\n")
    print(f"Saved sliding-window metrics: {metrics_path}")


def build_default_cache_filename(args: argparse.Namespace) -> str:
    updater_cache_tag = (
        args.updater_name if args.forecast_b_sampling == "updater" else "model"
    )
    return f"sliding_window_cache_{args.forecast_b_sampling}_{updater_cache_tag}.npz"


def print_available_figures(output_dir: Path) -> None:
    existing_paths = [
        output_dir / f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("pdf", "png")
    ]
    existing_paths = [path for path in existing_paths if path.exists()]
    if not existing_paths:
        print("No forecast plot files found.")
        return

    print("Saved/available forecast figures:")
    for path in existing_paths:
        print(f"  - {path}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    apply_publication_style()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEVICE: {device}")

    model, model_args, checkpoint_path = load_primary_model(args, device)
    checkpoint_dir = checkpoint_path.parent
    output_dir = resolve_path(args.output_dir) if args.output_dir else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    catalog_ds, registry_name, dataset_name = load_catalog(args, model_args)
    seq = select_sequence(catalog_ds, args.sequence_idx)

    print(f"Using catalog registry name: {registry_name}")
    print(f"Dataset: {dataset_name}")
    print(f"SEQUENCE_IDX: {args.sequence_idx} | seq length: {len(seq)}")
    print(
        "Sliding settings: "
        f"duration={args.duration}, step={args.step}, "
        f"quantiles={tuple(args.quantiles)}, samples_per_batch={args.samples_per_batch}, "
        f"view_mode={args.view_mode}"
    )

    predict_b = resolve_predict_b(args, model_args)
    if hasattr(model, "predict_b") and predict_b is not None:
        model.predict_b = bool(predict_b)
    print(f"Sampling uses checkpoint predict_b: {predict_b}")

    cache_filename = args.cache_filename or build_default_cache_filename(args)

    model.eval()
    sliding_result = evaluate_sliding_window_forecast_plots(
        model=model,
        seq=seq,
        device=device,
        catalog_ds=catalog_ds,
        checkpoint_dir=output_dir,
        sliding_duration=args.duration,
        sliding_step=args.step,
        sliding_quantiles=tuple(args.quantiles),
        samples_per_batch=args.samples_per_batch,
        predict_b=predict_b,
        bg_cache_seq=seq,
        sliding_view_mode=args.view_mode,
        load_sliding_cache=args.load_cache,
        force_recompute_sliding=args.force_recompute,
        sliding_cache_filename=cache_filename,
        plot_colors=PLOT_COLORS,
        save_plots=args.save_plots,
    )

    print_metrics(sliding_result)
    if not args.no_save_metrics:
        metrics_filename = Path(args.metrics_filename)
        metrics_path = (
            metrics_filename
            if metrics_filename.is_absolute()
            else output_dir / metrics_filename
        )
        metrics_payload = build_metrics_payload(
            sliding_result,
            args=args,
            cache_filename=cache_filename,
            predict_b=predict_b,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            registry_name=registry_name,
            dataset_name=dataset_name,
            device=device,
        )
        save_metrics_payload(metrics_payload, metrics_path)
    print_available_figures(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
