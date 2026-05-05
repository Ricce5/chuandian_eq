import argparse
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.train.config_setup import load_and_prepare_model
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts
from src.utils.utils import set_seed


logger = logging.getLogger("forecasting_repro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and diagnose forecasting sampling failures."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/mixer_tpp_20260205-110204/best_model_1.pth"),
        help="Checkpoint file path.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Dataset root directory (contains dataset folders).",
    )
    parser.add_argument("--t_forecast", type=float, default=13523.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--samples_per_batch", type=int, default=1000)
    parser.add_argument("--max_sample_len", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run torch.compile when loading model.",
    )
    parser.add_argument(
        "--predict_b",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diagnose_mix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Instrument conditional mixture sampling and dump invalid probability rows.",
    )
    parser.add_argument(
        "--strict_diagnose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise RuntimeError once invalid conditional probs are detected.",
    )
    parser.add_argument(
        "--model_mm",
        type=float,
        default=10.0,
        help="Override model.M_m if present; set a negative value to disable override.",
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("tmp/forecast_debug"),
        help="Where to save diagnostic snapshots.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _to_float_scalar(x: torch.Tensor) -> float | None:
    if x.numel() == 0:
        return None
    return float(x.item())


def _tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
    }
    if value.numel() == 0:
        stats.update(
            {
                "finite_count": 0,
                "nan_count": 0,
                "posinf_count": 0,
                "neginf_count": 0,
                "min": None,
                "max": None,
                "mean": None,
            }
        )
        return stats

    value = value.float()
    finite_mask = torch.isfinite(value)
    finite_vals = value[finite_mask]
    stats.update(
        {
            "finite_count": int(finite_mask.sum().item()),
            "nan_count": int(torch.isnan(value).sum().item()),
            "posinf_count": int(torch.isposinf(value).sum().item()),
            "neginf_count": int(torch.isneginf(value).sum().item()),
            "min": _to_float_scalar(finite_vals.min()) if finite_vals.numel() > 0 else None,
            "max": _to_float_scalar(finite_vals.max()) if finite_vals.numel() > 0 else None,
            "mean": _to_float_scalar(finite_vals.mean()) if finite_vals.numel() > 0 else None,
        }
    )
    return stats


def _install_mixture_probe(debug_dir: Path, strict: bool) -> None:
    from src.distributions.mixture import MixtureSameFamily

    original_sample_conditional = MixtureSameFamily.sample_conditional

    def wrapped_sample_conditional(self, lower_bound, sample_shape=torch.Size()):  # type: ignore[no-redef]
        with torch.no_grad():
            mix_probs = self.mixture_distribution.probs
            log_survival = self.component_distribution.log_survival(lower_bound)
            conditional_mix_probs_raw = mix_probs * log_survival.exp()
            probs_sum = conditional_mix_probs_raw.sum(dim=-1, keepdim=True)
            eps = torch.finfo(conditional_mix_probs_raw.dtype).eps
            conditional_mix_probs = conditional_mix_probs_raw / probs_sum.clamp_min(eps)

            invalid_probs = (~torch.isfinite(conditional_mix_probs)) | (conditional_mix_probs < 0)
            invalid_sum = (~torch.isfinite(probs_sum.squeeze(-1))) | (probs_sum.squeeze(-1) <= 0)
            invalid_rows = invalid_probs.any(dim=-1) | invalid_sum

            if invalid_rows.any():
                bad_idx = torch.nonzero(invalid_rows, as_tuple=False).flatten()
                max_rows = min(16, bad_idx.numel())
                selected_idx = bad_idx[:max_rows]
                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "bad_row_count": int(bad_idx.numel()),
                    "total_rows": int(invalid_rows.numel()),
                    "lower_bound_stats": _tensor_stats("lower_bound", lower_bound),
                    "mix_probs_stats": _tensor_stats("mix_probs", mix_probs),
                    "log_survival_stats": _tensor_stats("log_survival", log_survival),
                    "conditional_mix_probs_raw_stats": _tensor_stats(
                        "conditional_mix_probs_raw", conditional_mix_probs_raw
                    ),
                    "probs_sum_stats": _tensor_stats("probs_sum", probs_sum),
                    "conditional_mix_probs_stats": _tensor_stats(
                        "conditional_mix_probs", conditional_mix_probs
                    ),
                    "selected_bad_indices": selected_idx.detach().cpu().tolist(),
                    "selected_bad_rows_probs": conditional_mix_probs[selected_idx]
                    .detach()
                    .cpu()
                    .tolist(),
                    "selected_bad_rows_mix_probs": mix_probs[selected_idx].detach().cpu().tolist(),
                    "selected_bad_rows_log_survival": log_survival[selected_idx]
                    .detach()
                    .cpu()
                    .tolist(),
                }
                debug_dir.mkdir(parents=True, exist_ok=True)
                json_path = debug_dir / f"bad_conditional_mix_probs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)

                msg = (
                    "Detected invalid conditional mixture probabilities before Categorical sampling. "
                    f"bad_rows={int(bad_idx.numel())}/{int(invalid_rows.numel())}, snapshot={json_path}"
                )
                logger.error(msg)
                if strict:
                    raise RuntimeError(msg)

        return original_sample_conditional(self, lower_bound, sample_shape)

    MixtureSameFamily.sample_conditional = wrapped_sample_conditional  # type: ignore[assignment]


def _summarize_sequence_window(test_seq, t_forecast: float, duration: float) -> dict[str, Any]:
    past_seq = test_seq.get_subsequence(0, t_forecast)
    obs_seq = test_seq.get_subsequence(t_forecast, t_forecast + duration)
    arrival_times = past_seq.arrival_times
    if len(arrival_times) > 0:
        last_arrival = float(arrival_times[-1].item())
        time_remaining = float(past_seq.t_end - arrival_times[-1].item())
    else:
        last_arrival = None
        time_remaining = float(past_seq.t_end - past_seq.t_start)

    inter_times = past_seq.inter_times.detach().cpu()
    zero_inter_count = int((inter_times == 0).sum().item())
    tiny_inter_count = int((inter_times.abs() < 1e-12).sum().item())

    summary = {
        "t_forecast": float(t_forecast),
        "duration": float(duration),
        "past_num_events": int(len(past_seq)),
        "obs_num_events": int(len(obs_seq)),
        "past_t_end": float(past_seq.t_end),
        "past_last_arrival": last_arrival,
        "past_time_remaining": time_remaining,
        "past_zero_inter_times": zero_inter_count,
        "past_tiny_inter_times": tiny_inter_count,
    }
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", args.checkpoint_path)
    logger.info(
        "Params: t_forecast=%s duration=%s num_samples=%s samples_per_batch=%s max_sample_len=%s predict_b=%s",
        args.t_forecast,
        args.duration,
        args.num_samples,
        args.samples_per_batch,
        args.max_sample_len,
        args.predict_b,
    )

    if args.diagnose_mix:
        _install_mixture_probe(debug_dir=args.debug_dir, strict=args.strict_diagnose)
        logger.info("Mixture probe enabled (strict=%s).", args.strict_diagnose)

    model, checkpoint_args = load_and_prepare_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        compile=args.compile,
    )
    if args.model_mm >= 0 and hasattr(model, "M_m"):
        model.M_m = torch.tensor(float(args.model_mm), device=device)
        logger.info("Override model.M_m -> %.4f", args.model_mm)

    dataset_name = checkpoint_args.dataset
    catalog_ds, registry_name, _ = load_tpp_catalog(
        dataset_name,
        base_dir=args.data_root / dataset_name,
        catalog_cfg=getattr(checkpoint_args, "catalog_cfg", {}),
    )
    logger.info("Catalog: %s (dataset=%s)", registry_name, dataset_name)

    test_seq = catalog_ds.test[0]
    summary = _summarize_sequence_window(
        test_seq=test_seq,
        t_forecast=args.t_forecast,
        duration=args.duration,
    )
    logger.info("Window summary: %s", json.dumps(summary, ensure_ascii=False))

    past_seq = test_seq.get_subsequence(0, args.t_forecast)
    if device.type == "cuda":
        past_seq = past_seq.to(device)

    try:
        forecasts = sample_tpp_forecasts(
            model=model,
            past_seq=past_seq,
            duration=float(args.duration),
            num_samples=int(args.num_samples),
            samples_per_batch=int(args.samples_per_batch),
            seed=int(args.seed),
            sample_max_length=int(args.max_sample_len),
            predict_b=bool(args.predict_b),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Forecast sampling failed: %s", exc)
        logger.error("Traceback:\n%s", traceback.format_exc())
        if "device-side assert triggered" in str(exc):
            logger.error(
                "Likely root cause: conditional mixture probabilities become NaN/Inf or non-positive "
                "before Categorical sampling; inspect debug snapshots under %s.",
                args.debug_dir,
            )
        return 1

    logger.info("Forecast sampling succeeded. total=%d", len(forecasts))
    if len(forecasts) > 0:
        event_counts = [len(fc) for fc in forecasts]
        logger.info(
            "Event-count stats: min=%d max=%d mean=%.3f",
            int(np.min(event_counts)),
            int(np.max(event_counts)),
            float(np.mean(event_counts)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
