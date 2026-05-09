import argparse
import json
import logging
import os
from dataclasses import dataclass

import torch

from config import config_loader
from src.models.builders import ModelBuilder
from src.utils.file_utils import create_save_dir, find_latest_model_path
from src.utils.logging_utils import setup_logging
from src.utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    args_cli: argparse.Namespace
    args: object
    device: torch.device


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test", "optuna"], default="train", help="Run mode: train or test")
    parser.add_argument("--model", type=str, choices=ModelBuilder.list_available(), required=True, help="Model name")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to load checkpoint for test mode")
    parser.add_argument("--trial_index", type=int, default=1, help="Index of the trial for optuna")
    parser.add_argument(
        "--ckpt_select",
        type=str,
        choices=["best", "last", "epoch"],
        default="best",
        help="Which checkpoint to use in test mode (best, last, or epoch)",
    )
    parser.add_argument("--ckpt_epoch", type=int, default=None, help="Epoch number to load when --ckpt_select epoch")
    parser.add_argument("--threshold", type=float, default=None, help="If provided, use this threshold for classification test (overrides checkpoint)")
    parser.add_argument("--no_val_threshold", action="store_true", default=False, help="Do not use threshold stored in checkpoint val_metrics")
    parser.add_argument("--optuna_trials", type=int, default=None, help="Override number of Optuna trials")
    parser.add_argument("--optuna_profile", type=str, default=None, help="Optuna profile name (e.g. global/window)")
    parser.add_argument(
        "--optuna_all_profiles",
        action="store_true",
        default=False,
        help="Run Optuna for all configured profiles sequentially",
    )
    parser.add_argument("--optuna_storage", type=str, default=None, help="Optuna storage URI/path; e.g., sqlite:///tmp/study.db")
    parser.add_argument("--optuna_study_name", type=str, default=None, help="Override Optuna study name")
    parser.add_argument("--optuna_sampler_seed", type=int, default=None, help="Override Optuna sampler seed")
    return parser


def parse_and_prepare_runtime(argv=None):
    parser = build_parser()
    args_cli = parser.parse_args(argv)
    if args_cli.config is None:
        args_cli.config = f"config/{args_cli.model}.yaml"

    args = config_loader.load_args_from_yaml(args_cli.config)
    assert args.model.lower() == args_cli.model.lower(), "Model name in config must match command line argument"
    args.model = args.model.lower()
    torch.autograd.set_detect_anomaly(bool(getattr(args, "detect_anomaly", False)))

    seed = getattr(args, "seed", 0)
    set_seed(
        seed,
        deterministic=bool(getattr(args, "deterministic", True)),
        deterministic_warn_only=bool(getattr(args, "deterministic_warn_only", False)),
        use_deterministic_algorithms=bool(getattr(args, "use_deterministic_algorithms", False)),
    )

    if args_cli.mode in {"train", "optuna"}:
        args.save_dir = args_cli.checkpoint_dir or create_save_dir(base_dir="checkpoints", model_name=args.model)
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = args_cli.checkpoint_dir or find_latest_model_path(args_cli.model)

    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_id}" if args.cuda else "cpu")
    setup_logging(level=getattr(args, "log_level", None), log_file=os.path.join(args.save_dir, "run.log"))
    return RuntimeContext(args_cli=args_cli, args=args, device=device)


def build_test_checkpoint_path(args_cli, save_dir):
    if args_cli.ckpt_select == "epoch":
        if args_cli.ckpt_epoch is None:
            raise ValueError('When --ckpt_select is "epoch", --ckpt_epoch must be provided')
        return f"{save_dir}/epoch_{args_cli.ckpt_epoch}_model_{args_cli.trial_index}.pth"
    return f"{save_dir}/{args_cli.ckpt_select}_model_{args_cli.trial_index}.pth"


def resolve_test_threshold(args_cli, checkpoint):
    cli_thresh = getattr(args_cli, "threshold", None)
    if cli_thresh is not None:
        LOGGER.info("Using threshold %s from CLI argument for testing.", cli_thresh)
        return cli_thresh

    if getattr(args_cli, "no_val_threshold", False):
        return None

    if isinstance(checkpoint, dict):
        val_metrics = checkpoint.get("val_metrics", {})
        if isinstance(val_metrics, dict) and "threshold" in val_metrics:
            threshold = val_metrics["threshold"]
            LOGGER.info("Using threshold %s from checkpoint val_metrics for testing.", threshold)
            return threshold
    return None


def build_metrics_filename(args_cli):
    if args_cli.ckpt_select == "epoch":
        return f"metrics_test_epoch_{args_cli.ckpt_epoch}_{args_cli.trial_index}.json"
    return f"metrics_test_{args_cli.ckpt_select}_{args_cli.trial_index}.json"


def save_metrics(save_dir, metrics_filename, metrics):
    metrics_path = os.path.join(save_dir, metrics_filename)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Saved test metrics to %s", metrics_path)
    LOGGER.info("Metrics: %s", metrics)

