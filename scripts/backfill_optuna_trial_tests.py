#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from optuna.trial import TrialState

from config import config_loader
from src.cli.data_factory import get_model_and_data
from src.cli.optuna_workflow import (
    cfg_to_dict,
    _resolve_profile_cfg,
    _resolve_search_space,
    _resolve_trial_test_ckpt_selects,
    _resolve_trial_test_top_k,
    _run_trial_auto_tests,
    set_seed,
)
import src.train.config_setup as config_setup


def _load_profile_summary(run_root: Path, profile: str):
    summary_path = run_root / "runs" / profile / f"optuna_summary_{profile}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f), summary_path


def _select_trials_from_summary(summary: dict, top_k: int):
    direction = str(summary.get("direction", "StudyDirection.MINIMIZE"))
    is_minimize = "MINIMIZE" in direction.upper()

    selected = []
    best_trial = summary.get("best_trial")
    if isinstance(best_trial, dict):
        selected.append(best_trial)

    for row in summary.get("top_trials", []):
        if not isinstance(row, dict):
            continue
        selected.append(row)

    uniq = {}
    for row in selected:
        num = row.get("number")
        value = row.get("value")
        if num is None or value is None:
            continue
        uniq[int(num)] = row

    items = list(uniq.values())
    items.sort(key=lambda x: float(x["value"]), reverse=not is_minimize)
    return items[: max(1, int(top_k))]


def _build_trial_args(args_base, optuna_cfg, profile_name, trial_number, sampled_params):
    args_trial = OmegaConf.create(OmegaConf.to_container(args_base, resolve=True))
    profile_cfg = _resolve_profile_cfg(optuna_cfg, profile_name)
    if "t_elaps_mode" in profile_cfg:
        args_trial.t_elaps_mode = profile_cfg["t_elaps_mode"]
    args_trial.epochs = int(optuna_cfg.get("epochs", getattr(args_trial, "epochs", 120)))
    trial_seed_base = int(optuna_cfg.get("trial_seed_base", getattr(args_trial, "seed", 0)))
    args_trial.seed = int(trial_seed_base + trial_number)
    set_seed(
        args_trial.seed,
        deterministic=bool(getattr(args_trial, "deterministic", True)),
        deterministic_warn_only=bool(getattr(args_trial, "deterministic_warn_only", False)),
        use_deterministic_algorithms=bool(getattr(args_trial, "use_deterministic_algorithms", False)),
    )
    for name, value in sampled_params.items():
        node = args_trial
        if "." not in name:
            setattr(node, name, value)
            continue
        parts = name.split(".")
        for part in parts[:-1]:
            child = getattr(node, part, None)
            if child is None:
                child = OmegaConf.create({})
                setattr(node, part, child)
            node = child
        setattr(node, parts[-1], value)
    return args_trial


def backfill_profile(run_root: Path, args_base, optuna_cfg, profile: str, top_k: int, ckpt_selects: list[str], device: torch.device):
    summary, summary_path = _load_profile_summary(run_root, profile)
    chosen = _select_trials_from_summary(summary, top_k)
    if not chosen:
        print(f"[{profile}] no completed trials in {summary_path}")
        return

    profile_cfg = _resolve_profile_cfg(optuna_cfg, profile)
    search_space = _resolve_search_space(optuna_cfg, profile_cfg)

    print(f"[{profile}] selected trials: {[row['number'] for row in chosen]}")
    for row in chosen:
        trial_number = int(row["number"])
        trial_dir = run_root / "runs" / profile / "optuna_trials" / f"trial_{trial_number:04d}_{profile}"
        if not trial_dir.exists():
            print(f"  - skip trial {trial_number}: missing {trial_dir}")
            continue
        sampled = {}
        params = cfg_to_dict(row.get("params", {}))
        for key in search_space.keys():
            if key in params:
                sampled[key] = params[key]

        args_trial = _build_trial_args(args_base, optuna_cfg, profile, trial_number, sampled)
        args_trial.save_dir = str(trial_dir)

        train_step, _df, train_loader, val_loader, test_loader = get_model_and_data(
            args_trial,
            f"data/{args_trial.dataset}",
            device,
        )
        model, criterion, optimizer, scheduler, args_trial = config_setup.setup_config(
            args_trial,
            device,
            train_dataloader=train_loader,
            checkpoint=None,
            restore_weights=False,
        )
        del optimizer, scheduler

        trial_test_metrics = _run_trial_auto_tests(
            train_step=train_step,
            args_trial=args_trial,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            trial_dir=str(trial_dir),
            ckpt_selects=ckpt_selects,
        )

        metrics_optuna_path = trial_dir / "metrics_optuna_val.json"
        payload = {}
        if metrics_optuna_path.exists():
            try:
                with open(metrics_optuna_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        payload["trial_test_ckpt_selects"] = ckpt_selects
        payload["trial_test_metrics"] = trial_test_metrics
        with open(metrics_optuna_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  - done trial {trial_number}: wrote tests {ckpt_selects}")


def main():
    parser = argparse.ArgumentParser(description="Backfill optuna trial tests/plots on an existing run root without retraining.")
    parser.add_argument("--run_root", type=str, required=True, help="Existing run root, e.g. experiments/.../run_YYYYMMDD-HHMMSS")
    parser.add_argument("--config", type=str, default="config/reg_mixer_attnpl_t.yaml")
    parser.add_argument("--model", type=str, default="reg_mixer_attnpl_t")
    parser.add_argument("--profiles", type=str, default="auto", help="Comma-separated profiles or 'auto' from profile_optuna_summary.json")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k trials to backfill per profile.")
    parser.add_argument("--ckpt_selects", type=str, default=None, help="Comma-separated checkpoint selectors, e.g. best,last")
    parser.add_argument("--cuda_id", type=str, default=None, help="Override cuda id")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root not found: {run_root}")

    cfg = config_loader.load_args_from_yaml(args.config)
    cfg.model = args.model.lower()
    if args.cuda_id is not None:
        cfg.cuda_id = str(args.cuda_id)
    cfg.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.cuda_id}" if cfg.cuda else "cpu")

    optuna_cfg = cfg_to_dict(getattr(cfg, "optuna", {}))
    args_cli = argparse.Namespace(
        optuna_trial_test_ckpt_selects=args.ckpt_selects,
        optuna_trial_test_top_k=args.top_k,
    )
    ckpt_selects = _resolve_trial_test_ckpt_selects(args_cli, optuna_cfg)
    if not ckpt_selects:
        ckpt_selects = ["best", "last"]
    top_k = _resolve_trial_test_top_k(args_cli, optuna_cfg)
    if top_k <= 0:
        top_k = 5

    if args.profiles.strip().lower() == "auto":
        summary_path = run_root / "profile_optuna_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            profiles = list(summary.get("profiles", []))
        else:
            profiles = []
            runs_dir = run_root / "runs"
            if runs_dir.exists():
                profiles = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    else:
        profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]

    if not profiles:
        raise ValueError("No profiles resolved. Please pass --profiles explicitly.")

    print(f"run_root={run_root}")
    print(f"profiles={profiles}")
    print(f"top_k={top_k}")
    print(f"ckpt_selects={ckpt_selects}")
    for profile in profiles:
        backfill_profile(run_root, cfg, optuna_cfg, profile, top_k, ckpt_selects, device)


if __name__ == "__main__":
    main()

