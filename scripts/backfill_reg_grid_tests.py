#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_ckpt_selects(raw: str) -> list[str]:
    text = str(raw).strip().lower()
    if text == "both":
        return ["best", "last"]
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    valid = {"best", "last"}
    for item in items:
        if item not in valid:
            raise ValueError(f"Unsupported ckpt_select: {item!r}. Use best, last, both, or comma list.")
    ordered = []
    seen = set()
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered or ["last"]


def iter_run_dirs(exp_dir: Path):
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def resolve_model(cli_model: str | None, cfg_path: Path) -> str:
    if cli_model:
        return cli_model
    cfg = OmegaConf.load(str(cfg_path))
    model = str(getattr(cfg, "model", "")).strip()
    if not model:
        raise ValueError(f"Cannot resolve model from config: {cfg_path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Backfill test metrics for existing regression grid runs without retraining."
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing runs/summary.")
    parser.add_argument("--model", type=str, default=None, help="Optional model override.")
    parser.add_argument(
        "--ckpt_selects",
        type=str,
        default="last",
        help="best, last, both, or comma list (e.g. best,last). Default: last.",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        default=True,
        help="Only run tests when metrics_test_<ckpt>_1.json is missing (default true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if metrics file already exists.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_selects = parse_ckpt_selects(args.ckpt_selects)
    only_missing = bool(args.only_missing and not args.force)

    runs = iter_run_dirs(exp_dir)
    total_cmd = 0
    done_cmd = 0
    skip_cmd = 0
    fail_cmd = 0

    print(f"[INFO] exp_dir={exp_dir}")
    print(f"[INFO] runs={len(runs)} ckpt_selects={ckpt_selects} only_missing={only_missing}")

    for run_dir in runs:
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"[WARN] skip {run_dir.name}: missing config.yaml")
            continue

        model = resolve_model(args.model, cfg_path)
        for ckpt_select in ckpt_selects:
            ckpt_path = run_dir / f"{ckpt_select}_model_1.pth"
            metrics_path = run_dir / f"metrics_test_{ckpt_select}_1.json"

            if not ckpt_path.exists():
                print(f"[WARN] skip {run_dir.name} ({ckpt_select}): missing {ckpt_path.name}")
                skip_cmd += 1
                continue
            if only_missing and metrics_path.exists():
                skip_cmd += 1
                continue

            cmd = [
                sys.executable,
                "main.py",
                "--model",
                model,
                "--mode",
                "test",
                "--config",
                str(cfg_path),
                "--checkpoint_dir",
                str(run_dir),
                "--ckpt_select",
                ckpt_select,
            ]
            total_cmd += 1
            print("[CMD]", " ".join(cmd))
            if args.dry_run:
                continue
            ret = subprocess.run(cmd, cwd=str(repo_root))
            if ret.returncode == 0:
                done_cmd += 1
            else:
                fail_cmd += 1
                print(f"[ERROR] failed {run_dir.name} ({ckpt_select}), returncode={ret.returncode}")

    print(
        f"[DONE] planned={total_cmd}, success={done_cmd}, failed={fail_cmd}, skipped={skip_cmd}, dry_run={args.dry_run}"
    )
    if fail_cmd > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

