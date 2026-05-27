#!/usr/bin/env python3
import argparse
from pathlib import Path

import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


from omegaconf import OmegaConf

from automation import (
    apply_exp_config_overrides,
    adjust_parallel_limits,
    create_experiment_workspace,
    dump_json,
    expand_grid_points,
    execute_tasks,
    load_exp_config,
    normalize_run_name,
    parse_optional_csv,
    parse_set_overrides,
    resolve_point_seeds,
    set_key,
    try_load_summary,
)


def _expand_grid_to_points(exp_cfg: dict):
    """
    Expand exp config key `grid` into a normalized points list for etas grids.

    Supported schema:
      grid:
        name_prefix: etas_ab                # optional
        common:                             # optional
          seed: 0 | seeds: [0,1,2]          # optional
          overrides: {...}                  # optional common overrides
        points:                             # required non-empty list
          - name: p1                        # optional
            overrides: {...}                # optional
            # point-level extra keys (not reserved) are treated as overrides shorthand
    """
    reserved_keys = (
        "name",
        "seed",
        "seeds",
        "overrides",
    )
    forward_keys = (
        "seed",
        "seeds",
    )
    return expand_grid_points(
        exp_cfg,
        forward_keys=forward_keys,
        reserved_keys=reserved_keys,
        default_name_prefix="grid",
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run etas/etas_zhuang in two modes: "
            "matrix (seeds) or "
            "grid (expand grid.points then run)."
        )
    )
    parser.add_argument("--model", type=str, default="etas")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help=(
            "Path to external experiment config (.yaml/.yml/.json). "
            "Supports matrix keys (seeds) or grid.points."
        ),
    )
    parser.add_argument("--base_config", type=str, default="config/etas.yaml")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma separated seeds, e.g. 0,1,2",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra dotted config override, e.g. --set learning_rate=5e-3",
    )
    parser.add_argument(
        "--run_test",
        action="store_true",
        help="Also run test after train for each run.",
    )
    parser.add_argument(
        "--ckpt_select",
        type=str,
        default="best",
        help="Checkpoint for optional test mode: best, last, both, or comma list (e.g. best,last).",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip train command and only use existing run dirs (rarely needed).",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="experiments",
        help="Root folder where a new experiment folder will be created.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional experiment folder name under exp_root.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Max number of concurrent runs. 1 means serial.",
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=1,
        help="How many concurrent jobs are allowed on each GPU id.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Optional GPU list for parallel scheduling, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop remaining tasks once any task fails.",
    )
    parser.add_argument(
        "--resume_exp",
        action="store_true",
        help="Resume existing experiment folder if exp_name already exists.",
    )
    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="Skip runs that already have successful train/test results in summary.json.",
    )
    return parser


def _override_args_from_exp_config(args, exp_cfg: dict):
    direct_key_map = {
        "model": "model",
        "base_config": "base_config",
        "run_test": "run_test",
        "ckpt_select": "ckpt_select",
        "skip_train": "skip_train",
        "exp_root": "exp_root",
        "exp_name": "exp_name",
        "max_parallel": "max_parallel",
        "jobs_per_gpu": "jobs_per_gpu",
        "stop_on_error": "stop_on_error",
        "resume_exp": "resume_exp",
        "skip_done": "skip_done",
    }
    csv_like_keys = {
        "seeds": "seeds",
        "gpu_ids": "gpu_ids",
    }
    mapping_like_keys = {}
    return apply_exp_config_overrides(args, exp_cfg, direct_key_map, csv_like_keys, mapping_like_keys)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    exp_cfg = {}
    exp_cfg_path = None
    if args.exp_config:
        exp_cfg_path = Path(args.exp_config).resolve()
        exp_cfg = load_exp_config(exp_cfg_path)
        args = _override_args_from_exp_config(args, exp_cfg)

    seeds = parse_optional_csv(args.seeds, int)
    if not seeds:
        raise ValueError("No seed values found.")

    grid_points = _expand_grid_to_points(exp_cfg) if exp_cfg else None

    extra_overrides = parse_set_overrides(args.set)

    workspace = create_experiment_workspace(
        current_file=__file__,
        base_config=args.base_config,
        exp_root=args.exp_root,
        exp_name=args.exp_name,
        default_name_prefix="etas_grid",
        exp_cfg_path=exp_cfg_path,
        resume_exp=bool(args.resume_exp),
        base_snapshot_prefix="base_",
    )
    existing_summary, existing_by_run = try_load_summary(workspace.exp_dir / "summary.json")

    gpu_ids = parse_optional_csv(args.gpu_ids, int)
    max_parallel, jobs_per_gpu = adjust_parallel_limits(
        max_parallel_raw=args.max_parallel,
        jobs_per_gpu_raw=args.jobs_per_gpu,
        gpu_ids=gpu_ids,
    )

    mode = "grid" if grid_points is not None else "matrix"
    plan = {
        "model": args.model,
        "base_config": str(workspace.base_config_path),
        "base_config_snapshot": str(workspace.base_config_snapshot_path),
        "exp_config": str(exp_cfg_path) if exp_cfg_path is not None else None,
        "exp_config_snapshot": (
            str(workspace.exp_config_snapshot_path)
            if workspace.exp_config_snapshot_path is not None
            else None
        ),
        "mode": mode,
        "points": (
            [
                {
                    "name": point.get("name"),
                    "seed": point.get("seed"),
                    "seeds": point.get("seeds"),
                }
                for point in grid_points
            ]
            if grid_points is not None
            else None
        ),
        "seeds": seeds,
        "max_parallel": max_parallel,
        "gpu_ids": gpu_ids,
        "jobs_per_gpu": jobs_per_gpu,
        "stop_on_error": bool(args.stop_on_error),
        "resume_exp": bool(args.resume_exp),
        "skip_done": bool(args.skip_done),
        "extra_overrides": extra_overrides,
    }
    dump_json(workspace.exp_dir / "plan.json", plan)

    tasks = []
    if grid_points is None:
        for seed in seeds:
            variant_name = normalize_run_name(f"seed_{int(seed)}", fallback=f"seed_{int(seed)}")
            run_dir = workspace.runs_dir / variant_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(str(workspace.base_config_path))
            cfg.seed = int(seed)

            for key, value in extra_overrides.items():
                set_key(cfg, key, value)

            cfg_path = run_dir / "config_input.yaml"
            OmegaConf.save(cfg, str(cfg_path))

            skip_reason = None
            if args.skip_done:
                prev = existing_by_run.get(variant_name)
                if isinstance(prev, dict):
                    train_ok = prev.get("train_returncode") in (None, 0)
                    test_ok = prev.get("test_returncode") in (None, 0)
                    if train_ok and test_ok:
                        skip_reason = "summary_done"

            task = {
                "run": variant_name,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "cfg_path": str(cfg_path),
            }
            if skip_reason is None:
                tasks.append(task)
    else:
        for point in grid_points:
            point_name = str(point.get("name") or "grid")
            point_seeds = resolve_point_seeds(point, default_seeds=seeds)
            if not point_seeds:
                raise ValueError(f"grid point {point_name} resolved empty seeds.")

            point_overrides = point.get("overrides", {})
            if point_overrides is None:
                point_overrides = {}
            if not isinstance(point_overrides, dict):
                raise ValueError(f"grid point {point_name} overrides must be mapping/object.")

            for seed in point_seeds:
                variant_name = normalize_run_name(
                    f"{point_name}_seed_{int(seed)}",
                    fallback=f"grid_seed_{int(seed)}",
                )
                run_dir = workspace.runs_dir / variant_name
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = OmegaConf.load(str(workspace.base_config_path))
                cfg.seed = int(seed)

                for key, value in extra_overrides.items():
                    set_key(cfg, key, value)
                for key, value in point_overrides.items():
                    set_key(cfg, str(key), value)

                cfg_path = run_dir / "config_input.yaml"
                OmegaConf.save(cfg, str(cfg_path))

                skip_reason = None
                if args.skip_done:
                    prev = existing_by_run.get(variant_name)
                    if isinstance(prev, dict):
                        train_ok = prev.get("train_returncode") in (None, 0)
                        test_ok = prev.get("test_returncode") in (None, 0)
                        if train_ok and test_ok:
                            skip_reason = "summary_done"

                task = {
                    "run": variant_name,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "cfg_path": str(cfg_path),
                }
                if skip_reason is None:
                    tasks.append(task)

    skipped_count = 0
    if args.skip_done:
        if grid_points is None:
            total_planned = len(seeds)
        else:
            total_planned = 0
            for point in grid_points:
                if not isinstance(point, dict):
                    continue
                point_seeds = resolve_point_seeds(point, default_seeds=seeds)
                total_planned += len(point_seeds)
        skipped_count = max(0, total_planned - len(tasks))
        print(f"[INFO] skip_done enabled: {skipped_count} skipped, {len(tasks)} to run.")

    if not tasks:
        print("[INFO] No pending tasks to run.")
        return

    execute_tasks(
        tasks=tasks,
        args=args,
        repo_root=workspace.repo_root,
        exp_dir=workspace.exp_dir,
        gpu_ids=gpu_ids,
        max_parallel=max_parallel,
        jobs_per_gpu=jobs_per_gpu,
        stop_on_error=bool(args.stop_on_error),
        initial_summary=existing_summary,
    )

    print(f"\nDone. Experiment folder: {workspace.exp_dir}")
    print(f"Runs folder: {workspace.runs_dir}")
    print("Summary file:", workspace.exp_dir / "summary.json")


if __name__ == "__main__":
    main()
