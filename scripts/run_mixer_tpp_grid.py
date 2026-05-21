#!/usr/bin/env python3
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from automation import (
    apply_encoder_load_strategy,
    apply_exp_config_overrides,
    adjust_parallel_limits,
    build_attn_load_seed_variant_name,
    create_experiment_workspace,
    dump_json,
    expand_grid_points,
    execute_tasks,
    load_exp_config,
    normalize_run_name,
    parse_load_strategies,
    parse_optional_csv,
    parse_set_overrides,
    resolve_variant_load_strategy,
    set_key,
    try_load_summary,
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run mixer_tpp in three modes: "
            "matrix (attn_layers x load_strategies x seeds), "
            "variants (explicit variants list), or "
            "grid (expand grid.points to variants then run)."
        )
    )
    parser.add_argument("--model", type=str, default="mixer_tpp")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help=(
            "Path to external experiment config (.yaml/.yml/.json). "
            "Supports matrix keys (attn_layers/load_strategies), variants list, or grid.points."
        ),
    )
    parser.add_argument("--base_config", type=str, default="config/mixer_tpp.yaml")
    parser.add_argument(
        "--attn_layers",
        type=str,
        default="1,2",
        help=(
            "Comma separated attention layer indices (0-based), e.g. 1,2. "
            "Used in matrix mode; ignored when variants/grid provides attn_layer_idx."
        ),
    )
    parser.add_argument(
        "--load_strategies",
        type=str,
        default="none:none",
        help=(
            "Comma separated name:value mapping. value options: "
            "none|input_layer0_layer1|input_layer|layer0|layer1|layer0_input|layer1_input|encoder. "
            "In variants/grid mode, point-level load_strategy can be omitted to inherit base_config preload settings."
        ),
    )
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
        help="Extra dotted config override, e.g. --set mixer_model_config.n_layer=3",
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
        "attn_layers": "attn_layers",
        "seeds": "seeds",
        "gpu_ids": "gpu_ids",
    }
    mapping_like_keys = {
        "load_strategies": "load_strategies",
    }
    return apply_exp_config_overrides(args, exp_cfg, direct_key_map, csv_like_keys, mapping_like_keys)


def _build_tasks_matrix_mode(
    *,
    runs_dir: Path,
    base_config_path: Path,
    pretrain_resume_path,
    attn_layers,
    load_strategy_map: dict,
    seeds,
    extra_overrides: dict,
    existing_by_run: dict,
    skip_done: bool,
):
    tasks = []
    for attn_layer in attn_layers:
        for load_name, load_value in load_strategy_map.items():
            for seed in seeds:
                variant_name = build_attn_load_seed_variant_name(
                    attn_layer=attn_layer,
                    load_strategy=load_name,
                    seed=seed,
                )
                run_dir = runs_dir / variant_name
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = OmegaConf.load(str(base_config_path))
                cfg.pop("optuna", None)
                if "mixer_model_config" not in cfg or cfg.mixer_model_config is None:
                    cfg.mixer_model_config = {}
                cfg.mixer_model_config.attn_layer_idx = [int(attn_layer)]
                cfg.seed = int(seed)
                apply_encoder_load_strategy(cfg, load_value, pretrain_resume_path=pretrain_resume_path)

                for key, value in extra_overrides.items():
                    set_key(cfg, key, value)

                cfg_path = run_dir / "config.yaml"
                OmegaConf.save(cfg, str(cfg_path))

                skip_reason = None
                if skip_done:
                    prev = existing_by_run.get(variant_name)
                    if isinstance(prev, dict):
                        train_ok = prev.get("train_returncode") in (None, 0)
                        test_ok = prev.get("test_returncode") in (None, 0)
                        if train_ok and test_ok:
                            skip_reason = "summary_done"

                task = {
                    "run": variant_name,
                    "attn_layer_idx": int(attn_layer),
                    "load_strategy": load_name,
                    "load_strategy_value": load_value,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "cfg_path": str(cfg_path),
                }
                if skip_reason is None:
                    tasks.append(task)
    return tasks


def _build_tasks_variant_mode(
    *,
    runs_dir: Path,
    base_config_path: Path,
    pretrain_resume_path,
    variants,
    seeds,
    load_strategy_map: dict,
    extra_overrides: dict,
    existing_by_run: dict,
    skip_done: bool,
):
    if not isinstance(variants, list):
        raise ValueError("exp config key `variants` must be a list.")

    tasks = []
    seen_names = set()
    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variants[{idx}] must be a mapping/object.")
        attn_layer_raw = variant.get("attn_layer_idx")
        attn_layer = int(attn_layer_raw) if attn_layer_raw is not None else None
        load_name, load_value = resolve_variant_load_strategy(variant.get("load_strategy"), load_strategy_map)
        raw_name = variant.get("name", f"variant_{idx + 1:02d}")
        fallback_name = f"variant_{idx + 1:02d}"
        variant_name = normalize_run_name(raw_name, fallback=fallback_name)
        if variant_name in seen_names:
            variant_name = normalize_run_name(f"{variant_name}_{idx + 1:02d}", fallback=fallback_name)
        seen_names.add(variant_name)

        overrides = variant.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"variants[{idx}].overrides must be a mapping/object.")

        variant_seeds = variant.get("seeds")
        if variant_seeds is None:
            if "seed" in variant:
                seed_values = [int(variant.get("seed"))]
            else:
                seed_values = [int(s) for s in seeds]
        elif isinstance(variant_seeds, (list, tuple)):
            seed_values = [int(s) for s in variant_seeds]
        else:
            seed_values = [int(variant_seeds)]

        for seed in seed_values:
            run_name = normalize_run_name(f"{variant_name}_seed_{seed}", fallback=f"{fallback_name}_seed_{seed}")
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(str(base_config_path))
            cfg.pop("optuna", None)
            if attn_layer is not None:
                if "mixer_model_config" not in cfg or cfg.mixer_model_config is None:
                    cfg.mixer_model_config = {}
                cfg.mixer_model_config.attn_layer_idx = [int(attn_layer)]
            cfg.seed = int(seed)
            if load_value != "__KEEP_BASE__":
                apply_encoder_load_strategy(cfg, load_value, pretrain_resume_path=pretrain_resume_path)

            for key, value in overrides.items():
                set_key(cfg, str(key), value)
            for key, value in extra_overrides.items():
                set_key(cfg, key, value)

            cfg_path = run_dir / "config.yaml"
            OmegaConf.save(cfg, str(cfg_path))

            skip_reason = None
            if skip_done:
                prev = existing_by_run.get(run_name)
                if isinstance(prev, dict):
                    train_ok = prev.get("train_returncode") in (None, 0)
                    test_ok = prev.get("test_returncode") in (None, 0)
                    if train_ok and test_ok:
                        skip_reason = "summary_done"

            task = {
                "run": run_name,
                "attn_layer_idx": int(attn_layer) if attn_layer is not None else None,
                "load_strategy": load_name,
                "load_strategy_value": load_value,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "cfg_path": str(cfg_path),
                "variant_source": {
                    "run": variant.get("source_run"),
                    "profile": variant.get("source_profile"),
                    "trial": variant.get("source_trial"),
                },
            }
            if skip_reason is None:
                tasks.append(task)
    return tasks


def _expand_grid_to_variants(exp_cfg: dict):
    """
    Expand exp config key `grid` into a variants list so we can fully reuse
    existing variant-mode execution logic.

    Supported schema:
      grid:
        name_prefix: r04s            # optional
        common:                      # optional
          source_run: ...
          source_profile: ...
          source_trial: ...
          load_strategy: load_3
          attn_layer_idx: 2
          seed: 0 | seeds: [0,1,2]  # optional
          overrides: {...}           # optional common overrides
        points:                      # required non-empty list
          - name: g0_base            # optional
            overrides: {...}         # optional
            # point-level source_* / load_strategy / attn_layer_idx / seed(s) are allowed
            # point-level extra keys (not reserved) are treated as overrides shorthand
    """
    reserved_keys = (
        "name",
        "source_run",
        "source_profile",
        "source_trial",
        "load_strategy",
        "attn_layer_idx",
        "seed",
        "seeds",
        "overrides",
    )
    forward_keys = (
        "source_run",
        "source_profile",
        "source_trial",
        "load_strategy",
        "attn_layer_idx",
        "seed",
        "seeds",
    )
    return expand_grid_points(
        exp_cfg,
        forward_keys=forward_keys,
        reserved_keys=reserved_keys,
        default_name_prefix="grid",
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    exp_cfg = {}
    exp_cfg_path = None
    if args.exp_config:
        exp_cfg_path = Path(args.exp_config).resolve()
        exp_cfg = load_exp_config(exp_cfg_path)
        args = _override_args_from_exp_config(args, exp_cfg)

    attn_layers = parse_optional_csv(args.attn_layers, int)
    seeds = parse_optional_csv(args.seeds, int)
    load_strategy_map = parse_load_strategies(args.load_strategies)

    extra_overrides = parse_set_overrides(args.set)

    workspace = create_experiment_workspace(
        current_file=__file__,
        base_config=args.base_config,
        exp_root=args.exp_root,
        exp_name=args.exp_name,
        default_name_prefix="mixer_tpp_grid",
        exp_cfg_path=exp_cfg_path,
        resume_exp=bool(args.resume_exp),
    )
    base_config_path = workspace.base_config_path
    runs_dir = workspace.runs_dir

    existing_summary, existing_by_run = try_load_summary(workspace.exp_dir / "summary.json")

    gpu_ids = parse_optional_csv(args.gpu_ids, int)
    max_parallel, jobs_per_gpu = adjust_parallel_limits(
        max_parallel_raw=args.max_parallel,
        jobs_per_gpu_raw=args.jobs_per_gpu,
        gpu_ids=gpu_ids,
    )

    variant_defs_direct = exp_cfg.get("variants") if isinstance(exp_cfg, dict) else None
    variant_defs_from_grid = None
    if not (isinstance(variant_defs_direct, list) and variant_defs_direct):
        variant_defs_from_grid = _expand_grid_to_variants(exp_cfg)

    mode = "matrix"
    if isinstance(variant_defs_direct, list) and variant_defs_direct:
        mode = "variants"
    elif isinstance(variant_defs_from_grid, list) and variant_defs_from_grid:
        mode = "grid"

    plan = {
        "model": args.model,
        "base_config": str(base_config_path),
        "base_config_snapshot": str(workspace.base_config_snapshot_path),
        "exp_config": str(exp_cfg_path) if exp_cfg_path is not None else None,
        "exp_config_snapshot": (
            str(workspace.exp_config_snapshot_path)
            if workspace.exp_config_snapshot_path is not None
            else None
        ),
        "mode": mode,
        "attn_layers": attn_layers,
        "load_strategies": load_strategy_map,
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
    base_cfg_for_resume = OmegaConf.load(str(base_config_path))
    pretrain_resume_path = str(getattr(base_cfg_for_resume, "resume_path", "")).strip()
    if not pretrain_resume_path:
        pretrain_resume_path = None

    variant_defs = variant_defs_direct
    if not (isinstance(variant_defs, list) and variant_defs):
        variant_defs = variant_defs_from_grid

    if isinstance(variant_defs, list) and variant_defs:
        tasks = _build_tasks_variant_mode(
            runs_dir=runs_dir,
            base_config_path=base_config_path,
            pretrain_resume_path=pretrain_resume_path,
            variants=variant_defs,
            seeds=seeds,
            load_strategy_map=load_strategy_map,
            extra_overrides=extra_overrides,
            existing_by_run=existing_by_run,
            skip_done=bool(args.skip_done),
        )
    else:
        tasks = _build_tasks_matrix_mode(
            runs_dir=runs_dir,
            base_config_path=base_config_path,
            pretrain_resume_path=pretrain_resume_path,
            attn_layers=attn_layers,
            load_strategy_map=load_strategy_map,
            seeds=seeds,
            extra_overrides=extra_overrides,
            existing_by_run=existing_by_run,
            skip_done=bool(args.skip_done),
        )

    skipped_count = 0
    if args.skip_done:
        if isinstance(variant_defs, list) and variant_defs:
            total_planned = 0
            for variant in variant_defs:
                if not isinstance(variant, dict):
                    continue
                variant_seeds = variant.get("seeds")
                if variant_seeds is None:
                    if "seed" in variant:
                        total_planned += 1
                    else:
                        total_planned += len(seeds)
                elif isinstance(variant_seeds, (list, tuple)):
                    total_planned += len(variant_seeds)
                else:
                    total_planned += 1
        else:
            total_planned = len(attn_layers) * len(load_strategy_map) * len(seeds)
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
    print(f"Runs folder: {runs_dir}")
    print("Summary file:", workspace.exp_dir / "summary.json")


if __name__ == "__main__":
    main()
