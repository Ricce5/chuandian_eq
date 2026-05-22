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
    build_tf_mf_seed_variant_name,
    build_tf_mf_pairs,
    create_experiment_workspace,
    dump_json,
    expand_grid_points,
    execute_tasks,
    load_exp_config,
    normalize_run_name,
    parse_bool_text,
    parse_set_by_tf,
    parse_optional_csv,
    parse_set_overrides,
    parse_mapping,
    resolve_point_seeds,
    set_key,
    try_load_summary,
)


def _expand_grid_to_points(exp_cfg: dict):
    """
    Expand exp config key `grid` into a normalized points list for clf grids.

    Supported schema:
      grid:
        name_prefix: tf90                 # optional
        common:                           # optional
          Tfore: 90                       # optional
          Mf: 5.5                         # optional
          seed: 0 | seeds: [0,1,2]        # optional
          time_bias_type: log             # optional
          use_sampler: true               # optional
          criterion_alpha: 0.75           # optional
          overrides: {...}                # optional
        points:                           # required non-empty list
          - name: p1                      # optional
            overrides: {...}              # optional
            # point-level keys can override common keys
    """
    reserved_keys = (
        "name",
        "Tfore",
        "Mf",
        "seed",
        "seeds",
        "time_bias_type",
        "use_sampler",
        "criterion_alpha",
        "overrides",
    )
    forward_keys = (
        "Tfore",
        "Mf",
        "seed",
        "seeds",
        "time_bias_type",
        "use_sampler",
        "criterion_alpha",
    )
    return expand_grid_points(
        exp_cfg,
        forward_keys=forward_keys,
        reserved_keys=reserved_keys,
        default_name_prefix="grid",
    )


def _build_task_cfg(
    *,
    cfg,
    twindow: int,
    dt: int,
    context_len: int,
    tf: int,
    mf: float,
    seed: int,
    apply_ref_profile: bool,
    resume_path_override: str,
    global_time_bias_type,
    time_bias_map: dict,
    use_sampler_map: dict,
    alpha_map: dict,
    clear_criterion_for_unmapped_tf: bool,
    extra_overrides: dict,
    set_by_tf_map: dict,
    point_use_sampler=None,
    point_criterion_alpha=None,
    point_time_bias_type=None,
    point_overrides: dict | None = None,
):
    cfg.Twindow = int(twindow)
    cfg.Tfore = int(tf)
    cfg.Mf = float(mf)
    cfg.dt = int(dt)
    cfg.context_len = int(context_len)
    cfg.seed = int(seed)

    if apply_ref_profile:
        cfg.resume_path = resume_path_override
        cfg.load_specific_parts = ["encoder"]
        if "mixer_model_config" not in cfg or cfg.mixer_model_config is None:
            cfg.mixer_model_config = {}
        cfg.mlp_dropout = 0.5
        cfg.mlp_hdw = [128]
        cfg.use_sampler = True

    if global_time_bias_type is not None:
        cfg.time_bias_type = global_time_bias_type
    if int(tf) in time_bias_map:
        cfg.time_bias_type = time_bias_map[int(tf)]
    if point_time_bias_type is not None:
        cfg.time_bias_type = point_time_bias_type

    if int(tf) in use_sampler_map:
        cfg.use_sampler = bool(use_sampler_map[int(tf)])
    if point_use_sampler is not None:
        cfg.use_sampler = bool(point_use_sampler)

    alpha_value = None
    if point_criterion_alpha is not None:
        alpha_value = float(point_criterion_alpha)
    elif int(tf) in alpha_map:
        alpha_value = float(alpha_map[int(tf)])

    if alpha_value is not None:
        cfg.criterion_name = "focal"
        if "criterion_cfg" not in cfg or cfg.criterion_cfg is None:
            cfg.criterion_cfg = {}
        cfg.criterion_cfg.alpha = float(alpha_value)
        if "gamma" not in cfg.criterion_cfg:
            cfg.criterion_cfg.gamma = 2.0
    elif clear_criterion_for_unmapped_tf:
        cfg.pop("criterion_name", None)
        cfg.pop("criterion_cfg", None)

    for key, value in extra_overrides.items():
        set_key(cfg, key, value)

    if int(tf) in set_by_tf_map:
        for key, value in set_by_tf_map[int(tf)].items():
            set_key(cfg, str(key), value)

    if point_overrides:
        for key, value in point_overrides.items():
            set_key(cfg, str(key), value)

    return cfg


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run clf_mixer_attnpl_t over (Tfore, Mf, seed) grid. "
            "Each run writes checkpoint to a subfolder under one new experiments folder."
        )
    )
    parser.add_argument("--model", type=str, default="clf_mixer_attnpl_t")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help="Path to external experiment config (.yaml/.yml/.json). If provided, overrides CLI args.",
    )
    parser.add_argument("--base_config", type=str, default="config/clf_mixer_attnpl_t.yaml")
    parser.add_argument(
        "--tfs",
        type=str,
        default="10,20,30,60,90",
        help="Comma separated Tfore values.",
    )
    parser.add_argument(
        "--mfs",
        type=str,
        default="4,4.5,4.5,5,5.5",
        help="Comma separated Mf values; must have same length as --tfs unless --pair_mode false.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma separated seeds, e.g. 0,1,2",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "true: zip tfs and mfs as pairs; false: full cartesian product of all tfs x mfs."
        ),
    )
    parser.add_argument(
        "--time_bias_type",
        type=str,
        default=None,
        help="Optional global override, e.g. log or linear.",
    )
    parser.add_argument(
        "--time_bias_by_tf",
        type=str,
        default="",
        help="Optional per-Tfore time bias mapping, format tf:bias,tf:bias (e.g. 90:linear).",
    )
    parser.add_argument(
        "--use_sampler_by_tf",
        type=str,
        default="",
        help=(
            "Optional per-Tfore use_sampler mapping. "
            "Format: tf:true,tf:false (e.g. 10:true,90:false)."
        ),
    )
    parser.add_argument(
        "--alpha_by_tf",
        type=str,
        default="10:0.5,20:0.25,90:0.25",
        help=(
            "Optional per-Tfore focal alpha mapping. "
            "Format: tf:alpha,tf:alpha ; empty string disables alpha override."
        ),
    )
    parser.add_argument(
        "--set_by_tf",
        type=str,
        default="",
        help=(
            "Optional per-Tfore config overrides in YAML/JSON mapping string. "
            "Example: '{10: {batch_size: 64}, 90: {batch_size: 64}}'."
        ),
    )
    parser.add_argument(
        "--clear_criterion_for_unmapped_tf",
        type=str,
        default="true",
        choices=["true", "false"],
        help="If true, remove criterion_name/criterion_cfg when tf not in --alpha_by_tf.",
    )
    parser.add_argument(
        "--apply_ref_profile",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Apply defaults observed in reference ckpts (n_layer=3, attn_layer_idx=[1], mlp/use_sampler).",
    )
    parser.add_argument(
        "--resume_path_override",
        type=str,
        default="./checkpoints/mixer_tpp_20260508-182744/last_model_1.pth",
        help="resume_path used when --apply_ref_profile=true.",
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
    parser.add_argument("--twindow", type=int, default=180)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--context_len", type=int, default=1)
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
        help="How many concurrent jobs are allowed on each GPU id. Useful for single-GPU multi-task.",
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
        "pair_mode": "pair_mode",
        "time_bias_type": "time_bias_type",
        "set_by_tf": "set_by_tf",
        "clear_criterion_for_unmapped_tf": "clear_criterion_for_unmapped_tf",
        "apply_ref_profile": "apply_ref_profile",
        "resume_path_override": "resume_path_override",
        "run_test": "run_test",
        "ckpt_select": "ckpt_select",
        "skip_train": "skip_train",
        "exp_root": "exp_root",
        "exp_name": "exp_name",
        "twindow": "twindow",
        "dt": "dt",
        "context_len": "context_len",
        "max_parallel": "max_parallel",
        "jobs_per_gpu": "jobs_per_gpu",
        "stop_on_error": "stop_on_error",
        "resume_exp": "resume_exp",
        "skip_done": "skip_done",
    }
    csv_like_keys = {
        "tfs": "tfs",
        "mfs": "mfs",
        "seeds": "seeds",
        "gpu_ids": "gpu_ids",
    }
    mapping_like_keys = {
        "time_bias_by_tf": "time_bias_by_tf",
        "use_sampler_by_tf": "use_sampler_by_tf",
        "alpha_by_tf": "alpha_by_tf",
    }
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
    if grid_points is None:
        tfs = parse_optional_csv(args.tfs, int)
        mfs = parse_optional_csv(args.mfs, float)
        if not tfs:
            raise ValueError("No Tfore values found.")
        if not mfs:
            raise ValueError("No Mf values found.")
        tf_mf_pairs, pair_mode = build_tf_mf_pairs(tfs=tfs, mfs=mfs, pair_mode_raw=args.pair_mode)
    else:
        tf_mf_pairs = []
        pair_mode = "grid"

    time_bias_map = parse_mapping(args.time_bias_by_tf, int, str)
    use_sampler_map = parse_mapping(args.use_sampler_by_tf, int, parse_bool_text)
    alpha_map = parse_mapping(args.alpha_by_tf, int, float)
    set_by_tf_map = parse_set_by_tf(getattr(args, "set_by_tf", ""))
    clear_criterion_for_unmapped_tf = parse_bool_text(args.clear_criterion_for_unmapped_tf)
    apply_ref_profile = parse_bool_text(args.apply_ref_profile)

    extra_overrides = parse_set_overrides(args.set)

    workspace = create_experiment_workspace(
        current_file=__file__,
        base_config=args.base_config,
        exp_root=args.exp_root,
        exp_name=args.exp_name,
        default_name_prefix="clf_mf_tf_grid",
        exp_cfg_path=exp_cfg_path,
        resume_exp=bool(args.resume_exp),
    )

    existing_summary, existing_by_run = try_load_summary(workspace.exp_dir / "summary.json")

    gpu_ids = parse_optional_csv(args.gpu_ids, int)
    max_parallel, jobs_per_gpu = adjust_parallel_limits(
        max_parallel_raw=args.max_parallel,
        jobs_per_gpu_raw=args.jobs_per_gpu,
        gpu_ids=gpu_ids,
    )

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
        "pair_mode": pair_mode,
        "pairs": (
            [{"Tfore": tf, "Mf": mf} for tf, mf in tf_mf_pairs]
            if grid_points is None
            else [
                {
                    "name": point.get("name"),
                    "Tfore": point.get("Tfore"),
                    "Mf": point.get("Mf"),
                    "seed": point.get("seed"),
                    "seeds": point.get("seeds"),
                    "time_bias_type": point.get("time_bias_type"),
                    "use_sampler": point.get("use_sampler"),
                    "criterion_alpha": point.get("criterion_alpha"),
                }
                for point in grid_points
            ]
        ),
        "seeds": seeds,
        "time_bias_type": args.time_bias_type,
        "time_bias_by_tf": time_bias_map,
        "use_sampler_by_tf": use_sampler_map,
        "alpha_by_tf": alpha_map,
        "set_by_tf": set_by_tf_map,
        "apply_ref_profile": apply_ref_profile,
        "clear_criterion_for_unmapped_tf": clear_criterion_for_unmapped_tf,
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
        for tf, mf in tf_mf_pairs:
            for seed in seeds:
                variant_name = build_tf_mf_seed_variant_name(tf=tf, mf=mf, seed=seed)
                run_dir = workspace.runs_dir / variant_name
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = OmegaConf.load(str(workspace.base_config_path))
                cfg = _build_task_cfg(
                    cfg=cfg,
                    twindow=int(args.twindow),
                    dt=int(args.dt),
                    context_len=int(args.context_len),
                    tf=int(tf),
                    mf=float(mf),
                    seed=int(seed),
                    apply_ref_profile=apply_ref_profile,
                    resume_path_override=args.resume_path_override,
                    global_time_bias_type=args.time_bias_type,
                    time_bias_map=time_bias_map,
                    use_sampler_map=use_sampler_map,
                    alpha_map=alpha_map,
                    clear_criterion_for_unmapped_tf=clear_criterion_for_unmapped_tf,
                    extra_overrides=extra_overrides,
                    set_by_tf_map=set_by_tf_map,
                )

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
                    "Twindow": int(args.twindow),
                    "Tfore": int(tf),
                    "Mf": float(mf),
                    "seed": int(seed),
                    "time_bias_type": getattr(cfg, "time_bias_type", None),
                    "criterion_alpha": (
                        float(cfg.criterion_cfg.alpha)
                        if (
                            "criterion_cfg" in cfg
                            and cfg.criterion_cfg is not None
                            and "alpha" in cfg.criterion_cfg
                        )
                        else None
                    ),
                    "run_dir": str(run_dir),
                    "cfg_path": str(cfg_path),
                }
                if skip_reason is None:
                    tasks.append(task)
    else:
        for point in grid_points:
            if "Tfore" not in point or point.get("Tfore") is None:
                raise ValueError(f"grid point {point.get('name')} missing required key: Tfore")
            if "Mf" not in point or point.get("Mf") is None:
                raise ValueError(f"grid point {point.get('name')} missing required key: Mf")

            tf = int(point.get("Tfore"))
            mf = float(point.get("Mf"))
            point_name = str(point.get("name") or "grid")
            point_seeds = resolve_point_seeds(point, default_seeds=seeds)
            if not point_seeds:
                raise ValueError(f"grid point {point_name} resolved empty seeds.")

            point_overrides = point.get("overrides", {})
            if point_overrides is None:
                point_overrides = {}
            if not isinstance(point_overrides, dict):
                raise ValueError(f"grid point {point_name} overrides must be mapping/object.")

            point_time_bias_type = point.get("time_bias_type")
            point_use_sampler = point.get("use_sampler")
            point_criterion_alpha = point.get("criterion_alpha")

            for seed in point_seeds:
                variant_name = normalize_run_name(
                    f"{point_name}_seed_{int(seed)}",
                    fallback=f"grid_seed_{int(seed)}",
                )
                run_dir = workspace.runs_dir / variant_name
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = OmegaConf.load(str(workspace.base_config_path))
                cfg = _build_task_cfg(
                    cfg=cfg,
                    twindow=int(args.twindow),
                    dt=int(args.dt),
                    context_len=int(args.context_len),
                    tf=tf,
                    mf=mf,
                    seed=int(seed),
                    apply_ref_profile=apply_ref_profile,
                    resume_path_override=args.resume_path_override,
                    global_time_bias_type=args.time_bias_type,
                    time_bias_map=time_bias_map,
                    use_sampler_map=use_sampler_map,
                    alpha_map=alpha_map,
                    clear_criterion_for_unmapped_tf=clear_criterion_for_unmapped_tf,
                    extra_overrides=extra_overrides,
                    set_by_tf_map=set_by_tf_map,
                    point_use_sampler=point_use_sampler,
                    point_criterion_alpha=point_criterion_alpha,
                    point_time_bias_type=point_time_bias_type,
                    point_overrides=point_overrides,
                )

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
                    "Twindow": int(args.twindow),
                    "Tfore": tf,
                    "Mf": mf,
                    "seed": int(seed),
                    "time_bias_type": getattr(cfg, "time_bias_type", None),
                    "criterion_alpha": (
                        float(cfg.criterion_cfg.alpha)
                        if (
                            "criterion_cfg" in cfg
                            and cfg.criterion_cfg is not None
                            and "alpha" in cfg.criterion_cfg
                        )
                        else None
                    ),
                    "run_dir": str(run_dir),
                    "cfg_path": str(cfg_path),
                }
                if skip_reason is None:
                    tasks.append(task)

    skipped_count = 0
    if args.skip_done:
        if grid_points is None:
            total_planned = len(tf_mf_pairs) * len(seeds)
        else:
            total_planned = 0
            for point in grid_points:
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
