#!/usr/bin/env python3
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from automation import (
    apply_exp_config_overrides,
    adjust_parallel_limits,
    build_tf_mf_pairs,
    create_experiment_workspace,
    dump_json,
    execute_tasks,
    load_exp_config,
    parse_bool_text,
    parse_optional_csv,
    parse_set_overrides,
    parse_mapping,
    set_key,
)


def _build_variant_name(tf, mf, seed):
    mf_str = str(mf).replace(".", "p")
    return f"tf_{tf}_mf_{mf_str}_seed_{seed}"


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
    return parser


def _override_args_from_exp_config(args, exp_cfg: dict):
    direct_key_map = {
        "model": "model",
        "base_config": "base_config",
        "pair_mode": "pair_mode",
        "time_bias_type": "time_bias_type",
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

    exp_cfg_path = None
    if args.exp_config:
        exp_cfg_path = Path(args.exp_config).resolve()
        exp_cfg = load_exp_config(exp_cfg_path)
        args = _override_args_from_exp_config(args, exp_cfg)

    tfs = parse_optional_csv(args.tfs, int)
    mfs = parse_optional_csv(args.mfs, float)
    seeds = parse_optional_csv(args.seeds, int)
    if not tfs:
        raise ValueError("No Tfore values found.")
    if not mfs:
        raise ValueError("No Mf values found.")
    if not seeds:
        raise ValueError("No seed values found.")

    tf_mf_pairs, pair_mode = build_tf_mf_pairs(tfs=tfs, mfs=mfs, pair_mode_raw=args.pair_mode)

    time_bias_map = parse_mapping(args.time_bias_by_tf, int, str)
    use_sampler_map = parse_mapping(args.use_sampler_by_tf, int, parse_bool_text)
    alpha_map = parse_mapping(args.alpha_by_tf, int, float)
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
        allow_existing_without_resume=True,
    )

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
        "pairs": [{"Tfore": tf, "Mf": mf} for tf, mf in tf_mf_pairs],
        "seeds": seeds,
        "time_bias_type": args.time_bias_type,
        "time_bias_by_tf": time_bias_map,
        "use_sampler_by_tf": use_sampler_map,
        "alpha_by_tf": alpha_map,
        "apply_ref_profile": apply_ref_profile,
        "clear_criterion_for_unmapped_tf": clear_criterion_for_unmapped_tf,
        "max_parallel": max_parallel,
        "gpu_ids": gpu_ids,
        "jobs_per_gpu": jobs_per_gpu,
        "stop_on_error": bool(args.stop_on_error),
        "extra_overrides": extra_overrides,
    }
    dump_json(workspace.exp_dir / "plan.json", plan)

    tasks = []
    for tf, mf in tf_mf_pairs:
        for seed in seeds:
            variant_name = _build_variant_name(tf=tf, mf=mf, seed=seed)
            run_dir = workspace.runs_dir / variant_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(str(workspace.base_config_path))
            cfg.Twindow = int(args.twindow)
            cfg.Tfore = int(tf)
            cfg.Mf = float(mf)
            cfg.dt = int(args.dt)
            cfg.context_len = int(args.context_len)
            cfg.seed = int(seed)

            if apply_ref_profile:
                cfg.resume_path = args.resume_path_override
                cfg.load_specific_parts = ["encoder"]
                if "mixer_model_config" not in cfg or cfg.mixer_model_config is None:
                    cfg.mixer_model_config = {}
                cfg.mlp_dropout = 0.5
                cfg.mlp_hdw = [128]
                cfg.use_sampler = True

            if args.time_bias_type is not None:
                cfg.time_bias_type = args.time_bias_type
            if int(tf) in time_bias_map:
                cfg.time_bias_type = time_bias_map[int(tf)]
            if int(tf) in use_sampler_map:
                cfg.use_sampler = bool(use_sampler_map[int(tf)])

            if int(tf) in alpha_map:
                cfg.criterion_name = "focal"
                if "criterion_cfg" not in cfg or cfg.criterion_cfg is None:
                    cfg.criterion_cfg = {}
                cfg.criterion_cfg.alpha = float(alpha_map[int(tf)])
                if "gamma" not in cfg.criterion_cfg:
                    cfg.criterion_cfg.gamma = 2.0
            elif clear_criterion_for_unmapped_tf:
                cfg.pop("criterion_name", None)
                cfg.pop("criterion_cfg", None)

            for key, value in extra_overrides.items():
                set_key(cfg, key, value)

            cfg_path = run_dir / "config_input.yaml"
            OmegaConf.save(cfg, str(cfg_path))

            tasks.append(
                {
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
            )

    execute_tasks(
        tasks=tasks,
        args=args,
        repo_root=workspace.repo_root,
        exp_dir=workspace.exp_dir,
        gpu_ids=gpu_ids,
        max_parallel=max_parallel,
        jobs_per_gpu=jobs_per_gpu,
        stop_on_error=bool(args.stop_on_error),
    )

    print(f"\nDone. Experiment folder: {workspace.exp_dir}")
    print(f"Runs folder: {workspace.runs_dir}")
    print("Summary file:", workspace.exp_dir / "summary.json")


if __name__ == "__main__":
    main()
