#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
import yaml


def _parse_csv(raw: str, cast_fn):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [cast_fn(x) for x in items]


def _set_key(cfg, dotted_key: str, value):
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or node[key] is None:
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _build_variant_name(tf, mf, seed):
    mf_str = str(mf).replace(".", "p")
    return f"tf_{tf}_mf_{mf_str}_seed_{seed}"


def _run_cmd(cmd, cwd: Path, env=None, log_path: Path = None):
    print("[CMD]", " ".join(cmd))
    if log_path is None:
        return subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[CMD] " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def _parse_bool_text(raw: str) -> bool:
    norm = str(raw).strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool text: {raw!r}")


def _parse_mapping(raw: str, key_cast, value_cast):
    mapping = {}
    text = (raw or "").strip()
    if not text:
        return mapping
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Invalid mapping item: {item!r}, expected key:value")
        key_str, value_str = item.split(":", 1)
        mapping[key_cast(key_str.strip())] = value_cast(value_str.strip())
    return mapping


def _run_single_task(task, args, repo_root: Path):
    run_dir = Path(task["run_dir"])
    cfg_path = Path(task["cfg_path"])
    log_path = run_dir / "launcher.log"

    env = os.environ.copy()
    assigned_gpu = task.get("cuda_id", None)
    if assigned_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    train_ret_code = None
    test_ret_code = None
    test_skipped_reason = None

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "main.py",
            "--model",
            args.model,
            "--mode",
            "train",
            "--config",
            str(cfg_path),
            "--checkpoint_dir",
            str(run_dir),
        ]
        train_ret = _run_cmd(train_cmd, repo_root, env=env, log_path=log_path)
        train_ret_code = int(train_ret.returncode)

    can_run_test = args.run_test and (args.skip_train or train_ret_code == 0)
    if args.run_test and not can_run_test:
        test_skipped_reason = "train_failed"

    if can_run_test:
        test_cmd = [
            sys.executable,
            "main.py",
            "--model",
            args.model,
            "--mode",
            "test",
            "--config",
            str(cfg_path),
            "--checkpoint_dir",
            str(run_dir),
            "--ckpt_select",
            args.ckpt_select,
        ]
        test_ret = _run_cmd(test_cmd, repo_root, env=env, log_path=log_path)
        test_ret_code = int(test_ret.returncode)

    result = dict(task)
    result["train_returncode"] = train_ret_code
    result["test_returncode"] = test_ret_code
    result["test_skipped_reason"] = test_skipped_reason
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run clf_mixer_attnpl_t over (Tfore, Mf, seed) grid. "
            "Each run writes checkpoint to a subfolder under one new experiments folder."
        )
    )
    parser.add_argument("--model", type=str, default="clf_mixer_attnpl_t")
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
        default="./checkpoints/mixer_tpp_20260203-161948/last_model_1.pth",
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
        choices=["best", "last"],
        help="Checkpoint for optional test mode.",
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
        "--gpu_ids",
        type=str,
        default="",
        help="Optional GPU list for parallel scheduling, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop submitting/running remaining tasks once any task fails.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    tfs = _parse_csv(args.tfs, int)
    mfs = _parse_csv(args.mfs, float)
    seeds = _parse_csv(args.seeds, int)
    pair_mode = _parse_bool_text(args.pair_mode)
    clear_criterion_for_unmapped_tf = _parse_bool_text(args.clear_criterion_for_unmapped_tf)
    apply_ref_profile = _parse_bool_text(args.apply_ref_profile)

    if pair_mode:
        if len(tfs) != len(mfs):
            raise ValueError(
                f"pair_mode=true requires len(tfs)==len(mfs), got {len(tfs)} vs {len(mfs)}"
            )
        tf_mf_pairs = list(zip(tfs, mfs))
    else:
        tf_mf_pairs = [(tf, mf) for tf in tfs for mf in mfs]

    alpha_map = _parse_mapping(args.alpha_by_tf, int, float)
    time_bias_map = _parse_mapping(args.time_bias_by_tf, int, str)

    extra_overrides = {}
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"Invalid --set item: {kv!r}, expected key=value")
        key, value = kv.split("=", 1)
        extra_overrides[key.strip()] = yaml.safe_load(value)

    exp_root = (repo_root / args.exp_root).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    if args.exp_name:
        exp_name = args.exp_name
    else:
        now = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"clf_mf_tf_grid_{now}"

    exp_dir = exp_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = _parse_csv(args.gpu_ids, int) if (args.gpu_ids or "").strip() else []
    max_parallel = max(1, int(args.max_parallel))
    if gpu_ids:
        max_parallel = min(max_parallel, len(gpu_ids))

    plan = {
        "model": args.model,
        "base_config": str(base_config_path),
        "pair_mode": pair_mode,
        "pairs": [{"Tfore": tf, "Mf": mf} for tf, mf in tf_mf_pairs],
        "seeds": seeds,
        "time_bias_type": args.time_bias_type,
        "time_bias_by_tf": time_bias_map,
        "alpha_by_tf": alpha_map,
        "apply_ref_profile": apply_ref_profile,
        "clear_criterion_for_unmapped_tf": clear_criterion_for_unmapped_tf,
        "max_parallel": max_parallel,
        "gpu_ids": gpu_ids,
        "stop_on_error": bool(args.stop_on_error),
        "extra_overrides": extra_overrides,
    }
    with open(exp_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    tasks = []
    for tf, mf in tf_mf_pairs:
        for seed in seeds:
            variant_name = _build_variant_name(tf=tf, mf=mf, seed=seed)
            run_dir = runs_dir / variant_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(str(base_config_path))
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
                cfg.mixer_model_config.n_layer = 3
                cfg.mixer_model_config.attn_layer_idx = [1]
                cfg.mlp_dropout = 0.5
                cfg.mlp_hdw = [128]
                cfg.use_sampler = True

            if args.time_bias_type is not None:
                cfg.time_bias_type = args.time_bias_type
            if int(tf) in time_bias_map:
                cfg.time_bias_type = time_bias_map[int(tf)]

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
                _set_key(cfg, key, value)

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
                        if ("criterion_cfg" in cfg and cfg.criterion_cfg is not None and "alpha" in cfg.criterion_cfg)
                        else None
                    ),
                    "run_dir": str(run_dir),
                    "cfg_path": str(cfg_path),
                }
            )

    summary = []
    any_failed = False

    if max_parallel == 1:
        for task in tasks:
            task_result = _run_single_task(task, args, repo_root)
            summary.append(task_result)
            with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            failed = (task_result.get("train_returncode") not in {None, 0}) or (
                task_result.get("test_returncode") not in {None, 0}
            )
            if failed:
                any_failed = True
                if args.stop_on_error:
                    print(f"[STOP] stop_on_error enabled, failed run: {task_result['run']}")
                    break
    else:
        executor = ThreadPoolExecutor(max_workers=max_parallel)
        futures = {}
        running_by_gpu = {}
        pending = list(tasks)
        try:
            while pending or futures:
                while pending and len(futures) < max_parallel and (not (args.stop_on_error and any_failed)):
                    task = pending.pop(0)
                    assigned_gpu = None
                    if gpu_ids:
                        for gpu in gpu_ids:
                            if gpu not in running_by_gpu:
                                assigned_gpu = gpu
                                break
                        if assigned_gpu is None:
                            break
                        running_by_gpu[assigned_gpu] = task["run"]
                    task_submit = dict(task)
                    if assigned_gpu is not None:
                        task_submit["cuda_id"] = int(assigned_gpu)
                    future = executor.submit(_run_single_task, task_submit, args, repo_root)
                    futures[future] = assigned_gpu

                if not futures:
                    continue

                finished_future = None
                for future in as_completed(list(futures.keys()), timeout=None):
                    finished_future = future
                    break
                if finished_future is None:
                    continue

                gpu = futures.pop(finished_future)
                if gpu is not None and gpu in running_by_gpu:
                    running_by_gpu.pop(gpu, None)

                task_result = finished_future.result()
                summary.append(task_result)
                with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                failed = (task_result.get("train_returncode") not in {None, 0}) or (
                    task_result.get("test_returncode") not in {None, 0}
                )
                if failed:
                    any_failed = True
                    if args.stop_on_error:
                        print(f"[STOP] stop_on_error enabled, failed run: {task_result['run']}")
        finally:
            executor.shutdown(wait=True)

    print(f"\nDone. Experiment folder: {exp_dir}")
    print(f"Runs folder: {runs_dir}")
    print("Summary file:", exp_dir / "summary.json")


if __name__ == "__main__":
    main()
