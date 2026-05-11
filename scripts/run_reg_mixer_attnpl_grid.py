#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf
import yaml

from grid_runner_common import (
    apply_exp_config_overrides,
    execute_tasks,
    load_exp_config,
    parse_csv,
    set_key,
)


def _build_variant_name(attn_layer: int, load_strategy: str, seed: int):
    return f"attn_l{attn_layer}_load_{load_strategy}_seed_{seed}"


def _normalize_run_name(raw: str, fallback: str):
    text = str(raw or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^0-9a-zA-Z._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def _parse_load_strategy_item(raw: str):
    item = raw.strip()
    if not item:
        raise ValueError("Empty load strategy item.")
    if ":" not in item:
        raise ValueError(f"Invalid load strategy item: {raw!r}. Expected name:value")
    name, value = item.split(":", 1)
    name = name.strip()
    value = value.strip().lower()
    if value in {"none", "null"}:
        parsed = None
    elif value in {"input_layer0_layer1", "input_l0_l1", "layer01_input", "l01_input", "three"}:
        parsed = "input_layer0_layer1"
    elif value in {"input_layer", "input", "input_proj", "inproj"}:
        parsed = "input_layer"
    elif value in {"layer0", "l0"}:
        parsed = "layer0"
    elif value in {"layer1", "l1"}:
        parsed = "layer1"
    elif value in {"layer0_input", "l0_input", "first", "first_layer"}:
        parsed = "layer0_input"
    elif value in {"layer1_input", "l1_input", "second", "second_layer"}:
        parsed = "layer1_input"
    elif value in {"encoder"}:
        parsed = "encoder"
    else:
        raise ValueError(
            f"Unsupported load strategy value: {value!r}. "
            "Use one of: none, input_layer0_layer1, input_layer, layer0, layer1, layer0_input, layer1_input, encoder."
        )
    return name, parsed


def _parse_load_strategies(raw: str):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise ValueError("No load strategies provided.")
    parsed = {}
    for item in items:
        name, value = _parse_load_strategy_item(item)
        parsed[name] = value
    return parsed


def _parse_single_strategy_value(raw: str):
    _, parsed = _parse_load_strategy_item(f"variant:{raw}")
    return parsed


def _resolve_variant_load_strategy(raw_value, load_strategy_map: dict):
    if raw_value is None:
        raise ValueError("Variant must define load_strategy.")
    text = str(raw_value).strip()
    if not text:
        raise ValueError("Variant load_strategy cannot be empty.")
    if text in load_strategy_map:
        return text, load_strategy_map[text]
    parsed = _parse_single_strategy_value(text)
    if parsed is None:
        return "none", None
    alias_name = text.lower().replace(":", "_")
    alias_name = re.sub(r"[^0-9a-zA-Z._-]+", "_", alias_name)
    return alias_name, parsed


def _apply_load_strategy(cfg, strategy_value, pretrain_resume_path: str):
    cfg.pop("load_specific_parts", None)
    cfg.pop("encoder_param_keywords", None)
    cfg.pop("resume_path", None)
    if strategy_value is None:
        return
    if strategy_value == "encoder":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = ["encoder"]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "input_layer0_layer1":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.input_proj",
            "base_model.encoder.layers.0",
            "base_model.encoder.layers.1",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "input_layer":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer0":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.0",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer1":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.1",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer0_input":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.0",
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer1_input":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.1",
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    raise ValueError(f"Unsupported load strategy: {strategy_value!r}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run reg_mixer_attnpl_t by structural and preload strategy matrix: "
            "attn layer index x load strategy x seed."
        )
    )
    parser.add_argument("--model", type=str, default="reg_mixer_attnpl_t")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help="Path to external experiment config (.yaml/.yml/.json). If provided, overrides CLI args.",
    )
    parser.add_argument("--base_config", type=str, default="config/reg_mixer_attnpl_t.yaml")
    parser.add_argument(
        "--attn_layers",
        type=str,
        default="1,2",
        help="Comma separated attention layer indices (0-based), e.g. 1,2.",
    )
    parser.add_argument(
        "--load_strategies",
        type=str,
        default="none:none,load_3:input_layer0_layer1,input:input_layer,layer0:layer0,layer1:layer1,load_l1:layer0_input,load_l2:layer1_input",
        help=(
            "Comma separated name:value mapping. value options: "
            "none|input_layer0_layer1|input_layer|layer0|layer1|layer0_input|layer1_input|encoder."
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
    pretrain_resume_path: str,
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
                variant_name = _build_variant_name(
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
                _apply_load_strategy(cfg, load_value, pretrain_resume_path=pretrain_resume_path)

                for key, value in extra_overrides.items():
                    set_key(cfg, key, value)

                cfg_path = run_dir / "config_input.yaml"
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
    pretrain_resume_path: str,
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
        attn_layer = int(variant.get("attn_layer_idx", 2))
        load_name, load_value = _resolve_variant_load_strategy(variant.get("load_strategy"), load_strategy_map)
        raw_name = variant.get("name", f"variant_{idx + 1:02d}")
        fallback_name = f"variant_{idx + 1:02d}"
        variant_name = _normalize_run_name(raw_name, fallback=fallback_name)
        if variant_name in seen_names:
            variant_name = _normalize_run_name(f"{variant_name}_{idx + 1:02d}", fallback=fallback_name)
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
            run_name = _normalize_run_name(f"{variant_name}_seed_{seed}", fallback=f"{fallback_name}_seed_{seed}")
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(str(base_config_path))
            cfg.pop("optuna", None)
            if "mixer_model_config" not in cfg or cfg.mixer_model_config is None:
                cfg.mixer_model_config = {}
            cfg.mixer_model_config.attn_layer_idx = [int(attn_layer)]
            cfg.seed = int(seed)
            _apply_load_strategy(cfg, load_value, pretrain_resume_path=pretrain_resume_path)

            for key, value in overrides.items():
                set_key(cfg, str(key), value)
            for key, value in extra_overrides.items():
                set_key(cfg, key, value)

            cfg_path = run_dir / "config_input.yaml"
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
                "attn_layer_idx": int(attn_layer),
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


def main():
    parser = _build_parser()
    args = parser.parse_args()

    exp_cfg = {}
    exp_cfg_path = None
    if args.exp_config:
        exp_cfg_path = Path(args.exp_config).resolve()
        exp_cfg = load_exp_config(exp_cfg_path)
        args = _override_args_from_exp_config(args, exp_cfg)

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    attn_layers = parse_csv(args.attn_layers, int)
    seeds = parse_csv(args.seeds, int)
    load_strategy_map = _parse_load_strategies(args.load_strategies)

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
        exp_name = f"reg_mixer_attnpl_grid_{now}"

    exp_dir = exp_root / exp_name
    if exp_dir.exists():
        if args.resume_exp:
            print(f"[INFO] Resuming existing experiment folder: {exp_dir}")
        else:
            raise FileExistsError(
                f"Experiment folder already exists: {exp_dir}. "
                "Use --resume_exp to continue in-place."
            )
    else:
        exp_dir.mkdir(parents=True, exist_ok=False)

    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config_snapshot_path = configs_dir / base_config_path.name
    shutil.copy2(base_config_path, base_config_snapshot_path)

    exp_config_snapshot_path = None
    if exp_cfg_path is not None:
        exp_cfg_name = exp_cfg_path.name
        if exp_cfg_name == base_config_path.name:
            exp_cfg_name = f"exp_config_{exp_cfg_name}"
        exp_config_snapshot_path = configs_dir / exp_cfg_name
        shutil.copy2(exp_cfg_path, exp_config_snapshot_path)

    existing_summary = []
    existing_by_run = {}
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
            if isinstance(loaded_summary, list):
                existing_summary = loaded_summary
                existing_by_run = {
                    str(row.get("run")): row
                    for row in existing_summary
                    if isinstance(row, dict) and row.get("run") is not None
                }
                print(f"[INFO] Loaded existing summary with {len(existing_by_run)} runs.")
        except Exception as e:
            print(f"[WARN] Failed to load existing summary.json: {e}")

    gpu_ids = parse_csv(args.gpu_ids, int) if (args.gpu_ids or "").strip() else []
    max_parallel = max(1, int(args.max_parallel))
    jobs_per_gpu = max(1, int(args.jobs_per_gpu))
    if gpu_ids:
        max_slots = len(gpu_ids) * jobs_per_gpu
        if max_parallel > max_slots:
            print(
                f"[INFO] max_parallel={max_parallel} exceeds GPU slots={max_slots} "
                f"(len(gpu_ids)={len(gpu_ids)} x jobs_per_gpu={jobs_per_gpu}). "
                f"Use max_parallel={max_slots}."
            )
        max_parallel = min(max_parallel, max_slots)

    plan = {
        "model": args.model,
        "base_config": str(base_config_path),
        "base_config_snapshot": str(base_config_snapshot_path),
        "exp_config": str(exp_cfg_path) if exp_cfg_path is not None else None,
        "exp_config_snapshot": str(exp_config_snapshot_path) if exp_config_snapshot_path is not None else None,
        "mode": "variants" if (isinstance(exp_cfg.get("variants"), list) and exp_cfg.get("variants")) else "matrix",
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
    with open(exp_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    tasks = []
    base_cfg_for_resume = OmegaConf.load(str(base_config_path))
    pretrain_resume_path = str(getattr(base_cfg_for_resume, "resume_path", "")).strip()
    if not pretrain_resume_path:
        raise ValueError(
            f"Base config {base_config_path} does not define resume_path; "
            "cannot apply load strategies that require pretrained checkpoint."
        )

    variant_defs = exp_cfg.get("variants") if isinstance(exp_cfg, dict) else None
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
            total_planned = len(variant_defs)
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
        repo_root=repo_root,
        exp_dir=exp_dir,
        gpu_ids=gpu_ids,
        max_parallel=max_parallel,
        jobs_per_gpu=jobs_per_gpu,
        stop_on_error=bool(args.stop_on_error),
        initial_summary=existing_summary,
    )

    print(f"\nDone. Experiment folder: {exp_dir}")
    print(f"Runs folder: {runs_dir}")
    print("Summary file:", exp_dir / "summary.json")


if __name__ == "__main__":
    main()
