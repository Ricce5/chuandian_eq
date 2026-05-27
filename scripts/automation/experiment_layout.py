#!/usr/bin/env python3
import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .grid_common import parse_bool_text, parse_csv


@dataclass(frozen=True)
class ExperimentWorkspace:
    repo_root: Path
    exp_root: Path
    exp_dir: Path
    runs_dir: Path
    configs_dir: Path
    base_config_path: Path
    base_config_snapshot_path: Path
    exp_config_snapshot_path: Path | None


def resolve_repo_path(current_file: str, relative_path: str) -> Path:
    current_path = Path(current_file).resolve()
    if current_path.parent.name == "run" and current_path.parent.parent.name == "scripts":
        repo_root = current_path.parents[2]
    else:
        repo_root = current_path.parents[1]
    return (repo_root / relative_path).resolve()


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_set_overrides(set_items):
    extra = {}
    for item in set_items:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item!r}, expected key=value")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            raise ValueError(f"Invalid --set key in item: {item!r}")
        extra[key] = yaml.safe_load(raw_val)
    return extra


def parse_optional_csv(raw, cast_fn):
    return parse_csv(raw, cast_fn) if (raw or "").strip() else []


def build_tf_mf_pairs(tfs, mfs, pair_mode_raw):
    pair_mode = parse_bool_text(pair_mode_raw)
    if pair_mode:
        if len(tfs) != len(mfs):
            raise ValueError(
                f"pair_mode=true requires len(tfs)==len(mfs), got {len(tfs)} vs {len(mfs)}"
            )
        pairs = list(zip(tfs, mfs))
    else:
        pairs = [(tf, mf) for tf in tfs for mf in mfs]
    return pairs, pair_mode


def adjust_parallel_limits(max_parallel_raw: int, jobs_per_gpu_raw: int, gpu_ids):
    max_parallel = max(1, int(max_parallel_raw))
    jobs_per_gpu = max(1, int(jobs_per_gpu_raw))
    if gpu_ids:
        max_slots = len(gpu_ids) * jobs_per_gpu
        if max_parallel > max_slots:
            print(
                f"[INFO] max_parallel={max_parallel} exceeds GPU slots={max_slots} "
                f"(len(gpu_ids)={len(gpu_ids)} x jobs_per_gpu={jobs_per_gpu}). "
                f"Use max_parallel={max_slots}."
            )
        max_parallel = min(max_parallel, max_slots)
    return max_parallel, jobs_per_gpu


def create_experiment_workspace(
    *,
    current_file: str,
    base_config: str,
    exp_root: str,
    exp_name: str | None,
    default_name_prefix: str,
    exp_cfg_path: Path | None = None,
    resume_exp: bool = False,
    allow_existing_without_resume: bool = False,
    base_snapshot_prefix: str = "",
):
    repo_root = resolve_repo_path(current_file, ".")
    base_config_path = (repo_root / base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    exp_root_path = (repo_root / exp_root).resolve()
    exp_root_path.mkdir(parents=True, exist_ok=True)

    if exp_name:
        resolved_exp_name = exp_name
    else:
        now = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_exp_name = f"{default_name_prefix}_{now}"

    exp_dir = exp_root_path / resolved_exp_name
    if exp_dir.exists():
        if resume_exp:
            print(f"[INFO] Resuming existing experiment folder: {exp_dir}")
        elif allow_existing_without_resume:
            pass
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

    base_name = base_config_path.name
    if base_snapshot_prefix:
        base_name = f"{base_snapshot_prefix}{base_name}"
    base_config_snapshot_path = configs_dir / base_name
    shutil.copy2(base_config_path, base_config_snapshot_path)

    exp_config_snapshot_path = None
    if exp_cfg_path is not None:
        exp_cfg_name = exp_cfg_path.name
        if exp_cfg_name == base_config_path.name:
            exp_cfg_name = f"exp_config_{exp_cfg_name}"
        exp_config_snapshot_path = configs_dir / exp_cfg_name
        shutil.copy2(exp_cfg_path, exp_config_snapshot_path)

    return ExperimentWorkspace(
        repo_root=repo_root,
        exp_root=exp_root_path,
        exp_dir=exp_dir,
        runs_dir=runs_dir,
        configs_dir=configs_dir,
        base_config_path=base_config_path,
        base_config_snapshot_path=base_config_snapshot_path,
        exp_config_snapshot_path=exp_config_snapshot_path,
    )


def try_load_summary(summary_path: Path):
    if not summary_path.exists():
        return [], {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            loaded_summary = json.load(f)
        if not isinstance(loaded_summary, list):
            return [], {}
        existing_by_run = {
            str(row.get("run")): row
            for row in loaded_summary
            if isinstance(row, dict) and row.get("run") is not None
        }
        print(f"[INFO] Loaded existing summary with {len(existing_by_run)} runs.")
        return loaded_summary, existing_by_run
    except Exception as error:
        print(f"[WARN] Failed to load existing summary.json: {error}")
        return [], {}
