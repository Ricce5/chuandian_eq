#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_csv(raw: str, cast_fn):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [cast_fn(x) for x in items]


def set_key(cfg, dotted_key: str, value):
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or node[key] is None:
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def parse_bool_text(raw: str) -> bool:
    norm = str(raw).strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool text: {raw!r}")


def parse_mapping(raw: str, key_cast, value_cast):
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


def load_exp_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported exp config suffix: {suffix}. Use .yaml/.yml/.json")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Experiment config must be a mapping/object.")
    return data


def as_csv_str(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def mapping_to_csv(mapping):
    if mapping is None:
        return None
    if isinstance(mapping, str):
        return mapping
    if not isinstance(mapping, dict):
        raise ValueError(f"mapping value must be dict or str, got: {type(mapping)}")
    parts = []
    for key, value in mapping.items():
        parts.append(f"{key}:{value}")
    return ",".join(parts)


def apply_exp_config_overrides(args, exp_cfg: dict, direct_key_map: dict, csv_like_keys: dict, mapping_like_keys: dict):
    for key, arg_name in direct_key_map.items():
        if key in exp_cfg:
            setattr(args, arg_name, exp_cfg[key])

    for key, arg_name in csv_like_keys.items():
        if key in exp_cfg:
            setattr(args, arg_name, as_csv_str(exp_cfg[key]))

    for key, arg_name in mapping_like_keys.items():
        if key in exp_cfg:
            setattr(args, arg_name, mapping_to_csv(exp_cfg[key]))

    if "set" in exp_cfg:
        set_val = exp_cfg["set"]
        if isinstance(set_val, dict):
            args.set = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in set_val.items()]
        elif isinstance(set_val, list):
            args.set = [str(x) for x in set_val]
        elif isinstance(set_val, str):
            args.set = [set_val]
        else:
            raise ValueError("exp config key `set` must be dict/list/str.")

    return args


def run_cmd(cmd, cwd: Path, env=None, log_path: Path = None):
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


def parse_ckpt_selects(raw):
    text = str(raw or "best").strip().lower()
    if text in {"both", "all"}:
        return ["best", "last"]

    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        return ["best"]

    valid = {"best", "last"}
    out = []
    seen = set()
    for item in items:
        if item not in valid:
            raise ValueError(
                f"Unsupported ckpt_select item: {item!r}. Use best, last, both, or comma list."
            )
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def run_single_task(task, args, repo_root: Path):
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
    test_returncodes = {}
    test_skipped_reasons = {}

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
        train_ret = run_cmd(train_cmd, repo_root, env=env, log_path=log_path)
        train_ret_code = int(train_ret.returncode)

    can_run_test = args.run_test and (args.skip_train or train_ret_code == 0)
    if args.run_test and not can_run_test:
        test_skipped_reason = "train_failed"
        for ckpt_select in parse_ckpt_selects(getattr(args, "ckpt_select", "best")):
            test_skipped_reasons[ckpt_select] = "train_failed"

    if can_run_test:
        selected_ckpts = parse_ckpt_selects(getattr(args, "ckpt_select", "best"))
        for ckpt_select in selected_ckpts:
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
                ckpt_select,
            ]
            test_ret = run_cmd(test_cmd, repo_root, env=env, log_path=log_path)
            test_returncodes[ckpt_select] = int(test_ret.returncode)
        test_ret_code = max(test_returncodes.values()) if test_returncodes else None

    result = dict(task)
    result["train_returncode"] = train_ret_code
    result["test_returncode"] = test_ret_code
    result["test_skipped_reason"] = test_skipped_reason
    result["test_returncodes"] = test_returncodes
    result["test_skipped_reasons"] = test_skipped_reasons
    result["test_ckpt_selects"] = parse_ckpt_selects(getattr(args, "ckpt_select", "best"))
    return result


def _upsert_summary_row(summary, task_result, key_field: str = "run"):
    key_val = task_result.get(key_field)
    if key_val is None:
        summary.append(task_result)
        return
    for idx, row in enumerate(summary):
        if isinstance(row, dict) and row.get(key_field) == key_val:
            summary[idx] = task_result
            return
    summary.append(task_result)


def execute_tasks(
    tasks,
    args,
    repo_root: Path,
    exp_dir: Path,
    gpu_ids,
    max_parallel: int,
    jobs_per_gpu: int,
    stop_on_error: bool,
    initial_summary=None,
):
    summary = list(initial_summary) if initial_summary else []
    any_failed = False

    if max_parallel == 1:
        for task in tasks:
            task_result = run_single_task(task, args, repo_root)
            _upsert_summary_row(summary, task_result)
            with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            failed = (task_result.get("train_returncode") not in {None, 0}) or (
                task_result.get("test_returncode") not in {None, 0}
            )
            if failed:
                any_failed = True
                if stop_on_error:
                    print(f"[STOP] stop_on_error enabled, failed run: {task_result['run']}")
                    break
        return summary, any_failed

    executor = ThreadPoolExecutor(max_workers=max_parallel)
    futures = {}
    running_count_by_gpu = {gpu: 0 for gpu in gpu_ids}
    pending = list(tasks)
    try:
        while pending or futures:
            while pending and len(futures) < max_parallel and (not (stop_on_error and any_failed)):
                task = pending.pop(0)
                assigned_gpu = None
                if gpu_ids:
                    available = [gpu for gpu in gpu_ids if running_count_by_gpu[gpu] < jobs_per_gpu]
                    if not available:
                        break
                    assigned_gpu = min(available, key=lambda gpu: running_count_by_gpu[gpu])
                    running_count_by_gpu[assigned_gpu] += 1
                task_submit = dict(task)
                if assigned_gpu is not None:
                    task_submit["cuda_id"] = int(assigned_gpu)
                future = executor.submit(run_single_task, task_submit, args, repo_root)
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
            if gpu is not None and gpu in running_count_by_gpu:
                running_count_by_gpu[gpu] = max(0, running_count_by_gpu[gpu] - 1)

            task_result = finished_future.result()
            _upsert_summary_row(summary, task_result)
            with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            failed = (task_result.get("train_returncode") not in {None, 0}) or (
                task_result.get("test_returncode") not in {None, 0}
            )
            if failed:
                any_failed = True
                if stop_on_error:
                    print(f"[STOP] stop_on_error enabled, failed run: {task_result['run']}")
    finally:
        executor.shutdown(wait=True)

    return summary, any_failed

