#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
import yaml


def _parse_list(raw: str, cast_fn):
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


def _run_cmd(cmd, cwd: Path):
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=False)


def _extract_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    keep_keys = ["auc", "pr_auc", "f1", "precision", "recall", "threshold"]
    return {k: metrics.get(k) for k in keep_keys}


def _mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _resolve_checkpoint_path(run_dir: Path, ckpt_select: str):
    if ckpt_select == "best":
        return run_dir / "best_model_1.pth"
    if ckpt_select == "last":
        return run_dir / "last_model_1.pth"
    raise ValueError(f"Unsupported ckpt_select: {ckpt_select}")


def main():
    parser = argparse.ArgumentParser(description="Run small-data learning curve experiments and aggregate metrics.")
    parser.add_argument("--model", type=str, default="clf_mixer_attnpl_t")
    parser.add_argument("--config", type=str, default="config/clf_mixer_attnpl_t.yaml")
    parser.add_argument("--ratios", type=str, default="0.1,0.25,0.5,1.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--out_dir", type=str, default="tmp/small_data_curve")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra config override in dotted form, e.g. --set epochs=50 --set learning_rate=5e-4",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run test/aggregation.")
    parser.add_argument("--test_threshold", type=float, default=None, help="Fixed test threshold.")
    parser.add_argument(
        "--ckpt_select",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Checkpoint selection for test mode.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = out_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    ratios = _parse_list(args.ratios, float)
    seeds = _parse_list(args.seeds, int)
    for r in ratios:
        if not (0.0 < float(r) <= 1.0):
            raise ValueError(f"ratio must be in (0,1], got {r}")

    override_kv = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item!r}. Expected key=value.")
        k, v = item.split("=", 1)
        value = yaml.safe_load(v)
        override_kv[k.strip()] = value

    records = []
    for ratio in ratios:
        for seed in seeds:
            run_name = f"ratio_{ratio:g}_seed_{seed}"
            run_dir = run_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(args.config)
            cfg.seed = int(seed)
            cfg.train_subset_ratio = float(ratio)
            for key, value in override_kv.items():
                _set_key(cfg, key, value)

            cfg_path = run_dir / "learning_curve_config.yaml"
            OmegaConf.save(cfg, str(cfg_path))

            train_failed = False
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
                train_ret = _run_cmd(train_cmd, repo_root)
                if train_ret.returncode != 0:
                    print(f"[WARN] Train failed for {run_name}, returncode={train_ret.returncode}")
                    train_failed = True

            ckpt_path = _resolve_checkpoint_path(run_dir, args.ckpt_select)
            if train_failed or not ckpt_path.exists():
                print(f"[WARN] Skip test for {run_name}: checkpoint missing -> {ckpt_path}")
            else:
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
                if args.test_threshold is not None:
                    test_cmd.extend(["--threshold", str(args.test_threshold)])
                test_ret = _run_cmd(test_cmd, repo_root)
                if test_ret.returncode != 0:
                    print(f"[WARN] Test failed for {run_name}, returncode={test_ret.returncode}")

            metrics_file = run_dir / f"metrics_test_{args.ckpt_select}_1.json"
            metrics = _extract_metrics(metrics_file)
            rec = {
                "ratio": float(ratio),
                "seed": int(seed),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "metrics_file": str(metrics_file),
            }
            if metrics:
                rec.update(metrics)
            records.append(rec)

    detail_csv = out_dir / "curve_detail.csv"
    detail_json = out_dir / "curve_detail.json"
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    detail_fields = ["ratio", "seed", "run_name", "run_dir", "metrics_file", "auc", "pr_auc", "f1", "precision", "recall", "threshold"]
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in detail_fields})

    summary_rows = []
    for ratio in ratios:
        subset = [rec for rec in records if float(rec["ratio"]) == float(ratio)]
        row = {"ratio": float(ratio), "n_runs": len(subset)}
        for metric in ["auc", "pr_auc", "f1", "precision", "recall"]:
            mean, std = _mean_std([rec.get(metric) for rec in subset])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)

    summary_csv = out_dir / "curve_summary.csv"
    summary_json = out_dir / "curve_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_fields = [
        "ratio",
        "n_runs",
        "auc_mean",
        "auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nDone. Detail: {detail_csv}")
    print(f"Done. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
