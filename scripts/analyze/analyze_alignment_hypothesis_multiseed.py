#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "tmp" / "exp_align_hypothesis_multiseed" / "runs.json"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_load_summary(runlog_path: Path):
    info = {
        "pretrain": False,
        "resume_path": None,
        "target": None,
        "loaded": None,
        "missing_after_load": None,
        "shape_mismatch": None,
    }
    if not runlog_path.exists():
        return info
    lines = runlog_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if "src.train.trainer: Epoch 1" in line:
            break
        if "Loading checkpoint from:" in line:
            info["pretrain"] = True
            info["resume_path"] = line.split("Loading checkpoint from:", 1)[1].strip()
        if "Selective load summary:" in line:
            m = re.search(
                r"target_in_model=(\d+), matched_in_checkpoint=(\d+), loaded=(\d+), missing_after_load=(\d+), shape_mismatch=(\d+)",
                line,
            )
            if m:
                info["target"] = int(m.group(1))
                info["loaded"] = int(m.group(3))
                info["missing_after_load"] = int(m.group(4))
                info["shape_mismatch"] = int(m.group(5))
    return info


def _cohens_d(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return None
    return (ma - mb) / math.sqrt(pooled)


def _bootstrap_diff_ci(
    a: List[float], b: List[float], n_boot: int = 5000, seed: int = 42
) -> Optional[Tuple[float, float, float]]:
    if not a or not b:
        return None
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(mean(sa) - mean(sb))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    md = diffs[int(0.5 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, md, hi


def _describe(vals: List[float]) -> str:
    if not vals:
        return "n=0"
    if len(vals) == 1:
        return f"n=1 mean={vals[0]:.6f}"
    return f"n={len(vals)} mean={mean(vals):.6f} std={stdev(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze multiseed alignment hypothesis results.")
    parser.add_argument("--runs-json", type=str, default=str(DEFAULT_RUNS))
    parser.add_argument("--metric-clf", type=str, default="R", choices=["R", "f1", "auc", "pr_auc"])
    parser.add_argument("--metric-reg", type=str, default="MAE", choices=["MAE", "RMSE", "R2", "PearsonR"])
    args = parser.parse_args()

    runs_path = Path(args.runs_json)
    if not runs_path.exists():
        raise FileNotFoundError(f"未找到 runs 文件: {runs_path}")

    runs = _read_json(runs_path)
    rows = []
    for run in runs:
        ckpt_dir = ROOT / run["checkpoint_dir"]
        runlog = ckpt_dir / "run.log"
        load = _parse_load_summary(runlog)
        metrics_path = ckpt_dir / "metrics_test_best_1.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        rows.append(
            {
                **run,
                "pretrain": load["pretrain"],
                "target": load["target"],
                "loaded": load["loaded"],
                "missing_after_load": load["missing_after_load"],
                "shape_mismatch": load["shape_mismatch"],
                "complete_load": (
                    None
                    if load["target"] is None
                    else (
                        load["loaded"] == load["target"]
                        and load["missing_after_load"] == 0
                        and load["shape_mismatch"] == 0
                    )
                ),
                "metrics": metrics,
            }
        )

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    print("=== Variant Summary ===")
    for variant, items in sorted(by_variant.items()):
        model = items[0]["model"]
        metric_key = args.metric_clf if model == "clf_mixer_attnpl_t" else args.metric_reg
        vals = [it["metrics"].get(metric_key) for it in items if isinstance(it["metrics"].get(metric_key), (int, float))]
        complete = [it["complete_load"] for it in items if it["complete_load"] is not None]
        complete_rate = (sum(1 for x in complete if x) / len(complete)) if complete else None
        load_text = "-" if complete_rate is None else f"{complete_rate:.2%}"
        print(f"- {variant} ({model}) | {metric_key}: {_describe(vals)} | complete_load_rate={load_text}")

    def get_vals(variant: str, metric: str) -> List[float]:
        return [
            r["metrics"].get(metric)
            for r in by_variant.get(variant, [])
            if isinstance(r["metrics"].get(metric), (int, float))
        ]

    print("\n=== Hypothesis Checks ===")
    # H1
    h1_a = get_vals("reg_mismatch_pre", args.metric_reg)
    h1_b = get_vals("reg_aligned_pre", args.metric_reg)
    if h1_a and h1_b:
        ci = _bootstrap_diff_ci(h1_a, h1_b)
        d = _cohens_d(h1_a, h1_b)
        print(f"H1 (reg mismatch-pre vs aligned-pre) on {args.metric_reg}:")
        print(f"  mismatch: {_describe(h1_a)}")
        print(f"  aligned : {_describe(h1_b)}")
        if ci:
            print(f"  diff(mean mismatch - aligned) 95%CI: [{ci[0]:.6f}, {ci[2]:.6f}] median={ci[1]:.6f}")
        if d is not None:
            print(f"  Cohen's d: {d:.4f}")

    # H2
    h2_a = get_vals("reg_aligned_pre", args.metric_reg)
    h2_b = get_vals("reg_aligned_pre_freeze", args.metric_reg)
    if h2_a and h2_b:
        ci = _bootstrap_diff_ci(h2_a, h2_b)
        d = _cohens_d(h2_a, h2_b)
        print(f"H2 (reg aligned-pre vs aligned-pre-freeze) on {args.metric_reg}:")
        print(f"  no-freeze: {_describe(h2_a)}")
        print(f"  freeze   : {_describe(h2_b)}")
        if ci:
            print(f"  diff(mean no-freeze - freeze) 95%CI: [{ci[0]:.6f}, {ci[2]:.6f}] median={ci[1]:.6f}")
        if d is not None:
            print(f"  Cohen's d: {d:.4f}")

    # H3
    h3_clf_pre = get_vals("clf_aligned_pre", args.metric_clf)
    h3_clf_sc = get_vals("clf_aligned_scratch", args.metric_clf)
    h3_reg_pre = get_vals("reg_aligned_pre", args.metric_reg)
    h3_reg_sc = get_vals("reg_aligned_scratch", args.metric_reg)
    if h3_clf_pre and h3_clf_sc:
        ci = _bootstrap_diff_ci(h3_clf_pre, h3_clf_sc)
        d = _cohens_d(h3_clf_pre, h3_clf_sc)
        print(f"H3-clf (clf aligned-pre vs aligned-scratch) on {args.metric_clf}:")
        print(f"  pretrain: {_describe(h3_clf_pre)}")
        print(f"  scratch : {_describe(h3_clf_sc)}")
        if ci:
            print(f"  diff(mean pre - scratch) 95%CI: [{ci[0]:.6f}, {ci[2]:.6f}] median={ci[1]:.6f}")
        if d is not None:
            print(f"  Cohen's d: {d:.4f}")
    if h3_reg_pre and h3_reg_sc:
        ci = _bootstrap_diff_ci(h3_reg_pre, h3_reg_sc)
        d = _cohens_d(h3_reg_pre, h3_reg_sc)
        print(f"H3-reg (reg aligned-pre vs aligned-scratch) on {args.metric_reg}:")
        print(f"  pretrain: {_describe(h3_reg_pre)}")
        print(f"  scratch : {_describe(h3_reg_sc)}")
        if ci:
            print(f"  diff(mean pre - scratch) 95%CI: [{ci[0]:.6f}, {ci[2]:.6f}] median={ci[1]:.6f}")
        if d is not None:
            print(f"  Cohen's d: {d:.4f}")

    print("\n提示: 对 MAE/RMSE，越低越好；对 R/F1/AUC，越高越好。")


if __name__ == "__main__":
    main()
