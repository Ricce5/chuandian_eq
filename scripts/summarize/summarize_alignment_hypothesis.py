#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "tmp" / "exp_align_hypothesis"


def _read_json(path: Path) -> Dict:
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
        "freeze_summary": None,
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
        if "Freeze summary" in line:
            info["freeze_summary"] = line.split("Freeze summary", 1)[1].strip()
    return info


def _read_metrics(model: str, ckpt_dir: Path) -> Dict:
    metrics_path = ckpt_dir / "metrics_test_best_1.json"
    if not metrics_path.exists():
        return {}
    return _read_json(metrics_path)


def _is_complete_load(info: Dict) -> Optional[bool]:
    if info["target"] is None or info["loaded"] is None:
        return None
    return (
        info["loaded"] == info["target"]
        and info["missing_after_load"] == 0
        and info["shape_mismatch"] == 0
    )


def _fmt_float(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _collect_rows(runs):
    rows = []
    for run in runs:
        ckpt_dir = ROOT / run["checkpoint_dir"]
        runlog = ckpt_dir / "run.log"
        info = _parse_load_summary(runlog)
        metrics = _read_metrics(run["model"], ckpt_dir)
        row = {
            "name": run["name"],
            "model": run["model"],
            "checkpoint_dir": run["checkpoint_dir"],
            "notes": run.get("notes", ""),
            "pretrain": info["pretrain"],
            "resume_path": info["resume_path"],
            "target": info["target"],
            "loaded": info["loaded"],
            "missing_after_load": info["missing_after_load"],
            "shape_mismatch": info["shape_mismatch"],
            "complete_load": _is_complete_load(info),
            "freeze_summary": info["freeze_summary"],
            "metrics": metrics,
        }
        rows.append(row)
    return rows


def _summarize(rows):
    by_model = {"clf_mixer_attnpl_t": [], "reg_mixer_attnpl_t": []}
    for r in rows:
        by_model[r["model"]].append(r)

    print("=== 实验明细 ===")
    for r in rows:
        if r["model"] == "clf_mixer_attnpl_t":
            m = r["metrics"]
            metric_line = f"R={_fmt_float(m.get('R'))}, F1={_fmt_float(m.get('f1'))}, AUC={_fmt_float(m.get('auc'))}"
        else:
            m = r["metrics"]
            metric_line = f"MAE={_fmt_float(m.get('MAE'))}, RMSE={_fmt_float(m.get('RMSE'))}, R2={_fmt_float(m.get('R2'))}"
        print(
            f"- {r['name']} | pretrain={r['pretrain']} | complete_load={r['complete_load']} "
            f"| load={r['loaded']}/{r['target']} miss={r['missing_after_load']} shape={r['shape_mismatch']} | {metric_line}"
        )

    print("\n=== 分类结论辅助 ===")
    clf_rows = by_model["clf_mixer_attnpl_t"]
    clf_pre = [r for r in clf_rows if r["pretrain"]]
    clf_scratch = [r for r in clf_rows if not r["pretrain"]]
    if clf_pre and clf_scratch:
        pre_R = [r["metrics"].get("R") for r in clf_pre if isinstance(r["metrics"].get("R"), (int, float))]
        sc_R = [r["metrics"].get("R") for r in clf_scratch if isinstance(r["metrics"].get("R"), (int, float))]
        if pre_R and sc_R:
            print(f"- pretrain R均值={mean(pre_R):.6f}, scratch R均值={mean(sc_R):.6f}")
            print(f"- pretrain R最好={max(pre_R):.6f}, scratch R最好={max(sc_R):.6f}")

    print("\n=== 回归结论辅助 ===")
    reg_rows = by_model["reg_mixer_attnpl_t"]
    reg_pre = [r for r in reg_rows if r["pretrain"]]
    reg_scratch = [r for r in reg_rows if not r["pretrain"]]
    if reg_pre and reg_scratch:
        pre_mae = [r["metrics"].get("MAE") for r in reg_pre if isinstance(r["metrics"].get("MAE"), (int, float))]
        sc_mae = [r["metrics"].get("MAE") for r in reg_scratch if isinstance(r["metrics"].get("MAE"), (int, float))]
        pre_rmse = [r["metrics"].get("RMSE") for r in reg_pre if isinstance(r["metrics"].get("RMSE"), (int, float))]
        sc_rmse = [r["metrics"].get("RMSE") for r in reg_scratch if isinstance(r["metrics"].get("RMSE"), (int, float))]
        if pre_mae and sc_mae:
            print(f"- pretrain MAE均值={mean(pre_mae):.6f}, scratch MAE均值={mean(sc_mae):.6f}")
            print(f"- pretrain MAE最好(低)= {min(pre_mae):.6f}, scratch MAE最好(低)= {min(sc_mae):.6f}")
        if pre_rmse and sc_rmse:
            print(f"- pretrain RMSE均值={mean(pre_rmse):.6f}, scratch RMSE均值={mean(sc_rmse):.6f}")
            print(f"- pretrain RMSE最好(低)= {min(pre_rmse):.6f}, scratch RMSE最好(低)= {min(sc_rmse):.6f}")

    print("\n=== 解释检验模板 ===")
    print("- H1: 结构不对齐导致回归预训练无效/负迁移")
    print("  验证：比较 reg_mismatch_pre vs reg_aligned_pre（看加载完整度与MAE/RMSE变化）")
    print("- H2: 冻结策略会放大回归负迁移风险")
    print("  验证：比较 reg_aligned_pre vs reg_aligned_pre_freeze")
    print("- H3: 分类更受益于通用时序表示，而回归更依赖任务特定细粒度映射")
    print("  验证：比较 clf_aligned_pre vs clf_aligned_scratch 与 reg_aligned_pre vs reg_aligned_scratch")


def main():
    parser = argparse.ArgumentParser(description="Summarize alignment hypothesis experiments.")
    parser.add_argument(
        "--runs-json",
        type=str,
        default=str(TMP_DIR / "runs.json"),
        help="Path to runs.json produced by run_alignment_hypothesis_experiments.py",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs_json)
    if not runs_path.exists():
        raise FileNotFoundError(f"未找到 runs 索引文件: {runs_path}")

    runs = _read_json(runs_path)
    rows = _collect_rows(runs)
    _summarize(rows)


if __name__ == "__main__":
    main()
