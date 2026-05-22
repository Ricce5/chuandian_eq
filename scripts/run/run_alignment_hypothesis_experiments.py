#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "tmp" / "exp_align_hypothesis"


@dataclass
class Experiment:
    name: str
    model: str
    config_path: Path
    notes: str


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _set_nested(mapping: Dict, key_path: List[str], value):
    node = mapping
    for key in key_path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[key_path[-1]] = value


def _del_if_exists(mapping: Dict, key: str):
    if key in mapping:
        del mapping[key]


def build_experiment_configs(pretrain_ckpt: str) -> List[Experiment]:
    reg_base = _load_yaml(ROOT / "config" / "reg_mixer_attnpl_t.yaml")
    clf_base = _load_yaml(ROOT / "config" / "clf_mixer_attnpl_t.yaml")

    experiments: List[Experiment] = []

    # ========== Regression ==========
    # A) mismatch + pretrain (baseline)
    reg_mismatch_pre = json.loads(json.dumps(reg_base))
    reg_mismatch_pre["resume_path"] = pretrain_ckpt
    reg_mismatch_pre["load_specific_parts"] = ["encoder"]
    _del_if_exists(reg_mismatch_pre, "freeze_parts")
    reg_mismatch_pre["freeze_loaded_only"] = False
    reg_mismatch_pre["seed"] = 1
    p = TMP_DIR / "reg_mismatch_pre.yaml"
    _dump_yaml(p, reg_mismatch_pre)
    experiments.append(
        Experiment(
            name="reg_mismatch_pre",
            model="reg_mixer_attnpl_t",
            config_path=p,
            notes="结构不对齐(use_conv=true) + 预训练 + 不冻结",
        )
    )

    # B) aligned + pretrain + no freeze
    reg_aligned_pre = json.loads(json.dumps(reg_base))
    reg_aligned_pre["resume_path"] = pretrain_ckpt
    reg_aligned_pre["load_specific_parts"] = ["encoder"]
    _del_if_exists(reg_aligned_pre, "freeze_parts")
    reg_aligned_pre["freeze_loaded_only"] = False
    _set_nested(reg_aligned_pre, ["mixer_model_config", "ssm_cfg", "use_conv"], False)
    _set_nested(
        reg_aligned_pre,
        ["mixer_model_config", "ssm_cfg", "rotary_emb_scale_base"],
        8192,
    )
    _set_nested(
        reg_aligned_pre,
        ["mixer_model_config", "attn_cfg", "rotary_emb_scale_base"],
        8192,
    )
    _set_nested(reg_aligned_pre, ["mixer_model_config", "input_init_scale"], 1.5)
    _set_nested(reg_aligned_pre, ["mixer_model_config", "initializer_cfg"], {"initializer_range": 0.02})
    reg_aligned_pre["seed"] = 1
    p = TMP_DIR / "reg_aligned_pre.yaml"
    _dump_yaml(p, reg_aligned_pre)
    experiments.append(
        Experiment(
            name="reg_aligned_pre",
            model="reg_mixer_attnpl_t",
            config_path=p,
            notes="结构对齐(use_conv=false, rotary=8192) + 预训练 + 不冻结",
        )
    )

    # C) aligned + scratch + no freeze
    reg_aligned_scratch = json.loads(json.dumps(reg_aligned_pre))
    _del_if_exists(reg_aligned_scratch, "resume_path")
    _del_if_exists(reg_aligned_scratch, "load_specific_parts")
    _del_if_exists(reg_aligned_scratch, "freeze_parts")
    reg_aligned_scratch["freeze_loaded_only"] = False
    reg_aligned_scratch["seed"] = 1
    p = TMP_DIR / "reg_aligned_scratch.yaml"
    _dump_yaml(p, reg_aligned_scratch)
    experiments.append(
        Experiment(
            name="reg_aligned_scratch",
            model="reg_mixer_attnpl_t",
            config_path=p,
            notes="结构对齐 + 随机初始化 + 不冻结",
        )
    )

    # D) aligned + pretrain + freeze loaded encoder
    reg_aligned_pre_freeze = json.loads(json.dumps(reg_aligned_pre))
    reg_aligned_pre_freeze["freeze_parts"] = ["encoder"]
    reg_aligned_pre_freeze["freeze_loaded_only"] = True
    reg_aligned_pre_freeze["seed"] = 1
    p = TMP_DIR / "reg_aligned_pre_freeze.yaml"
    _dump_yaml(p, reg_aligned_pre_freeze)
    experiments.append(
        Experiment(
            name="reg_aligned_pre_freeze",
            model="reg_mixer_attnpl_t",
            config_path=p,
            notes="结构对齐 + 预训练 + 冻结已加载encoder",
        )
    )

    # ========== Classification ==========
    # E) aligned + pretrain
    clf_aligned_pre = json.loads(json.dumps(clf_base))
    clf_aligned_pre["resume_path"] = pretrain_ckpt
    clf_aligned_pre["load_specific_parts"] = ["encoder"]
    _del_if_exists(clf_aligned_pre, "freeze_parts")
    clf_aligned_pre["freeze_loaded_only"] = False
    _set_nested(clf_aligned_pre, ["mixer_model_config", "n_layer"], 2)
    _set_nested(clf_aligned_pre, ["mixer_model_config", "attn_layer_idx"], [])
    _set_nested(clf_aligned_pre, ["mixer_model_config", "ssm_cfg", "use_conv"], False)
    _set_nested(
        clf_aligned_pre,
        ["mixer_model_config", "ssm_cfg", "rotary_emb_scale_base"],
        8192,
    )
    _set_nested(
        clf_aligned_pre,
        ["mixer_model_config", "attn_cfg", "rotary_emb_scale_base"],
        8192,
    )
    clf_aligned_pre["seed"] = 1
    p = TMP_DIR / "clf_aligned_pre.yaml"
    _dump_yaml(p, clf_aligned_pre)
    experiments.append(
        Experiment(
            name="clf_aligned_pre",
            model="clf_mixer_attnpl_t",
            config_path=p,
            notes="结构对齐 + 预训练",
        )
    )

    # F) aligned + scratch
    clf_aligned_scratch = json.loads(json.dumps(clf_aligned_pre))
    _del_if_exists(clf_aligned_scratch, "resume_path")
    _del_if_exists(clf_aligned_scratch, "load_specific_parts")
    _del_if_exists(clf_aligned_scratch, "freeze_parts")
    clf_aligned_scratch["seed"] = 1
    p = TMP_DIR / "clf_aligned_scratch.yaml"
    _dump_yaml(p, clf_aligned_scratch)
    experiments.append(
        Experiment(
            name="clf_aligned_scratch",
            model="clf_mixer_attnpl_t",
            config_path=p,
            notes="结构对齐 + 随机初始化",
        )
    )

    # metadata
    metadata = {
        "pretrain_ckpt": pretrain_ckpt,
        "experiments": [
            {
                "name": e.name,
                "model": e.model,
                "config": str(e.config_path.relative_to(ROOT)),
                "notes": e.notes,
            }
            for e in experiments
        ],
    }
    with (TMP_DIR / "plan.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return experiments


def run_cmd(cmd: List[str], cwd: Path):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def find_new_checkpoint(model: str, before: set) -> Path:
    candidates = sorted((ROOT / "checkpoints").glob(f"{model}_*"), key=lambda p: p.stat().st_mtime)
    after = [c for c in candidates if str(c) not in before]
    if not after:
        raise RuntimeError(f"未找到新checkpoint目录: {model}")
    return after[-1]


def run_experiments(experiments: List[Experiment], python_bin: str = sys.executable):
    results = []
    for exp in experiments:
        before = {str(p) for p in (ROOT / "checkpoints").glob(f"{exp.model}_*")}
        train_cmd = [
            python_bin,
            "main.py",
            "--model",
            exp.model,
            "--mode",
            "train",
            "--config",
            str(exp.config_path.relative_to(ROOT)),
        ]
        run_cmd(train_cmd, ROOT)

        ckpt_dir = find_new_checkpoint(exp.model, before)
        test_cmd = [
            python_bin,
            "main.py",
            "--model",
            exp.model,
            "--mode",
            "test",
            "--checkpoint_dir",
            str(ckpt_dir),
            "--ckpt_select",
            "best",
        ]
        run_cmd(test_cmd, ROOT)

        results.append(
            {
                "name": exp.name,
                "model": exp.model,
                "config": str(exp.config_path.relative_to(ROOT)),
                "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)),
                "notes": exp.notes,
            }
        )

    with (TMP_DIR / "runs.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n所有实验运行完成，结果索引已写入:", TMP_DIR / "runs.json")


def main():
    parser = argparse.ArgumentParser(description="Run alignment hypothesis experiments for clf/reg mixer models.")
    parser.add_argument(
        "--pretrain-ckpt",
        type=str,
        default="./checkpoints/mixer_tpp_20260506-113707/last_model_1.pth",
        help="预训练权重路径",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable path",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="只生成配置，不执行训练/测试",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前清空 tmp/exp_align_hypothesis",
    )
    args = parser.parse_args()

    if args.clean and TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_experiment_configs(args.pretrain_ckpt)
    print("已生成配置:")
    for e in experiments:
        print("-", e.name, "->", e.config_path.relative_to(ROOT))

    if args.generate_only:
        print("generate-only模式，未执行训练。")
        return

    run_experiments(experiments, python_bin=args.python)


if __name__ == "__main__":
    main()
