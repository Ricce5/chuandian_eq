#!/usr/bin/env python3
import argparse
import copy
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "exp_align_hypothesis_multiseed"


@dataclass
class Variant:
    name: str
    model: str
    notes: str
    cfg: Dict


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _del_key(d: Dict, key: str):
    if key in d:
        del d[key]


def _set_nested(d: Dict, keys: List[str], value):
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def build_variants(pretrain_ckpt: str) -> List[Variant]:
    reg_base = _load_yaml(ROOT / "config" / "reg_mixer_attnpl_t.yaml")
    clf_base = _load_yaml(ROOT / "config" / "clf_mixer_attnpl_t.yaml")

    variants: List[Variant] = []

    # ---------- regression variants ----------
    reg_mismatch_pre = copy.deepcopy(reg_base)
    reg_mismatch_pre["resume_path"] = pretrain_ckpt
    reg_mismatch_pre["load_specific_parts"] = ["encoder"]
    _del_key(reg_mismatch_pre, "freeze_parts")
    reg_mismatch_pre["freeze_loaded_only"] = False
    variants.append(
        Variant(
            name="reg_mismatch_pre",
            model="reg_mixer_attnpl_t",
            notes="结构不对齐(use_conv=true)+预训练+不冻结",
            cfg=reg_mismatch_pre,
        )
    )

    reg_aligned_pre = copy.deepcopy(reg_base)
    reg_aligned_pre["resume_path"] = pretrain_ckpt
    reg_aligned_pre["load_specific_parts"] = ["encoder"]
    _del_key(reg_aligned_pre, "freeze_parts")
    reg_aligned_pre["freeze_loaded_only"] = False
    _set_nested(reg_aligned_pre, ["mixer_model_config", "ssm_cfg", "use_conv"], False)
    _set_nested(reg_aligned_pre, ["mixer_model_config", "ssm_cfg", "rotary_emb_scale_base"], 8192)
    _set_nested(reg_aligned_pre, ["mixer_model_config", "attn_cfg", "rotary_emb_scale_base"], 8192)
    _set_nested(reg_aligned_pre, ["mixer_model_config", "input_init_scale"], 1.5)
    _set_nested(reg_aligned_pre, ["mixer_model_config", "initializer_cfg"], {"initializer_range": 0.02})
    variants.append(
        Variant(
            name="reg_aligned_pre",
            model="reg_mixer_attnpl_t",
            notes="结构对齐(use_conv=false)+预训练+不冻结",
            cfg=reg_aligned_pre,
        )
    )

    reg_aligned_scratch = copy.deepcopy(reg_aligned_pre)
    _del_key(reg_aligned_scratch, "resume_path")
    _del_key(reg_aligned_scratch, "load_specific_parts")
    _del_key(reg_aligned_scratch, "freeze_parts")
    reg_aligned_scratch["freeze_loaded_only"] = False
    variants.append(
        Variant(
            name="reg_aligned_scratch",
            model="reg_mixer_attnpl_t",
            notes="结构对齐+随机初始化+不冻结",
            cfg=reg_aligned_scratch,
        )
    )

    reg_aligned_pre_freeze = copy.deepcopy(reg_aligned_pre)
    reg_aligned_pre_freeze["freeze_parts"] = ["encoder"]
    reg_aligned_pre_freeze["freeze_loaded_only"] = True
    variants.append(
        Variant(
            name="reg_aligned_pre_freeze",
            model="reg_mixer_attnpl_t",
            notes="结构对齐+预训练+冻结已加载encoder",
            cfg=reg_aligned_pre_freeze,
        )
    )

    # ---------- classification variants ----------
    clf_aligned_pre = copy.deepcopy(clf_base)
    clf_aligned_pre["resume_path"] = pretrain_ckpt
    clf_aligned_pre["load_specific_parts"] = ["encoder"]
    _del_key(clf_aligned_pre, "freeze_parts")
    clf_aligned_pre["freeze_loaded_only"] = False
    _set_nested(clf_aligned_pre, ["mixer_model_config", "n_layer"], 2)
    _set_nested(clf_aligned_pre, ["mixer_model_config", "attn_layer_idx"], [])
    _set_nested(clf_aligned_pre, ["mixer_model_config", "ssm_cfg", "use_conv"], False)
    _set_nested(clf_aligned_pre, ["mixer_model_config", "ssm_cfg", "rotary_emb_scale_base"], 8192)
    _set_nested(clf_aligned_pre, ["mixer_model_config", "attn_cfg", "rotary_emb_scale_base"], 8192)
    variants.append(
        Variant(
            name="clf_aligned_pre",
            model="clf_mixer_attnpl_t",
            notes="结构对齐+预训练",
            cfg=clf_aligned_pre,
        )
    )

    clf_aligned_scratch = copy.deepcopy(clf_aligned_pre)
    _del_key(clf_aligned_scratch, "resume_path")
    _del_key(clf_aligned_scratch, "load_specific_parts")
    _del_key(clf_aligned_scratch, "freeze_parts")
    variants.append(
        Variant(
            name="clf_aligned_scratch",
            model="clf_mixer_attnpl_t",
            notes="结构对齐+随机初始化",
            cfg=clf_aligned_scratch,
        )
    )

    return variants


def _run(cmd: List[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _find_new_ckpt(model: str, before: set) -> Path:
    candidates = sorted((ROOT / "checkpoints").glob(f"{model}_*"), key=lambda p: p.stat().st_mtime)
    after = [p for p in candidates if str(p) not in before]
    if not after:
        raise RuntimeError(f"未找到新checkpoint目录: {model}")
    return after[-1]


def main():
    parser = argparse.ArgumentParser(description="Run multiseed alignment hypothesis experiments.")
    parser.add_argument("--pretrain-ckpt", type=str, default="./checkpoints/mixer_tpp_20260506-113707/last_model_1.pth")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--only-generate", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None, help="debug用途，仅跑前N个variant")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("seeds 不能为空")

    if args.clean and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = build_variants(args.pretrain_ckpt)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    run_index = []
    for variant in variants:
        for seed in seeds:
            cfg = copy.deepcopy(variant.cfg)
            cfg["seed"] = seed
            cfg_name = f"{variant.name}_seed{seed}.yaml"
            cfg_path = OUT_DIR / "configs" / cfg_name
            _dump_yaml(cfg_path, cfg)
            run_index.append(
                {
                    "variant": variant.name,
                    "model": variant.model,
                    "seed": seed,
                    "notes": variant.notes,
                    "config": str(cfg_path.relative_to(ROOT)),
                }
            )

    with (OUT_DIR / "plan.json").open("w", encoding="utf-8") as f:
        json.dump(run_index, f, indent=2, ensure_ascii=False)
    print(f"已生成 {len(run_index)} 个配置: {OUT_DIR / 'plan.json'}")

    if args.only_generate:
        return

    results = []
    for item in run_index:
        model = item["model"]
        cfg_rel = item["config"]
        before = {str(p) for p in (ROOT / "checkpoints").glob(f"{model}_*")}
        _run([args.python, "main.py", "--model", model, "--mode", "train", "--config", cfg_rel])
        ckpt_dir = _find_new_ckpt(model, before)
        _run(
            [
                args.python,
                "main.py",
                "--model",
                model,
                "--mode",
                "test",
                "--checkpoint_dir",
                str(ckpt_dir),
                "--ckpt_select",
                "best",
            ]
        )
        row = dict(item)
        row["checkpoint_dir"] = str(ckpt_dir.relative_to(ROOT))
        results.append(row)

        with (OUT_DIR / "runs.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("全部完成，输出:", OUT_DIR / "runs.json")


if __name__ == "__main__":
    main()
