#!/usr/bin/env python3
import argparse
import copy
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VariantSpec:
    exp_name: str
    resume_path: str | None


DEFAULT_VARIANTS = (
    VariantSpec(
        exp_name="clf_pre_AZDX",
        resume_path="./checkpoints/mixer_tpp_20260508-182744/last_model_1.pth",
    ),
    VariantSpec(
        exp_name="clf_pre_AZDX_minus_b",
        resume_path="./checkpoints/mixer_tpp_20260510-211348/last_model_1.pth",
    ),
    VariantSpec(
        exp_name="clf_pre_SCEDC",
        resume_path="./checkpoints/mixer_tpp_20260514-211059/last_model_1.pth",
    ),
    VariantSpec(
        exp_name="clf_pre_SCEDC_minus_b",
        resume_path="./checkpoints/mixer_tpp_20260514-211318/last_model_1.pth",
    ),
    VariantSpec(
        exp_name="clf_pre_scratch",
        resume_path=None,
    ),
)


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/object: {path}")
    return data


def _prepare_run_exp_config(base_exp_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_exp_cfg)
    for key in (
        "exp_name",
        "exp_root",
        "set",
        "seeds",
        "run_test",
        "ckpt_select",
        "resume_exp",
        "skip_done",
    ):
        cfg.pop(key, None)
    return cfg


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run 5 clf-grid experiments in batch: "
            "clf_pre, clf_pre_AZDX_minus_b, clf_pre_SCEDC, "
            "clf_pre_SCEDC_minus_b, clf_pre_scratch."
        )
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="scripts/run/run_clf_mf_tf_grid.py",
        help="Runner script path (repo-relative or absolute).",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="config/experiments/clf_mf_tf_grid.yaml",
        help="Base experiment config used as template.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke runner.",
    )
    parser.add_argument("--exp_root", type=str, default="experiments/clf_pre")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--exp_name_suffix",
        type=str,
        default="",
        help="Optional suffix appended to each exp_name, e.g. v2 -> clf_pre_v2.",
    )
    parser.add_argument("--max_parallel", type=int, default=None)
    parser.add_argument("--jobs_per_gpu", type=int, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--ckpt_select", type=str, default="best")
    parser.add_argument("--run_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_exp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_missing_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow missing pretrained checkpoints (runner will then train from scratch).",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--keep_temp_config",
        action="store_true",
        help="Keep generated sanitized exp config file.",
    )
    return parser


def _variant_name(base_name: str, suffix: str) -> str:
    suffix = str(suffix or "").strip()
    if not suffix:
        return base_name
    return f"{base_name}_{suffix}"


def _resume_ckpt_abs(resume_path: str | None) -> Path | None:
    if not resume_path:
        return None
    path = Path(resume_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _run_one(args, run_cfg_path: Path, variant: VariantSpec, idx: int, total: int):
    exp_name = _variant_name(variant.exp_name, args.exp_name_suffix)

    command = [
        args.python_bin,
        str(_resolve_repo_path(args.runner)),
        "--exp_config",
        str(run_cfg_path),
        "--exp_root",
        args.exp_root,
        "--exp_name",
        exp_name,
        "--seeds",
        args.seeds,
    ]

    if args.run_test:
        command.extend(["--run_test", "--ckpt_select", args.ckpt_select])
    if args.resume_exp:
        command.append("--resume_exp")
    if args.skip_done:
        command.append("--skip_done")
    if args.max_parallel is not None:
        command.extend(["--max_parallel", str(args.max_parallel)])
    if args.jobs_per_gpu is not None:
        command.extend(["--jobs_per_gpu", str(args.jobs_per_gpu)])
    if args.gpu_ids is not None:
        command.extend(["--gpu_ids", str(args.gpu_ids)])

    if variant.resume_path:
        command.extend(
            [
                "--set",
                f"resume_path={variant.resume_path}",
                "--set",
                "load_specific_parts=[encoder]",
                "--set",
                "encoder_param_keywords=[encoder]",
            ]
        )
    else:
        command.extend(
            [
                "--set",
                "resume_path=null",
                "--set",
                "load_specific_parts=null",
                "--set",
                "encoder_param_keywords=null",
            ]
        )

    print(f"\n[RUN {idx}/{total}] {exp_name}")
    print("[CMD]", shlex.join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    base_exp_cfg_path = _resolve_repo_path(args.exp_config)
    base_exp_cfg = _load_yaml_mapping(base_exp_cfg_path)
    run_exp_cfg = _prepare_run_exp_config(base_exp_cfg)

    for variant in DEFAULT_VARIANTS:
        ckpt_path = _resume_ckpt_abs(variant.resume_path)
        if ckpt_path is None:
            continue
        if ckpt_path.exists():
            continue
        if args.allow_missing_resume:
            print(f"[WARN] Missing checkpoint, will fallback to scratch: {ckpt_path}")
            continue
        raise FileNotFoundError(
            f"Checkpoint not found for {variant.exp_name}: {ckpt_path}\n"
            "Use --allow_missing_resume to continue anyway."
        )

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="clf_",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    yaml.safe_dump(run_exp_cfg, tmp, allow_unicode=True, sort_keys=False)
    tmp.flush()
    tmp.close()

    print(f"[INFO] Prepared run config: {tmp_path}")
    print(f"[INFO] Template source: {base_exp_cfg_path}")

    try:
        total = len(DEFAULT_VARIANTS)
        for idx, variant in enumerate(DEFAULT_VARIANTS, start=1):
            _run_one(args, tmp_path, variant, idx=idx, total=total)
    finally:
        if args.keep_temp_config:
            print(f"[INFO] Keep temp config: {tmp_path}")
        else:
            tmp_path.unlink(missing_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
