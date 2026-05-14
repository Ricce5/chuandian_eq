#!/usr/bin/env bash
set -euo pipefail

# Sweep training hyperparameters for clf_mf_tf_grid.
# NOTE: This script only changes training hyperparameters via --set.
#       It does NOT change model structure or preload strategy.

EXP_CONFIG="${EXP_CONFIG:-config/experiments/clf_mf_tf_grid.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_ROOT="${EXP_ROOT:-experiments/clf_trainhp}"
RUN_EXP_CONFIG="${RUN_EXP_CONFIG:-}"

# You can override these arrays directly in this file,
# or export env vars and edit here to read from env if needed.
LRS=(0.0006 0.001 0.0015)
WDS=(0.0001 0.001 0.01)
SCHEDULERS=(step_warmup warmup_linear_decay)
WARMUP_RATIOS=(0.1 0.15 0.2)
BATCH_SIZES=(32 64)
SEEDS="0,1,2"

# Optional global prefix for experiment names.
NAME_PREFIX="${NAME_PREFIX:-hp}"

if [[ -z "${RUN_EXP_CONFIG}" ]]; then
  TMP_EXP_CONFIG="$(mktemp /tmp/clf_trainhp_exp_cfg.XXXXXX.yaml)"
  trap 'rm -f "${TMP_EXP_CONFIG}"' EXIT
  "${PYTHON_BIN}" - "${EXP_CONFIG}" "${TMP_EXP_CONFIG}" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise ValueError(f"Experiment config must be mapping, got: {type(cfg)}")

for key in ("exp_name", "exp_root"):
    cfg.pop(key, None)

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
  RUN_EXP_CONFIG="${TMP_EXP_CONFIG}"
fi

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for sch in "${SCHEDULERS[@]}"; do
      for wr in "${WARMUP_RATIOS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          exp_name="${NAME_PREFIX}_lr${lr}_wd${wd}_${sch}_wr${wr}_bs${bs}"
          exp_name="${exp_name//./p}"

          echo "[RUN] ${exp_name}"
          "${PYTHON_BIN}" scripts/run_clf_mf_tf_grid.py \
            --exp_config "${RUN_EXP_CONFIG}" \
            --exp_root "${EXP_ROOT}" \
            --exp_name "${exp_name}" \
            --seeds "${SEEDS}" \
            --set "learning_rate=${lr}" \
            --set "weight_decay=${wd}" \
            --set "scheduler_type=${sch}" \
            --set "warmup_ratio=${wr}" \
            --set "batch_size=${bs}" \
            --run_test \
            --ckpt_select best
        done
      done
    done
  done
done

echo "Done: training-hyperparameter sweep finished."
