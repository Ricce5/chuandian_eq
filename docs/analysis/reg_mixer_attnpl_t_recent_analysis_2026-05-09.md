# `reg_mixer_attnpl_t` 近期实验结果分析（2026-05-09）

## 1. 分析范围

- 时间范围：`2026-05-07` ~ `2026-05-09`
- 模型：`reg_mixer_attnpl_t`
- 主要数据来源：
  - `checkpoints/reg_mixer_attnpl_t_*`
  - `tmp/exp_align_hypothesis/runs.json`
  - `experiment_record.md`

说明：截至本次整理时，时间上最新的 checkpoint 为 `checkpoints/reg_mixer_attnpl_t_20260509-211257`（2026-05-09 21:12:57），但其性能并非最优。

---

## 2. 核心结论

1. **当晚最稳定且效果较好的一组**为：
   - `checkpoints/reg_mixer_attnpl_t_20260509-190327`
   - `checkpoints/reg_mixer_attnpl_t_20260509-205627`
   - `checkpoints/reg_mixer_attnpl_t_20260509-205706`

   这组共同特征是：`use_conv=true`、带预训练、但 selective load 为 `30/44`（存在缺失但无 shape mismatch）。

2. **明显退化的一组**是：
   - `checkpoints/reg_mixer_attnpl_t_20260509-191428`
   - `checkpoints/reg_mixer_attnpl_t_20260509-211257`

   二者都出现 `24/44` 且 `shape_mismatch=2`，性能显著下降。

3. 在 `tmp/exp_align_hypothesis/runs.json` 对应的一组“结构对齐假设实验”中：
   - `reg_aligned_pre` 并未优于 `reg_aligned_scratch`
   - `reg_mismatch_pre` 反而给出更低 MAE

   即：当前证据下，“结构对齐 + 预训练”没有稳定带来回归收益。

---

## 3. 关键指标摘要

### 3.1 当晚稳定组（推荐作为近期主结果参考）

| checkpoint | seed | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| `reg_mixer_attnpl_t_20260509-190327` | 1 | 0.631013 | 0.761169 | -0.731376 |
| `reg_mixer_attnpl_t_20260509-205627` | 0 | 0.665574 | 0.771867 | -0.780387 |
| `reg_mixer_attnpl_t_20260509-205706` | 2 | 0.648401 | 0.758748 | -0.720381 |

- MAE 均值：`0.648329`
- MAE 标准差：`0.017281`

### 3.2 退化组（加载质量问题明显）

| checkpoint | selective load | shape_mismatch | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|
| `reg_mixer_attnpl_t_20260509-191428` | 24/44 | 2 | 0.952032 | 1.108700 | -2.673310 |
| `reg_mixer_attnpl_t_20260509-211257` | 24/44 | 2 | 0.752130 | 0.870343 | -1.263652 |

### 3.3 结构对齐假设实验（`tmp/exp_align_hypothesis/runs.json`）

| variant | checkpoint | MAE | RMSE | R2 | 备注 |
|---|---|---:|---:|---:|---|
| `reg_mismatch_pre` | `20260509-162615` | 0.615716 | 0.716441 | -0.533877 | use_conv=true + 预训练 |
| `reg_aligned_pre` | `20260509-163252` | 0.703184 | 0.785909 | -0.845754 | use_conv=false + 预训练 |
| `reg_aligned_scratch` | `20260509-163910` | 0.619548 | 0.735449 | -0.616345 | use_conv=false + 随机初始化 |
| `reg_aligned_pre_freeze` | `20260509-164531` | 0.683813 | 0.778492 | -0.811080 | use_conv=false + 预训练 + freeze |

---

## 4. 诊断与解释

1. 当前性能波动更像是**加载覆盖率/匹配质量问题**，而不仅是“是否使用预训练”。
2. 当 `target_in_model` 扩到 44 且 only-load-encoder 时，若 `loaded` 进一步下降并出现 `shape_mismatch`，性能会明显劣化。
3. `encoder_param_keywords` 配置方式变化（例如从 `['encoder']` 切到更窄的键）与预训练源 checkpoint 切换，会放大不稳定性。

---

## 5. 建议（按优先级）

1. 固定近期主线配置，优先参考：
   - `checkpoints/reg_mixer_attnpl_t_20260509-205706/config.yaml`
2. 在训练启动时加入硬约束：
   - 若 selective load 覆盖率过低或存在 shape mismatch，直接中止本次训练。
3. 对外报告时，优先采用多次复现实验均值（如 `190327/205627/205706`），避免使用单次“最新”结果代表整体效果。

---

## 6. 附注

- 本文为阶段性记录，用于快速回顾 `2026-05-09` 当日回归实验状态。
- 若后续继续追加 seed 与配置组合，建议新增一份“自动汇总表”并滚动更新本结论。
