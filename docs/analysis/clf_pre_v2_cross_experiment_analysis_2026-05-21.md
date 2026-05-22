# clf_pre_v2 分析（2026-05-21）

## 范围
- 目录：`experiments/clf_pre_v2`
- 子实验：`clf_pre_AZDX_v2`、`clf_pre_AZDX_minus_b_v2`、`clf_pre_SCEDC_v2`、`clf_pre_SCEDC_minus_b_v2`、`clf_pre_scratch_v2`
- 口径：单实验汇总用 `summarize_clf_mf_tf_grid.py`，跨实验做横向比较

## 数据完整性
- 5 个实验均为 `25` 个 run
- 所有 run 均有有效结果：`status_ok = 100%`，`metrics_found = 100%`
- 所以下面的统计可视为完整样本

## 横向结果（按全网格平均）
| rank | experiment | auc_mean_over_grid | f1_mean_over_grid | R_mean_over_grid | best_auc_mean | best_point | auc_std_over_grid_mean |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | `clf_pre_scratch_v2` | 0.771660 | 0.481955 | 0.155797 | 0.903283 | `(60, 5.0)` | 0.049663 |
| 2 | `clf_pre_AZDX_v2` | 0.766081 | 0.511755 | 0.200173 | 0.874537 | `(60, 5.0)` | 0.043706 |
| 3 | `clf_pre_SCEDC_v2` | 0.747797 | 0.484476 | 0.172268 | 0.844150 | `(60, 5.0)` | 0.068107 |
| 4 | `clf_pre_SCEDC_minus_b_v2` | 0.743154 | 0.509267 | 0.195951 | 0.824116 | `(60, 5.0)` | 0.068906 |
| 5 | `clf_pre_AZDX_minus_b_v2` | 0.719414 | 0.484222 | 0.124807 | 0.878367 | `(60, 5.0)` | 0.116009 |

## 点位赢家（按 `auc_mean`）
| Tfore | Mf | winner | auc_mean | runner-up | gap |
|---|---:|---|---:|---|---:|
| 10 | 4.0 | `clf_pre_scratch_v2` | 0.778071 | `clf_pre_AZDX_v2` | 0.012807 |
| 20 | 4.5 | `clf_pre_AZDX_v2` | 0.792566 | `clf_pre_scratch_v2` | 0.007231 |
| 30 | 4.5 | `clf_pre_AZDX_v2` | 0.815736 | `clf_pre_scratch_v2` | 0.004632 |
| 60 | 5.0 | `clf_pre_scratch_v2` | 0.903283 | `clf_pre_AZDX_minus_b_v2` | 0.024916 |
| 90 | 5.5 | `clf_pre_SCEDC_v2` | 0.625575 | `clf_pre_SCEDC_minus_b_v2` | 0.024094 |

## `auc < 0.5` 的 run
- 共 `4` 个
- 全部集中在 `(Tfore=90, Mf=5.5)`，说明长预测窗口仍是主要风险位点

| experiment | run | seed | Tfore | Mf | auc |
|---|---|---:|---:|---:|---:|
| `clf_pre_AZDX_minus_b_v2` | `tf_90_mf_5p5_seed_4` | 4 | 90 | 5.5 | 0.467446 |
| `clf_pre_AZDX_minus_b_v2` | `tf_90_mf_5p5_seed_7` | 7 | 90 | 5.5 | 0.319298 |
| `clf_pre_SCEDC_v2` | `tf_90_mf_5p5_seed_4` | 4 | 90 | 5.5 | 0.459259 |
| `clf_pre_scratch_v2` | `tf_90_mf_5p5_seed_0` | 0 | 90 | 5.5 | 0.393372 |

## seed 稳定性（每个 seed 在 5 个点上的 AUC 平均）
- `clf_pre_AZDX_v2` 最稳：`seed_std = 0.018232`
- `clf_pre_scratch_v2` 次稳：`seed_std = 0.026183`
- `clf_pre_AZDX_minus_b_v2` 波动最大：`seed_std = 0.080050`

## 结论
- 按整体 `AUC`，`clf_pre_scratch_v2` 略优于 `clf_pre_AZDX_v2`。
- 按稳定性，`clf_pre_AZDX_v2` 最好，seed 间波动最小。
- 所有实验在 `Tfore=90, Mf=5.5` 仍存在明显退化风险（低 `AUC` 都在该点）。
- 若目标是“上限”，优先 `scratch_v2`；若目标是“稳态可复现”，优先 `AZDX_v2`。
