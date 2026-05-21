# clf_pre_v1 分析（2026-05-21）

## 范围
- 目录：`experiments/clf_pre_v1`
- 子实验：`clf_pre_AZDX_v1`、`clf_pre_AZDX_minus_b_v1`、`clf_pre_SCEDC_v1`、`clf_pre_SCEDC_minus_b_v1`、`clf_pre_scratch_v1`
- 口径：单实验汇总用 `summarize_clf_mf_tf_grid.py`，跨实验对比用 `summarize_clf_seed_across_experiments.py`

## 数据完整性
- 5 个实验均为 `40` 个 run
- 所有 run 均有有效结果：`status_ok = 100%`，`metrics_found = 100%`
- 因此下面的对比可视为完整样本

## 横向结果（按全网格平均）
| rank | experiment | auc_mean_over_grid | f1_mean_over_grid | R_mean_over_grid | best_auc_mean | best_point | auc_std_over_grid_mean |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | `clf_pre_scratch_v1` | 0.785926 | 0.495558 | 0.204443 | 0.873645 | `(60, 5.0)` | 0.072783 |
| 2 | `clf_pre_AZDX_v1` | 0.771722 | 0.490340 | 0.181544 | 0.845170 | `(60, 5.0)` | 0.066630 |
| 3 | `clf_pre_AZDX_minus_b_v1` | 0.758946 | 0.496198 | 0.197758 | 0.866977 | `(60, 5.0)` | 0.088083 |
| 4 | `clf_pre_SCEDC_minus_b_v1` | 0.757862 | 0.498637 | 0.169671 | 0.865425 | `(60, 5.0)` | 0.075361 |
| 5 | `clf_pre_SCEDC_v1` | 0.739775 | 0.486621 | 0.159711 | 0.871712 | `(60, 5.0)` | 0.071135 |

## 点位赢家（按 `auc_mean`）
| Tfore | Mf | winner | auc_mean | runner-up | gap |
|---|---:|---|---:|---|---:|
| 10 | 4.0 | `clf_pre_SCEDC_minus_b_v1` | 0.788068 | `clf_pre_AZDX_v1` | 0.003514 |
| 20 | 4.5 | `clf_pre_AZDX_v1` | 0.808534 | `clf_pre_SCEDC_minus_b_v1` | 0.033976 |
| 30 | 4.5 | `clf_pre_AZDX_v1` | 0.830951 | `clf_pre_AZDX_minus_b_v1` | 0.002272 |
| 60 | 5.0 | `clf_pre_scratch_v1` | 0.873645 | `clf_pre_SCEDC_v1` | 0.001933 |
| 90 | 5.5 | `clf_pre_scratch_v1` | 0.744542 | `clf_pre_AZDX_minus_b_v1` | 0.101096 |

## `auc < 0.5` 的 run
- 共 `13` 个
- 其中 `12/13` 都集中在 `(Tfore=90, Mf=5.5)`
- 唯一例外是 `clf_pre_scratch_v1` 的 `seed=6` 在 `(Tfore=20, Mf=4.5)`

| experiment | run | seed | Tfore | Mf | auc |
|---|---|---:|---:|---:|---:|
| `clf_pre_AZDX_minus_b_v1` | `tf_20_mf_4p5_seed_2` | 2 | 20 | 4.5 | 0.498230 |
| `clf_pre_AZDX_minus_b_v1` | `tf_90_mf_5p5_seed_6` | 6 | 90 | 5.5 | 0.330214 |
| `clf_pre_AZDX_v1` | `tf_90_mf_5p5_seed_1` | 1 | 90 | 5.5 | 0.484211 |
| `clf_pre_AZDX_v1` | `tf_90_mf_5p5_seed_5` | 5 | 90 | 5.5 | 0.370565 |
| `clf_pre_AZDX_v1` | `tf_90_mf_5p5_seed_6` | 6 | 90 | 5.5 | 0.369591 |
| `clf_pre_SCEDC_minus_b_v1` | `tf_90_mf_5p5_seed_0` | 0 | 90 | 5.5 | 0.441326 |
| `clf_pre_SCEDC_minus_b_v1` | `tf_90_mf_5p5_seed_1` | 1 | 90 | 5.5 | 0.325146 |
| `clf_pre_SCEDC_minus_b_v1` | `tf_90_mf_5p5_seed_3` | 3 | 90 | 5.5 | 0.415984 |
| `clf_pre_SCEDC_v1` | `tf_90_mf_5p5_seed_1` | 1 | 90 | 5.5 | 0.393762 |
| `clf_pre_SCEDC_v1` | `tf_90_mf_5p5_seed_3` | 3 | 90 | 5.5 | 0.491228 |
| `clf_pre_SCEDC_v1` | `tf_90_mf_5p5_seed_5` | 5 | 90 | 5.5 | 0.405848 |
| `clf_pre_SCEDC_v1` | `tf_90_mf_5p5_seed_6` | 6 | 90 | 5.5 | 0.378558 |
| `clf_pre_scratch_v1` | `tf_20_mf_4p5_seed_6` | 6 | 20 | 4.5 | 0.330721 |

## 结论
- `clf_pre_scratch_v1` 的整体 `AUC` 最好，且在长窗口 `(60, 5.0)`、`(90, 5.5)` 上都拿到最优点。
- `clf_pre_AZDX_v1` 的整体表现略低于 `scratch`，但 seed 波动更小，更稳。
- `Tfore=90, Mf=5.5` 是明显的风险点，大部分 `auc < 0.5` 都集中在这里。
- 如果目标是“稳”，优先看 `AZDX`；如果目标是“上限”，优先看 `scratch`。
