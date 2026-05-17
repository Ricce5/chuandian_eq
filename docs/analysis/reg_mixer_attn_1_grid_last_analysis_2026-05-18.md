# reg_mixer_attn_1_grid 实验分析（基于 `last`）

- 分析日期：2026-05-18
- 实验目录：`experiments/reg_mixer_attn_1_grid`
- 汇总文件：`experiments/reg_mixer_attn_1_grid/reports/reg_grid_metrics_summary.json`

## 1) 统计口径

本分析严格基于 `last` checkpoint：

- `ckpt_select = "last"`
- `best_metric = "RMSE"`，`best_mode = "min"`
- 总 run 数：`36`（`9` 个 variant × `4` 个 seed）

## 2) 按均值表现排序（跨 seed）

数据来源：`experiments/reg_mixer_attn_1_grid/reports/reg_grid_metrics_group_mean_std.csv`

> 说明：RMSE / MAE / MSE / MAPE / DTW 越低越好；R2 / PearsonR / SpearmanR 越高越好。

| 排名 | variant | RMSE_mean ± std | MAE_mean | R2_mean | PearsonR_mean | DTW_mean |
|---|---|---:|---:|---:|---:|---:|
| 1 | regmx_swu_beta03_drop020 | 0.7734 ± 0.0825 | 0.6254 | -0.8029 | 0.1038 | 76.65 |
| 2 | regmx_swu_warm008 | 0.7942 ± 0.0630 | 0.6674 | -0.8939 | -0.0867 | 73.77 |
| 3 | regmx_align_ckpt_20260515_180423 | 0.7991 ± 0.0563 | 0.6868 | -0.9156 | 0.1948 | 65.92 |
| 4 | regmx_align_ckpt_20260204_180531_safe | 0.8016 ± 0.1041 | 0.6660 | -0.9444 | 0.1825 | 77.39 |
| 5 | regmx_swu_core | 0.8017 ± 0.0864 | 0.6831 | -0.9373 | 0.0333 | 86.93 |
| 6 | regmx_swu_lr9e4_enc3p6e4 | 0.8178 ± 0.1207 | 0.6634 | -1.0312 | -0.0665 | 71.10 |
| 7 | regmx_baseline | 0.8238 ± 0.0735 | 0.7029 | -1.0404 | 0.0294 | 87.86 |
| 8 | regmx_swu_lr8e4_enc3p2e4 | 0.8497 ± 0.0828 | 0.6868 | -1.1728 | 0.0051 | 71.63 |
| 9 | regmx_swu_beta03_drop020_attn2 | 0.8539 ± 0.0920 | 0.7112 | -1.1978 | 0.0588 | 74.67 |

## 3) 相对 baseline 的变化（均值口径）

baseline：`regmx_baseline`（RMSE=0.8238, MAE=0.7029, DTW=87.86）

- `regmx_swu_beta03_drop020`：
  - RMSE：`-6.12%`
  - MAE：`-11.03%`
  - DTW：`-12.76%`
  - R2：`+0.2375`（仍为负）
  - PearsonR：`+0.0744`
- `regmx_swu_warm008`：
  - RMSE：`-3.60%`
  - MAE：`-5.05%`
  - DTW：`-16.05%`
  - PearsonR：`-0.1161`（相关性变差）
- `regmx_align_ckpt_20260515_180423`：
  - RMSE：`-3.00%`
  - MAE：`-2.29%`
  - DTW：`-24.98%`（时序形态距离改善最显著）
  - PearsonR：`+0.1655`

## 4) 最佳单 run 与稳定性

### 4.1 最佳单 run（按 RMSE）

数据来源：`experiments/reg_mixer_attn_1_grid/reports/reg_grid_metrics_group_best.csv`

- 全局最优单次 RMSE：`regmx_swu_lr9e4_enc3p6e4_seed_3`，`RMSE=0.6732`
- 但该 variant 的均值与方差较差：`0.8178 ± 0.1207`
- 结论：单次峰值较高，但可复现性弱，不宜直接按单次冠军选型

### 4.2 seed 稳定性（RMSE 变异系数 CV=std/mean）

- 最稳定：`regmx_align_ckpt_20260515_180423`（CV≈0.070）
- 次稳定：`regmx_swu_warm008`（CV≈0.079）
- 波动最大：`regmx_swu_lr9e4_enc3p6e4`（CV≈0.148）

## 5) 关键观察

1. **均值最优方案**是 `regmx_swu_beta03_drop020`，在 RMSE 与 MAE 上同时领先。  
2. **稳定性最优方案**是 `regmx_align_ckpt_20260515_180423`，且 DTW 最优，适合重视时序形态一致性的场景。  
3. `regmx_swu_beta03_drop020_attn2` 相比同系列明显退化，说明当前配置下把 attention 放在 layer2 并不合适。  
4. 所有方案 `R2_mean` 仍为负，说明虽然误差可下降，但整体解释度还未达到理想水平。  

## 6) 下一轮建议

- 主推两条线并行复验（seed 建议扩到 8~10）：
  - 线A（追求误差）：`regmx_swu_beta03_drop020`
  - 线B（追求稳定+形态）：`regmx_align_ckpt_20260515_180423`
- 暂缓：`regmx_swu_beta03_drop020_attn2` 与 `regmx_swu_lr8e4_enc3p2e4`
- 增加分析：补做 `best` vs `last` 并排报告，确认训练后期是否存在退化/过拟合

