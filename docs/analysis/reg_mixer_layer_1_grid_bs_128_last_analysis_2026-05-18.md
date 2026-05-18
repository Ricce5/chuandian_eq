# reg_mixer_layer_1_grid_bs_128 实验分析（基于 `last`）

- 分析日期：2026-05-18
- 实验目录：`experiments/reg_mixer_layer_1_grid_bs_128`
- 汇总文件：`experiments/reg_mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_summary.json`

## 1) 统计口径

- `ckpt_select = "last"`
- `best_metric = "RMSE"`，`best_mode = "min"`
- 总 run 数：`25`
- 成功数：`24`（`1` 个测试失败）

数据来源：
- `experiments/reg_mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_group_mean_std.csv`
- `experiments/reg_mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_group_best.csv`
- `experiments/reg_mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_per_run.csv`

> 说明：RMSE / MAE / MSE / MAPE / DTW 越低越好；R2 / PearsonR / SpearmanR 越高越好。

## 2) 按均值表现排序（跨 seed）

| 排名 | variant | RMSE_mean ± std | MAE_mean | R2_mean | DTW_mean | 成功数 |
|---|---|---:|---:|---:|---:|---:|
| 1 | regmx_lr1e3_enc4e4 | 0.6980 ± 0.1165 | 0.5717 | -0.4882 | 64.04 | 5/5 |
| 2 | regmx_lr9e4_enc3p6e4 | 0.7234 ± 0.1052 | 0.5967 | -0.5904 | 71.68 | 4/5 |
| 3 | regmx_ema | 0.7279 ± 0.0701 | 0.6161 | -0.5951 | 68.55 | 5/5 |
| 4 | regmx_warmup_linear_baseline | 0.7474 ± 0.1836 | 0.6004 | -0.7500 | 74.85 | 5/5 |
| 5 | regmx_ema_05 | 0.8043 ± 0.2128 | 0.6726 | -1.0414 | 87.04 | 5/5 |

## 3) 相对 baseline 的变化（均值口径）

baseline：`regmx_warmup_linear_baseline`（RMSE=0.7474，MAE=0.6004，DTW=74.85）

- `regmx_lr1e3_enc4e4`
  - RMSE：`-6.62%`
  - MAE：`-4.77%`
  - DTW：`-14.44%`
- `regmx_lr9e4_enc3p6e4`
  - RMSE：`-3.21%`
  - MAE：`-0.62%`
  - DTW：`-4.23%`
  - 但成功率 `4/5`，结论需谨慎
- `regmx_ema`
  - RMSE：`-2.61%`
  - MAE：`+2.62%`（略变差）
  - DTW：`-8.42%`
- `regmx_ema_05`
  - RMSE：`+7.61%`
  - MAE：`+12.04%`
  - DTW：`+16.29%`

## 4) 最佳单 run（按 RMSE）

数据来源：`experiments/reg_mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_group_best.csv`

- 全局最优单次：`regmx_warmup_linear_baseline_seed_0`，`RMSE=0.547730`
- 第二：`regmx_lr1e3_enc4e4_seed_0`，`RMSE=0.560637`
- 第三：`regmx_ema_05_seed_0`，`RMSE=0.572750`

说明：`ema_05` 虽然有较低的单次峰值，但均值和方差都较差，不建议按单次冠军选型。

## 5) 稳定性与异常

### 5.1 稳定性（CV=std/mean）

- `regmx_ema` 最稳：RMSE_CV ≈ `0.096`
- `regmx_lr1e3_enc4e4` 次稳：RMSE_CV ≈ `0.167`
- `regmx_warmup_linear_baseline` 与 `regmx_ema_05` 波动较大（CV 分别约 `0.246`、`0.265`）

### 5.2 失败样本

- 失败 run：`regmx_lr9e4_enc3p6e4_seed_2`
- 在 `per_run` 中体现为 `status_ok=0`（测试失败）
- 对应 run 目录现已存在 `metrics_test_last_1.json`，但当前汇总口径仍记录过该失败状态，建议后续重汇总前确认 `summary.json` 与脚本输出一致性。

## 6) 结论与建议

1. **当前最优均值方案**：`regmx_lr1e3_enc4e4`（误差与 DTW 综合最优）。  
2. **当前最优稳定方案**：`regmx_ema`（方差最小，适合追求稳健复现）。  
3. `regmx_lr9e4_enc3p6e4` 有潜力，但需要先补齐/确认失败 seed 后再与第一名做最终比较。  
4. `regmx_ema_05` 可考虑淘汰（均值、方差、DTW 均不占优）。  
5. 下一轮建议在 `lr1e3_enc4e4` 与 `ema` 两条线上做 8~10 seeds 扩展复验：前者追求精度，后者追求稳健。  

