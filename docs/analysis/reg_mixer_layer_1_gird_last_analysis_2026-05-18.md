# reg_mixer_layer_1_gird 实验分析（基于 `last`）

- 分析日期：2026-05-18
- 实验目录：`experiments/reg_mixer_layer_1_gird`
- 汇总文件：`experiments/reg_mixer_layer_1_gird/reports/reg_grid_metrics_summary.json`

## 1) 统计口径

本分析仅基于该实验目录，不与其他实验对比。

- `ckpt_select = "last"`
- `best_metric = "rmse"`，`best_mode = "min"`
- 总 run 数：`24`（`6` 个 variant × `4` 个 seed）
- 指标来源：
  - `experiments/reg_mixer_layer_1_gird/reports/reg_grid_metrics_group_mean_std.csv`
  - `experiments/reg_mixer_layer_1_gird/reports/reg_grid_metrics_group_best.csv`
  - `experiments/reg_mixer_layer_1_gird/reports/reg_grid_metrics_per_run.csv`

> 说明：RMSE / MAE / MSE / MAPE / DTW 越低越好；R2 / PearsonR / SpearmanR 越高越好。

## 2) 按均值表现排序（跨 seed）

| 排名 | variant | RMSE_mean ± std | MAE_mean | MAPE_mean | R2_mean | PearsonR_mean | DTW_mean | 成功数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | regmx_swu_core | 0.7712 ± 0.2509 | 0.6371 | 13.5768 | -0.9186 | 0.1742 | 81.43 | 4/4 |
| 2 | regmx_lr9e4_enc3p6e4 | 0.7759 ± 0.1983 | 0.6312 | 13.4839 | -0.8870 | 0.2174 | 77.14 | 4/4 |
| 3 | regmx_lr1e3_enc4e4 | 0.7823 ± 0.2887 | 0.6853 | 14.4272 | -1.0157 | 0.3062 | 74.49 | 4/4 |
| 4 | regmx_lr1.5e3_enc6e4 | 0.8255 ± 0.2007 | 0.6588 | 14.2818 | -1.0966 | 0.2443 | 92.85 | 2/4 |
| 5 | regmx_beta03_drop020 | 0.8942 ± 0.2795 | 0.7463 | 16.0668 | -1.5646 | 0.1040 | 92.58 | 4/4 |
| 6 | regmx_baseline | 0.8987 ± 0.2831 | 0.7512 | 16.1619 | -1.5933 | 0.1020 | 94.83 | 4/4 |

## 3) 相对 baseline 的变化（均值口径）

baseline：`regmx_baseline`（RMSE=0.8987, MAE=0.7512）

- `regmx_swu_core`
  - RMSE：`-14.19%`
  - MAE：`-15.19%`
  - MAPE：`-15.99%`
- `regmx_lr9e4_enc3p6e4`
  - RMSE：`-13.67%`
  - MAE：`-15.97%`
  - MAPE：`-16.57%`
- `regmx_lr1e3_enc4e4`
  - RMSE：`-12.95%`
  - MAE：`-8.77%`
  - MAPE：`-10.73%`
- `regmx_beta03_drop020`
  - RMSE：`-0.50%`（接近 baseline）
  - MAE：`-0.65%`

## 4) 最佳单 run（按 RMSE）

数据来源：`experiments/reg_mixer_layer_1_gird/reports/reg_grid_metrics_group_best.csv`

- 全局最优单次：`regmx_swu_core_seed_0`，`RMSE=0.551971`
- 其后：
  - `regmx_lr1e3_enc4e4_seed_3`，`RMSE=0.560146`
  - `regmx_baseline_seed_0`，`RMSE=0.580695`
  - `regmx_beta03_drop020_seed_0`，`RMSE=0.580985`

## 5) 稳定性与异常

### 5.1 seed 波动

各 variant 的 seed 间波动整体偏大（例如 baseline 的 RMSE 范围约 `0.668`，`0.581~1.249`），说明当前设置对随机种子较敏感。

### 5.2 训练失败

`regmx_lr1.5e3_enc6e4` 有 `2/4` 失败：

- `regmx_lr1.5e3_enc6e4_seed_0`：`train_failed`
- `regmx_lr1.5e3_enc6e4_seed_3`：`train_failed`

对应日志中可见 `CUDA out of memory`：

- `experiments/reg_mixer_layer_1_gird/runs/regmx_lr1.5e3_enc6e4_seed_0/launcher.log`
- `experiments/reg_mixer_layer_1_gird/runs/regmx_lr1.5e3_enc6e4_seed_3/launcher.log`

因此该 variant 的均值结论可靠性低于 4/4 成功的配置。

## 6) 结论与建议

1. **当前最优均值方案**：`regmx_swu_core`。在 RMSE、MAE 上综合最优。  
2. **可作为次优备选**：`regmx_lr9e4_enc3p6e4`，均值非常接近第一名，且成功率 4/4。  
3. `regmx_beta03_drop020` 相对 baseline 提升极小，优先级可降低。  
4. `regmx_lr1.5e3_enc6e4` 需先解决训练稳定性（OOM/失败）后再评估其真实水平。  
5. 下一轮建议对 `regmx_swu_core` 与 `regmx_lr9e4_enc3p6e4` 扩展 seeds（如 8~10）做复验，优先验证稳定性与可复现性。
