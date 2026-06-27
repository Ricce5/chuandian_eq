# ETAS / RECAST 滑动窗预测分析（包含 no_bg，bg_norm_weight=0.2）

## 分析口径

- 实验目录：
  - `experiments/etas_multi_ds_bg_norm_0.2`
  - `experiments/rtpp_v2_multi_bg_norm_0.2`
- 本文将 `rtpp_v2` 记为 **RECAST**。
- 本次分析包含 ETAS 的 `no_bg`。
- 主排序指标为 `RMSE`，越小越好。
- 辅助指标：
  - `MAE`：越小越好。
  - `CRPS`：越小越好。
  - `LP_NB`：越大越好。
  - `coverage`：95% 预测区间覆盖率，理想接近 `0.95`。
  - `W95`：95% 预测区间平均宽度，越小越尖锐，但必须结合 `coverage` 判断。
- 当前滑动窗汇总均为单 run 结果：每个 variant 的 `n_runs=1`，不是多 seed 平均。

## 数据完整性

当前两个滑动窗汇总文件均完整：

- ETAS：`20/20` 个 variant 成功，包含 `no_bg`。
- RECAST：`16/16` 个 variant 成功，不包含 `no_bg`。

使用的汇总文件：

- `experiments/etas_multi_ds_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`
- `experiments/rtpp_v2_multi_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`

## 主要结论

按 `RMSE` 排序：

| 数据集 | 最优模型 | 最优背景 | RMSE | MAE | CRPS | Coverage |
|---|---|---|---:|---:|---:|---:|
| PNR_1z | ETAS | mamba | 108.889 | 58.533 | 43.556 | 0.689 |
| CB_HAB1a | RECAST | mamba | 68.913 | 41.572 | 30.109 | 0.818 |
| St1_2018 | RECAST | proportional | 101.049 | 75.251 | 53.319 | 0.878 |
| CB_HAB4 | ETAS | mamba | 94.738 | 66.518 | 52.423 | 0.789 |

整体上，按滑动窗 `RMSE`：

- ETAS 赢 `PNR_1z` 和 `CB_HAB4`。
- RECAST 赢 `CB_HAB1a` 和 `St1_2018`。
- 两类模型在滑动窗预测中都明显偏好 `mamba` 背景。
- ETAS `no_bg` 在所有数据集上都不是最优，且多数情况下明显劣化。

## ETAS：包含 no_bg 的背景模型排名

### PNR_1z

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 108.889 | 58.533 | 43.556 | 0.689 | 136.2 | -5.451 |
| 2 | proportional | 114.663 | 58.989 | 41.183 | 0.672 | 169.1 | -5.299 |
| 3 | kernel_256 | 120.660 | 67.043 | 46.891 | 0.508 | 195.6 | -5.202 |
| 4 | kernel | 120.676 | 67.053 | 46.895 | 0.508 | 195.7 | -5.202 |
| 5 | no_bg | 156.252 | 108.650 | 70.741 | 0.344 | 378.5 | -5.434 |

`mamba` 的 RMSE 最低。`proportional` 的 CRPS 和 LP_NB 也较好，但 RMSE 略差。`no_bg` 明显偏差较大，coverage 也很低。

### CB_HAB1a

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 77.526 | 40.501 | 31.664 | 0.773 | 119.7 | -4.228 |
| 2 | proportional | 99.563 | 62.402 | 44.831 | 0.682 | 176.0 | -5.011 |
| 3 | kernel | 105.251 | 64.541 | 39.992 | 0.773 | 286.3 | -4.197 |
| 4 | kernel_256 | 105.620 | 64.785 | 40.198 | 0.773 | 285.6 | -4.221 |
| 5 | no_bg | 269.231 | 139.103 | 72.704 | 0.955 | 1064.6 | -4.616 |

`mamba` 在误差指标上明显最好。`no_bg` 的 coverage 接近 `0.95`，但 W95 极大，说明它主要是靠极宽预测区间提高覆盖率，点预测质量很差。

### St1_2018

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | kernel | 120.605 | 82.626 | 58.850 | 0.932 | 319.8 | -5.641 |
| 2 | kernel_256 | 120.606 | 82.612 | 58.853 | 0.932 | 320.0 | -5.641 |
| 3 | mamba | 121.315 | 89.276 | 69.592 | 0.649 | 205.3 | -6.321 |
| 4 | proportional | 123.724 | 87.042 | 62.277 | 0.892 | 288.8 | -5.735 |
| 5 | no_bg | 221.016 | 144.860 | 102.476 | 0.919 | 699.0 | -6.114 |

`kernel` 和 `kernel_256` 基本并列，说明该数据集上平滑背景更稳。`mamba` 的 RMSE 接近，但 coverage 明显偏低。`no_bg` 误差大幅变差。

### CB_HAB4

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 94.738 | 66.518 | 52.423 | 0.789 | 186.7 | -5.942 |
| 2 | kernel_256 | 106.937 | 83.827 | 64.589 | 0.684 | 220.5 | -6.668 |
| 3 | kernel | 106.951 | 83.845 | 64.599 | 0.684 | 220.6 | -6.669 |
| 4 | proportional | 131.341 | 111.497 | 91.572 | 0.421 | 163.1 | -12.156 |
| 5 | no_bg | 310.334 | 247.136 | 115.172 | 0.842 | 2389.4 | -6.385 |

`mamba` 在所有主要误差和概率指标上都最好。`no_bg` 的 W95 极大且 RMSE 极差，不适合滑动窗计数预测。

## RECAST：背景模型排名

### PNR_1z

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 111.592 | 46.181 | 35.673 | 0.885 | 130.1 | -4.297 |
| 2 | kernel_256 | 118.637 | 52.705 | 38.554 | 0.918 | 263.9 | -4.468 |
| 3 | proportional | 121.790 | 47.739 | 37.569 | 0.885 | 131.1 | -5.064 |
| 4 | kernel | 125.388 | 52.204 | 40.699 | 0.918 | 203.6 | -4.433 |

`mamba` 的 RMSE、MAE、CRPS、LP_NB 综合最好；`kernel/kernel_256` 的 coverage 更接近 `0.95`，但区间更宽。

### CB_HAB1a

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 68.913 | 41.572 | 30.109 | 0.818 | 124.9 | -4.258 |
| 2 | kernel_256 | 81.050 | 57.511 | 35.453 | 0.886 | 252.4 | -4.295 |
| 3 | kernel | 94.124 | 70.388 | 40.286 | 0.886 | 330.7 | -4.258 |
| 4 | proportional | 98.961 | 71.816 | 52.256 | 0.614 | 159.7 | -6.334 |

`mamba` 在 RMSE 和 CRPS 上明显最好。`kernel/kernel_256` 的 coverage 更高，但 W95 显著更宽。

### St1_2018

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | proportional | 101.049 | 75.251 | 53.319 | 0.878 | 295.7 | -5.728 |
| 2 | mamba | 102.119 | 75.139 | 58.094 | 0.608 | 164.5 | -7.383 |
| 3 | kernel_256 | 102.354 | 68.199 | 49.629 | 0.932 | 408.2 | -5.489 |
| 4 | kernel | 103.657 | 68.959 | 50.458 | 0.932 | 406.4 | -5.499 |

如果只看 RMSE，`proportional` 最好；但如果看 MAE、CRPS、LP_NB 和 coverage，`kernel_256` 更稳。`mamba` 虽然 RMSE 接近，但 coverage 只有 `0.608`，校准明显不足。

### CB_HAB4

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mamba | 102.493 | 71.565 | 58.669 | 0.684 | 158.3 | -9.577 |
| 2 | proportional | 108.371 | 87.953 | 73.909 | 0.316 | 121.4 | -18.986 |
| 3 | kernel | 108.839 | 79.749 | 64.745 | 0.684 | 182.7 | -10.138 |
| 4 | kernel_256 | 114.254 | 82.703 | 68.621 | 0.632 | 163.9 | -11.114 |

`mamba` 是 RECAST 在 `CB_HAB4` 上的最佳背景，但整体仍不如 ETAS `mamba`。

## ETAS no_bg 的影响

包含 `no_bg` 后，ETAS 的结论更加明确：`no_bg` 不适合滑动窗 count forecast。

| 数据集 | no_bg RMSE | ETAS 最佳 RMSE | 差距 |
|---|---:|---:|---:|
| PNR_1z | 156.252 | 108.889 | +47.363 |
| CB_HAB1a | 269.231 | 77.526 | +191.705 |
| St1_2018 | 221.016 | 120.605 | +100.411 |
| CB_HAB4 | 310.334 | 94.738 | +215.596 |

`no_bg` 的主要问题是缺少背景率调节，导致窗口计数均值预测系统性偏差。虽然在 `CB_HAB1a` 和 `CB_HAB4` 中 coverage 看起来较高，但对应 W95 极大，说明它不是更准确，而是不确定性区间过宽。

## ETAS 与 RECAST 对比

### PNR_1z

| 模型 | 最佳背景 | RMSE | MAE | CRPS | Coverage | LP_NB |
|---|---|---:|---:|---:|---:|---:|
| ETAS | mamba | 108.889 | 58.533 | 43.556 | 0.689 | -5.451 |
| RECAST | mamba | 111.592 | 46.181 | 35.673 | 0.885 | -4.297 |

按 RMSE，ETAS 略好；但 RECAST 在 MAE、CRPS、LP_NB 和 coverage 上明显更好。说明 ETAS 可能少数大误差更小，但 RECAST 的整体分布预测质量更好。

### CB_HAB1a

| 模型 | 最佳背景 | RMSE | MAE | CRPS | Coverage | LP_NB |
|---|---|---:|---:|---:|---:|---:|
| ETAS | mamba | 77.526 | 40.501 | 31.664 | 0.773 | -4.228 |
| RECAST | mamba | 68.913 | 41.572 | 30.109 | 0.818 | -4.258 |

RECAST `mamba` 在 RMSE、CRPS 和 coverage 上更好；ETAS `mamba` 的 MAE 和 LP_NB 略好。综合看 RECAST 更优。

### St1_2018

| 模型 | 最佳背景 | RMSE | MAE | CRPS | Coverage | LP_NB |
|---|---|---:|---:|---:|---:|---:|
| ETAS | kernel | 120.605 | 82.626 | 58.850 | 0.932 | -5.641 |
| RECAST | proportional | 101.049 | 75.251 | 53.319 | 0.878 | -5.728 |
| RECAST | kernel_256 | 102.354 | 68.199 | 49.629 | 0.932 | -5.489 |

按 RMSE，RECAST `proportional` 最好；按 MAE、CRPS、LP_NB 和 coverage，RECAST `kernel_256` 更稳。无论采用哪种辅助指标，RECAST 都优于 ETAS。

### CB_HAB4

| 模型 | 最佳背景 | RMSE | MAE | CRPS | Coverage | LP_NB |
|---|---|---:|---:|---:|---:|---:|
| ETAS | mamba | 94.738 | 66.518 | 52.423 | 0.789 | -5.942 |
| RECAST | mamba | 102.493 | 71.565 | 58.669 | 0.684 | -9.577 |

ETAS `mamba` 全面优于 RECAST `mamba`。这与此前 `nll_test_time` 中 `CB_HAB4` 上 ETAS 更稳的结论一致。

## 背景模型趋势

### mamba

滑动窗预测中，`mamba` 是最强背景模型。

- ETAS：`mamba` 在 3/4 个数据集上 RMSE 最优。
- RECAST：`mamba` 在 3/4 个数据集上 RMSE 最优。

原因是滑动窗 count forecast 直接依赖窗口内事件数的动态变化，灵活背景模型更容易追踪短期非平稳背景率。

### kernel / kernel_256

`kernel` 和 `kernel_256` 更平滑、更稳定，尤其在概率校准上经常有优势。

- ETAS `St1_2018` 中 `kernel/kernel_256` 最好。
- RECAST `St1_2018` 中 `kernel_256` 虽然 RMSE 略差于 `proportional`，但 MAE、CRPS、LP_NB 和 coverage 更好。

如果目标是稳定概率预测，而不是单纯 RMSE，`kernel_256` 仍然值得保留。

### proportional

`proportional` 在大多数场景下不是最优，但在 RECAST `St1_2018` 上 RMSE 最低。

这可能说明 `St1_2018` 的窗口计数变化中有较强的比例型背景成分，简单背景反而减少了过拟合。但从 CRPS 和 LP_NB 看，`kernel_256` 的分布预测更稳。

### no_bg

`no_bg` 在滑动窗预测中明显不推荐。

它在所有 ETAS 数据集上 RMSE 都是最差或接近最差。个别数据集 coverage 较高，是通过极宽 W95 实现的，并不代表预测更好。

## 与 nll_test_time 分析的差异

此前按 `nll_test_time` 分析时，RECAST 更偏好 `kernel`，ETAS 在部分数据集也偏好 `kernel/kernel_256`。

滑动窗预测则明显更偏好 `mamba`。原因是：

- `nll_test_time` 更关注逐事件时间似然。
- 滑动窗 `RMSE/MAE/CRPS` 更关注窗口计数预测。
- 窗口计数预测对背景率的动态变化更敏感。
- `mamba` 背景能更灵活地捕捉短期非平稳变化，因此在滑动窗指标上更强。

## 建议

### 如果目标是滑动窗 count forecast

| 场景 | 推荐 |
|---|---|
| ETAS 默认背景 | mamba |
| RECAST 默认背景 | mamba |
| PNR_1z | ETAS mamba 或 RECAST mamba；若重视 CRPS/coverage，选 RECAST mamba |
| CB_HAB1a | RECAST mamba |
| St1_2018 | RECAST proportional 看 RMSE；RECAST kernel_256 看综合概率质量 |
| CB_HAB4 | ETAS mamba |

### 后续实验建议

1. 对滑动窗评估补齐多 seed，而不是只看 seed 0。
2. 对 `St1_2018` 的 RECAST 比较 `proportional` 与 `kernel_256`，不要只按 RMSE 决策。
3. 对 `CB_HAB4` 继续重点排查 RECAST 的分布漂移问题，因为它在 NLL 和滑动窗指标上都弱于 ETAS。
4. 对 `mamba` 背景做容量/正则消融，确认其滑动窗优势是否稳定。
5. 不建议继续把 ETAS `no_bg` 作为滑动窗预测候选，除非只用于 ablation baseline。

