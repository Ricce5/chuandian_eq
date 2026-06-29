# ETAS / RECAST 滑动窗预测分析（补充 `CB_HAB4` 的 RECAST `norm_0`，bg_norm_weight=0.2）

## 分析口径

- 实验目录：
  - `experiments/etas_multi_ds_bg_norm_0.2`
  - `experiments/rtpp_v2_multi_bg_norm_0.2`
- 本文将 `rtpp_v2` 记为 **RECAST**。
- 本次 ETAS 分析包含：
  - 常规背景：`proportional`、`kernel`、`kernel_256`、`mamba`
  - 无背景：`no_bg`
  - `norm_0` 背景：`proportional_norm0`、`kernel_norm0`、`mamba_norm0`
- `norm_0` 目前只在 `PNR_1z`、`St1_2018`、`CB_HAB4` 上有对应 ETAS run；`CB_HAB1a` 没有 `norm_0` 组。
- RECAST 实验目录也包含 `no_bg` 与 `norm_0` 组；本文只补充 `CB_HAB4` 的 RECAST `norm_0/no_bg` 滑窗结果，其他数据集段落保持原口径。
- 当前滑动窗汇总为 3-seed 均值：ETAS `29/29` 个 variant 成功，RECAST `29/29` 个 variant 成功；本文只将 `CB_HAB4` 的 RECAST `norm_0/no_bg` 纳入跨模型更新。
- 主排序指标为 `RMSE`，越小越好。
- 辅助指标：
  - `MAE`：越小越好。
  - `CRPS`：越小越好。
  - `LP_NB`：越大越好。
  - `coverage`：95% 预测区间覆盖率，理想接近 `0.95`。
  - `W95`：95% 预测区间平均宽度，越小越尖锐，但必须结合 `coverage` 判断。

使用的汇总文件：

- `experiments/etas_multi_ds_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`
- `experiments/rtpp_v2_multi_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`

## 主要结论

加入 ETAS `norm_0` 后，按滑动窗 `RMSE` 的跨模型最优结果变为：

| 数据集 | 最优模型 | 最优背景 | RMSE | MAE | CRPS | Coverage |
|---|---|---|---:|---:|---:|---:|
| `PNR_1z` | ETAS | `mamba_norm0` | 102.218 | 55.726 | 38.448 | 0.656 |
| `CB_HAB1a` | RECAST | `mamba` | 67.009 | 42.383 | 30.586 | 0.795 |
| `St1_2018` | ETAS | `mamba_norm0` | 95.335 | 65.183 | 47.763 | 0.982 |
| `CB_HAB4` | RECAST | `mamba_norm_0` | 92.719 | 67.229 | 49.914 | 0.877 |

核心变化：

- 原先未考虑 `norm_0` 时，RECAST 在 `St1_2018` 上按 RMSE 更优；加入 `norm_0` 后，ETAS `mamba_norm0` 成为 `St1_2018` 最优。
- 加入 RECAST `CB_HAB4` 的 `norm_0` 后，RECAST `mamba_norm_0` 以极小差距超过 ETAS `mamba_norm0`，成为 `CB_HAB4` 按 RMSE 的第一；其他数据集结论保持不变。
- ETAS `mamba_norm0` 仍是 ETAS 内部最强的 `norm_0` 形态；但在 `CB_HAB4` 上，RECAST `mamba_norm_0` 的 RMSE/MAE/CRPS 略优。
- `norm_0` 的收益通常伴随 W95 变宽：它改善了多数组的 RMSE/CRPS/LP_NB，但预测区间更保守。
- ETAS `no_bg` 仍不适合作为滑动窗 count forecast 默认选择：虽然某些 NLL 指标可能好，但在滑窗 RMSE/MAE 上普遍明显变差，且 W95 经常极大。

## ETAS：包含 `no_bg` 与 `norm_0` 的背景模型排名

### `PNR_1z`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba_norm0` | 102.218 | 55.726 | 38.448 | 0.656 | 177.0 | -5.008 |
| 2 | `proportional_norm0` | 112.587 | 61.412 | 42.017 | 0.639 | 181.1 | -5.192 |
| 3 | `mamba` | 112.803 | 60.823 | 44.885 | 0.694 | 144.0 | -5.437 |
| 4 | `proportional` | 114.664 | 59.000 | 41.182 | 0.672 | 169.1 | -5.299 |
| 5 | `kernel_256` | 121.863 | 67.935 | 47.543 | 0.503 | 197.9 | -5.235 |
| 6 | `kernel` | 121.871 | 67.935 | 47.543 | 0.503 | 197.9 | -5.235 |
| 7 | `kernel_norm0` | 134.095 | 81.715 | 54.256 | 0.497 | 280.0 | -5.126 |
| 8 | `no_bg` | 156.341 | 106.830 | 69.556 | 0.361 | 374.8 | -5.388 |

`mamba_norm0` 明显降低 RMSE、MAE、CRPS，并提升 LP_NB；但 coverage 比原始 `mamba` 低，W95 也更宽。`kernel_norm0` 在该数据集上不适合，RMSE 明显变差。

### `CB_HAB1a`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba` | 77.232 | 42.091 | 32.004 | 0.773 | 123.3 | -4.311 |
| 2 | `proportional` | 99.563 | 62.402 | 44.831 | 0.682 | 176.0 | -5.011 |
| 3 | `kernel` | 107.227 | 65.053 | 40.223 | 0.788 | 290.7 | -4.172 |
| 4 | `kernel_256` | 107.810 | 65.287 | 40.383 | 0.788 | 291.4 | -4.175 |
| 5 | `no_bg` | 267.679 | 137.202 | 72.506 | 0.955 | 1052.0 | -4.645 |

`CB_HAB1a` 没有 `norm_0` 对应组。ETAS 内部仍是 `mamba` 最好；`no_bg` 虽然 coverage 接近 0.95，但 W95 极大，点预测质量很差。

### `St1_2018`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba_norm0` | 95.335 | 65.183 | 47.763 | 0.982 | 398.4 | -5.511 |
| 2 | `proportional_norm0` | 114.534 | 79.498 | 55.513 | 0.973 | 404.9 | -5.603 |
| 3 | `kernel_256` | 119.652 | 82.289 | 58.229 | 0.932 | 330.8 | -5.624 |
| 4 | `kernel` | 119.657 | 82.294 | 58.227 | 0.932 | 330.7 | -5.624 |
| 5 | `mamba` | 120.831 | 88.176 | 69.735 | 0.626 | 185.4 | -6.672 |
| 6 | `proportional` | 123.724 | 87.042 | 62.277 | 0.892 | 288.8 | -5.735 |
| 7 | `kernel_norm0` | 137.879 | 87.306 | 63.612 | 0.959 | 515.6 | -5.718 |
| 8 | `no_bg` | 220.650 | 144.178 | 102.237 | 0.910 | 695.9 | -6.112 |

`mamba_norm0` 是 `St1_2018` 的关键变化：它显著优于原始 `mamba`，也超过原先最稳的 `kernel/kernel_256`。但它的 coverage 已超过 0.95，W95 也明显增大，说明预测更保守。

### `CB_HAB4`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba_norm0` | 92.742 | 69.708 | 50.716 | 0.895 | 302.1 | -5.699 |
| 2 | `mamba` | 95.203 | 67.022 | 53.067 | 0.754 | 186.0 | -6.048 |
| 3 | `kernel_norm0` | 100.845 | 79.922 | 58.171 | 0.789 | 354.6 | -5.948 |
| 4 | `kernel_256` | 104.925 | 81.935 | 62.823 | 0.684 | 216.3 | -6.558 |
| 5 | `kernel` | 104.970 | 82.004 | 62.830 | 0.684 | 215.8 | -6.572 |
| 6 | `proportional_norm0` | 116.148 | 93.788 | 76.553 | 0.474 | 187.4 | -9.163 |
| 7 | `proportional` | 131.352 | 111.532 | 91.588 | 0.421 | 163.1 | -12.156 |
| 8 | `no_bg` | 312.556 | 250.648 | 115.799 | 0.877 | 2412.9 | -6.378 |

`CB_HAB4` 中 `norm_0` 对三个 ETAS 背景都有帮助，尤其 `mamba_norm0` 成为 ETAS 内部最优。补充 RECAST `mamba_norm_0` 后，跨模型第一略微转向 RECAST；`no_bg` 的 W95 极端偏大，不适合滑动窗计数预测。

## RECAST：背景模型排名

### `PNR_1z`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba` | 110.836 | 45.329 | 35.385 | 0.858 | 128.0 | -4.409 |
| 2 | `proportional` | 121.022 | 46.767 | 37.485 | 0.880 | 118.3 | -5.167 |
| 3 | `kernel_256` | 123.941 | 52.282 | 39.527 | 0.913 | 209.7 | -4.484 |
| 4 | `kernel` | 124.342 | 51.798 | 39.836 | 0.907 | 193.5 | -4.495 |

RECAST `mamba` 的 MAE、CRPS、coverage 都很好，但按 RMSE 被 ETAS `mamba_norm0` 超过。

### `CB_HAB1a`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba` | 67.009 | 42.383 | 30.586 | 0.795 | 116.2 | -4.328 |
| 2 | `kernel_256` | 85.009 | 57.679 | 36.240 | 0.871 | 245.2 | -4.313 |
| 3 | `kernel` | 87.368 | 61.221 | 37.176 | 0.879 | 279.7 | -4.279 |
| 4 | `proportional` | 100.916 | 73.489 | 53.284 | 0.614 | 162.6 | -6.454 |

RECAST `mamba` 是 `CB_HAB1a` 的全局最优。

### `St1_2018`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `proportional` | 101.170 | 73.191 | 52.521 | 0.869 | 289.5 | -5.860 |
| 2 | `mamba` | 101.405 | 74.646 | 58.308 | 0.626 | 154.3 | -6.938 |
| 3 | `kernel_256` | 107.214 | 74.096 | 53.066 | 0.928 | 351.9 | -5.534 |
| 4 | `kernel` | 107.725 | 74.285 | 53.358 | 0.923 | 350.4 | -5.529 |

RECAST 在 `St1_2018` 上仍较强，但已被 ETAS `mamba_norm0` 在 RMSE、MAE、CRPS 上超过。

### `CB_HAB4`

| 排名 | 背景模型 | RMSE | MAE | CRPS | Coverage | W95 | LP_NB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mamba_norm_0` | 92.719 | 67.229 | 49.914 | 0.877 | 253.4 | -5.537 |
| 2 | `mamba` | 105.149 | 74.078 | 61.269 | 0.579 | 151.7 | -9.710 |
| 3 | `proportional_norm_0` | 106.088 | 84.054 | 65.875 | 0.667 | 195.8 | -13.253 |
| 4 | `kernel` | 107.252 | 78.919 | 63.670 | 0.667 | 176.9 | -9.304 |
| 5 | `kernel_norm_0` | 107.474 | 83.859 | 60.134 | 0.789 | 293.1 | -5.877 |
| 6 | `proportional` | 107.858 | 87.225 | 72.084 | 0.386 | 138.2 | -37.788 |
| 7 | `kernel_256` | 110.058 | 79.594 | 65.452 | 0.667 | 171.1 | -10.011 |
| 8 | `no_bg` | 158.450 | 131.728 | 89.317 | 0.860 | 468.0 | -7.110 |

补充 `norm_0/no_bg` 后，RECAST `mamba_norm_0` 成为 `CB_HAB4` 的 RECAST 内部第一，并以极小差距超过 ETAS `mamba_norm0`；但 RECAST `kernel_norm_0` 和 `no_bg` 的 RMSE 并不理想。

## `norm_0` 对 ETAS 滑动窗指标的影响

下表为 `norm_0 - 非 norm_0`，负数表示误差下降。

| 数据集 | 背景 | ΔRMSE | ΔMAE | ΔCRPS | ΔCoverage | ΔW95 | ΔLP_NB |
|---|---|---:|---:|---:|---:|---:|---:|
| `PNR_1z` | `proportional` | -2.076 | +2.412 | +0.835 | -0.033 | +12.0 | +0.107 |
| `PNR_1z` | `kernel` | +12.224 | +13.781 | +6.713 | -0.005 | +82.1 | +0.108 |
| `PNR_1z` | `mamba` | -10.585 | -5.097 | -6.437 | -0.038 | +33.0 | +0.429 |
| `CB_HAB4` | `proportional` | -15.204 | -17.744 | -15.036 | +0.053 | +24.3 | +2.993 |
| `CB_HAB4` | `kernel` | -4.125 | -2.082 | -4.659 | +0.105 | +138.8 | +0.624 |
| `CB_HAB4` | `mamba` | -2.462 | +2.686 | -2.351 | +0.140 | +116.1 | +0.350 |
| `St1_2018` | `proportional` | -9.189 | -7.544 | -6.764 | +0.081 | +116.1 | +0.132 |
| `St1_2018` | `kernel` | +18.222 | +5.012 | +5.385 | +0.027 | +184.8 | -0.094 |
| `St1_2018` | `mamba` | -25.496 | -22.993 | -21.972 | +0.356 | +213.1 | +1.161 |

整体判断：

- `mamba_norm0` 是最稳定受益的配置：三个数据集 RMSE、CRPS、LP_NB 均改善。
- `proportional_norm0` 也通常改善 RMSE，但在 `PNR_1z` 上 MAE/CRPS 略变差。
- `kernel_norm0` 分化明显：`CB_HAB4` 改善，但 `PNR_1z` 和 `St1_2018` 的 RMSE 明显变差。
- W95 在所有 `norm_0` 组中都增大，说明它倾向于生成更宽、更保守的预测区间。

## ETAS 与 RECAST 对比

### `PNR_1z`

| 排名 | 模型 | 背景 | RMSE | MAE | CRPS | Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | ETAS | `mamba_norm0` | 102.218 | 55.726 | 38.448 | 0.656 |
| 2 | RECAST | `mamba` | 110.836 | 45.329 | 35.385 | 0.858 |
| 3 | ETAS | `proportional_norm0` | 112.587 | 61.412 | 42.017 | 0.639 |

按 RMSE，ETAS `mamba_norm0` 最好；按 MAE/CRPS/coverage，RECAST `mamba` 仍更均衡。

### `CB_HAB1a`

| 排名 | 模型 | 背景 | RMSE | MAE | CRPS | Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | RECAST | `mamba` | 67.009 | 42.383 | 30.586 | 0.795 |
| 2 | ETAS | `mamba` | 77.232 | 42.091 | 32.004 | 0.773 |
| 3 | RECAST | `kernel_256` | 85.009 | 57.679 | 36.240 | 0.871 |

该数据集没有 ETAS `norm_0`，RECAST `mamba` 仍是最优。

### `St1_2018`

| 排名 | 模型 | 背景 | RMSE | MAE | CRPS | Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | ETAS | `mamba_norm0` | 95.335 | 65.183 | 47.763 | 0.982 |
| 2 | RECAST | `proportional` | 101.170 | 73.191 | 52.521 | 0.869 |
| 3 | RECAST | `mamba` | 101.405 | 74.646 | 58.308 | 0.626 |

`norm_0` 使 ETAS 在 `St1_2018` 上反超 RECAST，但其 coverage 偏高、W95 较大，需要关注区间是否过宽。

### `CB_HAB4`

| 排名 | 模型 | 背景 | RMSE | MAE | CRPS | Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | RECAST | `mamba_norm_0` | 92.719 | 67.229 | 49.914 | 0.877 |
| 2 | ETAS | `mamba_norm0` | 92.742 | 69.708 | 50.716 | 0.895 |
| 3 | ETAS | `mamba` | 95.203 | 67.022 | 53.067 | 0.754 |
| 4 | ETAS | `kernel_norm0` | 100.845 | 79.922 | 58.171 | 0.789 |

补充 RECAST `norm_0` 后，`CB_HAB4` 的跨模型第一从 ETAS `mamba_norm0` 变为 RECAST `mamba_norm_0`，但二者 RMSE 只差 0.022，基本可视为同一梯队。

## 背景模型趋势

### `mamba`

`mamba` 是滑动窗 count forecast 中最强的背景形态。ETAS 加入 `norm_0` 后，`mamba_norm0` 在 `PNR_1z`、`St1_2018`、`CB_HAB4` 都成为 ETAS 内部第一；补充 RECAST `CB_HAB4` 的 `norm_0` 后，该数据集的跨模型 RMSE 第一变为 RECAST `mamba_norm_0`。

RECAST 中 `mamba` 也很强：它是 `PNR_1z`、`CB_HAB1a` 的 RECAST 内部第一；在 `CB_HAB4` 上则由 `mamba_norm_0` 取代原始 `mamba`。

### `kernel / kernel_256`

`kernel` 和 `kernel_256` 仍然稳定，但在滑动窗 RMSE 上不如 `mamba_norm0`。ETAS `kernel_norm0` 只在 `CB_HAB4` 上有明显收益，在 `PNR_1z` 与 `St1_2018` 上会放大误差；RECAST `kernel_norm_0` 在 `CB_HAB4` 上主要改善 CRPS/LP_NB/coverage，但 RMSE 略差于原始 `kernel`。

### `proportional`

`proportional_norm0` 通常比原始 `proportional` 好，但整体仍不如 `mamba_norm0`。它可以作为低复杂度备选，不建议作为默认首选。

### `no_bg`

ETAS `no_bg` 对滑动窗 count forecast 依然不友好：多数数据集 RMSE/MAE 极差，W95 也经常异常大。它在某些 likelihood 指标上可能看起来好，但不能直接代表滚动预测质量。

## 与 `nll_test_time` 分析的差异

滑动窗 count forecast 与 `nll_test_time` 的排序不完全一致：

- `nll_test_time` 只评估事件时间似然，滑动窗指标还评估未来窗口计数分布的均值、区间和概率校准。
- ETAS `no_bg` 在部分数据集的 `nll_test_time` 很强，但滑动窗 count forecast 明显变差。
- `norm_0` 在 `nll_test_time` 上整体也偏正面，但在滑动窗中更明显地体现为更宽的 W95 和更保守的区间。
- 因此，如果目标是实际滚动 count forecast，建议以滑动窗 RMSE/CRPS/coverage/W95 的联合表现为主，而不是单独看 NLL。

## 建议

### 如果目标是滑动窗 count forecast

| 数据集 | 推荐模型 | 推荐背景 | 备注 |
|---|---|---|---|
| `PNR_1z` | ETAS | `mamba_norm0` | RMSE 最低；若更重视 MAE/CRPS/coverage，可比较 RECAST `mamba`。 |
| `CB_HAB1a` | RECAST | `mamba` | 当前无 ETAS `norm_0`，RECAST `mamba` 最稳。 |
| `St1_2018` | ETAS | `mamba_norm0` | 误差最优，但 coverage 偏高、W95 较宽。 |
| `CB_HAB4` | RECAST / ETAS | `mamba_norm_0` / `mamba_norm0` | 二者 RMSE 几乎持平；RECAST `mamba_norm_0` 的 MAE/CRPS 略优，ETAS `mamba_norm0` coverage 略高。 |

### 后续实验建议

1. 对 `CB_HAB1a` 补跑 ETAS `proportional_norm0`、`kernel_norm0`、`mamba_norm0`，确认 `norm_0` 是否也能带来类似收益。
2. 对 `mamba_norm0` 做 W95/coverage 校准检查，尤其是 `St1_2018`，判断是否存在过度保守。
3. 对 `kernel_norm0` 做数据集分层诊断：ETAS `kernel_norm0` 在 `CB_HAB4` 有效，但在 `PNR_1z` 和 `St1_2018` 明显变差；RECAST `kernel_norm_0` 在 `CB_HAB4` 上也不是 RMSE 最优。
4. 对 RECAST 的 `norm_0/no_bg` 对照做完整数据集复核，判断 `CB_HAB4` 的收益是否能推广到其他数据集。
5. 对 `PNR_1z` 同时报告 RMSE 与 MAE/CRPS，因为 ETAS `mamba_norm0` 和 RECAST `mamba` 的最优指标不一致。
