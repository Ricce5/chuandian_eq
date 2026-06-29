# `etas_multi_ds_bg_norm_0.2` 中 `norm_0` 与对应非 `norm_0` 组的指标变化分析

## 结论摘要

- 本次共比较 9 对可配对实验：`CB_HAB4`、`PNR_1z`、`St1_2018` × `kernel/mamba/proportional`，每对均包含 `seed=0,1,2`。
- 滑窗预测指标上，`norm_0` 整体偏正面：MAE 5/9 组改善，RMSE 7/9 组改善，CRPS 6/9 组改善，`lp_nb` 8/9 组改善。
- `norm_0` 会显著扩大预测区间：W95 在 9/9 组都变大，平均约 +47.1%；这通常提高 coverage，但也意味着区间更宽、更保守。
- 最大滑窗收益出现在 `St1_2018_mamba`：MAE -26.1%、RMSE -21.1%、CRPS -31.5%，coverage 从 0.626 提升到 0.982。
- 最大滑窗负面出现在 `PNR_1z_kernel`：MAE +20.3%、RMSE +10.0%、CRPS +14.1%，coverage 从 0.503 略降到 0.497。
- 只看 `nll_time` 时，`norm_0` 在 test split 上整体有利：7/9 组改善，2/9 组变差；`CB_HAB4` 三个背景模型全部改善。

## 分析口径

实验目录：

- `/root/autodl-tmp/em_eqf/experiments/etas_multi_ds_bg_norm_0.2`

主要数据来源：

- 滑窗预测汇总：`experiments/etas_multi_ds_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`
- 每个 run 的测试 NLL：`experiments/etas_multi_ds_bg_norm_0.2/runs/*/metrics_test_best_1.json`

注意：

- `reports/etas_grid_metrics_group_mean_std.csv` 中只汇总到了 `etas__St1_2018_mamba_norm_0` 一个 `norm_0` 组。
- 但所有 `norm_0` run 目录下均存在 `metrics_test_best_1.json`，因此本文的 NLL 对比直接从各 run 的 JSON 重新按配对组聚合。
- 表中 `Δ = norm_0 - 非 norm_0`。
- `nll_time`、MAE、RMSE、CRPS、W95 越低越好；`lp_nb` 越高越好；coverage 越接近 0.95 越好。

## 可配对组

| 数据集 | 背景模型 | 非 `norm_0` | `norm_0` | seeds |
|---|---|---|---|---|
| `CB_HAB4` | `kernel` | `etas__CB_HAB4_kernel` | `etas__CB_HAB4_kernel_norm_0` | 0,1,2 |
| `CB_HAB4` | `mamba` | `etas__CB_HAB4_mamba` | `etas__CB_HAB4_mamba_norm_0` | 0,1,2 |
| `CB_HAB4` | `proportional` | `etas__CB_HAB4_proportional` | `etas__CB_HAB4_proportional_norm_0` | 0,1,2 |
| `PNR_1z` | `kernel` | `etas__PNR_1z_kernel` | `etas__PNR_1z_kernel_norm_0` | 0,1,2 |
| `PNR_1z` | `mamba` | `etas__PNR_1z_mamba` | `etas__PNR_1z_mamba_norm_0` | 0,1,2 |
| `PNR_1z` | `proportional` | `etas__PNR_1z_proportional` | `etas__PNR_1z_proportional_norm_0` | 0,1,2 |
| `St1_2018` | `kernel` | `etas__St1_2018_kernel` | `etas__St1_2018_kernel_norm_0` | 0,1,2 |
| `St1_2018` | `mamba` | `etas__St1_2018_mamba` | `etas__St1_2018_mamba_norm_0` | 0,1,2 |
| `St1_2018` | `proportional` | `etas__St1_2018_proportional` | `etas__St1_2018_proportional_norm_0` | 0,1,2 |

## 滑窗预测指标变化

| 数据集 | bg | coverage | MAE | RMSE | CRPS | W95 | `lp_nb` |
|---|---|---:|---:|---:|---:|---:|---:|
| `CB_HAB4` | `kernel` | 0.684 → 0.789 | -2.5% | -3.9% | -7.4% | +64.3% | +0.624 |
| `CB_HAB4` | `mamba` | 0.754 → 0.895 | +4.0% | -2.6% | -4.4% | +62.4% | +0.350 |
| `CB_HAB4` | `proportional` | 0.421 → 0.474 | -15.9% | -11.6% | -16.4% | +14.9% | +2.993 |
| `PNR_1z` | `kernel` | 0.503 → 0.497 | +20.3% | +10.0% | +14.1% | +41.5% | +0.108 |
| `PNR_1z` | `mamba` | 0.694 → 0.656 | -8.4% | -9.4% | -14.3% | +22.9% | +0.429 |
| `PNR_1z` | `proportional` | 0.672 → 0.639 | +4.1% | -1.8% | +2.0% | +7.1% | +0.107 |
| `St1_2018` | `kernel` | 0.932 → 0.959 | +6.1% | +15.2% | +9.2% | +55.9% | -0.094 |
| `St1_2018` | `mamba` | 0.626 → 0.982 | -26.1% | -21.1% | -31.5% | +114.9% | +1.161 |
| `St1_2018` | `proportional` | 0.892 → 0.973 | -8.7% | -7.4% | -10.9% | +40.2% | +0.132 |

### 滑窗指标解读

- `CB_HAB4`：`norm_0` 整体明显有利，尤其 `proportional` 的 MAE、RMSE、CRPS、`lp_nb` 均大幅改善；`mamba` 的 MAE 小幅变差，但 RMSE/CRPS/`lp_nb` 仍改善。
- `PNR_1z`：表现分化明显。`mamba` 是最受益的背景模型；`kernel` 明显变差；`proportional` 混合，RMSE 略好但 MAE/CRPS 变差。
- `St1_2018`：`mamba` 与 `proportional` 明显改善；`kernel` 虽然 coverage 更接近 0.95，但误差和 `lp_nb` 变差。
- W95 全部增大，说明 `norm_0` 的收益部分来自更宽预测区间带来的保守校准。

## `nll_time` 指标变化

以下为从每个 run 的 `metrics_test_best_1.json` 直接重新聚合的 `nll_test_time` 均值。

| 数据集 | bg | 非 `norm_0` | `norm_0` | Δ | 结论 |
|---|---|---:|---:|---:|---|
| `CB_HAB4` | `kernel` | -4.7456 | -4.7937 | -0.0481 | 改善 |
| `CB_HAB4` | `mamba` | -4.7222 | -4.7863 | -0.0642 | 改善 |
| `CB_HAB4` | `proportional` | -4.5327 | -4.5701 | -0.0374 | 改善 |
| `PNR_1z` | `kernel` | -6.3851 | -6.3850 | +0.0001 | 基本持平/微变差 |
| `PNR_1z` | `mamba` | -6.3926 | -6.4003 | -0.0078 | 改善 |
| `PNR_1z` | `proportional` | -6.3751 | -6.3858 | -0.0107 | 改善 |
| `St1_2018` | `kernel` | -4.8505 | -4.8496 | +0.0009 | 基本持平/微变差 |
| `St1_2018` | `mamba` | -4.8306 | -4.8503 | -0.0197 | 改善 |
| `St1_2018` | `proportional` | -4.8488 | -4.8496 | -0.0008 | 基本持平/微改善 |

### Train/Val/Test `nll_time` 对比

| 数据集 | bg | train time | val time | test time |
|---|---|---:|---:|---:|
| `CB_HAB4` | `kernel` | -5.1726 → -5.1741 (-0.0015) | -6.0018 → -6.0094 (-0.0076) | -4.7456 → -4.7937 (-0.0481) |
| `CB_HAB4` | `mamba` | -5.1803 → -5.1800 (+0.0003) | -5.9955 → -6.0072 (-0.0117) | -4.7222 → -4.7863 (-0.0642) |
| `CB_HAB4` | `proportional` | -5.1029 → -5.1489 (-0.0459) | -5.9652 → -5.9794 (-0.0142) | -4.5327 → -4.5701 (-0.0374) |
| `PNR_1z` | `kernel` | -5.9912 → -5.9926 (-0.0014) | -6.4739 → -6.4718 (+0.0022) | -6.3851 → -6.3850 (+0.0001) |
| `PNR_1z` | `mamba` | -6.0103 → -6.0180 (-0.0078) | -6.5439 → -6.5403 (+0.0035) | -6.3926 → -6.4003 (-0.0078) |
| `PNR_1z` | `proportional` | -5.9942 → -6.0076 (-0.0135) | -6.5290 → -6.5316 (-0.0026) | -6.3751 → -6.3858 (-0.0107) |
| `St1_2018` | `kernel` | -5.5437 → -5.5445 (-0.0008) | -5.5769 → -5.5763 (+0.0006) | -4.8505 → -4.8496 (+0.0009) |
| `St1_2018` | `mamba` | -5.5246 → -5.5452 (-0.0206) | -5.5639 → -5.5773 (-0.0133) | -4.8306 → -4.8503 (-0.0197) |
| `St1_2018` | `proportional` | -5.5402 → -5.5438 (-0.0036) | -5.5764 → -5.5771 (-0.0008) | -4.8488 → -4.8496 (-0.0008) |

### `nll_time` 解读

- Test split 上，`norm_0` 使 `nll_test_time` 在 7/9 组改善，平均 Δ 为 -0.0209。
- `CB_HAB4` 是最稳定受益的数据集，三个背景模型 test time NLL 均改善，其中 `mamba` 改善最大，Δ=-0.0642。
- `PNR_1z` 中 `mamba` 与 `proportional` 改善，`kernel` 基本持平但略微变差，Δ=+0.0001。
- `St1_2018` 中 `mamba` 改善较明显，`proportional` 基本持平但略有改善，`kernel` 基本持平但略微变差。
- 与滑窗指标相比，`nll_time` 对 `norm_0` 的评价更正面，尤其在 `CB_HAB4`、`PNR_1z_mamba`、`PNR_1z_proportional` 和 `St1_2018_mamba` 上更稳定。

## 方向性统计

按 9 个配对组统计：

| 指标 | 改善组数 | 变差组数 | 平均变化 |
|---|---:|---:|---:|
| `nll_test_time` | 7 | 2 | -0.0209 |
| MAE | 5 | 4 | -3.51 |
| RMSE | 7 | 2 | -4.30 |
| CRPS | 6 | 3 | -4.92 |
| W95 | 0 | 9 | +102.25 |
| `lp_nb` | 8 | 1 | +0.646 |
| coverage 接近 0.95 | 6 | 3 | 平均绝对偏差 -0.062 |

## 建议

- 如果关注滑窗预测误差与概率评分，优先保留 `norm_0` 的 `CB_HAB4_proportional`、`PNR_1z_mamba`、`St1_2018_mamba`、`St1_2018_proportional`。
- 如果关注 `nll_test_time`，`norm_0` 整体值得保留，尤其是 `CB_HAB4` 全部背景模型、`PNR_1z_mamba`、`PNR_1z_proportional`、`St1_2018_mamba`。
- `PNR_1z_kernel_norm_0` 不建议作为默认选择：滑窗误差明显变差，且 `nll_test_time` 仅基本持平但略微变差。
- `St1_2018_kernel_norm_0` 也不建议仅因 coverage 接近 0.95 而采用，因为 MAE/RMSE/CRPS 与 `lp_nb` 均变差，`nll_test_time` 也只是基本持平。
- 后续可以进一步检查 `norm_0` 导致 W95 全面扩大的原因，判断其是否是合理校准还是过度保守。
