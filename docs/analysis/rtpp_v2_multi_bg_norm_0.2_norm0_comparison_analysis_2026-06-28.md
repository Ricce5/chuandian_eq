# `rtpp_v2_multi_bg_norm_0.2` 中 `norm_0` 与对应非 `norm_0` 组的指标变化分析

## 结论摘要

- 本文只比较同一数据集、同一背景模型下 `bg_norm_weight=0.2` 与 `bg_norm_weight=0`（`norm_0`）的差异。
- 可配对组共 9 组：`CB_HAB4`、`PNR_1z`、`St1_2018` × `kernel/mamba/proportional`，每组均有 `seed=0,1,2`。
- `CB_HAB1a` 在该实验中没有 `norm_0` 组，因此不纳入 norm 主比较。
- 测试 `nll_time` 上，`norm_0` 在 8/9 组改善，仅 `St1_2018_kernel` 微变差；最大收益来自 `CB_HAB4_mamba`（Δ=-0.3041）、`CB_HAB4_kernel`（Δ=-0.2695）、`CB_HAB4_proportional`（Δ=-0.1531）。
- 滑窗预测指标上，`norm_0` 明显更分化：MAE 仅 3/9 组改善，RMSE 3/9 组改善，CRPS 4/9 组改善，但 `lp_nb` 7/9 组改善。
- `norm_0` 会系统性扩大预测区间：W95 在 9/9 组全部增大，且 coverage 在 8/9 组更接近 0.95；但部分组的误差显著恶化，说明校准改善可能来自过宽区间。
- 综合推荐保留 `CB_HAB4_mamba_norm_0`、`CB_HAB4_proportional_norm_0`、`St1_2018_proportional_norm_0`；谨慎使用 `PNR_1z_mamba_norm_0` 与 `PNR_1z_proportional_norm_0`；不建议默认使用 `PNR_1z_kernel_norm_0`、`St1_2018_kernel_norm_0`、`St1_2018_mamba_norm_0`。

## 分析口径

实验目录：

- `/root/autodl-tmp/em_eqf/experiments/rtpp_v2_multi_bg_norm_0.2`

主要数据来源：

- NLL 汇总：`experiments/rtpp_v2_multi_bg_norm_0.2/reports/rtpp_v2_grid_metrics_group_mean_std.csv`
- 滑窗汇总：`experiments/rtpp_v2_multi_bg_norm_0.2/reports/sliding_window_eval_metrics_group_mean_std.csv`
- 每 run 测试 NLL：`experiments/rtpp_v2_multi_bg_norm_0.2/runs/*/metrics_test_best_1.json`
- 每 run 滑窗指标：`experiments/rtpp_v2_multi_bg_norm_0.2/runs/*/sliding_window_eval_metrics.json`

注意：

- `norm_0` 指对应配置中的 `loss_weights.bg_norm_weight: 0`；非 `norm_0` 对应基础配置中的 `bg_norm_weight: 0.2`。
- 实验配置 `ckpt_select=best`，因此本文所有 NLL 使用 `metrics_test_best_1.json`。
- 表中 `Δ = norm_0 - 非 norm_0`。
- `nll_time`、`nll_total`、MAE、RMSE、CRPS、W95 越低越好；`lp_nb` 越高越好；coverage 越接近 0.95 越好。

## 可配对组

| 数据集 | 背景模型 | 非 `norm_0` | `norm_0` | seeds |
|---|---|---|---|---|
| `CB_HAB4` | `kernel` | `rtpp_v2_bg_CB_HAB4_kernel` | `rtpp_v2_bg_CB_HAB4_kernel_norm_0` | 0,1,2 |
| `CB_HAB4` | `mamba` | `rtpp_v2_bg_CB_HAB4_mamba` | `rtpp_v2_bg_CB_HAB4_mamba_norm_0` | 0,1,2 |
| `CB_HAB4` | `proportional` | `rtpp_v2_bg_CB_HAB4_proportional` | `rtpp_v2_bg_CB_HAB4_proportional_norm_0` | 0,1,2 |
| `PNR_1z` | `kernel` | `rtpp_v2_bg_PNR_1z_kernel` | `rtpp_v2_bg_PNR_1z_kernel_norm_0` | 0,1,2 |
| `PNR_1z` | `mamba` | `rtpp_v2_bg_PNR_1z_mamba` | `rtpp_v2_bg_PNR_1z_mamba_norm_0` | 0,1,2 |
| `PNR_1z` | `proportional` | `rtpp_v2_bg_PNR_1z_proportional` | `rtpp_v2_bg_PNR_1z_proportional_norm_0` | 0,1,2 |
| `St1_2018` | `kernel` | `rtpp_v2_bg_St1_2018_kernel` | `rtpp_v2_bg_St1_2018_kernel_norm_0` | 0,1,2 |
| `St1_2018` | `mamba` | `rtpp_v2_bg_St1_2018_mamba` | `rtpp_v2_bg_St1_2018_mamba_norm_0` | 0,1,2 |
| `St1_2018` | `proportional` | `rtpp_v2_bg_St1_2018_proportional` | `rtpp_v2_bg_St1_2018_proportional_norm_0` | 0,1,2 |

## `nll_time` 指标变化

以下为 `best` checkpoint 的测试集 NLL 均值，直接从每个 run 的 `metrics_test_best_1.json` 聚合。

| 数据集 | bg | 非 `norm_0` `nll_time` | `norm_0` `nll_time` | Δ time | 非 `norm_0` `nll_total` | `norm_0` `nll_total` | Δ total | time 改善 seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CB_HAB4` | `kernel` | -4.5508 ± 0.0887 | -4.8203 ± 0.0275 | -0.2695 | -4.5682 | -4.8203 | -0.2520 | 3/3 |
| `CB_HAB4` | `mamba` | -4.5079 ± 0.0575 | -4.8121 ± 0.0396 | -0.3041 | -4.5237 | -4.8121 | -0.2883 | 3/3 |
| `CB_HAB4` | `proportional` | -3.9871 ± 0.1490 | -4.1402 ± 0.1258 | -0.1531 | -4.0063 | -4.1402 | -0.1339 | 3/3 |
| `PNR_1z` | `kernel` | -6.3857 ± 0.0039 | -6.3999 ± 0.0095 | -0.0142 | -6.3950 | -6.3999 | -0.0048 | 3/3 |
| `PNR_1z` | `mamba` | -6.4092 ± 0.0073 | -6.4156 ± 0.0077 | -0.0064 | -6.4244 | -6.4156 | +0.0089 | 3/3 |
| `PNR_1z` | `proportional` | -6.3819 ± 0.0059 | -6.3887 ± 0.0024 | -0.0069 | -6.4035 | -6.3887 | +0.0148 | 3/3 |
| `St1_2018` | `kernel` | -4.8783 ± 0.0051 | -4.8762 ± 0.0039 | +0.0021 | -4.8896 | -4.8762 | +0.0134 | 1/3 |
| `St1_2018` | `mamba` | -4.8411 ± 0.0135 | -4.8753 ± 0.0041 | -0.0342 | -4.8733 | -4.8753 | -0.0020 | 3/3 |
| `St1_2018` | `proportional` | -4.8588 ± 0.0027 | -4.8770 ± 0.0016 | -0.0182 | -4.8748 | -4.8770 | -0.0022 | 3/3 |

### NLL 解读

- `CB_HAB4` 是 `norm_0` 收益最明显的数据集，三个背景模型在 3/3 seeds 上都改善，且幅度远大于其他数据集。
- `mamba` 的 `nll_time` 在三个数据集上都改善，其中 `CB_HAB4_mamba` 改善最大。
- `PNR_1z` 的 `nll_time` 全部改善，但幅度较小；`mamba/proportional` 的 `nll_total` 反而变差，说明去掉背景归一化项后总目标与时间似然的方向不完全一致。
- `St1_2018_kernel_norm_0` 是唯一 `nll_time` 平均变差的配对组，且只有 1/3 seeds 改善。
- 从 NLL 角度看，`norm_0` 多数有利；但是否作为默认配置，还需要结合滑窗误差与区间宽度判断。

## seed 级 `nll_time` 变化

以下表格展示每个 seed 的 `nll_test_time` 差值，负数表示 `norm_0` 更好。

| 数据集 | bg | seed 0 | seed 1 | seed 2 | mean Δ |
|---|---|---:|---:|---:|---:|
| `CB_HAB4` | `kernel` | -0.3566 | -0.3135 | -0.1384 | -0.2695 |
| `CB_HAB4` | `mamba` | -0.3604 | -0.1923 | -0.3596 | -0.3041 |
| `CB_HAB4` | `proportional` | -0.0392 | -0.2089 | -0.2111 | -0.1531 |
| `PNR_1z` | `kernel` | -0.0220 | -0.0018 | -0.0187 | -0.0142 |
| `PNR_1z` | `mamba` | -0.0074 | -0.0049 | -0.0067 | -0.0064 |
| `PNR_1z` | `proportional` | -0.0026 | -0.0091 | -0.0089 | -0.0069 |
| `St1_2018` | `kernel` | +0.0080 | -0.0070 | +0.0053 | +0.0021 |
| `St1_2018` | `mamba` | -0.0457 | -0.0156 | -0.0414 | -0.0342 |
| `St1_2018` | `proportional` | -0.0138 | -0.0220 | -0.0188 | -0.0182 |

## 滑窗预测指标变化

| 数据集 | bg | coverage | MAE Δ | RMSE Δ | CRPS Δ | W95 Δ | `lp_nb` Δ | 综合 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CB_HAB4` | `kernel` | 0.667 → 0.789 | +6.3% | +0.2% | -5.6% | +65.7% | +3.427 | 2/4 |
| `CB_HAB4` | `mamba` | 0.579 → 0.877 | -9.2% | -11.8% | -18.5% | +67.1% | +4.173 | 4/4 |
| `CB_HAB4` | `proportional` | 0.386 → 0.667 | -3.6% | -1.6% | -8.6% | +41.7% | +24.535 | 4/4 |
| `PNR_1z` | `kernel` | 0.907 → 0.984 | +48.4% | +9.8% | +19.2% | +149.8% | -0.434 | 0/4 |
| `PNR_1z` | `mamba` | 0.858 → 0.880 | +4.9% | +2.8% | +2.8% | +26.7% | +0.108 | 1/4 |
| `PNR_1z` | `proportional` | 0.880 → 0.896 | +3.0% | -0.1% | +0.2% | +13.4% | +0.185 | 2/4 |
| `St1_2018` | `kernel` | 0.923 → 0.874 | +77.8% | +56.2% | +62.9% | +63.7% | -0.649 | 0/4 |
| `St1_2018` | `mamba` | 0.626 → 0.901 | +50.4% | +47.4% | +29.3% | +249.2% | +0.933 | 1/4 |
| `St1_2018` | `proportional` | 0.869 → 0.946 | -4.9% | +1.0% | -4.7% | +35.6% | +0.384 | 3/4 |

表中“综合”统计 MAE、RMSE、CRPS、`lp_nb` 四项中有几项改善。

### 滑窗指标解读

- `CB_HAB4_mamba_norm_0` 与 `CB_HAB4_proportional_norm_0` 是滑窗上最稳的两组，MAE/RMSE/CRPS/`lp_nb` 全部改善。
- `CB_HAB4_kernel_norm_0` 的 CRPS、`lp_nb` 和 coverage 改善，但 MAE/RMSE 小幅变差，属于混合收益。
- `PNR_1z_kernel_norm_0` 明显不推荐：MAE +48.4%、CRPS +19.2%、`lp_nb` -0.434，且 coverage 从 0.907 提升到 0.984，偏过度保守。
- `St1_2018_kernel_norm_0` 是最差组合：MAE +77.8%、RMSE +56.2%、CRPS +62.9%、`lp_nb` -0.649，coverage 也从 0.923 退到 0.874。
- `St1_2018_mamba_norm_0` 虽然 `nll_time` 改善明显，但滑窗误差大幅变差，说明时间似然收益没有转化为窗口计数预测收益。
- `St1_2018_proportional_norm_0` 是 `St1_2018` 中最可取的 `norm_0` 组：coverage 接近 0.95，MAE/CRPS/`lp_nb` 改善，RMSE 仅小幅变差。

## 与无 `norm_0` 背景基线的关系

该实验还包含 `no_bg`、`kernel_256` 以及 `CB_HAB1a` 的无 `norm_0` 背景组。它们不参与 norm 配对，但可作为背景模型选择参考。

| 数据集 | 按 test `nll_time` 排名靠前的非 `norm_0` 组 |
|---|---|
| `CB_HAB1a` | `no_bg`(-5.2021) > `kernel`(-5.1610) > `kernel_256`(-5.1573) > `mamba`(-5.0521) > `proportional`(-4.8828) |
| `CB_HAB4` | `no_bg`(-4.8403) > `kernel`(-4.5508) > `kernel_256`(-4.5180) > `mamba`(-4.5079) > `proportional`(-3.9871) |
| `PNR_1z` | `mamba`(-6.4092) > `no_bg`(-6.3996) > `kernel`(-6.3857) > `proportional`(-6.3819) > `kernel_256`(-6.3816) |
| `St1_2018` | `kernel`(-4.8783) > `kernel_256`(-4.8782) > `no_bg`(-4.8681) > `proportional`(-4.8588) > `mamba`(-4.8411) |

注意：

- `CB_HAB1a` 没有 norm 对照，不能推断 `norm_0` 对它是否有利。
- `CB_HAB4` 的无背景模型 `no_bg` 在 test `nll_time` 上很强，但 forecast 滑窗 RMSE 明显差于 `mamba/kernel/proportional`，因此背景模型选择不能只看 NLL。
- `St1_2018` 的非 `norm_0` NLL 最优是 `kernel`，但其 `norm_0` 版本在滑窗上明显恶化。

## 按背景模型归纳

### `mamba`

- `nll_time` 在 3/3 数据集上改善，尤其 `CB_HAB4_mamba_norm_0` 改善最大。
- 滑窗表现分化：`CB_HAB4` 全面改善，`PNR_1z` 仅 `lp_nb` 改善，`St1_2018` 误差显著变差。
- 适合在 `CB_HAB4` 上使用 `norm_0`，但不适合无条件跨数据集推广。

### `proportional`

- `nll_time` 在 3/3 数据集上改善。
- 滑窗上 `CB_HAB4` 全面改善，`St1_2018` 较好，`PNR_1z` 混合。
- 是本实验中最稳健的 `norm_0` 候选背景之一。

### `kernel`

- `nll_time` 在 `CB_HAB4` 与 `PNR_1z` 改善，但 `St1_2018` 微变差。
- 滑窗上只有 `CB_HAB4` 有部分收益，`PNR_1z` 和 `St1_2018` 明显恶化。
- 不建议作为 `norm_0` 的默认背景模型。

## 最终建议

- 优先保留：`rtpp_v2_bg_CB_HAB4_mamba_norm_0`、`rtpp_v2_bg_CB_HAB4_proportional_norm_0`、`rtpp_v2_bg_St1_2018_proportional_norm_0`。
- 可以保留但需看目标：`rtpp_v2_bg_CB_HAB4_kernel_norm_0`，适合重视 CRPS/`lp_nb`/coverage，但不适合只看 MAE/RMSE。
- 谨慎使用：`rtpp_v2_bg_PNR_1z_mamba_norm_0`、`rtpp_v2_bg_PNR_1z_proportional_norm_0`，因为 NLL 改善但滑窗误差基本不改善。
- 不建议默认使用：`rtpp_v2_bg_PNR_1z_kernel_norm_0`、`rtpp_v2_bg_St1_2018_kernel_norm_0`、`rtpp_v2_bg_St1_2018_mamba_norm_0`。
- 后续建议增加 `bg_norm_weight=0.05/0.1` 的中间权重，验证当前现象是“0.2 过强”还是“完全去掉归一化约束”本身带来的收益。
