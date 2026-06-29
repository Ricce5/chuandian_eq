# ETAS / RECAST 背景模型结果分析（补充 `CB_HAB4` 的 RECAST `norm_0`，bg_norm_weight=0.2）

## 分析口径

- 实验目录：
  - `experiments/etas_multi_ds_bg_norm_0.2`
  - `experiments/rtpp_v2_multi_bg_norm_0.2`
- 本文将 `rtpp_v2` 记为 **RECAST**。
- 本文以 `nll_test_time` 为主指标，越低越好。
- ETAS 包含常规背景、`no_bg`、以及 `norm_0` 背景。
- `norm_0` 目前只在 `PNR_1z`、`St1_2018`、`CB_HAB4` 上有对应 ETAS run；`CB_HAB1a` 没有 `norm_0` 组。
- RECAST 实验目录也包含 `no_bg` 与 `norm_0` 组；本文只补充 `CB_HAB4` 的 RECAST `norm_0/no_bg` 分析，其他数据集段落保持原口径。
- ETAS 的 `norm_0` NLL 汇总直接从各 run 的 `metrics_test_best_1.json` 重新聚合，避免旧汇总文件遗漏 `norm_0` 组。

使用的汇总来源：

- ETAS：`experiments/etas_multi_ds_bg_norm_0.2/runs/*/metrics_test_best_1.json`
- RECAST：`experiments/rtpp_v2_multi_bg_norm_0.2/reports/rtpp_v2_grid_metrics_group_mean_std.csv`；`CB_HAB4` 的 `norm_0/no_bg` 结果从 `experiments/rtpp_v2_multi_bg_norm_0.2/runs/*/metrics_test_best_1.json` 补充聚合。

## 主要结论

按 `nll_test_time`，加入 ETAS `norm_0` 后的跨模型最佳结果为：

| 数据集 | 最优模型 | 最优背景 | nll_test_time |
|---|---|---|---:|
| `PNR_1z` | RECAST | `mamba` | -6.409206 ± 0.007262 |
| `CB_HAB1a` | RECAST | `kernel` | -5.160999 ± 0.011582 |
| `St1_2018` | RECAST | `kernel` | -4.878340 ± 0.005066 |
| `CB_HAB4` | ETAS | `no_bg` | -4.851626 ± 0.000000 |

核心变化：

- `norm_0` 对 ETAS 的 `nll_test_time` 整体偏正面：9 个可配对背景中 7 个改善，2 个基本持平但略变差。
- `CB_HAB4` 是 `norm_0` 收益最大的 ETAS 数据集：`proportional`、`kernel`、`mamba` 三个背景的 `nll_test_time` 全部改善。
- `CB_HAB4` 的 RECAST 也明显受益于 `norm_0/no_bg`：RECAST 内部最佳从原始 `kernel`（-4.550773）变为 `no_bg`（-4.840299），若排除 `no_bg` 则为 `kernel_norm_0`（-4.820282）。
- `PNR_1z` 中 `mamba_norm0` 成为 ETAS 内部最佳背景，但跨模型仍略弱于 RECAST `mamba`。
- `St1_2018` 中 `norm_0` 没有改变 NLL 排名第一：ETAS 内部仍是原始 `kernel/kernel_256` 最好；跨模型仍是 RECAST `kernel` 最好。
- `no_bg` 在 `CB_HAB1a` 和 `CB_HAB4` 的 `nll_test_time` 上很强，但它在滑动窗 count forecast 中明显变差，因此不能单独作为预测默认选择。

## ETAS：按数据集排序

### `PNR_1z`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `mamba_norm0` | -6.400337 ± 0.001244 |
| 2 | `mamba` | -6.392565 ± 0.002519 |
| 3 | `proportional_norm0` | -6.385776 ± 0.000000 |
| 4 | `kernel_256` | -6.385070 ± 0.001266 |
| 5 | `kernel` | -6.385069 ± 0.001267 |
| 6 | `kernel_norm0` | -6.384972 ± 0.000476 |
| 7 | `no_bg` | -6.384871 ± 0.000000 |
| 8 | `proportional` | -6.375057 ± 0.000000 |

`mamba_norm0` 是 ETAS 内部最优；`proportional_norm0` 也明显优于原始 `proportional`。`kernel_norm0` 与原始 `kernel` 基本持平但略差。

### `CB_HAB1a`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `no_bg` | -5.103908 ± 0.000000 |
| 2 | `mamba` | -5.001504 ± 0.014150 |
| 3 | `kernel_256` | -4.987374 ± 0.004708 |
| 4 | `kernel` | -4.987265 ± 0.004344 |
| 5 | `proportional` | -4.862419 ± 0.000000 |

该数据集没有 `norm_0` 组。`no_bg` 在时间似然上最好，但滑动窗 count forecast 中误差和 W95 都很差，因此不建议只凭 `nll_test_time` 选择 `no_bg`。

### `St1_2018`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `kernel` | -4.850541 ± 0.000022 |
| 2 | `kernel_256` | -4.850541 ± 0.000022 |
| 3 | `mamba_norm0` | -4.850255 ± 0.000366 |
| 4 | `proportional_norm0` | -4.849631 ± 0.000000 |
| 5 | `kernel_norm0` | -4.849629 ± 0.000057 |
| 6 | `proportional` | -4.848795 ± 0.000000 |
| 7 | `no_bg` | -4.847757 ± 0.000000 |
| 8 | `mamba` | -4.830576 ± 0.004180 |

`mamba_norm0` 大幅修复了原始 `mamba` 的弱表现，但仍没有超过原始 `kernel/kernel_256`。`kernel_norm0` 反而略弱于原始 `kernel`。

### `CB_HAB4`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `no_bg` | -4.851626 ± 0.000000 |
| 2 | `kernel_norm0` | -4.793688 ± 0.000545 |
| 3 | `mamba_norm0` | -4.786319 ± 0.002862 |
| 4 | `kernel_256` | -4.745848 ± 0.000723 |
| 5 | `kernel` | -4.745597 ± 0.000696 |
| 6 | `mamba` | -4.722153 ± 0.005033 |
| 7 | `proportional_norm0` | -4.570122 ± 0.000000 |
| 8 | `proportional` | -4.532691 ± 0.000000 |

`CB_HAB4` 是 `norm_0` 最明显受益的数据集。若排除 `no_bg`，ETAS 内部最佳从原始 `kernel/kernel_256` 切换到 `kernel_norm0`，`mamba_norm0` 也非常接近。

## RECAST：按数据集排序

### `PNR_1z`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `mamba` | -6.409206 ± 0.007262 |
| 2 | `kernel` | -6.385700 ± 0.003916 |
| 3 | `proportional` | -6.381883 ± 0.005879 |
| 4 | `kernel_256` | -6.381585 ± 0.008252 |

RECAST `mamba` 仍是 `PNR_1z` 的跨模型最佳，略优于 ETAS `mamba_norm0`。

### `CB_HAB1a`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `kernel` | -5.160999 ± 0.011582 |
| 2 | `kernel_256` | -5.157281 ± 0.010715 |
| 3 | `mamba` | -5.052124 ± 0.033484 |
| 4 | `proportional` | -4.882752 ± 0.002484 |

RECAST `kernel` 明显优于 ETAS 的所有可用配置。

### `St1_2018`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `kernel` | -4.878340 ± 0.005066 |
| 2 | `kernel_256` | -4.878236 ± 0.005256 |
| 3 | `proportional` | -4.858777 ± 0.002661 |
| 4 | `mamba` | -4.841114 ± 0.013529 |

RECAST `kernel` 仍是该数据集的跨模型最佳。

### `CB_HAB4`

| 排名 | 背景模型 | nll_test_time |
|---:|---|---:|
| 1 | `no_bg` | -4.840299 ± 0.005201 |
| 2 | `kernel_norm_0` | -4.820282 ± 0.027539 |
| 3 | `mamba_norm_0` | -4.812075 ± 0.039633 |
| 4 | `kernel` | -4.550773 ± 0.088656 |
| 5 | `kernel_256` | -4.517991 ± 0.005563 |
| 6 | `mamba` | -4.507941 ± 0.057451 |
| 7 | `proportional_norm_0` | -4.140222 ± 0.125820 |
| 8 | `proportional` | -3.987147 ± 0.149045 |

加入 RECAST `norm_0/no_bg` 后，`CB_HAB4` 的 RECAST 时间似然明显改善；`no_bg` 是 RECAST 内部最优，排除 `no_bg` 时 `kernel_norm_0` 最好，并超过 ETAS 的 `kernel_norm0/mamba_norm0`。

## `norm_0` 对 ETAS `nll_test_time` 的影响

下表为 `norm_0 - 非 norm_0`，负数表示 `nll_test_time` 改善。

| 数据集 | 背景 | 非 `norm_0` | `norm_0` | Δ |
|---|---|---:|---:|---:|
| `PNR_1z` | `proportional` | -6.375057 | -6.385776 | -0.010719 |
| `PNR_1z` | `kernel` | -6.385069 | -6.384972 | +0.000097 |
| `PNR_1z` | `mamba` | -6.392565 | -6.400337 | -0.007772 |
| `CB_HAB4` | `proportional` | -4.532691 | -4.570122 | -0.037431 |
| `CB_HAB4` | `kernel` | -4.745597 | -4.793688 | -0.048091 |
| `CB_HAB4` | `mamba` | -4.722153 | -4.786319 | -0.064166 |
| `St1_2018` | `proportional` | -4.848795 | -4.849631 | -0.000836 |
| `St1_2018` | `kernel` | -4.850541 | -4.849629 | +0.000913 |
| `St1_2018` | `mamba` | -4.830576 | -4.850255 | -0.019680 |

整体判断：

- `norm_0` 对 `nll_test_time` 是 7/9 改善、2/9 轻微变差。
- `mamba_norm0` 在三个可配对数据集上全部改善，是最稳定的 `norm_0` 背景。
- `kernel_norm0` 只在 `CB_HAB4` 上明显改善；在 `PNR_1z` 和 `St1_2018` 上基本持平但略差。
- `proportional_norm0` 三个数据集均改善，但绝对排名通常仍不如 `mamba` 或 `kernel` 类背景。

## ETAS 与 RECAST 的背景偏好差异

加入 `norm_0` 后，ETAS 对灵活背景的收益更明显：

- `PNR_1z`：ETAS 内部最佳从 `mamba` 变为 `mamba_norm0`；RECAST 仍偏好 `mamba`。
- `CB_HAB4`：ETAS 背景模型全面受益于 `norm_0`；RECAST 的 `norm_0/no_bg` 收益更大，RECAST 内部从原始 `kernel` 切换到 `no_bg`，排除 `no_bg` 时切换到 `kernel_norm_0`。
- `St1_2018`：ETAS 的 `mamba_norm0` 接近 `kernel`，但没有超过；RECAST 仍稳定偏好 `kernel`。
- `CB_HAB1a`：当前缺少 ETAS `norm_0`，暂不能判断是否会类似 `PNR_1z` 或 `St1_2018`。

RECAST 的 `CB_HAB4` 背景偏好已受 `norm_0/no_bg` 更新影响；其他数据集段落在本文中保持原口径。

## 数据集层面解释

### `PNR_1z`

两类模型都偏好 `mamba`，说明该数据集可能存在较明显的慢变背景率或非平稳外部驱动。`norm_0` 进一步提升了 ETAS `mamba`，但 RECAST `mamba` 仍略优：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | `mamba_norm0` | -6.400337 ± 0.001244 |
| RECAST | `mamba` | -6.409206 ± 0.007262 |

### `CB_HAB1a`

当前无 ETAS `norm_0` 对照。RECAST `kernel` 仍明显领先：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | `no_bg` | -5.103908 ± 0.000000 |
| RECAST | `kernel` | -5.160999 ± 0.011582 |

需要注意，ETAS `no_bg` 虽然 `nll_test_time` 好，但滑动窗 count forecast 很差。

### `St1_2018`

RECAST `kernel` 是跨模型最佳；ETAS 中原始 `kernel/kernel_256` 仍略优于 `mamba_norm0`：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | `kernel / kernel_256` | -4.850541 ± 0.000022 |
| RECAST | `kernel` | -4.878340 ± 0.005066 |

`mamba_norm0` 对 ETAS 的提升主要体现在把原始 `mamba` 从最差组提升到接近第一梯队。

### `CB_HAB4`

加入 RECAST `norm_0/no_bg` 后，`CB_HAB4` 的跨模型 NLL 排序发生变化：ETAS `no_bg` 仍是全局最优，但 RECAST `no_bg` 与 RECAST `kernel_norm_0/mamba_norm_0` 已超过 ETAS 的 `norm_0` 背景：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | `no_bg` | -4.851626 ± 0.000000 |
| RECAST | `no_bg` | -4.840299 ± 0.005201 |
| RECAST（排除 `no_bg`） | `kernel_norm_0` | -4.820282 ± 0.027539 |
| ETAS（排除 `no_bg`） | `kernel_norm0` | -4.793688 ± 0.000545 |

## 选择建议

### 若按 `nll_test_time` 选择默认背景

| 数据集 | ETAS 推荐 | RECAST 推荐 |
|---|---|---|
| `PNR_1z` | `mamba_norm0` | `mamba` |
| `CB_HAB1a` | `mamba` 或补跑 `mamba_norm0` 后再定 | `kernel` |
| `St1_2018` | `kernel / kernel_256`；`mamba_norm0` 作为备选 | `kernel` |
| `CB_HAB4` | `kernel_norm0`；若允许无背景则 `no_bg` | `kernel_norm_0`；若允许无背景则 `no_bg` |

### 若综合滑动窗预测

- `PNR_1z`、`St1_2018` 的 ETAS 首选应偏向 `mamba_norm0`；`CB_HAB4` 需同时比较 ETAS `mamba_norm0` 与 RECAST `mamba_norm_0`。
- `CB_HAB1a` 当前首选 RECAST `mamba`，并建议补跑 ETAS `mamba_norm0`。
- 不建议仅因 `nll_test_time` 好就使用 ETAS `no_bg`，因为其滑动窗 RMSE/MAE/W95 表现很差。
- `kernel_norm0` 适合 `CB_HAB4` 的 `nll_test_time`，但不适合作为 `PNR_1z` 或 `St1_2018` 的滑动窗默认配置。

## 后续验证建议

1. 补跑 `CB_HAB1a` 的 ETAS `proportional_norm0`、`kernel_norm0`、`mamba_norm0`，补齐可比性。
2. 对 RECAST `norm_0` 做系统化复核，尤其确认 `CB_HAB4` 中 `no_bg/kernel_norm_0/mamba_norm_0` 的 NLL 收益是否能稳定转化为滑动窗预测收益。
3. 对 `mamba_norm0` 检查 coverage 与 W95，尤其是 `St1_2018`，确认是否过度保守。
4. 对 `no_bg` 做 NLL 与滑动窗指标的分解分析，解释为何时间似然好但 count forecast 差。
5. 对 `CB_HAB4` 的 RECAST `norm_0` 做 seed-level 诊断，确认 `kernel_norm_0/mamba_norm_0` 的方差和测试收益来源。
