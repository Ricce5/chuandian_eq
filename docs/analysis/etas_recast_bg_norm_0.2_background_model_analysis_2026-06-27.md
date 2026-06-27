# ETAS / RECAST 背景模型结果分析（bg_norm_weight=0.2）

## 分析口径

- 实验目录：
  - `experiments/etas_multi_ds_bg_norm_0.2`
  - `experiments/rtpp_v2_multi_bg_norm_0.2`
- 本文将 `rtpp_v2` 记为 **RECAST**。
- 主性能指标使用 `nll_test_time`，越小越好。
- `nll_test_total` 只作为参考，不作为主要排序依据。
- 延续对比设定：ETAS 不考虑 `no_bg`，只比较带背景模型的配置。
- 每个配置有 3 个 seed，两个实验的所有 run 均成功：
  - ETAS：60/60 runs 成功，指标齐全。
  - RECAST：48/48 runs 成功，指标齐全。

## 主要结论

### ETAS

| 数据集 | 最佳背景模型 | nll_test_time | nll_test_total 参考 |
|---|---:|---:|---:|
| PNR_1z | mamba | -6.392565 ± 0.002519 | -6.430131 |
| CB_HAB1a | mamba | -5.001504 ± 0.014150 | -5.025208 |
| St1_2018 | kernel / kernel_256 | -4.850541 ± 0.000022 | -4.863876 |
| CB_HAB4 | kernel_256 | -4.745848 ± 0.000723 | -4.757907 |

ETAS 中没有单一背景模型在所有数据集上最优。`mamba` 在 `PNR_1z` 和 `CB_HAB1a` 最好，说明这两个数据集更受非平稳背景变化影响；`kernel/kernel_256` 在 `St1_2018` 和 `CB_HAB4` 更好，说明更平滑、更强先验的背景估计在这些数据集上泛化更稳。

如果必须选一个 ETAS 默认背景模型，建议优先选 `kernel_256` 或 `kernel`；如果目标数据集是 `PNR_1z` 或 `CB_HAB1a`，则应优先尝试 `mamba`。

### RECAST

| 数据集 | 最佳背景模型 | nll_test_time | nll_test_total 参考 |
|---|---:|---:|---:|
| PNR_1z | mamba | -6.409206 ± 0.007262 | -6.424412 |
| CB_HAB1a | kernel | -5.160999 ± 0.011582 | -5.177064 |
| St1_2018 | kernel | -4.878340 ± 0.005066 | -4.889602 |
| CB_HAB4 | kernel | -4.550773 ± 0.088656 | -4.568244 |

RECAST 中 `kernel` 是整体最稳的背景模型，在 3/4 个数据集上取得最佳 `nll_test_time`。`mamba` 只在 `PNR_1z` 最好，但在其他数据集上不如 `kernel` 稳定。

如果必须选一个 RECAST 默认背景模型，建议选 `kernel`。

## 各模型内背景模型排序

### ETAS：按数据集排序

#### PNR_1z

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | mamba | -6.392565 ± 0.002519 | -6.430131 |
| 2 | kernel_256 | -6.385070 ± 0.001266 | -6.398992 |
| 3 | kernel | -6.385069 ± 0.001267 | -6.398991 |
| 4 | proportional | -6.375057 ± 0.000000 | -6.398614 |

#### CB_HAB1a

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | mamba | -5.001504 ± 0.014150 | -5.025208 |
| 2 | kernel_256 | -4.987374 ± 0.004708 | -5.016346 |
| 3 | kernel | -4.987265 ± 0.004344 | -5.016251 |
| 4 | proportional | -4.862419 ± 0.000000 | -4.896653 |

#### St1_2018

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | kernel | -4.850541 ± 0.000022 | -4.863876 |
| 1 | kernel_256 | -4.850541 ± 0.000022 | -4.863876 |
| 3 | proportional | -4.848795 ± 0.000000 | -4.865610 |
| 4 | mamba | -4.830576 ± 0.004180 | -4.864904 |

#### CB_HAB4

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | kernel_256 | -4.745848 ± 0.000723 | -4.757907 |
| 2 | kernel | -4.745597 ± 0.000696 | -4.757667 |
| 3 | mamba | -4.722153 ± 0.005033 | -4.734452 |
| 4 | proportional | -4.532691 ± 0.000000 | -4.548924 |

### RECAST：按数据集排序

#### PNR_1z

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | mamba | -6.409206 ± 0.007262 | -6.424412 |
| 2 | kernel | -6.385700 ± 0.003916 | -6.395033 |
| 3 | proportional | -6.381883 ± 0.005879 | -6.403530 |
| 4 | kernel_256 | -6.381585 ± 0.008252 | -6.390972 |

#### CB_HAB1a

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | kernel | -5.160999 ± 0.011582 | -5.177064 |
| 2 | kernel_256 | -5.157281 ± 0.010715 | -5.174223 |
| 3 | mamba | -5.052124 ± 0.033484 | -5.081412 |
| 4 | proportional | -4.882752 ± 0.002484 | -4.917253 |

#### St1_2018

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | kernel | -4.878340 ± 0.005066 | -4.889602 |
| 2 | kernel_256 | -4.878236 ± 0.005256 | -4.889503 |
| 3 | proportional | -4.858777 ± 0.002661 | -4.874814 |
| 4 | mamba | -4.841114 ± 0.013529 | -4.873307 |

#### CB_HAB4

| 排名 | 背景模型 | nll_test_time | nll_test_total 参考 |
|---:|---|---:|---:|
| 1 | kernel | -4.550773 ± 0.088656 | -4.568244 |
| 2 | kernel_256 | -4.517991 ± 0.005563 | -4.535899 |
| 3 | mamba | -4.507941 ± 0.057451 | -4.523734 |
| 4 | proportional | -3.987147 ± 0.149045 | -4.006331 |

## 背景模型层面的整体判断

### kernel / kernel_256

`kernel` 和 `kernel_256` 整体最稳定，尤其适合作为默认背景模型。

在 ETAS 中，`kernel_256` 的平均 rank 最好，`kernel` 与其非常接近。两者在 `St1_2018` 和 `CB_HAB4` 上表现最好，且 seed 方差很小。

在 RECAST 中，`kernel` 明显是最优默认选择，在 `CB_HAB1a`、`St1_2018`、`CB_HAB4` 上均为最佳。`kernel_256` 通常与 `kernel` 接近，但多数情况下略差。

这说明背景卷积窗口长度不是主要瓶颈，背景模型形式及其与主时间模型的交互更关键。

### mamba

`mamba` 背景更灵活，适合背景率具有明显非平稳变化的数据集。

它在 ETAS 的 `PNR_1z`、`CB_HAB1a` 上最好，也在 RECAST 的 `PNR_1z` 上最好。这说明这些数据集中，灵活背景模型确实能帮助解释慢变背景或复杂非平稳结构。

但 `mamba` 在 `St1_2018` 和 `CB_HAB4` 上不如 `kernel`，尤其 RECAST 中更明显。这可能是因为 RECAST 的主时间模型本身表达能力已经较强，过于灵活的背景模型容易与主时间模型竞争解释权，从而降低测试泛化。

### proportional

`proportional` 整体最弱，不建议作为默认背景模型。

它的表达能力较刚性，只能描述简单比例型背景变化，难以适应复杂非平稳背景或跨时间段分布漂移。尤其在 `CB_HAB4` 的 RECAST 上，`nll_test_time=-3.987147±0.149045`，明显弱于其他背景模型，并且 seed 方差也大。

## ETAS 与 RECAST 的背景偏好差异

ETAS 更依赖背景模型本身来吸收非平稳变化。由于 ETAS 的时间触发结构有较强统计先验，背景模型的灵活性会直接影响它能否区分背景事件与触发事件。因此在 `PNR_1z` 和 `CB_HAB1a` 上，`mamba` 背景带来了最好的 `nll_test_time`。

RECAST 的主时间模型表达能力更强，因此不一定需要过于灵活的背景模型。对 RECAST 来说，`kernel` 这种平滑、低方差、带先验约束的背景更适合作为辅助项。它既能提供背景信息，又不太容易和主时间模型争夺时序解释。

这也是为什么 RECAST 中 `kernel` 胜出 3/4 个数据集，而 `mamba` 只在 `PNR_1z` 胜出。

## 数据集层面解释

### PNR_1z

ETAS 和 RECAST 都偏好 `mamba` 背景。说明该数据集可能存在较明显的慢变背景率或非平稳外部驱动，灵活背景模型更容易捕捉这种变化。

按 `nll_test_time`，RECAST `mamba` 优于 ETAS `mamba`：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | mamba | -6.392565 ± 0.002519 |
| RECAST | mamba | -6.409206 ± 0.007262 |

### CB_HAB1a

ETAS 偏好 `mamba`，RECAST 偏好 `kernel`。

这说明 ETAS 需要更灵活的背景项来补偿自身时间结构的限制；而 RECAST 的主时间模型已经能够表达更复杂的时序依赖，因此平滑背景 `kernel` 更合适。

按 `nll_test_time`，RECAST `kernel` 明显优于 ETAS `mamba`：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | mamba | -5.001504 ± 0.014150 |
| RECAST | kernel | -5.160999 ± 0.011582 |

### St1_2018

两类模型都偏好 `kernel` 或 `kernel_256`。

该数据集事件数较多，平滑背景估计更稳定，复杂背景模型的额外灵活性没有带来收益。RECAST 的主时间模型在该数据集上更强，因此 `kernel` 背景即可取得最佳表现。

按 `nll_test_time`，RECAST `kernel` 优于 ETAS `kernel`：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | kernel / kernel_256 | -4.850541 ± 0.000022 |
| RECAST | kernel | -4.878340 ± 0.005066 |

### CB_HAB4

两类模型都偏好 `kernel` 系背景，但 ETAS 明显优于 RECAST。

该数据集存在明显的验证集到测试集分布差异，RECAST 在测试段的时间似然退化较明显，并且 seed 方差较大。ETAS 的结构先验更强，因此在该数据集上更稳。

按 `nll_test_time`：

| 模型 | 最佳背景 | nll_test_time |
|---|---|---:|
| ETAS | kernel_256 | -4.745848 ± 0.000723 |
| RECAST | kernel | -4.550773 ± 0.088656 |

## 选择建议

### 若按模型选择默认背景

| 模型 | 推荐默认背景 | 备选 | 不推荐 |
|---|---|---|---|
| ETAS | kernel_256 / kernel | mamba | proportional |
| RECAST | kernel | kernel_256；PNR_1z 可试 mamba | proportional |

### 若按数据集选择背景

| 数据集 | ETAS 推荐 | RECAST 推荐 |
|---|---|---|
| PNR_1z | mamba | mamba |
| CB_HAB1a | mamba | kernel |
| St1_2018 | kernel / kernel_256 | kernel |
| CB_HAB4 | kernel_256 / kernel | kernel，但需注意稳定性 |

## 后续验证建议

1. 对 RECAST 补跑 `no_bg`，尤其是 `CB_HAB4`，判断其测试退化来自背景模块还是主时间模型泛化。
2. 对 RECAST 尝试 `time_use_bg_context: true`，验证背景信息直接进入时间模型是否改善 `CB_HAB4`。
3. 对 `CB_HAB4` 做 sliding-window 或 rolling evaluation，定位验证段到测试段的 rate shift。
4. 对 `mamba` 背景增加正则或降低容量，检查是否能提升 `St1_2018` 和 `CB_HAB4` 的泛化稳定性。
5. 保留 `kernel` 与 `kernel_256` 两个候选，但后续可减少对 kernel size 的大范围搜索，因为二者差异整体很小。

