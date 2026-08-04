# `ETAS` 与 `RTPP v2` 不同测试起点下的滑窗模型排序分析

## 问题与结论

本文只回答一个问题：同一数据集、同一滑窗预测指标下，模型的排序是否会随评估起点从全区间变为验证起点、再变为测试起点而改变？

结论是会改变，而且不能由全区间的排序直接推断验证或测试区间的排序。

- `CB_HAB1a`：两类模型的全区间到验证起点的点预测排序大多稳定，但测试起点会重排中后段模型；`ETAS` 的测试 CRPS 榜首从 `mamba` 变为 `no_bg`。
- `CB_HAB4`：全区间到验证起点已有明显重排，验证到测试又会改变榜首。`ETAS` 的 RMSE/CRPS 从 `mamba_norm_0` 转到 `kernel_norm_0`，MAE 保持 `mamba_norm_0`；`RTPP v2` 三项点预测指标都从 `mamba_norm_0` 转到 `kernel_norm_0`。
- `PNR_1z`：从全区间到验证起点的榜首变化较大；`RTPP v2` 的点预测从全区间的 `mamba`、验证的 `proportional` 变为测试的 `proportional_norm_0`。测试只有 3 个窗口，不能把该变化视为稳健结论。
- `St1-2018`：验证到测试的点预测排序最稳定，尤其 `RTPP v2` 的 MAE/RMSE/CRPS 完全不变；全区间与验证起点仍可能不同。
- `W95` 的榜首在所有数据集、两个实验、三个起点下都不变，但它只说明相对区间宽度稳定，不能单独说明区间校准好。

## 分析口径

### 三种评估位置

| 位置 | 来源文件 | 含义 |
|:---|:---|:---|
| 全区间 (`all`) | `sliding_window_eval_metrics_group_mean_std.csv` | 从原滑窗评估的第一个窗口开始，使用全部保存窗口。 |
| 验证起点 (`val`) | `sliding_window_eval_metrics_by_split_start_group_mean_std.csv` 的 `val_start_*` 字段 | 仅使用 `t_forecast >= val_start_t` 的窗口。 |
| 测试起点 (`test`) | 同一文件的 `test_start_*` 字段 | 仅使用 `t_forecast >= test_start_t` 的窗口。 |

所有数值都是同一 variant 的 3 个 seed 的均值。两个实验的 variant 集合一致：`CB_HAB1a` 各有 5 个，其余数据集各有 8 个。

### 排序表示法

下文 `>` 表示从该指标的最佳模型到最差模型：

- `K`=`kernel`，`K256`=`kernel_256`，`K0`=`kernel_norm_0`
- `M`=`mamba`，`M0`=`mamba_norm_0`，`N`=`no_bg`
- `P`=`proportional`，`P0`=`proportional_norm_0`

`MAE`、`RMSE`、`CRPS`、`W95` 越小越好；`LP_NB` 越大越好。Coverage 按距标称 0.95 的绝对距离排序。

相关性记为 `all/val` 与 `val/test` 的 Spearman 排名相关：1 为完整排序相同，0 为无明显关系，负数为趋向相反。

### 窗口数

| 数据集 | 全区间 | 验证起点 | 测试起点 | 解释 |
|:---|---:|---:|---:|:---|
| `CB_HAB1a` | 44 | 13 | 6 | 测试排序较易受少量窗口影响。 |
| `CB_HAB4` | 19 | 5 | 4 | 验证/测试排序只能作探索性比较。 |
| `PNR_1z` | 61 | 25 | 3 | 测试排序极不稳定，应以验证排序为主。 |
| `St1-2018` | 74 | 35 | 29 | 三种位置都具有相对充分的窗口数。 |

## 排序稳定性概览

下表每个单元格是 `all/val -> val/test`。例如 `1.00 -> 0.10` 表示全区间和验证排序相同，但验证和测试排序发生了明显重排。

### `ETAS`

| 数据集 | MAE | RMSE | CRPS | LP_NB | W95 | Coverage@0.95 |
|:---|:---|:---|:---|:---|:---|:---|
| `CB_HAB1a` | 1.00 -> 0.10 | 1.00 -> 0.00 | 1.00 -> -0.10 | 0.60 -> 1.00 | 1.00 -> 1.00 | 0.68 -> 1.00 |
| `CB_HAB4` | 0.48 -> 0.86 | 0.50 -> 0.40 | 0.64 -> 0.69 | 0.71 -> 1.00 | 0.95 -> 0.98 | 0.69 -> 0.89 |
| `PNR_1z` | 0.93 -> 1.00 | 0.71 -> 0.93 | 0.90 -> 0.93 | 0.40 -> 0.90 | 1.00 -> 0.98 | 0.86 -> 0.08 |
| `St1-2018` | 0.83 -> 0.95 | 0.71 -> 0.93 | 0.86 -> 0.95 | 0.55 -> 0.95 | 0.98 -> 0.97 | 0.22 -> 1.00 |

### `RTPP v2`

| 数据集 | MAE | RMSE | CRPS | LP_NB | W95 | Coverage@0.95 |
|:---|:---|:---|:---|:---|:---|:---|
| `CB_HAB1a` | 0.90 -> 0.70 | 1.00 -> 0.40 | 1.00 -> 0.70 | 0.10 -> 1.00 | 1.00 -> 1.00 | 0.67 -> 0.92 |
| `CB_HAB4` | 0.33 -> 0.88 | 0.62 -> 0.88 | 0.26 -> 0.81 | 0.95 -> 1.00 | 1.00 -> 0.93 | 0.94 -> 0.98 |
| `PNR_1z` | 0.88 -> 0.81 | 0.76 -> 0.76 | 0.79 -> 0.76 | 0.12 -> 0.36 | 1.00 -> 1.00 | 0.23 -> 0.58 |
| `St1-2018` | 0.90 -> 1.00 | 0.64 -> 1.00 | 0.93 -> 1.00 | 0.95 -> 1.00 | 1.00 -> 0.98 | 0.90 -> 0.93 |

## 点预测全排序：`all -> val -> test`

### MAE

| 实验 | 数据集 | 全区间 | 验证起点 | 测试起点 |
|:---|:---|:---|:---|:---|
| ETAS | `CB_HAB1a` | M>P>K>K256>N | M>P>K>K256>N | M>N>K>K256>P |
| ETAS | `CB_HAB4` | M>M0>K0>K256>K>P0>P>N | M0>P>K0>M>P0>K256>K>N | M0>K0>M>P>K256>K>P0>N |
| ETAS | `PNR_1z` | M0>P>M>P0>K256>K>K0>N | P>M0>P0>M>K>K256>K0>N | P>M0>P0>M>K>K256>K0>N |
| ETAS | `St1-2018` | M0>P0>K256>K>P>K0>M>N | M0>P0>K0>K>K256>P>M>N | M0>K0>P0>K256>K>P>M>N |
| RTPP | `CB_HAB1a` | M>K256>K>P>N | M>K256>K>N>P | M>N>K256>K>P |
| RTPP | `CB_HAB4` | M0>M>K>K256>K0>P0>P>N | M0>K0>P>P0>M>K>K256>N | K0>M0>P>K>M>P0>K256>N |
| RTPP | `PNR_1z` | M>P>M0>P0>K>K256>K0>N | P>P0>M>M0>K>K256>K0>N | P0>P>K>M0>K256>M>K0>N |
| RTPP | `St1-2018` | P0>P>K256>K>M>M0>K0>N | P0>K>K256>P>M>M0>K0>N | P0>K>K256>P>M>M0>K0>N |

### RMSE

| 实验 | 数据集 | 全区间 | 验证起点 | 测试起点 |
|:---|:---|:---|:---|:---|
| ETAS | `CB_HAB1a` | M>P>K>K256>N | M>P>K>K256>N | M>N>K256>K>P |
| ETAS | `CB_HAB4` | M0>M>K0>K256>K>P0>P>N | M0>P>M>P0>K0>K256>K>N | K0>M0>M>K256>K>P>P0>N |
| ETAS | `PNR_1z` | M0>P0>M>P>K256>K>K0>N | P>M0>P0>K>K256>M>K0>N | M0>P0>P>K>K256>M>K0>N |
| ETAS | `St1-2018` | M0>P0>K256>K>M>P>K0>N | M0>P0>K0>K>K256>P>M>N | K0>M0>P0>K>K256>P>M>N |
| RTPP | `CB_HAB1a` | M>K256>K>P>N | M>K256>K>P>N | M>N>K256>K>P |
| RTPP | `CB_HAB4` | M0>M>P0>K>K0>P>K256>N | M0>K0>P>P0>M>K>K256>N | K0>M0>P>K>M>P0>K256>N |
| RTPP | `PNR_1z` | M>M0>P0>P>K256>K>K0>N | P>P0>M>M0>K>K256>K0>N | P0>P>K>K256>M0>M>K0>N |
| RTPP | `St1-2018` | P>M>P0>K256>K>M0>K0>N | P0>K256>K>P>M>M0>K0>N | P0>K256>K>P>M>M0>K0>N |

### CRPS

| 实验 | 数据集 | 全区间 | 验证起点 | 测试起点 |
|:---|:---|:---|:---|:---|
| ETAS | `CB_HAB1a` | M>K>K256>P>N | M>K>K256>P>N | N>M>K256>K>P |
| ETAS | `CB_HAB4` | M0>M>K0>K256>K>P0>P>N | M0>K0>P>M>P0>K256>K>N | K0>M0>M>K>K256>P>P0>N |
| ETAS | `PNR_1z` | M0>P>P0>M>K256>K>K0>N | P>P0>M0>M>K>K256>K0>N | P>P0>M0>K>K256>M>K0>N |
| ETAS | `St1-2018` | M0>P0>K>K256>P>K0>M>N | M0>P0>K0>K>K256>P>M>N | M0>K0>P0>K>K256>P>N>M |
| RTPP | `CB_HAB1a` | M>K256>K>N>P | M>K256>K>N>P | M>N>K256>K>P |
| RTPP | `CB_HAB4` | M0>K0>M>K>K256>P0>P>N | M0>K0>N>P>P0>M>K>K256 | K0>M0>N>K>P>M>P0>K256 |
| RTPP | `PNR_1z` | M>M0>P>P0>K256>K>K0>N | P>P0>M>M0>K>K256>K0>N | P0>P>K>K256>M0>M>K0>N |
| RTPP | `St1-2018` | P0>P>K256>K>M>M0>K0>N | P0>K256>K>P>M>M0>K0>N | P0>K256>K>P>M>M0>K0>N |

## 概率预测和区间排序

### LP_NB 全排序

| 实验 | 数据集 | 全区间 | 验证起点 | 测试起点 |
|:---|:---|:---|:---|:---|
| ETAS | `CB_HAB1a` | K>K256>M>N>P | M>K256>K>N>P | M>K256>K>N>P |
| ETAS | `CB_HAB4` | M0>K0>M>N>K256>K>P0>P | N>K0>M0>K256>K>M>P0>P | N>K0>M0>K256>K>M>P0>P |
| ETAS | `PNR_1z` | M0>K0>P0>K256>K>P>N>M | P>P0>M0>K>K256>K0>M>N | P>P0>K>K256>M0>K0>N>M |
| ETAS | `St1-2018` | M0>P0>K256>K>K0>P>N>M | P>M0>K>K256>P0>K0>N>M | M0>P>K256>K>P0>K0>N>M |
| RTPP | `CB_HAB1a` | K>K256>M>N>P | N>M>K>K256>P | N>M>K>K256>P |
| RTPP | `CB_HAB4` | M0>K0>N>K>M>K256>P0>P | K0>M0>N>K>K256>M>P0>P | K0>M0>N>K>K256>M>P0>P |
| RTPP | `PNR_1z` | M0>M>K256>K>K0>P0>P>N | P>P0>M>M0>K256>K>K0>N | P0>P>K>K256>K0>N>M0>M |
| RTPP | `St1-2018` | P0>K>K256>P>M0>K0>N>M | P0>K>K256>M0>P>K0>M>N | P0>K>K256>M0>P>K0>M>N |

LP_NB 的全区间到验证起点变化尤其大：`ETAS/RTPP` 的 `CB_HAB1a` 都从 `kernel` 转到其他模型，两个实验的 `PNR_1z` 也分别从 `mamba_norm_0` 转到 `proportional`。因此，不能把全区间的概率预测排名用于验证或测试期模型选择。

### Coverage 与 W95 的榜首变化

Coverage 的榜首表示最接近 0.95；W95 的榜首表示区间最窄。

| 实验 | 数据集 | Coverage: all -> val -> test | W95: all -> val -> test |
|:---|:---|:---|:---|
| ETAS | `CB_HAB1a` | N -> N -> N | M -> M -> M |
| ETAS | `CB_HAB4` | M0 -> N -> N | P -> P -> P |
| ETAS | `PNR_1z` | M -> P -> K | M -> M -> M |
| ETAS | `St1-2018` | K0 -> N -> N | M -> M -> M |
| RTPP | `CB_HAB1a` | N -> N -> N | M -> M -> M |
| RTPP | `CB_HAB4` | M0 -> N -> K0 | P -> P -> P |
| RTPP | `PNR_1z` | N -> P -> K256 | P -> P -> P |
| RTPP | `St1-2018` | P0 -> K -> P0 | M -> M -> M |

Coverage 的榜首对评估位置敏感，最明显的是 `PNR_1z`；W95 的榜首则完全稳定。这并不意味着最窄区间最佳，因为 Coverage 可能远离 0.95。

## 如何使用这些排序

1. 需要选部署模型时，应使用验证起点的排序，而不能使用全区间或测试起点的榜首。
2. 测试起点只用于检验验证阶段选出的同一模型，不应根据测试榜首反向换模型。
3. `CB_HAB4` 与 `PNR_1z` 的测试窗口太少。即使看到 `M0 -> K0` 或 `P -> P0` 的榜首变化，也应补充同 seed 的 paired bootstrap 置信区间后再认为存在真实差异。
4. 点预测选择看 MAE/RMSE/CRPS；概率预测看 LP_NB；区间预测必须同时看 Coverage 和 W95。
