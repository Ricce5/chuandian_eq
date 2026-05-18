# reg_mixer_layer_1_gird vs reg_mixer_attn_1_grid 指标与 DTW 分析

- 分析日期：2026-05-18
- 对比实验：
  - `experiments/reg_mixer_layer_1_gird`
  - `experiments/reg_mixer_attn_1_grid`
- 汇总来源：
  - `reports/reg_grid_metrics_summary.json`
  - `reports/reg_grid_metrics_group_mean_std.csv`
  - `reports/reg_grid_metrics_group_best.csv`
  - `reports/reg_grid_metrics_per_run.csv`

## 1) 统计口径与可比性

- 两个实验 `ckpt_select` 均为 `last`，可直接按同口径对比。
- `reg_mixer_layer_1_gird`：`n_runs=24`，其中有效 `22/24`（2 个 train_failed）。
- `reg_mixer_attn_1_grid`：`n_runs=36`，有效 `36/36`。

## 2) 核心指标对比（RMSE/MAE/R2）

### 2.1 各实验 Top（按 RMSE_mean 升序）

**A. `reg_mixer_layer_1_gird`**

1. `regmx_swu_core`：RMSE `0.7712 ± 0.2509`，MAE `0.6371`，R2 `-0.9186`
2. `regmx_lr9e4_enc3p6e4`：RMSE `0.7759 ± 0.1983`，MAE `0.6312`，R2 `-0.8870`
3. `regmx_lr1e3_enc4e4`：RMSE `0.7823 ± 0.2887`，MAE `0.6853`，R2 `-1.0157`

**B. `reg_mixer_attn_1_grid`**

1. `regmx_swu_beta03_drop020`：RMSE `0.7734 ± 0.0825`，MAE `0.6254`，R2 `-0.8029`
2. `regmx_swu_warm008`：RMSE `0.7942 ± 0.0630`，MAE `0.6674`，R2 `-0.8939`
3. `regmx_align_ckpt_20260515_180423`：RMSE `0.7991 ± 0.0563`，MAE `0.6868`，R2 `-0.9156`

### 2.2 跨实验结论（看均值 + 稳定性）

- 两边最优均值几乎持平：
  - `layer_1_gird` 最优 RMSE_mean `0.7712`
  - `attn_1_grid` 最优 RMSE_mean `0.7734`
  - 差值约 `+0.0022`（attn 略高）
- 但稳定性差异显著：
  - `layer_1_gird` 最优项 std `0.2509`
  - `attn_1_grid` 最优项 std `0.0825`
- run 级离散度也同方向：
  - `layer_1_gird` 有效 run 的 RMSE std 约 `0.2331`
  - `attn_1_grid` 有效 run 的 RMSE std 约 `0.0801`

### 2.3 单次最优（best single run）

- `layer_1_gird` 单次最优更低：`0.55197`（`regmx_swu_core_seed_0`）
- `attn_1_grid` 单次最优：`0.67323`（`regmx_swu_lr9e4_enc3p6e4_seed_3`）
- 说明：`layer_1_gird` 有更强“峰值”，但整体复现性不如 `attn_1_grid`。

## 3) DTW 专项分析

## 3.1 各实验 DTW 最优项（按 DTW_mean 升序）

**A. `reg_mixer_layer_1_gird`（Top-3 DTW）**

1. `regmx_lr1e3_enc4e4`：DTW `74.490 ± 36.641`，DTW_norm `0.407047`
2. `regmx_lr9e4_enc3p6e4`：DTW `77.141 ± 16.364`，DTW_norm `0.421537`
3. `regmx_swu_core`：DTW `81.430 ± 26.170`，DTW_norm `0.444970`

baseline：DTW `94.831`，DTW_norm `0.518203`  
最佳 DTW 相对 baseline 改善约 `-21.45%`。

**B. `reg_mixer_attn_1_grid`（Top-3 DTW）**

1. `regmx_align_ckpt_20260515_180423`：DTW `65.920 ± 4.393`，DTW_norm `0.360218`
2. `regmx_swu_lr9e4_enc3p6e4`：DTW `71.097 ± 18.158`，DTW_norm `0.388510`
3. `regmx_swu_lr8e4_enc3p2e4`：DTW `71.634 ± 18.691`，DTW_norm `0.391442`

baseline：DTW `87.865`，DTW_norm `0.480135`  
最佳 DTW 相对 baseline 改善约 `-24.98%`。

### 3.2 DTW 跨实验结论

- 跨实验 DTW 最优对比：
  - `layer_1_gird` 最优 DTW：`74.490`
  - `attn_1_grid` 最优 DTW：`65.920`
  - `attn_1_grid` 额外下降约 `8.57`
- 标准化 DTW 也一致更优：
  - `layer_1_gird` 最优 DTW_norm：`0.407047`
  - `attn_1_grid` 最优 DTW_norm：`0.360218`
- 稳定性上，`attn_1_grid` 的 DTW 更稳（例如 `align_ckpt_20260515_180423` 的 DTW_CV 约 `0.0666`）。

### 3.3 RMSE 与 DTW 的关系

- `layer_1_gird`：variant 级 `corr(RMSE_mean, DTW_mean) ≈ 0.867`，两者强正相关。
- `attn_1_grid`：variant 级 `corr(RMSE_mean, DTW_mean) ≈ -0.061`，几乎解耦。
- 含义：
  - `layer_1_gird` 更像“误差降则 DTW 同步降”。
  - `attn_1_grid` 可以更独立地优化时序形状（DTW），不必完全跟随 RMSE 排序。

## 4) 风险点与解释

- `layer_1_gird` 中 `regmx_lr1.5e3_enc6e4` 有 `2/4` 训练失败，导致其统计可靠性偏低。
- 失败条目位于：
  - `regmx_lr1.5e3_enc6e4_seed_0`
  - `regmx_lr1.5e3_enc6e4_seed_3`
- 因此 `layer_1_gird` 的部分结论受可用样本数量影响。

## 5) 总结与下一步建议

1. 若目标是**稳健可复现**，优先选择 `reg_mixer_attn_1_grid` 路线。  
2. 若目标是**冲击单次最低 RMSE**，`reg_mixer_layer_1_gird` 仍有更低峰值潜力。  
3. 若目标是**时序形态一致性（DTW 优先）**，优先跟进 `regmx_align_ckpt_20260515_180423`。  
4. 建议下一轮将“误差目标”和“形态目标”分线调参，分别汇报 RMSE 排名与 DTW 排名，避免单指标决策偏差。  

