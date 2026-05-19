# `clf_grid_tf_90_stability_grid_v2` 重跑分析（2026-05-19）

## 1. 分析范围与口径

- 实验目录：`experiments/clf_grid_tf_90_stability_grid_v2`
- 汇总文件：
  - `experiments/clf_grid_tf_90_stability_grid_v2/reports/clf_grid_metrics_summary.json`
  - `experiments/clf_grid_tf_90_stability_grid_v2/reports/clf_grid_metrics_per_run.csv`
  - `experiments/clf_grid_tf_90_stability_grid_v2/reports/clf_grid_metrics_group_mean_std.csv`
  - `experiments/clf_grid_tf_90_stability_grid_v2/reports/clf_grid_metrics_group_best.csv`
- 配置文件：`experiments/clf_grid_tf_90_stability_grid_v2/configs/clf_mf_tf_grid.yaml`
- 口径：按 `variant` 分组，主指标 `AUC`。

本次重跑统计规模为 `6 variants × 6 seeds = 36 runs`（全成功）。

---

## 2. 配置摘要（对应本次重跑）

- 固定：`Tfore=90`, `Mf=5.5`, `use_sampler=true`
- seeds：`[0,1,2,3,4,5]`
- 关键对比轴：
  - `baseline`
  - 对齐 `SCEDC_minus_b_bs64` 的 `alpha=0.5` 组（`max_grad_norm=1/2/3`）
  - `a050_lr1e3_wd1e4_wld`
  - `a075_lr1e3_wd1e3_step`

---

## 3. 总体结果

- 全局 `AUC mean = 0.5544`, `std = 0.1550`, `min = 0.0924`, `max = 0.8386`。
- `AUC > 0.5`：`23/36`；`AUC < 0.5`：`13/36`。
- 说明：整体可用性比“随机水平”好，但 seed 敏感性依然明显（下限很低）。

---

## 4. 各 variant 对比（AUC）

按 `AUC mean` 排序：

1. `tf90_a050_lr1e3_wd1e4_wld`：`0.5973 ± 0.1441`，`>0.5` 为 `4/6`，`min=0.4265`
2. `tf90_a075_lr1e3_wd1e3_step`：`0.5796 ± 0.1040`，`>0.5` 为 `5/6`，`min=0.4019`
3. `tf90_a050_align_scedcmb64_seed8cfg`（maxgrad=3）：`0.5676 ± 0.0866`，`>0.5` 为 `5/6`，`min=0.4164`
4. `tf90_a050_align_scedcmb64_seed8cfg_maxgrad2`：`0.5563 ± 0.1242`，`>0.5` 为 `4/6`，`min=0.4035`
5. `tf90_baseline`：`0.5240 ± 0.1813`，`>0.5` 为 `2/6`，`min=0.3415`
6. `tf90_a050_align_scedcmb64_seed8cfg_maxgrad1`：`0.5013 ± 0.2704`，`>0.5` 为 `3/6`，`min=0.0924`

### 关键观察

- `max_grad_norm=1`（`maxgrad1`）波动最大（`std=0.2704`），虽然出现了全局最好单次 `AUC=0.8386`，但也出现了全局最差单次 `AUC=0.0924`，不稳定。
- 对齐配置下，`max_grad_norm=3` 比 `1`/`2` 更稳（更小 std、更高 `>0.5` 覆盖率）。
- 从“均值 + 覆盖率”综合看，当前最稳妥候选是：
  - `tf90_a050_align_scedcmb64_seed8cfg`（5/6 >0.5，std 最小组之一）
  - `tf90_a075_lr1e3_wd1e3_step`（5/6 >0.5，均值也较高）

---

## 5. seed 效应（是否有“都好/都差”）

按 seed 横向看 6 个 variant 的 AUC：

- `seed=3`：全部 variant 都 `>0.5`（`6/6`），且该 seed 平均最高（`mean=0.7028`）。
- `seed=4`：`4/6` 个 variant `<=0.5`，该 seed 平均较低（`mean=0.4585`）。
- `seed=2`：`3/6` 个 variant `<=0.5`（`mean=0.4671`）。
- `seed=1`：虽然均值接近 0.5（`mean=0.4955`），但出现一次严重崩塌（`AUC=0.0924`）。

结论：存在明显 seed 效应，且不是“所有 seed 都均匀波动”，而是个别 seed（如 3）普遍偏好、个别 seed（如 4）普遍偏差。

---

## 6. 低于 0.5 的情况（13 个 run）

主要集中在：

- `tf90_a050_align_scedcmb64_seed8cfg_maxgrad1`：`seed 0/1/4`（其中 `seed1=0.0924`）
- `tf90_baseline`：`seed 0/1/2/4`
- `tf90_a050_align_scedcmb64_seed8cfg_maxgrad2`：`seed 0/4`
- `tf90_a050_align_scedcmb64_seed8cfg`：`seed2`
- `tf90_a050_lr1e3_wd1e4_wld`：`seed4/5`
- `tf90_a075_lr1e3_wd1e3_step`：`seed2`

---

## 7. 结论与下一步建议

1. 本次重跑下，`v2` 仍未达到“稳定全 seed > 0.5”（所有 variant 都不是 `6/6`）。
2. 若优先追求稳定性，建议以以下两组做下一轮基线：
   - `tf90_a050_align_scedcmb64_seed8cfg`（`max_grad_norm=3`）
   - `tf90_a075_lr1e3_wd1e3_step`
3. 不建议继续用 `max_grad_norm=1` 作为主线（波动极大，存在崩塌）。
4. 下一轮优化建议以“抬高最差 seed”为第一目标：
   - 固定当前较稳配置，增大 seed 数（例如 10 seeds）
   - 围绕 `max_grad_norm ∈ {2.5, 3.0, 3.5}` 做微调
   - 以 `auc_min` 和 `AUC>0.5` 覆盖率作为主优化目标，`auc_mean` 作为次目标
