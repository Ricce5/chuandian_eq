# `clf_grid_tf_90_stability_grid` 分析（2026-05-19）

## 1. 分析范围与口径

- 实验目录：`experiments/clf_grid_tf_90_stability_grid`
- 结果文件：
  - `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_per_run.csv`
  - `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_group_mean_std.csv`
  - `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_group_best.csv`
- 任务设置：固定 `Tfore=90, Mf=5.5`，按 `variant` 分组汇总
- 统计规模：`6 variants × 10 seeds = 60 runs`
- 关注指标：`AUC`（主）、`PR-AUC`（辅）

> 目标是验证：`Tfore=90` 下能否做到 AUC 稳定大于 0.5（尽量所有 seed 都 > 0.5）。

---

## 2. 实验配置摘要

来自 `plan.json` 与 `configs/clf_mf_tf_grid.yaml`：

- 公共设置：
  - `Tfore=90`
  - `Mf=5.5`
  - `use_sampler=true`
  - `batch_size=64`
  - `max_grad_norm=1.0`
  - seeds: `[0..9]`

- 6 个 `variant`：
  1. `tf90_a050_lr1e3_wd1e4_wld`
  2. `tf90_a050_lr6e4_wd1e3_step`
  3. `tf90_a050_lr6e4_wd1e3_step_noresume`
  4. `tf90_a075_lr6e4_wd1e3_step`
  5. `tf90_a090_lr6e4_wd1e3_step`
  6. `tf90_a075_lr1e3_wd1e3_step`

---

## 3. 主结果（按 variant 汇总）

按 `AUC_mean` 从高到低：

1. `tf90_a050_lr1e3_wd1e4_wld`：`0.6289 ± 0.1350`
2. `tf90_a075_lr1e3_wd1e3_step`：`0.6287 ± 0.1690`
3. `tf90_a050_lr6e4_wd1e3_step_noresume`：`0.6226 ± 0.1715`
4. `tf90_a090_lr6e4_wd1e3_step`：`0.5600 ± 0.1206`
5. `tf90_a075_lr6e4_wd1e3_step`：`0.5412 ± 0.1123`
6. `tf90_a050_lr6e4_wd1e3_step`：`0.5041 ± 0.1708`

从均值看，前三名明显优于后三名；但各组标准差仍较大，seed 波动明显存在。

---

## 4. “AUC 是否稳定 > 0.5”检查

按每个 `variant` 统计 10 个 seed 中 `AUC > 0.5` 的个数：

- `tf90_a050_lr1e3_wd1e4_wld`：`9/10`（`auc_min=0.4550`）
- `tf90_a075_lr1e3_wd1e3_step`：`8/10`（`auc_min=0.4784`）
- `tf90_a050_lr6e4_wd1e3_step_noresume`：`7/10`（`auc_min=0.3520`）
- `tf90_a090_lr6e4_wd1e3_step`：`7/10`（`auc_min=0.3587`）
- `tf90_a075_lr6e4_wd1e3_step`：`6/10`（`auc_min=0.3828`）
- `tf90_a050_lr6e4_wd1e3_step`：`5/10`（`auc_min=0.1938`）

结论：本轮没有任何一个 `variant` 达到 `10/10 seed` 全部 `AUC > 0.5`。

---

## 5. 关键对比观察

### 5.1 学习率与调度

- `lr=1e-3` 的两组（`a050_lr1e3_wd1e4_wld`, `a075_lr1e3_wd1e3_step`）整体优于 `lr=6e-4` 的多数组。
- 仅“降学习率 + step_warmup”并未自动带来更稳定的 AUC>0.5。

### 5.2 `criterion_alpha` 影响

在 `lr=6e-4, wd=1e-3, step_warmup` 这条线上：

- `alpha=0.5`：`auc_mean=0.5041`
- `alpha=0.75`：`auc_mean=0.5412`
- `alpha=0.9`：`auc_mean=0.5600`

趋势上 `alpha` 增大带来均值提升，但最差 seed 仍低于 0.5，稳定性问题未根治。

### 5.3 预训练加载 (`noresume`)

- `a050_lr6e4_wd1e3_step_noresume` 相比同超参的 resume 版本均值更高（`0.6226` vs `0.5041`）。
- 说明 `tf90` 条件下，去掉该预训练初始化可能有帮助，但仍存在低 seed 崩塌。

---

## 6. 单次最好/最差 run（按 AUC）

- 全局最好单次：
  - `tf90_a075_lr1e3_wd1e3_step_seed_1`
  - `AUC=0.9524`

- 全局最差单次：
  - `tf90_a050_lr6e4_wd1e3_step_seed_1`
  - `AUC=0.1938`

说明当前配置对 seed 仍然高度敏感：同一网格内可从强可分到接近反向排序。

---

## 7. 结论与建议

1. **当前最优均值候选**：`tf90_a050_lr1e3_wd1e4_wld`（`9/10` seed >0.5，均值最高）。
2. **次优候选**：`tf90_a075_lr1e3_wd1e3_step`（单次峰值高，但波动更大）。
3. 在 `tf90` 上，`lr=6e-4` 路线并未体现出预期稳定优势；若继续沿此线优化，需联动更多稳态手段。 
4. 下一轮建议以 `tf90_a050_lr1e3_wd1e4_wld` 为基线，聚焦“最差 seed 过线”优化：
   - 小范围搜索 `criterion_alpha ∈ {0.5, 0.6, 0.7}`
   - 小范围搜索 `max_grad_norm ∈ {0.7, 1.0}`
   - 以 `auc_min` 为第一目标，`auc_mean` 为第二目标

---

## 8. 关键文件

- `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_summary.json`
- `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_tf_90_stability_grid/reports/clf_grid_metrics_group_best.csv`
- `experiments/clf_grid_tf_90_stability_grid/configs/clf_mf_tf_grid.yaml`
- `experiments/clf_grid_tf_90_stability_grid/plan.json`
