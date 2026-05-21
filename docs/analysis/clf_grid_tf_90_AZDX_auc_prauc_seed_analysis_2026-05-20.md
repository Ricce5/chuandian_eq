# `clf_grid_tf_90_AZDX` 分析（2026-05-20）

## 1. 分析范围

- 实验目录：`experiments/clf_grid_tf_90_AZDX`
- 结果文件：
  - `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_per_run.csv`
  - `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_group_mean_std.csv`
  - `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_group_best.csv`
- 当前规模：`15 groups × 10 seeds = 150 runs`
- 关注指标：`AUC`（主）、`PR-AUC`（辅）

---

## 2. 总体结论

- `AUC < 0.5` 的 run 共 **70/150**，占比 **46.7%**
- 组均值 `auc_mean < 0.5` 的有 **5 组**
- 低 AUC 主要集中在：
  - `tf90_a050_lr1e3`
  - `tf90_a050_lr6e4`
  - `tf90_a090_maxgrad1`
  - `tf90_ema98`
  - `tf90_stable`
- 这些低 AUC 组的 `PR-AUC` 也普遍偏低，基本在 `0.15 ~ 0.18`

---

## 3. 各组 `AUC < 0.5` 数量

按组统计如下：

| 组 | `AUC<0.5` | `auc_mean` | `pr_auc_mean` |
|---|---:|---:|---:|
| `tf90_a050_lr6e4` | 8/10 | 0.380 | 0.151 |
| `tf90_a050_lr1e3` | 6/10 | 0.419 | 0.166 |
| `tf90_a090_maxgrad1` | 6/10 | 0.475 | 0.158 |
| `tf90_stable` | 6/10 | 0.489 | 0.180 |
| `tf90_a090_b128_maxgrad1` | 5/10 | 0.502 | 0.174 |
| `tf90_b32` | 5/10 | 0.501 | 0.185 |
| `tf90_ema98` | 5/10 | 0.475 | 0.157 |
| `tf90_max_grad_norm_3` | 5/10 | 0.528 | 0.180 |
| `tf90_a090_b128` | 4/10 | 0.554 | 0.191 |
| `tf90_b128` | 4/10 | 0.546 | 0.247 |
| `tf90_max_grad_norm_1` | 4/10 | 0.551 | 0.188 |
| `tf90_max_grad_norm_2` | 4/10 | 0.510 | 0.192 |
| `tf90_a075` | 3/10 | 0.537 | 0.203 |
| `tf90_a090` | 3/10 | 0.563 | 0.195 |
| `tf90_a090_b128_maxgrad4` | 2/10 | 0.534 | 0.164 |

> 说明：`tf90_a090_b128_maxgrad1` 和 `tf90_b32` 的 `auc_mean` 都在 0.50 左右，但仍有一半或接近一半的 seed 低于 0.5，稳定性一般。

---

## 4. 各个 seed 的 `AUC` 均值排序

按 `seed` 的跨组平均 `AUC` 从高到低：

1. `seed=2`：`0.571280`
2. `seed=9`：`0.548720`
3. `seed=8`：`0.533827`
4. `seed=3`：`0.531228`
5. `seed=0`：`0.512177`

6. `seed=1`：`0.501884`
7. `seed=6`：`0.476595`
8. `seed=4`：`0.466875`
9. `seed=7`：`0.455595`
10. `seed=5`：`0.445094`

结论：`seed=2` 最稳，`seed=5` 最弱；整体上 seed 间仍有明显差异。

---

## 5. 结论

1. 这批 `tf90` 网格里，仍有近一半的 run `AUC < 0.5`，说明稳定性问题还在。
2. 最需要优先关注的是 `tf90_a050_lr6e4`、`tf90_a050_lr1e3`、`tf90_a090_maxgrad1`、`tf90_stable`。
3. 若下一轮继续扩展，建议优先围绕 `seed=2/9/8/3` 这类表现更好的随机种子做复核。

---

## 6. 关键文件

- `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_tf_90_AZDX/reports/clf_grid_metrics_group_best.csv`
