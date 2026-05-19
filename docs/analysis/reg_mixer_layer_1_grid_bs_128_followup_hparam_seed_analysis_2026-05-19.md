# reg_mixer_layer_1_grid_bs_128_followup 结果分析（超参数关系 + seed 效应）

- 分析日期：2026-05-19
- 实验目录：`experiments/reg_mixer_layer_1_grid_bs_128_followup`
- 汇总文件：`experiments/reg_mixer_layer_1_grid_bs_128_followup/reports/reg_grid_metrics_summary.json`

## 1) 统计口径

- `ckpt_select = "last"`
- `best_metric = "RMSE"`，`best_mode = "min"`
- 总 run 数：`54`（`6 variants × 9 seeds`）
- 成功数：`54/54`

数据来源：
- `experiments/reg_mixer_layer_1_grid_bs_128_followup/reports/reg_grid_metrics_per_run.csv`
- `experiments/reg_mixer_layer_1_grid_bs_128_followup/reports/reg_grid_metrics_group_mean_std.csv`
- `experiments/reg_mixer_layer_1_grid_bs_128_followup/reports/reg_grid_metrics_group_best.csv`
- `experiments/reg_mixer_layer_1_grid_bs_128_followup/configs/reg_mixer_attnpl_grid.yaml`

> 说明：RMSE / MAE / MSE / MAPE / DTW / DTW_normalized 越低越好；R2 / PearsonR / SpearmanR 越高越好。

## 2) 本轮超参数对照关系

固定项：`batch_size=128`，`learning_rate=1e-3`，`accumulation_steps=1`。

变化项（`variant`）：
- `regmxf_base`：`use_ema=false`，`encoder_learning_rate=4e-4`
- `regmxf_ema_d092`：`use_ema=true`，`ema_decay=0.92`，`encoder_learning_rate=4e-4`
- `regmxf_ema_d095`：`use_ema=true`，`ema_decay=0.95`，`encoder_learning_rate=4e-4`
- `regmxf_ema_d098`：`use_ema=true`，`ema_decay=0.98`，`encoder_learning_rate=4e-4`
- `regmxf_enc3e4`：`use_ema=false`，`encoder_learning_rate=3e-4`
- `regmxf_lrenc6e4`：`use_ema=false`，`encoder_learning_rate=6e-4`

## 3) 超参数与结果关系（跨 seed 均值）

按 `RMSE_mean` 排序（越低越好）：

1. `regmxf_ema_d098`：`0.7564 ± 0.1587`
2. `regmxf_ema_d092`：`0.7598 ± 0.1587`
3. `regmxf_base`：`0.7602 ± 0.1580`
4. `regmxf_ema_d095`：`0.7611 ± 0.1597`
5. `regmxf_lrenc6e4`：`0.7927 ± 0.1043`
6. `regmxf_enc3e4`：`0.8163 ± 0.1543`

结论：
- EMA 三个 decay 与 base 的差异非常小，`ema_d098` 略优，但提升幅度很有限。
- `encoder_learning_rate=3e-4`（`enc3e4`）明显退化。
- `encoder_learning_rate=6e-4`（`lrenc6e4`）也退化，但波动相对更小（std 更小）。

## 4) 是否存在“某些 seed 都好 / 某些 seed 都差”

结论：**存在，且非常明显**。

### 4.1 按 seed 的整体均值（跨所有 variant）

`RMSE` 从好到差：
- `seed 0 (0.5859)`
- `seed 8 (0.6610)`
- `seed 3 (0.6636)`
- `seed 7 (0.7067)`
- `seed 2 (0.7559)`
- `seed 4 (0.7843)`
- `seed 1 (0.8509)`
- `seed 5 (0.9666)`
- `seed 6 (0.9950)`

### 4.2 相对各 variant 均值的“好/差计数”

以“该 seed 在该 variant 上是否优于该 variant 的均值 RMSE”计数：
- `seed 0`: `better=6, worse=0`
- `seed 3`: `better=6, worse=0`
- `seed 8`: `better=6, worse=0`
- `seed 1`: `better=0, worse=6`
- `seed 5`: `better=0, worse=6`
- `seed 6`: `better=0, worse=6`

可见：
- **稳定偏好 seed**：`0 / 3 / 8`
- **稳定偏差 seed**：`1 / 5 / 6`

### 4.3 规模对比：seed 影响 vs 超参影响

以 `RMSE` 为例：
- seed 均值范围：`0.5859 ~ 0.9950`，跨度约 `0.4091`
- variant 均值范围：`0.7564 ~ 0.8163`，跨度约 `0.0599`

粗略方差分解（双因子无重复）显示：
- `seed` 贡献约 `84.83%`
- `variant` 贡献约 `2.44%`
- 交互项约 `12.73%`

说明本轮里 **seed 是主导因子**，超参数主效应较弱。

## 5) 补充观察

- `base / ema_d092 / ema_d095 / ema_d098` 四个配置的 seed 排名次序几乎完全一致（RMSE 排名一致）。
- `lrenc6e4` 是唯一一个 RMSE 最优 seed 不在 `seed 0` 的配置（其最优是 `seed 8`），显示该超参和 seed 有更强交互。
- 跨 9 个核心指标（RMSE/MAE/MSE/MAPE/R2/PearsonR/SpearmanR/DTW/DTW_normalized）平均排名看：
  - 最好：`seed 0`
  - 最差：`seed 6`

## 6) 最终结论与建议

1. 本实验确实有明显“seed 都好 / seed 都差”现象，不能用单 seed 结论判断超参数优劣。
2. 若按当前数据选型，`ema_d098` 可作为候选第一，但其领先幅度很小，应视为“微弱优势”。
3. 下一轮建议采用固定多 seed 口径（至少 `3~5` 个，建议含 `0/3/8` 与 `1/5/6` 的代表组合），以均值±方差做决策。
4. 若目标是降低 seed 敏感性，应优先继续做稳定性优化（正则、训练时长、学习率策略）而不只是微调 EMA decay。
