# `reg_grid_r04_stable8` 实验分析（2026-05-14）

## 1. 分析范围与口径

- 实验目录：`experiments/reg_grid_r04_stable8`
- 汇总文件：
  - `experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_summary.json`
  - `experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_group_mean_std.csv`
  - `experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_group_best.csv`
  - `experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_per_run.csv`
- 评估口径：
  - 主指标：`RMSE`（越小越好）
  - 当前汇总使用：`ckpt_select = last`

---

## 2. 整体结论

- 本轮共 `24` 个 run（8 个变体 × 3 seeds），训练/测试均成功，且无缺失指标。
- 所有组里，按**均值 RMSE** 最优的是 `r04s_g7_drop15`（`RMSE_mean=0.738823`）。
- 次优是 `r04s_g1_lr8e4`（`RMSE_mean=0.745076`），两者显著优于基线 `r04s_g0_base`（`RMSE_mean=0.836694`）。
- 相对基线 `r04s_g0_base`：
  - `r04s_g7_drop15` 的 RMSE 改善约 `11.70%`
  - `r04s_g1_lr8e4` 的 RMSE 改善约 `10.95%`

---

## 3. 关键排名与指标

### 3.1 按 RMSE_mean 排名（低→高）

1. `r04s_g7_drop15`：`0.738823`
2. `r04s_g1_lr8e4`：`0.745076`
3. `r04s_g4_wd15e4`：`0.793982`
4. `r04s_g6_warm12`：`0.794532`
5. `r04s_g5_warm08`：`0.810553`
6. `r04s_g0_base`：`0.836694`
7. `r04s_g2_lr12e4`：`0.838516`
8. `r04s_g3_wd8e4`：`0.927743`

### 3.2 其他主指标上的最优组

- `MAE_mean` 最优：`r04s_g7_drop15`（`0.615258`）
- `MAPE_mean` 最优：`r04s_g7_drop15`（`12.995192`）
- `R2_mean` 最优（数值最大）：`r04s_g7_drop15`（`-0.636709`）
- `PearsonR_mean` 最优：`r04s_g1_lr8e4`（`0.299993`）
- `SpearmanR_mean` 最优：`r04s_g1_lr8e4`（`0.315496`）
- `DTW_normalized_mean` 最优（最小）：`r04s_g1_lr8e4`（`0.341330`）

说明：全部组的 `R2_mean` 仍为负，当前设定下回归解释力仍偏弱。

---

## 4. 稳定性分析（跨 seed）

- `r04s_g1_lr8e4` 的 RMSE 标准差最小（`RMSE_std=0.018201`），是本轮**最稳**配置。
- `r04s_g7_drop15` 的均值最好，但波动更大（`RMSE_std=0.052526`），属于“性能优先、稳定性次之”。
- 基线 `r04s_g0_base` 波动较大（`RMSE_std=0.136434`），稳定性明显弱于 `g1/g7`。

稳健性（`mean+std`）排序前三：
1. `r04s_g1_lr8e4`
2. `r04s_g7_drop15`
3. `r04s_g6_warm12`

---

## 5. 最好单次 run 与差单次 run

- 全部 run 中单次 RMSE 最好：
  - `r04s_g7_drop15_seed_1`，`RMSE=0.683374`
- 各组“组内最佳”中也由 `r04s_g7_drop15` 夺冠：
  - `best_run = r04s_g7_drop15_seed_1`

较差尾部 run 主要集中在：
- `r04s_g3_wd8e4_seed_2`（`RMSE=0.987866`）
- `r04s_g0_base_seed_2`（`RMSE=0.986166`）
- `r04s_g2_lr12e4_seed_2`（`RMSE=0.969915`）

---

## 6. 参数影响解读（本轮网格）

本轮改动点来自：`learning_rate / weight_decay / warmup_ratio / mlp_dropout`。

- **降低学习率到 `8e-4`（g1）**：
  - 显著提升稳定性（RMSE std 最小）
  - 平均误差也显著优于基线
- **提升学习率到 `1.2e-3`（g2）**：
  - 相对基线无收益（略差）
- **weight_decay 降到 `8e-4`（g3）**：
  - 明显退化，为本轮最差组
- **weight_decay 升到 `1.5e-3`（g4）**：
  - 平均表现优于基线，但 seed 波动偏大
- **warmup_ratio 调到 `0.08/0.12`（g5/g6）**：
  - 均好于基线，但不及 `g1/g7`
- **mlp_dropout 从 `0.2` 降到 `0.15`（g7）**：
  - 本轮平均指标最优，收益最明显

---

## 7. 关于 checkpoint 的额外观察

- 实验配置中 `ckpt_select: both`，每个 run 都有：
  - `metrics_test_best_1.json`
  - `metrics_test_last_1.json`
- 但汇总报告当前按 `last` 聚合。
- 对比同一 run 的 `best vs last` 后发现：
  - `24` 个 run 中仅 `3` 个是 `best` 优于 `last`
  - 其余 `21` 个 run 为 `best` 更差

这提示：现有“best checkpoint”选取依据与最终测试目标可能不一致，或验证阶段噪声较大。

---

## 8. 推荐决策

### 8.1 如果追求榜单最优（最低误差）

- 首选：`r04s_g7_drop15`
- 原因：`RMSE/MAE/MAPE/R2_mean` 综合最优

### 8.2 如果追求上线稳健（低方差）

- 首选：`r04s_g1_lr8e4`
- 原因：RMSE 方差最小，跨 seed 一致性最好

---

## 9. 下一轮建议（可直接落地）

1. 双线并行细扫：
   - **性能线（g7）**：`mlp_dropout ∈ {0.12, 0.15, 0.18}`
   - **稳健线（g1）**：`learning_rate ∈ [7e-4, 9e-4]`（小步长）
2. 将 seeds 从 `3` 增加到 `5` 或 `10`，降低偶然性。
3. 复核 `best checkpoint` 的选模指标与测试目标是否一致；必要时加入更稳健的早停/平滑策略。

---

## 10. 关键文件定位

- 运行规模与汇总口径：`experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_summary.json`
- 组均值/方差：`experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_group_mean_std.csv`
- 组内最佳 run：`experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_group_best.csv`
- 每个 run 明细：`experiments/reg_grid_r04_stable8/reports/reg_grid_metrics_per_run.csv`
- 实验网格配置：`experiments/reg_grid_r04_stable8/configs/reg_mixer_attnpl_grid.yaml`
