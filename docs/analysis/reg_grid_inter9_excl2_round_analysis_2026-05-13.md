# `reg_grid_inter9_excl2_round` 实验分析（2026-05-13）

## 1. 分析范围与口径

- 实验目录：`experiments/reg_grid_inter9_excl2_round`
- 汇总文件：
  - `experiments/reg_grid_inter9_excl2_round/reports/reg_grid_metrics_summary.json`
  - `experiments/reg_grid_inter9_excl2_round/reports/reg_grid_metrics_group_mean_std.csv`
  - `experiments/reg_grid_inter9_excl2_round/reports/reg_grid_metrics_group_best.csv`
  - `experiments/reg_grid_inter9_excl2_round/reports/reg_grid_metrics_per_run.csv`
- 评估口径：
  - 主指标：`RMSE`（越小越好）
  - 汇总使用：`ckpt_select = last`

---

## 2. 整体结果结论

- 本轮共 `18` 个 run，成功产出指标 `17` 个。
- 单次最佳 run：`inter9_r04_base_t16_wld_seed_0`，`RMSE = 0.5479503480673907`。
- 按变体均值（`RMSE_mean`）排名：
  1. `inter9_r04_base_t16_wld`：`0.65397`
  2. `inter9_r05_base_t07_step`：`0.70433`
  3. `inter9_r07_base_t09_wld`：`0.75365`
  4. `inter9_r08_lowlr_t13_step`：`0.81754`
  5. `inter9_r03_lowlr_t20_step`：`0.89388`
  6. `inter9_r09_lowlr_t19_step`：`0.91064`
- 第一名相对第二名 RMSE 再下降约 `7.70%`。

---

## 3. 好组 vs 差组（Top3 vs Bottom3）

分组定义（按 `RMSE_mean`）：

- 好组 Top3：`r04 / r05 / r07`
- 差组 Bottom3：`r08 / r03 / r09`

均值对比：

- `RMSE_mean`：`0.7040`（好） vs `0.8740`（差） → 差组高约 `24.15%`
- `MAE_mean`：`0.5849`（好） vs `0.7438`（差） → 差组高约 `27.17%`
- `DTW_mean`：`69.52`（好） vs `95.34`（差） → 差组高约 `37.13%`
- `R2_mean`：`-0.5079`（好） vs `-1.3506`（差）

---

## 4. 参数共性总结

### 4.1 效果好的参数设置共性

1. **profile 共性最强**：好组全部来自 `base`，差组全部来自 `low_lr`。
2. **学习率整体更高**：好组主学习率均值约 `0.00103`，差组约 `0.00073`。
3. **调度参数更温和（需区分调度器类型）**：在同类型调度器内，表现更好的组合更接近
   - `warmup_ratio = 0.1`
   - `step_lr_step_size_ratio = 0.15`
   - `step_lr_gamma <= 0.5`
4. **dropout 更低**：`mlp_dropout = 0.2` 组整体优于 `0.3` 组。
5. **weight_decay 更居中**：`0.001 ~ 0.002` 区间优于极端值（`0.0005` 或 `0.003`）。

### 4.2 效果差的参数设置共性

1. 全部属于 `low_lr` profile。
2. 主学习率集中在 `0.0006 ~ 0.0008`。
3. 调度更激进（例如 `step_lr_step_size_ratio = 0.2/0.25`，`step_lr_gamma = 0.5/0.6`）。
4. 更常出现 `mlp_dropout = 0.3`。
5. `weight_decay` 偏极端（过低或过高）。

---

## 5. 可能机制解释（为什么有差异）

1. **欠拟合倾向**：低学习率 + 较高 dropout + 较激进后期衰减，会让有效更新变小，导致 RMSE/DTW 偏大、R2 更差。
2. **正则强度的 U 型效应**：`weight_decay` 过小与过大都不利，居中区间更稳。
3. **优化超参主导本轮差异**：本轮结构参数（如 `attn_layer_idx`、`load_strategy`）基本固定，差异更多来自训练超参组合。

---

## 5.1 按调度器类型拆分结论（修正）

上一版“调度更温和”是跨调度器观察，这里补充类型内分析：

1. `warmup_linear_decay`（2 个变体）
   - `inter9_r04_base_t16_wld`（`warmup_ratio=0.1`）优于 `inter9_r07_base_t09_wld`（`warmup_ratio=0.15`）。
   - 该类型下，`step_lr_step_size_ratio`/`step_lr_gamma` **不参与公式**，即使配置里有也不会影响 LR 轨迹。
2. `step_warmup`（4 个变体）
   - 最好的是 `inter9_r05_base_t07_step`（`warmup_ratio=0.1`, `step_ratio=0.15`, `gamma=0.4`）。
   - 中间是 `r03/r08`（`0.15/0.2/0.5`），最差是 `r09`（`0.05/0.25/0.6`）。
   - 在该类型内，`step_ratio` 更大、`gamma` 更高通常对应更差 RMSE。

结论更新：应表述为 **“在各自调度器内部，较温和参数更优；跨调度器比较需谨慎”**。

---

## 5.2 调度器参数是否生效（代码核验）

根据实现与调用链，结论如下：

1. 调度步进是按 **optimizer step** 触发（非按 epoch）：
   - 回归训练中每次 `optimizer.step()` 后都会 `step_scheduler(..., event='step')`。
2. `step_warmup` 生效参数：
   - `warmup_ratio`（决定 warmup steps）
   - `step_lr_step_size_ratio`（转成 `StepLR.step_size`）
   - `step_lr_gamma`
   - `scheduler_min_lr` **不生效**（该分支未使用）
3. `warmup_linear_decay` 生效参数：
   - `warmup_ratio`、`scheduler_min_lr`
   - `step_lr_step_size_ratio`/`step_lr_gamma` **不生效**
4. 本实验配置中这两类参数确实被传入并参与构造（见各 run 的 `config_input.yaml`）。

因此，本实验里“调度参数”不是全部都有效，必须按 `scheduler_type` 解读。

---

## 5.3 调度器参数生效矩阵（本实验）

说明：下表面向本实验实际用到的调度器（`step_warmup`、`warmup_linear_decay`）。

| 参数 | `step_warmup` | `warmup_linear_decay` | 备注 |
|---|---|---|---|
| `warmup_ratio` | 生效 | 生效 | 统一先换算 `warmup_steps = int(warmup_ratio * total_steps)` |
| `step_lr_step_size_ratio` | 生效 | 不生效 | 仅 `step_warmup` 用于 `StepLR.step_size` |
| `step_lr_gamma` | 生效 | 不生效 | 仅 `step_warmup` 用于 `StepLR.gamma` |
| `scheduler_min_lr` | 不生效 | 生效 | `step_warmup` 分支未使用该参数 |
| `learning_rate` | 生效（作为各 param group 初值） | 生效（作为非 encoder 组基准） | 两者都通过 optimizer 初始 LR 影响轨迹 |
| `encoder_learning_rate` | 生效 | 生效 | 若匹配到 encoder 参数组，会形成独立 LR 轨迹 |
| `accumulation_steps` | 间接生效 | 间接生效 | 影响 `total_steps` 计算（从而影响 warmup/step 边界） |
| `epochs` | 间接生效 | 间接生效 | 同上，参与 `total_steps` 计算 |

补充：

- `warmup_linear_decay` 的实现按各 param group 的 `base_lrs` 线性 warmup + 线性 decay 到各自 `min_lrs`，不会用到 `step_lr_*` 参数。
- `step_warmup` 是 `LinearLR(start_factor=0.01)` + `StepLR` 的 `SequentialLR` 组合。

---

## 6. 关于“好组学习率是不是更高？”

结论：**整体是更高的**，但不是“越高越好”。

- 好组主学习率：`[0.001, 0.0006, 0.0015]`，均值约 `0.00103`
- 差组主学习率：`[0.0008, 0.0006, 0.0008]`，均值约 `0.00073`
- 具体看分组效果：`lr=0.001` 最好；`lr=0.0015` 不如 `0.001`，说明存在“最优区间”。

---

## 7. “好组里 learning_rate=0.0006 对应哪个变体？”

- 对应变体：`inter9_r05_base_t07_step`
- 其配置（seed_0）：
  - `learning_rate: 0.0006`
  - `batch_size: 128`
  - `scheduler_type: step_warmup`
  - `warmup_ratio: 0.1`
  - `step_lr_step_size_ratio: 0.15`
  - `step_lr_gamma: 0.4`
  - `mlp_dropout: 0.2`

---

## 8. 异常说明

- `inter9_r03_lowlr_t20_step_seed_1` 在测试阶段失败（`test_returncode=1`）。
- 直接原因是测试时缺失 `config_input.yaml`（`FileNotFoundError`）。
- 这会让对应变体只统计到 2 个有效 seed，稳定性判断需谨慎。

---

## 9. 下一轮建议（简版）

1. 先以 `base` profile 为主线继续细化。
2. 将学习率聚焦在 `~1e-3` 附近（例如 `8e-4 ~ 1.2e-3`）。
3. 优先保留：
   - `warmup_ratio ≈ 0.1`
   - `step_lr_step_size_ratio ≈ 0.15`
   - `step_lr_gamma <= 0.5`
   - `mlp_dropout = 0.2`
   - `weight_decay` 在 `0.001 ~ 0.002`。
