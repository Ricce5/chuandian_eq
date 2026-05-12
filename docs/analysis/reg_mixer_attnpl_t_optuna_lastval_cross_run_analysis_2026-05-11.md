# `reg_mixer_attnpl_t` Optuna（`last val loss`）跨实验结论（2026-05-11）

## 1. 分析范围与口径

- 日期：`2026-05-11`
- 目标问题：分析“`last` 保存时 `val_loss` 较低的组有什么特点”
- 主数据来源（同一实验族）：
  - `experiments/reg_mixer_attnpl_t_optuna_profiles/20260510-232207/runs/base/optuna_trials`
  - `experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-000926/runs/base/optuna_trials`
  - `experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-001210/runs/base/optuna_trials`
- 每个 run 的分析结果文件：
  - `.../runs/base/analysis_last_val_loss/last_val_loss_group_analysis.json`
- 分组口径：
  - 默认 low/high 各取 25%（按 `last_model_1.pth` 的 `val_loss` 排序）

说明：`20260510-232207` 仅 3 个 trial，属于小样本；主结论以两个 30-trial run（`000926`、`001210`）为主。

---

## 2. 核心结论（可执行版）

1. **低 `last val loss` 组最稳定特征**（两个 30-trial run 一致）：
   - `criterion_cfg.beta = 0.3`
   - `batch_size = 64`
   - `step_lr_step_size_ratio = 0.15`
   - `warmup_ratio` 多为 `0.1`
2. **高 `last val loss` 组更常见**：
   - `criterion_cfg.beta = 0.2`
   - `batch_size = 128` 占比更高
   - `step_lr_step_size_ratio = 0.2 / 0.25` 占比更高
3. **稳定性（last-best 回弹）差异明显**：
   - low 组 `mean(last-best) ≈ 0.0070`
   - high 组 `mean(last-best) ≈ 0.0183`
   - 说明 low 组不仅“低”，还更稳。
4. **学习率有趋势但不是最强主因**：
   - 在 60-trial 合并视角里，`learning_rate` 与 `last val loss` 的相关性偏弱（Pearson 约 `0.13`）
   - `weight_decay` 的相关性相对更明显（Pearson 约 `0.27`）
   - 结论：`lr` 不是单独决定因素，需和 `beta/batch_size/scheduler` 联动看。

---

## 3. 关键证据（30-trial + 30-trial 合并）

### 3.1 分布对比（low/high 各 8，在每个 run 内部分位）

基于：
- `experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-000926/runs/base/analysis_last_val_loss/last_val_loss_group_analysis.json`
- `experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-001210/runs/base/analysis_last_val_loss/last_val_loss_group_analysis.json`

聚合计数（两个 run 合计）：

- `criterion_cfg.beta`
  - low：`0.3 -> 16/16`
  - high：`0.2 -> 11/16`, `0.3 -> 5/16`
- `batch_size`
  - low：`64 -> 16/16`
  - high：`64 -> 8/16`, `128 -> 8/16`
- `step_lr_step_size_ratio`
  - low：`0.15 -> 16/16`
  - high：`0.2 -> 8/16`, `0.25 -> 4/16`, `0.15 -> 4/16`
- `warmup_ratio`
  - low：`0.1 -> 13/16`
  - high：`0.05/0.1/0.15` 基本均分（各 `~5-6`）

稳定性：
- low 组平均 `last-best`：`~0.0070`
- high 组平均 `last-best`：`~0.0183`

### 3.2 学习率与正则（60-trial 合并）

合并对象：
- `run_20260511-000926` 的 `trial_last_val_loss_rows.csv`
- `run_20260511-001210` 的 `trial_last_val_loss_rows.csv`

将 60 个 trial 合并后再按 `last val loss` 切 low/high 各 15：

- `learning_rate`
  - low median：`7.96e-4`
  - high median：`9.30e-4`
  - 趋势：high 组略高，但差异不大
- `weight_decay`
  - low median：`9.40e-4`
  - high median：`1.39e-3`
  - 趋势：high 组更高更明显
- 相关性（全 60）
  - `corr(lr, last_val_loss)`：Pearson `~0.13`（弱）
  - `corr(weight_decay, last_val_loss)`：Pearson `~0.27`（弱-中等）

---

## 4. 与其他实验的交叉印证

作为“稳定性”旁证，参考：
- `experiments/reg_mixer_attnpl_grid_lr_on/reports_best/reg_grid_metrics_per_run.csv`
- `experiments/reg_mixer_attnpl_grid_lr_on/reports_last/reg_grid_metrics_per_run.csv`

观察到：
- 24 个 grid run 中，`best` 与 `last` 差异很常见，且组间波动大。
- 这支持一个现实：**只看单点最优不够，必须同时看 `last` 的回弹（稳定性）**。

该点与本次 Optuna low/high 结果一致：low 组 `last-best` gap 更小，更适合作为可复现实验主线。

---

## 5. 建议的下一轮搜索空间（针对 `last val loss`）

优先固定（建议主线）：
- `criterion_cfg.beta = 0.3`
- `batch_size = 64`
- `step_lr_step_size_ratio = 0.15`
- `warmup_ratio` 先以 `0.1` 为中心（可小范围试 `0.08~0.15`）

再微调（次优先）：
- `learning_rate`: 聚焦中等区间（例如 `~6e-4` 到 `~1.0e-3`）
- `weight_decay`: 避免偏高，优先在 low 组附近收窄

评估口径建议：
- 报告同时给出 `best_val_loss` 与 `last_val_loss`
- 必带 `last-best` gap 统计（mean/median/p90）

---

## 6. 注意事项（与代码变更相关）

`2026-05-11` 当天已修复 `warmup_linear_decay` 的参数组学习率问题（此前会把各组 LR 拉平）。  
因此，上述历史 run 结论属于“修复前行为”统计；建议在修复后按同脚本复跑一轮确认结论是否保持。

可复用脚本：
- `scripts/analyze_optuna_last_val_loss_groups.py`

