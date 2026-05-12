# `lstm` 近期实验结果分析（2026-05-09）

## 1. 分析范围

- 时间范围：`2026-05-05` 到 `2026-05-09`
- 目标模型：`lstm`（回归任务）
- 数据来源：
  - `checkpoints/lstm_202605*/`
  - `tmp/lstm_optuna_profiles/run_20x2/`
  - `experiment_record.md`

---

## 2. 关键结论

1. 近期最优两次实验为：
   - `checkpoints/lstm_20260509-204621`
   - `checkpoints/lstm_20260509-204708`

   对应 `metrics_test_best_1.json` 指标：
   - MAE 约 `0.604`
   - RMSE 约 `0.695~0.699`
   - R2 约 `-0.46 ~ -0.45`

2. 与 `experiment_record.md` 中记录的 `lstm_20260507-191427`（MAE `0.6399`，RMSE `0.7308`）相比，近期结果有明确提升。

3. 训练稳定性仍需关注：多数 run 存在 `metrics_test_last_1` 明显劣于 `metrics_test_best_1` 的情况，建议报告和部署统一使用 `best` checkpoint。

---

## 3. 近期代表性结果（test best）

| checkpoint | MAE | RMSE | R2 | DTW_normalized |
|---|---:|---:|---:|---:|
| `lstm_20260509-204621` | 0.603968 | 0.698505 | -0.458035 | 0.546375 |
| `lstm_20260509-204708` | 0.604044 | 0.695485 | -0.445457 | 0.559850 |
| `lstm_20260509-151903` | 0.609402 | 0.728540 | -0.586118 | 0.405264 |
| `lstm_20260507-191427` | 0.639857 | 0.730756 | -0.595785 | 0.456575 |

---

## 4. 配置差异与影响

### 4.1 网络规模

- 近期较优结果集中在：
  - `lstm_hidden_size: 256`
  - `lstm_num_layers: 4`
  - `lstm_dropout: 0.3`
- 相比此前常见的 `128 x 2`，整体表现更好。

### 4.2 调度器

- `lstm_20260509-204708` 使用 `scheduler_type: hf_cosine`
- `lstm_20260509-204621` 使用 `scheduler_type: step_warmup`
- 二者其他主配置接近，`hf_cosine` 在 RMSE 上略优，但差距较小。

### 4.3 时间特征模式（关键）

- `t_elaps_mode=window` 明显优于 `t_elaps_mode=global`。
- 例子：
  - `lstm_20260509-151903`（window）：MAE `0.6094`
  - `lstm_20260509-152433`（global）：MAE `0.7539`

---

## 5. Optuna 结果（20x2）

来自 `tmp/lstm_optuna_profiles/run_20x2/profile_optuna_summary.json`：

- `best_overall.profile = window`
- window 最优：
  - `best_rmse = 0.7428`
  - `best_dtw_normalized = 0.36295`
- global 最优：
  - `best_rmse = 0.7362`
  - `best_dtw_normalized = 0.46709`

说明：若综合考虑 DTW/RMSE 目标，`window` profile 更稳妥。

---

## 6. 特征重要性（基于 151903 的 permutation）

文件：`checkpoints/lstm_20260509-151903/analysis/lstm_permutation_importance/lstm_permutation_importance.csv`

按 `delta_rmse_mean` 排名前三：

1. `T_elaps5.5`
2. `dM_lsq`
3. `Mag_max`

对应 baseline：
- `RMSE = 0.7285`
- `MAE = 0.6094`
- `R2 = -0.5861`

---

## 7. 建议

1. 主线配置建议固定为：
   - `window + high_sentinel + 256x4 + smooth_l1(beta=0.2)`
   - 调度器优先尝试 `hf_cosine`
2. 评估口径统一使用 `metrics_test_best_1.json`，避免 `last` 带来的后期退化干扰。
3. 下一步应至少做 3 个随机种子复现（例如 `0/1/2`），报告均值和标准差，以验证当前提升是否稳定。

