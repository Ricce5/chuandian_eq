# `reg_mixer_attnpl_extractor` 实验分析（Extractor 配置专项，2026-05-16）

## 1. 分析范围与口径

- 实验目录：`experiments/reg_mixer_attnpl_extractor`
- 汇总文件：
  - `experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_summary.json`
  - `experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_group_mean_std.csv`
  - `experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_group_best.csv`
  - `experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_per_run.csv`
- 对比对象（5 个 variant，3 seeds）：
  - `reg_ex_base`（`concat, t_dim=16`）
  - `reg_ex_tdim8`（`concat, t_dim=8`）
  - `reg_ex_tdim32`（`concat, t_dim=32`）
  - `reg_ex_gate`（`gate, t_dim=16`）
  - `reg_ex_gate_tdim32`（`gate, t_dim=32`）
- 评估口径：
  - 主指标：`RMSE`（越小越好）
  - `ckpt_select = best`

---

## 2. 核心结论（先看这个）

1. **本轮均值最优**是 `reg_ex_tdim32`：
   - `RMSE_mean=0.856683`，优于 `base` 的 `0.872244`（约改善 `1.78%`）。
2. **本轮稳健性最优**（`mean+std`）是 `reg_ex_base`：
   - `reg_ex_base: 0.872244 + 0.018437 = 0.890681`
   - `reg_ex_tdim32: 0.856683 + 0.049182 = 0.905865`
3. `gate` 路线未带来收益：
   - `reg_ex_gate` 均值差于 `base`（`ΔRMSE=+0.017564`）。
4. `t_dim=8` 明显退化，`gate+t_dim=32` 明显不稳定：
   - `reg_ex_tdim8: RMSE_mean=0.958175`
   - `reg_ex_gate_tdim32: RMSE_mean=1.059741`, `RMSE_std=0.299196`

---

## 3. 组均值排名（按 RMSE_mean，低→高）

1. `reg_ex_tdim32`：`0.856683 ± 0.049182`
2. `reg_ex_base`：`0.872244 ± 0.018437`
3. `reg_ex_gate`：`0.889808 ± 0.039694`
4. `reg_ex_tdim8`：`0.958175 ± 0.121302`
5. `reg_ex_gate_tdim32`：`1.059741 ± 0.299196`

补充（同趋势指标）：

- `MAE_mean` 最优：`reg_ex_tdim32`（`0.730543`）
- `R2_mean` 最优（最大）：`reg_ex_tdim32`（`-1.197975`）
- `DTW_normalized_mean` 最优（最小）：`reg_ex_tdim32`（`0.557689`）

说明：虽然 `R2_mean` 仍整体为负，但 `tdim32` 在本轮相对最优。

---

## 4. 相对 `base` 的差值（组均值）

以 `reg_ex_base` 为参照：

- `reg_ex_tdim32`：
  - `ΔRMSE=-0.015561`（更好）
  - `ΔMAE=-0.011236`（更好）
  - `ΔR2=+0.076256`（更好）
- `reg_ex_gate`：
  - `ΔRMSE=+0.017564`（更差）
  - `ΔMAE=+0.017852`（更差）
- `reg_ex_tdim8`：
  - `ΔRMSE=+0.085930`（显著更差）
- `reg_ex_gate_tdim32`：
  - `ΔRMSE=+0.187497`（显著更差）

结论：本轮最有效改动是 **把 `t_dim` 从 `16` 提升到 `32`（且保持 `concat`）**。

---

## 5. 稳定性与“单次最优”解读

### 5.1 稳健排序（`RMSE_mean + RMSE_std`）

1. `reg_ex_base`：`0.890681`（最稳）
2. `reg_ex_tdim32`：`0.905865`
3. `reg_ex_gate`：`0.929502`
4. `reg_ex_tdim8`：`1.079477`
5. `reg_ex_gate_tdim32`：`1.358937`

### 5.2 单次最优 vs 组稳定性

- 单次最佳 run 是 `reg_ex_gate_tdim32_seed_2`（`RMSE=0.726803`）。
- 但同组另两个 seed 很差（`1.1463`、`1.3061`），导致组均值/方差最差。

结论：`gate+t_dim32` 属于“偶发高点”，不具备稳定可复现性，不建议作为主线。

---

## 6. 推荐决策

### 6.1 若追求榜单均值最优

- 推荐：`concat + t_dim=32`（即 `reg_ex_tdim32` 路线）。

### 6.2 若追求上线稳健

- 推荐：`concat + t_dim=16`（即 `reg_ex_base` 路线）。

### 6.3 不推荐继续投入

- `t_dim=8` 路线
- `gate+t_dim=32` 路线

---

## 7. 下一轮实验建议（已落地到配置）

针对本轮结论，下一轮建议只保留 `concat` 主线，细扫 `t_dim`：

- `base (t_dim=16)`
- `t_dim=24`
- `t_dim=32`
- `t_dim=40`
- seeds 扩展到 `5`（`[0,1,2,3,4]`）评估稳健性

对应配置文件：

- `config/experiments/reg_mixer_attnpl_grid.yaml`
- `config/reg_mixer_attnpl_t.yaml`

---

## 8. 关键文件定位

- 组均值与方差：`experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_group_mean_std.csv`
- 组内最佳 run：`experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_group_best.csv`
- 每个 run 明细：`experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_per_run.csv`
- 汇总元信息：`experiments/reg_mixer_attnpl_extractor/reports/reg_grid_metrics_summary.json`

