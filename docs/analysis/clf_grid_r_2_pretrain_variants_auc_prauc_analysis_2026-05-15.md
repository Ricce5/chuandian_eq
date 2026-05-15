# `clf_grid_r_2` 预训练变体 AUC/PR-AUC 对比分析（2026-05-15）

## 1. 分析范围与口径

- 实验目录：
  - `experiments/clf_grid_r_2`
  - `experiments/clf_grid_r_2_AZDX_minus_b`
  - `experiments/clf_grid_r_2_SCEDC`
  - `experiments/clf_grid_r_2_SCEDC_minus_b`
  - `experiments/clf_grid_r_2_scratch`
- 结果文件：
  - `reports/clf_grid_metrics_group_mean_std.csv`
  - `reports/clf_grid_metrics_per_run.csv`
- 指标重点：`AUC` 与 `PR-AUC`
- 汇总口径：
  1. 每个实验 `15 runs`（`5` 个 `Tfore` × `3` seeds）整体平均
  2. 每个 `Tfore` 下 `3` seeds 的组均值
  3. 与 `scratch` 按相同 `(Tfore, seed)` 的配对差值

---

## 2. 关键可比性检查（预训练差异）

5 组实验的网格与训练超参一致，主要差异在是否加载预训练 encoder 以及 `resume_path` 指向的 checkpoint：

- `clf_grid_r_2`：`resume_path=./checkpoints/mixer_tpp_20260508-182744/last_model_1.pth`
- `clf_grid_r_2_AZDX_minus_b`：`resume_path=./checkpoints/mixer_tpp_20260510-211348/last_model_1.pth`
- `clf_grid_r_2_SCEDC`：`resume_path=./checkpoints/mixer_tpp_20260514-211059/last_model_1.pth`
- `clf_grid_r_2_SCEDC_minus_b`：`resume_path=./checkpoints/mixer_tpp_20260514-211318/last_model_1.pth`
- `clf_grid_r_2_scratch`：未启用 `resume_path`（从头训练）

且预训练组均为：`load_specific_parts: [encoder]`。

---

## 3. 全局结果（15 runs 平均）

按整体平均（越高越好）：

1. `clf_grid_r_2`：`AUC=0.779439`, `PR-AUC=0.663752`
2. `clf_grid_r_2_SCEDC`：`AUC=0.756639`, `PR-AUC=0.652336`
3. `clf_grid_r_2_scratch`：`AUC=0.743912`, `PR-AUC=0.653976`
4. `clf_grid_r_2_AZDX_minus_b`：`AUC=0.740308`, `PR-AUC=0.625726`
5. `clf_grid_r_2_SCEDC_minus_b`：`AUC=0.690820`, `PR-AUC=0.601316`

相对 `scratch` 的均值差：

- `clf_grid_r_2`：`ΔAUC=+0.035527`, `ΔPR-AUC=+0.009776`
- `clf_grid_r_2_SCEDC`：`ΔAUC=+0.012726`, `ΔPR-AUC=-0.001640`
- `clf_grid_r_2_AZDX_minus_b`：`ΔAUC=-0.003604`, `ΔPR-AUC=-0.028250`
- `clf_grid_r_2_SCEDC_minus_b`：`ΔAUC=-0.053092`, `ΔPR-AUC=-0.052660`

结论：综合看 `clf_grid_r_2` 预训练最佳；`SCEDC` 次之；两个 `minus_b` 版本整体弱于或显著弱于 `scratch`。

---

## 4. 分 `Tfore` 结果（组均值）

### 4.1 AUC（每个 `Tfore`）

- `Tfore=10`：`SCEDC (0.753227)` 最好
- `Tfore=20`：`r_2 (0.834893)` 最好
- `Tfore=30`：`scratch (0.845745)` 最好
- `Tfore=60`：`SCEDC (0.899972)` 最好
- `Tfore=90`：`AZDX_minus_b (0.658220)` 最好

### 4.2 PR-AUC（每个 `Tfore`）

- `Tfore=10`：`SCEDC (0.810575)` 最好
- `Tfore=20`：`r_2 (0.684416)` 最好
- `Tfore=30`：`scratch (0.810386)` 最好
- `Tfore=60`：`scratch (0.860881)` 略优于 `SCEDC (0.860602)`
- `Tfore=90`：`scratch (0.276542)` 最好

结论：预训练收益集中在短中窗（尤其 `10/20`）；到 `30/90` 并不稳定，`scratch` 多次反超。

---

## 5. 与 `scratch` 的配对差值（同 `Tfore + seed`）

### 5.1 `clf_grid_r_2` vs `scratch`

- `mean ΔAUC=+0.035527`, `mean ΔPR-AUC=+0.009776`
- AUC 赢面 `8/15`，PR-AUC 赢面 `8/15`
- 分窗：`T10/T20` 提升明显，`T30/T60/T90` 小幅回落

### 5.2 `clf_grid_r_2_SCEDC` vs `scratch`

- `mean ΔAUC=+0.012726`, `mean ΔPR-AUC=-0.001640`
- AUC 赢面 `9/15`，PR-AUC 赢面 `6/15`
- 分窗：`T10/T20` 明显收益；`T30/T90` 明显退化

### 5.3 `clf_grid_r_2_AZDX_minus_b` vs `scratch`

- `mean ΔAUC=-0.003604`, `mean ΔPR-AUC=-0.028250`
- AUC 赢面 `8/15`，PR-AUC 赢面 `6/15`
- 分窗：`T10/T20` 有收益，但 `T30/T60` 退化较明显

### 5.4 `clf_grid_r_2_SCEDC_minus_b` vs `scratch`

- `mean ΔAUC=-0.053092`, `mean ΔPR-AUC=-0.052660`
- AUC 赢面 `4/15`，PR-AUC 赢面 `3/15`
- 分窗：仅 `T10/T20` 接近持平，其余窗长普遍退化

---

## 6. 结论与建议（面向 AUC/PR-AUC）

1. **首选预训练**：`clf_grid_r_2`（整体 AUC/PR-AUC 最优）。
2. **可选预训练**：`clf_grid_r_2_SCEDC`（AUC 尚可，但 PR-AUC 不稳定）。
3. **不建议当前使用**：`AZDX_minus_b` 与 `SCEDC_minus_b`（尤其后者明显退化）。
4. **按窗长策略**：
   - 短中窗（`Tfore=10/20`）可优先预训练版本；
   - 中长窗（`Tfore=30/90`）建议保留 `scratch` 作为强基线。
5. 下一轮建议：对每个 `Tfore` 扩 seed（如 `5~10`）后再定最终预训练方案，避免 3-seed 抖动影响结论。

---

## 7. 关键文件定位

- `experiments/clf_grid_r_2/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2_AZDX_minus_b/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2_SCEDC/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2_SCEDC_minus_b/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2_scratch/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2_AZDX_minus_b/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2_SCEDC/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2_SCEDC_minus_b/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2_scratch/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2_AZDX_minus_b/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2_SCEDC/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2_SCEDC_minus_b/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2_scratch/configs/clf_mixer_attnpl_t.yaml`
