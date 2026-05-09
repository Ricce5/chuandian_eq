# `clf_mixer_attnpl_t` 近期实验结果分析（AUC/PR-AUC主导，2026-05-09）

## 1. 分析范围

- 时间范围：`2026-05-08` 到 `2026-05-09`
- 模型：`clf_mixer_attnpl_t`
- 统计口径：统一使用 `metrics_test_best_1.json`
- 核心指标：`AUC`、`PR-AUC`（阈值无关），并补充 `sqrt(AUC*PR-AUC)` 作为综合排序分数
- 样本数量：`44` 个 checkpoint

---

## 2. 总体结论（阈值无关视角）

### 2.1 全体分布

- `AUC`: mean `0.8191`，std `0.0733`，min `0.5379`，max `0.9091`
- `PR-AUC`: mean `0.7530`，std `0.1008`，min `0.3619`，max `0.8590`
- `G-Score = sqrt(AUC*PR-AUC)`: mean `0.7849`，std `0.0871`，max `0.8837`

### 2.2 分日对比

- `2026-05-08`（`n=27`）：
  - `AUC_mean=0.8405`
  - `PR-AUC_mean=0.7800`
  - `G-Score_mean=0.8094`
- `2026-05-09`（`n=17`）：
  - `AUC_mean=0.7850`
  - `PR-AUC_mean=0.7101`
  - `G-Score_mean=0.7460`

结论：若以阈值无关指标为准，`2026-05-08` 的整体质量明显高于 `2026-05-09`。

---

## 3. 最优实验（按阈值无关指标）

### 3.1 按 AUC / PR-AUC / G-Score 综合最优

1. `clf_mixer_attnpl_t_20260508-154834`
   - `AUC=0.9091`, `PR-AUC=0.8590`, `G-Score=0.8837`
2. `clf_mixer_attnpl_t_20260508-151713`
   - `AUC=0.9005`, `PR-AUC=0.8568`, `G-Score=0.8784`
3. `clf_mixer_attnpl_t_20260509-124159`
   - `AUC=0.8934`, `PR-AUC=0.8394`, `G-Score=0.8660`
4. `clf_mixer_attnpl_t_20260509-124545`
   - `AUC=0.8921`, `PR-AUC=0.8387`, `G-Score=0.8650`
5. `clf_mixer_attnpl_t_20260509-165013`
   - `AUC=0.8867`, `PR-AUC=0.8232`, `G-Score=0.8544`

### 3.2 高质量 run 占比

- `AUC >= 0.85`: `14/44`
- `AUC >= 0.88`: `5/44`
- `PR-AUC >= 0.82`: `11/44`

---

## 4. 配置与 AUC/PR-AUC 的关系

### 4.1 三个关键分组（n>=3）

#### A 组：`resume=false, n_layer=3, attn=[1]`

- `n=6`
- `AUC_mean=0.8633`（最高）
- `PR-AUC_mean=0.8151`
- `G-Score_mean=0.8389`

说明：该组在“排序能力”上最稳最强，适合作为 AUC 主导的稳健基线。

#### B 组：`resume=true, n_layer=2, attn=[]`, 且 selective load=`30/30`

- `n=6`
- `AUC_mean=0.8587`
- `PR-AUC_mean=0.7984`
- `G-Score_mean=0.8280`

说明：该组同样很强，且单次峰值高（如 `20260509-124159`）。

#### C 组：`resume=true, n_layer=3, attn=[1]`

- `n=20`
- `AUC_mean=0.8394`
- `PR-AUC_mean=0.7799`
- `G-Score_mean=0.8088`

说明：均值不错，属于“可用且相对稳”的主力分组。

### 4.2 selective load 覆盖率影响

- load=`(30,30,30,0,0)`（完整）：
  - `AUC_mean=0.8587`, `PR-AUC_mean=0.7984`
- load=`(40,30,30,10,0)`（缺失10）：
  - `AUC_mean=0.7752`, `PR-AUC_mean=0.7142`

结论：在当前批次中，encoder 加载不完整（缺失10）与 AUC/PR-AUC 显著下降相关。

---

## 5. 对齐实验复核（AUC口径）

来自 `tmp/exp_align_hypothesis/runs.json`：

- `clf_aligned_pre`（`checkpoints/clf_mixer_attnpl_t_20260509-165013`）
  - `AUC=0.8867`, `PR-AUC=0.8232`
- `clf_aligned_scratch`（`checkpoints/clf_mixer_attnpl_t_20260509-165140`）
  - `AUC=0.8415`, `PR-AUC=0.8125`

结论：若按阈值无关指标判断，本组实验是 `pretrain` 优于 `scratch`。

---

## 6. 小样本曲线补充（阈值无关）

来自 `tmp/small_data_curve/curve_summary.csv`：

- ratio=`0.1`: `auc_std=0.3716`, `pr_auc_std=0.2946`（极不稳定）
- ratio=`0.25`: `auc_mean=0.8429`, `pr_auc_mean=0.7769`
- ratio=`1.0`: `auc_mean=0.8591`, `pr_auc_mean=0.8098`

结论：在低样本比例（尤其 0.1）下，AUC/PR-AUC 波动极大，建议实际结论至少基于 `ratio>=0.25`。

---

## 7. 建议（AUC优先）

1. 近期主线推荐优先级：
   - 第一优先：`resume=false + n_layer=3 + attn=[1]`（AUC/PR-AUC 均值最佳）
   - 第二优先：`resume=true + n_layer=2 + attn=[] + 完整加载30/30`
2. 在训练前设置加载门控：
   - 若出现 `target=40` 但 `loaded=30`（缺失10），优先视为风险 run。
3. 报告口径建议：
   - 主表使用 `AUC/PR-AUC/G-Score`
   - `R/F1` 仅作为阈值后决策表现补充。

---

## 8. 关键文件

- 近期高峰值之一：`checkpoints/clf_mixer_attnpl_t_20260508-154834/metrics_test_best_1.json`
- 近期高峰值之一：`checkpoints/clf_mixer_attnpl_t_20260509-124159/metrics_test_best_1.json`
- 对齐实验索引：`tmp/exp_align_hypothesis/runs.json`
- 小样本曲线汇总：`tmp/small_data_curve/curve_summary.csv`
