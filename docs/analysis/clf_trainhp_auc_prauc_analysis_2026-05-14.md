# clf_trainhp 超参数扫参分析（AUC / PR-AUC）

日期：2026-05-14  
实验根目录：`experiments/clf_trainhp`

## 1. 分析目标

围绕分类任务的两项核心指标：

- `AUC`
- `PR-AUC`

对 `scripts/run/run_clf_trainhp_sweep.sh` 生成的 `hp_*` 子实验进行聚合分析，并回答：

1. 哪些超参数组合效果好 / 差？
2. 为什么会出现“分数很高但不推荐”的配置？
3. 下一轮搜索该优先哪些区域？

---

## 2. 使用的数据与报告

本次基于以下聚合结果：

- `experiments/clf_trainhp/reports_auc_prauc_auc/clf_trainhp_metrics_hp_overall.csv`
- `experiments/clf_trainhp/reports_auc_prauc_auc/clf_trainhp_best_hp_overall.csv`
- `experiments/clf_trainhp/reports_auc_prauc_prauc/clf_trainhp_best_hp_overall.csv`
- `experiments/clf_trainhp/reports_auc_prauc_combo/clf_trainhp_combo_top10_all.csv`
- `experiments/clf_trainhp/reports_auc_prauc_combo/clf_trainhp_combo_top10_success1.csv`

样本规模：

- 超参数组合数：`108`
- 稳定组合数（`success_rate=1.0`）：`97`

---

## 3. 指标与口径说明

### 3.1 `success_rate` 定义

对一个超参数配置（一个 `hp_*` 子实验）：

- `n_runs`：该配置总 run 数（本轮为 15 = 5 个 `(Tfore,Mf)` × 3 seeds）
- `n_success`：训练 + 测试均成功的 run 数
- `success_rate = n_success / n_runs`

其中 run 成功判定为：

- `train_returncode == 0`
- 且测试返回码成功（`test_returncode == 0`，或所有 `test_returncodes[*] == 0`）

### 3.2 综合分定义

用于排序的综合分：

`combo = 0.5 * AUC + 0.5 * PR-AUC`

---

## 4. 关键结论

### 4.1 单指标全局最优

- **AUC 全局最优（pair_mean）**  
  `hp_lr0p001_wd0p0001_warmup_linear_decay_wr0p1_bs64`  
  - `auc_pair_mean = 0.7928762238819989`
  - `success_rate = 1.0`

- **PR-AUC 全局最优（pair_mean）**  
  `hp_lr0p0006_wd0p001_step_warmup_wr0p1_bs64`  
  - `pr_auc_pair_mean = 0.7048353130107072`
  - `success_rate = 0.2`（稳定性差）

### 4.2 综合分（0.5*AUC + 0.5*PR-AUC）

- **全量榜单 Top1**（未过滤稳定性）：  
  `hp_lr0p0006_wd0p001_step_warmup_wr0p1_bs64`，`combo=0.7482758363615847`，但 `success_rate=0.2`

- **稳定榜单 Top1**（仅 `success_rate=1.0`）：  
  `hp_lr0p001_wd0p0001_warmup_linear_decay_wr0p1_bs64`，`combo=0.7337300327156819`

---

## 5. “为什么 rank1 的 lr=0.0006，但又说它偏差？”

这是典型的“高分但不稳定（幸存者偏差）”：

- 配置：`hp_lr0p0006_wd0p001_step_warmup_wr0p1_bs64`
- `n_runs=15`，`n_success=3`，`success_rate=0.2`
- `n_metrics_found=3`，说明只有 3 个 run 产出可用测试指标
- 其余 12 个 run 失败（多为训练返回码非 0）

因此该配置的高分来自少数成功样本，不能代表可复现稳定收益。  
实际选型应优先参考稳定榜（`success_rate=1.0`）。

---

## 6. 好/差配置的共同特征（基于稳定组）

以下统计基于 `success_rate=1.0` 的 97 个组合。

### 6.1 最显著影响项

1. **learning_rate（最强信号）**  
   - 稳定组平均 `combo`：  
     - `lr=0.0015` → `0.708716`（最好）  
     - `lr=0.001` → `0.694011`  
     - `lr=0.0006` → `0.667470`（最差）
   - 结论：`0.0006` 整体弱于 `0.001/0.0015`。

2. **warmup_ratio（次强信号）**  
   - 稳定组平均 `combo`：  
     - `wr=0.1` → `0.700169`（最好）  
     - `wr=0.15` → `0.694172`  
     - `wr=0.2` → `0.683307`（最差）
   - 结论：`0.2` 风险较高，`0.1` 更稳。

### 6.2 影响较弱项

- **weight_decay**：`0.0001 / 0.001 / 0.01` 差异不大（量级远小于 lr 与 warmup_ratio）。
- **scheduler_type**：整体差距小；在 PR-AUC 上 `step_warmup` 略占优。
- **batch_size**：  
  - 对 AUC：`64` 略优于 `32`  
  - 但全量稳定性上 `64` 更容易失败（总体 `success_rate` 低于 `32`）

### 6.3 高分与低分分位对比（稳定组，Top 25% vs Bottom 25%）

- `lr=0.0015`：Top 组显著更多（15 vs 1）
- `lr=0.0006`：Bottom 组显著更多（2 vs 15）
- `wr=0.1`：Top 组更多（10 vs 4）
- `wr=0.2`：Bottom 组更多（7 vs 13）

结论：**高分组合通常是高学习率（0.001~0.0015）+ 较小 warmup（0.1~0.15）**。

---

## 7. 实操建议（下一轮搜索）

若目标是兼顾 `AUC + PR-AUC` 且确保稳定性，建议：

1. 优先在稳定榜 Top 区域继续细化：
   - `lr`: 以 `0.001`、`0.0015` 为中心（可加局部插值）
   - `warmup_ratio`: 优先 `0.1`、`0.15`
2. 将 `success_rate` 作为硬约束（例如至少 `>=0.8` 或直接 `==1.0`）。
3. 避免仅按全量 `combo` 排名直接选型，防止幸存者偏差。

---

## 8. 关键结果文件索引

- AUC 重点报告：  
  `experiments/clf_trainhp/reports_auc_prauc_auc/clf_trainhp_metrics_hp_overall.csv`
- PR-AUC 重点报告：  
  `experiments/clf_trainhp/reports_auc_prauc_prauc/clf_trainhp_metrics_hp_overall.csv`
- 综合分 Top10（全量）：  
  `experiments/clf_trainhp/reports_auc_prauc_combo/clf_trainhp_combo_top10_all.csv`
- 综合分 Top10（稳定）：  
  `experiments/clf_trainhp/reports_auc_prauc_combo/clf_trainhp_combo_top10_success1.csv`

