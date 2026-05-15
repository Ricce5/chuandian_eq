# `clf_grid_r_1 / r_1_scratch / r_2 / r_2_scratch` AUC/PR-AUC 对比分析（2026-05-14）

## 1. 分析范围与口径

- 实验目录：
  - `experiments/clf_grid_r_1`
  - `experiments/clf_grid_r_1_scratch`
  - `experiments/clf_grid_r_2`
  - `experiments/clf_grid_r_2_scratch`
- 结果文件：
  - `reports/clf_grid_metrics_summary.json`
  - `reports/clf_grid_metrics_group_mean_std.csv`
  - `reports/clf_grid_metrics_group_best.csv`
  - `reports/clf_grid_metrics_per_run.csv`
- 比较重点：`AUC` 与 `PR-AUC`
- 汇总口径：每组 `(Tfore, Mf)` 下 `3` 个 seed 的均值/标准差；主指标为 `AUC(max)`。

---

## 2. 实验设置差异（影响对比解释）

四个实验的 TF/MF 网格一致（`(10,4.0),(20,4.5),(30,4.5),(60,5.0),(90,5.5)`，每点 3 seeds），主要差异在：

1. `r_1` vs `r_2` 基础训练超参不同：
   - `r_1`：`batch_size=64`, `learning_rate=1e-3`, `warmup_ratio=0.1`
   - `r_2`：`batch_size=32`, `learning_rate=1.5e-3`, `warmup_ratio=0.15`
2. `*_scratch` 与非 scratch 的差异：
   - 非 scratch 启用 `resume_path + load_specific_parts: [encoder]`
   - scratch 版本注释掉预训练加载（从头训练）

---

## 3. 全局结果（15 runs 平均）

按 `15` 个 run 的平均指标：

1. `clf_grid_r_1_scratch`：`AUC=0.786685`, `PR-AUC=0.675273`
2. `clf_grid_r_2`：`AUC=0.779439`, `PR-AUC=0.663752`
3. `clf_grid_r_1`：`AUC=0.767343`, `PR-AUC=0.656538`
4. `clf_grid_r_2_scratch`：`AUC=0.743912`, `PR-AUC=0.653976`

结论：从“跨所有 TF 点的整体平均”看，`r_1_scratch` 最好，`r_2_scratch` 最弱。

---

## 4. 每个实验内的最佳组（按组均值）

### 4.1 最佳 AUC 组

- `clf_grid_r_1`：`(Tfore=30, Mf=4.5)`，`AUC_mean=0.867760`（`std=0.014271`）
- `clf_grid_r_1_scratch`：`(30,4.5)`，`AUC_mean=0.867908`（`std=0.014426`）
- `clf_grid_r_2`：`(60,5.0)`，`AUC_mean=0.882295`（`std=0.039969`）
- `clf_grid_r_2_scratch`：`(60,5.0)`，`AUC_mean=0.891554`（`std=0.024050`）

### 4.2 最佳 PR-AUC 组

- `clf_grid_r_1`：`(10,4.0)`，`PR-AUC_mean=0.836771`
- `clf_grid_r_1_scratch`：`(10,4.0)`，`PR-AUC_mean=0.861096`
- `clf_grid_r_2`：`(60,5.0)`，`PR-AUC_mean=0.843146`
- `clf_grid_r_2_scratch`：`(60,5.0)`，`PR-AUC_mean=0.860881`

结论：
- `r_1` 系列的最优点偏向 `Tfore=30`（AUC）和 `Tfore=10`（PR-AUC）。
- `r_2` 系列的最优点明显集中到 `Tfore=60`。

---

## 5. 跨实验 Top 组（只看 AUC/PR-AUC）

### 5.1 AUC 前四组（组均值）

1. `r_2_scratch @ (60,5.0)`：`0.891554`
2. `r_2 @ (60,5.0)`：`0.882295`
3. `r_1_scratch @ (30,4.5)`：`0.867908`
4. `r_1 @ (30,4.5)`：`0.867760`

### 5.2 PR-AUC 前四组（组均值）

1. `r_1_scratch @ (10,4.0)`：`0.861096`
2. `r_2_scratch @ (60,5.0)`：`0.860881`
3. `r_2 @ (60,5.0)`：`0.843146`
4. `r_1 @ (10,4.0)`：`0.836771`

这说明：如果只追单点最优，`r_2_scratch` 的 `(60,5.0)` 在 AUC 上最强；而 PR-AUC 最强点在 `r_1_scratch` 的 `(10,4.0)`。

---

## 6. scratch vs 预训练加载：增益/退化

### 6.1 `r_1_scratch - r_1`

- 宏平均（按 5 个 TF 组平均）
  - `ΔAUC = +0.01934`
  - `ΔPR-AUC = +0.01874`
- 按 TF 看：
  - 改善最大：`Tfore=60`（`AUC +0.1418`, `PR-AUC +0.1158`）
  - 退化点：`Tfore=20`（`AUC -0.0959`, `PR-AUC -0.0974`）

结论：`r_1` 设定下，从头训练整体优于加载 encoder。

### 6.2 `r_2_scratch - r_2`

- 宏平均（按 5 个 TF 组平均）
  - `ΔAUC = -0.03553`
  - `ΔPR-AUC = -0.00978`
- 按 TF 看：
  - 明显退化：`Tfore=10/20`
  - 轻微改善：`Tfore=30/60/90`

结论：`r_2` 设定下，加载 encoder 整体更稳、更优。

---

## 7. 预训练影响分析（两种配置）与可能原因

### 7.1 定量影响（paired 口径）

将 `pretrain` 与 `scratch` 按相同 `(Tfore, Mf, seed)` 配对后（共 15 对）：

- `r1` 配置（`r1_scratch - r1`）：
  - `AUC` 均值差：`+0.01934`
  - `PR-AUC` 均值差：`+0.01874`
  - 方向：**scratch 略优**
- `r2` 配置（`r2_scratch - r2`）：
  - `AUC` 均值差：`-0.03553`
  - `PR-AUC` 均值差：`-0.00978`
  - 方向：**pretrain 略优**

补充：当前样本量仅 `15` 对（每组 3 seeds），AUC/PR-AUC 的配对检验未达显著（p 值较大），更适合解释为“趋势”而非定论。

### 7.2 分配置解读

1. `r1`（`batch=64, lr=1e-3, warmup=0.1`）  
   预训练影响表现为“分化”：
   - `Tfore=60` 上 pretrain 明显偏弱，scratch 拉升幅度大（`AUC +0.1418`, `PR-AUC +0.1158`）。
   - `Tfore=20` 上反过来，scratch 明显退化（`AUC -0.0959`, `PR-AUC -0.0974`）。
   - 说明：预训练初始化与该配置在不同时间窗存在适配差异，属于“局部正迁移 + 局部负迁移”并存。

2. `r2`（`batch=32, lr=1.5e-3, warmup=0.15`）  
   预训练影响更一致：
   - `Tfore=10/20` scratch 均显著下滑；
   - `Tfore=30/60/90` scratch 仅小幅改善或接近。
   - 整体上 pretrain 更稳，尤其在短窗任务上更有帮助。

### 7.3 可能原因（结合实现与指标行为）

1. **预训练表示与目标窗长存在匹配度差异**  
   同一 encoder 初始化在不同 `Tfore` 上迁移效果不同；`r1` 的 `Tfore=60` 可能出现负迁移，而 `r2` 的短窗（10/20）更依赖预训练先验。

2. **训练超参与预训练初始化有交互**  
   `r1` 与 `r2` 的 `batch_size / lr / warmup` 不同，改变了优化轨迹；预训练是否有利取决于该轨迹能否“利用”而不是“破坏”已有表示。

3. **排序能力与阈值指标分离**  
   本报告关注的 `AUC/PR-AUC` 与阈值无关，反映分数排序质量。实现中二者直接由连续分数计算（`roc_auc_score` 与 `average_precision_score`）。  
   因此这里观测到的差异更接近“模型打分排序变化”，而不是单纯阈值变化导致。

4. **可见的校准/决策边界现象**  
   在 `r2_scratch` 的部分组（尤其短窗）可见 `precision` 下降、`fpr` 上升而 `recall` 上升，常见于模型更激进地判正类；这会压低 PR-AUC 或拉低 AUC 稳定性。

### 7.4 实操建议（专门针对预训练影响）

1. 下一轮建议固定一个训练超参版本，做纯粹的 `pretrain vs scratch` A/B，避免与 `r1/r2` 训练策略耦合。  
2. 每个 `(Tfore,Mf)` 增加 seeds（如 5~10），再做 paired 检验，减少偶然性。  
3. 增加“冻结比例”对照（仅加载不冻结、部分冻结、完全可训练），定位负迁移来源。  
4. 对短窗（10/20）和中窗（60）分开建模决策，不强求一个初始化策略覆盖全部窗长。

---

## 8. 结论与建议（仅面向 AUC/PR-AUC）

1. **若追求全局平均表现**（跨全部 TF）：优先 `clf_grid_r_1_scratch`。
2. **若追求最高单点 AUC**：选 `clf_grid_r_2_scratch @ (Tfore=60, Mf=5.0)`。
3. **若追求最高单点 PR-AUC**：优先 `clf_grid_r_1_scratch @ (10,4.0)`，次选 `clf_grid_r_2_scratch @ (60,5.0)`。
4. 建议下一轮按目标拆两条线：
   - 短期预警线：围绕 `Tfore=10` 优化（偏 PR-AUC）
   - 中期预警线：围绕 `Tfore=60` 优化（偏 AUC）
5. 鉴于 `scratch` 在 `r_1` 与 `r_2` 呈现相反效果，建议固定一套训练超参后再单独比较“是否加载预训练”，避免交互混淆。

---

## 9. 关键文件定位

- `experiments/clf_grid_r_1/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_1_scratch/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_2_scratch/reports/clf_grid_metrics_group_mean_std.csv`
- `experiments/clf_grid_r_1/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_1_scratch/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_2_scratch/reports/clf_grid_metrics_per_run.csv`
- `experiments/clf_grid_r_1/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_1_scratch/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2/configs/clf_mixer_attnpl_t.yaml`
- `experiments/clf_grid_r_2_scratch/configs/clf_mixer_attnpl_t.yaml`
- `src/utils/metrics.py`
