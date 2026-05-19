# clf_grid_r_2 系列 seed 一致性与低 AUC 汇总（精简）

## 范围
- 目录：
  - `experiments/clf_grid_r_2`
  - `experiments/clf_grid_r_2_AZDX_minus_b`
  - `experiments/clf_grid_r_2_layer_1`
  - `experiments/clf_grid_r_2_SCEDC`
  - `experiments/clf_grid_r_2_SCEDC_minus_b`
  - `experiments/clf_grid_r_2_scratch`
- 数据文件：各目录 `reports/clf_grid_metrics_per_run.csv`
- 总样本：`6 实验 × 5 窗口 × 3 seed = 90` 条

## 1) 同窗口下 seed 影响是否一致
- 结论：整体不一致（按 F1），局部有倾向（按 AUC/PR-AUC）。
- F1：30 个“实验-窗口”单元里最佳 seed 次数
  - `seed0=11, seed1=9, seed2=10`（接近打平）
  - Kendall’s W（各窗口）约 `0.083 ~ 0.250`（一致性偏低）
- AUC：最佳 seed 次数 `seed0=6, seed1=9, seed2=15`（seed2 偏多）
- PR-AUC：最佳 seed 次数 `seed0=6, seed1=11, seed2=13`（seed2 略多）

## 2) AUC < 0.5 明细（8 条）
- `clf_grid_r_2`：`Tw=180,Tf=90,Mf=5.5,seed=2`，`auc=0.3341`
- `clf_grid_r_2_AZDX_minus_b`：`Tw=180,Tf=10,Mf=4.0,seed=1`，`auc=0.4486`
- `clf_grid_r_2_SCEDC`：`Tw=180,Tf=90,Mf=5.5,seed=2`，`auc=0.4674`
- `clf_grid_r_2_SCEDC_minus_b`：`Tw=180,Tf=10,Mf=4.0,seed=1`，`auc=0.4719`
- `clf_grid_r_2_SCEDC_minus_b`：`Tw=180,Tf=90,Mf=5.5,seed=1`，`auc=0.2565`
- `clf_grid_r_2_layer_1`：`Tw=180,Tf=90,Mf=5.5,seed=1`，`auc=0.4265`
- `clf_grid_r_2_scratch`：`Tw=180,Tf=10,Mf=4.0,seed=1`，`auc=0.3876`
- `clf_grid_r_2_scratch`：`Tw=180,Tf=90,Mf=5.5,seed=1`，`auc=0.4086`

## 3) 低 AUC 频次汇总（窗口 × seed）
- `Tw180_Tf10_Mf4.0`：`seed0=0, seed1=3, seed2=0`（共 3）
- `Tw180_Tf90_Mf5.5`：`seed0=0, seed1=3, seed2=2`（共 5）

## 4) 一句话结论
- 低 AUC 主要集中在 `Tf=90,Mf=5.5` 与 `Tf=10,Mf=4.0`；且以 `seed=1` 最多（`6/8`）。
