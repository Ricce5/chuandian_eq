# `lstm_5_seeds_v5` 结果分析（`last`，2026-05-19）

## 1. 分析对象与口径

- 实验目录：`experiments/lstm_5_seeds_v5`
- 汇总文件：
  - `experiments/lstm_5_seeds_v5/reports/lstm_grid_metrics_summary.json`
  - `experiments/lstm_5_seeds_v5/reports/lstm_grid_metrics_group_mean_std.csv`
  - `experiments/lstm_5_seeds_v5/reports/lstm_grid_metrics_group_best.csv`
  - `experiments/lstm_5_seeds_v5/reports/lstm_grid_metrics_per_run.csv`
- 评估口径：`ckpt_select = last`，`best_metric = RMSE (min)`，`group_by = variant`
- 规模：`42 runs`（`7 variants x 6 seeds`），所有 run 均成功产出指标。

---

## 2. 核心结论（先看结论）

1. **最佳簇仍是 `step_warmup + bs128 + h64/l2`**，且三组很接近：
   - `top_h64_l2_bs128_step_ema098`: `RMSE_mean = 0.8817`（全体最优）
   - `top_h64_l2_bs128_step_wd2e2`: `0.8841`
   - `top_h64_l2_bs128_step`: `0.8848`

2. **`hf_cosine + beta=0.2` 在 v5 上明显退化**（无论 bs128 还是 bs64）：
   - `top_h64_l2_bs128_hfcos_b02`: `RMSE_mean = 1.1276`
   - `top_h64_l2_bs128_hfcos_b02_ema098`: `1.1272`
   - 相比最佳组，差约 `+0.245` RMSE，属于显著退化。

3. **bs64 在该任务与该配置下整体不占优**：
   - `top_h64_l2_bs64_step_ema098`: `1.0828`
   - `top_h64_l2_bs64_hfcos_b02`: `1.2559`
   - 相比对应 bs128 方案均明显更差。

4. **EMA 在 step 系列里仍有小幅收益**（但量级小）：
   - `step_ema098` 相对 `step`：`RMSE_mean -0.0031`
   - `DTW_mean` 略变差（`+0.19`），属于可接受的小 trade-off。

---

## 3. 排名（按 `RMSE_mean`）

1. `lstm__top_h64_l2_bs128_step_ema098`：`0.8817 ± 0.1827`
2. `lstm__top_h64_l2_bs128_step_wd2e2`：`0.8841 ± 0.1842`
3. `lstm__top_h64_l2_bs128_step`：`0.8848 ± 0.1823`
4. `lstm__top_h64_l2_bs64_step_ema098`：`1.0828 ± 0.2819`
5. `lstm__top_h64_l2_bs128_hfcos_b02_ema098`：`1.1272 ± 0.2136`
6. `lstm__top_h64_l2_bs128_hfcos_b02`：`1.1276 ± 0.2141`
7. `lstm__top_h64_l2_bs64_hfcos_b02`：`1.2559 ± 0.2351`

补充：最佳单 run（`group_best`）仍在 `step` 系列：
- `step_ema098` best seed：RMSE `0.7085`
- `step_wd2e2` best seed：`0.7155`
- `step` best seed：`0.7159`

---

## 4. 稳定性与 seed 行为（`last`）

`step + bs128` 三组的 seed 形态高度一致：
- 好 seed：`4/7/2/0`（大致 `0.71~0.83`）
- 中等 seed：`3`（约 `0.95`）
- 差 seed：`8`（约 `1.22`）

说明当前瓶颈主要还是 **跨 seed 方差**，不是“找不到更好的超参中心”。

`hf_cosine + beta=0.2` 系列则整体上移，多个 seed 落在 `1.2~1.37`，与 step 系列拉开明显差距。

---

## 5. 与 v4 的关系（避免误读）

v5 中三组 step 主线（`step / step_ema098 / step_wd2e2`）的均值与 v4 对应组**数值完全一致**。  
这是因为这三组配置本质上与 v4 对应组相同，v5 主要新增的是 `hf_cosine+beta0.2` 和 `bs64` 对照组。

因此，v5 的主要新增信息是：
- `hf_cosine+beta0.2` 在当前设置下不适合；
- `bs64` 也不优于 `bs128`；
- 最优点依旧落在 `step + bs128 + h64/l2` 附近。

---

## 6. 下一步建议（按“显著差异优先”）

1. **保留并继续主跑**：
   - `top_h64_l2_bs128_step_ema098`
   - `top_h64_l2_bs128_step_wd2e2`
   - `top_h64_l2_bs128_step`

2. **下一轮建议删除**（已证伪方向）：
   - `hf_cosine_b02` 两组
   - `bs64` 两组

3. **若继续优化（不做过细微调）**，建议仅做 2~3 组粗粒度试验：
   - 在 `step + bs128 + h64/l2` 上尝试 `learning_rate` 两档：`8e-4` 与 `1.2e-3`
   - 保持 `beta=0.6`，不要再与 `0.2` 混跑
   - EMA 只保留 `0.98`（不再扫更细 decay）

---

## 7. 一句话结论

`lstm_5_seeds_v5` 的新增探索表明：**当前应回到 `step_warmup + bs128 + h64/l2 (+EMA/WD)` 主线，放弃 `hf_cosine+beta0.2` 与 `bs64` 路线。**

