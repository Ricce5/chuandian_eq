# clf_grid_tf_90_stability_grid：按 seed 分窗口跨模型性能排序（AUC / PR-AUC）

## 分析范围
- 结果目录：`/root/autodl-tmp/em_eqf/experiments/clf_grid_tf_90_stability_grid/runs`
- 指标文件：每个 run 的 `metrics_test_best_1.json`
- run 数量：`60`
- seed 集合：`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Twindow 集合：`[180]`
- Tfore 集合：`[90]`
- model 集合：`['clf_mixer_attnpl_t']`
- 配置族数量（跨模型对比单元）：`6`

> 说明：当前目录下窗口参数只有一个，即 `Twindow=180`，因此“按窗口”只有一个窗口组。

## 配置族（参与排名）
- `tf90_a050_lr1e3_wd1e4_wld`
- `tf90_a050_lr6e4_wd1e3_step`
- `tf90_a050_lr6e4_wd1e3_step_noresume`
- `tf90_a075_lr1e3_wd1e3_step`
- `tf90_a075_lr6e4_wd1e3_step`
- `tf90_a090_lr6e4_wd1e3_step`

## 总体排名（跨全部 seed 的平均名次）

### AUC 平均名次（越小越好）

| 排名 | 配置族 | 平均名次 | 平均AUC |
|---:|---|---:|---:|
| 1 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 3.100 | 0.622612 |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` | 3.200 | 0.628889 |
| 3 | `tf90_a075_lr1e3_wd1e3_step` | 3.300 | 0.628694 |
| 4 | `tf90_a075_lr6e4_wd1e3_step` | 3.600 | 0.541189 |
| 5 | `tf90_a090_lr6e4_wd1e3_step` | 3.700 | 0.560039 |
| 6 | `tf90_a050_lr6e4_wd1e3_step` | 4.100 | 0.504055 |

### PR-AUC 平均名次（越小越好）

| 排名 | 配置族 | 平均名次 | 平均PR-AUC |
|---:|---|---:|---:|
| 1 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 2.700 | 0.271539 |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` | 3.000 | 0.267736 |
| 3 | `tf90_a090_lr6e4_wd1e3_step` | 3.500 | 0.225601 |
| 4 | `tf90_a075_lr1e3_wd1e3_step` | 3.700 | 0.280352 |
| 5 | `tf90_a075_lr6e4_wd1e3_step` | 3.900 | 0.205471 |
| 6 | `tf90_a050_lr6e4_wd1e3_step` | 4.200 | 0.206479 |

## 每个 seed 的 Top1（按窗口）

| Twindow | seed | AUC Top1 | AUC | PR-AUC Top1 | PR-AUC |
|---:|---:|---|---:|---|---:|
| 180 | 0 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 0.811306 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 0.355701 |
| 180 | 1 | `tf90_a075_lr1e3_wd1e3_step` | 0.952437 | `tf90_a075_lr1e3_wd1e3_step` | 0.744635 |
| 180 | 2 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 0.863938 | `tf90_a050_lr1e3_wd1e4_wld` | 0.442436 |
| 180 | 3 | `tf90_a090_lr6e4_wd1e3_step` | 0.679532 | `tf90_a090_lr6e4_wd1e3_step` | 0.357857 |
| 180 | 4 | `tf90_a075_lr6e4_wd1e3_step` | 0.571540 | `tf90_a050_lr1e3_wd1e4_wld` | 0.336385 |
| 180 | 5 | `tf90_a050_lr1e3_wd1e4_wld` | 0.820273 | `tf90_a050_lr1e3_wd1e4_wld` | 0.320420 |
| 180 | 6 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 0.810136 | `tf90_a090_lr6e4_wd1e3_step` | 0.380410 |
| 180 | 7 | `tf90_a050_lr6e4_wd1e3_step` | 0.815984 | `tf90_a050_lr6e4_wd1e3_step` | 0.327454 |
| 180 | 8 | `tf90_a075_lr6e4_wd1e3_step` | 0.686940 | `tf90_a050_lr6e4_wd1e3_step_noresume` | 0.306825 |
| 180 | 9 | `tf90_a075_lr1e3_wd1e3_step` | 0.912671 | `tf90_a075_lr1e3_wd1e3_step` | 0.470607 |

## 每个 seed 的完整排序（window=180）

### seed = 0

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.811306) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.355701) |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` (0.794152) | `tf90_a050_lr1e3_wd1e4_wld` (0.330490) |
| 3 | `tf90_a090_lr6e4_wd1e3_step` (0.630799) | `tf90_a090_lr6e4_wd1e3_step` (0.213727) |
| 4 | `tf90_a075_lr1e3_wd1e3_step` (0.554386) | `tf90_a075_lr1e3_wd1e3_step` (0.190276) |
| 5 | `tf90_a050_lr6e4_wd1e3_step` (0.437037) | `tf90_a075_lr6e4_wd1e3_step` (0.158625) |
| 6 | `tf90_a075_lr6e4_wd1e3_step` (0.392593) | `tf90_a050_lr6e4_wd1e3_step` (0.155489) |

### seed = 1 --

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a075_lr1e3_wd1e3_step` (0.952437) | `tf90_a075_lr1e3_wd1e3_step` (0.744635) |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` (0.611306) | `tf90_a050_lr1e3_wd1e4_wld` (0.303505) |
| 3 | `tf90_a075_lr6e4_wd1e3_step` (0.577388) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.229094) |
| 4 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.564522) | `tf90_a075_lr6e4_wd1e3_step` (0.183992) |
| 5 | `tf90_a090_lr6e4_wd1e3_step` (0.358674) | `tf90_a090_lr6e4_wd1e3_step` (0.143442) |
| 6 | `tf90_a050_lr6e4_wd1e3_step` (0.193762) | `tf90_a050_lr6e4_wd1e3_step` (0.127751) |

### seed = 2

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.863938) | `tf90_a050_lr1e3_wd1e4_wld` (0.442436) |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` (0.829630) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.373888) |
| 3 | `tf90_a075_lr1e3_wd1e3_step` (0.614035) | `tf90_a090_lr6e4_wd1e3_step` (0.286533) |
| 4 | `tf90_a050_lr6e4_wd1e3_step` (0.588694) | `tf90_a050_lr6e4_wd1e3_step` (0.260533) |
| 5 | `tf90_a090_lr6e4_wd1e3_step` (0.569591) | `tf90_a075_lr1e3_wd1e3_step` (0.224004) |
| 6 | `tf90_a075_lr6e4_wd1e3_step` (0.382846) | `tf90_a075_lr6e4_wd1e3_step` (0.150087) |

### seed = 3 --

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a090_lr6e4_wd1e3_step` (0.679532) | `tf90_a090_lr6e4_wd1e3_step` (0.357857) |
| 2 | `tf90_a050_lr6e4_wd1e3_step` (0.654191) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.251173) |
| 3 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.649903) | `tf90_a050_lr6e4_wd1e3_step` (0.238757) |
| 4 | `tf90_a075_lr1e3_wd1e3_step` (0.569981) | `tf90_a075_lr1e3_wd1e3_step` (0.185701) |
| 5 | `tf90_a050_lr1e3_wd1e4_wld` (0.534893) | `tf90_a050_lr1e3_wd1e4_wld` (0.179806) |
| 6 | `tf90_a075_lr6e4_wd1e3_step` (0.422222) | `tf90_a075_lr6e4_wd1e3_step` (0.154387) |

### seed = 4 --

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a075_lr6e4_wd1e3_step` (0.571540) | `tf90_a050_lr1e3_wd1e4_wld` (0.336385) |
| 2 | `tf90_a050_lr1e3_wd1e4_wld` (0.557115) | `tf90_a075_lr6e4_wd1e3_step` (0.234548) |
| 3 | `tf90_a075_lr1e3_wd1e3_step` (0.548148) | `tf90_a075_lr1e3_wd1e3_step` (0.215777) |
| 4 | `tf90_a090_lr6e4_wd1e3_step` (0.437037) | `tf90_a090_lr6e4_wd1e3_step` (0.163110) |
| 5 | `tf90_a050_lr6e4_wd1e3_step` (0.407797) | `tf90_a050_lr6e4_wd1e3_step` (0.160594) |
| 6 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.352047) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.153995) |

### seed = 5

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a050_lr1e3_wd1e4_wld` (0.820273) | `tf90_a050_lr1e3_wd1e4_wld` (0.320420) |
| 2 | `tf90_a075_lr1e3_wd1e3_step` (0.651462) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.253001) |
| 3 | `tf90_a075_lr6e4_wd1e3_step` (0.635478) | `tf90_a075_lr1e3_wd1e3_step` (0.250541) |
| 4 | `tf90_a090_lr6e4_wd1e3_step` (0.531774) | `tf90_a075_lr6e4_wd1e3_step` (0.229136) |
| 5 | `tf90_a050_lr6e4_wd1e3_step` (0.452632) | `tf90_a050_lr6e4_wd1e3_step` (0.158429) |
| 6 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.431969) | `tf90_a090_lr6e4_wd1e3_step` (0.150485) |

### seed = 6

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.810136) | `tf90_a090_lr6e4_wd1e3_step` (0.380410) |
| 2 | `tf90_a090_lr6e4_wd1e3_step` (0.778168) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.368581) |
| 3 | `tf90_a050_lr1e3_wd1e4_wld` (0.570370) | `tf90_a050_lr1e3_wd1e4_wld` (0.245478) |
| 4 | `tf90_a050_lr6e4_wd1e3_step` (0.507992) | `tf90_a075_lr6e4_wd1e3_step` (0.209148) |
| 5 | `tf90_a075_lr6e4_wd1e3_step` (0.487914) | `tf90_a050_lr6e4_wd1e3_step` (0.184475) |
| 6 | `tf90_a075_lr1e3_wd1e3_step` (0.478752) | `tf90_a075_lr1e3_wd1e3_step` (0.172945) |

### seed = 7 --

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a050_lr6e4_wd1e3_step` (0.815984) | `tf90_a050_lr6e4_wd1e3_step` (0.327454) |
| 2 | `tf90_a075_lr6e4_wd1e3_step` (0.660039) | `tf90_a075_lr6e4_wd1e3_step` (0.281160) |
| 3 | `tf90_a090_lr6e4_wd1e3_step` (0.561014) | `tf90_a090_lr6e4_wd1e3_step` (0.179992) |
| 4 | `tf90_a075_lr1e3_wd1e3_step` (0.526706) | `tf90_a075_lr1e3_wd1e3_step` (0.178783) |
| 5 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.476023) | `tf90_a050_lr1e3_wd1e4_wld` (0.174873) |
| 6 | `tf90_a050_lr1e3_wd1e4_wld` (0.454971) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.170993) |

### seed = 8

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a075_lr6e4_wd1e3_step` (0.686940) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.306825) |
| 2 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.644834) | `tf90_a050_lr6e4_wd1e3_step` (0.303587) |
| 3 | `tf90_a050_lr6e4_wd1e3_step` (0.597271) | `tf90_a075_lr6e4_wd1e3_step` (0.249412) |
| 4 | `tf90_a050_lr1e3_wd1e4_wld` (0.587135) | `tf90_a050_lr1e3_wd1e4_wld` (0.197383) |
| 5 | `tf90_a075_lr1e3_wd1e3_step` (0.478363) | `tf90_a090_lr6e4_wd1e3_step` (0.184321) |
| 6 | `tf90_a090_lr6e4_wd1e3_step` (0.475244) | `tf90_a075_lr1e3_wd1e3_step` (0.170250) |

### seed = 9 --

| 名次 | AUC 排序 | PR-AUC 排序 |
|---:|---|---|
| 1 | `tf90_a075_lr1e3_wd1e3_step` (0.912671) | `tf90_a075_lr1e3_wd1e3_step` (0.470607) |
| 2 | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.621442) | `tf90_a050_lr6e4_wd1e3_step_noresume` (0.252137) |
| 3 | `tf90_a075_lr6e4_wd1e3_step` (0.594932) | `tf90_a075_lr6e4_wd1e3_step` (0.204215) |
| 4 | `tf90_a090_lr6e4_wd1e3_step` (0.578558) | `tf90_a090_lr6e4_wd1e3_step` (0.196138) |
| 5 | `tf90_a050_lr1e3_wd1e4_wld` (0.529045) | `tf90_a050_lr6e4_wd1e3_step` (0.147724) |
| 6 | `tf90_a050_lr6e4_wd1e3_step` (0.385185) | `tf90_a050_lr1e3_wd1e4_wld` (0.146589) |

## 结论（面向 AUC / PR-AUC）
- 若按**平均名次**综合看，`tf90_a050_lr6e4_wd1e3_step_noresume` 最稳健（AUC 与 PR-AUC 都是第 1）。
- `tf90_a050_lr1e3_wd1e4_wld` 在 PR-AUC 上整体也较强，且多 seed 下进入前二。
- `tf90_a075_lr1e3_wd1e3_step` 在个别 seed（如 1、9）出现很高峰值，但跨 seed 波动较大。

---
- 生成时间：2026-05-19
0 1 2 4 5 6 8