# 诱发地震预测实用指南

本指南聚焦本仓库中的诱发地震（Induced Seismicity）时间点过程（TPP）预测流程，涵盖：
- 可用模型
- 可用数据集
- 如何加入背景模块（BG）
- 如何评估模型预测

---

## 1. 可用模型（诱发地震预测）

先查看当前注册模型：

```bash
python - <<'PY'
from src.models.builders import ModelBuilder
print(sorted(ModelBuilder.list_available()))
PY
```

其中适用于诱发地震预测（`task_type: tpp`）的常用模型：
- `etas`
- `etas_zhuang`
- `rtpp`
- `rtpp_v2`

对应实现入口在：
- `src/models/builders/tpp.py`

---

## 2. 可用数据集（诱发地震）

先查看当前注册 catalog：

```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print(sorted(Catalog.list_available()))
PY
```

诱发地震相关数据集主要包括：
- `Basel-Standard`
- `FORGE2022-Standard`
- `St1-2018-Standard`
- `St1-2020-Standard`
- `PNR-Standard`、`PNR_1z-Standard`、`PNR_2-Standard`
- `CB_HAB1a-Standard`、`CB_HAB1b-Standard`、`CB_HAB4-Standard`
- `CooperBasin-Standard`
- `SSFS-Standard`、`SSFS1993-Standard`、`SSFS2000-Standard`、`SSFS2003-Standard`、`SSFS2004-Standard`、`SSFS2005-Standard`

说明：
- 训练配置中通常写 `dataset: <Name>`，TPP 数据准备会解析为 `<Name>-Standard`。
- 例如 `dataset: Geysers` 会对应 `Geysers-Standard`。

---

## 3. 如何加入背景模块（BG）

### 3.1 查看可用 BG 模块

```bash
python - <<'PY'
from src.models.bg import BGModel
print(sorted(BGModel.list_available()))
PY
```

当前可用 BG 包括：
- `kernel`
- `mamba`
- `ssm`
- `rnn`
- `proportional`
- `conv_mlp`
- `latent_bg`
- `gp_latent_bg`
- `gp_latent_bg_gpytorch`
- `gp_latent_bg_svgp`
- `ncde`

实现入口：
- `src/models/bg/__init__.py`
- `src/models/bg/*.py`

### 3.2 在配置里启用 BG

对于支持 BG 的 TPP 模型（如 `rtpp/rtpp_v2/etas/etas_zhuang/mixer_tpp/nhpp`），在 `config/*.yaml` 中设置：

```yaml
task_type: tpp
model: rtpp_v2
dataset: Geysers

bg_model: kernel
bg_model_cfg:
  d_feature: 1
  scale_init: 300.0
  kernel_type: gamma
```

或：

```yaml
bg_model: mamba
bg_model_cfg:
  d_feature: 1
  d_state: 64
  d_model: 16
  model_type: mamba
  scale_init: 1000.0
  smooth_kernel_size: 3
```

模型构建时会走：
- `BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)`
- 位置见 `src/models/builders/tpp.py`

---

## 4. 如何评估模型预测

评估通常分两层：**训练/测试集 NLL 指标** + **预测统计检验/滑窗评估**。

### 4.1 训练后测试（标准流程）

```bash
python main.py --mode test --model rtpp_v2 --checkpoint_dir <你的ckpt目录>
```

会输出测试指标 JSON（如）：
- `metrics_test_best_1.json`
- `metrics_test_last_1.json`

相关入口：
- `src/cli/workflows.py`
- `src/cli/common.py`

### 4.2 预测采样与可视化

仓库提供了预测脚本与 notebook：
- `notebooks/forecasting_induced_eq.ipynb`
- `notebooks/forecasting_induced_eq_triplet.ipynb`
- `notebooks/forecasting_induced_eq_triplet_multi_bg_compare.ipynb`


### 4.3 统计评估（Number/Magnitude Test）

可用模块：
- `src/utils/catalog_tests.py`

常用接口：
- `run_number_test_result(...)`
- `run_magnitude_test_result(...)`

用途：
- Number Test：比较预测与观测的事件数分布
- Magnitude Test：比较预测与观测的震级分布

### 4.4 滑动窗口预测评估

可用模块：
- `src/utils/forecast_eval.py`
- `src/utils/forecast_sliding.py`
- `src/utils/forecast_eval_helpers.py`

能力包括：
- 滑动窗口 forecast 轨迹评估
- 覆盖率（PI coverage）
- 观测 vs 预测散点
- 误差随时间变化图

---

