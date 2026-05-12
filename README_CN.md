# em_eqf（中文说明）

地震目录深度学习预测框架，覆盖：
- 分类/回归任务（基于滑动窗口样本）
- 时间点过程（TPP）任务（事件序列建模）
- 训练、测试、Optuna 搜索、可视化与评估

## 1. 安装

### 1.1 基础依赖
```bash
pip install -r requirements.txt
pip install -e .
```

### 1.2 可选加速依赖
仅当你的模型/配置需要时再安装：
- 建议优先使用各项目官方预编译轮子（wheels）。
- 推荐顺序：先装 `causal_conv1d`，再装 `mamba_ssm` 和/或 `flash_attn`
- `causal_conv1d`（wheels）：https://github.com/Dao-AILab/causal-conv1d/releases
- `mamba_ssm`（wheels）：https://github.com/state-spaces/mamba/releases
- `flash_attn`（wheels）：https://github.com/Dao-AILab/flash-attention/releases

请按各项目官方安装指引，确保 CUDA / PyTorch 版本匹配。

## 2. 仓库结构

```text
.
├── main.py                 # 主入口（train/test/optuna）
├── forecasting.py          # 采样与预测可视化
├── config/                 # YAML 配置
├── data/                   # 数据目录（raw/processed）
├── notebooks/              # 预处理/分析/绘图
├── checkpoints/            # 训练产物
├── src/
│   ├── catalogs/           # 目录加载器 + 数据集注册
│   │   ├── event_features/ # 事件特征构造（oracle 特征）
│   │   ├── induced_triplet_base.py     # 单个目录的 triplet 加载
│   │   ├── induced_triplet_grouped.py  # 多目录聚合的 triplet 加载
│   │   ├── pathing.py      # 目录路径工具
│   │   └── <dataset>.py    # 各数据集的加载与注册
│   ├── data/               # 数据集对象、预处理、批处理
│   │   ├── preparation.py  # 主要数据准备入口
│   │   ├── preprocessing.py
│   │   ├── tpp_dataset.py  # TPP 数据集
│   │   ├── lstm_loader.py  # LSTM 特定数据加载
│   │   └── batch.py / sequence.py / event_loader.py
│   ├── distributions/      # 统计分布与混合分布
│   │   ├── mixture.py
│   │   └── gamma.py / weibull.py / lomax.py / gutenberg_richter.py
│   ├── features/           # 地震特征工程
│   │   └── seismic_features.py
│   ├── models/             # 模型定义与组件
│   │   ├── builders/       # 模型注册与构建（ModelBuilder）
│   │   ├── core/           # BaseModel / TaskModel / TaskHead 组合基元
│   │   ├── adapters/       # 按任务/领域组织的输入适配器
│   │   ├── base_model.py / task_model.py / heads.py / input_adapters.py
│   │   ├── bg/             # 背景率模型（kernel/mamba/rnn 等）
│   │   ├── tpp/            # TPP 模型（etas/rtpp/mixer_tpp/nhpp 等）
│   │   ├── transformer/    # Transformer 编码器（已弃用）
│   │   ├── mamba/          # Mamba 相关模块
│   │   ├── mha/            # 多头注意力 + 时间旋转嵌入
│   │   ├── ncde/           # Neural CDE（已弃用）
│   │   ├── extractors/     # 表征抽取器（注意力池化、last-step 等）
│   │   └── layers/         # 通用层（MLP、Conv、ReVIN 等）
│   ├── train/              # 训练流程与步骤
│   │   ├── trainer.py      # 训练/验证 + 保存
│   │   ├── config_setup.py # 优化器/调度器/模型配置
│   │   ├── model_routing.py
│   │   ├── scheduler.py
│   │   └── *_train_step.py # 任务专用 train/validate/test
│   └── utils/              # 工具（日志、指标、可视化等）
│       ├── logging_utils.py / metrics.py / viz_sequences.py
│       ├── file_utils.py / registrable.py
│       └── forecast_eval.py / interpretability.py / mask_utils.py / runtime_utils.py
└── tests/                  # 测试
```

## 3. 快速开始

### 3.1 训练
```bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml
```

若未指定 `--config`，默认使用 `config/<model>.yaml`。

指定输出目录：
```bash
python main.py --model mixer_tpp --mode train --checkpoint_dir checkpoints/my_exp
```

若 `train`/`optuna` 未指定 `--checkpoint_dir`，会自动创建：
`checkpoints/<model>_YYYYMMDD-HHMMSS`（例如 `checkpoints/mixer_tpp_20260413-153045`）。

### 3.2 测试
默认使用 `best_model_{trial_index}.pth`：
```bash
python main.py \
  --model mixer_tpp \
  --mode test \
  --checkpoint_dir checkpoints/my_exp \
  --ckpt_select best
```

若 `test` 未指定 `--checkpoint_dir`，会自动选择包含 `last_model_1.pth` 的最新 `checkpoints/<model>_YYYYMMDD-HHMMSS` 目录。

若未指定 `--ckpt_select`，默认使用 `best`。

使用指定 epoch 的 checkpoint：
```bash
python main.py \
  --model mixer_tpp \
  --mode test \
  --checkpoint_dir checkpoints/my_exp \
  --ckpt_select epoch \
  --ckpt_epoch 10
```

注意：epoch 测试需要对应的 epoch checkpoint（如 `epoch_10_model_1.pth`）已存在，请先配置并启用周期保存。

分类任务可手动阈值：
```bash
python main.py --model clf_mixer_attnpl_t --mode test --threshold 0.5
```

### 3.3 Optuna 搜索
```bash
python main.py --model mixer_tpp --mode optuna --config config/mixer_tpp.yaml
```

## 4. CLI 参数（main.py）

- `--mode`: `train` / `test` / `optuna`
- `--model`: 模型名（需与配置中的 `model` 一致）
- `--config`: 配置路径，默认 `config/<model>.yaml`
- `--checkpoint_dir`: 训练输出或测试输入目录
- `--trial_index`: checkpoint 序号（默认 `1`）
- `--ckpt_select`: `best` / `last` / `epoch`
- `--ckpt_epoch`: 当 `--ckpt_select epoch` 时必须提供
- `--threshold`: 分类测试阈值（覆盖 checkpoint 中的阈值）
- `--no_val_threshold`: 不使用 checkpoint 中的验证阈值

说明：
- `train`/`optuna` 模式若未指定 `--checkpoint_dir`，会自动创建 `checkpoints/<model>_YYYYMMDD-HHMMSS`
- `test` 模式若未指定 `--checkpoint_dir`，会自动选取最近的对应模型目录

## 5. 配置系统

- 配置文件位于 `config/`
- 使用 `OmegaConf` 读取
- 常见字段：`model`, `dataset`, `task_type`, 优化器/调度器参数、模型结构参数等

推荐起步配置：
- `config/mixer_tpp.yaml`
- `config/clf_mixer_attnpl_t.yaml`
- `config/clf_rnn.yaml`
- `config/reg_mixer_attnpl_t.yaml`
- `config/reg_rnn.yaml`
- `config/etas.yaml`
- `config/etas_zhuang.yaml`

## 6. 数据

### 6.1 分类/回归数据
常见路径：`data/<dataset>/raw/*.csv`

主要字段：
- `t`
- `Magnitude`
- `Latitude`
- `Longitude`
- `Depth`
- `dt`（预处理中由 `t` 计算）

### 6.2 TPP 数据
TPP 管线通过注册系统解析 `<dataset>-Standard`：
- 例：`dataset: ChuanDian` -> `ChuanDian-Standard`

列出已注册目录：
```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print(sorted(Catalog.list_available()))
PY
```

当前可用数据集：
`AZDX`, `Basel`, `CB_HAB1a`, `CB_HAB1b`, `CB_HAB4`, `ChinaArray`, `ChuanDian`, `CooperBasin`, `FORGE2022`,
`PNR`, `PNR_1z`, `PNR_2`, `QTMSaltonSea`, `QTMSanJacinto`, `SCEDC`, `SSFS`, `SSFS1993`, `SSFS2000`,
`SSFS2003`, `SSFS2004`, `SSFS2005`, `St1-2018`, `St1-2020`, `White`。

### 6.3 添加新的 TPP 数据集
1. 在 `data/<YourDataset>/processed/` 下准备处理后的文件：
   - `<YourDataset>_eq_processed.csv`
   - `<YourDataset>_inj_<resample_freq_min>min_processed.csv`（诱发地震 triplet 数据）
   - `<YourDataset>_summary.json`（需包含 `resample_freq_min`, `mc`, `inj_fill_policy`, `is_upsample`, `start_time_iso`, `end_time_iso`）
   - 注意：诱发地震 triplet 的预处理遵循 Oracle 数据格式（EQ/Inj/Head）及 Oracle 风格事件特征构建。
2. 在 `src/catalogs/` 实现并注册一个 catalog 类：
   - 使用 `@Catalog.register(name="<YourDataset>-Standard")`
   - 诱发地震 triplet 数据请继承 `InducedTripletBase`（示例：`src/catalogs/st1.py`）
3. 在 `src/catalogs/__init__.py` 中导入新的 catalog 模块以触发注册。
4. 在配置文件中设置 `dataset: <YourDataset>`，管线会自动解析为 `<YourDataset>-Standard`。
5. 验证是否可被识别：
   ```bash
   python - <<'PY'
   import src.catalogs
   from src.data.catalog import Catalog
   print(sorted(Catalog.list_available()))
   PY
   ```

## 7. 可用模型

当前已注册模型（可通过 `ModelBuilder.list_available()` 获取）：

`btpp`, `classifier`, `classifier_se`, `classifier_stm`, `classifier_stm_s`, `classifier_tm_s`, `clf_attnpl`, `clf_attnpl_t`, `clf_mixer_attnpl_t`, `clf_rnn`, `clf_tm_attnpl`, `clf_tm_attnpl_t`, `clf_tm_cv_attnpl_t`, `etas`, `etas_zhuang`, `lstm`, `mhp`, `mixer_tpp`, `mtpp`, `nhpp`, `njdtpp`, `oracle`, `reg_attnpl`, `reg_mixer_attnpl_t`, `reg_rnn`, `rtpp`, `rtpp_v2`, `thp`, `thp_deltat`

适用的诱发地震点过程模型（常用）：`etas`, `etas_zhuang`, `rtpp`, `rtpp_v2`, `oracle`。
- `etas` / `etas_zhuang` / `rtpp` / `rtpp_v2`：基于背景率建模。
- `oracle`：基于特征工程（注水相关事件特征通过 `catalog_cfg.event_feature_builder`）。

`rtpp`（无背景项，RECAST）参考：
Dascher-Cousineau, K., Shchur, O., Brodsky, E. E., & Günnemann, S. (2023).
*Using Deep Learning for Flexible and Scalable Earthquake Forecasting*.
*Geophysical Research Letters, 50*(17), e2023GL103909. https://doi.org/10.1029/2023GL103909

`oracle` 参考：
Schultz, R., & Wiemer, S. (2026).
*Forecasting the Rate of Induced Seismicity as a Neural Temporal Point Process*.
*Journal of Geophysical Research: Machine Learning and Computation, 3*(1), e2025JH001052.
https://doi.org/10.1029/2025JH001052

在 `config/*.yaml` 中添加背景模型：
1. 在模型配置中设置 `bg_model` 与 `bg_model_cfg`（如 `config/rtpp.yaml` 或 `config/etas.yaml`）。
2. 模型主体参数保持不变，仅替换背景模块以便对比。
3. 可选项参考 `config/bg.yaml`（`mamba`, `kernel`, `gp_latent_bg`, `gp_latent_bg_svgp`, `conv_mlp`, `proportional`）。
4. `scale_init` 用于加速收敛，通常建议 `300–1000`。

示例（Mamba 背景）：
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

示例（Kernel 背景）：
```yaml
bg_model: kernel
bg_model_cfg:
  d_feature: 1
  scale_init: 300.0
  kernel_type: gamma
  kernel_size: 256
  normalize_kernel: true
  use_mlp: true
  hidden: 32
  gamma_init_k: 4.0
  gamma_init_beta: 0.08
```

随机森林（`rf`）基线请参考 `notebooks/classifier_baseline.ipynb`。

### 7.1 新增模型流程
1. 在 `src/models/builders/` 中添加 builder：
   - 继承 `ModelBuilder`
   - 使用 `@ModelBuilder.register("<new_model_name>")` 注册
   - 实现 `__call__(self, args, device)` 并返回模型
2. 在 `src/train/model_routing.py` 中添加路由：
   - 将 `<new_model_name>` 放入对应集合（`CLASSIFIER_MODELS`, `REGRESSOR_MODELS`, `TPP_MODELS`, 或 `TPP_M_MODELS`）
3. 新增 `config/<new_model_name>.yaml`：
   - 确保 `model: <new_model_name>`
   - 配置 `task_type` 与必要超参
4. 训练：
   ```bash
   python main.py --model <new_model_name> --mode train --config config/<new_model_name>.yaml
   ```

## 8. 产物与日志

`checkpoints/<exp>/` 常见文件：
- `config.yaml`（运行时配置快照）
- `run.log`（统一日志）
- `best_model_<idx>.pth`
- `last_model_<idx>.pth`
- `epoch_<k>_model_<idx>.pth`（若开启周期保存）
- `tensorboard/`

测试输出：
- `metrics_test_<...>.json`

TensorBoard：
```bash
tensorboard --logdir checkpoints/<exp>/tensorboard --port 6006
```



## 10. Notebook

Notebook 位于 `notebooks/`（项目相对路径：`./notebooks`）：
- `b_t_estimation.ipynb`：时变 b 值估计（主要用于 mixer_tpp）。
- `classifier_analysis.ipynb`：clf_mixer_attnpl_t 分类结果分析与事件重要性。
- `classifier_baseline.ipynb`：特征工程分类器基线（RF）。
- `cumulative_likelihood_over_time.ipynb`：点过程模型的时间累积似然可视化。
- `forecasting.ipynb`：点过程模型生成预测目录。
- `forecasting_induced_eq.ipynb`：诱发地震点过程模型生成预测目录（单个目录）。
- `forecasting_induced_eq_triplet.ipynb`：诱发地震点过程模型生成预测目录（可使用多目录）
- `forecasting_induced_eq_triplet_multi_bg_compare.ipynb`：比较多种诱发地震点过程模型的背景模型。
- `induced_seismicity_analysis.ipynb`：诱发地震探索性分析（直观分析比较注水与地震率关系）。
- `induced_seismicity_analysis_pnr.ipynb`：PNR 数据集专题分析。
- `preprocessing.ipynb`：地震目录数据预处理（不是诱发地震）。
- `preprocessing_geysers.ipynb`：Geysers 数据预处理。（已弃用）
- `preprocessing_hengill.ipynb`：Hengill 数据预处理。（已弃用）
- `preprocessing_induced_eq_data.ipynb`：诱发地震数据预处理（主要是 Oracle 的数据）。
- `preprocessing_pnr.ipynb`：PNR 数据预处理。
- `preprocessing_pnr_nsta.ipynb`：PNR 来源于NSTA的数据预处理。(已弃用)
- `regression_comparison.ipynb`：回归任务对比可视化。
- `regressor_analysis.ipynb`：reg_mixer_attnpl_t 回归结果分析与事件重要性。
- `lstm_regression_permutation_importance.ipynb`：LSTM 回归特征置换重要性分析。
- `tpp_analysis.ipynb`：TPP 模型内部与行为分析。
- `tpp_comparison.ipynb`：多种 TPP 模型对比分析。
- `eval_new_datasets_forecasting.ipynb`：新数据集预测评估流程。
- `tpp_evaluating.ipynb`：基于时间变换定理的时间点过程模型分析。
- 输出图与缓存数据位于 `notebooks/figs/` 与 `notebooks/figs_data/`。

项目根目录额外 notebook：
- `ckpt_utils.ipynb`：按配置条件筛选/搜索 checkpoint 目录。

## 11. 测试

运行所有测试：
```bash
pytest -q
```

运行指定分组：
```bash
pytest -q tests/features
pytest -q tests/data
pytest -q tests/model
```

## 12. 复现实验提示

提高 CUDA 可复现性：
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 13. 相关文档

- `docs/guides/add_induced_triplet_catalog_guide.md`
- `docs/guides/induced_seismicity_forecasting_guide.md`

## 14. 许可证

本项目采用 MIT License，详见 `LICENSE`。
