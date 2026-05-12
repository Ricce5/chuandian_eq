# 新增诱发地震目录（Induced Triplet Catalog）操作指南

本文给出在当前项目中新增诱发地震目录的完整流程，覆盖：
- 单目录接入（`InducedTripletBase`）
- 多目录聚合接入（`InducedTripletGroupedCatalog`）
- 数据准备规范、注册、配置、验证与常见报错排查

---

## 1. 先理解当前框架约定

当前诱发地震目录走 `src/catalogs/induced_triplet_base.py` 的统一逻辑，要求：

1. 数据放在 `data/<DatasetName>/processed/`
2. 至少包含 3 个文件：
   - `<DatasetName>_eq_processed.csv`
   - `<DatasetName>_inj_<resample_freq_min>min_processed.csv`
   - `<DatasetName>_summary.json`
3. `summary` 必含键（缺一不可）：
   - `resample_freq_min`
   - `mc`
   - `inj_fill_policy`
   - `is_upsample`
   - `start_time_iso`
   - `end_time_iso`

代码位置参考：
- `src/catalogs/induced_triplet_base.py`
- `src/utils/catalog_pathing.py`
- 示例：`src/catalogs/geysers.py`、`src/catalogs/st1.py`

---

## 2. 准备数据文件

以数据集名 `MyField` 为例，目录应为：

```text
data/MyField/processed/
├── MyField_eq_processed.csv
├── MyField_inj_60min_processed.csv
└── MyField_summary.json
```

> 其中 `60` 对应 `summary` 里的 `resample_freq_min`。

### 2.1 EQ 文件建议列

`InducedTripletBase` 内置了列别名，常见可识别列：
- 时间列（其一即可）：`time_iso`, `ts`, `time`, `timestamp`, `origin_time`, `Date`
- 震级列：`magnitude`, `mag`, `Magnitude`, `ML`, `Mw`
- 经纬深：`latitude/lat`, `longitude/lon/lng`, `depth/depth_m/depth_km`

### 2.2 注水文件建议列

- 时间列：`time_iso`, `ts`, `time`, `timestamp`, `Date`
- 注水速率列：`inj_rate_m3_min` 或 `inj_rate`

---

## 3. 新增“单目录”Catalog 类

在 `src/catalogs/` 新建文件，例如 `src/catalogs/myfield.py`：

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.data import Catalog, default_catalogs_dir
from .induced_triplet_base import InducedTripletBase


@Catalog.register(name="MyField-Standard")
class MyFieldStandard(InducedTripletBase):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "MyField" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
        end_ts: Optional[Union[pd.Timestamp, str]] = None,
        train_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        val_start_ts: Optional[Union[pd.Timestamp, str]] = None,
        test_start_ts: Optional[Union[pd.Timestamp, str]] = None,
    ):
        super().__init__(
            dataset_name="MyField",
            root_dir=root_dir,
            data_dir=data_dir,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
            end_ts=end_ts,
            train_start_ts=train_start_ts,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
```

### 3.1 注册导入

编辑 `src/catalogs/__init__.py`，把新模块加入导入列表（用于触发注册）：

```python
from . import (
    ...,
    myfield,
)
```

并补充 `__all__`。

---

## 4. 在配置文件中使用

在你的模型配置（如 `config/rtpp_v2.yaml`）中设置：

```yaml
dataset: MyField
catalog_cfg:
  mag_completeness: 1.5
  normalize: true
  freq: 1h
  # 可选时间切分边界
  # end_ts: "2021-12-31T23:00:00"
  # train_start_ts: "2018-01-01T00:00:00"
  # val_start_ts: "2020-01-01T00:00:00"
  # test_start_ts: "2021-01-01T00:00:00"
```

说明：
- `dataset: MyField` 会优先解析为 `MyField-Standard`。
- 若 `mag_completeness` 为空，默认取 `summary.mc`。
- 缓存目录采用哈希子目录（在 `data/MyField/catalogs/<hash>/` 下）。

### 4.1 如何在 config 中切换到新增数据集

以 `config/rtpp_v2.yaml` 为例，最小改动只有两步：

1. 把 `dataset` 改为你的新增目录名（不带 `-Standard`）：
   ```yaml
   dataset: MyField
   ```
2. 在 `catalog_cfg` 中填入数据相关参数（见下一节字段说明）。

注意：
- 代码会优先尝试注册名 `MyField-Standard`，再尝试 `MyField`。
- 因此你的 catalog 类注册名建议保持 `@Catalog.register(name="MyField-Standard")`。

### 4.2 `catalog_cfg` 可配置字段（单目录 `InducedTripletBase`）

下表字段会通过 `build_tpp_catalog_init_kwargs(...)` 传给 catalog 构造函数（仅保留该类签名中支持的字段）：

- `mag_completeness`: `float | null`
  - 含义：震级完备阈值（Mc）。
  - 默认：`null` 时使用 `summary.mc`。
  - 要求：建议与训练/评估设定一致；过高会导致样本过少。

- `normalize`: `bool`
  - 含义：是否归一化 time-series 特征。
  - 默认：`true`（取类默认值）。
  - 要求：训练与测试尽量保持一致。

- `freq`: `str`（如 `1h`, `1D`, `30min`）
  - 含义：模型时间单位（`pd.Timedelta(freq)`）。
  - 要求：必须是正时间间隔，否则报 `freq must be positive`。

- `end_ts`: `str | null`（ISO 时间）
  - 含义：截断目录终止时间。
  - 默认：`null` 表示用 `summary.end_time_iso`。
  - 要求：会被裁剪到 `[start_time_iso, end_time_iso]` 区间内，且必须大于起点。

- `train_start_ts` / `val_start_ts` / `test_start_ts`: `str | null`
  - 含义：切分起始时间。
  - 默认：`null` 时按内部规则自动切分。
  - 要求：最终会强制满足 `train <= val <= test <= end`。

- `data_dir`: `str`（可选）
  - 含义：显式指定数据目录。
  - 默认：自动用 `data/<dataset>`。
  - 要求：目录下必须有 `processed` 三文件。

- `root_dir`: `str`（可选）
  - 含义：显式指定 catalog 缓存根目录。
  - 默认：`data/<dataset>/catalogs`。
  - 要求：目录可写；缓存将按哈希子目录创建。

### 4.3 `catalog_cfg` 可配置字段（多目录 `InducedTripletGroupedCatalog`）

对 grouped family（如 `CooperBasin-Standard`）常用字段：

- `split_groups`: `{train, val, test}`
  - 每个 split 可填字符串或列表（支持 `+`/`,` 分隔）。
  - 要求：
    - 必须包含 `train`/`val`/`test` 三个键；
    - 每个 split 非空；
    - 不同 split 间数据集不能重复（必须互斥）。

- `mag_completeness`: `float | null`
  - 全局 Mc，作用于所有子目录。

- `mag_completeness_map`: `dict[str, float]`（仅代码调用常用）
  - 按子目录细分 Mc。
  - 要求：若未设置全局 `mag_completeness`，则被选中的每个子目录都必须有值。

- `dataset_aliases`: `dict[str, str]`（通常在类里定义）
  - 用于把简写 token 映射到标准子目录名。

- `normalize` / `freq` / `root_dir` / `data_dir`
  - 语义与单目录一致。

### 4.4 字段配置必须满足的基础要求

1. `dataset` 必须能解析到已注册 catalog（建议 `<Dataset>-Standard`）。
2. `data/<Dataset>/processed/` 下三文件齐全且命名严格匹配：
   - `<Dataset>_eq_processed.csv`
   - `<Dataset>_inj_<resample_freq_min>min_processed.csv`
   - `<Dataset>_summary.json`
3. `summary` 必含 6 个键：
   - `resample_freq_min`, `mc`, `inj_fill_policy`, `is_upsample`, `start_time_iso`, `end_time_iso`
4. `catalog_cfg.freq` 与 `summary` 时间跨度组合后应保证有效持续时长（`t_end > 0`）。
5. 若命中历史缓存但元信息不一致，会报 `FileExistsError`（需清缓存或改参数生成新哈希目录）。

---

## 5. 验证是否接入成功

### 5.1 验证注册名

```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print("MyField-Standard" in Catalog.list_available())
PY
```

### 5.2 验证可实例化

```bash
python - <<'PY'
import src.catalogs
from src.utils.tpp_experiments import load_tpp_catalog
from pathlib import Path

ds, registry_name, kwargs = load_tpp_catalog(
    "MyField",
    base_dir=Path("data") / "MyField",
    catalog_cfg={"normalize": True},
    candidates=["MyField-Standard", "MyField"],
)
print("registry:", registry_name)
print("train/val/test lens:", len(ds.train[0]), len(ds.val[0]), len(ds.test[0]))
print("root_dir:", ds.root_dir)
PY
```

---

## 6. 新增“多目录聚合”Catalog（可选）

如果你希望把多个子目录组合成一个 family（类似 `CooperBasin-Standard`），可继承 `InducedTripletGroupedCatalog`。

示例骨架：

```python
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from src.data import Catalog, default_catalogs_dir
from .induced_triplet_grouped import InducedTripletGroupedCatalog

MYFIELD_DATASETS = ("MyA", "MyB", "MyC")
MYFIELD_DEFAULT_SPLIT_GROUPS = {
    "train": ("MyA",),
    "val": ("MyB",),
    "test": ("MyC",),
}
MYFIELD_ALIASES = {
    "a": "MyA",
    "b": "MyB",
    "c": "MyC",
}


@Catalog.register(name="MyFieldFamily-Standard")
class MyFieldFamilyStandard(InducedTripletGroupedCatalog):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "MyFieldFamily" / "catalogs",
        data_dir: Union[str, Path, None] = None,
        split_groups: Optional[Mapping[str, Union[str, Sequence[str], Path]]] = None,
        mag_completeness: Optional[float] = None,
        normalize: bool = True,
        freq: str = "1h",
    ):
        super().__init__(
            family_name="MyFieldFamily",
            valid_datasets=MYFIELD_DATASETS,
            default_split_groups=MYFIELD_DEFAULT_SPLIT_GROUPS,
            root_dir=root_dir,
            data_dir=data_dir,
            split_groups=split_groups,
            dataset_aliases=MYFIELD_ALIASES,
            mag_completeness=mag_completeness,
            normalize=normalize,
            freq=freq,
        )
```

---

## 7. 常见报错与排查

### 7.1 `FileNotFoundError: Dataset directory not found`

原因：`dataset` 名字与 `data/<dataset>` 目录不一致（常见拼写问题）。

排查：
```bash
ls data
```

### 7.2 `FileExistsError: A different catalog already exists in ...`

原因：同一哈希目录已存在旧缓存，但当前参数/元数据已变化。

处理方式（任选其一）：
1. 删除冲突缓存目录后重建（最常用）
2. 修改 `catalog_cfg`（例如 `freq`、切分时间）让其生成新哈希目录

### 7.3 `Summary file not found` / `Missing required summary keys`

原因：`processed/<Dataset>_summary.json` 缺失或字段不全。  
按第 1 节要求补齐后重试。

---

## 8. 建议实践

1. **先用 notebook 完成预处理**，确保 `eq/inj/summary` 三文件齐全。
2. **单目录先跑通再做 grouped**，降低排障成本。
3. **固定 `freq` 和时间边界策略**，避免频繁触发缓存冲突。
4. 每次改完 catalog 类后，先跑“注册验证 + 实例化验证”两个最小脚本。

---

## 9. 通过 `notebooks/induced_seismicity_analysis.ipynb` 可视化验证

接入完成后，建议直接用该 notebook 做端到端可视化检查，确认：
- catalog 能被正确加载；
- 注水与地震时间序列可正常提取；
- 峰值识别、单目录统计与批量对比图可正常输出。

### 9.1 启动 notebook

```bash
jupyter lab
```

打开：
- `notebooks/induced_seismicity_analysis.ipynb`

### 9.2 关键参数建议（最小改动）

在 notebook 参数区将数据集相关变量改为你的数据集，例如：
- `DATASET` 设为你的目录名（如 `MyField`）
- `CATALOG_CFG` 中 `mag_completeness`、`normalize` 与你的数据处理配置一致
- 批量分析单元中的数据集列表加入你的目录（或 family 成员）

### 9.3 重点观察输出

至少确认以下图或指标正常生成：
- 单目录事件-注水时间序列图
- 注水峰值检测结果图
- 单目录统计摘要（峰值、事件率、时间范围）
- 批量对比导出图（all triplet datasets）

若输出异常（空序列、测试集为 0、图像全空），优先检查：
1. `summary` 时间边界与 `train/val/test` 切分时间是否冲突；
2. `catalog_cfg.end_ts` 是否裁剪过度；
3. `mag_completeness` 是否过高导致事件几乎被过滤。

---

## 10. 最小上线清单（Checklist）

- [ ] `data/<Dataset>/processed/` 三文件存在
- [ ] `summary` 包含 6 个必需键
- [ ] 新增 `src/catalogs/<dataset>.py` 并 `@Catalog.register(...)`
- [ ] 在 `src/catalogs/__init__.py` 导入新模块
- [ ] 配置文件中 `dataset: <Dataset>`
- [ ] `Catalog.list_available()` 中可见 `<Dataset>-Standard`
- [ ] `load_tpp_catalog(...)` 可实例化并可得到 `train/val/test`
