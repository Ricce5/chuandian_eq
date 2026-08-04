# RTPP-v2 注水背景采样与数学一致性审计

审计日期：2026-08-03  
范围：`src/models/tpp/recurrent/model_v2.py` 及其实际使用的继承采样、背景 NHPP 和序列构造代码。

修复状态：本报告列出的实现问题已在本次工作区修复，尚未提交。针对性回归测试
`tests/model/test_rtpp_v2_sampling.py` 与既有 RTPP 时间 NLL / 背景 / 公共 TPP 测试共 36 项已通过。

## 结论

`RTPP-v2` 的基本建模与竞争风险（race）采样思路是正确的：若触发部分的条件 hazard 为 `h`，注水驱动背景强度为 `f`，训练目标的时间部分对应总强度

\[
\lambda(t_i+u\mid\mathcal H_{t_i})=h_i(u)+f(t_i+u).
\]

先抽触发候选时间 `T_h`，再在 `[t_i, t_i + T_h]` 中抽背景 NHPP 的首事件 `T_f`，并返回 `min(T_h, T_f)`，数学上确实会产生该总强度。

但当前代码包含多处实现不一致。其中“带历史的多步预测 RNN hidden 状态错误”会影响所有 RTPP-v2 预测；缓存边界错误会使注水背景在缓存范围外被静默置零。若启用注水 context 融合，还存在首事件条件分布和震级分布与训练不一致的问题。因此，不应把当前版本生成的带 `past_seq` 模拟结果视作已从训练模型精确采样。

## 范围与调用链

`RecurrentTPPV2` 本身没有覆写 `sample()`；它通过 `RecurrentTPPSamplingMixin` 继承采样实现：

- 触发/背景竞争：`src/models/tpp/recurrent/sampling.py::sample_next_inter_time`
- 自回归生成：`src/models/tpp/recurrent/sampling.py::sample`
- RNN history 编码与一步更新：`src/models/tpp/common/recurrent_blocks.py`
- 注水背景 NHPP 反演：`src/models/bg/base.py::sample_nhpp_inverse`
- 输出 batch 构造及 compensator：`src/models/tpp/common/sequence_ops.py`

因此，以下问题虽然有些不在 `model_v2.py` 文件中，但会直接作用于 RTPP-v2 的实际采样行为。

## 本次修复摘要

| 审计问题 | 修复位置 | 修复方式 |
| --- | --- | --- |
| 历史 hidden 多推进一次零输入 | `recurrent_blocks.py`、`model_v2.py` | 新增按 `end_idx` 恢复“最后真实事件后”的 hidden；RTPP-v2 sampling 使用该 hidden。 |
| 背景缓存静默截断 | `bg/base.py` | 对原始 `t0/t1` 严格范围检查，越界抛出 `ValueError`。 |
| 背景查询超出预测窗口 | `recurrent/sampling.py` | 在 NHPP 调用前按每条路径的 remaining duration 截断触发候选。 |
| 条件首事件注水 context 时刻错误 | `model_v2.py` | time decoder 固定查询最后事件时间；`lower_bound` 仅用于条件等待时间。 |
| b-value 注水 context 缺失 | `model_v2.py`、`sampling.py` | 增加 sampling b-context helper，并在 model-b 震级采样时使用。 |
| 绝对 arrival time 与空历史 | `sequence_ops.py`、`sampling.py` | 输出 batch 加回 `t_start`；零事件 history 使用区间起点作为条件起点。 |
| 总 compensator 缺背景 | `sequence_ops.py` | 叠加背景积分，并使用 RTPP-v2 的 time context。 |

## 数学核对

### 训练目标

令触发 inter-time 分布的 survival 为 `S_h(u)=exp(-H(u))`、hazard 为 `h(u)`。在事件时刻，RTPP-v2 计算触发时间对数似然；背景项 `BGModel.nll_change()` 再加入

\[
-\sum_i\log\left(1+\frac{f(t_i)}{h_i(\tau_i)}\right)+\int f(t)\,dt.
\]

与触发项相加后为

\[
\sum_i\log[h_i(\tau_i)+f(t_i)]-\sum_iH_i(\tau_i)-\int f(t)\,dt,
\]

即总强度 `h + f` 的标准点过程对数似然（当 `time_weight=bg_weight=1` 时）。当前 RTPP-v2 配置中这两个权重默认均为 1；`bg_norm_weight` 是额外正则化，不改变上述基础似然的形式。

### 无条件 race 采样

若 `T_h` 的 survival 是 `S_h(u)`，背景 NHPP 首事件的 survival 是

\[
S_f(u)=\exp\left[-\int_0^u f(t_i+s)\,ds\right],
\]

则

\[
\Pr(\min(T_h,T_f)>u)=S_h(u)S_f(u),
\]

其 hazard 正是 `h_i(u)+f(t_i+u)`。`sampling.py::sample_next_inter_time` 的基本逻辑符合这个推导。

### 有历史 censoring 的条件分布

若最后一个历史事件在 `t_i`，预测从 `t_c=t_i+d` 开始，正确的触发残余等待时间应满足

\[
\Pr(T_h-d>u\mid T_h>d)=\frac{S_h(d+u)}{S_h(d)}.
\]

背景部分由于 NHPP 独立增量，从 `t_c` 重新抽首事件即可。因而，只要触发分布的参数仍是最后事件时刻确定的参数，`sample_conditional(lower_bound=d) - d` 加上从 `t_c` 开始的背景 race 是正确的。

## 已确认问题

### P0：历史 RNN hidden 被末尾零输入推进，第二个未来事件起偏离模型

位置：

- `src/models/tpp/recurrent/sampling.py:313-328`
- `src/models/tpp/common/recurrent_blocks.py:234-244`

`Batch` 对一个含 `N` 个事件的历史序列保存 `N+1` 个 inter-time；最后一个是 censoring/survival 区间。`input_mask` 将该位置的特征乘为零，但 `get_context_and_hidden()` 仍把完整长度输入 GRU/RNN，再返回 `hidden`。

与此同时，返回的 `context[:, -1]` 是最后一个真实事件后的状态。采样器把这个 context 用作第一件未来事件的分布状态，却把“额外经过一次零输入”的 `hidden` 用于接收第一件未来事件后的一步递推。对于 GRU/RNN，零输入不是 identity（存在 bias），所以 `current_state` 与 `current_hidden` 不表示同一历史状态。

影响：第一件未来事件的时间分布通常仍使用正确 context；从第一件未来事件更新完成后，第二件及以后事件的时间和震级条件分布都偏离训练模型。此问题不依赖 `bg_model`，故影响所有使用 `past_seq` 的 RTPP-v2 滑窗/预测模拟。

数值复现（真实 `RNNTPPBackbone`、随机初始化 GRU）：

| 检查量 | 实测最大绝对误差 |
| --- | ---: |
| 返回 `hidden` 与仅处理真实事件后 hidden 的差 | 0.11795 |
| 对同一新事件输入得到的下一 state 的差 | 0.06214 |
| `context[:, -1]` 与正确最后事件 state 的差 | 0.0 |

修复要求：采样起点的 hidden 必须是“最后一个真实事件输入后”的 hidden。对于当前 sampling 用途（history batch 固定为 1），可只对有效事件输入重跑 RNN，或让 backbone 显式返回按 `end_idx` 截断后的 hidden；不可复用完整 padded/censoring 序列的 final hidden。

### P0：背景缓存终点被静默截断，缓存外注水强度被当作零

位置：`src/models/bg/base.py:535-589`。

`sample_nhpp_inverse()` 先执行

```python
t1_max = t1_b.max().clamp_max(ts_times_full[-1])
```

再使用 `t1_max` 做范围断言。因此当实际 `t1_b` 超出缓存终点时，断言不会失败；而 `cif_at_rel(t1_rel)` 会在最后一个格点饱和，令缓存外累计强度不再增加。这等价于把缓存外背景强度错误设为零。

数值复现：以常数背景 `f=1` 缓存于 `[0,10]`，调用 `sample_nhpp_inverse(B=100000, t0=9, dt=10)`：

| 量 | 实测 | 正确值 |
| --- | ---: | ---: |
| 返回 `tau=10` 的比例（被当作“10 内无事件”） | 0.3712 | `exp(-10)=0.0000454` |
| 当前实现实际对应的无事件概率 | 约 `exp(-1)` | `exp(-1)=0.3679` |

作为对照，完全位于缓存内的 `[8,10]` 调用实测无事件概率为 0.1370，接近 `exp(-2)=0.1353`，说明反演器在定义域内的逻辑正确。

修复要求：检查未经截断的 `t0_b.min()` 与 `t1_b.max()` 都在缓存范围内；超出时直接抛出带有真实区间的 `ValueError`。不应 clamp 后继续采样。

### P1：NHPP 查询在裁剪到剩余预测窗口之前发生

位置：

- `src/models/tpp/recurrent/sampling.py:41-47`
- `src/models/tpp/recurrent/sampling.py:395`

背景 NHPP 使用完整触发候选 `T_h` 作为窗口长度；直到背景抽样结束后，外层才把结果裁剪到总 duration，且裁剪上限是完整 duration 而不是本行样本的 remaining duration。

当模拟已接近预测终点、但 `T_h` 很大时，代码会请求预测窗口之外的注水数据。这与上一个缓存截断问题相互掩盖：缓存刚好覆盖预测窗口时，当前代码通常仍能给出窗口内正确的“是否有首事件”结果，但实现依赖缓存外强度被静默截成零，且无法安全改成严格的越界检查。

修复要求：在调用背景 sampler 之前计算

\[
d_{\mathrm{query}}=\min(T_h^{\mathrm{residual}},\;D_{\mathrm{remain}}),
\]

仅在 `[t_current, t_current + d_query]` 抽背景首事件。若两者均未在窗口内发生，直接生成 survival/censoring，而不是创建窗口外的伪事件状态。

### P1：`time_use_bg_context=True` 时，历史条件化的首个未来事件使用了错误时刻的注水 context

位置：`src/models/tpp/recurrent/model_v2.py:271-297`。

训练阶段的 `_context_query_times()` 对最后一个 survival context 查询最后事件时间 `t_i`；这意味着该 inter-time 分布参数是由 `z(t_i)` 确定的。采样阶段，如果存在历史 censoring `d`，`_get_sampling_time_context()` 却将 query time 改为 `t_i+d=t_c`，随后仍调用 `sample_conditional(lower_bound=d)`。

于是代码采的是

\[
p_{\theta(z(t_c))}(T\mid T>d),
\]

但训练模型定义的是

\[
p_{\theta(z(t_i))}(T\mid T>d).
\]

二者一般不相等。若想让触发 hazard 在 inter-event 区间内连续地随注水变化，则必须定义 `h(u, z(t_i+u))` 并对其积分；仅在预测起点替换一次 decoder 参数不是该模型的似然或采样。

修复要求：对当前离散 context 模型，首个条件事件的 time context 应查询 `t_i`，而背景 NHPP 的绝对起点独立地使用 `t_c`。现有实验配置的 `time_use_bg_context=false`，所以该问题目前未触发，但功能启用前必须修复。

### P1：`b_use_bg_context=True` 时采样震级忽略背景 context

位置：

- 训练：`src/models/tpp/recurrent/model_v2.py:380-383`
- 采样：`src/models/tpp/recurrent/sampling.py:401-406`

训练时 `get_magnitude_dist(context, batch=batch)` 传入 batch，可通过 `_get_b_context()` 融合背景 context；采样仅传入 `current_state`，没有提供等价的采样时 query/context。因此在 `predict_b=True` 且 `b_use_bg_context=True` 时，生成的 Gutenberg--Richter `b` 与训练分布不一致。

当前配置中 `b_use_bg_context=false`，故该问题暂未触发。

### P2：默认返回 `Batch` 的绝对 arrival time 错误

位置：`src/models/tpp/common/sequence_ops.py:78-87`。

`build_sample_batch()` 写入

```python
arrival_times = inter_times.cumsum(-1)
```

却没有加 `t_start`。例如 `t_start=10`，返回 batch 的 arrival times 是 `[1, 5]`，而 metadata `t_start=[10]`；转换为 `Sequence` 后才会从 inter-times 重建成 `[11]`。

因此 `return_sequences=True` 的主要预测路径通常得到正确 absolute time，但默认 `return_sequences=False` 返回的 `Batch.arrival_times` 不符合 `Batch.from_list()` 和其他模型的约定。

### P2：空历史序列无法采样

位置：`src/models/tpp/recurrent/sampling.py:328`。

零事件历史（只含一个 survival inter-time）是 `Sequence` 支持的合法对象，但这里无条件索引 `past_seq.arrival_times[-1]`。预测窗口早于首个观测事件时会失败。当前滑窗入口从首个事件之后开始，通常避开了该分支。

### P2：带背景时的 compensator/time-rescaling 诊断不对应总过程

位置：

- `src/models/tpp/recurrent/sampling.py:480-489`
- `src/models/tpp/common/sequence_ops.py:91-125`

共享 `evaluate_compensator_from_model()` 只用 `get_context()` 与 inter-time decoder 的 `log_survival`，因此只累计触发项 `H`：

- 没有加入背景补偿 `\int f(t)dt`；
- RTPP-v2 启用 `time_use_bg_context` 时，也没有先经过 `_get_time_context()`。

它不能用于带注水背景模型的总过程校准、rescaling QQ 图或补偿曲线解释。

## 验证过的正确部分

使用仓库中的实际 `ProportionalBGModel` 和 `RecurrentTPPSamplingMixin.sample_next_inter_time()`，构造常数触发 hazard `h=2` 与常数背景 `f=1`，并确保背景缓存覆盖整个采样范围。理论上 `T` 应为 rate 3 的指数分布。

| 统计量 | 实测 | 理论 |
| --- | ---: | ---: |
| `E[T]` | 0.3342 | 0.3333 |
| `P(T>0.5)` | 0.2243 | `exp(-1.5)=0.2231` |

这确认问题不在“先抽触发候选、再在该时间窗内抽背景 NHPP、取最小值”的基本构造，而在其状态、边界、context 与输出实现。

数值复现使用 `fa_mamba_clean` Conda 环境、PyTorch `2.8.0+cu128`，随机种子分别固定为 7、3、19。结果为 Monte Carlo 检查，不用于模型性能评估。

## 对现有实验的影响

现有 `rtpp_v2_multi_bg_norm_0.2` 配置中，`time_use_bg_context=false`、`b_use_bg_context=false`。因此两个 fusion 专属问题未启用；注水背景仍以显式加法强度 `f` 进入总过程。

但这些配置的滑窗预测均使用 `past_seq`，故 P0 hidden 问题会影响每条模拟路径的第二个及以后事件。若当前滑窗缓存和指标由该版本代码生成，修复后应重新生成模拟缓存、均值、分位区间及派生的 MAE/RMSE/CRPS/覆盖率。此前的模型比较不应被解释为“已从各 checkpoint 精确采样”的结果。

滑窗评估入口 `src/utils/forecast_eval.py` 会在模拟前缓存 `bg_cache_seq`，通常会传入完整序列，因而只要完整注水序列覆盖窗口，背景缓存边界问题通常不触发。`forecasting.py` 的单次预测路径没有传入 `bg_cache_seq`；对于带 `bg_model` 的 checkpoint，它依赖既存缓存，容易直接失败或误用旧缓存。

## 推荐修复顺序

1. 修复 P0：返回/恢复最后真实事件后的 RNN hidden，并新增 state-equivalence 测试。
2. 同时修复 P0/P1 边界：先按每条样本的 remaining duration 截断背景查询窗口；背景 sampler 对真实越界严格报错。
3. 修复注水 context 条件化语义：首个条件事件的 time decoder 固定使用最后事件时刻 context；为 `b` 增加对等的 sampling context helper。
4. 修复输出 `Batch.arrival_times`，并支持空 history。
5. 为带背景模型实现总 compensator，包含 `H + \int f`，并使用与 NLL 相同的 time context。
6. 修复后重新生成受影响的 RTPP-v2 预测缓存和报告。

## 必需回归测试

1. 常数 `h`、常数 `f` 的 race 采样检验：均值、survival 或 KS 检验匹配 `h+f`。
2. 给定历史 censoring `d` 的首事件检验：触发 residual 与背景 NHPP residual 同时满足条件生存函数。
3. history state 等价性：给历史加一个已知未来事件后，`sample` 的一步 state 必须等于以“历史 + 该事件”整段重算得到的最后 context/hidden。
4. 缓存边界：恰好覆盖窗口时成功；任一 `t0`/`t1` 越界时明确报错；不得静默截断。
5. `time_use_bg_context`：验证条件首事件 query 使用最后事件时间；验证随后每个新事件使用自己的事件时间。
6. `b_use_bg_context`：训练和采样在相同 history/query 时给出相同 `b` 分布参数。
7. `build_sample_batch(t_start != 0)`：返回 `Batch.arrival_times` 为绝对时刻，且与 `to_list()` 一致。
8. 总 compensator：常数背景下其斜率比无背景触发 compensator 多出常数 `f`。
