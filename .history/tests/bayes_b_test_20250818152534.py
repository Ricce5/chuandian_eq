# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("../config/mixer_tpp.yaml")
base_dir = f"../data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
# %%
import math
from typing import Dict, Optional, Union
import torch
import matplotlib.pyplot as plt

LN10 = math.log(10.0)

class BayesianGRBUpdater:
    """
    Event-by-event Bayesian updater for GR b-value with time-variation
    using discounted conjugate Gamma prior on alpha = b * ln(10).

    更新逻辑（逐事件）：
      1) 折扣先验: (a, s) <- (delta*a, delta*s)
      2) 若 m_i >= Mc: a <- a + 1; s <- s + (m_i - Mc)
      3) 后验在 b 上的摘要：
         E[b] = a / (s * ln 10)
         SD[b] = sqrt(a) / (s * ln 10)
         95% 区间使用正态近似（也可自行改为分位数法）

    参数
    ----
    Mc : float
        完备震级阈值
    delta : float in (0,1]
        遗忘系数/折扣，越小越快适应 b 的时变
    a0, s0 : float
        Gamma(a0, s0) 先验（shape, rate）
    mag_key : str
        Sequence 中震级字段名
    write_back : bool
        是否把结果字段写回到传入的 seq（DotDict/自定义对象）

    结果字段（长度 = num_events）
    ----
    a_t, s_t, b_mean, b_sd, b_lo, b_hi, used_mask
    """

    def __init__(
        self,
        Mc: float = 3,
        delta: float = 0.97,
        a0: float = 1e-3,
        s0: Optional[float] = None, # rate 可以自动算
        init_b_target: Optional[float] = None,
        mag_key: str = "mag",
        write_back: bool = True,
        ci_z: float = 1.959963984540054,  # ~95%
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.Mc = float(Mc)
        if not (0.0 < delta <= 1.0):
            raise ValueError("delta must be in (0, 1].")
        self.delta = float(delta)
        
        if init_b_target is not None:
            # 根据目标 b 期望，重算 s0
            if init_b_target <= 0:
                raise ValueError("init_b_target 必须为正数")
            if s0 is not None:
                raise ValueError("不能同时设定 s0 和 init_b_target")
            s0 = a0 / (init_b_target * LN10)
        if a0 <= 0 or s0 <= 0:
            raise ValueError("a0, s0 must be positive.")
        self.a0 = float(a0)
        self.s0 = float(s0) if s0 is not None else 1e-3
        self.mag_key = mag_key
        self.write_back = write_back
        self.ci_z = float(ci_z)
        self.dtype = dtype
        self.device = device

        # 在线状态（标量）
        self._a = None
        self._s = None

    # ---------- 核心：对整个 Sequence 逐事件更新 ----------
    def fit(self, seq) -> Dict[str, torch.Tensor]:
        if self.mag_key not in seq:
            raise KeyError(f"Sequence 缺少震级属性 '{self.mag_key}'。已有键：{list(seq.keys())}")

        mags: torch.Tensor = seq[self.mag_key]
        if self.dtype is not None:
            mags = mags.to(dtype=self.dtype)
        else:
            mags = mags.to(dtype=torch.get_default_dtype())
        if self.device is not None:
            mags = mags.to(self.device)

        device = mags.device
        n = mags.shape[0]

        # 创建结果容器
        a_t = torch.empty(n, dtype=mags.dtype, device=device)
        s_t = torch.empty(n, dtype=mags.dtype, device=device)
        b_mean = torch.empty(n, dtype=mags.dtype, device=device)
        b_sd = torch.empty(n, dtype=mags.dtype, device=device)
        b_lo = torch.empty(n, dtype=mags.dtype, device=device)
        b_hi = torch.empty(n, dtype=mags.dtype, device=device)
        used_mask = torch.zeros(n, dtype=torch.int64, device=device)

        # 状态初始化（标量）
        a = torch.tensor(self.a0, dtype=mags.dtype, device=device)
        s = torch.tensor(self.s0, dtype=mags.dtype, device=device)
        z = torch.tensor(self.ci_z, dtype=mags.dtype, device=device)

        # 逐事件递推
        for i in range(n):
            # 折扣
            a = a * self.delta
            s = s * self.delta

            m_i = mags[i]
            if m_i >= self.Mc:
                x_i = m_i - self.Mc
                a = a + 1.0
                s = s + x_i
                used_mask[i] = 1

            # 后验到 b 的摘要
            bm = a / (s * LN10)
            bstd = torch.sqrt(a) / (s * LN10)
            blo = torch.clamp(bm - z * bstd, min=0.0)
            bhi = bm + z * bstd

            a_t[i] = a
            s_t[i] = s
            b_mean[i] = bm
            b_sd[i] = bstd
            b_lo[i] = blo
            b_hi[i] = bhi

        # 保存最终标量状态（便于继续在线更新）
        self._a = a
        self._s = s

        out = dict(
            a_t=a_t, s_t=s_t, b_mean=b_mean, b_sd=b_sd, b_lo=b_lo, b_hi=b_hi, used_mask=used_mask
        )

        if self.write_back:
            for k, v in out.items():
                seq[k] = v

        return out

    # ---------- 继续在线：添加一个新事件（可选） ----------
    def update_one(self, m: Union[float, torch.Tensor]) -> Dict[str, float]:
        """
        若你想在 fit 之后继续在线更新，可调用本方法。
        返回该事件后的 (a, s, b_mean, b_sd, b_lo, b_hi)
        """
        if self._a is None or self._s is None:
            # 首次单步更新：初始化
            dtype = self.dtype or (m.dtype if isinstance(m, torch.Tensor) else torch.get_default_dtype())
            device = self.device or (m.device if isinstance(m, torch.Tensor) else None)
            self._a = torch.tensor(self.a0, dtype=dtype, device=device)
            self._s = torch.tensor(self.s0, dtype=dtype, device=device)

        # 折扣
        self._a = self._a * self.delta
        self._s = self._s * self.delta

        if isinstance(m, torch.Tensor):
            m_val = m
        else:
            m_val = torch.tensor(float(m), dtype=self._a.dtype, device=self._a.device)

        if m_val >= self.Mc:
            self._a = self._a + 1.0
            self._s = self._s + (m_val - self.Mc)

        bm = self._a / (self._s * LN10)
        bstd = torch.sqrt(self._a) / (self._s * LN10)
        z = torch.tensor(self.ci_z, dtype=bm.dtype, device=bm.device)
        blo = torch.clamp(bm - z * bstd, min=0.0)
        bhi = bm + z * bstd

        return dict(
            a=float(self._a.item()),
            s=float(self._s.item()),
            b_mean=float(bm.item()),
            b_sd=float(bstd.item()),
            b_lo=float(blo.item()),
            b_hi=float(bhi.item()),
        )

    # ---------- 绘图 ----------
    @staticmethod
    def plot(
        seq,
        field_mean: str = "b_mean",
        field_lo: str = "b_lo",
        field_hi: str = "b_hi",
        truth_lines: Optional[Dict[str, float]] = None,  # e.g., {"phase1": 1.0, "phase2": 0.8}
        switch_index: Optional[int] = None,
        title: str = "Dynamic Bayesian b-value (event-by-event)",
        ax: Optional[plt.Axes] = None,
    ):
        """
        直接从 seq（已写回结果）画出逐事件 b 的后验均值与置信带。
        """
        if field_mean not in seq or field_lo not in seq or field_hi not in seq:
            raise KeyError("Sequence 未包含绘图所需字段，请先调用 fit() 完成更新并写回。")

        b_mean = seq[field_mean].detach().cpu().numpy()
        b_lo = seq[field_lo].detach().cpu().numpy()
        b_hi = seq[field_hi].detach().cpu().numpy()
        n = len(b_mean)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))

        xs = range(n)
        ax.plot(xs, b_mean, label="Posterior mean b")
        ax.fill_between(xs, b_lo, b_hi, alpha=0.3, label="~95% credible band")

        if switch_index is not None:
            ax.axvline(switch_index, linestyle="--", label="Switch index")

        if truth_lines:
            for lab, val in truth_lines.items():
                ax.axhline(val, linestyle=":", label=f"Truth {lab}")

        ax.set_xlabel("Event index")
        ax.set_ylabel("b-value")
        ax.set_title(title)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


# ---------------- 使用示例 ----------------
# %%
updater = BayesianGRBUpdater(Mc=3, delta=0.99, a0=5, init_b_target=0.8, mag_key="mag", write_back=True)
results = updater.fit(seq)            # 逐事件更新；同时把结果字段写回0 seq
BayesianGRBUpdater.plot(seq)          # 绘图（可选传入 truth_lines / switch_index）
# 在线追加一个事件：
updater.update_one(3.2)               # 返回该事件后的摘要

# %%
from src.distributions.gutenberg_richter import GutenbergRichter
dist = GutenbergRichter(b=seq.b_mean, mag_min=2.0, mag_max=10.0)
