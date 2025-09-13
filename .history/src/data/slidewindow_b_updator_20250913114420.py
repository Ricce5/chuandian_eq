import math
from typing import Dict, Optional, Literal
import torch
import matplotlib.pyplot as plt

LN10 = math.log(10.0)

class SlidingTimeWindowGRB:
    """
    滑动“时间窗”估计 b，但时间窗长度 Δt 由目标事件数 target_count 自动确定：
      对第 k 个“有效事件”(m>=Mc)，取它与前面最近 target_count-1 个有效事件构成的集合，
      窗口长度 Δt = t_k - t_{k - target_count + 1}（若 k < target_count-1，则用现有的所有有效事件）。

    估计器：
      - method="mle": b = log10(e) / (mean(M) - Mc)
      - method="bayes": 先验 alpha=b*ln(10) ~ Gamma(a0,s0)，窗口内做一次共轭更新，返回后验均值：
            E[b] = (a0 + n) / ((s0 + sum(M-Mc)) * ln 10)

    输出（对齐到每个全局事件索引 i）：
      - b_mean[i]       : 截至第 i 个事件的窗口 b 估计（不足 min_count 则为 NaN）
      - n_used[i]       : 当前窗口内有效事件数
      - win_len[i]      : 当前窗口长度（同时间单位）
      - win_start_idx[i]: 窗口起点（全局索引）
      - used_mask[i]    : 该事件是否用于本次窗口（仅在 i 为有效事件时为 1）
    """

    def __init__(
        self,
        Mc: float = 3.0,
        target_count: int = 100,
        min_count: Optional[int] = None,
        mag_key: str = "mag",
        time_key: str = "arrival_times",
        method: Literal["mle", "bayes"] = "mle",
        a0: float = 1e-3,
        s0: float = 1e-3,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        write_back: bool = True,
    ):
        if target_count <= 0:
            raise ValueError("target_count 必须为正整数")
        self.Mc = float(Mc)
        self.target_count = int(target_count)
        self.min_count = int(min_count) if min_count is not None else max(30, min(50, target_count//2))
        self.mag_key = mag_key
        self.time_key = time_key
        self.method = method
        if method == "bayes":
            if a0 <= 0 or s0 <= 0:
                raise ValueError("Bayes 先验 a0, s0 必须为正")
        self.a0 = float(a0)
        self.s0 = float(s0)
        self.dtype = dtype
        self.device = device
        self.write_back = write_back

    def _estimate_window_b(self, mags_win: torch.Tensor) -> float:
        """在一个窗口内估计 b（返回 float）"""
        if mags_win.numel() == 0:
            return float("nan")
        if self.method == "mle":
            mean_mag = mags_win.mean()
            denom = (mean_mag - self.Mc)
            if denom <= 0:
                return float("nan")
            b = (1.0 / LN10) / denom  # == log10(e)/(mean(M) - Mc)
            return float(b.item())
        else:  # bayes
            n = mags_win.numel()
            sum_x = (mags_win - self.Mc).sum()
            a_post = self.a0 + n
            s_post = self.s0 + sum_x
            if s_post <= 0:
                return float("nan")
            b_mean = a_post / (s_post * LN10)
            return float(b_mean.item())

    @torch.no_grad()
    def fit(self, seq) -> Dict[str, torch.Tensor]:
        if self.mag_key not in seq:
            raise KeyError(f"Sequence 缺少 '{self.mag_key}'")
        if self.time_key not in seq:
            raise KeyError(f"Sequence 缺少 '{self.time_key}'")

        mags: torch.Tensor = seq[self.mag_key]
        times: torch.Tensor = seq[self.time_key]

        # dtype / device 统一
        if self.dtype is not None:
            mags = mags.to(dtype=self.dtype)
            times = times.to(dtype=self.dtype)
        else:
            d = torch.get_default_dtype()
            mags = mags.to(dtype=d)
            times = times.to(dtype=d)
        if self.device is not None:
            mags = mags.to(self.device)
            times = times.to(self.device)

        device = mags.device
        n = mags.shape[0]

        # 结果容器（长度 = 所有事件）
        b_mean = torch.full((n,), float("nan"), dtype=mags.dtype, device=device)
        n_used = torch.zeros(n, dtype=torch.int64, device=device)
        win_len = torch.full((n,), float("nan"), dtype=mags.dtype, device=device)
        win_start_idx = torch.full((n,), -1, dtype=torch.int64, device=device)
        used_mask = torch.zeros(n, dtype=torch.int64, device=device)

        # 仅保留有效事件的索引
        used_idx = torch.nonzero(mags >= self.Mc, as_tuple=False).flatten()
        if used_idx.numel() == 0:
            out = dict(
                b_mean=b_mean, n_used=n_used, win_len=win_len, win_start_idx=win_start_idx, used_mask=used_mask
            )
            if self.write_back:
                for k, v in out.items():
                    seq[k] = v
            return out

        # 对每个有效事件做一次窗口估计
        last_b = float("nan")
        last_len = float("nan")
        last_n = 0
        last_start = -1

        for k in range(used_idx.numel()):
            i = int(used_idx[k].item())  # 全局索引
            used_mask[i] = 1

            # 选择窗口内的有效事件索引范围 [k0, k]
            if k >= self.target_count - 1:
                k0 = k - (self.target_count - 1)
            else:
                k0 = 0
            win_used_idx = used_idx[k0 : k + 1]  # 有效事件的全局索引
            # 窗口长度（与 times 的单位一致）
            t_end = times[i]
            t_start = times[int(win_used_idx[0].item())]
            dt = (t_end - t_start).abs()  # 防御性 abs

            # 事件数
            n_in = win_used_idx.numel()

            # 若事件不足 min_count，则跳过估计，仅把“最近一次可用估计”沿用到当前全局位置
            if n_in < self.min_count:
                # 延续上一次
                b_mean[i] = torch.tensor(last_b, dtype=mags.dtype, device=device)
                n_used[i] = last_n
                win_len[i] = torch.tensor(last_len, dtype=mags.dtype, device=device)
                win_start_idx[i] = last_start
                continue

            # 在窗口内做 b 值估计
            mags_win = mags[win_used_idx]
            b_hat = self._estimate_window_b(mags_win)

            # 记录
            last_b = b_hat
            last_len = float(dt.item())
            last_n = int(n_in)
            last_start = int(win_used_idx[0].item())

            b_mean[i] = torch.tensor(b_hat, dtype=mags.dtype, device=device)
            n_used[i] = n_in
            win_len[i] = dt
            win_start_idx[i] = last_start

        # 将“无效事件”位置也填上最近一次可用估计（可选）
        # 这样每个事件索引上都有 b 值曲线（开头可能是 NaN）
        cur_b = float("nan")
        cur_len = float("nan")
        cur_n = 0
        cur_start = -1
        for i in range(n):
            if not torch.isnan(b_mean[i]):
                cur_b = float(b_mean[i].item())
                cur_len = float(win_len[i].item()) if not torch.isnan(win_len[i]) else float("nan")
                cur_n = int(n_used[i].item())
                cur_start = int(win_start_idx[i].item())
            else:
                b_mean[i] = torch.tensor(cur_b, dtype=mags.dtype, device=device) if not math.isnan(cur_b) else b_mean[i]
                win_len[i] = torch.tensor(cur_len, dtype=mags.dtype, device=device) if not math.isnan(cur_len) else win_len[i]
                n_used[i] = torch.tensor(cur_n, dtype=torch.int64, device=device)
                win_start_idx[i] = torch.tensor(cur_start, dtype=torch.int64, device=device)

        out = dict(
            b_mean=b_mean,
            n_used=n_used,
            win_len=win_len,
            win_start_idx=win_start_idx,
            used_mask=used_mask,
        )
        if self.write_back:
            for k, v in out.items():
                seq[k] = v
        return out

    # 简单的画图函数：x轴可选事件索引或时间；可叠加显示“窗口长度”
    @staticmethod
    def plot(
        seq,
        field_b: str = "b_mean",
        field_win_len: str = "win_len",
        time_key: str = "arrival_times",
        x_axis: str = "event",  # "event" 或 "time"
        title: str = "Sliding time window b-value (auto Δt by target count)",
        ax: Optional[plt.Axes] = None,
        show_window_len: bool = False,
        window_len_scale: float = 1.0,  # 把窗长映射到副轴（可选）
    ):
        if field_b not in seq:
            raise KeyError(f"Sequence 未包含字段 '{field_b}'，请先 fit()")
        b = seq[field_b].detach().cpu().numpy()

        if x_axis == "time":
            if time_key not in seq:
                raise KeyError(f"Sequence 缺少时间字段 '{time_key}'")
            xs = seq[time_key].detach().cpu().numpy()
            xlabel = "Time (days)"
        elif x_axis == "event":
            xs = range(len(b))
            xlabel = "Event index"
        else:
            raise ValueError("x_axis 必须是 'event' 或 'time'")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4.8))

        ax.plot(xs, b, label="b (windowed)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("b-value")
        ax.set_title(title)
        ax.legend(loc="best")

        if show_window_len and field_win_len in seq:
            win_len = seq[field_win_len].detach().cpu().numpy()
            ax2 = ax.twinx()
            ax2.plot(xs, win_len * window_len_scale, alpha=0.35, label="window length")
            ax2.set_ylabel(f"Window length × {window_len_scale:g}")
            ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
