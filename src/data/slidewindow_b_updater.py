import math
from typing import Dict, Optional, Literal
import torch
import matplotlib.pyplot as plt

LN10 = math.log(10.0)

class FixedTimeWindowGRB:
    """
    固定时间窗滑动估计 b 值（与最常用做法一致）：
      对每个“评估时刻”（默认逐事件的到时），取 [t - window_len, t] 内所有满足 m>=Mc 的事件，
      用 MLE（默认）或窗口内一次性共轭 Bayes 估计 b。

    设窗方式：
      - 直接传入 window_len（时间单位与 arrival_times 一致，建议 days）
      - 或仅传 target_count：通过目录整体发生率 λ=N/T 估算固定窗长 Δt=target_count/λ

    输出（与目录等长，对齐到每个事件时刻）：
      - b_mean[i]  : 在以第 i 个事件时刻为“右端点”的固定时间窗估计 b（不足 min_count 则 NaN）
      - n_used[i]  : 该窗内有效事件数
      - win_len[i] : 固定窗长（常数向量，便于检查）
      - used_mask  : 标注哪些事件满足 m>=Mc（与输入一致）

    注：MLE 公式 b = log10(e) / (mean(M) - Mc)
    """

    def __init__(
        self,
        Mc: float = 3.0,
        window_len: Optional[float] = None,     # 固定窗长（与 arrival_times 同单位；优先）
        target_count: Optional[int] = None,     # 若未给 window_len，可用目标事件数估窗
        min_count: int = 30,                    # 窗内最少有效事件数，否则返回 NaN
        mag_key: str = "mag",
        time_key: str = "arrival_times",
        method: Literal["mle", "bayes"] = "mle",
        a0: float = 1e-3, s0: float = 1e-3,     # 仅 method="bayes" 时使用
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        write_back: bool = True,
    ):
        if window_len is None and (target_count is None or target_count <= 0):
            raise ValueError("请提供 window_len，或提供正整数 target_count 以自动估算固定窗长。")
        if method == "bayes" and (a0 <= 0 or s0 <= 0):
            raise ValueError("Bayes 模式需要正的先验 a0, s0")
        self.Mc = float(Mc)
        self.window_len = None if window_len is None else float(window_len)
        self.target_count = None if target_count is None else int(target_count)
        self.min_count = int(min_count)
        self.mag_key = mag_key
        self.time_key = time_key
        self.method = method
        self.a0 = float(a0); self.s0 = float(s0)
        self.dtype = dtype; self.device = device
        self.write_back = write_back

    def _estimate_b_window(self, mags_win: torch.Tensor) -> float:
        """在一个固定窗内估计 b（返回 float）"""
        n = mags_win.numel()
        if n == 0:
            return float("nan")
        if self.method == "mle":
            mean_mag = mags_win.mean()
            denom = (mean_mag - self.Mc)
            if denom <= 0:
                return float("nan")
            b = (1.0 / LN10) / denom
            return float(b.item())
        else:  # bayes
            sum_x = (mags_win - self.Mc).sum()
            a_post = self.a0 + n
            s_post = self.s0 + sum_x
            if s_post <= 0:
                return float("nan")
            b_mean = a_post / (s_post * LN10)
            return float(b_mean.item())

    @torch.no_grad()
    def fit(self, seq, prefix: str = "") -> Dict[str, torch.Tensor]:
        if self.mag_key not in seq:  raise KeyError(f"Sequence 缺少 '{self.mag_key}'")
        if self.time_key not in seq: raise KeyError(f"Sequence 缺少 '{self.time_key}'")

        mags: torch.Tensor = seq[self.mag_key]
        times: torch.Tensor = seq[self.time_key]

        # 统一 dtype / device
        if self.dtype is not None:
            mags = mags.to(self.dtype); times = times.to(self.dtype)
        else:
            d = torch.get_default_dtype()
            mags = mags.to(d); times = times.to(d)
        if self.device is not None:
            mags = mags.to(self.device); times = times.to(self.device)

        device = mags.device
        n = mags.shape[0]

        # 仅标记有效事件
        used_mask = (mags >= self.Mc).to(torch.int64)

        # 若未给 window_len，用总体速率估算固定窗长（单位与 times 相同）
        win_len_val: float
        if self.window_len is None:
            used_idx = torch.nonzero(used_mask, as_tuple=False).flatten()
            if used_idx.numel() < max(self.min_count, 5):
                raise ValueError("有效事件过少，无法根据 target_count 估算固定时间窗。")
            t0 = times[used_idx[0]]
            t1 = times[used_idx[-1]]
            total_T = float((t1 - t0).abs().item())
            N = int(used_idx.numel())
            if total_T <= 0:
                raise ValueError("时间跨度为 0，无法估算事件发生率。请直接指定 window_len。")
            lam = N / total_T               # 事件率（每单位时间）
            win_len_val = self.target_count / lam
        else:
            win_len_val = float(self.window_len)

        # 结果容器
        b_mean = torch.full((n,), float("nan"), dtype=times.dtype, device=device)
        n_used = torch.zeros(n, dtype=torch.int64, device=device)
        win_len_vec = torch.full((n,), win_len_val, dtype=times.dtype, device=device)

        # 双指针法维护时间窗 [t - win_len, t]
        left = 0
        # 为加速，预先把 times, mags 搬到 CPU numpy 不是必须；我们直接用 torch
        for i in range(n):
            t_right = times[i]
            t_left  = t_right - win_len_vec[i]

            # 移动左指针，使 times[left] >= t_left（保持窗口内）
            while left < n and times[left] < t_left:
                left += 1

            # 当前窗口范围 [left, i]
            if left <= i:
                idx_slice = slice(left, i + 1)
                # 窗内有效事件
                mask_win = (mags[idx_slice] >= self.Mc)
                n_in = int(mask_win.sum().item())
                if n_in >= self.min_count:
                    mags_win = mags[idx_slice][mask_win]
                    b_hat = self._estimate_b_window(mags_win)
                    b_mean[i] = torch.tensor(b_hat, dtype=times.dtype, device=device)
                    n_used[i] = n_in
                else:
                    # 不足 min_count，返回 NaN（也可选择延续上一值，按需修改）
                    n_used[i] = n_in
            else:
                # 窗口为空
                n_used[i] = 0

        out = {
        f"{prefix}b_mean": b_mean,
        f"{prefix}n_used": n_used,
        f"{prefix}win_len": win_len_vec,
        f"{prefix}used_mask": used_mask,
    }
        if self.write_back:
            for k, v in out.items():
                seq[k] = v
        return out

    # 画图：横轴可选事件或时间；时间轴默认单位“天”
    @staticmethod
    def plot(
        seq,
        prefix: str = "",   
        field_b: str = "b_mean",
        field_win_len: str = "win_len",
        time_key: str = "arrival_times",
        x_axis: str = "time",   # 与常用展示一致，默认时间轴
        title: str = "Fixed time-window b-value (MLE)",
        ax: Optional[plt.Axes] = None,
        show_counts: bool = False, counts_key: str = "n_used",
        show: bool = True,  
    ):
        field_b = prefix + field_b
        field_win_len = prefix + field_win_len
        counts_key = prefix + counts_key
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
            fig, ax = plt.subplots(figsize=(9, 4.6))

        ax.plot(xs, b, label="b (fixed window)")
        ax.set_xlabel(xlabel); ax.set_ylabel("b-value"); ax.set_title(title)
        ax.legend(loc="best")

        if show_counts and (counts_key in seq):
            counts = seq[counts_key].detach().cpu().numpy()
            ax2 = ax.twinx()
            ax2.plot(xs, counts, alpha=0.35, label="#events in window")
            ax2.set_ylabel("Count in window")
            ax2.legend(loc="upper right")
        if show:
            plt.tight_layout(); plt.show()
