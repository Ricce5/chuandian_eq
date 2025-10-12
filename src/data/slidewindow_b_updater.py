import math
from typing import Dict, Optional, Literal
import torch
import matplotlib.pyplot as plt

from  src.utils.utils import _to_np_datetime64_seconds, _to_py_datetime,set_xaxis_time_locator
import matplotlib.dates as mdates
LN10 = math.log(10.0)

class FixedTimeWindowGRB:
    """
    Fixed time-window sliding estimation of b-value (consistent with the most common approach):
      For each "evaluation moment" (default is the arrival time of each event), take all events 
      within [t - window_len, t] that satisfy m >= Mc, and estimate b using MLE (default) or 
      one-time conjugate Bayes estimation within the window.

    Window setup:
      - Directly pass window_len (time unit consistent with arrival_times, recommended in days)
      - Or pass only target_count: estimate the fixed window length Δt=target_count/λ using the 
        overall occurrence rate λ=N/T of the catalog.

    Output (aligned with the catalog, corresponding to each event moment):
      - b_mean[i]  : Estimated b-value in the fixed time window with the i-th event as the 
                     "right endpoint" (NaN if less than min_count)
      - n_used[i]  : Number of valid events in the window
      - win_len[i] : Fixed window length (constant vector for verification)
      - used_mask  : Marks which events satisfy m >= Mc (consistent with input)

    Note: MLE formula b = log10(e) / (mean(M) - Mc)
    """

    def __init__(
        self,
        Mc: float = 3.0,
        window_len: Optional[float] = None,   
        target_count: Optional[int] = None,   
        min_count: int = 30,                  
        mag_key: str = "mag",
        time_key: str = "arrival_times",
        method: Literal["mle", "bayes"] = "mle",
        a0: float = 1e-3, s0: float = 1e-3,     #  # Only used when method="bayes"
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        write_back: bool = True,
    ):
        if window_len is None and (target_count is None or target_count <= 0):
            raise ValueError("Please provide window_len, or a positive integer target_count to automatically estimate the fixed window length.")
        if method == "bayes" and (a0 <= 0 or s0 <= 0):
            raise ValueError("Bayesian mode requires positive prior values for a0 and s0")
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
        """Estimate b-value within a fixed window (returns float)."""
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
        if self.mag_key not in seq:
            raise KeyError(f"Sequence is missing the key '{self.mag_key}'")
        if self.time_key not in seq:
            raise KeyError(f"Sequence is missing the key '{self.time_key}'")

        mags: torch.Tensor = seq[self.mag_key]
        times: torch.Tensor = seq[self.time_key]

        if self.dtype is not None:
            mags = mags.to(self.dtype); times = times.to(self.dtype)
        else:
            d = torch.get_default_dtype()
            mags = mags.to(d); times = times.to(d)
        if self.device is not None:
            mags = mags.to(self.device); times = times.to(self.device)

        device = mags.device
        n = mags.shape[0]
        used_mask = (mags >= self.Mc).to(torch.int64)

        # If window_len is not provided, estimate the fixed window length using the overall rate (unit consistent with times)
        win_len_val: float
        if self.window_len is None:
            used_idx = torch.nonzero(used_mask, as_tuple=False).flatten()
            if used_idx.numel() < max(self.min_count, 5):
                raise ValueError("Too few valid events to estimate the fixed time window based on target_count.")
            t0 = times[used_idx[0]]
            t1 = times[used_idx[-1]]
            total_T = float((t1 - t0).abs().item())
            N = int(used_idx.numel())
            if total_T <= 0:
                raise ValueError("Time span is 0, unable to estimate event rate. Please specify window_len directly.")
            lam = N / total_T               
            win_len_val = self.target_count / lam
        else:
            win_len_val = float(self.window_len)

        b_mean = torch.full((n,), float("nan"), dtype=times.dtype, device=device)
        n_used = torch.zeros(n, dtype=torch.int64, device=device)
        win_len_vec = torch.full((n,), win_len_val, dtype=times.dtype, device=device)

        # Use the two-pointer technique to maintain the time window [t - win_len, t]
        left = 0
        for i in range(n):
            t_right = times[i]
            t_left  = t_right - win_len_vec[i]

            while left < n and times[left] < t_left:
                left += 1

            # Current window range [left, i]
            if left <= i:
                idx_slice = slice(left, i + 1)
                mask_win = (mags[idx_slice] >= self.Mc)
                n_in = int(mask_win.sum().item())
                if n_in >= self.min_count:
                    mags_win = mags[idx_slice][mask_win]
                    b_hat = self._estimate_b_window(mags_win)
                    b_mean[i] = torch.tensor(b_hat, dtype=times.dtype, device=device)
                    n_used[i] = n_in
                else:
                    # If less than min_count, return NaN (can also choose to carry forward the previous value, modify as needed)
                    n_used[i] = n_in
            else:
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

    @staticmethod
    def plot(
        seq,
        prefix: str = "",   
        field_b: str = "b_mean",
        field_win_len: str = "win_len",
        time_key: str = "arrival_times",
        x_axis: str = "time", 
        title: str = "Fixed time-window b-value (MLE)",
        ax: Optional[plt.Axes] = None,
        show_counts: bool = False, counts_key: str = "n_used",
        show: bool = True,  
        start_time = None,     
    ):
        field_b = prefix + field_b
        field_win_len = prefix + field_win_len
        counts_key = prefix + counts_key
        if field_b not in seq:
            raise KeyError(f"Sequence does not contain the field '{field_b}', please run fit() first.")
        b = seq[field_b].detach().cpu().numpy()

        if x_axis == "time":
            if time_key not in seq:
                raise KeyError(f"Sequence is missing the time field '{time_key}'")
            days = seq[time_key].detach().cpu().numpy()  

            if start_time is None:
                xs = days
                xlabel = "Time (days)"
            else:
                t0_np = _to_np_datetime64_seconds(start_time)
                xs = t0_np + days.astype('timedelta64[D]')  # Convert to calendar dates
                xlabel = "Year"
        elif x_axis == "event":
            xs = range(len(b))
            xlabel = "Event index"
        else:
            raise ValueError("x_axis must be either 'event' or 'time'")

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
        
        if x_axis == "time" and start_time is not None:
            set_xaxis_time_locator(ax, start_time)
        if show:
            plt.tight_layout(); plt.show()
