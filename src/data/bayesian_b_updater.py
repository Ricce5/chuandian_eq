import math
from typing import Dict, Optional, Union
import torch
import matplotlib.pyplot as plt
from .sequence import Sequence
import numpy as np

from  src.utils.utils import _to_np_datetime64_seconds, _to_py_datetime,set_xaxis_time_locator
import matplotlib.dates as mdates

LN10 = math.log(10.0)

    
class BayesianGRBUpdater:
    """
    Event-by-event Bayesian updater for GR b-value with time-variation
    using discounted conjugate Gamma prior on alpha = b * ln(10).

    Update logic (event-by-event):
      1) Discount prior: (a, s) <- (delta*a, delta*s)
      2) If m_i >= Mc: a <- a + 1; s <- s + (m_i - Mc)
      3) Posterior summary for b:
         E[b] = a / (s * ln 10)
         SD[b] = sqrt(a) / (s * ln 10)
         95% interval uses normal approximation (can be modified to use quantiles)
        b∼Gamma(a,s⋅ln(10)).
    Parameters
    ----
    Mc : float
        Completeness magnitude threshold
    delta : float in (0,1]
        Forgetting factor/discount, smaller values adapt faster to time-varying b
    a0, s0 : float
        Gamma(a0, s0) prior (shape, rate)
    mag_key : str
        Magnitude field name in the Sequence
    write_back : bool
        Whether to write the result fields back to the input seq (DotDict/custom object)

    Result fields (length = num_events)
    ----
    a_t, s_t, b_mean, b_sd, b_lo, b_hi, used_mask
    """

    def __init__(
        self,
        Mc: float = 3,
        delta: float = 0.97,
        a0: float = 1e-3,
        s0: Optional[float] = None,
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
            # Recalculate s0 based on the target b expectation
            if init_b_target <= 0:
                raise ValueError("init_b_target must be positive")
            if s0 is not None:
                raise ValueError("Cannot set both s0 and init_b_target simultaneously")
            s0 = a0 / (init_b_target * LN10)
        elif s0 is None:
            s0 = 1e-3 
        if a0 <= 0 or s0 <= 0:
            raise ValueError("a0, s0 must be positive.")
        self.init_b_target = init_b_target
        self.a0 = float(a0)
        self.s0 = float(s0) if s0 is not None else 1e-3
        self.mag_key = mag_key
        self.write_back = write_back
        self.ci_z = float(ci_z)
        self.dtype = dtype
        self.device = device


        self._a = None
        self._s = None


    def fit(self, seq: Sequence,prefix: str = "") -> Dict[str, torch.Tensor]:
        if self.mag_key not in seq:
            raise KeyError(f"Sequence is missing the magnitude attribute '{self.mag_key}'. Available keys: {list(seq.keys())}")

        mags: torch.Tensor = seq[self.mag_key]
        if self.dtype is not None:
            mags = mags.to(dtype=self.dtype)
        else:
            mags = mags.to(dtype=torch.get_default_dtype())
        if self.device is not None:
            mags = mags.to(self.device)

        device = mags.device
        n = mags.shape[0]

        a_t = torch.empty(n, dtype=mags.dtype, device=device)
        s_t = torch.empty(n, dtype=mags.dtype, device=device)
        b_mean = torch.empty(n, dtype=mags.dtype, device=device)
        b_sd = torch.empty(n, dtype=mags.dtype, device=device)
        b_lo = torch.empty(n, dtype=mags.dtype, device=device)
        b_hi = torch.empty(n, dtype=mags.dtype, device=device)
        used_mask = torch.zeros(n, dtype=torch.int64, device=device)

        a = torch.tensor(self.a0, dtype=mags.dtype, device=device)
        s = torch.tensor(self.s0, dtype=mags.dtype, device=device)
        z = torch.tensor(self.ci_z, dtype=mags.dtype, device=device)

        for i in range(n):
            a = a * self.delta
            s = s * self.delta

            m_i = mags[i]
            if m_i >= self.Mc:
                x_i = m_i - self.Mc
                a = a + 1.0
                s = s + x_i
                used_mask[i] = 1

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

        self._a = a
        self._s = s

        out = {
        f"{prefix}a_t": a_t,
        f"{prefix}s_t": s_t,
        f"{prefix}b_mean": b_mean,
        f"{prefix}b_sd": b_sd,
        f"{prefix}b_lo": b_lo,
        f"{prefix}b_hi": b_hi,
        f"{prefix}used_mask": used_mask,
    }

        if self.write_back:
            for k, v in out.items():
                seq[k] = v

        return out

    def update_one(self, m: Union[float, torch.Tensor]) -> Dict[str, float]:
        """
        If you want to continue updating online after calling fit, 
        you can use this method. 
        Returns (a, s, b_mean, b_sd, b_lo, b_hi) after processing the event.
        """
        if self._a is None or self._s is None:
            dtype = self.dtype or (m.dtype if isinstance(m, torch.Tensor) else torch.get_default_dtype())
            device = self.device or (m.device if isinstance(m, torch.Tensor) else None)
            self._a = torch.tensor(self.a0, dtype=dtype, device=device)
            self._s = torch.tensor(self.s0, dtype=dtype, device=device)

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

    @staticmethod
    def plot(
        seq,
        prefix: str = "",
        field_mean: str = "b_mean",
        field_lo: str = "b_lo",
        field_hi: str = "b_hi",
        truth_lines: Optional[Dict[str, float]] = None,
        switch_index: Optional[int] = None,
        title: str = "Dynamic Bayesian b-value",
        ax: Optional[plt.Axes] = None,
        x_axis: str = "event",
        time_key: str = "arrival_days", 
        start_time=None,                 
        show: bool = True,
    ):

        field_mean = prefix + field_mean
        field_lo   = prefix + field_lo
        field_hi   = prefix + field_hi
        if field_mean not in seq or field_lo not in seq or field_hi not in seq:
            raise KeyError("Sequence does not contain the required fields for plotting. Please run fit() to write them back.")

        b_mean = seq[field_mean].detach().cpu().numpy()
        b_lo   = seq[field_lo].detach().cpu().numpy()
        b_hi   = seq[field_hi].detach().cpu().numpy()
        n = len(b_mean)

        if x_axis == "event":
            xs = np.arange(n)
            xlabel = "Event index"
        elif x_axis == "time":
            if time_key not in seq:
                raise KeyError(f"Sequence is missing the time field '{time_key}'")
            days = seq[time_key].detach().cpu().numpy() 

            if start_time is None:
                xs = days
                xlabel = "Time (days)"
            else:
                t0_np = _to_np_datetime64_seconds(start_time)
                xs = t0_np + days.astype('timedelta64[D]')  
                xlabel = "Year"
        else:
            raise ValueError("x_axis must be either 'event' or 'time'")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.figure


        ax.plot(xs, b_mean, label="Posterior mean b")
        ax.fill_between(xs, b_lo, b_hi, alpha=0.3, label="~95% credible band")

        if switch_index is not None and x_axis == "event":
            ax.axvline(switch_index, linestyle="--", label="Switch index")
        if truth_lines:
            for lab, val in truth_lines.items():
                ax.axhline(val, linestyle=":", label=f"Truth {lab}")

        ax.set_ylabel("b-value")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(loc="best")

        if x_axis == "time" and start_time is not None:
            set_xaxis_time_locator(ax, start_time)


        if show:
            plt.tight_layout()
            plt.show()