import math
import copy
from typing import Dict, Optional, Union
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.sequence import Sequence
from src.utils.utils import day_offsets_to_np_datetime64, set_xaxis_time_locator
from .base import BValueUpdaterBase

LN10 = math.log(10.0)


@BValueUpdaterBase.register("bayesian_gr")
@BValueUpdaterBase.register("bayesian_grb")
@BValueUpdaterBase.register("bayesian")
class BayesianGRBUpdater(BValueUpdaterBase):
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

    def expand_state(self, batch_size: int) -> "BayesianGRBUpdater":
        """Expand scalar internal state to batched state for sampling/update."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        target_device = self.device if self.device is not None else torch.device("cpu")

        if self._a is None or self._s is None:
            self._a = torch.tensor(self.a0, dtype=target_dtype, device=target_device)
            self._s = torch.tensor(self.s0, dtype=target_dtype, device=target_device)
        else:
            self._a = self._a.to(dtype=target_dtype, device=target_device)
            self._s = self._s.to(dtype=target_dtype, device=target_device)

        if self._a.ndim == 0:
            self._a = self._a.reshape(1).expand(batch_size).clone()
            self._s = self._s.reshape(1).expand(batch_size).clone()
        elif self._a.ndim == 1:
            if self._a.shape[0] == 1:
                self._a = self._a.expand(batch_size).clone()
                self._s = self._s.expand(batch_size).clone()
            elif self._a.shape[0] != batch_size:
                raise ValueError(
                    f"Cannot expand state from batch={self._a.shape[0]} to batch={batch_size}."
                )
        else:
            raise ValueError("Internal state must be scalar or 1D tensor.")
        return self

    def clone_for_batch(self, batch_size: int):
        """Return list of deep-copied updaters for per-sequence independent state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [copy.deepcopy(self) for _ in range(batch_size)]

    def sample_b_value(self, mode: str = "mean") -> torch.Tensor:
        """Sample or return posterior mean b based on current updater state."""
        if self._a is None or self._s is None:
            target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
            target_device = self.device if self.device is not None else torch.device("cpu")
            self._a = torch.tensor(self.a0, dtype=target_dtype, device=target_device)
            self._s = torch.tensor(self.s0, dtype=target_dtype, device=target_device)
        a = self._a
        s = self._s
        if mode == "mean":
            return a / (s * LN10)
        if mode == "sample":
            alpha = torch.clamp(a, min=1e-6)
            beta = torch.clamp(s, min=1e-6)
            alpha_sample = torch.distributions.Gamma(alpha, beta).sample()
            return alpha_sample / LN10
        raise ValueError(f"Unknown mode: {mode}. Use 'mean' or 'sample'.")

    def _infer_batch_valid_mask(
        self,
        seq,
        mags: torch.Tensor,
    ) -> torch.Tensor:
        """Infer valid event positions for batched magnitudes."""
        if "input_mask" in seq:
            input_mask = seq["input_mask"]
            if not isinstance(input_mask, torch.Tensor):
                raise ValueError("input_mask must be a torch.Tensor when provided.")
            if input_mask.shape != mags.shape:
                raise ValueError(
                    "input_mask must match batched magnitude shape. "
                    f"Got input_mask={tuple(input_mask.shape)} vs mags={tuple(mags.shape)}."
                )
            return input_mask.to(device=mags.device) > 0

        if "end_idx" in seq:
            end_idx = seq["end_idx"]
            if not isinstance(end_idx, torch.Tensor):
                raise ValueError("end_idx must be a torch.Tensor when provided.")
            if end_idx.ndim != 1 or end_idx.shape[0] != mags.shape[0]:
                raise ValueError(
                    "end_idx must have shape [batch_size] for batched input. "
                    f"Got {tuple(end_idx.shape)} for batch_size={mags.shape[0]}."
                )
            end_idx = end_idx.to(device=mags.device, dtype=torch.long)
            arange = torch.arange(mags.shape[1], device=mags.device).unsqueeze(0)
            return arange < end_idx.unsqueeze(1)

        return torch.ones_like(mags, dtype=torch.bool, device=mags.device)

    def _run_single_sequence(
        self,
        mags: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run discounted Bayesian update for one 1D magnitude sequence."""
        if mags.ndim != 1:
            raise ValueError("Single-sequence update expects 1D magnitudes.")
        n = mags.shape[0]
        if n == 0:
            raise ValueError("Input sequence contains zero events.")

        device = mags.device
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
                a = a + 1.0
                s = s + (m_i - self.Mc)
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

        return {
            "a_t": a_t,
            "s_t": s_t,
            "b_mean": b_mean,
            "b_sd": b_sd,
            "b_lo": b_lo,
            "b_hi": b_hi,
            "used_mask": used_mask,
        }

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

        if mags.ndim not in (1, 2):
            raise ValueError(
                "Only 1D [num_events] or 2D [batch_size, seq_len] tensors are supported."
            )

        if mags.ndim == 1:
            out_single = self._run_single_sequence(mags)
            self._a = out_single["a_t"][-1]
            self._s = out_single["s_t"][-1]
            out = {
                f"{prefix}a_t": out_single["a_t"],
                f"{prefix}s_t": out_single["s_t"],
                f"{prefix}b_mean": out_single["b_mean"],
                f"{prefix}b_sd": out_single["b_sd"],
                f"{prefix}b_lo": out_single["b_lo"],
                f"{prefix}b_hi": out_single["b_hi"],
                f"{prefix}used_mask": out_single["used_mask"],
            }
        else:
            batch_size, seq_len = mags.shape
            valid_mask = self._infer_batch_valid_mask(seq=seq, mags=mags)
            if valid_mask.shape != mags.shape:
                raise ValueError(
                    f"Resolved valid mask has invalid shape {tuple(valid_mask.shape)} "
                    f"(expected {tuple(mags.shape)})."
                )

            a_t = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            s_t = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            b_mean = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            b_sd = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            b_lo = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            b_hi = torch.full((batch_size, seq_len), float("nan"), dtype=mags.dtype, device=mags.device)
            used_mask = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=mags.device)

            a_last = torch.full((batch_size,), float("nan"), dtype=mags.dtype, device=mags.device)
            s_last = torch.full((batch_size,), float("nan"), dtype=mags.dtype, device=mags.device)
            for b in range(batch_size):
                valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    continue
                out_b = self._run_single_sequence(mags[b, valid_indices])
                a_t[b, valid_indices] = out_b["a_t"]
                s_t[b, valid_indices] = out_b["s_t"]
                b_mean[b, valid_indices] = out_b["b_mean"]
                b_sd[b, valid_indices] = out_b["b_sd"]
                b_lo[b, valid_indices] = out_b["b_lo"]
                b_hi[b, valid_indices] = out_b["b_hi"]
                used_mask[b, valid_indices] = out_b["used_mask"]
                a_last[b] = out_b["a_t"][-1]
                s_last[b] = out_b["s_t"][-1]

            self._a = a_last
            self._s = s_last
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

    def update_one(
        self,
        m: Union[float, torch.Tensor] = None,
        *,
        mag: Union[float, torch.Tensor, None] = None,
        time: Union[float, torch.Tensor, None] = None,
        active_mask: Union[bool, torch.Tensor, None] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        If you want to continue updating online after calling fit, 
        you can use this method. 
        Returns (a, s, b_mean, b_sd, b_lo, b_hi) after processing the event.
        """
        del time  # Kept for updater interface compatibility (some updaters need time).
        if mag is not None and m is not None:
            raise ValueError("Pass only one of 'm' or 'mag'.")
        if mag is not None:
            m = mag
        if m is None:
            raise ValueError("Missing magnitude input. Pass 'm' or 'mag'.")

        if isinstance(m, torch.Tensor):
            m_val = m
        else:
            target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
            target_device = self.device if self.device is not None else torch.device("cpu")
            m_val = torch.tensor(float(m), dtype=target_dtype, device=target_device)

        if m_val.ndim > 1:
            raise ValueError(
                "update_one supports scalar () or batched vector (B,) magnitudes only."
            )
        if m_val.ndim == 0:
            m_val = m_val.reshape(1)
            squeeze_out = True
        else:
            squeeze_out = False

        if active_mask is None:
            active = torch.ones_like(m_val, dtype=torch.bool, device=m_val.device)
        else:
            active = active_mask if isinstance(active_mask, torch.Tensor) else torch.tensor(active_mask)
            active = active.to(device=m_val.device)
            if active.ndim == 0:
                active = active.reshape(1)
            if active.shape != m_val.shape:
                raise ValueError(
                    "active_mask shape mismatch in update_one. "
                    f"Got active_mask={tuple(active.shape)} vs m={tuple(m_val.shape)}."
                )
            active = active.to(dtype=torch.bool)

        if self._a is None or self._s is None:
            self._a = torch.full_like(m_val, self.a0)
            self._s = torch.full_like(m_val, self.s0)
        else:
            if not isinstance(self._a, torch.Tensor) or not isinstance(self._s, torch.Tensor):
                raise ValueError("Internal updater state is invalid; expected torch.Tensor.")
            self._a = self._a.to(dtype=m_val.dtype, device=m_val.device)
            self._s = self._s.to(dtype=m_val.dtype, device=m_val.device)
            if self._a.ndim == 0:
                self._a = self._a.reshape(1)
                self._s = self._s.reshape(1)
            if self._a.shape != m_val.shape or self._s.shape != m_val.shape:
                raise ValueError(
                    "update_one batch size mismatch with internal state. "
                    f"Got m shape={tuple(m_val.shape)} vs state shape={tuple(self._a.shape)}. "
                    "Reset updater or call fit() before changing batch size."
                )

        self._a[active] = self._a[active] * self.delta
        self._s[active] = self._s[active] * self.delta

        used_bool = (m_val >= self.Mc) & active
        self._a[used_bool] = self._a[used_bool] + 1.0
        self._s[used_bool] = self._s[used_bool] + (m_val[used_bool] - self.Mc)
        used = used_bool.to(dtype=self._a.dtype)

        bm = self._a / (self._s * LN10)
        bstd = torch.sqrt(self._a) / (self._s * LN10)
        z = torch.tensor(self.ci_z, dtype=bm.dtype, device=bm.device)
        blo = torch.clamp(bm - z * bstd, min=0.0)
        bhi = bm + z * bstd

        if squeeze_out:
            return dict(
                a=float(self._a[0].item()),
                s=float(self._s[0].item()),
                b_mean=float(bm[0].item()),
                b_sd=float(bstd[0].item()),
                b_lo=float(blo[0].item()),
                b_hi=float(bhi[0].item()),
                used_mask=int(used[0].item()),
            )
        return dict(
            a=self._a.clone(),
            s=self._s.clone(),
            b_mean=bm.clone(),
            b_sd=bstd.clone(),
            b_lo=blo.clone(),
            b_hi=bhi.clone(),
            used_mask=used.to(dtype=torch.int64).clone(),
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
                xs = day_offsets_to_np_datetime64(start_time, days)
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
