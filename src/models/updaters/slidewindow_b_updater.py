import math
from typing import Dict, Literal, Optional, Union

import matplotlib.pyplot as plt
import torch

from src.utils.utils import day_offsets_to_np_datetime64, set_xaxis_time_locator
from .base import BValueUpdaterBase

LN10 = math.log(10.0)


@BValueUpdaterBase.register("fixed_window_gr")
@BValueUpdaterBase.register("slidewindow_gr")
@BValueUpdaterBase.register("slidewindow")
class FixedTimeWindowGRB(BValueUpdaterBase):
    """
    Fixed time-window sliding estimator for Gutenberg-Richter b-value.

    Supports:
      - Full-sequence estimation via ``fit(seq)``
      - Online single-step update via ``update_one(mag, time)``
      - Batched online update via ``update_one(mag[B], time[B])``
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
        a0: float = 1e-3,
        s0: float = 1e-3,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        write_back: bool = True,
    ):
        if window_len is None and (target_count is None or target_count <= 0):
            raise ValueError(
                "Please provide window_len, or a positive target_count to estimate "
                "the fixed time window automatically."
            )
        if method == "bayes" and (a0 <= 0 or s0 <= 0):
            raise ValueError("Bayesian mode requires positive prior a0 and s0.")

        self.Mc = float(Mc)
        self.window_len = None if window_len is None else float(window_len)
        self.target_count = None if target_count is None else int(target_count)
        self.min_count = int(min_count)
        self.mag_key = mag_key
        self.time_key = time_key
        self.method = method
        self.a0 = float(a0)
        self.s0 = float(s0)
        self.dtype = dtype
        self.device = device
        self.write_back = write_back

        self._stream_single_mags: Optional[torch.Tensor] = None
        self._stream_single_times: Optional[torch.Tensor] = None
        self._stream_batch_histories: Optional[list[dict[str, torch.Tensor]]] = None
        self._target_dtype: Optional[torch.dtype] = None
        self._target_device: Optional[torch.device] = None
        self._last_summary: Optional[Dict[str, Union[float, torch.Tensor]]] = None

    def sample_b_value(self, mode: str = "mean") -> torch.Tensor:
        if mode not in {"mean", "sample"}:
            raise ValueError("mode must be one of ['mean', 'sample'].")
        target_dtype = self._target_dtype or self.dtype or torch.get_default_dtype()
        target_device = self._target_device or self.device or torch.device("cpu")
        eps = torch.finfo(target_dtype).eps

        if self._last_summary is None:
            if self.method == "bayes":
                default_b = self.a0 / (self.s0 * LN10)
            else:
                default_b = 1.0
            return torch.tensor(default_b, dtype=target_dtype, device=target_device).clamp_min(eps)

        b_mean = self._last_summary["b_mean"]
        if isinstance(b_mean, torch.Tensor):
            b = b_mean.to(dtype=target_dtype, device=target_device)
        else:
            b = torch.tensor(float(b_mean), dtype=target_dtype, device=target_device)
        return b.clamp_min(eps)

    def _estimate_b_window(self, mags_win: torch.Tensor) -> float:
        if mags_win.numel() == 0:
            return float("nan")
        if self.method == "mle":
            denom = mags_win.mean() - self.Mc
            if denom <= 0:
                return float("nan")
            return float(((1.0 / LN10) / denom).item())

        sum_x = (mags_win - self.Mc).sum()
        a_post = self.a0 + mags_win.numel()
        s_post = self.s0 + sum_x
        if s_post <= 0:
            return float("nan")
        return float((a_post / (s_post * LN10)).item())

    def _resolve_window_len(
        self,
        mags: torch.Tensor,
        times: torch.Tensor,
        *,
        strict: bool,
    ) -> float:
        if self.window_len is not None:
            return float(self.window_len)

        used_idx = torch.nonzero(mags >= self.Mc, as_tuple=False).flatten()
        min_events = max(self.min_count, 5)
        if used_idx.numel() < min_events:
            if strict:
                raise ValueError(
                    "Too few valid events to estimate fixed window length from target_count."
                )
            return float("nan")

        t0 = times[used_idx[0]]
        t1 = times[used_idx[-1]]
        total_T = float((t1 - t0).abs().item())
        if total_T <= 0:
            if strict:
                raise ValueError(
                    "Time span is 0, unable to estimate event rate. "
                    "Please specify window_len directly."
                )
            return float("nan")

        rate = float(used_idx.numel()) / total_T
        return float(self.target_count / rate)

    def _init_target_dtype_device_for_update(
        self,
        mag: Union[float, torch.Tensor],
        time: Union[float, torch.Tensor],
    ) -> None:
        if self._target_dtype is None:
            self._target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self._target_device is None:
            if isinstance(mag, torch.Tensor):
                self._target_device = mag.device
            elif isinstance(time, torch.Tensor):
                self._target_device = time.device
            else:
                self._target_device = torch.device("cpu")

    def _to_target_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        assert self._target_dtype is not None
        assert self._target_device is not None
        if isinstance(value, torch.Tensor):
            return value.to(dtype=self._target_dtype, device=self._target_device)
        return torch.tensor(float(value), dtype=self._target_dtype, device=self._target_device)

    def _compute_online_summary(
        self,
        mags: torch.Tensor,
        times: torch.Tensor,
    ) -> Dict[str, float]:
        win_len = self._resolve_window_len(mags=mags, times=times, strict=False)
        used_now = int((mags[-1] >= self.Mc).item())
        if not math.isfinite(win_len):
            return {
                "b_mean": float("nan"),
                "n_used": 0.0,
                "win_len": float("nan"),
                "used_mask": float(used_now),
            }

        t_right = times[-1]
        t_left = t_right - win_len
        in_window = (times >= t_left) & (times <= t_right) & (mags >= self.Mc)
        n_in = int(in_window.sum().item())
        b_hat = float("nan")
        if n_in >= self.min_count:
            b_hat = self._estimate_b_window(mags[in_window])

        return {
            "b_mean": float(b_hat),
            "n_used": float(n_in),
            "win_len": float(win_len),
            "used_mask": float(used_now),
        }

    @torch.no_grad()
    def fit(self, seq, prefix: str = "") -> Dict[str, torch.Tensor]:
        if self.mag_key not in seq:
            raise KeyError(f"Sequence is missing key '{self.mag_key}'.")
        if self.time_key not in seq:
            raise KeyError(f"Sequence is missing key '{self.time_key}'.")

        mags = seq[self.mag_key]
        times = seq[self.time_key]

        target_dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        mags = mags.to(dtype=target_dtype)
        times = times.to(dtype=target_dtype)
        if self.device is not None:
            mags = mags.to(self.device)
            times = times.to(self.device)

        if mags.ndim != 1 or times.ndim != 1:
            raise ValueError("fit expects 1D magnitude/time tensors.")
        if mags.shape != times.shape:
            raise ValueError("Magnitude and time arrays must share the same shape.")
        if mags.numel() == 0:
            raise ValueError("Input sequence contains zero events.")
        if torch.any(times[1:] < times[:-1]):
            raise ValueError("Event times must be sorted in non-decreasing order.")

        n = mags.shape[0]
        used_mask = (mags >= self.Mc).to(torch.int64)
        win_len_val = self._resolve_window_len(mags=mags, times=times, strict=True)

        b_mean = torch.full((n,), float("nan"), dtype=times.dtype, device=times.device)
        n_used = torch.zeros(n, dtype=torch.int64, device=times.device)
        win_len_vec = torch.full((n,), win_len_val, dtype=times.dtype, device=times.device)

        left = 0
        for idx in range(n):
            t_left = times[idx] - win_len_vec[idx]
            while left < n and times[left] < t_left:
                left += 1

            if left > idx:
                continue
            mask_win = mags[left : idx + 1] >= self.Mc
            n_in = int(mask_win.sum().item())
            n_used[idx] = n_in
            if n_in >= self.min_count:
                mags_win = mags[left : idx + 1][mask_win]
                b_hat = self._estimate_b_window(mags_win)
                b_mean[idx] = torch.tensor(b_hat, dtype=times.dtype, device=times.device)

        out = {
            f"{prefix}b_mean": b_mean,
            f"{prefix}n_used": n_used,
            f"{prefix}win_len": win_len_vec,
            f"{prefix}used_mask": used_mask,
        }
        if self.write_back:
            for key, value in out.items():
                seq[key] = value
        self._last_summary = {
            "b_mean": out[f"{prefix}b_mean"][-1].detach().clone(),
            "n_used": out[f"{prefix}n_used"][-1].detach().clone(),
            "win_len": out[f"{prefix}win_len"][-1].detach().clone(),
            "used_mask": out[f"{prefix}used_mask"][-1].detach().clone(),
        }
        return out

    @torch.no_grad()
    def update_one(
        self,
        mag: Union[float, torch.Tensor],
        time: Union[float, torch.Tensor],
        active_mask: Optional[Union[bool, torch.Tensor]] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        self._init_target_dtype_device_for_update(mag=mag, time=time)
        mag_t = self._to_target_tensor(mag)
        time_t = self._to_target_tensor(time)
        if mag_t.shape != time_t.shape:
            raise ValueError("mag and time must have identical shape in update_one().")
        if active_mask is None:
            active_mask_t = torch.ones_like(mag_t, dtype=torch.bool, device=mag_t.device)
        else:
            active_mask_t = (
                active_mask if isinstance(active_mask, torch.Tensor) else torch.tensor(active_mask)
            )
            active_mask_t = active_mask_t.to(device=mag_t.device)
            if active_mask_t.shape != mag_t.shape:
                raise ValueError(
                    "active_mask shape mismatch in update_one. "
                    f"Got active_mask={tuple(active_mask_t.shape)} vs mag={tuple(mag_t.shape)}."
                )
            active_mask_t = active_mask_t.to(dtype=torch.bool)

        if mag_t.ndim == 0:
            if self._stream_batch_histories is not None:
                self._stream_batch_histories = None
            if not bool(active_mask_t.item()):
                if self._stream_single_mags is None:
                    win_len = float(self.window_len) if self.window_len is not None else float("nan")
                    out = {
                        "b_mean": float("nan"),
                        "n_used": 0.0,
                        "win_len": win_len,
                        "used_mask": 0.0,
                    }
                    self._last_summary = out
                    return out
                summary = self._compute_online_summary(
                    mags=self._stream_single_mags,
                    times=self._stream_single_times,
                )
                summary["used_mask"] = 0.0
                self._last_summary = summary
                return summary
            if self._stream_single_times is not None and bool((time_t < self._stream_single_times[-1]).item()):
                raise ValueError("Incoming event time must be non-decreasing.")

            mag_vec = mag_t.reshape(1)
            time_vec = time_t.reshape(1)
            if self._stream_single_mags is None:
                self._stream_single_mags = mag_vec
                self._stream_single_times = time_vec
            else:
                self._stream_single_mags = torch.cat([self._stream_single_mags, mag_vec], dim=0)
                self._stream_single_times = torch.cat([self._stream_single_times, time_vec], dim=0)

            summary = self._compute_online_summary(
                mags=self._stream_single_mags,
                times=self._stream_single_times,
            )
            self._last_summary = summary
            return summary

        if mag_t.ndim != 1:
            raise ValueError(
                "update_one supports only scalar () or batched vector (B,) mag/time."
            )

        batch_size = int(mag_t.shape[0])
        if batch_size <= 0:
            raise ValueError("Batched update_one requires a non-empty batch.")

        if self._stream_batch_histories is None:
            self._stream_single_mags = None
            self._stream_single_times = None
            self._stream_batch_histories = [
                {
                    "mags": torch.empty(0, dtype=mag_t.dtype, device=mag_t.device),
                    "times": torch.empty(0, dtype=time_t.dtype, device=time_t.device),
                }
                for _ in range(batch_size)
            ]
        elif len(self._stream_batch_histories) != batch_size:
            raise ValueError(
                "Batched update_one size mismatch with existing state. "
                f"Expected batch={len(self._stream_batch_histories)}, got batch={batch_size}."
            )

        b_mean = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        n_used = torch.empty(batch_size, dtype=torch.int64, device=mag_t.device)
        win_len = torch.empty(batch_size, dtype=mag_t.dtype, device=mag_t.device)
        used_mask = torch.empty(batch_size, dtype=torch.int64, device=mag_t.device)

        for idx in range(batch_size):
            history = self._stream_batch_histories[idx]
            hist_times = history["times"]
            if (
                bool(active_mask_t[idx].item())
                and hist_times.numel() > 0
                and bool((time_t[idx] < hist_times[-1]).item())
            ):
                raise ValueError(
                    f"Incoming event time must be non-decreasing for batch index {idx}."
                )
            if not bool(active_mask_t[idx].item()):
                if history["times"].numel() == 0:
                    summary = {
                        "b_mean": float("nan"),
                        "n_used": 0.0,
                        "win_len": float(self.window_len) if self.window_len is not None else float("nan"),
                        "used_mask": 0.0,
                    }
                else:
                    summary = self._compute_online_summary(mags=history["mags"], times=history["times"])
                    summary["used_mask"] = 0.0
                b_mean[idx] = torch.tensor(summary["b_mean"], dtype=mag_t.dtype, device=mag_t.device)
                n_used[idx] = int(summary["n_used"])
                win_len[idx] = torch.tensor(summary["win_len"], dtype=mag_t.dtype, device=mag_t.device)
                used_mask[idx] = int(summary["used_mask"])
                continue

            history["mags"] = torch.cat([history["mags"], mag_t[idx].reshape(1)], dim=0)
            history["times"] = torch.cat([history["times"], time_t[idx].reshape(1)], dim=0)
            summary = self._compute_online_summary(mags=history["mags"], times=history["times"])
            b_mean[idx] = torch.tensor(summary["b_mean"], dtype=mag_t.dtype, device=mag_t.device)
            n_used[idx] = int(summary["n_used"])
            win_len[idx] = torch.tensor(summary["win_len"], dtype=mag_t.dtype, device=mag_t.device)
            used_mask[idx] = int(summary["used_mask"])

        out = {
            "b_mean": b_mean,
            "n_used": n_used,
            "win_len": win_len,
            "used_mask": used_mask,
        }
        self._last_summary = out
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
        show_counts: bool = False,
        counts_key: str = "n_used",
        show: bool = True,
        start_time=None,
    ):
        field_b = prefix + field_b
        field_win_len = prefix + field_win_len
        counts_key = prefix + counts_key
        if field_b not in seq:
            raise KeyError(
                f"Sequence does not contain '{field_b}'. Please run fit() before plotting."
            )
        b = seq[field_b].detach().cpu().numpy()

        if x_axis == "time":
            if time_key not in seq:
                raise KeyError(f"Sequence is missing the time field '{time_key}'")
            days = seq[time_key].detach().cpu().numpy()
            if start_time is None:
                xs = days
                xlabel = "Time (days)"
            else:
                xs = day_offsets_to_np_datetime64(start_time, days)
                xlabel = "Year"
        elif x_axis == "event":
            xs = range(len(b))
            xlabel = "Event index"
        else:
            raise ValueError("x_axis must be either 'event' or 'time'")

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4.6))

        ax.plot(xs, b, label="b (fixed window)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("b-value")
        ax.set_title(title)
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
            plt.tight_layout()
            plt.show()
