from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch


class OracleEventFeatureBuilder:
    """Build Oracle-style event-level injection/seismicity marks.

    Returned features are per-event tensors aligned with ``df_eq`` rows:
    - ``vm``: log10(abs(interpolated injection rate))
    - ``dVc``: log10(abs(sequential cumulative-volume change at events))
    - ``sv``: sign of sequential cumulative-volume change at events
    - ``dTS``: log10(time since last non-zero injection sample), Oracle legacy style
    - ``aRs``: smoothed causal seismicity rate in log10(1/min), Oracle legacy style
    """

    REQUIRED_EQ_COLS = ("t",)
    REQUIRED_INJ_COLS = ("t", "rate")

    def __init__(
        self,
        *,
        time_unit_minutes: float,
        smoothing_window: int = 10,
        activity_epsilon: float = 0.0,
        log_floor: float = -10.0,
    ) -> None:
        self.time_unit_minutes = float(time_unit_minutes)
        self.smoothing_window = int(smoothing_window)
        self.activity_epsilon = float(activity_epsilon)
        self.log_floor = float(log_floor)

        if not np.isfinite(self.time_unit_minutes) or self.time_unit_minutes <= 0:
            raise ValueError(f"time_unit_minutes must be finite and > 0, got {time_unit_minutes}.")
        if self.smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}.")
        if self.activity_epsilon < 0:
            raise ValueError(f"activity_epsilon must be >= 0, got {activity_epsilon}.")

    def build(self, *, df_eq: pd.DataFrame, df_inj: pd.DataFrame) -> Dict[str, torch.Tensor]:
        self._validate_columns(df_eq, df_inj)

        eq_t = pd.to_numeric(df_eq["t"], errors="raise").to_numpy(dtype=np.float64)
        inj_t = pd.to_numeric(df_inj["t"], errors="raise").to_numpy(dtype=np.float64)
        inj_rate = pd.to_numeric(df_inj["rate"], errors="raise").to_numpy(dtype=np.float64)

        if not np.isfinite(eq_t).all():
            raise ValueError("Found non-finite EQ relative times while building Oracle event features.")
        if not np.isfinite(inj_t).all() or not np.isfinite(inj_rate).all():
            raise ValueError("Found non-finite injection data while building Oracle event features.")
        if np.any(np.diff(eq_t) <= 0):
            raise ValueError("EQ times must be strictly increasing for Oracle event feature construction.")
        if np.any(np.diff(inj_t) < 0):
            raise ValueError("Injection times must be non-decreasing for Oracle event feature construction.")

        n_events = eq_t.shape[0]
        if n_events == 0:
            empty = torch.empty(0, dtype=torch.float32)
            return {"vm": empty, "dVc": empty, "sv": empty, "dTS": empty, "aRs": empty}

        dt_inj_min = np.diff(inj_t, prepend=inj_t[0]) * self.time_unit_minutes
        cumulative_volume = np.cumsum(inj_rate * dt_inj_min)

        vm_lin = np.interp(eq_t, inj_t, inj_rate)
        vm_cum = np.interp(eq_t, inj_t, cumulative_volume)
        dvm = np.diff(vm_cum, prepend=0.0)
        sv = np.sign(dvm)

        elapsed_inj = self._legacy_time_since_nonzero_injection(inj_t, inj_rate, self.activity_epsilon)
        elapsed_event = np.interp(eq_t, inj_t, elapsed_inj)

        inter_times_min = np.diff(eq_t, prepend=0.0) * self.time_unit_minutes
        ars = self._smoothed_log10_rate(inter_times_min, window=self.smoothing_window, floor=self.log_floor)

        with np.errstate(divide="ignore", invalid="ignore"):
            vm = np.log10(np.abs(vm_lin))
            dvc = np.log10(np.abs(dvm))
            dts = np.log10(elapsed_event)

        return {
            "vm": torch.tensor(vm, dtype=torch.float32),
            "dVc": torch.tensor(dvc, dtype=torch.float32),
            "sv": torch.tensor(sv, dtype=torch.float32),
            "dTS": torch.tensor(dts, dtype=torch.float32),
            "aRs": torch.tensor(ars, dtype=torch.float32),
        }

    def _validate_columns(self, df_eq: pd.DataFrame, df_inj: pd.DataFrame) -> None:
        missing_eq = [col for col in self.REQUIRED_EQ_COLS if col not in df_eq.columns]
        if missing_eq:
            raise KeyError(f"Missing required EQ columns for Oracle event features: {missing_eq}")

        missing_inj = [col for col in self.REQUIRED_INJ_COLS if col not in df_inj.columns]
        if missing_inj:
            raise KeyError(f"Missing required injection columns for Oracle event features: {missing_inj}")

        if df_inj.shape[0] < 2:
            raise ValueError(
                "Oracle event features require at least 2 injection samples to build cumulative marks."
            )

    @staticmethod
    def _legacy_time_since_nonzero_injection(
        inj_t: np.ndarray,
        inj_rate: np.ndarray,
        activity_epsilon: float,
    ) -> np.ndarray:
        active = np.abs(inj_rate) > activity_epsilon
        elapsed = np.zeros_like(inj_t, dtype=np.float64)
        last_active_t = np.nan
        for idx in range(len(inj_t)):
            if active[idx]:
                last_active_t = inj_t[idx]
                elapsed[idx] = 0.0
            elif np.isnan(last_active_t):
                elapsed[idx] = 0.0
            else:
                elapsed[idx] = inj_t[idx] - last_active_t
        return elapsed

    @staticmethod
    def _smoothed_log10_rate(inter_times_min: np.ndarray, *, window: int, floor: float) -> np.ndarray:
        if np.any(inter_times_min < 0):
            raise ValueError("Inter-event times in minutes must be non-negative for Oracle aRs construction.")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rates = np.log10(np.reciprocal(inter_times_min))

        history = np.full(window, floor, dtype=np.float64)
        out = np.empty_like(log_rates, dtype=np.float64)
        for idx, value in enumerate(log_rates):
            history = np.concatenate([history[1:], np.array([value], dtype=np.float64)])
            out[idx] = history.mean()
        return out
