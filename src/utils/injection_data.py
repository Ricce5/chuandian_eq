"""Shared extraction helpers for injection-rate and earthquake-count series."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def time_to_days_scale(metadata: Mapping[str, Any] | None) -> float:
    frequency = metadata.get("freq") if hasattr(metadata, "get") else None
    if frequency is None:
        return 1.0
    try:
        return float(pd.Timedelta(frequency) / pd.Timedelta("1D"))
    except Exception:
        return 1.0


def extract_injection_count_series(
    sequence: Any, metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Extract interval-midpoint injection rates and event counts.

    Raw covariates are preferred because they preserve the original injection
    resolution.  Processed ``time_series`` values are used only when the raw
    fields are unavailable.
    """

    source = "raw_time_series"
    injection = getattr(sequence, "raw_time_series", None)
    injection_times = getattr(sequence, "raw_time_series_times", None)
    if injection is None or injection_times is None:
        source = "time_series"
        injection = getattr(sequence, "time_series", None)
        injection_times = getattr(sequence, "time_series_times", None)
    if injection is None or injection_times is None:
        raise ValueError("Sequence has no injection time_series/time_series_times values.")

    injection_times = to_numpy(injection_times).reshape(-1)
    injection_array = to_numpy(injection)
    if injection_array.shape[0] != injection_times.shape[0]:
        raise ValueError(
            "Injection values and timestamps have inconsistent first dimensions: "
            f"{injection_array.shape[0]} vs {injection_times.shape[0]}."
        )
    if injection_times.size < 2:
        raise ValueError("Injection timestamps must contain at least two samples.")
    if np.any(np.diff(injection_times) <= 0):
        raise ValueError("Injection timestamps must be strictly increasing.")

    injection_values = injection_array.reshape(injection_times.shape[0], -1)[:, 0]
    arrival_times = to_numpy(sequence.arrival_times).reshape(-1)
    event_counts, _ = np.histogram(arrival_times, bins=injection_times)
    midpoint_times = 0.5 * (injection_times[1:] + injection_times[:-1])
    midpoint_injection = 0.5 * (injection_values[1:] + injection_values[:-1])
    return {
        "absolute_days": midpoint_times * time_to_days_scale(metadata),
        "injection": midpoint_injection,
        "counts": event_counts.astype(float),
        "source": source,
    }
