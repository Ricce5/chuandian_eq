# Enhanced sequence-to-event sequence conversion used in EasyTPP.
# Reference: https://zenodo.org/records/8161777 - Using Deep Learning for Flexible and Scalable Earthquake Forecasting.
from typing import List, Optional
import warnings
from typing import Optional, Union
import numpy as np
import torch
from .dot_dict import DotDict

class EventSequence:
    def __init__(self, arrival_times, inter_times, 
                 t_start: Optional[float] = None, t_end: Optional[float] = None,
                 time_series: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
                 time_series_times: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
                 **attributes):
        self.arrival_times = torch.as_tensor(arrival_times)
        self.inter_times = torch.as_tensor(inter_times)

        if self.arrival_times.shape != self.inter_times.shape:
            raise ValueError("arrival_times and inter_times must be the same shape.")

        self.t_start = float(t_start) if t_start is not None else 0.0
        self.t_end = float(t_end) if t_end is not None else self.arrival_times[-1].item()

        self.attributes = {}
        for k, v in attributes.items():
            v_tensor = torch.as_tensor(v)
            if v_tensor.shape[0] != len(self.arrival_times):
                raise ValueError(f"Attribute {k} must have length {len(self.arrival_times)}.")
            self.attributes[k] = v_tensor

        # Handle optional continuous time series and their timestamps. These are
        # independent of per-event attributes so store them separately.
        if time_series is not None:
            assert time_series_times is not None, "time_series_times must be provided if time_series is provided."
            ts = torch.as_tensor(time_series)
            ts_times = torch.as_tensor(time_series_times)
            if ts_times.shape[0] != ts.shape[0]:
                raise ValueError("time_series and time_series_times must have the same length")
            self.time_series = ts
            self.time_series_times = ts_times
        else:
            self.time_series = None
            self.time_series_times = None
        

    def __len__(self):
        return len(self.arrival_times)

    def state_dict(self):
         return {
            "arrival_times": self.arrival_times.tolist(),
            "inter_times": self.inter_times.tolist(),
            **{k: v.tolist() for k, v in self.attributes.items()},
            **({"time_series": self.time_series.tolist(), "time_series_times": self.time_series_times.tolist()} if self.time_series is not None else {})
        }


class Sequence(DotDict):
    """Sequence of events (potentially with marks).

    Args:
        inter_times: Inter-event times, including the last survival time t_end - t_N.
            shape [num_events + 1]
        t_start: Start of the observed time interval.
        t_nll_start: The negative log-likelihood (NLL) will be evaluated on the interval
            (t_nll_start, t_end]. Used when evaluating predictive performance of a TPP
            model given past events. Defaults to t_start.
        **kwargs: Additional dynamical attributes associated with each event in the
            sequence (e.g., magnitude, location), each with shape [num_events, ...].

    Example:
        >>> t_start = 10.0
        >>> arrival_times = np.array([11.1, 11.5, 12.1, 14.2])
        >>> # If you don't know t_end, just set t_end = arrival_times[-1]
        >>> t_end = 20.0
        >>> mag = np.array([4.0, 2.1, 2.5, 2.9])
        >>> loc = np.array([
            [33.1, -115.0],
            [33.2, -115.3],
            [33.1, -116.2],
            [32.3, -115.2],
        ])
        >>> inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])  # inter_times 定义
        >>> seq = Sequence(inter_times, t_start=0.0, mag=mag, loc=loc)
    """

    default_sequence_attrs = {
        "arrival_times",
        "inter_times",
        "t_start",
        "t_end",
        "t_nll_start",
        "time_series",
        "time_series_times",
        "raw_time_series",
        "raw_time_series_times",
    }

    def __init__(
        self,
        inter_times: Union[torch.Tensor, np.ndarray, list],
        t_start: float = 0.0,
        t_nll_start: Optional[float] = None,
        time_series: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        time_series_times: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        raw_time_series: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        raw_time_series_times: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        **kwargs,
    ):
        super().__init__()
        self.inter_times = torch.flatten(torch.as_tensor(inter_times))
        if not self.inter_times.dtype in [torch.float32, torch.float64]:
            raise ValueError(
                f"inter_times must be of type torch.float32 or torch.float64 "
                "(got {self.inter_times.dtype})"
            )
        self.arrival_times = self.inter_times.cumsum(dim=-1)[:-1] + t_start

        self.t_start = float(t_start)
        self.t_end = float(self.inter_times.sum().item() + self.t_start)
        if t_nll_start is None:
            t_nll_start = t_start
        self.t_nll_start = float(t_nll_start)

        for key, value in kwargs.items():
            self[key] = torch.as_tensor(value)

        # Handle optional continuous time series and their sample times.
        if time_series is not None:
            assert time_series_times is not None, "time_series_times must be provided if time_series is provided."
            ts = torch.as_tensor(time_series)
            ts_times = torch.as_tensor(time_series_times)
            if ts_times.shape[0] != ts.shape[0]:
                raise ValueError("time_series and time_series_times must have the same length")
            # Store as sequence attributes so they are carried in state_dict and
            # visible to routines that expect extra attributes.
            self.time_series = ts
            self.time_series_times = ts_times
        if raw_time_series is not None:
            if raw_time_series_times is None:
                raw_time_series_times = time_series_times
            assert raw_time_series_times is not None, (
                "raw_time_series_times must be provided if raw_time_series is provided."
            )
            raw_ts = torch.as_tensor(raw_time_series)
            raw_ts_times = torch.as_tensor(raw_time_series_times)
            if raw_ts_times.shape[0] != raw_ts.shape[0]:
                raise ValueError("raw_time_series and raw_time_series_times must have the same length")
            self.raw_time_series = raw_ts
            self.raw_time_series_times = raw_ts_times

        self._validate_args()
        # Move all tensors to the same device as inter_times
        self.to(self.inter_times.device)

    @property
    def num_events(self):
        return len(self.arrival_times)

    @property
    def num_nll_events(self):
        """Number of events in the interval where the NLL is computed."""
        return (self.arrival_times >= self.t_nll_start).sum().item()

    def __len__(self):
        return self.num_events

    @staticmethod
    def compute_inter_times(
        arrival_times: Union[np.ndarray, list, torch.Tensor],
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        return np.diff(arrival_times, prepend=[t_start], append=[t_end])

    def get_subsequence(self, start: float, end: float, reset_t_nll_to_end: bool = False) -> "Sequence":
        """Select a subset of events in the interval [start, end].

        Args:
            start: window start (inclusive).
            end: window end (inclusive).
            reset_t_nll_to_end: if True, set the returned sequence's t_nll_start == end
                (useful for creating pure test sequences with no NLL interval). Otherwise
                keep t_nll_start = max(self.t_nll_start, start).
        """
        if start < self.t_start or end > self.t_end:
            raise ValueError(
                f"start must be >= {self.t_start} and end must be <= {self.t_end}"
            )
        mask = (self.arrival_times >= start) & (self.arrival_times <= end)

        new_arrival_times = self.arrival_times[mask]
        if len(new_arrival_times) > 0:
            last_inter_time = torch.tensor(
                [end - new_arrival_times[-1]],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )
            new_inter_times = torch.cat([self.inter_times[:-1][mask], last_inter_time])
            first_inter_time = new_arrival_times[0] - start
            new_inter_times[0] = first_inter_time
        else:
            new_inter_times = torch.tensor(
                [end - start],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )

        # Deal with other sequence attributes
        other_attr = {}
        for key, value in self.items():
            if key not in self.default_sequence_attrs:
                other_attr[key] = value[mask].contiguous()
        # Handle continuous time series (slice by time window if present)
        if hasattr(self, 'time_series') and hasattr(self, 'time_series_times'):
            ts = self.time_series
            ts_times = self.time_series_times
            ts_mask = (ts_times >= start) & (ts_times <= end)
            # It's fine if no samples fall into the window — return empty tensors
            new_ts = ts[ts_mask].contiguous()
            new_ts_times = ts_times[ts_mask].contiguous()
            other_attr['time_series'] = new_ts
            other_attr['time_series_times'] = new_ts_times
        if hasattr(self, 'raw_time_series') and hasattr(self, 'raw_time_series_times'):
            raw_ts = self.raw_time_series
            raw_ts_times = self.raw_time_series_times
            raw_ts_mask = (raw_ts_times >= start) & (raw_ts_times <= end)
            other_attr['raw_time_series'] = raw_ts[raw_ts_mask].contiguous()
            other_attr['raw_time_series_times'] = raw_ts_times[raw_ts_mask].contiguous()

        # When the window is very short (end - start < 0.1), end-1e-1 can fall before start;
        # clamp to start to keep 0 <= t_start <= t_nll_start <= t_end.
        t_nll_start = max(end - 1e-1, start) if reset_t_nll_to_end else max(self.t_nll_start, start)

        return Sequence(
            inter_times=new_inter_times,
            t_start=start,
            t_nll_start=t_nll_start,
            **other_attr,
        )

    def _subset_by_event_mask(self, keep_mask: Union[torch.Tensor, np.ndarray, list]) -> "Sequence":
        """Return a new sequence containing only events selected by ``keep_mask``."""
        keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=self.arrival_times.device).flatten()
        if keep_mask.numel() != self.num_events:
            raise ValueError(
                f"keep_mask must have length {self.num_events} (got {keep_mask.numel()})"
            )

        kept_arrival_times = self.arrival_times[keep_mask]
        if kept_arrival_times.numel() > 0:
            boundaries = torch.empty(
                kept_arrival_times.numel() + 2,
                dtype=self.inter_times.dtype,
                device=self.inter_times.device,
            )
            boundaries[0] = self.t_start
            boundaries[1:-1] = kept_arrival_times.to(dtype=self.inter_times.dtype)
            boundaries[-1] = self.t_end
            new_inter_times = torch.diff(boundaries)
        else:
            new_inter_times = torch.tensor(
                [self.t_end - self.t_start],
                dtype=self.inter_times.dtype,
                device=self.inter_times.device,
            )

        other_attr = {}
        for key, value in self.items():
            if key not in self.default_sequence_attrs:
                other_attr[key] = value[keep_mask].contiguous()

        if hasattr(self, "time_series") and hasattr(self, "time_series_times"):
            other_attr["time_series"] = self.time_series.clone()
            other_attr["time_series_times"] = self.time_series_times.clone()
        if hasattr(self, "raw_time_series") and hasattr(self, "raw_time_series_times"):
            other_attr["raw_time_series"] = self.raw_time_series.clone()
            other_attr["raw_time_series_times"] = self.raw_time_series_times.clone()

        return Sequence(
            inter_times=new_inter_times,
            t_start=self.t_start,
            t_nll_start=self.t_nll_start,
            **other_attr,
        )

    def drop_events(
        self,
        drop_prob: float,
        *,
        generator: Optional[torch.Generator] = None,
        min_total_events: int = 1,
        min_nll_events: int = 1,
    ) -> "Sequence":
        """Randomly drop events and rebuild a valid sequence.

        Args:
            drop_prob: Probability of deleting each event independently.
            generator: Optional torch random generator used for reproducible sampling.
            min_total_events: Minimum number of events to keep in total.
            min_nll_events: Minimum number of kept events with arrival time >= ``t_nll_start``.
        """
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1] (got {drop_prob})")
        if min_total_events < 0:
            raise ValueError(f"min_total_events must be >= 0 (got {min_total_events})")
        if min_nll_events < 0:
            raise ValueError(f"min_nll_events must be >= 0 (got {min_nll_events})")
        if self.num_events == 0 or drop_prob == 0.0:
            return self._subset_by_event_mask(torch.ones(self.num_events, dtype=torch.bool, device=self.arrival_times.device))

        keep_mask = torch.rand(
            self.num_events,
            generator=generator,
            device=self.arrival_times.device,
        ) >= drop_prob

        def ensure_min_kept(mask: torch.Tensor, eligible_idx: torch.Tensor, min_keep: int) -> torch.Tensor:
            if min_keep == 0 or eligible_idx.numel() == 0:
                return mask
            current = int(mask[eligible_idx].sum().item())
            if current >= min_keep:
                return mask
            dropped_idx = eligible_idx[~mask[eligible_idx]]
            need = min(min_keep - current, dropped_idx.numel())
            if need <= 0:
                return mask
            selected = dropped_idx[
                torch.randperm(
                    dropped_idx.numel(),
                    generator=generator,
                    device=dropped_idx.device,
                )[:need]
            ]
            mask[selected] = True
            return mask

        all_idx = torch.arange(self.num_events, device=self.arrival_times.device)
        nll_idx = torch.nonzero(self.arrival_times >= self.t_nll_start, as_tuple=False).flatten()
        keep_mask = ensure_min_kept(keep_mask, nll_idx, min_nll_events)
        keep_mask = ensure_min_kept(keep_mask, all_idx, min_total_events)
        return self._subset_by_event_mask(keep_mask)

    def init_sample_sequence(self) -> "Sequence":
        """Initialize a sample sequence for training."""
        last_event_time = self.arrival_times[-1]
        return self.get_subsequence(last_event_time, last_event_time)


    def state_dict(self) -> dict:
        # These attributes are computed from inter_times and t_start, no need to save them to disk
        inferred_attributes = ["arrival_times", "t_end"]
        return {k: v for (k, v) in self.items() if k not in inferred_attributes}

    def _validate_args(self):
        """Check if the event sequence is valid and the shapes are correct."""
        if not (0 <= self.t_start <= self.t_nll_start <= self.t_end):
            raise ValueError(
                "It should hold 0 <= t_start <= t_nll_start < t_end. "
                f"Received {self.t_start}, {self.t_nll_start}, {self.t_end}."
            )

        if torch.any(self.inter_times < 0):
            raise ValueError(
                "Negative inter-event times detected, this isn't a valid event sequence."
            )
        num_zero_inter_times = (self.inter_times[:-1] == 0).sum()
        if num_zero_inter_times > 0:
            warnings.warn(
                f"Found {num_zero_inter_times} zero inter-event times in the sequence. "
                f"This violates fundamental assumptions of TPP models and may lead to "
                f"incorrect log-likelihood values."
            )

        for key, value in self.items():
            if key not in self.default_sequence_attrs and value.shape[0] != len(self):
                raise ValueError(
                    f"Attribute {key} must have shape [{len(self)}, ...] (got {list(value.shape)})"
                )
    def to_event_sequence(self) -> EventSequence:
        """
        Convert to event-only EventSequence:
        - Removes the survival time (last inter_time)
        - Sets first inter_time to 0.0
        - Recomputes arrival_times
        """
        if len(self.inter_times) <= 1:
            raise ValueError("Sequence too short to remove survival time.")

        inter_times = self.inter_times[:-1].clone()
        inter_times[0] = 0.0
        arrival_times = inter_times.cumsum(dim=0) 

        other_attr = {
            k: v.clone() for k, v in self.items()
            if k not in self.default_sequence_attrs
        }

        # Pass through continuous time series if present
        ts = self.get('time_series', None)
        ts_times = self.get('time_series_times', None)

        return EventSequence(
            t_start=self.t_start,
            t_end=self.t_end,
            arrival_times=arrival_times,
            inter_times=inter_times,
            time_series=ts,
            time_series_times=ts_times,
            **other_attr
        )
