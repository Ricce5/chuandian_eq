"""Small data containers and sampling helpers for NETAS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.data.sequence import Sequence


def _to_numpy_array(x: torch.Tensor) -> np.ndarray:
    """Detach a tensor and move it to a NumPy array."""
    return x.detach().cpu().numpy()


def _gen_mag(
    rng: np.random.Generator,
    shape: int,
    *,
    b: float,
    m_min: float,
    m_max: float,
) -> np.ndarray:
    """Sample magnitudes from a truncated Gutenberg-Richter law."""
    if shape <= 0:
        return np.empty((0,), dtype=np.float64)
    u = rng.random(shape)
    mag = (
        -1.0
        / b
        * np.log10(
            -u * (10.0 ** (-b * m_min) - 10.0 ** (-b * m_max))
            + 10.0 ** (-b * m_min)
        )
    )
    return mag.astype(np.float64, copy=False)


def _empty_initial_event_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return empty ``(times, magnitudes)`` arrays for sampled initial events."""
    return (
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.float32),
    )


@dataclass(frozen=True)
class _MagnitudeSamplingParams:
    """Magnitude-law parameters used by NETAS branching samplers."""

    b: float
    m_min: float
    m_max: float


@dataclass(frozen=True)
class _ParentParameters:
    """Cached parent-event context and trigger parameters."""

    context: torch.Tensor
    eta: torch.Tensor
    omega: torch.Tensor
    event_mask: torch.Tensor

    @property
    def event_mask_bool(self) -> torch.Tensor:
        return self.event_mask.bool()

    @property
    def weighted_mixture(self) -> torch.Tensor:
        return self.eta.unsqueeze(-1) * self.omega

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "context": self.context,
            "eta": self.eta,
            "omega": self.omega,
            "event_mask": self.event_mask,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, torch.Tensor]) -> "_ParentParameters":
        return cls(
            context=value["context"],
            eta=value["eta"],
            omega=value["omega"],
            event_mask=value["event_mask"],
        )


@dataclass(frozen=True)
class _HistorySelection:
    """Selected parent-history slice for a query-time chunk.

    ``start`` and ``end`` are column bounds on the padded history axis. The
    optional tensors carry full-sequence event positions and per-query counts so
    downstream chunk computations can apply exact per-query truncation masks.
    """

    start: int
    end: int
    positions: Optional[torch.Tensor] = None
    previous_counts: Optional[torch.Tensor] = None

    @property
    def empty(self) -> bool:
        """Whether this selection contains no candidate history events."""
        return self.end <= self.start

    @classmethod
    def empty_for(
        cls,
        positions: Optional[torch.Tensor],
        previous_counts: Optional[torch.Tensor],
    ) -> "_HistorySelection":
        """Create an empty selection while preserving reusable context tensors."""
        return cls(0, 0, positions, previous_counts)

    def iter_chunks(self, chunk_size: int):
        """Yield absolute history-axis chunks inside ``[start, end)``."""
        for rel_start in range(0, self.end - self.start, int(chunk_size)):
            rel_end = min(rel_start + int(chunk_size), self.end - self.start)
            yield self.start + rel_start, self.start + rel_end

    def position_chunk(self, start: int, end: int) -> Optional[torch.Tensor]:
        """Return the event-position slice aligned to a history chunk."""
        if self.positions is None:
            return None
        return self.positions[:, start:end]


@dataclass
class _SequenceEventAccumulator:
    """Incrementally collect simulated event times and magnitudes."""

    times: list[float] = field(default_factory=list)
    magnitudes: list[float] = field(default_factory=list)

    def append(self, event_time: float, magnitude: float) -> None:
        self.times.append(float(event_time))
        self.magnitudes.append(float(magnitude))

    def to_sequence(self, *, t_start: float, t_end: float) -> Sequence:
        if not self.times:
            inter_times = np.asarray([float(t_end) - float(t_start)], dtype=np.float32)
            magnitudes = np.empty((0,), dtype=np.float32)
        else:
            arrival_times = np.asarray(self.times, dtype=np.float64)
            magnitudes = np.asarray(self.magnitudes, dtype=np.float32)
            inter_times = np.diff(
                arrival_times,
                prepend=[float(t_start)],
                append=[float(t_end)],
            ).astype(np.float32, copy=False)
        return Sequence(
            inter_times=inter_times,
            t_start=float(t_start),
            mag=magnitudes,
        )


__all__ = [
    "_HistorySelection",
    "_MagnitudeSamplingParams",
    "_ParentParameters",
    "_SequenceEventAccumulator",
    "_empty_initial_event_arrays",
    "_gen_mag",
    "_to_numpy_array",
]
