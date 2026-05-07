from typing import Dict

import torch
import src.data


class SpatialMagnitudeTimeAdapter:
    """Adapter: spatial+magnitude as mark, plus event time."""

    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),
            "event_time": bx[:, :, 1],
        }


class LocationMagnitudeTimeAdapter:
    """Adapter: separate location, magnitude and time tensors."""

    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_loc": bx[:, :, 3:5],
            "event_mag": bx[:, :, 2:3],
            "event_time": bx[:, :, 1],
        }


class SpatialMagnitudeTimeAdapterWithAccessor:
    """Same as `SpatialMagnitudeTimeAdapter` but exposes `get_extra_inputs` for time."""

    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),
            "event_time": bx[:, :, 1],
        }

    def get_extra_inputs(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": bx[:, :, 1]}


class SpatialMagnitudeTimeBatchAdapter:
    """Adapter for batch objects exposing `loc`, `mag`, `arrival_times`."""

    def __call__(self, batch: src.data.Batch) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat([batch.loc, batch.mag[..., None]], dim=-1),
            "event_time": batch.arrival_times,
        }


class TypeTimeBatchAdapter:
    """Adapter for categorical event types + times in batch objects."""

    def __call__(self, batch: src.data.Batch) -> Dict[str, torch.Tensor]:
        return {
            "event_type": batch.type_seq,
            "event_time": batch.arrival_times,
        }


class MagnitudeTimeAdapter:
    """Adapter: magnitude-only mark plus time."""

    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": bx[:, :, 2:3],
            "event_time": bx[:, :, 1],
        }


class MagnitudeTimeAdapterWithAccessor:
    """Magnitude-only adapter with extra time accessor."""

    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_mark": bx[:, :, 2:3], "event_time": bx[:, :, 1]}

    def get_extra_inputs(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": bx[:, :, 1]}
