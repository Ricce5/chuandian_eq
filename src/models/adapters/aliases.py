"""Backward-compatible adapter aliases used by existing builders/configs."""

from .basic import (
    LocationMagnitudeTimeAdapter,
    MagnitudeTimeAdapter,
    MagnitudeTimeAdapterWithAccessor,
    SpatialMagnitudeTimeAdapter,
    SpatialMagnitudeTimeAdapterWithAccessor,
    SpatialMagnitudeTimeBatchAdapter,
    TypeTimeBatchAdapter,
)
from .thp import THPBatchAdapter, THPLogDeltaTBatchAdapter


SM_T_InputAdapter = SpatialMagnitudeTimeAdapter
M_T_InputAdapter = MagnitudeTimeAdapter
M_T_InputAdapterWithTime = MagnitudeTimeAdapterWithAccessor
S_T_M_InputAdapter = LocationMagnitudeTimeAdapter
SM_T_InputAdapterWithTime = SpatialMagnitudeTimeAdapterWithAccessor
THP_BatchInputAdapter = THPBatchAdapter
THP_Logdeltat_BatchInputAdapter = THPLogDeltaTBatchAdapter

__all__ = [
    "LocationMagnitudeTimeAdapter",
    "MagnitudeTimeAdapter",
    "MagnitudeTimeAdapterWithAccessor",
    "M_T_InputAdapter",
    "M_T_InputAdapterWithTime",
    "SM_T_InputAdapter",
    "SM_T_InputAdapterWithTime",
    "S_T_M_InputAdapter",
    "SpatialMagnitudeTimeAdapter",
    "SpatialMagnitudeTimeAdapterWithAccessor",
    "SpatialMagnitudeTimeBatchAdapter",
    "THPBatchAdapter",
    "THPLogDeltaTBatchAdapter",
    "THP_BatchInputAdapter",
    "THP_Logdeltat_BatchInputAdapter",
    "TypeTimeBatchAdapter",
]
