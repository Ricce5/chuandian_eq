"""Input adapters grouped by task domain.

This package keeps compatibility with the old flat `src.models.input_adapters`
module by re-exporting the same names and aliases.
"""

from .aliases import (
    M_T_InputAdapter,
    M_T_InputAdapterWithTime,
    SM_T_InputAdapter,
    SM_T_InputAdapterWithTime,
    S_T_M_InputAdapter,
    THP_BatchInputAdapter,
    THP_Logdeltat_BatchInputAdapter,
)
from .basic import (
    LocationMagnitudeTimeAdapter,
    MagnitudeTimeAdapter,
    MagnitudeTimeAdapterWithAccessor,
    SpatialMagnitudeTimeAdapter,
    SpatialMagnitudeTimeAdapterWithAccessor,
    SpatialMagnitudeTimeBatchAdapter,
    TypeTimeBatchAdapter,
)
from .mixer_task import MixerAdapter
from .mixer_tpp import MixerBatchAdapter
from .rnn_task import RTPPTaskInputAdapter
from .thp import THPBatchAdapter, THPLogDeltaTBatchAdapter

__all__ = [
    "LocationMagnitudeTimeAdapter",
    "MagnitudeTimeAdapter",
    "MagnitudeTimeAdapterWithAccessor",
    "M_T_InputAdapter",
    "M_T_InputAdapterWithTime",
    "MixerAdapter",
    "MixerBatchAdapter",
    "RTPPTaskInputAdapter",
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
