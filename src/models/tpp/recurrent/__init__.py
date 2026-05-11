"""Recurrent TPP subpackage.

Canonical locations:
- `model_v1.py`: legacy RTPP model (`RecurrentTPP`)
- `model_v2.py`: modular RTPP v2 model (`RecurrentTPPV2`)
- `sampling.py`: shared sampling mixin
- `utils.py`: recurrent utility helpers
- `src.models.tpp.common`: shared reusable building blocks
"""

from .model_v1 import RecurrentTPP
from .model_v2 import RecurrentTPPV2
from .sampling import RecurrentTPPSamplingMixin
from ..common.inter_time_decoding import WeibullMixtureDecoder
from ..common.recurrent_blocks import (
    FiLMContextFuse,
    RNNTPPBackbone,
    TimeHypernetMLP,
    build_time_hypernet,
)

__all__ = [
    "RecurrentTPP",
    "RecurrentTPPV2",
    "FiLMContextFuse",
    "RNNTPPBackbone",
    "TimeHypernetMLP",
    "build_time_hypernet",
    "RecurrentTPPSamplingMixin",
    "WeibullMixtureDecoder",
]
