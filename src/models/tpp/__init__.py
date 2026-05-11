"""Temporal Point Process (TPP) model package."""

import importlib
import sys

from . import common, etas, etas_zhuang, njdtpp, oracle, recurrent
from .tpp_model import TPPModel

_LEGACY_IMPORT_ALIASES = {
    "inter_time_decoders": "common.inter_time_decoding",
    "oracle_components": "common.oracle_blocks",
    "recurrent_components": "common.recurrent_blocks",
    "recurrent_v2": "recurrent.model_v2",
    "recurrent_sampling": "recurrent.sampling",
}


def _install_legacy_import_aliases() -> None:
    prefix = __name__
    for old_suffix, new_suffix in _LEGACY_IMPORT_ALIASES.items():
        old_name = f"{prefix}.{old_suffix}"
        if old_name in sys.modules:
            continue
        new_name = f"{prefix}.{new_suffix}"
        sys.modules[old_name] = importlib.import_module(new_name)


_install_legacy_import_aliases()

__all__ = [
    "TPPModel",
    "common",
    "recurrent",
    "etas",
    "etas_zhuang",
    "njdtpp",
    "oracle",
]
