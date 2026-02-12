"""Catalog module registry.

Importing this package triggers side-effect registration of catalog classes
decorated with ``@Catalog.register(...)``.
"""

from . import (
    azdx,
    china_array,
    chuandian,
    geysers,
    hauksson,
    pnr,
    qtm,
    scedc,
    synthetic_etas,
    visualization,
    white,
)

__all__ = [
    "azdx",
    "china_array",
    "chuandian",
    "geysers",
    "hauksson",
    "pnr",
    "qtm",
    "scedc",
    "synthetic_etas",
    "visualization",
    "white",
]
