"""Catalog module registry.

Importing this package triggers side-effect registration of catalog classes
decorated with ``@Catalog.register(...)``.
"""

from . import (
    azdx,
    basel,
    ccl,
    china_array,
    chuandian,
    cooper_basin,
    forge,
    geysers,
    hauksson,
    pnr,
    qtm,
    scedc,
    ssfs,
    st1,
    synthetic_etas,
    virtual_induced,
    white,
)

__all__ = [
    "azdx",
    "basel",
    "ccl",
    "china_array",
    "chuandian",
    "cooper_basin",
    "forge",
    "geysers",
    "hauksson",
    "pnr",
    "qtm",
    "scedc",
    "ssfs",
    "st1",
    "synthetic_etas",
    "virtual_induced",
    "white",
]
