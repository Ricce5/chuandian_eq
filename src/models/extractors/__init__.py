"""Representation extractor registry entrypoint."""

from . import (
    attention_pooling,
    attn_pool_with_time,
    attn_time_biased,
    attn_time_biased_mh,
    last_step,
    pma_time_biased,
)
from .base import RepresentationExtractor

__all__ = ["RepresentationExtractor"]
