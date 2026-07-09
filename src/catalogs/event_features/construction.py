from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Protocol

import pandas as pd
import torch

from .oracle import OracleEventFeatureBuilder


class EventFeatureBuilder(Protocol):
    """Interface for catalog event-feature builders."""

    def build(self, *, df_eq: pd.DataFrame, df_inj: pd.DataFrame) -> Dict[str, torch.Tensor]:
        ...


BuilderCtor = Callable[..., EventFeatureBuilder]


class EventFeatureBuilderFactory:
    """Create event-feature builders from a small explicit registry."""

    _REGISTRY: Mapping[str, BuilderCtor] = {
        "oracle": OracleEventFeatureBuilder,
    }

    @classmethod
    def create(
        cls,
        builder_name: Optional[str],
        *,
        time_unit_minutes: float,
        builder_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Optional[EventFeatureBuilder]:
        if builder_name is None:
            return None

        normalized_name = str(builder_name).strip().lower()
        if not normalized_name:
            raise ValueError("event_feature_builder cannot be an empty string.")

        builder_ctor = cls._REGISTRY.get(normalized_name)
        if builder_ctor is None:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unsupported event_feature_builder={builder_name!r}. "
                f"Supported values: {available}."
            )

        return builder_ctor(time_unit_minutes=time_unit_minutes, **dict(builder_cfg or {}))


class EventFeatureConstruction:
    """Adapter used by catalogs to materialize per-event feature tensors."""

    def __init__(self, builder: Optional[EventFeatureBuilder]) -> None:
        self._builder = builder

    def build(self, *, df_eq: pd.DataFrame, df_inj: pd.DataFrame) -> Dict[str, torch.Tensor]:
        if self._builder is None:
            return {}
        return self._builder.build(df_eq=df_eq, df_inj=df_inj)
