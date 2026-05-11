from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from src.utils.registrable import Registrable


class BValueUpdaterBase(Registrable, ABC):
    """Common interface for b-value updater components."""

    _default_impl = "bayesian_gr"

    @classmethod
    def create(cls, name: str | None = None, **kwargs):
        updater_name = cls._default_impl if name is None else str(name)
        if updater_name is None:
            raise ValueError("No updater name provided and no default implementation is configured.")
        constructor = cls.by_name(updater_name)
        return constructor(**kwargs)

    @classmethod
    def build(
        cls,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        params = dict(config or {})
        params.update(kwargs)
        return cls.create(name=name, **params)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        **kwargs,
    ):
        """Instantiate updater from config with optional registry name fields."""
        params = dict(config or {})
        resolved_name = name
        if resolved_name is None:
            raw_name = params.pop("name", None)
            if raw_name is None:
                raw_name = params.pop("type", None)
            if raw_name is not None:
                resolved_name = str(raw_name)
        params.update(kwargs)
        return cls.create(name=resolved_name, **params)

    @abstractmethod
    def fit(self, seq: Any, prefix: str = "") -> dict[str, torch.Tensor]:
        """Fit/update state from a sequence-like input."""

    def update_one(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not support online update.")

    def sample_b_value(self, mode: str = "mean") -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not support b-value sampling.")

    def expand_state(self, batch_size: int):
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch state expansion.")

    def clone_for_batch(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [copy.deepcopy(self) for _ in range(batch_size)]
