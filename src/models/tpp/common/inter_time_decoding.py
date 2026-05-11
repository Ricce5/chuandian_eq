"""Inter-time distribution decoding utilities for TPP models.

This module contains reusable decoders that map model context into valid
inter-event-time distributions. It is intentionally independent from concrete
model backbones (RNN/Mixer/Transformer/etc.) to reduce coupling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import src.distributions as dist


class WeibullMixtureDecoder:
    """Shared decoder for Weibull-mixture inter-time distributions.

    This class separates two concerns:
    1) mapping context -> raw parameters via an external hypernetwork;
    2) mapping raw parameters -> valid mixture-distribution parameters.

    It supports both the legacy RTPP parameterization and Oracle-style
    parameterization while keeping backward compatibility with existing configs.
    """

    _VALID_PARAMETRIZATIONS = {"legacy", "oracle"}
    _VALID_SCALE_RANGES = {"positive", "decay"}

    def __init__(
        self,
        *,
        num_components: Optional[int],
        parametrization: str = "legacy",
        scale_range: str = "positive",
        normalize_mixture_logits: bool = True,
    ) -> None:
        self.num_components = None if num_components is None else int(num_components)
        if self.num_components is not None and self.num_components < 1:
            raise ValueError("num_components must be >= 1 when provided.")

        self.parametrization = str(parametrization).strip().lower()
        if self.parametrization not in self._VALID_PARAMETRIZATIONS:
            raise ValueError(
                f"parametrization must be one of {sorted(self._VALID_PARAMETRIZATIONS)} "
                f"(got {parametrization!r})."
            )

        self.scale_range = str(scale_range).strip().lower()
        if self.scale_range not in self._VALID_SCALE_RANGES:
            raise ValueError(
                f"scale_range must be one of {sorted(self._VALID_SCALE_RANGES)} "
                f"(got {scale_range!r})."
            )
        self.normalize_mixture_logits = bool(normalize_mixture_logits)

        self._legacy_min_pre_activation = -5.0
        self._oracle_min_shape = 1e-1
        self._oracle_max_shape = 1e2
        self._oracle_min_log_scale = -20.0
        self._oracle_max_log_scale = 20.0
        self._oracle_log_mean_scale = 5.0 * 2.302585093

    def from_context(
        self,
        context: torch.Tensor,
        hypernet_time: torch.nn.Module,
    ) -> dist.MixtureSameFamily:
        params = hypernet_time(context)
        return self.from_params(params)

    def from_params(self, params: torch.Tensor) -> dist.MixtureSameFamily:
        num_components = self._resolve_num_components(params)
        if self.parametrization == "legacy":
            return self._decode_legacy(params, num_components=num_components)
        return self._decode_oracle_style(params, num_components=num_components)

    def _resolve_num_components(self, params: torch.Tensor) -> int:
        if self.num_components is not None:
            expected = 3 * self.num_components
            got = params.shape[-1]
            if got != expected:
                raise ValueError(
                    f"Expected last param dimension = {expected} (3 * num_components), got {got}."
                )
            return self.num_components

        last_dim = params.shape[-1]
        if last_dim % 3 != 0:
            raise ValueError(
                f"Cannot infer num_components from params.shape[-1]={last_dim}; must be divisible by 3."
            )
        inferred = last_dim // 3
        if inferred < 1:
            raise ValueError(f"Inferred num_components must be >= 1, got {inferred}.")
        return inferred

    def _decode_legacy(self, params: torch.Tensor, *, num_components: int) -> dist.MixtureSameFamily:
        scale_raw, shape_raw, weight_logits = torch.split(
            params,
            [num_components, num_components, num_components],
            dim=-1,
        )
        scale_raw = scale_raw.clamp_min(self._legacy_min_pre_activation)
        if self.scale_range == "decay":
            scale = F.softplus(scale_raw)
            scale = scale + (scale.clamp_max(1.0) - scale).detach()
        else:
            scale = F.softplus(scale_raw)
        shape = F.softplus(shape_raw.clamp_min(self._legacy_min_pre_activation))
        if self.normalize_mixture_logits:
            weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def _decode_oracle_style(
        self,
        params: torch.Tensor,
        *,
        num_components: int,
    ) -> dist.MixtureSameFamily:
        log_mean, shape_raw, weight_logits = torch.split(
            params,
            [num_components, num_components, num_components],
            dim=-1,
        )
        log_mean = torch.nan_to_num(log_mean, nan=0.0, posinf=20.0, neginf=-20.0)
        shape_raw = torch.nan_to_num(shape_raw, nan=0.0, posinf=20.0, neginf=-20.0)
        weight_logits = torch.nan_to_num(weight_logits, nan=0.0, posinf=0.0, neginf=0.0)

        log_mean = torch.tanh(log_mean) * self._oracle_log_mean_scale
        shape = F.softplus(shape_raw).clamp(self._oracle_min_shape, self._oracle_max_shape)
        if self.normalize_mixture_logits:
            weight_logits = F.log_softmax(weight_logits, dim=-1)

        log_scale = (torch.lgamma(1 + shape.reciprocal()) - log_mean) * shape
        log_scale = torch.nan_to_num(
            log_scale,
            nan=0.0,
            posinf=self._oracle_max_log_scale,
            neginf=self._oracle_min_log_scale,
        ).clamp(self._oracle_min_log_scale, self._oracle_max_log_scale)
        scale = torch.exp(log_scale)

        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )


__all__ = ["WeibullMixtureDecoder"]
