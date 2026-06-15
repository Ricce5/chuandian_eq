"""Fixed triggering basis kernels for NETAS."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .types import _to_numpy_array


class _KernelFamily:
    """Stateless interface for a fixed normalized triggering-kernel family."""

    name: str = ""
    parameter_names: tuple[str, ...] = ()
    size_parameter_name: str = ""

    @staticmethod
    def _broadcast_params(lag: torch.Tensor, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Broadcast basis parameters over ``lag`` plus a trailing basis axis."""
        shape = (*([1] * lag.ndim), -1)
        return tuple(param.view(*shape) for param in params)

    @staticmethod
    def _interpolate_truncated_uniform(
        lower_cdf: np.ndarray | float,
        upper_cdf: np.ndarray | float,
        u: np.ndarray,
    ) -> np.ndarray:
        values = lower_cdf + (upper_cdf - lower_cdf) * u
        return np.clip(values, 1e-12, 1.0 - 1e-12)

    @classmethod
    def pdf(cls, lag: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis PDFs at nonnegative lags."""
        raise NotImplementedError

    @classmethod
    def cdf(cls, lag: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis CDFs at nonnegative lags."""
        raise NotImplementedError

    @classmethod
    def interval_mass_np(
        cls,
        lower: float,
        upper: float,
        *params: np.ndarray,
    ) -> np.ndarray:
        """Return basis mass over ``[lower, upper]`` in NumPy space."""
        raise NotImplementedError

    @classmethod
    def component_cdf_np(
        cls,
        lag: np.ndarray | float,
        params: tuple[float, ...],
    ) -> np.ndarray:
        """Evaluate one component CDF in NumPy space."""
        raise NotImplementedError

    @classmethod
    def sample_from_component_cdf_np(
        cls,
        cdf_values: np.ndarray,
        params: tuple[float, ...],
    ) -> np.ndarray:
        """Invert one component CDF for already-interpolated CDF values."""
        raise NotImplementedError

    @classmethod
    def sample_truncated_np(
        cls,
        *,
        params: tuple[float, ...],
        lower: float,
        upper: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample lags from one component conditioned on ``lower <= lag <= upper``."""
        if int(size) <= 0:
            return np.empty((0,), dtype=np.float64)

        lower = max(float(lower), 0.0)
        upper = max(float(upper), lower)
        cdf_values = cls._interpolate_truncated_uniform(
            cls.component_cdf_np(lower, params),
            cls.component_cdf_np(upper, params),
            rng.random(int(size)),
        )
        return cls.sample_from_component_cdf_np(cdf_values, params)

    @classmethod
    def sample_truncated_vectorized_np(
        cls,
        *,
        params: tuple[float, ...],
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Vectorized truncated sampling for one component and per-parent bounds."""
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have identical shapes.")
        if lower.size == 0:
            return np.empty((0,), dtype=np.float64)

        lower = np.clip(lower, a_min=0.0, a_max=None)
        upper = np.maximum(np.asarray(upper, dtype=np.float64), lower)
        cdf_values = cls._interpolate_truncated_uniform(
            cls.component_cdf_np(lower, params),
            cls.component_cdf_np(upper, params),
            rng.random(lower.shape),
        )
        return cls.sample_from_component_cdf_np(cdf_values, params)


class _ExponentialKernel(_KernelFamily):
    """Exponential basis kernel family."""

    name = "exponential"
    parameter_names = ("rates",)
    size_parameter_name = "rates"

    @classmethod
    def pdf(cls, lag: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        (rates,) = cls._broadcast_params(lag, rates)
        return rates * torch.exp(-rates * lag.unsqueeze(-1))

    @classmethod
    def cdf(cls, lag: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        (rates,) = cls._broadcast_params(lag, rates)
        return 1.0 - torch.exp(-rates * lag.unsqueeze(-1))

    @staticmethod
    def tail_np(lag: np.ndarray | float, rate: np.ndarray | float) -> np.ndarray:
        return np.exp(-rate * lag)

    @classmethod
    def cdf_np(cls, lag: np.ndarray | float, rate: np.ndarray | float) -> np.ndarray:
        return 1.0 - cls.tail_np(lag, rate)

    @staticmethod
    def icdf_np(cdf_values: np.ndarray, rate: float) -> np.ndarray:
        return -np.log1p(-cdf_values) / rate

    @classmethod
    def interval_mass_np(
        cls,
        lower: float,
        upper: float,
        rates: np.ndarray,
    ) -> np.ndarray:
        lower = max(float(lower), 0.0)
        upper = max(float(upper), lower)
        return cls.tail_np(lower, rates) - cls.tail_np(upper, rates)

    @classmethod
    def component_cdf_np(
        cls,
        lag: np.ndarray | float,
        params: tuple[float, ...],
    ) -> np.ndarray:
        (rate,) = params
        return cls.cdf_np(lag, rate)

    @classmethod
    def sample_from_component_cdf_np(
        cls,
        cdf_values: np.ndarray,
        params: tuple[float, ...],
    ) -> np.ndarray:
        (rate,) = params
        return cls.icdf_np(cdf_values, rate)


class _LomaxKernel(_KernelFamily):
    """Lomax basis kernel family."""

    name = "lomax"
    parameter_names = ("scales", "shapes")
    size_parameter_name = "scales"

    @staticmethod
    def _one_plus(lag: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return 1.0 + lag.unsqueeze(-1) / scales

    @classmethod
    def pdf(
        cls,
        lag: torch.Tensor,
        scales: torch.Tensor,
        shapes: torch.Tensor,
    ) -> torch.Tensor:
        scales, shapes = cls._broadcast_params(lag, scales, shapes)
        one_plus = cls._one_plus(lag, scales)
        return (shapes / scales) * one_plus.pow(-(shapes + 1.0))

    @classmethod
    def cdf(
        cls,
        lag: torch.Tensor,
        scales: torch.Tensor,
        shapes: torch.Tensor,
    ) -> torch.Tensor:
        scales, shapes = cls._broadcast_params(lag, scales, shapes)
        one_plus = cls._one_plus(lag, scales)
        return 1.0 - one_plus.pow(-shapes)

    @staticmethod
    def tail_np(
        lag: np.ndarray | float,
        scale: np.ndarray | float,
        shape: np.ndarray | float,
    ) -> np.ndarray:
        return (1.0 + lag / scale) ** (-shape)

    @classmethod
    def cdf_np(
        cls,
        lag: np.ndarray | float,
        scale: np.ndarray | float,
        shape: np.ndarray | float,
    ) -> np.ndarray:
        return 1.0 - cls.tail_np(lag, scale, shape)

    @staticmethod
    def icdf_np(
        cdf_values: np.ndarray,
        scale: float,
        shape: float,
    ) -> np.ndarray:
        tail = np.clip(1.0 - cdf_values, 1e-12, 1.0)
        return scale * (tail ** (-1.0 / shape) - 1.0)

    @classmethod
    def interval_mass_np(
        cls,
        lower: float,
        upper: float,
        scales: np.ndarray,
        shapes: np.ndarray,
    ) -> np.ndarray:
        lower = max(float(lower), 0.0)
        upper = max(float(upper), lower)
        return cls.tail_np(lower, scales, shapes) - cls.tail_np(upper, scales, shapes)

    @classmethod
    def component_cdf_np(
        cls,
        lag: np.ndarray | float,
        params: tuple[float, ...],
    ) -> np.ndarray:
        scale, shape = params
        return cls.cdf_np(lag, scale, shape)

    @classmethod
    def sample_from_component_cdf_np(
        cls,
        cdf_values: np.ndarray,
        params: tuple[float, ...],
    ) -> np.ndarray:
        scale, shape = params
        return cls.icdf_np(cdf_values, scale, shape)


_KERNEL_FAMILIES = {
    _ExponentialKernel.name: _ExponentialKernel,
    _LomaxKernel.name: _LomaxKernel,
}


class FixedKernelBasis(nn.Module):
    """Normalized basis kernels for the NETAS trigger.

    By default the basis is fixed. Setting ``learnable`` adds bounded log-space
    deltas on top of the fixed positive parameters, preserving the ETAS-style
    nonnegative normalized kernels while allowing modest data-driven adaptation.
    """

    def __init__(
        self,
        *,
        family: str = "exponential",
        num_basis: int = 4,
        rates: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        shapes: Optional[torch.Tensor] = None,
        rate_min: float = 1e-3,
        rate_max: float = 1e1,
        scale_min: float = 1e-2,
        scale_max: float = 1e2,
        lomax_shape: float = 0.35,
        learnable: str = "fixed",
        max_log_deviation: Optional[float] = 0.0,
        learn_shapes: bool = False,
    ) -> None:
        super().__init__()
        family_name = str(family).strip().lower()
        if family_name not in _KERNEL_FAMILIES:
            raise ValueError("family must be one of ['exponential', 'lomax'].")
        if int(num_basis) < 1:
            raise ValueError("num_basis must be >= 1.")
        self.family = family_name
        self.kernel = _KERNEL_FAMILIES[family_name]
        learnable_name = str(learnable).strip().lower().replace("-", "_")
        learnable_aliases = {
            "none": "fixed",
            "false": "fixed",
            "fixed": "fixed",
            "global": "global",
            "global_scale": "global",
            "shared": "global",
            "per_basis": "per_basis",
            "basis": "per_basis",
            "full": "per_basis",
            "true": "per_basis",
        }
        if learnable_name not in learnable_aliases:
            raise ValueError(
                "learnable must be one of ['fixed', 'global', 'per_basis']."
            )
        self.learnable_mode = learnable_aliases[learnable_name]
        if max_log_deviation is None:
            self.max_log_deviation = 0.0
        else:
            if float(max_log_deviation) < 0.0:
                raise ValueError("max_log_deviation must be non-negative.")
            self.max_log_deviation = float(max_log_deviation)
        self.learn_shapes = bool(learn_shapes)

        if self.family == "exponential":
            if rates is None:
                if rate_min <= 0 or rate_max <= 0:
                    raise ValueError("rate_min/rate_max must be positive.")
                if rate_max < rate_min:
                    raise ValueError("rate_max must be >= rate_min.")
                if num_basis == 1:
                    rates = torch.tensor([float(rate_min)], dtype=torch.float32)
                else:
                    rates = torch.logspace(
                        math.log10(float(rate_min)),
                        math.log10(float(rate_max)),
                        steps=int(num_basis),
                        dtype=torch.float32,
                    )
            rates = torch.as_tensor(rates, dtype=torch.float32).flatten()
            if rates.numel() != int(num_basis):
                raise ValueError("rates must have shape [num_basis].")
            if torch.any(rates <= 0):
                raise ValueError("All exponential basis rates must be positive.")
            self.register_buffer("rates", rates)
            self._register_log_delta("rate_log_delta", rates)
        else:
            if scales is None:
                if scale_min <= 0 or scale_max <= 0:
                    raise ValueError("scale_min/scale_max must be positive.")
                if scale_max < scale_min:
                    raise ValueError("scale_max must be >= scale_min.")
                if num_basis == 1:
                    scales = torch.tensor([float(scale_min)], dtype=torch.float32)
                else:
                    scales = torch.logspace(
                        math.log10(float(scale_min)),
                        math.log10(float(scale_max)),
                        steps=int(num_basis),
                        dtype=torch.float32,
                    )
            if shapes is None:
                shapes = torch.full(
                    (int(num_basis),),
                    float(lomax_shape),
                    dtype=torch.float32,
                )
            scales = torch.as_tensor(scales, dtype=torch.float32).flatten()
            shapes = torch.as_tensor(shapes, dtype=torch.float32).flatten()
            if scales.numel() != int(num_basis) or shapes.numel() != int(num_basis):
                raise ValueError("scales/shapes must have shape [num_basis].")
            if torch.any(scales <= 0) or torch.any(shapes <= 0):
                raise ValueError("All Lomax basis scales/shapes must be positive.")
            self.register_buffer("scales", scales)
            self.register_buffer("shapes", shapes)
            self._register_log_delta("scale_log_delta", scales)
            if self.learn_shapes:
                self._register_log_delta("shape_log_delta", shapes)

    @property
    def num_basis(self) -> int:
        return int(getattr(self, self.kernel.size_parameter_name).numel())

    def _register_log_delta(self, name: str, base: torch.Tensor) -> None:
        if self.learnable_mode == "fixed":
            return
        shape = (1,) if self.learnable_mode == "global" else tuple(base.shape)
        self.register_parameter(
            name,
            nn.Parameter(torch.zeros(shape, dtype=base.dtype)),
        )

    def _bounded_log_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        if self.max_log_deviation > 0.0:
            return self.max_log_deviation * torch.tanh(raw_delta)
        return raw_delta

    def _apply_log_delta(self, base: torch.Tensor, name: str) -> torch.Tensor:
        raw_delta = getattr(self, name, None)
        if raw_delta is None:
            return base
        log_delta = self._bounded_log_delta(raw_delta).to(
            device=base.device,
            dtype=base.dtype,
        )
        if log_delta.numel() == 1:
            log_delta = log_delta.expand_as(base)
        return base * torch.exp(log_delta)

    @property
    def effective_rates(self) -> torch.Tensor:
        return self._apply_log_delta(self.rates, "rate_log_delta")

    @property
    def effective_scales(self) -> torch.Tensor:
        return self._apply_log_delta(self.scales, "scale_log_delta")

    @property
    def effective_shapes(self) -> torch.Tensor:
        return self._apply_log_delta(self.shapes, "shape_log_delta")

    @property
    def _kernel_params(self) -> tuple[torch.Tensor, ...]:
        """Return effective torch parameters in the active kernel's order."""
        return tuple(
            getattr(self, f"effective_{name}") for name in self.kernel.parameter_names
        )

    @property
    def _kernel_np_params(self) -> tuple[np.ndarray, ...]:
        """Return effective NumPy parameters in the active kernel's order."""
        return tuple(
            _to_numpy_array(param).astype(np.float64, copy=False)
            for param in self._kernel_params
        )

    def _component_np_params(self, component_idx: int) -> tuple[float, ...]:
        """Return scalar parameters for one basis component as NumPy-friendly floats."""
        idx = int(component_idx)
        return tuple(float(param[idx].item()) for param in self._kernel_params)

    def pdf(self, lag: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis PDFs at nonnegative lags."""
        return self.kernel.pdf(torch.clamp_min(lag, 0.0), *self._kernel_params)

    def cdf(self, lag: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis CDFs at nonnegative lags."""
        return self.kernel.cdf(torch.clamp_min(lag, 0.0), *self._kernel_params)

    def interval_mass(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Return basis mass over ``[lower, upper]`` in torch space."""
        lower = torch.clamp_min(lower, 0.0)
        upper = torch.clamp_min(upper, 0.0)
        return (self.cdf(upper) - self.cdf(lower)).clamp_min(0.0)

    def interval_mass_np(self, lower: float, upper: float) -> np.ndarray:
        """Return basis mass over ``[lower, upper]`` in NumPy space."""
        return self.kernel.interval_mass_np(lower, upper, *self._kernel_np_params)

    def sample_truncated_np(
        self,
        *,
        component_idx: int,
        lower: float,
        upper: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample lags from one component conditioned on ``lower <= lag <= upper``."""
        return self.kernel.sample_truncated_np(
            params=self._component_np_params(component_idx),
            lower=lower,
            upper=upper,
            size=size,
            rng=rng,
        )

    def sample_truncated_vectorized_np(
        self,
        *,
        component_idx: int,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Vectorized truncated sampling for one component and per-parent bounds."""
        return self.kernel.sample_truncated_vectorized_np(
            params=self._component_np_params(component_idx),
            lower=lower,
            upper=upper,
            rng=rng,
        )


__all__ = ["FixedKernelBasis"]
