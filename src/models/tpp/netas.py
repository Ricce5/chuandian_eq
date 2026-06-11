"""Neural-ETAS with fixed nonnegative basis kernels.

This module implements a branching-preserving Hawkes / ETAS trigger of the form

    λ_tr(t) = Σ_{i: t_i < t} η_i Σ_{ℓ=1}^L ω_{iℓ} q_ℓ(t - t_i),

where the parent-event productivity and basis-mixture weights are controlled by
an event encoder state h_i:

    η_i = η_max σ(a + α (m_i - M_c) + v^T h_i),
    ω_i = softmax(W h_i).

The basis kernels q_ℓ are fixed, nonnegative, and normalized. This keeps the
trigger self-exciting in the ETAS sense while preserving closed-form integrals
and branching-process sampling.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src
from src.data.batch import Batch, get_mask, pad_sequence
from src.data.sequence import Sequence

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

try:
    from src.models.mamba.mamba_time import MambaTime
except ImportError:
    MambaTime = None

from .common.recurrent_blocks import RNNTPPBackbone
from .tpp_model import TPPModel

logger = logging.getLogger(__name__)


def _to_numpy_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _gen_mag(
    rng: np.random.Generator,
    shape: int,
    *,
    b: float,
    m_min: float,
    m_max: float,
) -> np.ndarray:
    if shape <= 0:
        return np.empty((0,), dtype=np.float64)
    u = rng.random(shape)
    mag = (
        -1.0
        / b
        * np.log10(
            -u * (10.0 ** (-b * m_min) - 10.0 ** (-b * m_max))
            + 10.0 ** (-b * m_min)
        )
    )
    return mag.astype(np.float64, copy=False)


def _empty_initial_event_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.float32),
    )


@dataclass(frozen=True)
class _MagnitudeSamplingParams:
    b: float
    m_min: float
    m_max: float


@dataclass(frozen=True)
class _ParentParameters:
    context: torch.Tensor
    eta: torch.Tensor
    omega: torch.Tensor
    event_mask: torch.Tensor

    @property
    def event_mask_bool(self) -> torch.Tensor:
        return self.event_mask.bool()

    @property
    def weighted_mixture(self) -> torch.Tensor:
        return self.eta.unsqueeze(-1) * self.omega

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "context": self.context,
            "eta": self.eta,
            "omega": self.omega,
            "event_mask": self.event_mask,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, torch.Tensor]) -> "_ParentParameters":
        return cls(
            context=value["context"],
            eta=value["eta"],
            omega=value["omega"],
            event_mask=value["event_mask"],
        )


@dataclass
class _SequenceEventAccumulator:
    times: list[float] = field(default_factory=list)
    magnitudes: list[float] = field(default_factory=list)

    def append(self, event_time: float, magnitude: float) -> None:
        self.times.append(float(event_time))
        self.magnitudes.append(float(magnitude))

    def to_sequence(self, *, t_start: float, t_end: float) -> Sequence:
        if not self.times:
            inter_times = np.asarray([float(t_end) - float(t_start)], dtype=np.float32)
            magnitudes = np.empty((0,), dtype=np.float32)
        else:
            arrival_times = np.asarray(self.times, dtype=np.float64)
            magnitudes = np.asarray(self.magnitudes, dtype=np.float32)
            inter_times = np.diff(
                arrival_times,
                prepend=[float(t_start)],
                append=[float(t_end)],
            ).astype(np.float32, copy=False)
        return Sequence(
            inter_times=inter_times,
            t_start=float(t_start),
            mag=magnitudes,
        )


def masked_select_per_row(
    matrix: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.shape != mask.shape or matrix.ndim != 2:
        raise ValueError(
            "masked_select_per_row expects matrix/mask with identical 2D shapes."
        )
    selected_rows = []
    for matrix_row, mask_row in zip(matrix, mask.bool()):
        selected_rows.append(matrix_row.masked_select(mask_row))
    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(row) for row in selected_rows])
    return new_matrix, new_mask.float()


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
        if family_name not in {"exponential", "lomax"}:
            raise ValueError("family must be one of ['exponential', 'lomax'].")
        if int(num_basis) < 1:
            raise ValueError("num_basis must be >= 1.")
        self.family = family_name
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
        if self.family == "exponential":
            return int(self.rates.numel())
        return int(self.scales.numel())

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

    def pdf(self, lag: torch.Tensor) -> torch.Tensor:
        lag = torch.clamp_min(lag, 0.0)
        if self.family == "exponential":
            rates = self.effective_rates.view(*([1] * lag.ndim), -1)
            return rates * torch.exp(-rates * lag.unsqueeze(-1))

        scales = self.effective_scales.view(*([1] * lag.ndim), -1)
        shapes = self.effective_shapes.view(*([1] * lag.ndim), -1)
        one_plus = 1.0 + lag.unsqueeze(-1) / scales
        return (shapes / scales) * one_plus.pow(-(shapes + 1.0))

    def cdf(self, lag: torch.Tensor) -> torch.Tensor:
        lag = torch.clamp_min(lag, 0.0)
        if self.family == "exponential":
            rates = self.effective_rates.view(*([1] * lag.ndim), -1)
            return 1.0 - torch.exp(-rates * lag.unsqueeze(-1))

        scales = self.effective_scales.view(*([1] * lag.ndim), -1)
        shapes = self.effective_shapes.view(*([1] * lag.ndim), -1)
        one_plus = 1.0 + lag.unsqueeze(-1) / scales
        return 1.0 - one_plus.pow(-shapes)

    def interval_mass(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        lower = torch.clamp_min(lower, 0.0)
        upper = torch.clamp_min(upper, 0.0)
        return (self.cdf(upper) - self.cdf(lower)).clamp_min(0.0)

    def interval_mass_np(self, lower: float, upper: float) -> np.ndarray:
        lower = max(float(lower), 0.0)
        upper = max(float(upper), lower)
        if self.family == "exponential":
            rates = _to_numpy_array(self.effective_rates).astype(np.float64, copy=False)
            return np.exp(-rates * lower) - np.exp(-rates * upper)

        scales = _to_numpy_array(self.effective_scales).astype(np.float64, copy=False)
        shapes = _to_numpy_array(self.effective_shapes).astype(np.float64, copy=False)
        return (1.0 + lower / scales) ** (-shapes) - (1.0 + upper / scales) ** (-shapes)

    def sample_truncated_np(
        self,
        *,
        component_idx: int,
        lower: float,
        upper: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if int(size) <= 0:
            return np.empty((0,), dtype=np.float64)

        lower = max(float(lower), 0.0)
        upper = max(float(upper), lower)
        u = rng.random(int(size))

        if self.family == "exponential":
            rate = float(self.effective_rates[int(component_idx)].item())
            f_lower = 1.0 - math.exp(-rate * lower)
            f_upper = 1.0 - math.exp(-rate * upper)
            u = f_lower + (f_upper - f_lower) * u
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            return -np.log1p(-u) / rate

        scale = float(self.effective_scales[int(component_idx)].item())
        shape = float(self.effective_shapes[int(component_idx)].item())
        f_lower = 1.0 - (1.0 + lower / scale) ** (-shape)
        f_upper = 1.0 - (1.0 + upper / scale) ** (-shape)
        u = f_lower + (f_upper - f_lower) * u
        tail = np.clip(1.0 - u, 1e-12, 1.0)
        return scale * (tail ** (-1.0 / shape) - 1.0)

    def sample_truncated_vectorized_np(
        self,
        *,
        component_idx: int,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have identical shapes.")
        if lower.size == 0:
            return np.empty((0,), dtype=np.float64)

        lower = np.clip(lower, a_min=0.0, a_max=None)
        upper = np.maximum(np.asarray(upper, dtype=np.float64), lower)
        u = rng.random(lower.shape)

        if self.family == "exponential":
            rate = float(self.effective_rates[int(component_idx)].item())
            f_lower = 1.0 - np.exp(-rate * lower)
            f_upper = 1.0 - np.exp(-rate * upper)
            u = f_lower + (f_upper - f_lower) * u
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            return -np.log1p(-u) / rate

        scale = float(self.effective_scales[int(component_idx)].item())
        shape = float(self.effective_shapes[int(component_idx)].item())
        f_lower = 1.0 - (1.0 + lower / scale) ** (-shape)
        f_upper = 1.0 - (1.0 + upper / scale) ** (-shape)
        u = f_lower + (f_upper - f_lower) * u
        tail = np.clip(1.0 - u, 1e-12, 1.0)
        return scale * (tail ** (-1.0 / shape) - 1.0)


class MambaNETASEncoder(nn.Module):
    """Mamba-based event encoder for NETAS."""

    def __init__(
        self,
        *,
        context_size: int,
        tau_mean: float,
        mag_mean: float,
        input_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        dropout: float = 0.0,
        time_preprocess: str = "legacy",
        log_tau_mean: Optional[float] = None,
        log_tau_std: Optional[float] = None,
        inter_time_min: float = 1e-10,
        inter_time_max: float = 1e10,
        d_state: Optional[int] = None,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        use_conv: bool = True,
        max_inference_len: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if MambaTime is None or InferenceParams is None:
            raise ImportError(
                "MambaNETASEncoder requires `mamba_ssm` and `src.models.mamba.mamba_time`."
            )
        if int(context_size) < 1:
            raise ValueError("context_size must be >= 1.")
        if float(inter_time_min) <= 0.0:
            raise ValueError("inter_time_min must be positive.")
        if float(inter_time_max) <= float(inter_time_min):
            raise ValueError("inter_time_max must exceed inter_time_min.")
        if int(max_inference_len) < 1:
            raise ValueError("max_inference_len must be >= 1.")

        self.context_size = int(context_size)
        self.input_magnitude = bool(input_magnitude)
        self.num_extra_features = num_extra_features
        self.time_preprocess = str(time_preprocess).strip().lower()
        if self.time_preprocess not in {"legacy", "oracle"}:
            raise ValueError("time_preprocess must be one of ['legacy', 'oracle'].")
        self.inter_time_min = float(inter_time_min)
        self.inter_time_max = float(inter_time_max)
        self.max_inference_len = int(max_inference_len)

        tau_mean_value = max(float(tau_mean), self.inter_time_min)
        if log_tau_mean is None:
            log_tau_mean = (
                math.log10(tau_mean_value)
                if self.time_preprocess == "oracle"
                else math.log(tau_mean_value)
            )
        if log_tau_std is None:
            log_tau_std = 1.0
        if float(log_tau_std) <= 0.0:
            raise ValueError("log_tau_std must be positive.")

        self.register_buffer("tau_mean", torch.tensor(tau_mean_value, dtype=torch.float32))
        self.register_buffer("log_tau_mean", torch.tensor(float(log_tau_mean), dtype=torch.float32))
        self.register_buffer("log_tau_std", torch.tensor(float(log_tau_std), dtype=torch.float32))
        self.register_buffer("mag_mean", torch.tensor(float(mag_mean), dtype=torch.float32))

        self.num_inputs = (
            1
            + int(self.input_magnitude)
            + (0 if self.num_extra_features is None else int(self.num_extra_features))
        )
        self.layer_idx = 0
        resolved_device = None if device is None else torch.device(device)
        effective_use_conv = bool(use_conv)
        if resolved_device is not None and resolved_device.type != "cuda":
            effective_use_conv = False
        self.mamba = MambaTime(
            d_model=self.context_size,
            d_state=int(d_state) if d_state is not None else max(1, self.context_size // 2),
            d_conv=int(d_conv),
            expand=int(expand),
            dt_rank=dt_rank,
            layer_idx=self.layer_idx,
            device=resolved_device,
            use_conv=effective_use_conv,
        )
        self.input_proj = nn.Linear(self.num_inputs, self.context_size)
        self.dropout = nn.Dropout(float(dropout))

    def _require_cuda_runtime(self) -> None:
        param_device = next(self.parameters()).device
        if param_device.type != "cuda":
            raise RuntimeError(
                "MambaNETASEncoder currently requires a CUDA device in this repository "
                "because the underlying `mamba_ssm` selective-scan kernels are CUDA-backed."
            )

    def encode_time(self, inter_times: torch.Tensor) -> torch.Tensor:
        inter_times = inter_times.clamp(self.inter_time_min, self.inter_time_max)
        if self.time_preprocess == "oracle":
            log_tau = torch.log10(inter_times).unsqueeze(-1)
            return (log_tau - self.log_tau_mean) / self.log_tau_std
        log_tau = torch.log(inter_times).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def encode_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        return mag.unsqueeze(-1) - self.mag_mean

    def encode_extra_features(self, extra_feat: torch.Tensor) -> torch.Tensor:
        return extra_feat

    def build_features(self, batch: Batch) -> torch.Tensor:
        feat_list = [self.encode_time(batch.inter_times)]
        if self.input_magnitude:
            if not hasattr(batch, "mag") or batch.mag is None:
                raise ValueError("MambaNETASEncoder requires `batch.mag` when input_magnitude=True.")
            feat_list.append(self.encode_magnitude(batch.mag))
        if self.num_extra_features is not None:
            if not hasattr(batch, "extra_features"):
                raise ValueError(
                    "num_extra_features is set but batch.extra_features is missing."
                )
            feat_list.append(self.encode_extra_features(batch.extra_features))
        features = torch.cat(feat_list, dim=-1).contiguous()
        return features * batch.input_mask[:, :, None]

    def get_parent_context(self, batch: Batch) -> torch.Tensor:
        self._require_cuda_runtime()
        features = self.build_features(batch)
        hidden_state = self.input_proj(features)
        context = self.mamba(
            hidden_state,
            inter_times=batch.inter_times.to(device=hidden_state.device, dtype=hidden_state.dtype),
        )
        context = context * batch.input_mask[:, :, None]
        return self.dropout(context)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._require_cuda_runtime()
        _ = device
        cache = self.mamba.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=self.max_inference_len,
            dtype=dtype,
        )
        return InferenceParams(
            max_seqlen=self.max_inference_len,
            max_batch_size=batch_size,
            key_value_memory_dict={self.layer_idx: cache},
        )

    def step_event(
        self,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
        prev_state,
    ):
        self._require_cuda_runtime()
        feat_list = [self.encode_time(inter_time.squeeze(-1) if inter_time.ndim == 3 else inter_time)]
        if self.input_magnitude:
            mag_tensor = magnitude.squeeze(-1) if magnitude.ndim == 3 else magnitude
            feat_list.append(self.encode_magnitude(mag_tensor))
        features = torch.cat(feat_list, dim=-1).contiguous()
        hidden_state = self.input_proj(features)
        inter_time_input = (
            inter_time.squeeze(-1) if inter_time.ndim == 3 else inter_time
        ).to(device=hidden_state.device, dtype=hidden_state.dtype)
        context = self.mamba(
            hidden_state,
            inference_params=prev_state,
            inter_times=inter_time_input,
        )
        prev_state.seqlen_offset += hidden_state.shape[1]
        return self.dropout(context), prev_state


class NETAS(TPPModel):
    """Neural-basis ETAS trigger model with fixed basis kernels."""

    def __init__(
        self,
        *,
        event_encoder: nn.Module,
        context_size: int,
        basis_family: str = "exponential",
        num_basis: int = 4,
        basis_rates: Optional[torch.Tensor] = None,
        basis_scales: Optional[torch.Tensor] = None,
        basis_shapes: Optional[torch.Tensor] = None,
        basis_rate_min: float = 1e-3,
        basis_rate_max: float = 1e1,
        basis_scale_min: float = 1e-2,
        basis_scale_max: float = 1e2,
        basis_lomax_shape: float = 0.35,
        basis_learnable: str = "fixed",
        basis_max_log_deviation: Optional[float] = 0.0,
        basis_learn_shapes: bool = False,
        base_rate_init: Union[float, torch.Tensor] = 0.02,
        productivity_alpha_init: float = 1.0,
        productivity_bias_init: float = 0.0,
        head_init_std: float = 1e-2,
        productivity_mode: str = "bounded",
        eta_max: float = 0.95,
        branching_penalty_weight: float = 0.0,
        branching_penalty_target: float = 0.95,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        mag_max: float = 10.0,
        device: Optional[torch.device] = None,
        bg_model=None,
        fix_mu: bool = False,
        fixed_mu_value: Optional[float] = None,
        loss_reduction: str = "per_time",
        query_chunk_size: int = 0,
        history_chunk_size: int = 0,
        loss_weights: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if int(context_size) < 1:
            raise ValueError("context_size must be >= 1.")
        productivity_mode = str(productivity_mode).strip().lower()
        if productivity_mode not in {"bounded", "softplus", "exp"}:
            raise ValueError("productivity_mode must be one of ['bounded', 'softplus', 'exp'].")
        if eta_max <= 0.0:
            raise ValueError("eta_max must be positive.")
        if productivity_mode == "bounded" and eta_max >= 1.0:
            raise ValueError(
                "eta_max must lie in (0, 1) when productivity_mode='bounded'."
            )
        if branching_penalty_weight < 0.0:
            raise ValueError("branching_penalty_weight must be non-negative.")
        if branching_penalty_target <= 0.0:
            raise ValueError("branching_penalty_target must be positive.")
        if head_init_std < 0.0:
            raise ValueError("head_init_std must be non-negative.")
        if mag_max <= mag_completeness:
            raise ValueError("mag_max must be greater than mag_completeness.")
        if richter_b <= 0:
            raise ValueError("richter_b must be strictly positive.")

        self.device = device if device is not None else torch.device("cpu")
        self.event_encoder = event_encoder
        self.context_size = int(context_size)
        self.bg_model = bg_model
        self.fix_mu = bool(fix_mu)
        self.reduction = str(loss_reduction)
        self.query_chunk_size = int(query_chunk_size)
        self.history_chunk_size = int(history_chunk_size)
        self.productivity_mode = productivity_mode
        self.branching_penalty_target = float(branching_penalty_target)

        self.basis = FixedKernelBasis(
            family=basis_family,
            num_basis=int(num_basis),
            rates=basis_rates,
            scales=basis_scales,
            shapes=basis_shapes,
            rate_min=float(basis_rate_min),
            rate_max=float(basis_rate_max),
            scale_min=float(basis_scale_min),
            scale_max=float(basis_scale_max),
            lomax_shape=float(basis_lomax_shape),
            learnable=basis_learnable,
            max_log_deviation=basis_max_log_deviation,
            learn_shapes=bool(basis_learn_shapes),
        )

        self.productivity_head = nn.Linear(self.context_size, 1, bias=False)
        self.productivity_alpha = nn.Parameter(
            torch.tensor(float(productivity_alpha_init), dtype=torch.float32)
        )
        self.productivity_bias = nn.Parameter(
            torch.tensor(float(productivity_bias_init), dtype=torch.float32)
        )
        self.mixture_head = nn.Linear(self.context_size, self.basis.num_basis, bias=False)
        if float(head_init_std) > 0.0:
            nn.init.normal_(
                self.productivity_head.weight,
                mean=0.0,
                std=float(head_init_std),
            )
            nn.init.normal_(
                self.mixture_head.weight,
                mean=0.0,
                std=float(head_init_std),
            )
        else:
            nn.init.zeros_(self.productivity_head.weight)
            nn.init.zeros_(self.mixture_head.weight)

        self.register_buffer("eta_max", torch.tensor(float(eta_max), dtype=torch.float32))
        self.register_buffer("M_c", torch.tensor(float(mag_completeness), dtype=torch.float32))
        self.register_buffer("M_m", torch.tensor(float(mag_max), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(richter_b), dtype=torch.float32))

        base_rate_init_t = torch.as_tensor(
            base_rate_init,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        tiny = torch.finfo(base_rate_init_t.dtype).tiny
        self.log_mu = nn.Parameter(base_rate_init_t.clamp_min(tiny).log())
        if self.fix_mu:
            self.log_mu.requires_grad = False
            mu_value = (
                torch.as_tensor(
                    fixed_mu_value,
                    device=self.device,
                    dtype=base_rate_init_t.dtype,
                )
                if fixed_mu_value is not None
                else torch.zeros(1, device=self.device, dtype=base_rate_init_t.dtype)
            )
            self.register_buffer("mu_fixed", mu_value)

        raw_weights = dict(loss_weights or {})
        self.bg_kl_weight = float(raw_weights.get("bg_kl_weight", 1.0))
        self.bg_norm_weight = float(raw_weights.get("bg_norm_weight", 0.0))
        self.branching_penalty_weight = float(
            raw_weights.get("branching_penalty_weight", branching_penalty_weight)
        )
        if self.bg_kl_weight < 0.0 or self.bg_norm_weight < 0.0:
            raise ValueError("bg_kl_weight and bg_norm_weight must be non-negative.")
        if self.branching_penalty_weight < 0.0:
            raise ValueError("branching_penalty_weight must be non-negative.")

        self.to(self.device)

    @property
    def mu(self) -> torch.Tensor:
        if self.fix_mu:
            return self.mu_fixed
        return torch.exp(self.log_mu)

    @property
    def num_basis(self) -> int:
        return self.basis.num_basis

    @staticmethod
    def _resolve_chunk_size(total: int, configured: int) -> int:
        if configured and configured > 0:
            return max(1, min(int(configured), int(total)))
        return max(1, int(total))

    @staticmethod
    def _iter_chunks(total: int, chunk_size: int):
        for start in range(0, int(total), int(chunk_size)):
            end = min(start + int(chunk_size), int(total))
            yield start, end

    def _sampling_magnitude_params(self) -> _MagnitudeSamplingParams:
        return _MagnitudeSamplingParams(
            b=float(self.b.detach().cpu().item()),
            m_min=float(self.M_c.detach().cpu().item()),
            m_max=float(self.M_m.detach().cpu().item()),
        )

    def _event_mask(self, batch: Batch) -> torch.Tensor:
        return get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )

    def _require_magnitude(self, batch: Batch) -> torch.Tensor:
        if not hasattr(batch, "mag") or batch.mag is None:
            raise ValueError("NETAS requires `batch.mag`.")
        return batch.mag.to(self.device)

    def _get_rnn_parent_context(self, batch: Batch, encoder: RNNTPPBackbone) -> torch.Tensor:
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling/context extraction currently supports no extra features.")
        features = encoder.build_features(batch)
        rnn_output, _ = encoder.rnn(features.contiguous())
        rnn_output = encoder._post_rnn_transform(rnn_output, features)
        return encoder.dropout(rnn_output * batch.input_mask[:, :, None])

    @staticmethod
    def _build_rnn_event_features(
        encoder: RNNTPPBackbone,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> torch.Tensor:
        feat_list = [encoder.encode_time(inter_time)]
        if encoder.input_magnitude:
            feat_list.append(encoder.encode_magnitude(magnitude))
        return torch.cat(feat_list, dim=-1).contiguous()

    def get_parent_context(self, batch: Batch) -> torch.Tensor:
        encoder = self.event_encoder
        if hasattr(encoder, "get_parent_context"):
            context = encoder.get_parent_context(batch)
        elif isinstance(encoder, RNNTPPBackbone):
            context = self._get_rnn_parent_context(batch, encoder)
        else:
            output = encoder(batch)
            context = output[0] if isinstance(output, tuple) else output

        if not torch.is_tensor(context) or context.ndim != 3:
            raise ValueError("Event encoder must return a tensor of shape [B, L, C].")
        if context.shape[0] != batch.batch_size or context.shape[1] != batch.seq_len:
            raise ValueError(
                "Parent context must be aligned with the padded event axis "
                f"(expected {(batch.batch_size, batch.seq_len)}, got {tuple(context.shape[:2])})."
            )
        if context.shape[-1] != self.context_size:
            raise ValueError(
                f"Parent context last dimension must equal context_size={self.context_size}."
            )
        return context.to(self.device)

    def _productivity_from_score(self, score: torch.Tensor) -> torch.Tensor:
        if self.productivity_mode == "bounded":
            return self.eta_max * torch.sigmoid(score)
        elif self.productivity_mode == "softplus":
            return F.softplus(score)
        elif self.productivity_mode == "exp":
            return torch.exp(score)
        raise RuntimeError(f"Unsupported productivity_mode={self.productivity_mode!r}.")

    def parent_params_from_context(
        self,
        parent_context: torch.Tensor,
        magnitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prod_score = self.productivity_head(parent_context).squeeze(-1)
        prod_score = prod_score + self.productivity_bias
        prod_score = prod_score + self.productivity_alpha * (magnitudes - self.M_c)
        eta = self._productivity_from_score(prod_score)
        omega = F.softmax(self.mixture_head(parent_context), dim=-1)
        return eta, omega

    def _make_parent_parameters(
        self,
        batch: Batch,
        *,
        parent_context: Optional[torch.Tensor] = None,
    ) -> _ParentParameters:
        magnitudes = self._require_magnitude(batch)
        event_mask = self._event_mask(batch).to(self.device)
        if parent_context is None:
            parent_context = self.get_parent_context(batch)
        eta, omega = self.parent_params_from_context(parent_context, magnitudes)
        eta = eta * event_mask
        omega = omega * event_mask.unsqueeze(-1)
        return _ParentParameters(
            context=parent_context,
            eta=eta,
            omega=omega,
            event_mask=event_mask,
        )

    @staticmethod
    def _coerce_parent_parameters(
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> _ParentParameters:
        if isinstance(parent_params, _ParentParameters):
            return parent_params
        return _ParentParameters.from_mapping(parent_params)

    def parent_parameters(
        self,
        batch: Batch,
        *,
        parent_context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self._make_parent_parameters(
            batch,
            parent_context=parent_context,
        ).as_dict()

    def _history_contrib(
        self,
        *,
        t_query_chunk: torch.Tensor,
        t_hist_chunk: torch.Tensor,
        eta_hist_chunk: torch.Tensor,
        omega_hist_chunk: torch.Tensor,
        event_mask_chunk: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = t_query_chunk.unsqueeze(-1) - t_hist_chunk.unsqueeze(-2)
        prev_mask = (delta_t > 0.0) & event_mask_chunk.unsqueeze(-2)
        basis_pdf = self.basis.pdf(delta_t)
        parent_weight = eta_hist_chunk.unsqueeze(-1) * omega_hist_chunk
        contrib = basis_pdf * parent_weight.unsqueeze(-3) * prev_mask.unsqueeze(-1)
        return contrib.sum(dim=(-2, -1))

    def _trigger_intensity_from_history(
        self,
        *,
        t_query: torch.Tensor,
        t_history: torch.Tensor,
        eta: torch.Tensor,
        omega: torch.Tensor,
        event_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        s_total = t_query.size(1)
        l_total = t_history.size(1)
        out = torch.zeros_like(t_query)
        query_chunk_size = self._resolve_chunk_size(s_total, self.query_chunk_size)
        history_chunk_size = self._resolve_chunk_size(l_total, self.history_chunk_size)

        for q_start, q_end in self._iter_chunks(s_total, query_chunk_size):
            t_query_chunk = t_query[:, q_start:q_end]
            intensity_chunk = torch.zeros_like(t_query_chunk)
            for h_start, h_end in self._iter_chunks(l_total, history_chunk_size):
                intensity_chunk = intensity_chunk + self._history_contrib(
                    t_query_chunk=t_query_chunk,
                    t_hist_chunk=t_history[:, h_start:h_end],
                    eta_hist_chunk=eta[:, h_start:h_end],
                    omega_hist_chunk=omega[:, h_start:h_end, :],
                    event_mask_chunk=event_mask_bool[:, h_start:h_end],
                )
            out[:, q_start:q_end] = intensity_chunk
        return out

    def trigger_intensity(
        self,
        batch: Batch,
        *,
        t_query: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)
        t_history = batch.arrival_times.to(self.device)
        if t_query is None:
            t_query = t_history
        return self._trigger_intensity_from_history(
            t_query=t_query.to(self.device),
            t_history=t_history,
            eta=parent_params.eta,
            omega=parent_params.omega,
            event_mask_bool=parent_params.event_mask_bool,
        )

    def intensity(
        self,
        batch: Batch,
        *,
        t_query: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        trigger = self.trigger_intensity(batch, t_query=t_query, parent_params=parent_params)
        total = trigger + self.mu.to(trigger.dtype)
        if self.bg_model is not None:
            total = total + self.bg_model.intensity(batch, t_query=t_query)
        return total

    def trigger_integral_between(
        self,
        batch: Batch,
        *,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if t_start.ndim == 1:
            t_start = t_start.unsqueeze(-1)
        if t_end.ndim == 1:
            t_end = t_end.unsqueeze(-1)
        if t_start.shape != t_end.shape:
            raise ValueError("t_start and t_end must have identical shapes.")
        t_start = t_start.to(self.device)
        t_end = t_end.to(self.device)

        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)

        t_history = batch.arrival_times.to(self.device)
        dt_end = (t_end.unsqueeze(-1) - t_history.unsqueeze(-2)).clamp_min(0.0)
        dt_start = (t_start.unsqueeze(-1) - t_history.unsqueeze(-2)).clamp_min(0.0)
        mass = self.basis.interval_mass(dt_start, dt_end)
        parent_weight = parent_params.weighted_mixture
        masked_weight = (
            parent_weight.unsqueeze(1)
            * parent_params.event_mask_bool.unsqueeze(1).unsqueeze(-1)
        )
        return (mass * masked_weight).sum(dim=(-2, -1))

    def compensator(
        self,
        batch: Batch,
        *,
        t_query: torch.Tensor,
        lower: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if t_query.ndim == 1:
            t_query = t_query.unsqueeze(-1)
        if lower is None:
            lower = batch.t_start
        if lower.ndim == 1:
            lower = lower.unsqueeze(-1).expand_as(t_query)
        trigger = self.trigger_integral_between(
            batch,
            t_start=lower,
            t_end=t_query,
            parent_params=parent_params,
        )
        total = trigger + self.mu.to(trigger.dtype) * (t_query - lower)
        if self.bg_model is not None:
            total = total + self.bg_model.intensity_integral_between(
                batch,
                t_start=lower,
                t_end=t_query,
            )
        return total

    def _nll_event_log_intensity(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = batch.arrival_times.to(self.device)
        t_select, intensity_mask = masked_select_per_row(
            t,
            batch.nll_event_mask.to(self.device),
        )
        intensity = self.intensity(
            batch,
            t_query=t_select,
            parent_params=parent_params,
        )
        log_intensity = torch.log(intensity.clamp_min(eps))
        return (log_intensity * intensity_mask).sum(dim=-1), t_select, intensity_mask

    def _nll_integral(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> torch.Tensor:
        t_start = batch.t_nll_start.to(self.device).unsqueeze(-1)
        t_end = batch.t_end.to(self.device).unsqueeze(-1)
        integral = self.trigger_integral_between(
            batch,
            t_start=t_start,
            t_end=t_end,
            parent_params=parent_params,
        ).squeeze(-1)
        interval_length = (batch.t_end - batch.t_nll_start).to(
            device=integral.device,
            dtype=integral.dtype,
        )
        integral = integral + interval_length * self.mu.to(integral.dtype)
        if self.bg_model is not None:
            integral = integral + self.bg_model.intensity_integral(batch)
        return integral

    def _background_regularization_terms(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
        t_query: torch.Tensor,
        event_mask: torch.Tensor,
        eps: float,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bg_kl = None
        if self.bg_model is not None and hasattr(self.bg_model, "kl_term"):
            bg_kl = self.bg_model.kl_term(batch, eps=eps)

        bg_norm = None
        if (
            self.bg_model is not None
            and self.bg_norm_weight > 0.0
            and hasattr(self.bg_model, "normalizing_term")
        ):
            trigger_intensity = self.trigger_intensity(
                batch,
                t_query=t_query,
                parent_params=parent_params,
            )
            bg_norm = self.bg_model.normalizing_term(
                batch,
                torch.log(trigger_intensity.clamp_min(eps)),
                eps=eps,
                t_query=t_query,
                event_mask=event_mask,
            )
        return bg_kl, bg_norm

    def _branching_penalty(
        self,
        batch: Batch,
        *,
        parent_params: _ParentParameters,
    ) -> Optional[torch.Tensor]:
        if self.branching_penalty_weight <= 0.0:
            return None

        event_mask = parent_params.event_mask.to(parent_params.eta.dtype)
        event_counts = event_mask.sum(dim=-1).clamp_min(1.0)
        mean_eta = (parent_params.eta * event_mask).sum(dim=-1) / event_counts
        target = torch.as_tensor(
            self.branching_penalty_target,
            device=mean_eta.device,
            dtype=mean_eta.dtype,
        )
        penalty = F.relu(mean_eta - target).pow(2)
        nll_event_counts = batch.nll_event_mask.to(
            device=penalty.device,
            dtype=penalty.dtype,
        ).sum(dim=-1).clamp_min(1.0)
        return penalty * nll_event_counts

    def nll_loss(
        self,
        batch: Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        reduction = self.reduction if reduction is None else reduction

        parent_params = self._make_parent_parameters(batch)
        log_intensity, t_select, intensity_mask = self._nll_event_log_intensity(
            batch,
            parent_params=parent_params,
            eps=eps,
        )
        integral = self._nll_integral(batch, parent_params=parent_params)
        nll_time = -log_intensity + integral
        bg_kl, bg_norm = self._background_regularization_terms(
            batch,
            parent_params=parent_params,
            t_query=t_select,
            event_mask=intensity_mask,
            eps=eps,
        )
        branching_penalty = self._branching_penalty(
            batch,
            parent_params=parent_params,
        )

        nll_total = nll_time
        if bg_kl is not None:
            nll_total = nll_total + self.bg_kl_weight * bg_kl
        if bg_norm is not None:
            nll_total = nll_total + self.bg_norm_weight * bg_norm
        if branching_penalty is not None:
            nll_total = nll_total + self.branching_penalty_weight * branching_penalty

        out_dict = {"time": nll_time, "total": nll_total}
        if bg_kl is not None:
            out_dict["bg_kl"] = bg_kl
        if bg_norm is not None:
            out_dict["bg_norm"] = bg_norm
        if branching_penalty is not None:
            out_dict["branching_penalty"] = branching_penalty

        out = self.reduce_nll_dict(out_dict, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]

    def _encoder_initial_state(self, *, batch_size: int, dtype: torch.dtype):
        encoder = self.event_encoder
        if hasattr(encoder, "initial_state"):
            return encoder.initial_state(
                batch_size=batch_size,
                device=self.device,
                dtype=dtype,
            )
        if isinstance(encoder, RNNTPPBackbone):
            if encoder.num_extra_features is not None:
                raise ValueError("NETAS sampling does not support extra encoder features.")
            return torch.zeros(
                encoder.num_rnn_layers,
                batch_size,
                encoder.context_size,
                device=self.device,
                dtype=dtype,
            )
        raise NotImplementedError(
            "Sampling requires either an RNNTPPBackbone encoder or a custom encoder "
            "implementing `initial_state(...)` and `step_event(...)`."
        )

    def _encoder_step_event(
        self,
        *,
        inter_time: float,
        magnitude: float,
        prev_state: Any,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Any]:
        encoder = self.event_encoder
        inter_time_t = torch.tensor([[float(inter_time)]], device=self.device, dtype=dtype)
        magnitude_t = torch.tensor([[float(magnitude)]], device=self.device, dtype=dtype)

        if hasattr(encoder, "step_event"):
            context, next_state = encoder.step_event(
                inter_time=inter_time_t,
                magnitude=magnitude_t,
                prev_state=prev_state,
            )
            return context, next_state

        if isinstance(encoder, RNNTPPBackbone):
            rnn_input = self._build_rnn_event_features(
                encoder,
                inter_time=inter_time_t,
                magnitude=magnitude_t,
            )
            return encoder.step(rnn_input, prev_state)

        raise NotImplementedError(
            "Sampling requires either an RNNTPPBackbone encoder or a custom encoder "
            "implementing `step_event(...)`."
        )

    def _encoder_step_event_batch_rnn(
        self,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.event_encoder
        if not isinstance(encoder, RNNTPPBackbone):
            raise TypeError("_encoder_step_event_batch_rnn requires RNNTPPBackbone.")
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        rnn_input = self._build_rnn_event_features(
            encoder,
            inter_time=inter_time,
            magnitude=magnitude,
        )
        return encoder.step(rnn_input, prev_state)

    def _sample_background_times_np(
        self,
        *,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
        preset_bg_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        duration = float(t_end - t_start)
        all_times: list[np.ndarray] = []

        mu_value = float(self.mu.detach().cpu().item())
        if mu_value > 0.0:
            n_back = int(rng.poisson(mu_value * duration))
            if n_back > 0:
                times = t_start + np.sort(rng.random(n_back) * duration)
                all_times.append(times.astype(np.float64, copy=False))

        if preset_bg_times is not None:
            bg_times = np.asarray(preset_bg_times, dtype=np.float64)
            if bg_times.size > 0:
                all_times.append(np.sort(bg_times))
        elif self.bg_model is not None:
            times_list = self.bg_model.sample_nhpp_inverse(
                B=1,
                t0=torch.tensor([t_start], device=self.device),
                dt=torch.tensor([duration], device=self.device),
                sample_sequence=True,
                mu=0.0,
            )
            if len(times_list) != 1:
                raise ValueError("Expected a single sampled background sequence.")
            bg_times = times_list[0]
            if torch.is_tensor(bg_times):
                bg_times = bg_times.detach().cpu().numpy()
            bg_times = np.asarray(bg_times, dtype=np.float64)
            if bg_times.size > 0:
                all_times.append(np.sort(bg_times))

        if not all_times:
            return np.empty((0,), dtype=np.float64)
        return np.sort(np.concatenate(all_times, axis=0))

    def _bg_cache_covers_window(
        self,
        cached: Any,
        *,
        t_start: float,
        t_end: float,
    ) -> bool:
        if cached is None or not hasattr(cached, "time_series_times"):
            return False
        ts_times = torch.as_tensor(cached.time_series_times).reshape(-1)
        if ts_times.numel() == 0:
            return False
        ts_min = float(ts_times[0].item())
        ts_max = float(ts_times[-1].item())
        return ts_min <= float(t_start) and ts_max >= float(t_end)

    def _ensure_bg_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        bg_cache_seq: Optional[Sequence],
    ) -> None:
        if self.bg_model is None:
            return

        cached = getattr(self.bg_model, "ts_batch_cache", None)
        if cached is None and not hasattr(self.bg_model, "cache_batch"):
            return
        if self._bg_cache_covers_window(
            cached,
            t_start=float(t_start),
            t_end=float(t_end),
        ):
            return

        for source_seq in (bg_cache_seq, past_seq):
            if source_seq is None:
                continue
            time_series = getattr(source_seq, "time_series", None)
            time_series_times = getattr(source_seq, "time_series_times", None)
            if time_series is None or time_series_times is None:
                continue
            ts_times = torch.as_tensor(time_series_times).reshape(-1)
            if ts_times.numel() == 0:
                continue
            ts_min = float(ts_times[0].item())
            ts_max = float(ts_times[-1].item())
            if ts_min <= float(t_start) and ts_max >= float(t_end):
                self.bg_model.cache_batch(
                    time_series=torch.as_tensor(time_series).unsqueeze(0),
                    time_series_times=ts_times.unsqueeze(0),
                )
                return

        raise ValueError(
            "NETAS background sampling requires a cached time series covering "
            f"[{float(t_start):.4f}, {float(t_end):.4f}]. "
            "Pass `bg_cache_seq=full_seq` to `model.sample(...)` or pre-call "
            "`model.bg_model.cache_batch(...)` with the full forcing series."
        )

    def _prepare_batched_bg_times(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
    ) -> list[np.ndarray]:
        if self.bg_model is None:
            return [np.empty((0,), dtype=np.float64) for _ in range(int(batch_size))]

        duration = float(t_end - t_start)
        times_list = self.bg_model.sample_nhpp_inverse(
            B=int(batch_size),
            t0=torch.full((int(batch_size),), float(t_start), device=self.device),
            dt=torch.full((int(batch_size),), duration, device=self.device),
            sample_sequence=True,
            mu=0.0,
        )
        bg_times_batch: list[np.ndarray] = []
        for bg_times in times_list:
            if torch.is_tensor(bg_times):
                bg_times = bg_times.detach().cpu().numpy()
            bg_times = np.asarray(bg_times, dtype=np.float64)
            bg_times_batch.append(np.sort(bg_times) if bg_times.size > 0 else np.empty((0,), dtype=np.float64))
        if len(bg_times_batch) != int(batch_size):
            raise ValueError(
                f"Expected {int(batch_size)} background sequences, got {len(bg_times_batch)}."
            )
        return bg_times_batch

    def _sample_offspring_for_parent(
        self,
        *,
        parent_time: float,
        parent_mag: float,
        parent_context: torch.Tensor,
        lower: float,
        upper: float,
        rng: np.random.Generator,
        mag_params: Optional[_MagnitudeSamplingParams] = None,
    ) -> list[tuple[float, float]]:
        if upper <= lower:
            return []

        magnitude_t = torch.tensor(
            [[float(parent_mag)]],
            device=parent_context.device,
            dtype=parent_context.dtype,
        )
        eta_t, omega_t = self.parent_params_from_context(parent_context, magnitude_t)
        eta = float(eta_t.squeeze().detach().cpu().item())
        omega = _to_numpy_array(omega_t.squeeze(0).squeeze(0)).astype(np.float64, copy=False)
        interval_mass = self.basis.interval_mass_np(lower, upper)
        offspring_means = eta * omega * interval_mass
        offspring_counts = rng.poisson(np.clip(offspring_means, a_min=0.0, a_max=None))

        children: list[tuple[float, float]] = []
        if mag_params is None:
            mag_params = self._sampling_magnitude_params()
        for basis_idx, count in enumerate(offspring_counts.tolist()):
            count = int(count)
            if count <= 0:
                continue
            lags = self.basis.sample_truncated_np(
                component_idx=basis_idx,
                lower=lower,
                upper=upper,
                size=count,
                rng=rng,
            )
            mags = _gen_mag(
                rng,
                count,
                b=mag_params.b,
                m_min=mag_params.m_min,
                m_max=mag_params.m_max,
            )
            for lag, magnitude in zip(lags.tolist(), mags.tolist()):
                children.append((parent_time + float(lag), float(magnitude)))
        return children

    def _sample_offspring_from_cached_means(
        self,
        *,
        parent_time: float,
        lower: float,
        upper: float,
        offspring_means: np.ndarray,
        rng: np.random.Generator,
        mag_params: Optional[_MagnitudeSamplingParams] = None,
    ) -> list[tuple[float, float]]:
        if upper <= lower:
            return []

        offspring_counts = rng.poisson(np.clip(offspring_means, a_min=0.0, a_max=None))
        children: list[tuple[float, float]] = []
        if mag_params is None:
            mag_params = self._sampling_magnitude_params()
        for basis_idx, count in enumerate(offspring_counts.tolist()):
            count = int(count)
            if count <= 0:
                continue
            lags = self.basis.sample_truncated_np(
                component_idx=basis_idx,
                lower=lower,
                upper=upper,
                size=count,
                rng=rng,
            )
            mags = _gen_mag(
                rng,
                count,
                b=mag_params.b,
                m_min=mag_params.m_min,
                m_max=mag_params.m_max,
            )
            for lag, magnitude in zip(lags.tolist(), mags.tolist()):
                children.append((parent_time + float(lag), float(magnitude)))
        return children

    def _clone_sampling_state(self, state: Any) -> Any:
        try:
            return copy.deepcopy(state)
        except Exception:
            if torch.is_tensor(state):
                return state.clone()
            if isinstance(state, tuple):
                return tuple(self._clone_sampling_state(item) for item in state)
            if isinstance(state, list):
                return [self._clone_sampling_state(item) for item in state]
            if isinstance(state, dict):
                return {
                    key: self._clone_sampling_state(value)
                    for key, value in state.items()
                }
            raise

    def _prepare_rnn_history_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        encoder: RNNTPPBackbone,
    ) -> dict[str, Any]:
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        dtype = self.M_c.dtype
        state = self._encoder_initial_state(batch_size=1, dtype=dtype)
        cached_parents: list[dict[str, Any]] = []

        if past_seq is None:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": float(t_start),
                "parents": cached_parents,
            }

        if not hasattr(past_seq, "mag") or past_seq.mag is None:
            raise ValueError("Sampling with past_seq requires `past_seq.mag`.")

        past_times = _to_numpy_array(past_seq.arrival_times).astype(np.float64, copy=False)
        past_mags = _to_numpy_array(past_seq.mag).astype(np.float64, copy=False)
        if past_times.size == 0:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": float(past_seq.t_end),
                "parents": cached_parents,
                "parent_times": np.empty((0,), dtype=np.float64),
                "parent_lower": np.empty((0,), dtype=np.float64),
                "parent_upper": np.empty((0,), dtype=np.float64),
                "parent_offspring_means": np.empty((0, self.num_basis), dtype=np.float64),
            }

        inter_times = np.diff(past_times, prepend=[float(past_seq.t_start)])
        inter_time_t = torch.as_tensor(
            inter_times,
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0)
        magnitude_t = torch.as_tensor(
            past_mags,
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0)

        features = self._build_rnn_event_features(
            encoder,
            inter_time=inter_time_t,
            magnitude=magnitude_t,
        )
        rnn_output, state = encoder.rnn(features, state.contiguous())
        context = encoder._post_rnn_transform(rnn_output, features)
        context = encoder.dropout(context)

        parent_lower = np.maximum(float(t_start) - past_times, 0.0).astype(
            np.float64,
            copy=False,
        )
        parent_upper = np.maximum(float(t_end) - past_times, 0.0).astype(
            np.float64,
            copy=False,
        )
        lower_t = torch.as_tensor(parent_lower, device=self.device, dtype=dtype)
        upper_t = torch.as_tensor(parent_upper, device=self.device, dtype=dtype)
        eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
        interval_mass = self.basis.interval_mass(lower_t, upper_t)
        valid = (upper_t > lower_t).to(dtype=interval_mass.dtype).unsqueeze(-1)
        parent_offspring_means_t = (
            eta_t.squeeze(0).unsqueeze(-1)
            * omega_t.squeeze(0)
            * interval_mass
            * valid
        )
        parent_offspring_means = _to_numpy_array(parent_offspring_means_t).astype(
            np.float64,
            copy=False,
        )

        for event_time, lower, upper, offspring_means in zip(
            past_times.tolist(),
            parent_lower.tolist(),
            parent_upper.tolist(),
            parent_offspring_means,
        ):
            cached_parents.append(
                {
                    "time": float(event_time),
                    "lower": float(lower),
                    "upper": float(upper),
                    "offspring_means": offspring_means,
                }
            )

        return {
            "state": self._clone_sampling_state(state),
            "last_time": float(past_times[-1]),
            "parents": cached_parents,
            "parent_times": past_times,
            "parent_lower": parent_lower,
            "parent_upper": parent_upper,
            "parent_offspring_means": parent_offspring_means,
        }

    def _prepare_history_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
    ) -> dict[str, Any]:
        encoder = self.event_encoder
        if isinstance(encoder, RNNTPPBackbone):
            return self._prepare_rnn_history_sampling_cache(
                t_start=t_start,
                t_end=t_end,
                past_seq=past_seq,
                encoder=encoder,
            )

        dtype = self.M_c.dtype
        state = self._encoder_initial_state(batch_size=1, dtype=dtype)
        last_time = float(t_start)
        cached_parents: list[dict[str, Any]] = []

        if past_seq is None:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": last_time,
                "parents": cached_parents,
            }

        if not hasattr(past_seq, "mag") or past_seq.mag is None:
            raise ValueError("Sampling with past_seq requires `past_seq.mag`.")

        last_time = float(past_seq.t_start)
        past_times = _to_numpy_array(past_seq.arrival_times).astype(np.float64, copy=False)
        past_mags = _to_numpy_array(past_seq.mag).astype(np.float64, copy=False)
        has_past_events = past_times.size > 0
        for event_time, event_mag in zip(past_times.tolist(), past_mags.tolist()):
            context, state = self._encoder_step_event(
                inter_time=float(event_time - last_time),
                magnitude=float(event_mag),
                prev_state=state,
                dtype=dtype,
            )
            last_time = float(event_time)
            lower = max(t_start - float(event_time), 0.0)
            upper = max(t_end - float(event_time), 0.0)
            if upper > lower:
                magnitude_t = torch.tensor(
                    [[float(event_mag)]],
                    device=context.device,
                    dtype=context.dtype,
                )
                eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
                eta = float(eta_t.squeeze().detach().cpu().item())
                omega = _to_numpy_array(omega_t.squeeze(0).squeeze(0)).astype(
                    np.float64,
                    copy=False,
                )
                interval_mass = self.basis.interval_mass_np(lower, upper)
                offspring_means = eta * omega * interval_mass
            else:
                offspring_means = np.zeros((self.num_basis,), dtype=np.float64)
            cached_parents.append(
                {
                    "time": float(event_time),
                    "lower": float(lower),
                    "upper": float(upper),
                    "offspring_means": offspring_means,
                }
            )
        if not has_past_events:
            last_time = float(past_seq.t_end)

        if cached_parents:
            parent_times = np.asarray(
                [payload["time"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_lower = np.asarray(
                [payload["lower"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_upper = np.asarray(
                [payload["upper"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_offspring_means = np.stack(
                [payload["offspring_means"] for payload in cached_parents],
                axis=0,
            ).astype(np.float64, copy=False)
        else:
            parent_times = np.empty((0,), dtype=np.float64)
            parent_lower = np.empty((0,), dtype=np.float64)
            parent_upper = np.empty((0,), dtype=np.float64)
            parent_offspring_means = np.empty(
                (0, self.num_basis),
                dtype=np.float64,
            )

        return {
            "state": self._clone_sampling_state(state),
            "last_time": last_time,
            "parents": cached_parents,
            "parent_times": parent_times,
            "parent_lower": parent_lower,
            "parent_upper": parent_upper,
            "parent_offspring_means": parent_offspring_means,
        }

    def _resolve_sampling_parent_chunk_size(
        self,
        *,
        batch_size: int,
        num_parents: int,
    ) -> int:
        if int(num_parents) <= 0:
            return 0
        target_count_elements = 1_000_000
        denom = max(1, int(batch_size) * int(self.num_basis))
        chunk_size = max(32, target_count_elements // denom)
        return min(int(num_parents), max(1, int(chunk_size)))

    def _build_batched_initial_events(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
        history_cache: dict[str, Any],
        init_rng: np.random.Generator,
        batched_bg_times: Optional[list[np.ndarray]] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        initial_time_chunks: list[list[np.ndarray]] = [[] for _ in range(int(batch_size))]
        initial_mag_chunks: list[list[np.ndarray]] = [[] for _ in range(int(batch_size))]

        mag_params = self._sampling_magnitude_params()

        parent_times = np.asarray(history_cache.get("parent_times", []), dtype=np.float64)
        parent_lower = np.asarray(history_cache.get("parent_lower", []), dtype=np.float64)
        parent_upper = np.asarray(history_cache.get("parent_upper", []), dtype=np.float64)
        parent_offspring_means = np.asarray(
            history_cache.get("parent_offspring_means", []),
            dtype=np.float64,
        )

        num_parents = int(parent_times.shape[0])
        if num_parents > 0:
            parent_chunk_size = self._resolve_sampling_parent_chunk_size(
                batch_size=int(batch_size),
                num_parents=num_parents,
            )
            for parent_start in range(0, num_parents, parent_chunk_size):
                parent_end = min(num_parents, parent_start + parent_chunk_size)
                chunk_times = parent_times[parent_start:parent_end]
                chunk_lower = parent_lower[parent_start:parent_end]
                chunk_upper = parent_upper[parent_start:parent_end]
                chunk_means = parent_offspring_means[parent_start:parent_end]
                if (
                    chunk_means.size == 0
                    or not np.any(chunk_upper > chunk_lower)
                    or not np.any(chunk_means > 0.0)
                ):
                    continue

                counts_cube = init_rng.poisson(
                    chunk_means[None, :, :],
                    size=(int(batch_size), chunk_means.shape[0], chunk_means.shape[1]),
                )
                if counts_cube.size == 0 or int(counts_cube.sum()) == 0:
                    continue

                for basis_idx in range(chunk_means.shape[1]):
                    counts_bp = counts_cube[:, :, basis_idx]
                    flat_counts = counts_bp.reshape(-1)
                    nonzero_mask = flat_counts > 0
                    if not np.any(nonzero_mask):
                        continue

                    contributing_pairs = np.nonzero(nonzero_mask)[0]
                    pair_counts = flat_counts[nonzero_mask].astype(np.int64, copy=False)
                    total_count = int(pair_counts.sum())
                    if total_count <= 0:
                        continue

                    chunk_parent_count = chunk_means.shape[0]
                    seq_ids = contributing_pairs // chunk_parent_count
                    parent_ids = contributing_pairs % chunk_parent_count
                    expanded_seq_ids = np.repeat(seq_ids, pair_counts)
                    expanded_parent_ids = np.repeat(parent_ids, pair_counts)

                    lags = self.basis.sample_truncated_vectorized_np(
                        component_idx=basis_idx,
                        lower=chunk_lower[expanded_parent_ids],
                        upper=chunk_upper[expanded_parent_ids],
                        rng=init_rng,
                    )
                    event_times = chunk_times[expanded_parent_ids] + lags
                    event_mags = _gen_mag(
                        init_rng,
                        total_count,
                        b=mag_params.b,
                        m_min=mag_params.m_min,
                        m_max=mag_params.m_max,
                    ).astype(np.float32, copy=False)

                    order = np.argsort(expanded_seq_ids, kind="stable")
                    expanded_seq_ids = expanded_seq_ids[order]
                    event_times = event_times[order]
                    event_mags = event_mags[order]

                    unique_seq_ids, group_starts = np.unique(
                        expanded_seq_ids,
                        return_index=True,
                    )
                    group_ends = np.concatenate(
                        [group_starts[1:], np.asarray([expanded_seq_ids.shape[0]])]
                    )
                    for seq_idx, start_idx, end_idx in zip(
                        unique_seq_ids.tolist(),
                        group_starts.tolist(),
                        group_ends.tolist(),
                    ):
                        initial_time_chunks[int(seq_idx)].append(event_times[start_idx:end_idx])
                        initial_mag_chunks[int(seq_idx)].append(event_mags[start_idx:end_idx])

        mu_value = float(self.mu.detach().cpu().item())
        duration = float(t_end - t_start)
        if mu_value > 0.0:
            counts_bg = init_rng.poisson(mu_value * duration, size=int(batch_size))
            total_count = int(counts_bg.sum())
            if total_count > 0:
                bg_times = t_start + np.sort(init_rng.random(total_count) * duration)
                bg_mags = _gen_mag(
                    init_rng,
                    total_count,
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                offset = 0
                for seq_idx, seq_count in enumerate(counts_bg.tolist()):
                    seq_count = int(seq_count)
                    if seq_count <= 0:
                        continue
                    seq_times = bg_times[offset : offset + seq_count]
                    seq_mags = bg_mags[offset : offset + seq_count]
                    initial_time_chunks[seq_idx].append(seq_times)
                    initial_mag_chunks[seq_idx].append(
                        seq_mags.astype(np.float32, copy=False)
                    )
                    offset += seq_count

        if batched_bg_times is not None:
            for seq_idx, bg_times in enumerate(batched_bg_times):
                if bg_times.size == 0:
                    continue
                bg_mags = _gen_mag(
                    init_rng,
                    int(bg_times.size),
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                initial_time_chunks[seq_idx].append(np.asarray(bg_times, dtype=np.float64))
                initial_mag_chunks[seq_idx].append(bg_mags.astype(np.float32, copy=False))

        initial_events: list[tuple[np.ndarray, np.ndarray]] = []
        for time_parts, mag_parts in zip(initial_time_chunks, initial_mag_chunks):
            if not time_parts:
                initial_events.append(_empty_initial_event_arrays())
                continue
            seq_times = np.concatenate(time_parts, axis=0).astype(np.float64, copy=False)
            seq_mags = np.concatenate(mag_parts, axis=0).astype(np.float32, copy=False)
            order = np.argsort(seq_times, kind="mergesort")
            initial_events.append((seq_times[order], seq_mags[order]))
        return initial_events

    def _sample_single_sequence(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        rng: np.random.Generator,
        max_length: int,
        history_cache: Optional[dict[str, Any]] = None,
        preset_bg_times: Optional[np.ndarray] = None,
        preset_initial_events: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Sequence:
        dtype = self.M_c.dtype
        if history_cache is None:
            history_cache = self._prepare_history_sampling_cache(
                t_start=t_start,
                t_end=t_end,
                past_seq=past_seq,
            )
        mag_params = self._sampling_magnitude_params()
        state = self._clone_sampling_state(history_cache["state"])
        last_time = float(history_cache["last_time"])
        event_heap: list[tuple[float, int, float]] = []
        event_counter = itertools.count()
        future_events = _SequenceEventAccumulator()
        pending_initial_times: Optional[np.ndarray] = None
        pending_initial_magnitudes: Optional[np.ndarray] = None
        pending_initial_idx = 0

        if preset_initial_events is not None:
            pending_initial_times = np.asarray(
                preset_initial_events[0],
                dtype=np.float64,
            )
            pending_initial_magnitudes = np.asarray(
                preset_initial_events[1],
                dtype=np.float32,
            )
            if pending_initial_times.shape != pending_initial_magnitudes.shape:
                raise ValueError(
                    "preset_initial_events time and magnitude arrays must have identical shapes."
                )
        else:
            for parent_payload in history_cache["parents"]:
                children = self._sample_offspring_from_cached_means(
                    parent_time=float(parent_payload["time"]),
                    lower=float(parent_payload["lower"]),
                    upper=float(parent_payload["upper"]),
                    offspring_means=parent_payload["offspring_means"],
                    rng=rng,
                    mag_params=mag_params,
                )
                for child_time, child_mag in children:
                    heapq.heappush(
                        event_heap,
                        (float(child_time), next(event_counter), float(child_mag)),
                    )

            background_times = self._sample_background_times_np(
                t_start=t_start,
                t_end=t_end,
                rng=rng,
                preset_bg_times=preset_bg_times,
            )
            if background_times.size > 0:
                background_mags = _gen_mag(
                    rng,
                    int(background_times.size),
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                for event_time, event_mag in zip(background_times.tolist(), background_mags.tolist()):
                    heapq.heappush(
                        event_heap,
                        (float(event_time), next(event_counter), float(event_mag)),
                    )

        if past_seq is None:
            last_time = float(t_start)

        while (
            event_heap
            or (
                pending_initial_times is not None
                and pending_initial_idx < int(pending_initial_times.shape[0])
            )
        ):
            next_pending_time = math.inf
            if (
                pending_initial_times is not None
                and pending_initial_idx < int(pending_initial_times.shape[0])
            ):
                next_pending_time = float(pending_initial_times[pending_initial_idx])

            next_heap_time = event_heap[0][0] if event_heap else math.inf
            if next_pending_time <= next_heap_time:
                event_time = next_pending_time
                assert pending_initial_magnitudes is not None
                event_mag = float(pending_initial_magnitudes[pending_initial_idx])
                pending_initial_idx += 1
            else:
                event_time, _, event_mag = heapq.heappop(event_heap)
            if event_time > t_end:
                continue

            context, state = self._encoder_step_event(
                inter_time=float(event_time - last_time),
                magnitude=float(event_mag),
                prev_state=state,
                dtype=dtype,
            )
            last_time = float(event_time)
            future_events.append(event_time, event_mag)
            if len(future_events.times) > int(max_length):
                raise RuntimeError(
                    f"NETAS sampling exceeded max_length={max_length}; "
                    "consider lowering eta_max or shortening the horizon."
                )

            children = self._sample_offspring_for_parent(
                parent_time=float(event_time),
                parent_mag=float(event_mag),
                parent_context=context,
                lower=0.0,
                upper=max(t_end - float(event_time), 0.0),
                rng=rng,
                mag_params=mag_params,
            )
            for child_time, child_mag in children:
                heapq.heappush(
                    event_heap,
                    (float(child_time), next(event_counter), float(child_mag)),
                )

        return future_events.to_sequence(t_start=t_start, t_end=t_end)

    def _can_use_batched_rnn_sampling(self, *, batch_size: int) -> bool:
        encoder = self.event_encoder
        return (
            int(batch_size) > 1
            and isinstance(encoder, RNNTPPBackbone)
            and encoder.num_extra_features is None
        )

    def _sample_batch_sequences_rnn(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
        child_seeds: list[np.random.SeedSequence],
        max_length: int,
        history_cache: dict[str, Any],
        preset_initial_events_batch: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[Sequence]:
        encoder = self.event_encoder
        if not isinstance(encoder, RNNTPPBackbone):
            raise TypeError("_sample_batch_sequences_rnn requires RNNTPPBackbone.")
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        dtype = self.M_c.dtype
        mag_params = self._sampling_magnitude_params()
        base_state = history_cache["state"]
        if not torch.is_tensor(base_state) or base_state.ndim != 3 or base_state.shape[1] != 1:
            raise ValueError("RNN sampling state must have shape [num_layers, 1, context_size].")
        state = base_state.repeat(1, int(batch_size), 1).contiguous()

        rngs = [np.random.default_rng(seed) for seed in child_seeds]
        event_heaps: list[list[tuple[float, int, float]]] = [[] for _ in range(int(batch_size))]
        event_counters = [itertools.count() for _ in range(int(batch_size))]
        future_events = [
            _SequenceEventAccumulator()
            for _ in range(int(batch_size))
        ]
        last_times = np.full(
            int(batch_size),
            float(history_cache["last_time"]),
            dtype=np.float64,
        )

        pending_initial_times: list[np.ndarray] = []
        pending_initial_magnitudes: list[np.ndarray] = []
        pending_initial_idx = np.zeros(int(batch_size), dtype=np.int64)
        for event_times, event_mags in preset_initial_events_batch:
            event_times = np.asarray(event_times, dtype=np.float64)
            event_mags = np.asarray(event_mags, dtype=np.float32)
            if event_times.shape != event_mags.shape:
                raise ValueError(
                    "preset_initial_events time and magnitude arrays must have identical shapes."
                )
            pending_initial_times.append(event_times)
            pending_initial_magnitudes.append(event_mags)

        while True:
            active_indices: list[int] = []
            active_times: list[float] = []
            active_magnitudes: list[float] = []

            for seq_idx in range(int(batch_size)):
                next_pending_time = math.inf
                if pending_initial_idx[seq_idx] < int(pending_initial_times[seq_idx].shape[0]):
                    next_pending_time = float(
                        pending_initial_times[seq_idx][pending_initial_idx[seq_idx]]
                    )

                next_heap_time = (
                    event_heaps[seq_idx][0][0] if event_heaps[seq_idx] else math.inf
                )
                if math.isinf(next_pending_time) and math.isinf(next_heap_time):
                    continue

                if next_pending_time <= next_heap_time:
                    event_time = next_pending_time
                    event_mag = float(
                        pending_initial_magnitudes[seq_idx][pending_initial_idx[seq_idx]]
                    )
                    pending_initial_idx[seq_idx] += 1
                else:
                    event_time, _, event_mag = heapq.heappop(event_heaps[seq_idx])

                if event_time > t_end:
                    continue

                active_indices.append(seq_idx)
                active_times.append(float(event_time))
                active_magnitudes.append(float(event_mag))

            if not active_indices:
                break

            active_indices_t = torch.as_tensor(
                active_indices,
                device=self.device,
                dtype=torch.long,
            )
            active_times_np = np.asarray(active_times, dtype=np.float64)
            active_magnitudes_np = np.asarray(active_magnitudes, dtype=np.float64)
            active_indices_np = np.asarray(active_indices, dtype=np.int64)
            inter_times_np = active_times_np - last_times[active_indices_np]
            inter_time_t = torch.as_tensor(
                inter_times_np,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(-1)
            magnitude_t = torch.as_tensor(
                active_magnitudes_np,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(-1)

            prev_state = state.index_select(1, active_indices_t).contiguous()
            context, next_state = self._encoder_step_event_batch_rnn(
                inter_time=inter_time_t,
                magnitude=magnitude_t,
                prev_state=prev_state,
            )
            state.index_copy_(1, active_indices_t, next_state)

            for seq_idx, event_time, event_mag in zip(
                active_indices,
                active_times,
                active_magnitudes,
            ):
                last_times[seq_idx] = float(event_time)
                future_events[seq_idx].append(event_time, event_mag)
                if len(future_events[seq_idx].times) > int(max_length):
                    raise RuntimeError(
                        f"NETAS sampling exceeded max_length={max_length}; "
                        "consider lowering eta_max or shortening the horizon."
                    )

            event_time_t = torch.as_tensor(
                active_times_np,
                device=self.device,
                dtype=dtype,
            )
            upper_t = (
                torch.as_tensor(float(t_end), device=self.device, dtype=dtype)
                - event_time_t
            ).clamp_min(0.0)
            interval_mass = self.basis.interval_mass(torch.zeros_like(upper_t), upper_t)
            eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
            offspring_means = (
                eta_t.squeeze(1).unsqueeze(-1)
                * omega_t.squeeze(1)
                * interval_mass
            )
            offspring_means_np = _to_numpy_array(offspring_means).astype(
                np.float64,
                copy=False,
            )

            for local_idx, seq_idx in enumerate(active_indices):
                upper = max(float(t_end) - float(active_times[local_idx]), 0.0)
                children = self._sample_offspring_from_cached_means(
                    parent_time=float(active_times[local_idx]),
                    lower=0.0,
                    upper=upper,
                    offspring_means=offspring_means_np[local_idx],
                    rng=rngs[seq_idx],
                    mag_params=mag_params,
                )
                for child_time, child_mag in children:
                    heapq.heappush(
                        event_heaps[seq_idx],
                        (
                            float(child_time),
                            next(event_counters[seq_idx]),
                            float(child_mag),
                        ),
                    )

        return [
            events.to_sequence(t_start=t_start, t_end=t_end)
            for events in future_events
        ]

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        bg_cache_seq: Optional[src.data.Sequence] = None,
        random_state: int = 123,
        max_length: int = 50_000,
        return_sequences: bool = False,
    ) -> Union[Batch, list[Sequence]]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1.")
        if float(duration) <= 0.0:
            raise ValueError("duration must be positive.")
        if past_seq is not None:
            t_start = float(past_seq.t_end)

        t_end = float(t_start + duration)
        self._ensure_bg_sampling_cache(
            t_start=float(t_start),
            t_end=t_end,
            past_seq=past_seq,
            bg_cache_seq=bg_cache_seq,
        )
        seed_sequence = np.random.SeedSequence(int(random_state))
        child_seeds = seed_sequence.spawn(int(batch_size))
        history_cache = self._prepare_history_sampling_cache(
            t_start=float(t_start),
            t_end=t_end,
            past_seq=past_seq,
        )
        batched_bg_times = self._prepare_batched_bg_times(
            batch_size=int(batch_size),
            t_start=float(t_start),
            t_end=t_end,
        )
        init_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        preset_initial_events_batch = self._build_batched_initial_events(
            batch_size=int(batch_size),
            t_start=float(t_start),
            t_end=t_end,
            history_cache=history_cache,
            init_rng=init_rng,
            batched_bg_times=batched_bg_times,
        )

        if past_seq is not None:
            estimated_history_work = int(batch_size) * int(past_seq.num_events)
            if estimated_history_work >= 20_000:
                logger.warning(
                    "NETAS sampling may be slow: batch_size=%s, past_events=%s, "
                    "estimated parent-history work=%s. Consider reducing "
                    "`samples_per_batch` or forecast horizon.",
                    int(batch_size),
                    int(past_seq.num_events),
                    estimated_history_work,
                )

        progress_stride = 0
        if int(batch_size) >= 200:
            progress_stride = max(1, int(batch_size) // 10)

        if self._can_use_batched_rnn_sampling(batch_size=int(batch_size)):
            sequences = self._sample_batch_sequences_rnn(
                batch_size=int(batch_size),
                t_start=float(t_start),
                t_end=t_end,
                child_seeds=child_seeds,
                max_length=int(max_length),
                history_cache=history_cache,
                preset_initial_events_batch=preset_initial_events_batch,
            )
        else:
            sequences = []
            for sample_idx, child_seed in enumerate(child_seeds, start=1):
                rng = np.random.default_rng(child_seed)
                sequences.append(
                    self._sample_single_sequence(
                        t_start=float(t_start),
                        t_end=t_end,
                        past_seq=past_seq,
                        rng=rng,
                        max_length=int(max_length),
                        history_cache=history_cache,
                        preset_bg_times=batched_bg_times[sample_idx - 1],
                        preset_initial_events=preset_initial_events_batch[sample_idx - 1],
                    )
                )
                if progress_stride and (
                    sample_idx % progress_stride == 0 or sample_idx == int(batch_size)
                ):
                    logger.info(
                        "NETAS sampling progress: %s/%s sequences completed.",
                        sample_idx,
                        int(batch_size),
                    )

        if return_sequences:
            return sequences
        return Batch.from_list(sequences).to(self.device)

    def _evaluation_grid(
        self,
        sequence: Sequence,
        *,
        num_grid_points: int,
    ) -> tuple[Batch, torch.Tensor]:
        if int(num_grid_points) < 1:
            raise ValueError("num_grid_points must be >= 1.")
        batch = Batch.from_list([sequence]).to(self.device)
        seq_len = int(batch.end_idx[0].item()) + 1
        inter_times = batch.inter_times[0, :seq_len]
        interval_starts = torch.cat(
            [
                batch.t_start[:1].to(inter_times.dtype),
                batch.arrival_times[0, : seq_len - 1].to(inter_times.dtype),
            ],
            dim=0,
        )
        fractions = torch.linspace(
            1e-4,
            1.0,
            int(num_grid_points),
            device=self.device,
            dtype=inter_times.dtype,
        )
        grid = interval_starts[:, None] + inter_times[:, None] * fractions[None, :]
        return batch, grid.reshape(1, -1)

    def evaluate_intensity(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, grid = self._evaluation_grid(sequence, num_grid_points=num_grid_points)
        parent_params = self.parent_parameters(batch)
        intensity = self.intensity(batch, t_query=grid, parent_params=parent_params)
        return grid.squeeze(0), intensity.squeeze(0)

    def evaluate_compensator(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, grid = self._evaluation_grid(sequence, num_grid_points=num_grid_points)
        parent_params = self.parent_parameters(batch)
        lower = batch.t_start.unsqueeze(-1).expand_as(grid)
        compensator = self.compensator(
            batch,
            t_query=grid,
            lower=lower,
            parent_params=parent_params,
        )
        return grid.squeeze(0), compensator.squeeze(0)


__all__ = [
    "FixedKernelBasis",
    "MambaNETASEncoder",
    "NETAS",
    "masked_select_per_row",
]
