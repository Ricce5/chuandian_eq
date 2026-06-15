"""Mamba-based event encoder used by NETAS."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn

from src.data.batch import Batch

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

try:
    from src.models.mamba.mamba_time import MambaTime
except ImportError:
    MambaTime = None


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


__all__ = ["MambaNETASEncoder"]
