"""Reusable recurrent building blocks for TPP models.

This module provides composable recurrent backbone pieces and small helper
heads/fusion modules shared by recurrent-style TPP variants.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import src


class FiLMContextFuse(nn.Module):
    """Feature-wise linear modulation block for context + bg-context fusion."""

    def __init__(
        self,
        *,
        context_dim: int,
        bg_context_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(context_dim)
        if hidden < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden}.")
        self.bg_norm = nn.LayerNorm(int(bg_context_dim))
        self.mod_in = nn.Linear(int(bg_context_dim), hidden)
        self.mod_out = nn.Linear(hidden, 2 * int(context_dim))
        nn.init.zeros_(self.mod_out.weight)
        nn.init.zeros_(self.mod_out.bias)

    def forward(self, context: torch.Tensor, bg_context: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.mod_in(self.bg_norm(bg_context)))
        gamma_beta = self.mod_out(h)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = torch.tanh(gamma)
        return context * (1.0 + gamma) + beta


class TimeHypernetMLP(nn.Module):
    """Small MLP head for mapping context -> inter-time distribution parameters."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(input_dim)
        if hidden < 1:
            raise ValueError(f"hidden_dim must be >= 1 (got {hidden})")

        act_name = str(activation).strip().lower()
        if act_name == "relu":
            act = nn.ReLU()
        elif act_name == "gelu":
            act = nn.GELU()
        elif act_name == "silu":
            act = nn.SiLU()
        else:
            raise ValueError("activation must be one of ['relu', 'gelu', 'silu']")

        self.fc1 = nn.Linear(int(input_dim), hidden)
        self.act = act
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(hidden, int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


def build_time_hypernet(
    *,
    input_dim: int,
    output_dim: int,
    use_mlp: bool = True,
    hidden_dim: Optional[int] = None,
    activation: str = "silu",
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for context -> inter-time-parameter heads."""
    if use_mlp:
        return TimeHypernetMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
    return nn.Linear(int(input_dim), int(output_dim))


class RNNTPPBackbone(nn.Module):
    """Composable RNN backbone for temporal point process models."""

    def __init__(
        self,
        *,
        context_size: int,
        tau_mean: float,
        mag_mean: float,
        rnn_type: str = "GRU",
        num_rnn_layers: int = 1,
        dropout: float = 0.0,
        input_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        use_residual: bool = False,
        use_layernorm: bool = False,
        time_preprocess: str = "legacy",
        log_tau_mean: Optional[float] = None,
        log_tau_std: Optional[float] = None,
        inter_time_min: float = 1e-10,
        inter_time_max: float = 1e10,
    ) -> None:
        super().__init__()
        if rnn_type not in {"RNN", "GRU"}:
            raise ValueError(f"rnn_type must be one of ['RNN', 'GRU'] (got {rnn_type})")
        if num_rnn_layers < 1:
            raise ValueError(f"num_rnn_layers must be >= 1 (got {num_rnn_layers})")

        self.context_size = int(context_size)
        self.input_magnitude = bool(input_magnitude)
        self.num_extra_features = num_extra_features
        self.num_rnn_layers = int(num_rnn_layers)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        self.time_preprocess = str(time_preprocess).strip().lower()
        if self.time_preprocess not in {"legacy", "oracle"}:
            raise ValueError("time_preprocess must be one of ['legacy', 'oracle']")
        self.inter_time_min = float(inter_time_min)
        self.inter_time_max = float(inter_time_max)
        if self.inter_time_min <= 0:
            raise ValueError(f"inter_time_min must be > 0, got {self.inter_time_min}")
        if self.inter_time_max <= self.inter_time_min:
            raise ValueError(
                f"inter_time_max must be > inter_time_min, got {self.inter_time_max} <= {self.inter_time_min}"
            )

        tau_mean_value = max(float(tau_mean), self.inter_time_min)
        if log_tau_mean is None:
            if self.time_preprocess == "oracle":
                log_tau_mean = torch.log10(torch.tensor(tau_mean_value)).item()
            else:
                log_tau_mean = torch.log(torch.tensor(tau_mean_value)).item()
        if log_tau_std is None:
            log_tau_std = 2.0 if self.time_preprocess == "oracle" else 1.0
        if float(log_tau_std) <= 0:
            raise ValueError(f"log_tau_std must be > 0, got {log_tau_std}")

        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.float32))
        self.register_buffer("log_tau_mean", torch.tensor(float(log_tau_mean), dtype=torch.float32))
        self.register_buffer("log_tau_std", torch.tensor(float(log_tau_std), dtype=torch.float32))
        self.register_buffer("mag_mean", torch.tensor(mag_mean, dtype=torch.float32))

        self.num_rnn_inputs = (
            1
            + int(self.input_magnitude)
            + (0 if self.num_extra_features is None else int(self.num_extra_features))
        )
        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs,
            self.context_size,
            num_layers=self.num_rnn_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        if self.use_residual:
            if self.num_rnn_inputs == self.context_size:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Linear(self.num_rnn_inputs, self.context_size)
        else:
            self.residual_proj = None
        self.layer_norm = nn.LayerNorm(self.context_size) if self.use_layernorm else None

    def _post_rnn_transform(
        self,
        rnn_output: torch.Tensor,
        residual_input: torch.Tensor,
    ) -> torch.Tensor:
        out = rnn_output
        if self.use_residual:
            out = out + self.residual_proj(residual_input)
        if self.use_layernorm:
            out = self.layer_norm(out)
        return out

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

    def build_features(self, batch: src.data.Batch) -> torch.Tensor:
        feat_list = [self.encode_time(batch.inter_times)]
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))

        if self.num_extra_features is not None:
            if not hasattr(batch, "extra_features"):
                raise ValueError(
                    "num_extra_features is set but batch.extra_features is missing."
                )
            feat_list.append(self.encode_extra_features(batch.extra_features))

        features = torch.cat(feat_list, dim=-1).contiguous()
        return features * batch.input_mask[:, :, None]

    def get_context(self, batch: src.data.Batch) -> torch.Tensor:
        context, _ = self.get_context_and_hidden(batch)
        return context

    def get_context_and_hidden(
        self,
        batch: src.data.Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.build_features(batch)
        rnn_output, hidden = self.rnn(features.contiguous())
        rnn_output = self._post_rnn_transform(rnn_output, features)
        rnn_output = rnn_output * batch.input_mask[:, :, None]
        rnn_output = rnn_output[:, :-1, :]
        context = F.pad(rnn_output, (0, 0, 1, 0))
        return self.dropout(context), hidden

    def forward(self, batch: src.data.Batch):
        features = self.build_features(batch)
        rnn_output, hidden = self.rnn(features.contiguous())
        rnn_output = self._post_rnn_transform(rnn_output, features)
        return rnn_output, hidden

    def step(
        self,
        rnn_input: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.num_rnn_layers, rnn_input.shape[0], self.context_size)
        if tuple(prev_hidden.shape) != expected:
            raise ValueError(
                f"prev_hidden must have shape {expected}, got {tuple(prev_hidden.shape)}."
            )
        rnn_output, next_hidden = self.rnn(rnn_input, prev_hidden.contiguous())
        next_state = rnn_output[:, -1:, :].contiguous()
        next_state = self._post_rnn_transform(next_state, rnn_input)
        return self.dropout(next_state), next_hidden


__all__ = [
    "FiLMContextFuse",
    "TimeHypernetMLP",
    "build_time_hypernet",
    "RNNTPPBackbone",
]
