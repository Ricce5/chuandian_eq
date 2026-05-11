"""Oracle-specific reusable network blocks.

This module groups encoder/decoder components used by Oracle-style TPP models
while keeping them reusable and separately testable from the high-level model.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import src.distributions as dist


class OracleRNNEncoder(nn.Module):
    """Residual RNN encoder used by Oracle."""

    def __init__(
        self,
        *,
        rnn_type: str,
        d_model_in: int,
        num_layers: int = 1,
        dropout_prob: float = 0.2,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        rnn_type = str(rnn_type).strip().upper()
        rnn_cls_by_name = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        if rnn_type not in rnn_cls_by_name:
            raise ValueError(f"rnn_type must be one of {{'RNN','GRU','LSTM'}} (got {rnn_type!r}).")
        recurrent_dropout = dropout_prob if num_layers > 1 else 0.0
        self.rnn = rnn_cls_by_name[rnn_type](
            d_model_in,
            d_model_in,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(d_model_in)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.norm(self.dropout(out) + x)
        return out, hidden


class OracleFCNDecoder(nn.Module):
    """Oracle fully-connected decoder with optional forecast conditioning."""

    def __init__(
        self,
        *,
        d_model_in: int,
        d_model_ff: int,
        d_model_out: int,
        lookback_size: int = 1,
        num_hidden_layers: int = 1,
        dropout_prob: float = 0.2,
        future_feature_idx: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.lookback_size = int(lookback_size)
        if self.lookback_size < 1:
            raise ValueError(f"lookback_size must be >= 1, got {lookback_size}.")
        self.future_feature_indices = list(int(i) for i in future_feature_idx)
        self.future_feature_idx = self.future_feature_indices

        d_in = d_model_in * self.lookback_size + len(self.future_feature_indices) + 1
        self.first = nn.Linear(d_in, d_model_ff)
        self.act1 = nn.PReLU(1)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(d_model_ff, d_model_ff) for _ in range(num_hidden_layers)
        )
        self.hidden_acts = nn.ModuleList(nn.PReLU(1) for _ in range(num_hidden_layers))
        self.dropout = nn.Dropout(dropout_prob)
        self.final = nn.Linear(d_model_ff, d_model_out)

    def forward(
        self,
        x: torch.Tensor,
        *,
        forecasting: bool = False,
        idx: int = 0,
        pad_value: float = -10.0,
    ) -> torch.Tensor:
        if self.lookback_size > 1:
            x = self._apply_lookback(x, pad_value=pad_value)
        x = self._prep_data(x, forecasting=forecasting, idx=idx)

        x = self.act1(self.dropout(self.first(x)))
        for layer, act in zip(self.hidden_layers, self.hidden_acts):
            x = act(self.dropout(layer(x)))
        return self.final(self.dropout(x))

    def _apply_lookback(self, x: torch.Tensor, *, pad_value: float) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_pad = F.pad(x, (0, 0, self.lookback_size, 0), value=pad_value)
        windows = x_pad.unfold(dimension=1, size=self.lookback_size, step=1)[:, :seq_len, :, :]
        return windows.permute(0, 1, 3, 2).reshape(batch_size, seq_len, self.lookback_size * d_model)

    def _select_future_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.future_feature_indices:
            return x[..., :0]
        return x[..., self.future_feature_indices]

    def _prep_data(self, x: torch.Tensor, *, forecasting: bool, idx: int) -> torch.Tensor:
        seq_len = x.shape[1]
        if not forecasting:
            forecast_flag = x.new_zeros(x.shape[:-1] + (1,))
            return torch.cat([x, forecast_flag, self._select_future_features(x)], dim=-1)

        if idx <= 0 or idx >= seq_len:
            raise ValueError(f"forecast idx must satisfy 0 < idx < seq_len={seq_len}, got idx={idx}.")
        out = x[:, idx - 1 : idx, :].expand(-1, seq_len - idx, -1)
        forecast_flag = out.new_ones(out.shape[:-1] + (1,))
        future_marks = self._select_future_features(x[:, idx:, :])
        return torch.cat([out, forecast_flag, future_marks], dim=-1)


class OracleDistDecoder(nn.Module):
    """Decode Oracle time pre-parameters into a Weibull mixture."""

    def __init__(self, *, num_dist_components: int) -> None:
        super().__init__()
        self.num_dist_components = int(num_dist_components)
        if self.num_dist_components < 1:
            raise ValueError("num_dist_components must be >= 1.")
        self.min_shape = 1e-1
        self.max_shape = 1e2
        self.min_log_scale = -20.0
        self.max_log_scale = 20.0

    def forward(self, x: torch.Tensor) -> dist.MixtureSameFamily:
        log_mean, shape, weight_logits = torch.split(
            x,
            [self.num_dist_components, self.num_dist_components, self.num_dist_components],
            dim=-1,
        )

        log_mean = torch.nan_to_num(log_mean, nan=0.0, posinf=20.0, neginf=-20.0)
        shape = torch.nan_to_num(shape, nan=0.0, posinf=20.0, neginf=-20.0)
        weight_logits = torch.nan_to_num(weight_logits, nan=0.0, posinf=0.0, neginf=0.0)

        log_mean = torch.tanh(log_mean) * (5.0 * 2.302585093)
        shape = F.softplus(shape).clamp(self.min_shape, self.max_shape)
        weight_logits = F.log_softmax(weight_logits, dim=-1)

        log_scale = (torch.lgamma(1 + shape.reciprocal()) - log_mean) * shape
        log_scale = torch.nan_to_num(
            log_scale,
            nan=0.0,
            posinf=self.max_log_scale,
            neginf=self.min_log_scale,
        ).clamp(self.min_log_scale, self.max_log_scale)
        scale = torch.exp(log_scale)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )


__all__ = ["OracleRNNEncoder", "OracleFCNDecoder", "OracleDistDecoder"]
