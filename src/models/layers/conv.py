import torch
import torch.nn as nn
from typing import Optional
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class Causalconv(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 3, num_layers: int = 3,
                 activation: str = "silu", residual: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.activation = activation
        self.residual = residual

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(num_layers):
            w = nn.Parameter(torch.randn(channels, kernel_size, dtype=torch.float32) * 0.01)
            b = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) or (B, L) if C==1
        return: same shape as x
        """
        if causal_conv1d_fn is None:
            return x
        squeeze_last = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeeze_last = True

        # (B, C, L)
        y = x.transpose(1, 2).contiguous()

        for w, b in zip(self.weights, self.biases):
            # activation 可选：None / "silu"
            out = causal_conv1d_fn(y, w, b, activation=self.activation)
            out = out[:, :, : y.size(-1)]
            if self.residual:
                y = y + out
            else:
                y = out
        y = y.transpose(1, 2).contiguous()  # (B, L, C)
        return y.squeeze(-1) if squeeze_last else y