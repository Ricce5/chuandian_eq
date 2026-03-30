import abc
import torch
import torch.nn as nn
from src.models.layers.mlp import MLP
from .base import BGModel
from causal_conv1d import causal_conv1d_fn
from einops import rearrange


@BGModel.register("conv_mlp")
class ConvMLPBGModel(BGModel):
    def __init__(self,
                 d_feature,
                 scale_init,
                 hidden_layers_width=[16, 8],
                 activation=torch.nn.SiLU,
                 dropout=0.2,
                 device=None,
                 conv_kernel_size=4,
                 conv_bias=True,
                 mlp_bias=True,
                 conv_activation="silu",   #  "silu"/"swish"/None
                 num_conv_layers=1,     
                 ):
        super().__init__(device=device, scale_init=scale_init)
        self.device = device

        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_feature if i == 0 else d_feature,
                out_channels=d_feature,
                kernel_size=conv_kernel_size,
                groups=d_feature,  # depthwise conv
                padding=conv_kernel_size - 1,
                bias=conv_bias
            ) for i in range(num_conv_layers)
        ])

        self.conv_activation = conv_activation
        # MLP expects last-dim size = conv_out_channels
        self.mlp = MLP(
            input_size=d_feature,
            output_size=1,
            hidden_layers_width=hidden_layers_width,
            activation=activation,
            dropout_rate=dropout,
            use_norm=False,
            linear_bias=mlp_bias,
        )

        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        #  (B, T, d_feature)
        B, T, D = time_series.shape

        x = time_series.permute(0, 2, 1).contiguous()  # (B, D, T)
        for conv in self.conv_layers:
            x = causal_conv1d_fn(
                x,
                weight=rearrange(conv.weight, "d 1 w -> d w"),
                bias=conv.bias,
                activation=self.conv_activation,
                seq_idx=None
            )
        # x shape: (B, conv_out_channels, T)
        x = x.permute(0, 2, 1)  # -> (B, T, conv_out_channels)
        # MLP expects (B, T, out_channels)
        out = self.mlp(x)  # (B, T, 1)
        return out
