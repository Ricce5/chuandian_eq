import torch.nn as nn
from typing import List, Type

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable hidden layers, activation, dropout,
    and optional normalization type ('layer' or 'batch'). Set use_norm=False to disable.
    """
    def __init__(self,
                 hidden_layers_width: List[int] = [128, 64, 32],
                 input_size: int = 30,
                 output_size: int = 32,
                 dropout_rate: float = 0.2,
                 activation: Type[nn.Module] = nn.GELU,
                 use_norm: bool = True,
                 norm_type: str = "layer",
                 linear_bias: bool = True):
        """
        norm_type: 'layer' (LayerNorm) or 'batch' (BatchNorm1d). use_norm=False disables normalization.
        activation: the activation function class (not instance), e.g., nn.ReLU, nn.GELU, nn.Tanh
        """
        super(MLP, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_type = (norm_type or "").lower()
        self.linear_bias = bool(linear_bias)

        layers: List[nn.Module] = []
        self.layers_width = [self.input_size] + self.hidden_layers_width

        for i in range(len(self.layers_width) - 1):
            out_features = self.layers_width[i + 1]
            layers.append(
                nn.Linear(
                    self.layers_width[i],
                    out_features,
                    bias=self.linear_bias,
                )
            )

            if self.use_norm:
                if self.norm_type in ("layer", "layernorm"):
                    layers.append(nn.LayerNorm(out_features))
                elif self.norm_type in ("batch", "batchnorm"):
                    layers.append(nn.BatchNorm1d(out_features))
                else:
                    raise ValueError(f"Unsupported norm_type: {norm_type!r}. Use 'layer' or 'batch'.")

            layers.append(activation())  # instantiate the activation function
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(
            nn.Linear(
                self.layers_width[-1],
                self.output_size,
                bias=self.linear_bias,
            )
        )
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
