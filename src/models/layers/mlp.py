import torch.nn as nn
from typing import List, Type

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation with configurable hidden layers, activation functions, and dropout.
    """
    def __init__(self,
                 hidden_layers_width: List[int] = [128, 64, 32],
                 input_size: int = 30,
                 output_size: int = 32,
                 dropout_rate: float = 0.2,
                 activation: Type[nn.Module] = nn.GELU):
        """
        activation: the activation function class (not instance), e.g., nn.ReLU, nn.GELU, nn.Tanh
        """
        super(MLP, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.dropout_rate = dropout_rate

        layers: List[nn.Module] = []
        self.layers_width = [self.input_size] + self.hidden_layers_width

        for i in range(len(self.layers_width) - 1):
            layers += [
                nn.Linear(self.layers_width[i], self.layers_width[i + 1]),
                nn.LayerNorm(self.layers_width[i + 1]),
                activation(),  # instantiate the activation function
                nn.Dropout(self.dropout_rate)
            ]

        layers += [nn.Linear(self.layers_width[-1], self.output_size)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)