import torch.nn as nn
from src.models.layers.mlp import MLP

class TaskHead(nn.Module):
    """
    Task head module that encapsulates different types of output layer structures, such as:
    - MLP (Multi-Layer Perceptron)
    - Single-layer Linear
    """

    def __init__(self, input_dim, output_dim, head_type="mlp", hidden_layers=None, dropout=0.1,device=None):
        """
        Args:
            input_dim (int): Input dimension provided by encoder + extractor.
            output_dim (int): Output dimension (usually 1, unless multi-class).
            head_type (str): Type of head, options are 'mlp' or 'linear'.
            hidden_layers (list, optional): List of hidden layer sizes for MLP, e.g., [128, 64].
            dropout (float): Dropout probability for MLP.
        """
        super().__init__()

        self.head_type = head_type.lower()

        if self.head_type == "mlp":
            if hidden_layers is None:
                hidden_layers = [input_dim]  
            self.head = MLP(
                hidden_layers_width=hidden_layers,
                input_size=input_dim,
                output_size=output_dim,
                dropout_rate=dropout
            ).to(device)
        elif self.head_type == "linear":
            self.head = nn.Linear(input_dim, output_dim,bias=True).to(device)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x):
        return self.head(x)
