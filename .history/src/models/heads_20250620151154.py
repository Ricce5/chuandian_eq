import torch.nn as nn
from layers.mlp import MLP

class TaskHead(nn.Module):
    """
    任务头模块，封装不同类型的输出层结构，如：
    - MLP
    - 单层 Linear
    """

    def __init__(self, input_dim, output_dim, head_type="mlp", hidden_layers=None, dropout=0.1):
        """
        Args:
            input_dim: 输入维度（encoder + extractor 提供）
            output_dim: 输出维度（通常为1，除非多分类）
            head_type: 头类型，可选 'mlp', 'linear'
            hidden_layers: 隐藏层列表（用于 MLP），如 [128, 64]
            dropout: dropout 概率（MLP）
        """
        super().__init__()

        self.head_type = head_type.lower()

        if self.head_type == "mlp":
            if hidden_layers is None:
                hidden_layers = [input_dim]  # 如果未指定则使用一层
            self.head = MLP(
                hidden_layers_width=hidden_layers,
                input_size=input_dim,
                output_size=output_dim,
                dropout_rate=dropout
            )
        elif self.head_type == "linear":
            self.head = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x):
        return self.head(x)
