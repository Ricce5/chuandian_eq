import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from .layers import MLP, AttentionPooling
from .transformer import Transformer, Transformer_ST, Transformer_STM, Transformer_SE
from .TppModels import THP
from src.utils.registrable import Registrable
from src.utils.mask_utils import get_last_valid_step


class BaseModel(nn.Module):
    """
    通用模型骨干（Backbone），包含：
    - encoder: Transformer / LSTM / 其他序列建模器
    - input_adapter: 负责从原始输入构造 encoder 所需的输入
    """

    def __init__(self, encoder, input_adapter, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.input_adapter = input_adapter  # callable: x -> (features, t_n_seq, ...)
        self.device = device

    def forward(self, x):
        """
        输入:
            x: Tensor [B, L, F]
        输出:
            encoder_out: [B, L, D]
            non_pad_mask: [B, L, 1]
        """
        x = x.float()
        inputs = self.input_adapter(x)  # e.g. (f_seq, t_n_seq, ...)
        encoder_out, non_pad_mask = self.encoder(*inputs)
        return encoder_out, non_pad_mask


    

class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim, head_type="mlp", dropout=0.1):
        super().__init__()
        if head_type == "mlp":
            self.head = MLP([input_dim], input_dim, output_dim, dropout)
        elif head_type == "linear":
            self.head = nn.Linear(input_dim, output_dim)

        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def forward(self, x):
        return self.head(x)


