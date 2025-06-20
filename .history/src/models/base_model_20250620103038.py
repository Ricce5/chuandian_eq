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
    def __init__(self, encoder: nn.Module, input_adapter: Callable, device: torch.device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.input_adapter = input_adapter
        self.device = device

    def forward(self, x):
        x = x.float()
        model_inputs = self.input_adapter(x)
        features, mask = self.encoder(*model_inputs)
        return features, mask

    

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




class TaskModel(nn.Module):
    def __init__(self, base_model: BaseModel, head: TaskHead, final_activation=None):
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.final_activation = final_activation

    def forward(self, x):
        features, mask = self.base_model(x)
        last = self._extract_last_valid(features, mask)
        out = self.head(last)
        if self.final_activation:
            out = self.final_activation(out)
        return out.squeeze(1)

    @staticmethod
    def _extract_last_valid(enc_out, mask):
        # mask: [B, T, 1]
        lengths = mask.sum(dim=1).squeeze(-1).long()
        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(mask.size(0), device=mask.device)
        return enc_out[batch_idx, idx]
