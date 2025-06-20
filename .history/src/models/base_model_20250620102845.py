import torch
import torch.nn as nn
import torch.nn.functional as F
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

    


