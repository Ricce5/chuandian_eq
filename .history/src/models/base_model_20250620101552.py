import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, AttentionPooling
from .transformer import Transformer, Transformer_ST, Transformer_STM, Transformer_SE
from .TppModels import THP
from src.utils.registrable import Registrable
from src.utils.mask_utils import get_last_valid_step



class BaseTransformerModel(nn.Module,Registrable):
    def __init__(self, transformer, mlp, device):
        super().__init__()
        self.transformer = transformer.to(device)
        self.mlp = mlp.to(device)
        self.device = device

    def forward(self, x):
        x = x.float()
        inputs = self._batch_to_input(x)
        enc_out, non_pad_mask = self.transformer(*inputs)
        enc_last, _ = get_last_valid_step(enc_out, non_pad_mask)
        out = self.mlp(enc_last)
        return out.squeeze(1)

    def _batch_to_input(self, x):
        raise NotImplementedError("Child class must implement this method.")
    


