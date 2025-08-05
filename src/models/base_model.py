import torch.nn as nn
import torch.nn.functional as F
from src.utils.registrable import Registrable
from typing import Optional, Dict, Any


class BaseModel(nn.Module):
    def __init__(self, encoder, input_adapter, device,unpack_adapter_output=False):
        super().__init__()
        self.encoder = encoder.to(device)
        self.input_adapter = input_adapter  # callable: batch -> dict {"features": ..., "event_time": ...}
        self.device = device
        self.unpack_adapter_output = unpack_adapter_output  

    def forward(self,batch,caches):
        """
        输入:
            batch: Tensor [B, L, F]
        输出:
            encoder_out: [B, L, D]
            non_pad_mask: [B, L, 1]
        """
        assert batch.device == self.device, f"Input tensor on {batch.device}, but model on {self.device}"
        batch = batch.float()
        batch = batch.to(self.device)
        # 输入适配器返回 dict，而非 tuple
        
        inputs = self.input_adapter(batch)  # dict: {"features": ..., "event_time": ...}
        if self.unpack_encoder_output:
            out, non_pad_mask, new_cache = self.encoder(inputs, caches=caches)
            return out, non_pad_mask, new_cache
        else:
            out = self.encoder(inputs, caches=caches)
            return out

    def set_attn_type(self, new_type: str):
        self.encoder.set_attn_type(new_type)

        
    def set_attn_dropout(self, p: float):
        self.encoder.set_attn_dropout(p)



