import torch.nn as nn
import torch.nn.functional as F
from src.utils.registrable import Registrable


class BaseModel(nn.Module):
    def __init__(self, encoder, input_adapter, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.input_adapter = input_adapter  # callable: x -> dict {"features": ..., "event_time": ...}
        self.device = device

    def forward(self, x):
        """
        输入:
            x: Tensor [B, L, F]
        输出:
            encoder_out: [B, L, D]
            non_pad_mask: [B, L, 1]
        """
        assert x.device == self.device, f"Input tensor on {x.device}, but model on {self.device}"
        x = x.float()

        # 输入适配器返回 dict，而非 tuple
        inputs = self.input_adapter(x)  # dict: {"features": ..., "event_time": ...}

        # 只传一个参数给 encoder（BaseTransformer）
        encoder_out, non_pad_mask, _ = self.encoder(inputs)

        return encoder_out, non_pad_mask

    def set_attn_type(self, new_type: str):
        self.encoder.set_attn_type(new_type)

        
    def set_attn_dropout(self, p: float):
        self.encoder.set_attn_dropout(p)



