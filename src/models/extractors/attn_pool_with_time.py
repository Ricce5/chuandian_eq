# models/extractors/attn_pool_with_time.py
from .base import RepresentationExtractor
from ..layers.attention_pooling import AttentionPooling
import torch

@RepresentationExtractor.register("attn_time")
class AttentionPoolingWithTimeExtractor(RepresentationExtractor):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pool = AttentionPooling(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, enc_out, non_pad_mask, extra_inputs=None):
        if extra_inputs is None or 't_n_seq' not in extra_inputs:
            raise ValueError("t_n_seq is required for AttentionPoolingWithTimeExtractor")
        
        t_n_seq = extra_inputs['t_n_seq'].unsqueeze(-1)  # [B, L, 1]
        pooling_input = torch.cat((enc_out, t_n_seq), dim=-1)  # [B, L, D+1]
        mask = non_pad_mask.squeeze(-1)  # [B, L]
        pooled, _ = self.pool(pooling_input, mask)
        return pooled
