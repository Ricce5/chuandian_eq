from .base import RepresentationExtractor
from ..layers.attention_pooling import AttentionPooling
import torch



@RepresentationExtractor.register("attn_time")
class AttentionPoolingWithTimeExtractor(RepresentationExtractor):
    """
    Attention Pooling with Event Time as Additional Feature
    """
    def __init__(self, input_dim, hidden_dim,device=None):
        super().__init__()
        self.pool = AttentionPooling(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    def forward(self, enc_out, non_pad_mask, extra_inputs=None):
        """
        enc_out: Tensor [B, L, D]
        non_pad_mask: Tensor [B, L, 1]
        extra_inputs: dict, must contain 'event_time' (or 't_n_seq')
        """
        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("Missing 'event_time' in extra_inputs for AttentionPoolingWithTimeExtractor")
        
        event_time = extra_inputs['event_time'].unsqueeze(-1)  # [B, L, 1]
        pooling_input = torch.cat((enc_out, event_time), dim=-1)  # [B, L, D + 1]
        mask = non_pad_mask.squeeze(-1)  # [B, L]

        pooled, _ = self.pool(pooling_input, mask)
        return pooled




