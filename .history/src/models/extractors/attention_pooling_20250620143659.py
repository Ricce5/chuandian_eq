# models/extractors/attention_pooling.py
from .base import RepresentationExtractor
from layers.attention_pooling import AttentionPooling

@RepresentationExtractor
class AttentionPoolingExtractor(RepresentationExtractor):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pool = AttentionPooling(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, enc_out, non_pad_mask, extra_inputs=None):
        mask = non_pad_mask.squeeze(-1)  # [B, L]
        pooled, _ = self.pool(enc_out, mask)  # [B, D]
        return pooled
