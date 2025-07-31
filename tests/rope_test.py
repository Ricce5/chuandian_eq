import torch
from src.models.mha.rotary_embedding import RotaryEmbedding
rotary_emb = RotaryEmbedding(dim=64)

q = torch.randn(2, 10, 8, 64)  # (batch, seqlen, nheads, dim)
kv = torch.randn(2, 10, 2, 8, 64)  # (batch, seqlen, 2, nheads, dim)

positions = torch.linspace(0.0, 1.0, steps=10).unsqueeze(0).repeat(2, 1)  # shape (2, 10)

q_rot, kv_rot = rotary_emb(q, kv, positions=positions)