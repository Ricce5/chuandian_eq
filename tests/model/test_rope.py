import torch
from flash_attn.layers.rotary import RotaryEmbedding

rotary_emb = RotaryEmbedding(dim=64).cuda()  # Move the module to CUDA

q = torch.randn(2, 10, 8, 64).cuda()  # Move the tensor to CUDA
kv = torch.randn(2, 10, 2, 8, 64).cuda()  # Move the tensor to CUDA

q_rot, kv_rot = rotary_emb(q, kv)