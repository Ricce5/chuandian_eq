import torch
import torch.nn as nn
from functools import partial

from src.models.mha.mha_time import MHATime
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

B, S, D = 2, 8, 64   # batch size, sequence length, embedding dim
H = 4               # number of heads
rotary_emb_dim = D // H // 2 
mlp_hidden_dim = 256          

x = torch.randn(B, S, D).cuda()  
times = torch.arange(S).unsqueeze(0).expand(B, -1).to(dtype=torch.float32, device="cuda")


block = Block(
    dim=D,
    mixer_cls=partial(MHATime,
                      num_heads=H,
                      rotary_emb_dim=rotary_emb_dim,
                      layer_idx=0),
    mlp_cls=partial(GatedMLP,
                    hidden_features=mlp_hidden_dim,
                    out_features=D),
    norm_cls=nn.LayerNorm,
    fused_add_norm=False,
    residual_in_fp32= True,
).cuda()  


output, residual = block(x, residual=None, times=times, inference_params=None)

print("Output shape:", output.shape)     # Expect: (B, S, D)
print("Residual shape:", residual.shape) # Expect: (B, S, D)
