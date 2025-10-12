import torch

from src.models.mha.mha_time import MHATime  
from mamba_ssm.modules.mha import MHA

B, S, D, H = 2, 6, 64, 4  # batch, seqlen, embed_dim, num_heads
T = torch.arange(S).unsqueeze(0).expand(B, -1).float()  # time indices

model = MHATime(
    embed_dim=D,
    num_heads=H,
    rotary_emb_dim=D // H // 2,
    layer_idx=0
).to(dtype=torch.float16, device="cuda")

x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
T = T.to(dtype=torch.float16, device="cuda")  # <== 时间索引也需要同样类型

out = model(x, times=T, inference_params=None)
print(out.shape)  # (B, S, D)

