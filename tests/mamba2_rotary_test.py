import torch
from src.models.mamba.mamba2_rotary import Mamba2Rotary

B, S, D = 2, 6, 64
T = torch.arange(S).unsqueeze(0).expand(B, -1).float()

model = Mamba2Rotary(
    d_model=D,
    rotary_emb_scale_base=512,
    rotary_emb_center_mode='dynamic',
    layer_idx=0
).to(dtype=torch.float16, device="cuda")

x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
T = T.to(dtype=torch.float16, device="cuda")

# ===== 全序列一次性 forward =====
out_full = model(x, times=T, inference_params=None)

# ===== 逐 token 推理 =====
conv_state, ssm_state = model.allocate_inference_cache(batch_size=B, max_seqlen=S, dtype=torch.float16)
out_tokens = []
for t in range(S):
    xt = x[:, t:t+1]
    Tt = T[:, t:t+1]
    yt, conv_state, ssm_state = model.step(xt, conv_state, ssm_state, times=Tt)
    out_tokens.append(yt)
out_step = torch.cat(out_tokens, dim=1)

# ===== 对比差异 =====
diff = (out_full - out_step).abs().max()
print(f"Max difference between forward and step: {diff.item()}")
