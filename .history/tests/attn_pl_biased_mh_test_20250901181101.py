import torch
import torch.nn as nn
from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH

# ====== 构造假数据并测试 ======
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, L, D = 2, 6, 8           # batch=2, 序列长=6, 特征维=8
H, Dh = 4, 16               # 头数=4, 每头hidden=16
x = torch.randn(B, L, D, device=device)

# 构造 mask：每个样本有效长度不同
valid_len = torch.tensor([6, 4], device=device)  # 第2个样本后两步是padding
mask = torch.arange(L, device=device).unsqueeze(0).repeat(B,1) < valid_len.unsqueeze(1)  # [B,L]

# 事件时间（已缩放到[0,1]）；确保 padding 位置的时间不会影响（mask会屏蔽）
t = torch.linspace(0.2, 1.0, L, device=device).unsqueeze(0).repeat(B,1)  # [B,L]
t[1, 4:] = 0.0  # 给padding位置随便填，反正会被mask掉

# ====== 1) 线性偏置 + concat 输出 ======
pool_lin = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                               n_heads=H, agg="concat", device=device)
pooled_lin, alpha_lin = pool_lin(x, mask, extra_inputs={"event_time": t}, return_score=True)
print("linear/concat -> pooled:", pooled_lin.shape, " alpha:", alpha_lin.shape)
# 期望: pooled = [B, H*D] = [2, 32]；alpha = [B,H,L] = [2,4,6]

# 反传测试
loss = pooled_lin.mean()
loss.backward()
print("backward ok (linear)")

# ====== 2) 对数偏置 + mean 输出 ======
pool_log = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="log",
                               alpha0=10.0, n_heads=H, agg="mean", device=device)
pooled_log, alpha_log = pool_log(x, mask, extra_inputs={"event_time": t}, return_score=True)
print("log/mean -> pooled:", pooled_log.shape, " alpha:", alpha_log.shape)
# 期望: pooled = [B, D] = [2, 8]；alpha = [2,4,6]

# 反传测试
loss = (pooled_log**2).sum()
loss.backward()
print("backward ok (log)")

# ====== 3) 检查 mask 是否生效：第2个样本的 alpha 在无效位应接近0 ======
with torch.no_grad():
    print("alpha (sample=1):\n", alpha_lin[1])  # [H,L]
    print("mask (sample=1):\n", mask[1].int())  # [L]
