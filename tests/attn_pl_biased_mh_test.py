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
# %%

# ====== 4) 新增功能测试：融合 last_token ======
from copy import deepcopy

# 构造一个可导的 last_token（模拟编码器最后一个 token 的表示）
last_token = torch.randn(B, D, device=device, requires_grad=True)

# 4.1 残差加和 + 方差匹配 s（不开 LN，方便数值对齐验证）
pool_add = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                               n_heads=H, agg="mean",
                               fuse_mode="add", use_ln=False, use_var_scale=True,
                               device=device)
pooled_add, alpha_add = pool_add(x, mask, extra_inputs={"event_time": t},
                                 return_score=True, last_token=last_token)
print("fuse=add(use_var_scale=True) ->", pooled_add.shape)  # 期望 [B, D]

# 数值校验：手动计算 s 并核对 fused = lt + s*pt
with torch.no_grad():
    # 先得到不融合的 pooled（复用内部逻辑最稳，这里简单再前向一次但 fuse_mode=none）
    pool_none = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                    n_heads=H, agg="mean",
                                    fuse_mode="none", device=device)
    # 拷贝权重，保证输出一致
    pool_none.load_state_dict({k: v for k, v in pool_add.state_dict().items()
                               if "ln_" not in k and "fuse_" not in k and "gate_" not in k}, strict=False)
    pooled_plain = pool_none(x, mask, extra_inputs={"event_time": t}, return_score=False)  # [B,D]

    lt = last_token.detach()
    pt = pooled_plain.detach()
    var_lt = lt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
    var_pt = pt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
    s = torch.sqrt(var_lt / var_pt)  # [B,1]
    fused_manual = lt + s * pt
    diff = (pooled_add.detach() - fused_manual).abs().max().item()
    print(f"[check add+scale] max|diff| = {diff:.6g}")

# 反传看梯度是否能回到 last_token
(pooled_add.sum()).backward(retain_graph=True)
print("grad on last_token (add):", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.2 concat 融合 + 线性降回 d_model
pool_concat = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                  n_heads=H, agg="mean",
                                  fuse_mode="concat", repr_dim=D, use_ln=True,
                                  device=device)
pooled_concat, _ = pool_concat(x, mask, extra_inputs={"event_time": t},
                               return_score=True, last_token=last_token)
print("fuse=concat ->", pooled_concat.shape)  # 期望 [B, D]
loss = pooled_concat.mean()
loss.backward()
print("backward ok (concat) | grad last_token:", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.3 gate 门控融合（样本级自适应）
pool_gate = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                n_heads=H, agg="mean",
                                fuse_mode="gate", use_ln=True,
                                device=device)
pooled_gate, _ = pool_gate(x, mask, extra_inputs={"event_time": t},
                           return_score=True, last_token=last_token)
print("fuse=gate ->", pooled_gate.shape)  # 期望 [B, D]
loss = (pooled_gate**2).sum()
loss.backward()
print("backward ok (gate) | grad last_token:", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.4 未提供 last_token 的 fallback（应使用 mask 找到每样本最后有效 token）
pool_add_fb = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                  n_heads=H, agg="mean",
                                  fuse_mode="add", use_ln=True, use_var_scale=False,
                                  device=device)
pooled_fb = pool_add_fb(x, mask, extra_inputs={"event_time": t}, return_score=False)  # 不传 last_token
print("fuse=add (fallback last_token) ->", pooled_fb.shape)  # 期望 [B, D]

# 4.5 兼容 agg='concat' 的情形：建议用 fuse_mode='concat'/'gate'（'add' 维度不配）
pool_cat_heads = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                     n_heads=H, agg="concat",
                                     fuse_mode="concat", repr_dim=D, device=device)
pooled_cat_heads = pool_cat_heads(x, mask, extra_inputs={"event_time": t},
                                  return_score=False, last_token=last_token)
print("agg=concat + fuse=concat ->", pooled_cat_heads.shape)  # 期望 [B, D]
