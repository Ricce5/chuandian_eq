import torch
import torch.nn.functional as F
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_fn

device = "cuda"
dtype_x = torch.bfloat16   # 或 float16 / float32
D = 2048                   # dim
K = 4                      # width
B = 8                      # batch
T = 16                     # seqlen
state_len = K - 1          # 最小也可以；更大也行（>= K-1）

# 为了可复现
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 1) 初始化权重
weight = torch.randn(D, K, device=device, dtype=torch.float32)
bias = torch.randn(D, device=device, dtype=torch.float32)  # 可选；不要就设 None
activation = "silu"  # "silu"/"swish"/None

# 2) 构造“一次性输入”
# x_full: (B, D, T)
x_full = torch.randn(B, D, T, device=device, dtype=dtype_x)

# -------------------------
# A) 一次性前向（对照组）
# -------------------------
y_full = causal_conv1d_fn(x_full, weight, bias, activation=activation)  # (B, D, T)

# -------------------------
# B) 逐 token update（实验组）
# -------------------------
conv_state = torch.zeros(B, D, state_len, device=device, dtype=dtype_x)

ys = []
for t in range(T):
    x_t = x_full[:, :, t:t+1]  # (B, D, 1)
    out_t = causal_conv1d_update(
        x_t, conv_state, weight, bias,
        activation=activation
    )  # (B, D, 1)
    ys.append(out_t)

y_step = torch.cat(ys, dim=-1)  # (B, D, T)

# -------------------------
# C) 比较一致性
# -------------------------
# 误差阈值：bf16/fp16 会更松一点
if dtype_x == torch.float32:
    rtol, atol = 3e-4, 1e-3
elif dtype_x == torch.float16:
    rtol, atol = 3e-3, 5e-3
else:  # bfloat16
    rtol, atol = 1e-2, 5e-2

max_diff = (y_full - y_step).abs().max().item()
mean_diff = (y_full - y_step).abs().mean().item()
print("y_full shape:", y_full.shape)
print("y_step shape:", y_step.shape)
print("max diff:", max_diff)
print("mean diff:", mean_diff)

ok = torch.allclose(y_full, y_step, rtol=rtol, atol=atol)
print("allclose:", ok)

# 如果你希望严格检查（不通过就报错）
assert ok, f"Mismatch! max_diff={max_diff}, mean_diff={mean_diff}, rtol={rtol}, atol={atol}"
