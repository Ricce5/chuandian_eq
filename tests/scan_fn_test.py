import torch
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例参数，保持与之前相同的形状
B_batch, D_inner, L_len, D_state = 4, 128, 10, 16  # 批次大小、状态维度、序列长度、输入维度

# 创建变量，形状与之前给定的张量一致
x = torch.randn(B_batch, D_inner, L_len, device=device)   # x: (4, 128, 10)
dt = torch.randn(B_batch, D_inner, L_len, device=device)  # dt: (4, 128, 10)
A = torch.randn(D_inner, D_state, device=device)                # A: (128, 16)
# B = torch.randn(B_batch, D_state, L_len, device=device)       # B: (4, 16, 10)
# C = torch.randn(B_batch, D_state, L_len, device=device)       # C: (4, 16, 10)
B = torch.randn(B_batch, D_state, device=device)       # B: (4, 16, 10)
C = torch.randn(B_batch, D_state, device=device)       # C: (4, 16, 10)
D = torch.randn(D_inner, device=device)                   # D: (128)
# z = torch.randn(B_batch, D_inner, L_len, device=device)   # z: (4, 128, 10)
z =None

# 设置变化后的形状
x_new = torch.randn(B_batch, D_inner, 1, device=device)   # 新的 x: (4, 128, 1)
dt_new = torch.randn(B_batch, D_inner, 1, device=device)  # 新的 dt: (4, 128, 1)
B_new = torch.randn(B_batch, D_state, 1, device=device)       # 新的 B: (4, 16, 1)
C_new = torch.randn(B_batch, D_state, 1, device=device)       # 新的 C: (4, 16, 1)
z_new = torch.randn(B_batch, D_inner, 1, device=device)   # 新的 z: (4, 128, 1)

# 使用 selective_scan_fn 执行操作
x_out = selective_scan_fn(
    u=x_new, 
    delta=dt_new,  # delta: 与 x_new 和 dt_new 相同
    A=A, 
    B=B_new,  # 新的 B: (4, 16, 1)
    C=C_new,  # 新的 C: (4, 16, 1)
    D=D, 
    z=z_new,   # 新的 z: (4, 128, 1)
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False
)

# 打印输出的形状，应该与输入 x_new 的形状一致
print(x_out.shape)  # 结果应为 (4, 128, 1)
