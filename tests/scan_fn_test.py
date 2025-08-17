import torch
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例参数，保持与之前相同的形状
B_batch, D_inner, L_len, D_state = 4, 128, 10, 1  # D_inner 对应输入特征维数乘expand

# 创建变量，形状与之前给定的张量一致
x = torch.randn(B_batch, D_inner, L_len, device=device)   # x: (4, 128, 10)
dt = torch.randn(B_batch, D_inner, L_len, device=device)  # dt: (4, 128, 10)
A = torch.randn(D_inner, D_state, device=device)                # A: (128, 16) # D_inner个对角阵
# B = torch.randn(B_batch, D_state, L_len, device=device)       # B: (4, 16, 10)
# C = torch.randn(B_batch, D_state, L_len, device=device)       # C: (4, 16, 10)
B = torch.randn(D_inner, D_state, device=device)
C = torch.randn(D_inner, D_state, device=device)
D = torch.randn(D_inner, device=device)                   # D: (128)
# z = torch.randn(B_batch, D_inner, L_len, device=device)   # z: (4, 128, 10)
z = None


# 使用 selective_scan_fn 执行操作
x_out = selective_scan_fn(
    u=x, 
    delta=dt,  # delta: 与 x_new 和 dt_new 相同
    A=A, 
    B=B,  # 新的 B: (4, 16, 1)
    C=C,  # 新的 C: (4, 16, 1)
    D=D, 
    z=z,   # 新的 z: (4, 128, 1)
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False
)

# 打印输出的形状，应该与输入 x_new 的形状一致
print(x_out.shape)  # 结果应为 (4, 128, 1)
