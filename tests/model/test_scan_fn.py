import torch
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B_batch, D_inner, L_len, D_state = 4, 128, 10, 1  

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

x_out = selective_scan_fn(
    u=x, 
    delta=dt, 
    A=A, 
    B=B,  
    C=C,  
    D=D, 
    z=z,  
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False
)


print(x_out.shape)  
