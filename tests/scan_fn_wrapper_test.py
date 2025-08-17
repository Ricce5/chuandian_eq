import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelectiveScanWrapper(nn.Module):
    def __init__(self, d_model, d_state, device, **kwargs):
        """
        d_model: 模型的特征维度
        d_state: 状态空间的维度
        device: 当前设备，cpu或cuda
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device

        # A: (d_model, d_state)
        self.A = torch.zeros(d_model, d_state, device=device)  # 此处 A 为零，你可以根据需要调整为非零值
        # B 和 C 是可学习的参数，我们为它们创建可训练的张
        self.B = nn.Parameter(torch.randn(d_model, d_state, device=device))
        self.C = nn.Parameter(torch.randn(d_model, d_state, device=device))
        # D 是一个与 d_model 维度相同的张量，这里我们将其初始化为零
        self.D = torch.zeros(d_model, device=device)  # 此处 D 为零，你可以根据需要调整为非零值
        # delta_bias 参数
        self.delta_bias = None  # nn.Parameter(torch.randn(d_model, device=device))  # 可训练的偏置

    def forward(self, x, delta):
        """
        x: 输入张量，形状为 (batch, length, d_model)
        delta: 时间步长张量，形状为 (batch, length, d_model)
        """
        # 将输入张量从 (B, L, D) 转换为 (B, D, L)，以符合 selective_scan_fn 的要求
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')

        y_out = selective_scan_fn(
            u=u, 
            delta=delta_rearranged, 
            A=self.A, 
            B=self.B, 
            C=self.C, 
            D=self.D, 
            delta_bias=self.delta_bias
        )
        
        # 将输出从 (B, D, L) 转换回 (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    
        return y_out

# --- 使用示例 ---
B_batch, L_len, D_model, D_state = 1, 10, 1, 1  # 4, 10, 128, 16

ssm_wrapper = SelectiveScanWrapper(d_model=D_model, d_state=D_state, device=device).to(device)

x_input = torch.randn(B_batch, L_len, D_model, device=device)
delta_input = torch.randn(B_batch, L_len, D_model, device=device)

output = ssm_wrapper(x_input, delta_input) 

print(f"输入形状: {x_input.shape}")
print(f"输出形状: {output.shape}")
print(output)
