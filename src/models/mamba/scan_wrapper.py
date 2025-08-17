import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

class BoundedSelectiveScanWrapper(nn.Module):
    def __init__(self, d_model, d_state, device, B_range=None, C_range=None, output_range=None, **kwargs):
        """
        d_model: 模型的特征维度
        d_state: 状态空间的维度
        device: 当前设备，cpu或cuda
        B_range: B 参数的范围 (min_val, max_val)，可选
        C_range: C 参数的范围 (min_val, max_val)，可选
        output_range: 输出的范围 (min_val, max_val)，可选
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device
        self.B_range = B_range if B_range is not None else (0, 1)  # 默认范围 [0, 1]
        self.C_range = C_range if C_range is not None else (1, 2)  # 默认范围 [-1, 1]
        self.output_range = output_range 

        # A: (d_model, d_state)
        self.A = torch.zeros(d_model, d_state, device=device)
        # B, C 是可学习的参数，将它们初始化并应用范围
        self.B = nn.Parameter(self._init_bounded_tensor(d_model, d_state, *self.B_range))  # 根据范围初始化 B
        if self.C_range is None:
            self.C = torch.ones(d_model, d_state, device=device)
        else:
            self.C = nn.Parameter(self._init_bounded_tensor(d_model, d_state, *self.C_range))
        self.D = torch.zeros(d_model, device=device) 
        self.delta_bias = None  # 可训练的偏置，可以根据需要启用

    def forward(self, x, delta):
        """
        x: 输入张量，形状为 (batch, length, d_model)
        delta: 时间步长张量，形状为 (batch, length, d_model)
        """
        # 将输入张量从 (B, L, D) 转换为 (B, D, L)，以符合 selective_scan_fn 的要求
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')

        y_out = selective_scan_fn(
            u=self._bounded_tanh(u), 
            delta=delta_rearranged, 
            A=self.A, 
            B=self._bounded_tanh(self.B),  # 应用范围约束的 B
            C=self._bounded_tanh(self.C),  # 应用范围约束的 C
            D=self.D, 
            delta_bias=self.delta_bias
        )
        
        # 将输出从 (B, D, L) 转换回 (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    

        if self.output_range is not None:
            y_out = self._bounded_tanh(y_out, *self.output_range)

        return y_out

    def _bounded_tanh(self, input, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        使用带有范围限制的 tanh 激活函数，将输出限制在[min_val, max_val]范围内。
        """
        output = min_val + (max_val - min_val) * 0.5 * (torch.tanh(input) + 1)
        return output

    def _init_bounded_tensor(self, d_model, d_state, min_val, max_val):
        """
        初始化一个带有指定范围限制的张量，并返回这个张量。
        """
        tensor = torch.randn(d_model, d_state)  # 初始化为标准正态分布
        # 将张量值限制在 [min_val, max_val] 范围内
        return min_val + (max_val - min_val) * 0.5 * (torch.tanh(tensor) + 1)