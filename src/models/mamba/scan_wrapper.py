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
    

class BoundedDiscreteSSM(nn.Module):
    def __init__(self, device, B_range=None, output_range=None,output_init=None):
        super(BoundedDiscreteSSM, self).__init__()
        self.device = device  # Ensure that device is correctly initialized
        self.B_range = B_range if B_range is not None else (0, 1)
        self.output_range = output_range if output_range is not None else (0.5, 2)
        self.kappa = self._init_bounded_tensor(*self.B_range)  # Use self.device here
        if output_init is None:
            output_init = torch.tensor(0.8, device=self.device)  # Default initial output
        else:
            output_init = torch.tensor(output_init, device=self.device)
        out_min, out_max = self.output_range
        self.state_init = self._inverse_bounded_tanh(output=output_init, min_val=out_min, max_val=out_max)  # Use self.device here

    def forward(self, x, delta_t=None, return_last_state=False):
        B, L, D = x.shape
        state = torch.zeros(B, L, D, device=x.device)
        if delta_t is not None and delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)  # Make it [B, L, 1]
            delta_t = delta_t.expand(-1, -1, D)  # Expand to [B, L, D]
        if delta_t is not None:
            sequence = self.kappa * torch.tanh(x) * delta_t  # [B, L, D]
        else:
            sequence = self.kappa * torch.tanh(x) 
        state = torch.cumsum(sequence, dim=1)  
        if self.state_init is not None:
            state += self.state_init
        out = self._bounded_tanh(state, *self.output_range)
        if return_last_state:
            return out, state
        else:
            return out

    def _bounded_tanh(self, input, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        使用带有范围限制的 tanh 激活函数，将输出限制在[min_val, max_val]范围内。
        """
        output = min_val + (max_val - min_val) * 0.5 * (torch.tanh(input) + 1)
        return output

    def _inverse_bounded_tanh(self, output, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        计算给定 output 对应的原始 input。
        """
        clamped_output = torch.clamp(output, min=min_val + 1e-6, max=max_val - 1e-6)

        term = (clamped_output - min_val) / (max_val - min_val) * 2 - 1
        input = torch.atanh(term)
        return input
    
    def _init_bounded_tensor(self, min_val, max_val):
        """
        初始化一个带有指定范围限制的张量，并返回这个张量。
        """
        tensor = torch.randn(1, device=self.device)  # Use self.device here
        return min_val + (max_val - min_val) * 0.5 * (torch.tanh(tensor) + 1)