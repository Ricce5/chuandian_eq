import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.output_range = output_range if output_range is not None else (0.5, 2)  # 默认范围 [0.5, 2]

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

    def forward(self, x, delta, return_last_state=False):
        """
        x: 输入张量，形状为 (batch, length, d_model)
        delta: 时间步长张量，形状为 (batch, length, d_model)
        """
        # 将输入张量从 (B, L, D) 转换为 (B, D, L)，以符合 selective_scan_fn 的要求
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')
        if return_last_state:
            y_out, ssm_state = selective_scan_fn(
                u=self._bounded_tanh(u),
                delta=delta_rearranged,
                A=self.A,
                B=self._bounded_tanh(self.B),  # 应用范围约束的 B
                C=self._bounded_tanh(self.C),  # 应用范围约束的 C
                D=self.D,
                delta_bias=self.delta_bias,
                return_last_state=return_last_state
            )
        else:
            y_out = selective_scan_fn(
                u=self._bounded_tanh(u),
                delta=delta_rearranged,
                A=self.A,
                B=self._bounded_tanh(self.B),  # 应用范围约束的 B
                C=self._bounded_tanh(self.C),  # 应用范围约束的 C
                D=self.D,
                delta_bias=self.delta_bias,
                return_last_state=return_last_state
            )

        # 将输出从 (B, D, L) 转换回 (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    
        # 对输出进行范围约束
        y_out = self._bounded_tanh(y_out, *self.output_range)
        if return_last_state:
            return y_out, ssm_state
        else:
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

# --- 使用示例 ---
B_batch, L_len, D_model, D_state = 10, 1000, 4, 16  # 示例参数
B_range = (0.2, 1.0)  # 设定 B 参数的范围 [0.2, 1.0]
C_range = (-2.0, 2.0)  # 设定 C 参数的范围 [-2.0, 2.0]
ssm_wrapper = BoundedSelectiveScanWrapper(d_model=D_model, d_state=D_state, device=device, B_range=B_range, C_range=C_range).to(device)
x_input = torch.randn(B_batch, L_len, D_model, device=device) * 100  # 随机输入
delta_input = torch.abs(torch.randn(B_batch, L_len, D_model, device=device))  # 保证 delta_input 为正
output,ssm_state = ssm_wrapper(x_input, delta_input,return_last_state=True)

print(f"ssm_state {ssm_state.shape}")

# 打印输入和输出的形状，确认其一致性
print(f"输入形状: {x_input.shape}")
print(f"输出形状: {output.shape}")
# print(f"输入数据: {x_input}")
# print(f"输出数据: {output}")

# 生成保存路径和文件夹
output_dir = "/root/autodl-tmp/chuandian_eq/tests"
os.makedirs(output_dir, exist_ok=True)

# 假设 x_input 和 output 是模型的输入和输出
# x_input.shape = (B_batch, L_len, D_model)
# output.shape = (B_batch, L_len, D_model)

# 这里我们假设是单个 batch (B_batch = 1)
x_input_single = x_input[0].detach().cpu().numpy()  # 转换为 numpy 格式，以便绘图
output_single = output[0].detach().cpu().numpy()

# 生成时间步索引
time_steps = range(L_len)

# 绘制输入与输出的图像
plt.figure(figsize=(10, 6))

# 绘制输入图
plt.subplot(2, 1, 1)  # 2 行 1 列，第 1 个子图
plt.plot(time_steps, x_input_single, label="Input", color="blue")
plt.title("Input over Time")
plt.xlabel("Time Step")
plt.ylabel("Input Value")
plt.grid(True)
plt.legend()

# 绘制输出图
plt.subplot(2, 1, 2)  # 2 行 1 列，第 2 个子图
plt.plot(time_steps, output_single, label="Output", color="red")
plt.title("Output over Time")
plt.xlabel("Time Step")
plt.ylabel("Output Value")
plt.grid(True)
plt.legend()

# 保存图像
plt.savefig(os.path.join(output_dir, "input_output_plot.png"))

# 显示图像
plt.tight_layout()
plt.show()
