import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from src.models.mamba.scan_wrapper import BoundedDiscreteSSM

B_batch, L_len, D_model, D_state = 10, 1000, 4, 16  # 示例参数
B_range = (0.2, 1.0)  # 设定 B 参数的范围 [0.2, 1.0]
out_range = (0.5,2)

# 创建 BoundedDiscreteSSM 实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssm_wrapper = BoundedDiscreteSSM(device=device, B_range=B_range,output_range=out_range).to(device)

# 创建随机输入数据
x_input = torch.randn(B_batch, L_len, D_model, device=device) * 100  # 随机输入
delta_input = torch.abs(torch.randn(B_batch, L_len, D_model, device=device))  # 保证 delta_input 为正

# 进行前向计算，获得输出和状态
output, ssm_state = ssm_wrapper(x_input, delta_input, return_last_state=True)

# 打印输出形状，确认一致性
print(f"ssm_state shape: {ssm_state.shape}")
print(f"输入形状: {x_input.shape}")
print(f"输出形状: {output.shape}")

output_dir = "/root/autodl-tmp/chuandian_eq/tests"


# 假设 x_input 和 output 是模型的输入和输出
# 这里我们假设是单个 batch (B_batch = 1)，提取第一个 batch
x_input_single = x_input[0,:,0].detach().cpu().numpy()  # 转换为 numpy 格式，以便绘图
output_single = output[0,:,0].detach().cpu().numpy()

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
