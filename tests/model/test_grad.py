import torch
# %%
# 创建一个batch_size=3的输入张量，设置 requires_grad=True 以计算梯度
a = torch.tensor([[2.0], [1.0], [3.0]], requires_grad=True)  # batch_size=3
b = torch.tensor([[3.0], [4.0], [2.0]], requires_grad=True)  # batch_size=3

# 定义多个输出（这里我们定义两个输出）
z1 = a**2 + b  # 输出1
z2 = a * b + b**2  # 输出2

# 创建一个张量列表表示多个输出
output = torch.stack([z1, z2], dim=-1)  # (batch_size, 2) -> 2个输出，batch_size=3

# 计算多个输出相对于输入的梯度
output.backward(torch.ones_like(output))  # 这里对每个输出都使用相同的权重（这里是1）

# 打印a和b的梯度
print(f"a's gradient: {a.grad}")
print(f"b's gradient: {b.grad}")
# %%
import torch
import torch.nn as nn

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 设置输入大小和输出大小
input_size = 3  # 输入是3维向量
output_size = 2  # 输出是2维向量

# 创建模型实例
model = SimpleModel(input_size, output_size)

# 创建输入张量，要求计算梯度
x = torch.randn(input_size, requires_grad=True)

# 通过模型前向传播计算输出
y = model(x)

# 计算雅可比矩阵：y对x的梯度
jacobian = []
for i in range(output_size):
    grad_output = torch.zeros_like(y)
    grad_output[i] = 1
    jacobian_row = torch.autograd.grad(y, x, grad_outputs=grad_output, create_graph=True)[0]
    jacobian.append(jacobian_row)

jacobian_matrix = torch.stack(jacobian).T
print("Jacobian Matrix:\n", jacobian_matrix)
# %%
import torch
import torch.nn as nn

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 设置输入大小、输出大小和批量大小
input_size = 3  # 输入是3维向量
output_size = 2  # 输出是2维向量
batch_size = 4   # 假设我们有4个样本

# 创建模型实例
model = SimpleModel(input_size, output_size)

# 创建批量输入张量，要求计算梯度
x = torch.randn(batch_size, input_size, requires_grad=True)

# 通过模型前向传播计算输出
y = model(x)

# 计算雅可比矩阵：y对x的梯度
# 这里我们将计算每个样本的雅可比矩阵
jacobian = []
for i in range(output_size):
    grad_output = torch.zeros_like(y)
    grad_output[:, i] = 1  # 只保留第i个输出
    jacobian_batch = torch.autograd.grad(y, x, grad_outputs=grad_output, create_graph=True)[0]
    jacobian.append(jacobian_batch)

# 将所有样本的雅可比矩阵堆叠起来
jacobian_matrix = torch.stack(jacobian, dim=2)  # 变成 (batch_size, output_size, input_size)
print("Jacobian Matrix Shape: ", jacobian_matrix.shape)
print("Jacobian Matrix:\n", jacobian_matrix)

# %%
import torch
import torch.nn as nn

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 设置输入大小、输出大小和批量大小
input_size = 3  # 输入是3维向量
output_size = 2  # 输出是2维向量
batch_size = 4   # 假设我们有4个样本

# 创建模型实例
model = SimpleModel(input_size, output_size)

# 创建批量输入张量，要求计算梯度
x = torch.randn(batch_size, input_size, requires_grad=True)

# 通过模型前向传播计算输出
y = model(x)

# 计算雅可比矩阵：y对x的梯度
# grad_output将包含每个输出的梯度，shape为(batch_size, output_size)
grad_output = torch.eye(output_size).expand(batch_size, -1, -1)

# 并行计算雅可比矩阵
jacobian_matrix = torch.autograd.grad(y, x, grad_outputs=grad_output, create_graph=True)[0]

# jacobian_matrix的shape为(batch_size, output_size, input_size)
print("Jacobian Matrix Shape: ", jacobian_matrix.shape)
print("Jacobian Matrix:\n", jacobian_matrix)

# %%

import torch
import torch.nn as nn

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 设置输入大小、输出大小和批量大小
input_size = 3  # 输入是3维向量
output_size = 2  # 输出是2维向量
batch_size = 4   # 假设我们有4个样本

# 创建模型实例
model = SimpleModel(input_size, output_size)

# 创建批量输入张量，要求计算梯度
x = torch.randn(batch_size, input_size, requires_grad=True)

# 通过模型前向传播计算输出
y = model(x)

# 计算雅可比矩阵：y对x的梯度
# grad_output将包含每个输出的梯度，shape为(batch_size, output_size)
jacobian = []

for i in range(output_size):
    grad_output = torch.zeros_like(y)
    grad_output[:, i] = 1  # 对每个输出维度设置梯度为1
    
    # 计算每个输出对输入的梯度
    jacobian_row = torch.autograd.grad(y, x, grad_outputs=grad_output, create_graph=True)[0]
    
    # 将每个输出的雅可比矩阵添加到jacobian列表
    jacobian.append(jacobian_row)

# 将所有输出的雅可比矩阵堆叠起来，得到最终的雅可比矩阵
jacobian_matrix = torch.stack(jacobian, dim=2)  # shape will be (batch_size, output_size, input_size)

# 输出结果
print("Jacobian Matrix Shape: ", jacobian_matrix.shape)
print("Jacobian Matrix:\n", jacobian_matrix)

# %%
import torch
import torch.nn as nn
import torch.autograd as autograd

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, h_n = self.rnn(x)  # out: (batch_size, seq_len, hidden_size), h_n: (1, batch_size, hidden_size)
        out = self.fc(out)       # (batch_size, seq_len, output_size)
        return out, h_n

# 参数初始化
input_size = 1
hidden_size = 64
output_size = 1
model = RNNModel(input_size, hidden_size, output_size)

# 假设输入是一个序列
x = torch.randn(1, 5, input_size, requires_grad=True)  # (batch_size, seq_len, input_size)

# 计算输出
y_pred, _ = model(x)

# 选择一个时间步（例如第1个时间步）
output_t = y_pred[:, 0, :]  # 选择第1个时间步的输出

# 对输入 x 的梯度进行反向传播
output_t.backward(torch.ones_like(output_t))  # 计算输出对输入的导数

# 获取 x 对第一个时间步输出的导数
x_grad = x.grad  # (batch_size, seq_len, input_size)

print("x_grad:", x_grad)
