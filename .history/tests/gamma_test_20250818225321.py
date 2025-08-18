import torch
from torch.distributions import constraints
import src
from src.distributions.gamma import Gamma

# 模拟包含 PAD 值的输入数据
PAD = -1  # 假设 -1 是我们的 PAD 值
alpha = torch.rand(5, 3) * 3 + 0.2
beta  = torch.rand(5, 3) * 2 + 0.2
x     = torch.rand(5, 3) * 5 + 1e-6

# 创建 mask：假设 PAD 值在 x 中表示为 -1
mask = (x != PAD)

# 创建 Gamma 分布
mine = Gamma(alpha, beta)
ref  = torch.distributions.Gamma(alpha, beta)

# 测试 log_prob 函数的输出
log_prob_mine = mine.log_prob(x, mask)
log_prob_ref  = ref.log_prob(x)
print("max |Δ log_prob| =", (log_prob_mine - log_prob_ref).abs().max().item())

# --- 2) 测试 rsample 方法 --- 
alpha_r = torch.tensor([1.1, 2.2], requires_grad=True)
beta_r  = torch.tensor([0.7, 1.3], requires_grad=True)
dist_r  = Gamma(alpha_r, beta_r)

# 使用 mask 模拟填充数据并进行采样
mask_r = torch.tensor([True, False])  # 模拟两个位置的掩码
z = dist_r.rsample((10_000,))

# 打印结果并验证 mask 的影响
loss = z.mean()
loss.backward()

print("alpha.grad:", alpha_r.grad)
print("beta.grad :", beta_r.grad)
