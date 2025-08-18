import torch
from torch.distributions import constraints
import src
from src.distributions.gamma import Gamma

# --- 测试 alpha, beta 和 x 中都含有相同形状的 0 PAD ---
PAD = 0  # 假设 0 是我们的 PAD 值

# 创建 alpha, beta 和 x，其中包含 0 作为 PAD
alpha = torch.tensor([[1.5, 0.0, 2.0], [0.5, 3.0, 0.0], [2.5, 4.0, 1.0]])
beta = torch.tensor([[0.5, 0.0, 1.5], [1.2, 1.3, 0.0], [0.8, 0.9, 1.0]])
x = torch.tensor([[2.0, 0.0, 3.0], [1.0, 4.0, 0.0], [5.0, 6.0, 0.0]])

# 创建 mask：0 为 PAD，其他为有效值
mask = (x != PAD)

# 创建 Gamma 分布实例
mine = Gamma(alpha, beta)

# 计算 log_prob
log_prob_masked = mine.log_prob(x, mask)
print("log_prob with mask:")
print(log_prob_masked)

# --- 采样部分 ---
N = 1000  # 采样的数量
samples = mine.rsample((N,))  # 从 Gamma 分布中采样
print("Sampled values (first 5):")
print(samples[:5])

# --- 检查带 mask 采样后的均值和方差 ---
# 为了计算均值和方差，我们只关心有效值的位置（mask=True 的地方）
masked_samples = torch.where(mask.unsqueeze(0), samples, torch.zeros_like(samples))
masked_mean = masked_samples.mean((0, 1))  # 对 batch 和 sample 维度求均值
masked_var = masked_samples.var((0, 1), unbiased=False)  # 对 batch 和 sample 维度求方差

print(f"Masked Mean (excluding PAD): {masked_mean}")
print(f"Masked Variance (excluding PAD): {masked_var}")
