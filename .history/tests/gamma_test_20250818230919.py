import torch
from torch.distributions import constraints
import src
from src.distributions.gamma import Gamma

# --- 1) log_prob vs 官方实现 ---
torch.manual_seed(0)
alpha = torch.rand(5, 3) * 3 + 0.2
beta  = torch.rand(5, 3) * 2 + 0.2
x     = torch.rand(5, 3) * 5 + 1e-6

mine = Gamma(alpha, beta)
ref  = torch.distributions.Gamma(alpha, beta)
print("max |Δ log_prob| =", (mine.log_prob(x) - ref.log_prob(x)).abs().max().item())

# --- 2) 采样均值/方差 ---
alpha_s = torch.tensor([0.5, 1.0, 2.5, 7.0])
beta_s  = torch.tensor([0.7, 1.2, 0.5, 2.0])
dist_s  = Gamma(alpha_s, beta_s)
N = 200_000
samples = dist_s.rsample((N,))
emp_mean = samples.mean(0)
emp_var  = samples.var(0, unbiased=False)
print("empirical mean   =", emp_mean)
print("theoretical mean =", alpha_s / beta_s)
print("empirical var    =", emp_var)
print("theoretical var  =", alpha_s / (beta_s**2))

# --- 3) gradcheck（需要 double） ---
dtype = torch.double
alpha_g = torch.rand(2, 2, dtype=dtype, requires_grad=True) + 0.5
beta_g  = torch.rand(2, 2, dtype=dtype, requires_grad=True) + 0.5
x_g     = torch.rand(2, 2, dtype=dtype, requires_grad=True) + 0.2

mine_g = Gamma(alpha_g, beta_g)

def f_x(xv):    return mine_g.log_prob(xv).sum()
def f_alpha(a): return Gamma(a, beta_g).log_prob(x_g).sum()
def f_beta(b):  return Gamma(alpha_g, b).log_prob(x_g).sum()

print("gradcheck x    =", torch.autograd.gradcheck(f_x, (x_g,), eps=1e-6, atol=1e-4, rtol=1e-3))
print("gradcheck α    =", torch.autograd.gradcheck(f_alpha, (alpha_g,), eps=1e-6, atol=1e-4, rtol=1e-3))
print("gradcheck β    =", torch.autograd.gradcheck(f_beta, (beta_g,), eps=1e-6, atol=1e-4, rtol=1e-3))

# --- 4) rsample 可回传 ---
alpha_r = torch.tensor([1.1, 2.2], requires_grad=True)
beta_r  = torch.tensor([0.7, 1.3], requires_grad=True)
dist_r  = Gamma(alpha_r, beta_r)
z = dist_r.rsample((10_000,))
loss = z.mean()
loss.backward()
print("alpha.grad:", alpha_r.grad)
print("beta.grad :", beta_r.grad)

# --- 5) 测试带mask的 log_prob ---
PAD = 0  # 假设 0 是我们的 PAD 值
alpha_mask = torch.rand(5, 3) * 3 + 0.2
beta_mask  = torch.rand(5, 3) * 2 + 0.2
x_mask     = torch.rand(5, 3) * 5 + 1e-6
mask = (x_mask != PAD)

mine_mask = Gamma(alpha_mask, beta_mask)
log_prob_mask = mine_mask.log_prob(x_mask, mask)
print("max |Δ log_prob with mask| =", log_prob_mask.abs().max().item())
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
