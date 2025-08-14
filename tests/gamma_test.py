import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
import src
from  src.distributions.gamma import Gamma
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
