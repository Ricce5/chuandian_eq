import src
from  src.distributions.lomax import Lomax
import torch
import math

# 如果 Lomax 类在同一文件，取消下面注释并粘贴类定义
# from your_module import Lomax    # ← 若你封装在模块里就这样引入

torch.manual_seed(0)

# ==== 参数设定 ====
lam = torch.tensor(2.0)   # λ
alpha = torch.tensor(2.5) # α > 1 才有有限均值
lomax = Lomax(scale=lam, shape=alpha)

# ==== 1) 基本检验：样本均值 vs 理论均值 ====
# 理论均值：E[X] = λ/(α-1)  (α>1)
theory_mean = lam / (alpha - 1.0)

N = 200_000
x = lomax.rsample((N,))   # [N] 样本
emp_mean = x.mean()

print("[Lomax] theory mean:", theory_mean.item())
print("[Lomax] sample mean:", emp_mean.item())

# ==== 2) 生存函数 S(x) 与模拟对比（点估计） ====
def S_theory(x_tensor):
    return (1.0 + x_tensor / lam).pow(-alpha)

grid = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
# 模拟估计 S(x) = P(X>=x) ≈ 1/N * sum(1_{X>=x})
S_emp = torch.tensor([(x >= t).float().mean().item() for t in grid])
S_th = S_theory(grid)

print("\n[Lomax] Survival comparison at grid points")
for t, se, st in zip(grid.tolist(), S_emp.tolist(), S_th.tolist()):
    print(f"  x={t:>4}:  empirical S≈{se:.4f}  theory S={st:.4f}")

# ==== 3) 平移性质：m = x + M0 ====
M0 = torch.tensor(3.0)
m = x + M0

# m 的理论：起点在 M0 的 Lomax：
#   F_M(m) = 1 - (1 + (m - M0)/λ)^(-α),  m >= M0
def Fm_theory(m_tensor):
    z = torch.clamp_min(m_tensor - M0, 0.0)
    return 1.0 - (1.0 + z / lam).pow(-alpha)

# 在若干点比较经验 CDF 和理论 CDF
m_grid = torch.tensor([3.0, 3.5, 4.0, 5.0, 7.0, 11.0])
F_emp = torch.tensor([(m <= t).float().mean().item() for t in m_grid])
F_th = Fm_theory(m_grid)

print("\n[Shifted Lomax] CDF comparison at grid points (m=x+M0)")
for t, fe, ft in zip(m_grid.tolist(), F_emp.tolist(), F_th.tolist()):
    print(f"  m={t:>4}:  empirical F≈{fe:.4f}  theory F={ft:.4f}")

# ==== 4) 简单 KS 统计量（非严格 p 值，仅供参考） ====
# 计算 KS 距离 sup |F_emp(m) - F_th(m)| 在若干网格上
# 注意：严格 KS 需要排序并按经验 CDF 对齐，这里给近似版本
m_grid_dense = torch.linspace(M0.item(), M0.item()+20.0, 200)
F_emp_dense = torch.tensor([(m <= t).float().mean().item() for t in m_grid_dense])
F_th_dense = Fm_theory(m_grid_dense)
ks_stat = (F_emp_dense - F_th_dense).abs().max().item()
print(f"\n[Shifted Lomax] approx. KS distance on dense grid: {ks_stat:.4f}")
