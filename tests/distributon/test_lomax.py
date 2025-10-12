import src
from src.distributions.lomax import Lomax
import torch
import math

# If the Lomax class is in the same file, uncomment the following line and paste the class definition
# from your_module import Lomax    # ← Import like this if encapsulated in a module

torch.manual_seed(0)

# ==== Parameter Settings ====
lam = torch.tensor(2.0)   # λ
alpha = torch.tensor(2.5) # α > 1 for finite mean
lomax = Lomax(scale=lam, shape=alpha)

# ==== 1) Basic Test: Sample Mean vs Theoretical Mean ====
# Theoretical mean: E[X] = λ/(α-1)  (α>1)
theory_mean = lam / (alpha - 1.0)

N = 200_000
x = lomax.rsample((N,))   # [N] samples
emp_mean = x.mean()

print("[Lomax] theory mean:", theory_mean.item())
print("[Lomax] sample mean:", emp_mean.item())

# ==== 2) Survival Function S(x) Comparison (Point Estimation) ====
def S_theory(x_tensor):
    return (1.0 + x_tensor / lam).pow(-alpha)

grid = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
# Empirical estimate S(x) = P(X>=x) ≈ 1/N * sum(1_{X>=x})
S_emp = torch.tensor([(x >= t).float().mean().item() for t in grid])
S_th = S_theory(grid)

print("\n[Lomax] Survival comparison at grid points")
for t, se, st in zip(grid.tolist(), S_emp.tolist(), S_th.tolist()):
    print(f"  x={t:>4}:  empirical S≈{se:.4f}  theory S={st:.4f}")

# ==== 3) Shifted Property: m = x + M0 ====
M0 = torch.tensor(3.0)
m = x + M0

# Theoretical CDF for m: shifted Lomax starting at M0:
#   F_M(m) = 1 - (1 + (m - M0)/λ)^(-α),  m >= M0
def Fm_theory(m_tensor):
    z = torch.clamp_min(m_tensor - M0, 0.0)
    return 1.0 - (1.0 + z / lam).pow(-alpha)

# Compare empirical CDF and theoretical CDF at several points
m_grid = torch.tensor([3.0, 3.5, 4.0, 5.0, 7.0, 11.0])
F_emp = torch.tensor([(m <= t).float().mean().item() for t in m_grid])
F_th = Fm_theory(m_grid)

print("\n[Shifted Lomax] CDF comparison at grid points (m=x+M0)")
for t, fe, ft in zip(m_grid.tolist(), F_emp.tolist(), F_th.tolist()):
    print(f"  m={t:>4}:  empirical F≈{fe:.4f}  theory F={ft:.4f}")

# ==== 4) Simple KS Statistic (Non-strict p-value, for reference only) ====
# Compute KS distance sup |F_emp(m) - F_th(m)| on a dense grid
# Note: Strict KS requires sorting and aligning with empirical CDF, this is an approximate version
m_grid_dense = torch.linspace(M0.item(), M0.item()+20.0, 200)
F_emp_dense = torch.tensor([(m <= t).float().mean().item() for t in m_grid_dense])
F_th_dense = Fm_theory(m_grid_dense)
ks_stat = (F_emp_dense - F_th_dense).abs().max().item()
print(f"\n[Shifted Lomax] approx. KS distance on dense grid: {ks_stat:.4f}")
