from src.distributions.seismic_mixture_factory import SeismicMixtureFactory
from torch import Tensor
import torch
K = 3
weight_logits = torch.randn(K)                  # 混合 logits
alpha = torch.tensor([2.0, 3.0, 5.0])          # b 的 shape (concentration)
beta  = torch.tensor([0.8, 1.2, 2.0])          # b 的 rate
M0    = 3.0

factory = SeismicMixtureFactory(weight_logits, alpha, beta, M0)
b_mix, m_mix = factory.build()

# 现在就可以：
x = torch.tensor([1.0, 2.0, 4.0])
print("log S_b at 1.0:", b_mix.log_survival(torch.tensor(1.0)))
print("log p_m at m=5.0:", m_mix.log_prob(torch.tensor(5.0)))
samples_m = m_mix.rsample((1000,))             # 震级样本（起点在 M0）
