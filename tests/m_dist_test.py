from src.distributions.seismic_mixture_factory import SeismicMixtureFactory
from src.distributions.gutenberg_richter import GutenbergRichter
from torch import Tensor
import torch
from src.utils.utils import set_seed    
set_seed(42)
K = 1
weight_logits = torch.randn(K)                  # 混合 logits
alpha = torch.tensor([2000.0])          # b 的 shape (concentration)
beta  = torch.tensor([2000.0])          # b 的 rate
M0    = 3.0

factory = SeismicMixtureFactory(weight_logits, alpha, beta, M0)
b_mix, m_mix = factory.build()
m_gr = GutenbergRichter(b=torch.tensor(1.0), mag_min=M0)


# 现在就可以：
x = torch.tensor([1.0, 2.0, 4.0])
print("b sample:", b_mix.sample())
print("m sample:", m_mix.sample())
print("b log_prob:", b_mix.log_prob(x))
