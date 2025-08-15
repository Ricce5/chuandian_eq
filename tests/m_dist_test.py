from src.distributions.seismic_mixture_factory import SeismicMixtureFactory
from src.distributions.gutenberg_richter import GutenbergRichter
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from src.utils.utils import set_seed    
set_seed(42)
K = 1
# weight_logits = torch.randn(K) 
weight_logits = torch.tensor([1.0])  # 混合 logits
print(weight_logits.shape)
alpha = torch.tensor([0.5])          # b 的 shape (concentration)
beta  = torch.tensor([0.5])          # b 的 rate
M0    = 3.0

factory = SeismicMixtureFactory(weight_logits, alpha, beta, M0)
b_mix, m_mix = factory.build()
m_gr = GutenbergRichter(b=torch.tensor(1.0), mag_min=M0)
mix_samples = m_mix.sample((100000,))  # 10 万个样本
print(mix_samples)
gr_samples = m_gr.rsample((100000,))  # 10 万个样本
plt.hist(mix_samples.numpy(), bins=50, density=True, alpha=0.6, color='blue')
plt.xlabel("Magnitude (M)")
plt.ylabel("Probability Density")
plt.title("Gutenberg–Richter Magnitude Distribution (Using Class)")
plt.savefig("mix_gr_distribution.png", dpi=150)
plt.show()
