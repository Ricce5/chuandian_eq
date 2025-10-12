import torch
import matplotlib.pyplot as plt
from torch.distributions import constraints
from src.distributions.gutenberg_richter import GutenbergRichter

b = torch.tensor(1.0)  
gr_dist = GutenbergRichter(b, mag_min=2.0, mag_max=10.0)


samples = gr_dist.rsample((100000,)) 


plt.hist(samples.numpy(), bins=50, density=True, alpha=0.6, color='blue')
plt.xlabel("Magnitude (M)")
plt.ylabel("Probability Density")
plt.title("Gutenberg–Richter Magnitude Distribution (Using Class)")
plt.savefig("gr_distribution.png", dpi=150)
plt.show()


x = torch.linspace(1, 7, steps=7) 
print("x =", x)
print("log_prob(x) =", gr_dist.log_prob(x))

xs = torch.linspace(2.0, 6.0, steps=1000)
pdf = gr_dist.log_prob(xs).exp()
dx = xs[1] - xs[0]
integral = torch.sum(pdf * dx)
print("Integral result (should be close to 1) =", integral.item())