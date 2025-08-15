import torch
import matplotlib.pyplot as plt
from torch.distributions import constraints

# 你的 GutenbergRichter 类
class Distribution:
    def __init__(self, batch_shape=torch.Size(), validate_args=False):
        self.batch_shape = batch_shape

class GutenbergRichter(Distribution):
    arg_constraints = {"b": constraints.positive}

    def __init__(self, b, mag_min=2.0, mag_max=10):
        self.b = b
        self.mag_min = mag_min
        self.mag_max = mag_max
        batch_shape = b.shape
        super().__init__(batch_shape, validate_args=False)

    def log_hazard(self, x):
        return torch.zeros_like(x)

    def log_survival(self, x):
        return torch.zeros_like(x)

    def log_prob(self, x):
        return torch.zeros_like(x)  # 占位符

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(shape, device=self.b.device, dtype=self.b.dtype).uniform_()
        return self.b.reciprocal().neg() * torch.log10(
            -u * (10 ** (-self.b * self.mag_min) - 10 ** (-self.b * self.mag_max)) 
            + 10 ** (-self.b * self.mag_min)
        )

# 创建分布对象
b = torch.tensor(1.0)  # 典型 b 值
gr_dist = GutenbergRichter(b, mag_min=2.0, mag_max=10.0)

# 采样
samples = gr_dist.rsample((100000,))  # 10 万个样本

# 绘图
plt.hist(samples.numpy(), bins=50, density=True, alpha=0.6, color='blue')
plt.xlabel("Magnitude (M)")
plt.ylabel("Probability Density")
plt.title("Gutenberg–Richter Magnitude Distribution (Using Class)")
plt.savefig("gr_distribution.png", dpi=150)
plt.show()
