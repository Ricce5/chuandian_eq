import torch
from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily

from .distribution import Distribution


class MixtureSameFamily(TorchMixtureSameFamily, Distribution):  # 混合分布的概率密度函数等于各成分的概率密度函数的加权和
    def __init__(
        self, mixture_distribution, component_distribution, validate_args=False
    ):
        super(MixtureSameFamily, self).__init__(
            mixture_distribution=mixture_distribution,  # weights for the mixture components
            component_distribution=component_distribution,   # distribution for each component
            validate_args=False,
        )

    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x) - self.log_survival(x)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)  # 每个分量的生存函数和混合权重相乘后求和

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape) # 注意是形状的长度不是长度
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len 
            es = self.event_shape

            # Since we know that the sample x > lower_bound, we have to adjust the
            # mixing probabilities as p(z_i = k) * Pr(x >= lower_bound | z_i = k)
            # mixture samples [n, B]
            conditional_mix_probs = (
                self.mixture_distribution.probs
                * self.component_distribution.log_survival(lower_bound).exp() # 隐变量满足条件的后验概率
            )
            mix_sample = Categorical(probs=conditional_mix_probs).sample(sample_shape)
            mix_shape = mix_sample.shape  # [n, B]

            # component samples [n, B, k, E] n:采样的数量 B:不同参数分布的个数 k:成分的个数 E:每个成分的事件形状
            comp_samples = self.component_distribution.sample_conditional(
                lower_bound, sample_shape
            )

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))  # 使得 mix_sample_r 的形状为 [n, B, 1, 1, ..., 1] (es 的长度+1个 1)
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es # 扩展event shape对应的维度，使得mix_sample_r的形状[n,B,1,E]
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r) # torch.gather 在第 gather_dim 维度上按照 mix_sample_r 中的索引提取出我们想要的 component 的样本
            return samples.squeeze(gather_dim)
