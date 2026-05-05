# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
import torch
from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily

from .distribution import Distribution


class MixtureSameFamily(TorchMixtureSameFamily, Distribution):
    def __init__(
        self, mixture_distribution, component_distribution, validate_args=False
    ):
        super(MixtureSameFamily, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=validate_args,
        )

    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x) - self.log_survival(x)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)  # sum over mixture components

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # Since we know that the sample x > lower_bound, we have to adjust the
            # mixing probabilities as p(z_i = k) * Pr(x >= lower_bound | z_i = k)
            # mixture samples [n, B]
            mix_probs = self.mixture_distribution.probs
            log_survival = self.component_distribution.log_survival(lower_bound)
            conditional_mix_probs = mix_probs * log_survival.exp()
            eps = torch.finfo(conditional_mix_probs.dtype).eps

            conditional_mix_probs = torch.nan_to_num(
                conditional_mix_probs, nan=0.0, posinf=0.0, neginf=0.0
            )
            conditional_mix_probs = conditional_mix_probs.clamp_min(0.0)
            probs_sum = conditional_mix_probs.sum(dim=-1, keepdim=True)

            fallback_probs = torch.nan_to_num(
                mix_probs, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            fallback_sum = fallback_probs.sum(dim=-1, keepdim=True)
            num_components = fallback_probs.size(-1)
            uniform_probs = torch.full_like(
                fallback_probs, 1.0 / max(1, num_components)
            )
            fallback_probs = torch.where(
                fallback_sum > 0,
                fallback_probs / fallback_sum.clamp_min(eps),
                uniform_probs,
            )

            conditional_mix_probs = torch.where(
                probs_sum > 0,
                conditional_mix_probs / probs_sum.clamp_min(eps),
                fallback_probs,
            )
            conditional_mix_probs = torch.nan_to_num(
                conditional_mix_probs,
                nan=1.0 / max(1, num_components),
                posinf=1.0 / max(1, num_components),
                neginf=0.0,
            ).clamp_min(0.0)
            final_sum = conditional_mix_probs.sum(dim=-1, keepdim=True)
            conditional_mix_probs = torch.where(
                final_sum > 0,
                conditional_mix_probs / final_sum.clamp_min(eps),
                uniform_probs,
            )
            mix_sample = Categorical(probs=conditional_mix_probs).sample(sample_shape)
            mix_shape = mix_sample.shape  # [n, B]

            # component samples [n, B, k, E] n: number of samples, B: batch size, k: number of components, E: event shape
            comp_samples = self.component_distribution.sample_conditional(
                lower_bound, sample_shape
            )

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)
