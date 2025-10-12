import torch
import math
from src.distributions.gutenberg_richter import GutenbergRichter
#%%
b_values = torch.tensor([0.6,0.8,1.0,1.2,1.2])  # Example b values for different batches
dist = GutenbergRichter(b_values, mag_min=2.0, mag_max=10.0)

sample_mags = torch.tensor([2.5, 3.0, 5.0, 7.0, 9.0])

mask = torch.tensor([True, True, True, False, False])

log_prob_values = dist.log_prob(sample_mags, mask=mask)
print("log_prob values:", log_prob_values)
#%%
# Compute log_likelihood
log_likelihood_value = dist.log_likelihood(sample_mags, mask=mask)
print("log_likelihood value:", log_likelihood_value)

# Compute AIC and BIC
aic_value = dist.aic(sample_mags, mask=mask, k=1)
bic_value = dist.bic(sample_mags, mask=mask, k=1)
print("AIC:", aic_value)
print("BIC:", bic_value)

# Compute KS test (D statistic and p-value)
D, p_value = dist.ks_stat_pvalue(sample_mags, mask=mask)
print("KS Statistic D:", D)
print("KS p-value:", p_value)

# Generate samples
generated_samples = dist.rsample(sample_shape=torch.Size([10]))  # Generate 10 samples
print("Generated samples:", generated_samples)
