import torch
import math
from src.distributions.gutenberg_richter import GutenbergRichter
#%%
# 假设每个事件有不同的 b 值
b_values = torch.tensor([0.6])  # 这里是每个事件不同的 b 值
dist = GutenbergRichter(b_values, mag_min=2.0, mag_max=10.0)

# 假设有 5 个事件的震级
sample_mags = torch.tensor([2.5, 3.0, 5.0, 7.0, 9.0])

# 假设只关心前 3 个有效事件
mask = torch.tensor([True, True, True, False, False])

# 计算每个事件的 log_prob
log_prob_values = dist.log_prob(sample_mags, mask=mask)
print("log_prob values:", log_prob_values)
#%%
# 计算 log_likelihood
log_likelihood_value = dist.log_likelihood(sample_mags, mask=mask)
print("log_likelihood value:", log_likelihood_value)

# 计算 AIC 和 BIC
aic_value = dist.aic(sample_mags, mask=mask, k=1)
bic_value = dist.bic(sample_mags, mask=mask, k=1)
print("AIC:", aic_value)
print("BIC:", bic_value)

# 计算 KS 测试（D 统计量和 p 值）
D, p_value = dist.ks_stat_pvalue(sample_mags, mask=mask)
print("KS Statistic D:", D)
print("KS p-value:", p_value)

# 生成样本
generated_samples = dist.rsample(sample_shape=torch.Size([10]))  # 生成 10 个样本
print("Generated samples:", generated_samples)
