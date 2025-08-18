import torch
import math
from src.distributions.gutenberg_richter import GutenbergRichter

# 假设我们定义了 GutenbergRichter 类
b = torch.tensor([0.8])  # 假设 b = 0.8
mag_min = 2.0
mag_max = 10.0

# 创建一个 GutenbergRichter 分布实例
dist = GutenbergRichter(b, mag_min=mag_min, mag_max=mag_max)

# 生成一些样本数据（模拟的震级）
sample_mags = torch.tensor([2.5, 3.0, 5.0, 7.0, 9.0, 10.0])  # 测试的震级值

# 创建一个 mask，假设我们只关心前 4 个有效的震级值
mask = torch.tensor([True, True, True, True, False, False])  # 只考虑前四个有效数据

# 测试 log_prob
log_prob_values = dist.log_prob(sample_mags, mask=mask)
print("log_prob values:", log_prob_values)

# 测试 log_likelihood
log_likelihood_value = dist.log_likelihood(sample_mags, mask=mask)
print("log_likelihood value:", log_likelihood_value)

# 测试 AIC 和 BIC
aic_value = dist.aic(sample_mags, mask=mask, k=1)
bic_value = dist.bic(sample_mags, mask=mask, k=1)
print("AIC:", aic_value)
print("BIC:", bic_value)

# 测试 KS 测试（计算 D 统计量和 p 值）
D, p_value = dist.ks_stat_pvalue(sample_mags, mask=mask)
print("KS Statistic D:", D)
print("KS p-value:", p_value)

# 测试样本生成（rsample）
generated_samples = dist.rsample(sample_shape=torch.Size([10]))  # 生成 10 个样本
print("Generated samples:", generated_samples)
