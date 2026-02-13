import torch
from torch.distributions import constraints
import src
from src.distributions.gamma import Gamma

# --- Test case where alpha, beta, and x contain the same shape with 0 as PAD ---
PAD = 0  # Assume 0 is our PAD value

# Create alpha, beta, and x tensors, which include 0 as PAD
alpha = torch.tensor([[1.5, 0.0, 2.0], [0.5, 3.0, 0.0], [2.5, 4.0, 1.0]])
beta = torch.tensor([[0.5, 0.0, 1.5], [1.2, 1.3, 0.0], [0.8, 0.9, 1.0]])
x = torch.tensor([[2.0, 0.0, 3.0], [1.0, 4.0, 0.0], [5.0, 6.0, 0.0]])

# Create a mask: 0 is PAD, other values are valid
mask = (x != PAD)

# Create an instance of the Gamma distribution
mine = Gamma(alpha, beta)

# Compute log_prob
log_prob_masked = mine.log_prob(x, mask)
print("log_prob with mask:")
print(log_prob_masked)

# ---
N = 1000  # Number of samples to draw
samples = mine.rsample((N,))  # Draw samples from the Gamma distribution
print("Sampled values (first 5):")
print(samples[:5])

# --- Check the mean and variance of the samples with the mask applied ---
# To compute the mean and variance, we only consider valid positions (where mask=True)
masked_samples = torch.where(mask.unsqueeze(0), samples, torch.zeros_like(samples))
masked_mean = masked_samples.mean((0, 1))  # Compute mean over batch and sample dimensions
masked_var = masked_samples.var((0, 1), unbiased=False)  # Compute variance over batch and sample dimensions

print(f"Masked Mean (excluding PAD): {masked_mean}")
print(f"Masked Variance (excluding PAD): {masked_var}")
