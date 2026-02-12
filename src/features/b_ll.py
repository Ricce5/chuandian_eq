import numpy as np
from scipy.optimize import minimize_scalar


def log_likelihood_b(b, magnitudes, Mc, Mmax):
    """
    log-likelihood function for truncated Gutenberg–Richter distribution.
    
    Formula:
    L = N ln(b) + N ln(ln 10) 
        - b ln(10) * sum(m_n) 
        - N ln(10^(-b Mc) - 10^(-b Mmax))
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    if b <= 0 or magnitudes.size == 0 or Mmax <= Mc:
        return -np.inf

    N = magnitudes.size
    sum_m = np.sum(magnitudes)
    norm = np.power(10.0, -b * Mc) - np.power(10.0, -b * Mmax)
    if norm <= 0 or not np.isfinite(norm):
        return -np.inf

    term1 = N * np.log(b)
    term2 = N * np.log(np.log(10))
    term3 = - b * np.log(10) * sum_m
    term4 = - N * np.log(norm)

    return term1 + term2 + term3 + term4


def estimate_b_value(magnitudes, Mc, Mmax):
    """
    Estimate b-value by maximizing log-likelihood.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.size == 0:
        return np.nan, np.nan

    # maximize log-likelihood == minimize negative log-likelihood
    def neg_LL(b):
        return -log_likelihood_b(b, magnitudes, Mc, Mmax)

    result = minimize_scalar(neg_LL, bounds=(0.01, 10), method='bounded')
    if not result.success or not np.isfinite(result.x):
        return np.nan, np.nan
    return result.x, -result.fun


