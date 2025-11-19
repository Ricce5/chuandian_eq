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
    if b <= 0:
        return -np.inf

    N = len(magnitudes)
    sum_m = np.sum(magnitudes)

    term1 = N * np.log(b)
    term2 = N * np.log(np.log(10))
    term3 = - b * np.log(10) * sum_m
    term4 = - N * np.log(10**(-b * Mc) - 10**(-b * Mmax))

    return term1 + term2 + term3 + term4


def estimate_b_value(magnitudes, Mc, Mmax):
    """
    Estimate b-value by maximizing log-likelihood.
    """

    # maximize log-likelihood == minimize negative log-likelihood
    def neg_LL(b):
        return -log_likelihood_b(b, magnitudes, Mc, Mmax)

    result = minimize_scalar(neg_LL, bounds=(0.01, 10), method='bounded')
    return result.x, -result.fun



