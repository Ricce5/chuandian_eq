import numpy as np

def log_lhood_comp(theta, rate):
    """
    Python equivalent of the MATLAB function log_lhood_comp.
    
    Parameters
    ----------
    theta : list or np.ndarray
        Contains the parameters [af, b].
    rate : dict
        Contains various parameters and data required for likelihood calculation. Keys include:
        'm_0', 't_b_s', 'dot_V_bs', 't_sbs', 'tot_V', 'data_magn', 'm_0_m', etc.
    
    Returns
    -------
    log_lhood_comp : float
        The computed negative log-likelihood.
    """

    af = theta[0]
    b = theta[1]
    rate['N'] = len(rate['data_magn'])
    # First term in the likelihood
    A1 = rate['N'] * (af - b * rate['m_0']) / np.log10(np.exp(1))

    # Interpolation of dot_V_bs at the time points t_sbs
    dotV_bs_ts = np.interp(rate['t_sbs'], rate['t_b_s'], rate['dot_V_bs'])
    
    # Second term in the likelihood
    K2 = np.sum(np.log(dotV_bs_ts))
    
    # If K2 is large, add small value to avoid log(0)
    if K2 < 1.e30:
        dotV_bs_ts += 1e-10
        K2 = np.sum(np.log(dotV_bs_ts))

    # Third term (currently unused)
    K3 = 0  # (rate['N'] - rate['N_sbs']) * np.log(rate['dot_V_shut_in'])

    # Fourth term (currently unused)
    A4 = 0  # -1/tau * (np.sum(rate['t_sas'] - rate['T_s']))

    # Fifth term in the likelihood
    A5 = - 10**(af - b * rate['m_0']) * rate['tot_V']
    
    from .b_ll import log_likelihood_b
    b_ll = log_likelihood_b(b,rate['data_magn'], rate['m_0'],10)
    log_lhood_comp = -(A1 + K2 + K3 + A4 + A5 + b_ll)/10000

    return log_lhood_comp
