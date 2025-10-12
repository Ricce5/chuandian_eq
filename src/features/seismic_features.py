# Features for Small Earthquakes Can Help Predict Large Earthquakes: A Machine Learning Perspective
# Reference: https://www.mdpi.com/2076-3417/13/11/6424
import numpy as np
import math
import pandas as pd
from datetime import datetime

def cal2jd(date):
    if isinstance(date, (pd.Timestamp, datetime)):
        year = date.year
        month = date.month
        day = date.day
        hour = date.hour
        minute = date.minute
        second = date.second + date.microsecond / 1e6
    else:
        year, month, day, hour, minute, second = date

    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JDN = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + B - 1524.5
    jd = JDN + (hour - 12) / 24 + minute / 1440 + second / 86400
    return jd

def calculate_magnitudes_and_features(Mag, Mc, dMag):
    """
    Calculate b-value and a-value for each time window, along with other features related to earthquake magnitudes.

    Parameters:
    Mag - Earthquake magnitude data for the current time window
    Mc - Minimum magnitude
    dMag - Step size for magnitude intervals

    Returns:
    b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk
    """
    if len(Mag) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    Mag_max = np.max(Mag)
    Mag_mean = np.mean(Mag)

    Mag_int = np.arange(Mc, Mag_max + dMag, dMag)
    Mag_int = np.round(Mag_int, 1)  
    num_Mag_int = np.array([len(np.where(Mag >= m)[0]) for m in Mag_int])

    log_NumM = np.log10(np.maximum(num_Mag_int, 1e-10))  


    # Calculate b_lsq (slope)
    denominator_b_lsq = (np.sum(Mag_int) * np.sum(Mag_int) - len(Mag_int) * np.sum(Mag_int * Mag_int))
    if denominator_b_lsq != 0:
        b_lsq = (len(Mag_int) * np.sum(Mag_int * log_NumM) - np.sum(Mag_int) * np.sum(log_NumM)) / denominator_b_lsq
    else:
        b_lsq = np.nan

    # Calculate a_lsq (intercept)
    if b_lsq != np.nan:
        a_lsq = np.sum(np.log10(num_Mag_int + b_lsq * Mag_int)) / len(Mag_int)
    else:
        a_lsq = np.nan

    # calculate std_gr_lsq 
    if len(Mag_int) > 1 and denominator_b_lsq != 0:
        std_gr_lsq = np.sum((log_NumM - a_lsq - b_lsq * Mag_int) ** 2) / (len(Mag_int) - 1)
    else:
        std_gr_lsq = np.nan

    # maximum likelihood estimation for b_mlk and a_mlk
    if Mag_mean > Mc:
        b_mlk = np.log10(np.exp(1)) / (Mag_mean - Mc)
        a_mlk = np.log10(len(Mag)) + b_mlk * Mc
    else:
        b_mlk = np.nan
        a_mlk = np.nan

    # std_gr_mlk
    if len(Mag_int) > 1 and b_mlk != np.nan:
        std_gr_mlk = np.sum((log_NumM - a_mlk - b_mlk * Mag_int) ** 2) / (len(Mag_int) - 1)
    else:
        std_gr_mlk = np.nan

    # max_magnitude_dificit
    if b_lsq != 0:
        dM_lsq = Mag_max - (a_lsq / b_lsq)
    else:
        dM_lsq = np.nan

    if b_mlk != 0:
        dM_mlk = Mag_max - (a_mlk / b_mlk)
    else:
        dM_mlk = np.nan

    # standard deviation of b-value
    if len(Mag_int) > 1:
        denominator_std = np.sum((Mag_int - Mag_mean) ** 2) / len(Mag_int) / (len(Mag_int) - 1)
        if denominator_std != 0:
            b_std_lsq = 2.3 * b_lsq ** 2 * np.sqrt(denominator_std)
            b_std_mlk = 2.3 * b_mlk ** 2 * np.sqrt(denominator_std)
        else:
            b_std_lsq = np.nan
            b_std_mlk = np.nan
    else:
        b_std_lsq = np.nan
        b_std_mlk = np.nan

    return b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk, num_Mag_int



def calculate_elapsed_times(jd, t, mag, Mag_elaps=[6.0, 6.5, 7.0, 7.5]):
    """
    Calculate the elapsed time since the last event with a magnitude greater than the specified thresholds.

    Parameters:
    jd - Julian dates of all earthquakes
    t - Current time window
    mag - Magnitudes of all earthquakes
    Mag_elaps - List of magnitude thresholds (default [6.0, 6.5, 7.0, 7.5])

    Returns:
    T_elaps - Elapsed times corresponding to the magnitude thresholds
    """
    T_elaps = np.zeros(len(Mag_elaps))
    indexT_elaps = np.where(jd < t)[0]
    for i, Mag_threshold in enumerate(Mag_elaps):
        indexT_elaps_mag = np.where(mag[indexT_elaps] >= Mag_threshold)[0]
        if len(indexT_elaps_mag) > 0:
            T_elaps[i] = t - jd[indexT_elaps_mag[-1]]
        else:
            T_elaps[i] = np.nan
    return T_elaps

def calculate_seismic_change_rate(sub_jd, Twindow, t):
    """
    Calculate seismic change rates, including z-value and beta value.

    Parameters:
    sub_jd - Julian dates of all earthquakes
    Twindow - Size of the time window
    t - Current time window time

    Returns:
    zvalue - z-value of seismic change rate
    beta - beta value of seismic change rate
    """
    def calculate_zvalue(R1, R2, S1, S2, N1, N2):
        try:
            if N1 == 0 or N2 == 0:
                print(f"Error: Division by zero detected! N1={N1}, N2={N2}, S1={S1}, S2={S2}")
                print(t)
                return np.nan
            
            denominator = (S1 / N1) + (S2 / N2)
            
            if np.isclose(denominator, 0):
                print(f"Warning: Denominator is close to zero! Denominator={denominator}, S1={S1}, N1={N1}, S2={S2}, N2={N2}")
                zvalue = np.nan
            else:
                zvalue = (R1 - R2) / np.sqrt(denominator)
            
            return zvalue
        
        except Exception as e:
            print(f"Exception occurred: {e}, S1={S1}, N1={N1}, S2={S2}, N2={N2}")
            return np.nan

    fTstart = t - Twindow  # start time of the window
    fT = t - 0.5 * Twindow  # midpoint of the window
    fTw = 0.5 * Twindow  
    Tbin = 0.05 * Twindow  # bin size for histogram

    # get indices for the two time periods
    indexr1 = np.where((sub_jd >= fTstart) & (sub_jd < fT))[0]
    indexr2 = np.where((sub_jd >= fT) & (sub_jd < (fT + fTw)))[0]

    # number of events in each period
    N1 = len(indexr1)
    N2 = len(indexr2)

    # calculate event rates
    R1 = N1 / (fT - fTstart)
    R2 = N2 / fTw
                                                                                                     
    # calculate histograms for variance calculation
    nR1, _ = np.histogram(sub_jd[indexr1], bins=np.arange(fTstart, fT + 2 * Tbin, Tbin))
    nR2, _ = np.histogram(sub_jd[indexr2], bins=np.arange(fT, fT + fTw + 2 * Tbin, Tbin))

    # calculate sample variance
    S1 = np.var(nR1, ddof=1)
    S2 = np.var(nR2, ddof=1)

    # calculate event frequency
    vR1, _ = np.histogram(sub_jd, bins=np.arange(fTstart, fT + fTw + 2 * Tbin, Tbin))
    nEq1 = np.sum(vR1)
    nBin1 = len(vR1)

    # calculate parameters for change rate  
    winlen_days = fTw / Tbin
    fNormInvalLength = winlen_days / nBin1

    zvalue = calculate_zvalue(R1, R2, S1, S2, N1, N2)
    beta = (N2 - nEq1 * fNormInvalLength) / np.sqrt(nEq1 * fNormInvalLength * (1 - fNormInvalLength))

    return beta, zvalue

def get_max_magnitude_in_forecast(jd, mag, t, Tfore):
    """
    Get the maximum magnitude within the forecast period (Tfore).

    Parameters:
    jd - Julian dates of all earthquakes
    mag - Magnitudes of all earthquakes
    t - End time of the current time window
    Tfore - Forecast period (time range)

    Returns:
    max_mag - Maximum magnitude within the forecast period
    """
    index_Max_mag_obs = np.where((jd >= t) & (jd < (t + Tfore)))[0]
    
    if len(index_Max_mag_obs) > 0:
        return np.max(mag[index_Max_mag_obs])
    else:
        return np.nan

def calculate_seismic_features(data_input, Mc=4.7, Mf=5.5, Twindow=[20], Tfore=30, dt=30, dMag=0.1, Mag_elaps=[6, 6.5], t_arrary=None,L_max=60,context_len=2):
    """
    Calculate seismic features, including b-value fitting, maximum/average magnitude, seismic energy, occurrence times, and other statistical features.

    Parameters:
    data_input : ndarray
        Earthquake data array in the form [JD, mag, reg]
    Mc : float
        Minimum magnitude threshold
    Mf : float
        Threshold for determining the maximum magnitude after the mainshock (used for marking Negative)
    Twindow : float
        Time window length (days)
    Tfore : float
        Forecast period (days)
    dt : float
        Time step
    dMag : float
        Bin width for magnitude distribution
    Mag_elaps : list of float
        Magnitude thresholds for calculating elapsed times
    t_arrary : array-like or None
        Specified time points (JD), if None, they are generated automatically
    L_max : int
        Maximum number of magnitude bins to consider (default is 60)
    context_len : int
        Context length multiplier for determining Negative samples (default is 2)

    Returns:
    features_df : pd.DataFrame
        Table of various statistical features
    num_mag : ndarray
        Magnitude frequency statistics corresponding to each time point (up to 40 bins)
    """

    data1 = data_input[data_input[:, 1] >= Mc]
    if len(data1) == 0:
        raise ValueError("No earthquake events meet the Mc condition")

    jd = data1[:, 0]
    mag = data1[:, 1]
    # reg = data1[:, 2]

    if t_arrary is not None:
        t_arrary = np.array(t_arrary)
        if len(t_arrary) == 0:
            raise ValueError("t_arrary cannot be empty")
        if np.any(t_arrary < jd[0] + Twindow) or np.any(t_arrary > jd[-1]):
            print("Warning: t_arrary contains values outside the valid range. They will be ignored.")
        Nloop = len(t_arrary)
        t_array_final = t_arrary
    else:
        Nloop = int(np.ceil((jd[-1] - jd[0] - Twindow - Tfore) / dt))
        t_array_final = Twindow + jd[0] + np.arange(Nloop) * dt

    features = {
        "t": t_array_final.copy(),
        "Num": np.zeros(Nloop),
        "Mag_max": np.zeros(Nloop),
        "Mag_max_obs": np.zeros(Nloop),
        "Mag_mean": np.zeros(Nloop),
        "b_lsq": np.zeros(Nloop),
        "a_lsq": np.zeros(Nloop),
        "b_std_lsq": np.zeros(Nloop),
        "std_gr_lsq": np.zeros(Nloop),
        "b_mlk": np.zeros(Nloop),
        "a_mlk": np.zeros(Nloop),
        "b_std_mlk": np.zeros(Nloop),
        "std_gr_mlk": np.zeros(Nloop),
        "dM_lsq": np.zeros(Nloop),
        "dM_mlk": np.zeros(Nloop),
        "Energy_sqrt": np.zeros(Nloop),
        "prob_x7_lsq": np.zeros(Nloop),
        "prob_x7_mlk": np.zeros(Nloop),
        "zvalue": np.zeros(Nloop),
        "beta": np.zeros(Nloop),
        "shock_len": np.zeros(Nloop),
        "Negative": np.zeros(Nloop), 
    }

    for mag_threshold in Mag_elaps:
        features[f"T_elaps{mag_threshold}"] = np.zeros(Nloop)

    num_mag = np.zeros((Nloop, L_max))

    for i in range(Nloop):
        t_now = t_array_final[i]
        index = np.where((jd >= t_now - Twindow) & (jd < t_now))[0]
    
        shock_len = len(np.where((jd >= t_now) & (jd < t_now + Tfore) & (mag >= Mf))[0])
        sub_jd, sub_mag= jd[index], mag[index]
        index_n = np.where((jd >= t_now - context_len * Tfore) & (jd < t_now + context_len * Tfore))[0]
        Negative = True if (len(index_n) == 0 or np.max(mag[index_n]) < Mf) else False
        # print(i,jd[index_n],mag[index_n],Negative)
        features["shock_len"][i] = shock_len
        if context_len ==0:
            features["Negative"][i] = bool(shock_len==0)
        else:
            features["Negative"][i] = Negative

        if len(index) == 0:
            print(f"Warning: No data in window for index {i}, skipping.")
            continue

        b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk, num_mag_int = calculate_magnitudes_and_features(sub_mag, Mc, dMag)
        T_elaps = calculate_elapsed_times(jd, t_now, mag, Mag_elaps)
        beta,zvalue = calculate_seismic_change_rate(sub_jd, Twindow, t_now)
        Mag_max_obs = get_max_magnitude_in_forecast(jd, mag, t_now, Tfore)
       

        features["Num"][i] = len(index)
        features["Mag_max"][i] = np.max(sub_mag)
        features["Mag_mean"][i] = np.mean(sub_mag)
        features["b_lsq"][i] = b_lsq
        features["a_lsq"][i] = a_lsq
        features["std_gr_lsq"][i] = std_gr_lsq
        features["b_mlk"][i] = b_mlk
        features["a_mlk"][i] = a_mlk
        features["std_gr_mlk"][i] = std_gr_mlk
        features["dM_lsq"][i] = dM_lsq
        features["dM_mlk"][i] = dM_mlk
        features["b_std_lsq"][i] = b_std_lsq
        features["b_std_mlk"][i] = b_std_mlk
        features["prob_x7_lsq"][i] = np.exp(-3 * b_lsq / np.log10(np.exp(1)))
        features["prob_x7_mlk"][i] = np.exp(-3 * b_mlk / np.log10(np.exp(1)))
        features["Energy_sqrt"][i] = np.sqrt(np.sum(10 ** (12 + 1.8 * sub_mag)))
        features["beta"][i] = beta
        features["zvalue"][i] = zvalue
        features["Mag_max_obs"][i] = Mag_max_obs
        

        for j, mag_threshold in enumerate(Mag_elaps):
            features[f"T_elaps{mag_threshold}"][i] = T_elaps[j]

        L = min(len(num_mag_int), 40)
        num_mag[i, :L] = num_mag_int[:L]

    features_df = pd.DataFrame(features)
    return features_df, num_mag

def calculate_seismic_features_n(data_input, Mc=4.7, Mf=5.5, Twindow_list=[200], Tfore=30, dt=30, dMag=0.1, Mag_elaps=[6, 6.5], t_arrary=None):
    """
    Calculate seismic features over multiple time windows.

    This function computes seismic features for a given dataset over a list of 
    specified time windows. It uses the `calculate_seismic_features` function 
    internally and aggregates the results for each time window.

    Args:
        data_input (pd.DataFrame): The input seismic data.
        Mc (float, optional): The magnitude cutoff for completeness. Defaults to 4.7.
        Mf (float, optional): The magnitude threshold for filtering. Defaults to 5.5.
        Twindow_list (list of int, optional): A list of time windows (in seconds) 
            over which to calculate features. Defaults to [200].
        Tfore (int, optional): The forecast time window (in seconds). Defaults to 30.
        dt (int, optional): The time step for feature calculation (in seconds). Defaults to 30.
        dMag (float, optional): The magnitude bin size for histogram calculations. Defaults to 0.1.
        Mag_elaps (list of float, optional): A list of magnitude ranges for elapsed time calculations. Defaults to [6, 6.5].
        t_arrary (np.ndarray or None, optional): An optional array of time values to use 
            for feature calculations. If None, it will be updated during the process. Defaults to None.

    Returns:
        tuple:
            - results (dict): A dictionary where keys are time windows and values are 
              the calculated seismic features for each window.
            - num_mag_all (dict): A dictionary where keys are time windows and values are 
              the number of events in each magnitude bin for each window.
    """
    Twindow_list = sorted(Twindow_list, reverse=True)
    results = {}
    num_mag_all = {}
    for Twindow in Twindow_list:

        out, num_mag = calculate_seismic_features(
            data_input, 
            Mc=Mc, 
            Mf=Mf, 
            Twindow=Twindow, 
            Tfore=Tfore, 
            dt=dt, 
            dMag=dMag, 
            Mag_elaps=Mag_elaps, 
            t_arrary=t_arrary
        )

        results[Twindow] = out
        num_mag_all[Twindow] = num_mag

        t_arrary = out["t"].values
    return results, num_mag_all


