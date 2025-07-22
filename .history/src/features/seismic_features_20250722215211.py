# 'b_std_mlk', 'std_gr_mlk' 'b_std_lsq', 'std_gr_lsq'与matlab程序计算结果有差异
import numpy as np
import math
import pandas as pd
from datetime import datetime
from astropy.time import Time


def cal2jd_astropy(date):
    """
    用 Astropy 实现儒略日计算，支持 datetime 或 Timestamp 或 [year, month, day, hour, minute, second] 格式
    """
    if isinstance(date, (pd.Timestamp, datetime)):
        t = Time(date, scale='utc')
    else:
        year, month, day, hour, minute, second = date
        dt = datetime(year, month, day, hour, minute, int(second), int((second % 1) * 1e6))
        t = Time(dt, scale='utc')
    return t.jd

def calculate_magnitudes_and_features(Mag, Mc, dMag):
    """
    计算每个时间窗口内地震数据的b值和a值，以及与地震震级相关的其他特征。
    
    输入参数:
    Mag - 当前时间窗口的地震震级数据
    Mc - 最小震级
    dMag - 震级区间的步长

    返回值:
    b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk
    """
    if len(Mag) == 0:
        # 如果没有地震数据，返回 NaN 或默认值
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    Mag_max = np.max(Mag)
    Mag_mean = np.mean(Mag)

    # 震级区间及对应的地震事件数量
    Mag_int = np.arange(Mc, Mag_max + dMag, dMag)
    Mag_int = np.round(Mag_int, 1)  # 保留一位小数
    num_Mag_int = np.array([len(np.where(Mag >= m)[0]) for m in Mag_int])

    log_NumM = np.log10(np.maximum(num_Mag_int, 1e-10))  # 替换0为一个很小的正数


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

    # 标准差计算 std_gr_lsq
    if len(Mag_int) > 1 and denominator_b_lsq != 0:
        std_gr_lsq = np.sum((log_NumM - a_lsq - b_lsq * Mag_int) ** 2) / (len(Mag_int) - 1)
    else:
        std_gr_lsq = np.nan

    # 最大似然估计 (Maximum Likelihood)
    if Mag_mean > Mc:
        b_mlk = np.log10(np.exp(1)) / (Mag_mean - Mc)
        a_mlk = np.log10(len(Mag)) + b_mlk * Mc
    else:
        b_mlk = np.nan
        a_mlk = np.nan

    # 最大似然的标准差计算 std_gr_mlk
    if len(Mag_int) > 1 and b_mlk != np.nan:
        std_gr_mlk = np.sum((log_NumM - a_mlk - b_mlk * Mag_int) ** 2) / (len(Mag_int) - 1)
    else:
        std_gr_mlk = np.nan

    # 最大震级缺口
    if b_lsq != 0:
        dM_lsq = Mag_max - (a_lsq / b_lsq)
    else:
        dM_lsq = np.nan

    if b_mlk != 0:
        dM_mlk = Mag_max - (a_mlk / b_mlk)
    else:
        dM_mlk = np.nan

    # 计算标准差（用于误差估算）
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
    计算自上次震级大于指定值的事件发生以来经过的时间。
    
    输入参数:
    jd - 所有地震的 Julian 日期
    t - 当前时间窗口的时间
    mag - 当前所有地震的震级
    Mag_elaps - 震级阈值列表（默认 [6.0, 6.5, 7.0, 7.5]）
    
    返回值:
    T_elaps - 对应震级阈值的经过时间
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
    计算地震性变化率，包括 zvalue 和 beta 值。

    输入参数:
    sub_jd - 所有地震的 Julian 日期
    Twindow - 时间窗口大小
    t - 当前时间窗口的时间

    返回值:
    zvalue - 地震性变化率的 z 值
    beta - 地震性变化率的 beta 值
    """
    def calculate_zvalue(R1, R2, S1, S2, N1, N2):
        try:
            # 检查 N1 或 N2 是否为零
            if N1 == 0 or N2 == 0:
                print(f"错误：检测到除以零！N1={N1}, N2={N2}, S1={S1}, S2={S2}")
                print(t)
                return np.nan
            
            # 计算分母
            denominator = (S1 / N1) + (S2 / N2)
            
            # 检查分母是否接近零
            if np.isclose(denominator, 0):
                print(f"警告：分母接近零！Denominator={denominator}, S1={S1}, N1={N1}, S2={S2}, N2={N2}")
                zvalue = np.nan
            else:
                zvalue = (R1 - R2) / np.sqrt(denominator)
                # print(f"成功：zvalue={zvalue}, R1={R1}, R2={R2}, Denominator={denominator}, S1={S1}, N1={N1}, S2={S2}, N2={N2}")
            
            return zvalue
        
        except Exception as e:
            print(f"发生异常：{e}, S1={S1}, N1={N1}, S2={S2}, N2={N2}")
            return np.nan

    fTstart = t - Twindow  # 时间窗口开始时间
    fT = t - 0.5 * Twindow  # 时间窗口中间时间
    fTw = 0.5 * Twindow  
    Tbin = 0.05 * Twindow  # 时间窗口的分bin大小

    # 获取时间段内的地震事件索引
    indexr1 = np.where((sub_jd >= fTstart) & (sub_jd < fT))[0]
    indexr2 = np.where((sub_jd >= fT) & (sub_jd < (fT + fTw)))[0]

    # 计算每个时间段内地震事件的数量
    N1 = len(indexr1)
    N2 = len(indexr2)

    # 计算震级比率
    R1 = N1 / (fT - fTstart)
    R2 = N2 / fTw
                                                                                                     
    # 计算震级事件的分布
    nR1, _ = np.histogram(sub_jd[indexr1], bins=np.arange(fTstart, fT + 2 * Tbin, Tbin))
    nR2, _ = np.histogram(sub_jd[indexr2], bins=np.arange(fT, fT + fTw + 2 * Tbin, Tbin))

    # 计算样本方差
    S1 = np.var(nR1, ddof=1)  
    S2 = np.var(nR2, ddof=1)

    # 计算地震发生频率
    vR1, _ = np.histogram(sub_jd, bins=np.arange(fTstart, fT + fTw + 2 * Tbin, Tbin))
    nEq1 = np.sum(vR1)
    nBin1 = len(vR1)

    # 计算变化率的参数
    winlen_days = fTw / Tbin
    fNormInvalLength = winlen_days / nBin1

    zvalue = calculate_zvalue(R1, R2, S1, S2, N1, N2)
    beta = (N2 - nEq1 * fNormInvalLength) / np.sqrt(nEq1 * fNormInvalLength * (1 - fNormInvalLength))

    return  beta,zvalue

def get_max_magnitude_in_forecast(jd, mag, t, Tfore):
    """
    获取预测期（Tfore）内的最大震级。

    输入参数:
    jd - 所有地震的 Julian 日期
    mag - 所有地震的震级
    t - 当前时间窗口结束时间
    Tfore - 预测期（时间范围）

    返回值:
    max_mag - 预测期内的最大震级
    """
    # 获取预测期内的地震事件索引
    index_Max_mag_obs = np.where((jd >= t) & (jd < (t + Tfore)))[0]
    
    # 如果有地震事件，返回最大震级
    if len(index_Max_mag_obs) > 0:
        return np.max(mag[index_Max_mag_obs])
    else:
        return np.nan

def calculate_seismic_features(data_input, Mc=4.7, Mf=5.5, Twindow=[20], Tfore=30, dt=30, dMag=0.1, Mag_elaps=[6, 6.5], t_arrary=None,L_max=60,context_len=2):
    """
    计算地震特征，包括 b值拟合、最大/平均震级、地震能量、发生时间等统计特征。

    参数:
    data_input : ndarray
        地震数据数组，形如 [JD, mag, reg]
    Mc : float
        最小震级阈值
    Mf : float
        用于判断主震后的最大震级阈值（用于标注 Negative）
    Twindow : float
        时间窗口长度（天）
    Tfore : float
        预测期（天）
    dt : float
        时间步长
    dMag : float
        用于震级分布的 bin 宽度
    Mag_elaps : list of float
        用于计算发生时间间隔的震级阈值
    t_arrary : array-like or None
        指定的时间节点（JD），若为 None 则自动生成

    返回:
    features_df : pd.DataFrame
        各种统计特征组成的表格
    num_mag : ndarray
        每个时间点对应的震级频数统计（最多40个bin）
    """

    # 只保留震级大于 Mc 的事件
    data1 = data_input[data_input[:, 1] >= Mc]
    if len(data1) == 0:
        raise ValueError("没有符合 Mc 条件的地震事件")

    jd = data1[:, 0]
    mag = data1[:, 1]
    # reg = data1[:, 2]

    # 时间数组设定
    if t_arrary is not None:
        t_arrary = np.array(t_arrary)
        if len(t_arrary) == 0:
            raise ValueError("t_arrary 不能为空")
        if np.any(t_arrary < jd[0] + Twindow) or np.any(t_arrary > jd[-1]):
            print("Warning: t_arrary 有值超出数据时间范围，会导致空窗口")
        Nloop = len(t_arrary)
        t_array_final = t_arrary
    else:
        Nloop = int(np.ceil((jd[-1] - jd[0] - Twindow - Tfore) / dt))
        t_array_final = Twindow + jd[0] + np.arange(Nloop) * dt

    # 初始化特征字典
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

    # 遍历每个时间节点
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

        # 调用子函数计算特征
        b_lsq, a_lsq, std_gr_lsq, b_mlk, a_mlk, std_gr_mlk, dM_lsq, dM_mlk, b_std_lsq, b_std_mlk, num_mag_int = calculate_magnitudes_and_features(sub_mag, Mc, dMag)
        T_elaps = calculate_elapsed_times(jd, t_now, mag, Mag_elaps)
        beta,zvalue = calculate_seismic_change_rate(sub_jd, Twindow, t_now)
        Mag_max_obs = get_max_magnitude_in_forecast(jd, mag, t_now, Tfore)
       

        # 写入特征
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

    # 返回 DataFrame 和 num_mag 数组
    features_df = pd.DataFrame(features)
    return features_df, num_mag

def calculate_seismic_features_n(data_input, Mc=4.7, Mf=5.5, Twindow_list=[200], Tfore=30, dt=30, dMag=0.1, Mag_elaps=[6, 6.5], t_arrary=None):
    Twindow_list = sorted(Twindow_list, reverse=True)
    results = {}
    num_mag_all = {}
    for Twindow in Twindow_list:
        # 计算地震特征
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

        # 存储当前时间窗口的结果
        results[Twindow] = out
        num_mag_all[Twindow] = num_mag

        # 更新 t_arrary，保证后续窗口使用相同的时间点
        t_arrary = out["t"].values
    return results, num_mag_all


