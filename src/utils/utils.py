import os
import random
import numpy as np
import torch
import math

def set_seed(seed: int = 42):
    """为所有可能的随机源设置种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)




def cal2jd(date):
    """
    计算给定日期的儒略日 (Julian Date)
    
    输入参数:
    date - 日期格式：[year, month, day, hour, minute, second]
    
    返回值:
    jd - 对应的儒略日 (Julian Date)
    """
    year, month, day, hour, minute, second = date
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JDN = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + B - 1524.5
    jd = JDN + (hour - 12) / 24 + minute / 1440 + second / 86400
    return jd