from astropy.time import Time
import pandas as pd
from datetime import datetime
import math

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
