import os
import random
import numpy as np
import torch
import math
from datetime import datetime, timezone
import matplotlib.dates as mdates
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False




def cal2jd(date):
    """
    Calculate the Julian Date (JD) for a given date.
    
    Parameters:
    date - Date in the format: [year, month, day, hour, minute, second]
    
    Returns:
    jd - The corresponding Julian Date (JD)
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


def _to_np_datetime64_seconds(t0) -> np.datetime64:
    """Convert various input formats to numpy.datetime64 (seconds precision)."""
    if isinstance(t0, np.datetime64):
        return t0.astype('datetime64[s]')
    if isinstance(t0, datetime):
        return np.datetime64(t0).astype('datetime64[s]')
    if isinstance(t0, (int, float)):  # Assume it is a UNIX timestamp in seconds.
        return np.datetime64(int(t0), 's')
    if isinstance(t0, str):  # ISO string, e.g., '2000-01-01' or '2000-01-01T00:00:00'.
        return np.datetime64(t0).astype('datetime64[s]')
    raise TypeError("start_time must be datetime / np.datetime64 / ISO string / UNIX timestamp in seconds.")


def _to_py_datetime(t64: np.datetime64) -> datetime:
    """Convert np.datetime64 to Python datetime (second precision, tz-aware UTC)."""
    ts = (t64 - np.datetime64('1970-01-01T00:00:00', 's')) / np.timedelta64(1, 's')
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)

def set_xaxis_time_locator(ax, start_time, x_axis="time"):
    """
    Function to set X axis ticks for time-based data.

    Parameters:
    ax : matplotlib.axes.Axes
        The axis to modify.
    start_time : str or datetime
        The start time to align the ticks.
    x_axis : str, optional
        Whether to use time-based x-axis. Default is "time".
    """
    if x_axis == "time" and start_time is not None:
        # Convert np.datetime64 start point to Python datetime to extract month/day
        t0_py = _to_py_datetime(_to_np_datetime64_seconds(start_time))

        # Major ticks: Align with the "year-month-day" of the start time, one tick every 5 years
        ax.xaxis.set_major_locator(
            mdates.YearLocator(base=5, month=t0_py.month, day=t0_py.day)
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # (Optional) Minor ticks: Quarterly positioning for easier reading
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.tick_params(axis='x', which='minor', bottom=False)