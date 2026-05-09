import os
import random
import math
from datetime import datetime, timezone

import matplotlib.dates as mdates
import numpy as np
import torch
from .runtime_utils import resolve_project_root, unwrap_compiled_model

__all__ = [
    "set_seed",
    "cal2jd",
    "_to_np_datetime64_seconds",
    "_to_py_datetime",
    "set_xaxis_time_locator",
    "resolve_project_root",
    "unwrap_compiled_model",
]

def set_seed(
    seed: int = 42,
    *,
    deterministic: bool = True,
    deterministic_warn_only: bool = False,
    use_deterministic_algorithms: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=bool(deterministic_warn_only))

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



def set_xaxis_time_locator(
    ax,
    start_time,
    x_axis: str = "time",
    major_date_fmt: str = "%Y",
    major_unit: str = "year",   # "day" | "month" | "year"
    major_interval: int = 5,  
    minor_unit: str | None = "month",  # None | "day" | "month" | "quarter"
    hide_minor_ticks: bool = True,
):
    """
    Set X axis ticks for time-based data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify.
    start_time : str | datetime | np.datetime64
        Start time used for alignment (month/day when using year/month locators).
    x_axis : str
        Use time-based x-axis when == "time".
    major_date_fmt : str
        Major tick label format (matplotlib.dates.DateFormatter).
        Examples: "%Y", "%Y-%m", "%Y-%m-%d".
    major_unit : str
        Major tick unit: "day", "month", "year".
    major_interval : int
        Major tick interval count (e.g., 5 days / 5 months / 5 years).
    minor_unit : str | None
        Minor tick unit: None, "day", "month", "quarter".
    hide_minor_ticks : bool
        If True, do not draw minor tick marks on bottom (locator may still help grids).
    """
    if x_axis != "time" or start_time is None:
        return

    if major_interval < 1:
        raise ValueError("major_interval must be >= 1")

    # Convert start_time to Python datetime to extract month/day, etc.
    t0_py = _to_py_datetime(_to_np_datetime64_seconds(start_time))
    if not isinstance(t0_py, datetime):
        raise TypeError("start_time could not be converted to datetime")

    # ---- Major locator ----
    major_unit = major_unit.lower()
    if major_unit == "year":
        ax.xaxis.set_major_locator(
            mdates.YearLocator(base=major_interval, month=t0_py.month, day=t0_py.day)
        )
    elif major_unit == "month":
        # Note: if t0_py.day is 29/30/31 some months may not have that day -> ticks may skip.
        ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=major_interval, bymonthday=1)  # 固定每月1号
    )
    elif major_unit == "day":
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=major_interval)
        )
    else:
        raise ValueError('major_unit must be one of: "day", "month", "year"')

    ax.xaxis.set_major_formatter(mdates.DateFormatter(major_date_fmt))

    # ---- Minor locator (optional) ----
    if minor_unit is None:
        ax.xaxis.set_minor_locator(mdates.NullLocator())
    else:
        minor_unit = minor_unit.lower()
        if minor_unit == "quarter":
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        elif minor_unit == "month":
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        elif minor_unit == "day":
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        else:
            raise ValueError('minor_unit must be one of: None, "day", "month", "quarter"')

        if hide_minor_ticks:
            ax.tick_params(axis="x", which="minor", bottom=False)
