from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from src.utils.utils import day_offsets_to_np_datetime64, set_xaxis_time_locator


def plot_b_value(
    seq,
    b_seq: torch.Tensor,
    x_axis: str = "time",
    time_key: str = "arrival_times",
    title: str = "",
    label: str = "b-value",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    smooth: bool = False,
    smooth_sigma: float = 1.0,
    start_time: pd.Timestamp | None = None,
):
    if isinstance(b_seq, torch.Tensor):
        b_seq = b_seq.squeeze().detach().cpu().numpy()

    if smooth:
        b_seq = gaussian_filter1d(b_seq, sigma=smooth_sigma)

    if x_axis == "time":
        if time_key not in seq:
            raise KeyError(f"Sequence is missing the time field '{time_key}'")
        days = seq[time_key].detach().cpu().numpy().squeeze()
        if len(days) != len(b_seq):
            days = days[: len(b_seq)]

        if start_time is None:
            xs = days
            xlabel = "Time (days)"
        else:
            xs = day_offsets_to_np_datetime64(start_time, days)
            xlabel = "Year"
    elif x_axis == "event":
        xs = range(len(b_seq))
        xlabel = "Event index"
    else:
        raise ValueError("x_axis must be 'time' or 'event'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(xs, b_seq, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("b-value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    if x_axis == "time" and start_time is not None:
        set_xaxis_time_locator(ax, start_time)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
