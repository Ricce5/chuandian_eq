from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

from src.data import Sequence
from src.data import Catalog

__all__ = [
    "visualize_sequence",
    "visualize_trajectories",
    "visualize_catalog",
]

def visualize_catalog(
    catalog: Catalog,
    ax = None,
    figsize: tuple = (9, 3),
    dpi: int = 100,
    event_color="C0",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    plot_style: str = 'scatter',
):
    
    if t_start is None:
        t_start = catalog.metadata["start_ts"]
    if t_end is None:
        t_end = catalog.metadata["end_ts"]
    
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi,figsize=figsize)
    
    def day2timestamp(N,start_time_stamp=catalog.metadata["start_ts"]):
        return start_time_stamp + N * pd.Timedelta(1,'D')
    
    plt_fun = {
        'hist':    lambda ax, seq: ax.hist(day2timestamp(seq.arrival_times.numpy()),100,alpha=0.5),
        'scatter': lambda ax, seq: visualize_sequence(
            seq,
            ax,
            event_color=event_color,
            t_start = t_start,
            t_end = t_end,
            show_legend = False,
            time_transform = day2timestamp,
        )
    }
    
    plt_fun[plot_style](ax,catalog.full_sequence)
    for k in ['start_ts','end_ts','train_start_ts','val_start_ts','test_start_ts']:
        ax.axvline(catalog.metadata[k],c='k',ls='--')
    ax.text(
        1, 1, 
        f'Catalog: {catalog.metadata["name"]}\n$M_c$: {catalog.metadata["mag_completeness"]}',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )
    return ax
    

    
def visualize_sequence(
    seq: Sequence,
    ax=None,
    show_nll: bool = False,
    mag_completeness: Optional[float] = None,
    figsize: tuple = (9, 3),
    dpi: int = 100,
    event_color="C0",
    nll_interval_color="C1",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    show_legend: bool = True,
    time_transform = lambda x: x,
    xlabel: str = "Arrival time (days)"
):
    if t_start is None:
        t_start = seq.t_start
    if t_end is None:
        t_end = seq.t_end

    t = time_transform(seq.arrival_times.cpu().numpy())
    
    mag = seq.mag.cpu().numpy()
    if mag_completeness is None:
        mag_completeness = np.min(mag)

    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
    ax.scatter(t, mag, s=np.exp(2 * mag - 6) , c=event_color, label="Events")  # =np.exp(2 * mag - 6)  0.1*np.exp(2 * mag - 2)
    _, y_max = ax.get_ylim()
    if show_nll:
        ax.add_patch(
            Rectangle(
                [seq.t_nll_start, mag_completeness],
                t_end - seq.t_nll_start,
                y_max - mag_completeness,
                alpha=0.2,
                facecolor=nll_interval_color,
                label="Interval on which NLL is computed",
            )
        )
    ax.set_xlabel(f"{xlabel}")
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(mag_completeness, y_max)
    ax.set_ylabel("Magnitude")
    if show_legend:
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=[0.5, 1.15])
    return ax


def plot_counting_process(seq, ax=None, color="k", alpha=1.0, T0=None, T=None):
    if ax is None:
        ax = plt.gca()

    t = np.append(seq.arrival_times.numpy(), T)
    t = np.insert(t, 0, T0, axis=0)

    N = np.arange(len(t) - 1)
    N = np.append(N, N[-1])  
    ax.plot(t, N, c=color, alpha=alpha)


def plot_intensity(
    seq,
    ax=None,
    color="k",
    alpha=1.0,
    T0=None,
    T=None,
    dt=1.0,         
    logy=False,
):
    """plot intensity of a sequence"""
    if ax is None:
        ax = plt.gca()

    if T0 is None:
        T0 = float(seq.arrival_times.min())
    if T is None:
        T = float(seq.arrival_times.max())

    t_arr = seq.arrival_times.cpu().numpy()
    edges = np.arange(T0, T + dt, dt)
    counts, _ = np.histogram(t_arr, bins=edges)
    lam = counts / dt

    t_plot = np.repeat(edges, 2)[1:-1]
    l_plot = np.repeat(lam, 2)

    ax.plot(t_plot, l_plot, color=color, alpha=alpha)
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(T0, T)
    ax.set_ylabel("Intensity")
    return ax



def visualize_trajectories(
    seq: Sequence,
    forecast: List[Sequence],
    ax=None,
    figsize: tuple = (6.6, 3.0),
    dpi: int = 150,
    event_color="C0",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    t_before: Optional[float] = None,
    num_examples: int = 10,
    offset: int = 0,
    save_path: Optional[str] = None,
    xlabel: str = "Arrival time (days)",
    reset_t_nll_to_end: bool = False,
    bins: int = 40,
):
    """Visualize observed sequence + example forecast trajectories + forecast-count histogram.

    Returns
    -------
    fig, (axA, axB, axAA)
    """
    assert len(forecast) > 0, "forecast must be non-empty"
    sample_forecast = forecast[0]

    # Resolve time bounds
    if t_start is None:
        t_start = sample_forecast.t_start
    if t_end is None:
        t_end = sample_forecast.t_end

    duration = t_end - t_start
    if t_before is None:
        t_before = duration

    # Clamp offset/num_examples
    offset = int(offset)
    num_examples = int(num_examples)
    offset = max(0, min(offset, len(forecast) - 1))
    num_examples = min(num_examples, len(forecast) - offset)

    # Create axes
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=(3.2, 1.3),
            wspace=0.05,
        )
        axA = fig.add_subplot(gs[0])
        axAA = axA.twinx()
        axB = fig.add_subplot(gs[1], sharey=axAA)
    else:
        fig = ax[0].figure
        assert len(ax) == 2, "ax must be (axA, axB)"
        axA, axB = ax
        axAA = axA.twinx()

    # Data slices
    s_viz = seq.get_subsequence(
        t_start - t_before, t_end, reset_t_nll_to_end=reset_t_nll_to_end
    ).cpu()
    s_obs = seq.get_subsequence(
        t_start, t_end, reset_t_nll_to_end=reset_t_nll_to_end
    ).cpu()

    # --- Left panel: events + forecast window highlight ---
    axA.margins(x=0)
    axA.axvspan(t_start, t_end, color="k", alpha=0.06, lw=0)
    axA.axvline(t_start, c="k", lw=1, ls="--", alpha=0.8)

    visualize_sequence(seq=s_viz, ax=axA, event_color=event_color, show_legend=False, xlabel=xlabel)

    axA.set_title("Observed sequence + example forecast trajectories", fontsize=10)
    axA.set_xlabel(xlabel, fontsize=9)
    axA.set_ylabel("Magnitude", fontsize=9)
    axA.tick_params(axis="both", labelsize=8)
    axA.grid(axis="x", alpha=0.25)

    # Counting process (right y of left panel)
    plot_counting_process(s_obs, axAA, "k", T0=t_start, T=t_end)
    for i_samp in forecast[offset:offset + num_examples]:
        plot_counting_process(i_samp.cpu(), axAA, "k", 0.18, T0=t_start, T=t_end)

    axAA.set_ylabel("Cumulative count", fontsize=9)
    axAA.tick_params(axis="y", labelsize=8)
    axAA.grid(False)

    # Reduce x tick density
    xt = axA.get_xticks()
    if len(xt) > 6:
        axA.set_xticks(xt[::2])

    # --- Right panel: histogram of event counts per forecast ---
    counts_per_forecast = np.array([len(s) for s in forecast], dtype=float)
    obs_count = len(s_obs)

    # y-axis (shared with cumulative count axis); choose sensible limits
    q025, q975 = np.quantile(counts_per_forecast, [0.025, 0.975])
    y_max = max(obs_count, q975) * 1.05
    axB.set_ylim(0, y_max)

    axB.hist(
        counts_per_forecast,
        bins=bins,
        range=(0, y_max),
        orientation="horizontal",
        facecolor="k",
        alpha=0.22,
        edgecolor="w",
        linewidth=0.8,
        label="Simulated",
    )
    axB.axhline(obs_count, c="C1", lw=1.6, label="Observed")

    axB.set_title("Forecast counts", fontsize=10)
    axB.set_xlabel("Frequency", fontsize=9)
    axB.yaxis.set_ticks_position("right")
    axB.yaxis.set_label_position("right")
    axB.set_ylabel("Event count in window", fontsize=9)
    axB.tick_params(axis="both", labelsize=8)
    axB.grid(axis="y", alpha=0.15)
    axB.legend(fontsize=8, loc="upper right", frameon=False)

    # Annotation (mode + 95% interval)
    txt = f"Observed: {len(s_obs)}\n95%: [{int(q025)}, {int(q975)}]"

    axB.text(
        0.05, 0.55, txt,
        transform=axB.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, (axA, axB, axAA)
