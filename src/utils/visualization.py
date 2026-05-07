from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import pandas as pd

from src.data import Sequence
from src.data import Catalog

__all__ = [
    "visualize_sequence",
    "visualize_trajectories",
    "visualize_trajectories_multi_model",
    "visualize_catalog",
    "visualize_forecast_with_tests",
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
    has_events = mag.size > 0
    if mag_completeness is None:
        mag_completeness = float(np.min(mag)) if has_events else 0.0

    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
    if has_events:
        ax.scatter(
            t,
            mag,
            s=np.exp(2 * mag - 6),
            c=event_color,
            label="Events",
        )  # =np.exp(2 * mag - 6)  0.1*np.exp(2 * mag - 2)
    else:
        # Keep the plotting pipeline alive for empty subsequences.
        ax.scatter([], [], c=event_color, label="Events")
    _, y_max = ax.get_ylim()
    if y_max <= mag_completeness:
        y_max = mag_completeness + 1.0
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


def plot_counting_process(
    seq,
    ax=None,
    color="k",
    alpha=1.0,
    T0=None,
    T=None,
    linewidth: float = 1.2,
    linestyle: str = "-",
    zorder: Optional[float] = None,
):
    if ax is None:
        ax = plt.gca()

    t = np.append(seq.arrival_times.numpy(), T)
    t = np.insert(t, 0, T0, axis=0)

    N = np.arange(len(t) - 1)
    N = np.append(N, N[-1])
    ax.step(
        t,
        N,
        where="post",
        c=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def plot_intensity(
    seq,
    ax=None,
    color="k",
    alpha=1.0,
    T0=None,
    T=None,
    dt=1.0,
    logy=False,
    figsize=(8, 3),  # 新增参数
):
    """plot intensity of a sequence"""
    if ax is None:
        plt.figure(figsize=figsize)
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
    event_color: str = "C0",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    t_before: Optional[float] = None,
    num_examples: int = 10,
    offset: int = 0,
    save_path: Optional[str] = None,
    xlabel: str = "Arrival time (days)",
    reset_t_nll_to_end: bool = False,
    bins: int = 40,
    align_count_axes: bool = True,
    simulated_color: str = "#AB6209",
    observed_color: str = "k",
    forecast_window_color: str = "#C2D9F8",
) -> tuple:
    """
    Visualize the observed sequence, example forecast trajectories, and the forecast-count histogram.

    Args:
        seq (Sequence): Observed sequence.
        forecast (List[Sequence]): List of forecast sequences.
        ax (Optional): Axes to plot on.
        figsize (tuple): Figure size.
        dpi (int): Figure resolution.
        event_color (str): Color for events.
        t_start (Optional[float]): Start time for visualization.
        t_end (Optional[float]): End time for visualization.
        t_before (Optional[float]): Time window before t_start.
        num_examples (int): Number of forecast examples to display.
        offset (int): Offset for starting the examples.
        save_path (Optional[str]): Path to save the figure.
        xlabel (str): X-axis label.
        reset_t_nll_to_end (bool): Whether to reset NLL interval to sequence end.
        bins (int): Number of bins for histogram.
        align_count_axes (bool): If True, force strict alignment between the left-panel
            right y-axis (cumulative count) and right-panel y-axis (event count).
        simulated_color (str): Color used for simulated forecast trajectories and histogram.
        observed_color (str): Color used for the observed counting-process line.
        forecast_window_color (str): Fill color for the forecast-time window highlight.

    Returns:
        tuple: (fig, (axA, axB, axAA))
    """
    assert len(forecast) > 0, "Forecast must be non-empty"
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
    offset = max(0, min(int(offset), len(forecast) - 1))
    num_examples = min(int(num_examples), len(forecast) - offset)

    # Create axes with reduced space between subplots
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=(3.2, 1.3), wspace=0.02)
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

    # Left panel: Events + forecast window highlight
    # Use a cool, light shade for the forecast window to avoid
    # clashing with warm simulated-line colors.
    axA.margins(x=0)
    axA.axvspan(t_start, t_end, color=forecast_window_color, alpha=0.22, lw=0)
    axA.axvline(t_start, c="k", lw=1, ls="--", alpha=0.8)

    visualize_sequence(
        seq=s_viz, ax=axA, event_color=event_color, show_legend=False, xlabel=xlabel
    )
    axA.set_title("Observed Sequence and Example Forecast Trajectories", fontsize=10)
    axA.set_xlabel(xlabel, fontsize=9)
    axA.set_ylabel("Magnitude", fontsize=9)
    axA.tick_params(axis="both", labelsize=8)
    axA.grid(axis="x", alpha=0.25)

    # Counting process on the right y-axis of the left panel

    plot_counting_process(
        s_obs,
        axAA,
        color=observed_color,
        alpha=0.98,
        T0=t_start,
        T=t_end,
        linewidth=1.9,
        zorder=5,
    )
    for i_samp in forecast[offset : offset + num_examples]:
        plot_counting_process(
            i_samp.cpu(),
            axAA,
            color=simulated_color,
            alpha=0.46,
            T0=t_start,
            T=t_end,
            linewidth=1.2,
            zorder=2,
        )

    axAA.set_ylabel("Cumulative count", fontsize=9)
    axAA.tick_params(axis="y", labelsize=8)
    # axAA.yaxis.set_major_formatter(
    #     mticker.FuncFormatter(lambda y, _: _format_compact_thousands(y))
    # )
    axAA.grid(False)

    # Reduce x tick density without expanding x-limits to out-of-window ticks.
    xlo, xhi = axA.get_xlim()
    xt = np.asarray(axA.get_xticks(), dtype=float)
    xt = xt[(xt >= xlo) & (xt <= xhi)]
    if len(xt) > 6:
        xt = xt[::2]
    if len(xt) > 0:
        axA.set_xticks(xt)
    # set_xticks can expand limits if any tick is outside; restore exact window.
    axA.set_xlim(xlo, xhi)

    # Right panel: Histogram of event counts per forecast
    counts_per_forecast = np.array([len(s) for s in forecast], dtype=float)
    obs_count = len(s_obs)

    q025, q975 = np.quantile(counts_per_forecast, [0.025, 0.975])
    y_max = max(obs_count, q975) * 1.05

    axB.hist(
        counts_per_forecast,
        bins=bins,
        range=(0, y_max),
        orientation="horizontal",
        facecolor=simulated_color,
        alpha=0.28,
        edgecolor=simulated_color,
        linewidth=0.6,
        label="Simulated",
    )
    axB.axhline(obs_count, c=observed_color, lw=1.8, label="Observed")

    if align_count_axes:
        # Strictly share the same visible count range and ticks on both axes.
        left_line_max = max(
            [obs_count]
            + [len(s) for s in forecast[offset : offset + num_examples]]
        )
        common_ymax = max(float(y_max), float(left_line_max))
        axAA.set_ylim(0.0, common_ymax)
        axB.set_ylim(0.0, common_ymax)

        tick_count = 6
        common_ticks = np.linspace(0.0, common_ymax, tick_count)
        # Keep count ticks visually clean.
        common_ticks = np.unique(np.round(common_ticks).astype(int))
        axAA.set_yticks(common_ticks)
        axB.set_yticks(common_ticks)
    else:
        axB.set_ylim(0.0, y_max)

    axB.set_title("Forecast counts", fontsize=10)
    axB.set_xlabel("Frequency", fontsize=9)
    axB.yaxis.set_ticks_position("right")
    axB.yaxis.set_label_position("right")
    axB.set_ylabel("Event count in window", fontsize=9)
    axB.tick_params(axis="both", labelsize=8)
    # axB.yaxis.set_major_formatter(
    #     mticker.FuncFormatter(lambda y, _: _format_compact_thousands(y))
    # )
    axB.grid(axis="y", alpha=0.15)
    axB.legend(fontsize=8, loc="upper right", frameon=False)

    # Annotation for observed count and 95% interval
    txt = f"Observed: {len(s_obs)}\n95%: [{int(q025)}, {int(q975)}]"
    axB.text(
        0.62,
        0.55,
        txt,
        transform=axB.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # Save the figure if a path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, (axA, axB, axAA)


def visualize_trajectories_multi_model(
    seq: Sequence,
    forecasts_by_model: Dict[str, List[Sequence]],
    figsize: tuple = (10.0, 0.0),
    dpi: int = 150,
    row_height: float = 2.8,
    model_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    **trajectory_kwargs,
) -> tuple:
    """
    Visualize forecast trajectories for multiple models in vertical rows.

    Each model occupies one row, and each row reuses the standard
    `visualize_trajectories` layout (left trajectory panel + right count panel).
    Rows are ordered from top to bottom according to `model_order`
    (or insertion order of `forecasts_by_model` when None).

    Returns:
        tuple: (fig, axes_by_model)
            where axes_by_model[model_name] = (ax_left, ax_right, ax_right_y)
    """
    if not forecasts_by_model:
        raise ValueError("forecasts_by_model must be non-empty")

    if model_order is None:
        model_names = list(forecasts_by_model.keys())
    else:
        model_names = list(model_order)
        missing = [name for name in model_names if name not in forecasts_by_model]
        if missing:
            raise KeyError(f"model_order contains unknown models: {missing}")

    n_models = len(model_names)
    if n_models == 0:
        raise ValueError("No models to visualize")

    fig_w = float(figsize[0])
    fig_h = float(figsize[1]) if len(figsize) > 1 and float(figsize[1]) > 0 else row_height * n_models

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    try:
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    except Exception:
        pass

    gs = fig.add_gridspec(
        n_models,
        2,
        width_ratios=(3.2, 1.3),
        hspace=0.06,
        wspace=0.02,
    )

    axes_by_model = {}
    first_left = None

    for idx, model_name in enumerate(model_names):
        forecast_i = forecasts_by_model[model_name]
        if len(forecast_i) == 0:
            raise ValueError(f"Forecast list for model '{model_name}' is empty")

        if first_left is None:
            ax_left = fig.add_subplot(gs[idx, 0])
            first_left = ax_left
        else:
            ax_left = fig.add_subplot(gs[idx, 0], sharex=first_left)
        ax_right = fig.add_subplot(gs[idx, 1])

        _, (ax_left, ax_right, ax_right_y) = visualize_trajectories(
            seq=seq,
            forecast=forecast_i,
            ax=(ax_left, ax_right),
            dpi=dpi,
            save_path=None,
            **trajectory_kwargs,
        )

        # Normalize header semantics:
        # - column titles appear only on the first row
        # - model name is shown as a row label on the left panel
        ax_left.set_title("", loc="left")
        ax_left.set_title("", loc="center")
        ax_right.set_title("", loc="left")
        ax_right.set_title("", loc="center")

        if idx == 0:
            ax_left.set_title(
                "Observed Sequence and Example Forecast Trajectories",
                loc="center",
                fontsize=10,
            )
            ax_right.set_title("Forecast counts", loc="center", fontsize=10)

        # ax_left.text(
        #     0.0,
        #     1.02,
        #     str(model_name),
        #     transform=ax_left.transAxes,
        #     ha="left",
        #     va="bottom",
        #     fontsize=9,
        #     fontweight="bold",
        # )

        # Subplot labels: label every panel separately
        left_label = f"({chr(ord('a') + 2 * idx)})"
        right_label = f"({chr(ord('a') + 2 * idx + 1)})"

        ax_left.text(
            0.02,
            0.98,
            left_label,
            transform=ax_left.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.7,
            ),
        )

        ax_right.text(
            0.02,
            0.98,
            right_label,
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.7,
            ),
        )

        if idx < n_models - 1:
            ax_left.set_xlabel("")
            ax_right.set_xlabel("")
            ax_left.tick_params(axis="x", labelbottom=False)
            ax_right.tick_params(axis="x", labelbottom=False)

        axes_by_model[model_name] = (ax_left, ax_right, ax_right_y)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes_by_model

def visualize_forecast_with_tests(
    seq: Sequence,
    forecast: List[Sequence],
    m_test_result: Optional[Any] = None,
    n_test_result: Optional[Any] = None,
    figsize: tuple = (8.8, 6.2),
    trajectory_ref_figsize: Optional[tuple] = None,
    row_hspace: float = 0.06,
    bottom_scale: float = 1.0,
    dpi: int = 160,
    save_path: Optional[str] = None,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
    m_plot_args: Optional[Dict[str, Any]] = None,
    n_plot_args: Optional[Dict[str, Any]] = None,
):
    """
    Compose a single figure:
      - top row (spanning both columns): forecast visualization
      - bottom-left: M-test
      - bottom-right: N-test

    Notes:
      - `m_test_result`/`n_test_result` should provide `.plot(ax=..., show=False, plot_args=...)`.
      - If one test result is None, the corresponding panel will show a placeholder.
    """
    def _axes_has_main_artists(ax) -> bool:
        return bool(
            len(ax.lines)
            or len(ax.patches)
            or len(ax.collections)
            or len(ax.images)
            or len(ax.containers)
        )

    def _plot_test_panel(result_obj, ax, kind: str, plot_args: Dict[str, Any]) -> None:
        """Robustly draw test results into the provided axis."""
        errors = []

        # Preferred path: bypass result_obj.plot() and draw directly via csep plotting API.
        try:
            from csep.utils import plots as csep_plots

            if kind == "m":
                csep_plots.plot_magnitude_test(
                    result_obj, axes=ax, show=False, plot_args=plot_args
                )
            else:
                csep_plots.plot_number_test(
                    result_obj, axes=ax, show=False, plot_args=plot_args
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            errors.append(exc)

        if _axes_has_main_artists(ax):
            return

        # Fallback path: handle custom result_obj.plot signatures.
        for kwargs in (
            {"ax": ax, "show": False, "plot_args": plot_args},
            {"axes": ax, "show": False, "plot_args": plot_args},
            {"show": False, "plot_args": plot_args},
        ):
            before = set(plt.get_fignums())
            try:
                result_obj.plot(**kwargs)
            except TypeError as exc:
                errors.append(exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive fallback
                errors.append(exc)
                continue
            finally:
                # Close standalone figures opened by fallback calls.
                for fig_num in set(plt.get_fignums()) - before:
                    if fig_num != ax.figure.number:
                        plt.close(fig_num)

            if _axes_has_main_artists(ax):
                return

        raise RuntimeError(
            f"Failed to render {'M' if kind == 'm' else 'N'}-test into target axis. "
            f"Last error: {errors[-1]!r}" if errors else "No plotting backend succeeded."
        )

    def _add_test_stats_annotation(
        result_obj, ax, kind: str, plot_args: Dict[str, Any]
    ) -> None:
        """
        Add M/N-test stats text when chained plotting suppresses csep default annotations.
        csep's plot_*_test annotates only when `axes is None`.
        """
        if result_obj is None:
            return

        # Avoid duplicate stats text if backend already wrote one.
        existing_text = " ".join(t.get_text() for t in ax.texts)
        if any(k in existing_text for k in ("\\gamma", "\\delta", "\\omega", "P(X")):
            return

        text_fontsize = plot_args.get("text_fontsize", 10)
        annotation_kwargs = {"xycoords": "axes fraction", "fontsize": text_fontsize}

        if kind == "m":
            xy = plot_args.get("xy", (0.55, 0.6))
            try:
                quantile = result_obj.quantile
                if isinstance(quantile, (list, tuple, np.ndarray)):
                    quantile = np.asarray(quantile).reshape(-1)[0]
                observed_statistic = float(result_obj.observed_statistic)
                txt = (
                    rf"$\gamma = P(X \geq x) = {float(quantile):.2f}$"
                    + "\n"
                    + rf"$\omega = {observed_statistic:.2f}$"
                )
                ax.annotate(txt, xy=xy, **annotation_kwargs)
            except Exception:
                return
        else:
            xy = plot_args.get("xy", (0.5, 0.3))
            try:
                quantile_arr = np.asarray(result_obj.quantile).reshape(-1)
                observed_statistic = float(result_obj.observed_statistic)
                if quantile_arr.size >= 2:
                    if abs(observed_statistic - round(observed_statistic)) < 1e-9:
                        observed_repr = f"{int(round(observed_statistic))}"
                    else:
                        observed_repr = f"{observed_statistic:.2f}"
                    txt = (
                        rf"$\delta_1 = P(X \geq x) = {float(quantile_arr[0]):.2f}$"
                        + "\n"
                        + rf"$\delta_2 = P(X \leq x) = {float(quantile_arr[1]):.2f}$"
                        + "\n"
                        + rf"$\omega = {observed_repr}$"
                    )
                elif quantile_arr.size == 1:
                    txt = (
                        rf"$\gamma = P(X \leq x) = {float(quantile_arr[0]):.2f}$"
                        + "\n"
                        + rf"$\omega = {observed_statistic:.2f}$"
                    )
                else:
                    return
                ax.annotate(txt, xy=xy, **annotation_kwargs)
            except Exception:
                return

    def _add_panel_label(ax, label: str) -> None:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.8),
            zorder=20,
        )

    def _force_upper_right_legend(ax, fontsize: float = 8) -> None:
        """Force existing legend to the upper-right while preserving style."""
        legend = ax.get_legend()
        if legend is None:
            return
        frameon = legend.get_frame_on()
        legend.remove()
        kwargs = {"loc": "upper right", "frameon": frameon, "fontsize": fontsize}
        ax.legend(**kwargs)

    def _force_test_legend_blue(
        ax, fontsize: float = 8, simulated_color: str = "C0"
    ) -> None:
        """
        Force M/N-test legend handles so Simulated is always blue.
        csep may color rejection-tail bars red, which can leak into auto legend handles.
        """
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()] if legend is not None else []
        frameon = legend.get_frame_on() if legend is not None else True
        if legend is not None:
            legend.remove()

        observed_label = next((x for x in labels if "obs" in x.lower()), "observed")
        simulated_label = next((x for x in labels if "sim" in x.lower()), "Simulated")
        handles = [
            Line2D([0], [0], color="k", linestyle="--", lw=1.6),
            Patch(facecolor=simulated_color, edgecolor=simulated_color),
        ]
        ax.legend(
            handles,
            [observed_label, simulated_label],
            loc="upper right",
            frameon=frameon,
            fontsize=fontsize,
        )

    if len(forecast) == 0:
        raise ValueError("forecast must be non-empty")

    forecast_kwargs = dict(forecast_kwargs or {})
    m_plot_args = dict(m_plot_args or {})
    n_plot_args = dict(n_plot_args or {})

    # Keep forecast visualization strictly within `seq` bounds to avoid
    # get_subsequence(...) range errors when users pass forecast-window-only sequences.
    sample_forecast = forecast[0]
    seq_start = float(seq.t_start)
    seq_end = float(seq.t_end)
    req_t_start = float(forecast_kwargs.get("t_start", sample_forecast.t_start))
    req_t_end = float(forecast_kwargs.get("t_end", sample_forecast.t_end))

    safe_t_start = max(req_t_start, seq_start)
    safe_t_end = min(req_t_end, seq_end)
    if safe_t_end <= safe_t_start:
        raise ValueError(
            f"Invalid plotting window after clipping: [{safe_t_start}, {safe_t_end}]. "
            f"`seq` range is [{seq_start}, {seq_end}]."
        )

    if "t_before" in forecast_kwargs:
        req_t_before = float(forecast_kwargs["t_before"])
        avail_before = max(0.0, safe_t_start - seq_start)
        safe_t_before = max(0.0, min(req_t_before, avail_before))
    else:
        # Match visualize_trajectories default intent (`duration`) but clamp to available context.
        default_t_before = safe_t_end - safe_t_start
        avail_before = max(0.0, safe_t_start - seq_start)
        safe_t_before = min(default_t_before, avail_before)

    forecast_kwargs["t_start"] = safe_t_start
    forecast_kwargs["t_end"] = safe_t_end
    forecast_kwargs["t_before"] = safe_t_before

    fig_w, fig_h = float(figsize[0]), float(figsize[1])
    if trajectory_ref_figsize is not None:
        # Strictly align top panel height with standalone visualize_trajectories(figsize=(W, H)).
        top_h = float(trajectory_ref_figsize[1])
    else:
        top_h = 2.0
    bottom_scale = max(0.5, min(float(bottom_scale), 1.0))
    bottom_h = max((fig_h - top_h) * bottom_scale, 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    # Use tighter constrained-layout pads so subplot gaps stay compact by default.
    try:
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    except Exception:
        pass
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(top_h, bottom_h),
        hspace=float(row_hspace),
        wspace=0.08,
    )
    gs_top = gs[0, :].subgridspec(1, 2, width_ratios=(3.25, 1.25), wspace=0.005)

    ax_fore_left = fig.add_subplot(gs_top[0, 0])
    ax_fore_right = fig.add_subplot(gs_top[0, 1])
    _, (ax_fore_left, ax_fore_right, ax_fore_right_y) = visualize_trajectories(
        seq=seq,
        forecast=forecast,
        ax=(ax_fore_left, ax_fore_right),
        save_path=None,
        dpi=dpi,
        **forecast_kwargs,
    )

    ax_m = fig.add_subplot(gs[1, 0])
    ax_n = fig.add_subplot(gs[1, 1])
    for _ax in (ax_m, ax_n):
        # Preserve the same shape while shrinking both bottom subplots proportionally.
        try:
            _ax.set_box_aspect(0.68 * bottom_scale)
        except Exception:
            pass

    if m_test_result is not None:
        m_plot_args.setdefault("title", "M-Test")
        m_plot_args.setdefault("tight_layout", False)
        m_plot_args.setdefault("text_fontsize", 10)
        try:
            _plot_test_panel(m_test_result, ax_m, kind="m", plot_args=m_plot_args)
            _add_test_stats_annotation(
                m_test_result, ax_m, kind="m", plot_args=m_plot_args
            )
        except Exception:
            ax_m.text(0.5, 0.5, "M-test plot failed", ha="center", va="center")
            ax_m.set_axis_off()
    else:
        ax_m.text(0.5, 0.5, "M-test unavailable", ha="center", va="center")
        ax_m.set_axis_off()

    if n_test_result is not None:
        n_plot_args.setdefault("title", "N-Test")
        n_plot_args.setdefault("tight_layout", False)
        n_plot_args.setdefault("text_fontsize", 10)
        try:
            _plot_test_panel(n_test_result, ax_n, kind="n", plot_args=n_plot_args)
            _add_test_stats_annotation(
                n_test_result, ax_n, kind="n", plot_args=n_plot_args
            )
        except Exception:
            ax_n.text(0.5, 0.5, "N-test plot failed", ha="center", va="center")
            ax_n.set_axis_off()
    else:
        ax_n.text(0.5, 0.5, "N-test unavailable", ha="center", va="center")
        ax_n.set_axis_off()

    # Keep forecast row labels readable after the extra subplots are added
    ax_fore_left.set_title(ax_fore_left.get_title(), fontsize=10)
    ax_fore_right.set_title(ax_fore_right.get_title(), fontsize=10)
    ax_fore_right_y.tick_params(axis="y", labelsize=8)
    # Keep legends in merged figure fixed at upper-right.
    _force_upper_right_legend(ax_fore_right, fontsize=8)
    _force_test_legend_blue(ax_m, fontsize=8, simulated_color="C0")
    _force_test_legend_blue(ax_n, fontsize=8, simulated_color="C0")

    # Panel labels: (a), (b), (c), (d)
    _add_panel_label(ax_fore_left, "(a)")
    _add_panel_label(ax_fore_right, "(b)")
    _add_panel_label(ax_m, "(c)")
    _add_panel_label(ax_n, "(d)")

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, {
        "forecast_left": ax_fore_left,
        "forecast_right": ax_fore_right,
        "forecast_right_y": ax_fore_right_y,
        "m_test": ax_m,
        "n_test": ax_n,
    }
