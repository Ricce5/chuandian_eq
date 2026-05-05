"""Reusable plotting style helpers.

This module centralizes common Matplotlib styling utilities shared by
forecasting/visualization helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
from matplotlib import cycler
from matplotlib import ticker as mticker

__all__ = [
    "apply_publication_style",
    "save_pub_figure",
    "format_time_axis",
    "style_axes",
]


def apply_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.prop_cycle": cycler(
                color=[
                    "#4E79A7",
                    "#F28E2B",
                    "#59A14F",
                    "#E15759",
                    "#76B7B2",
                    "#B07AA1",
                    "#EDC948",
                    "#9C755F",
                    "#BAB0AC",
                ]
            ),
        }
    )


def save_pub_figure(fig, output_path, dpi: int = 300, file_format: str = "pdf") -> Path:
    output_path = Path(output_path)
    file_format = str(file_format).lower().lstrip(".")
    if file_format not in {"pdf", "png"}:
        raise ValueError("file_format must be 'pdf' or 'png'")
    target_path = output_path.with_suffix(f".{file_format}")
    save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = dpi
    fig.savefig(target_path, format=file_format, **save_kwargs)
    return target_path


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=0)


def style_axes(
    ax,
    xlabel="Forecast start date",
    ylabel=None,
    integer_y=False,
    compact_y_thousands=False,
):
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if integer_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if compact_y_thousands:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _pos: (
                    f"{int(round(value / 1000.0))}k"
                    if abs(float(value)) >= 1000.0 and abs((value / 1000.0) - round(value / 1000.0)) < 1e-9
                    else (
                        f"{value / 1000.0:.1f}k"
                        if abs(float(value)) >= 1000.0
                        else (f"{int(round(value))}" if abs(value - round(value)) < 1e-9 else f"{value:g}")
                    )
                )
            )
        )

