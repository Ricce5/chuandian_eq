"""Reusable workflow helpers for induced-earthquake background comparisons.

The notebook that uses this module should own experiment choices and execution
switches.  This module owns model-spec validation, checkpoint/catalog loading,
background-intensity extraction, plotting, forecast sampling, catalog tests,
cache selection, and concise output naming.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import Batch
from src.models.updaters import UpdaterSamplingWrapper, build_sampling_updater
from src.train.config_setup import load_and_prepare_model
from src.utils import catalog_tests
from src.utils.forecast_eval import (
    build_default_sliding_cache_filename,
    build_sliding_cache_metadata,
    load_sliding_window_cache_if_compatible,
    save_pub_figure,
    style_current_figure,
)
from src.utils.runtime_utils import unwrap_compiled_model
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts
from src.utils import viz_sequences as vis


@dataclass(frozen=True)
class BackgroundComparisonStyle:
    model_palette: tuple[str, ...] = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    )
    line_styles: tuple[Any, ...] = (
        "-",
        (0, (6, 2)),
        (0, (2, 1.4)),
        (0, (4, 1.6, 1.2, 1.6)),
        (0, (8, 2, 2, 2)),
        (0, (1, 1)),
    )
    injection_color: str = "#6B7280"
    seismicity_color: str = "#1F1F1F"
    full_width: float = 7.2
    wide_width: float = 12.0
    medium_width: float = 6.0
    font_size: float = 8.5
    label_size: float = 8.5
    tick_size: float = 7.5
    legend_size: float = 7.5
    grid_color: str = "#D9D9D9"
    spine_color: str = "#4A4A4A"
    dpi: int = 600
    figure_formats: tuple[str, ...] = ("pdf", "png")
    legend_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: {
            "frameon": False,
            "handlelength": 2.2,
            "handletextpad": 0.45,
            "borderaxespad": 0.25,
            "labelspacing": 0.25,
            "columnspacing": 0.85,
        }
    )


@dataclass(frozen=True)
class BackgroundComparisonLayout:
    overlay_background: bool = True
    panel_b_only: bool = False
    suppress_titles: bool = True

    @property
    def show_panel_labels(self) -> bool:
        return not self.panel_b_only


@dataclass(frozen=True)
class ModelSpecSelection:
    config_name: str
    config: Mapping[str, Any]
    specs: tuple[dict[str, str], ...]
    primary_model: str
    available_configs: tuple[str, ...]


@dataclass
class BackgroundComparisonContext:
    selection: ModelSpecSelection
    model_registry: OrderedDict[str, dict[str, Any]]
    model: Any
    args: Any
    checkpoint_path: Path
    catalog_ds: Any
    catalog_registry_name: str
    catalog_init_kwargs: Mapping[str, Any]
    metadata: Mapping[str, Any]
    magnitude_completeness: float
    full_sequence: Any
    sequences: Sequence[Any]


@dataclass(frozen=True)
class SlidingCacheSelection:
    path: Path
    contract_name: str
    predict_b: bool | None
    sampling_seed: int | None
    metadata: Mapping[str, Any]


INJECTION_YLABEL = r"Injection rate (m$^3$/min)"
BACKGROUND_YLABEL = r"Background intensity ($\mathrm{day}^{-1}$)"
SEISMICITY_YLABEL = r"Seismicity rate ($\mathrm{day}^{-1}$)"


def filename_token(value: Any, *, lowercase: bool = False) -> str:
    """Return a filesystem-safe token without obscuring configuration names."""

    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    )
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_") or "item"
    return token.lower() if lowercase else token


def comparison_output_stem(
    kind: str,
    spec_config_name: str,
    *,
    layout: BackgroundComparisonLayout | None = None,
    primary_model: str | None = None,
) -> str:
    """Build concise names that always identify the selected spec config."""

    parts = [filename_token(kind, lowercase=True), filename_token(spec_config_name)]
    if layout is not None:
        if not layout.overlay_background:
            parts.append("stacked")
        if layout.panel_b_only:
            parts.append("panel")
    if primary_model is not None:
        parts.append(filename_token(primary_model))
    return "_".join(parts)


def load_model_spec_selection(
    config_path: str | Path,
    config_name: str,
    *,
    primary_override: str | None = None,
) -> ModelSpecSelection:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model spec config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config_map = payload.get("configs", {})
    if not isinstance(config_map, dict) or not config_map:
        raise ValueError(f"Invalid config format in {config_path}: missing non-empty 'configs'.")
    if config_name not in config_map:
        available = ", ".join(sorted(config_map))
        raise KeyError(f"Unknown MODEL_SPEC_CONFIG_NAME={config_name}. Available: {available}")

    config = config_map[config_name]
    raw_specs = config.get("models", [])
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Config {config_name!r} has no models in {config_path}.")
    specs: list[dict[str, str]] = []
    for index, item in enumerate(raw_specs):
        label = item.get("label")
        relative_path = item.get("checkpoint_rel_path", item.get("path"))
        if not label:
            raise KeyError(f"Model item {index} in config {config_name!r} must contain 'label'.")
        if not relative_path:
            raise KeyError(
                f"Model item {index} in config {config_name!r} must contain "
                "'checkpoint_rel_path' or 'path'."
            )
        specs.append({"label": str(label), "path": str(relative_path)})

    labels = [spec["label"] for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Config {config_name!r} contains duplicate model labels.")
    primary_model = primary_override or config.get("primary_model_label") or labels[0]
    if primary_model not in labels:
        raise KeyError(
            f"PRIMARY_MODEL_NAME={primary_model!r} is not in config {config_name!r}; "
            f"available={labels}."
        )
    return ModelSpecSelection(
        config_name=str(config_name),
        config=config,
        specs=tuple(specs),
        primary_model=str(primary_model),
        available_configs=tuple(sorted(config_map)),
    )


def resolve_checkpoint_path(path_like: str | Path, project_root: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def build_model_style_map(
    specs: Sequence[Mapping[str, Any]],
    style: BackgroundComparisonStyle,
    overrides: Mapping[Any, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    overrides = dict(overrides or {})
    result: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs):
        label = str(spec["label"])
        model_style = {
            "color": style.model_palette[index % len(style.model_palette)],
            "linestyle": style.line_styles[index % len(style.line_styles)],
            "linewidth": 1.5,
            "alpha": 0.95,
        }
        model_style.update(overrides.get(index, {}))
        model_style.update(overrides.get(label, {}))
        result[label] = model_style
    return result


def configure_publication_style(style: BackgroundComparisonStyle) -> None:
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.legend_size,
            "axes.prop_cycle": plt.cycler(color=style.model_palette),
            "axes.linewidth": 0.75,
            "grid.color": style.grid_color,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "savefig.dpi": style.dpi,
            "savefig.pad_inches": 0.015,
        }
    )


def load_comparison_context(
    selection: ModelSpecSelection,
    *,
    project_root: str | Path,
    data_root: str | Path,
    device: Any,
    compile_models: bool = False,
    model_magnitude_max: float | None = None,
) -> BackgroundComparisonContext:
    project_root = Path(project_root)
    registry: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for spec in selection.specs:
        name = str(spec["label"])
        checkpoint_path = resolve_checkpoint_path(spec["path"], project_root)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {checkpoint_path}")
        model, args = load_and_prepare_model(
            checkpoint_path=checkpoint_path,
            device=device,
            compile=compile_models,
        )
        model = unwrap_compiled_model(model)
        if model_magnitude_max is not None and hasattr(model, "M_m"):
            model.M_m = torch.tensor(float(model_magnitude_max), device=device)
        registry[name] = {
            "model": model,
            "args": args,
            "checkpoint_path": checkpoint_path,
        }

    primary = registry[selection.primary_model]
    args = primary["args"]
    catalog_config = getattr(args, "catalog_cfg", {})
    catalog_ds, registry_name, init_kwargs = load_tpp_catalog(
        args.dataset,
        base_dir=Path(data_root) / args.dataset,
        catalog_cfg=catalog_config,
    )
    metadata = dict(getattr(catalog_ds, "metadata", {}) or {})
    full_sequence = catalog_ds.full_sequence
    sequences = (
        catalog_ds.sequences if hasattr(catalog_ds, "sequences") else [full_sequence]
    )
    return BackgroundComparisonContext(
        selection=selection,
        model_registry=registry,
        model=primary["model"],
        args=args,
        checkpoint_path=primary["checkpoint_path"],
        catalog_ds=catalog_ds,
        catalog_registry_name=registry_name,
        catalog_init_kwargs=init_kwargs,
        metadata=metadata,
        magnitude_completeness=float(metadata["mag_completeness"]),
        full_sequence=full_sequence,
        sequences=sequences,
    )


def model_registry_summary(
    context: BackgroundComparisonContext,
    *,
    branching_t_max: float | None = 10.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, record in context.model_registry.items():
        model = record["model"]
        background = getattr(model, "bg_model", None)
        background_scale = None
        if background is not None and hasattr(background, "_scale"):
            background_scale = float(background._scale.detach().cpu().item())
        branching_ratio = None
        if branching_t_max is not None and hasattr(model, "effective_branching_ratio"):
            value = model.effective_branching_ratio(t_max=float(branching_t_max))
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            branching_ratio = float(value)
        rows.append(
            {
                "model": name,
                "primary": name == context.selection.primary_model,
                "dataset": str(record["args"].dataset),
                "background_scale": background_scale,
                "effective_branching_ratio": branching_ratio,
                "checkpoint": str(record["checkpoint_path"]),
            }
        )
    return pd.DataFrame(rows)


def catalog_summary(context: BackgroundComparisonContext, sequence_index: int) -> pd.DataFrame:
    sequence = context.sequences[int(sequence_index)]
    return pd.DataFrame(
        [
            {
                "spec_config": context.selection.config_name,
                "primary_model": context.selection.primary_model,
                "catalog_registry": context.catalog_registry_name,
                "dataset": str(context.args.dataset),
                "sequence_index": int(sequence_index),
                "events": len(sequence),
                "magnitude_completeness": context.magnitude_completeness,
                "start_ts": context.metadata.get("start_ts"),
            }
        ]
    )


def catalog_time_label(metadata: Mapping[str, Any]) -> str:
    start_timestamp = metadata.get("start_ts", "start")
    if hasattr(start_timestamp, "strftime"):
        start_timestamp = start_timestamp.strftime("%Y-%m-%d")
    return f"Days since {start_timestamp}"


def _model_style(
    label: str,
    index: int,
    model_style_map: Mapping[str, Mapping[str, Any]],
    style: BackgroundComparisonStyle,
) -> dict[str, Any]:
    configured = model_style_map.get(str(label))
    if configured is not None:
        return dict(configured)
    return {
        "color": style.model_palette[index % len(style.model_palette)],
        "linestyle": style.line_styles[index % len(style.line_styles)],
        "linewidth": 1.5,
        "alpha": 0.95,
    }


def _panel_label(index: int) -> str:
    n = int(index) + 1
    chars: list[str] = []
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return f"({''.join(reversed(chars))})"


def _add_panel_label(axis, index: int, *, fontsize: float = 10.0) -> None:
    axis.text(
        0.014,
        0.965,
        _panel_label(index),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.45},
        zorder=20,
    )


def _clean_axis(
    axis,
    style: BackgroundComparisonStyle,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    suppress_titles: bool = True,
) -> None:
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel, labelpad=8)
    axis.xaxis.label.set_clip_on(False)
    axis.yaxis.label.set_clip_on(False)
    if title is not None and not suppress_titles:
        axis.set_title(title)
    axis.grid(True, color=style.grid_color, alpha=0.18, linestyle="--", linewidth=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(style.spine_color)
        axis.spines[side].set_linewidth(0.75)
    axis.tick_params(
        axis="both",
        direction="out",
        length=2.6,
        width=0.7,
        color=style.spine_color,
        pad=2,
    )
    axis.margins(x=0.01)


def _style_figure(
    figure,
    style: BackgroundComparisonStyle,
    *,
    suppress_titles: bool,
) -> None:
    try:
        layout_engine = figure.get_layout_engine()
        if layout_engine is not None and hasattr(layout_engine, "set"):
            layout_engine.set(w_pad=0.08, h_pad=0.035, wspace=0.02, hspace=0.04)
    except Exception:
        pass
    for axis in figure.axes:
        if not axis.has_data():
            continue
        if suppress_titles:
            axis.set_title("")
        axis.grid(
            True,
            color=style.grid_color,
            alpha=0.18,
            linestyle="--",
            linewidth=0.45,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(style.spine_color)
            axis.spines[side].set_linewidth(0.75)
        axis.tick_params(
            axis="both",
            direction="out",
            length=2.6,
            width=0.7,
            color=style.spine_color,
            pad=2,
        )
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            if hasattr(legend, "set_alignment"):
                legend.set_alignment("left")
    try:
        figure.align_ylabels(
            [axis for axis in figure.axes if axis.get_visible() and axis.has_data()]
        )
    except Exception:
        pass
    figure.canvas.draw()


def save_figure_bundle(
    figure,
    output_dir: str | Path,
    stem: str,
    *,
    style: BackgroundComparisonStyle,
    suppress_titles: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _style_figure(figure, style, suppress_titles=suppress_titles)
    return {
        file_format: save_pub_figure(
            figure,
            output_dir / stem,
            dpi=style.dpi,
            file_format=file_format,
        )
        for file_format in style.figure_formats
    }


def _background_records_from_batch(
    model_registry: Mapping[str, Mapping[str, Any]],
    batch: Any,
    model_style_map: Mapping[str, Mapping[str, Any]],
    style: BackgroundComparisonStyle,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (name, record) in enumerate(model_registry.items()):
        model = record["model"]
        background = getattr(model, "bg_model", None)
        if background is None:
            continue
        intensity = background.intensity_trajectory(batch).reshape(-1)
        records.append(
            {
                "model": name,
                "intensity": intensity.detach().cpu().numpy(),
                "style": _model_style(name, index, model_style_map, style),
            }
        )
    if not records:
        raise RuntimeError("No model with bg_model was found in the model registry.")
    return records


def prepare_full_background_data(
    context: BackgroundComparisonContext,
    *,
    sequence_index: int,
    device: Any,
    model_style_map: Mapping[str, Mapping[str, Any]],
    style: BackgroundComparisonStyle,
) -> dict[str, Any]:
    sequence = context.sequences[int(sequence_index)]
    batch = Batch.from_list([sequence]).to(device)
    times = batch.time_series_times.reshape(-1)
    start = float(times.min().item()) if times.numel() else float(sequence.t_start)
    end = float(times.max().item()) if times.numel() else float(sequence.t_end)
    if end <= start:
        end = start + 1e-6
    return {
        "sequence": sequence,
        "batch": batch,
        "time_values": batch.time_series_times.squeeze().detach().cpu().numpy(),
        "injection_values": batch.time_series.squeeze().detach().cpu().numpy(),
        "background_records": _background_records_from_batch(
            context.model_registry,
            batch,
            model_style_map,
            style,
        ),
        "seismicity_sequence": sequence,
        "seismicity_start": start,
        "seismicity_end": end,
        "display_start": start,
        "display_end": end,
        "xlabel": catalog_time_label(context.metadata),
    }


def focused_window_end(
    window_times: np.ndarray,
    injection_times: np.ndarray,
    injection_values: np.ndarray,
    background_matrix: np.ndarray,
    *,
    window_start: float,
    window_end: float,
) -> float:
    """Trim an inactive tail while preserving a small amount of context."""

    window_times = np.asarray(window_times, dtype=float)
    if window_times.size == 0:
        raise ValueError("window_times must contain at least one point.")
    injection_values = np.asarray(injection_values, dtype=float).reshape(-1)
    background_matrix = np.asarray(background_matrix, dtype=float)
    injection_peak = float(np.nanmax(np.abs(injection_values))) if injection_values.size else 0.0
    background_peak = (
        float(np.nanmax(np.abs(background_matrix))) if background_matrix.size else 0.0
    )
    activity = np.zeros(window_times.shape, dtype=bool)
    if np.asarray(injection_times).size and injection_values.size:
        interpolated = np.interp(
            window_times,
            np.asarray(injection_times, dtype=float),
            injection_values,
            left=0.0,
            right=0.0,
        )
        activity |= np.abs(interpolated) > max(1e-8, injection_peak * 1e-3)
    if background_matrix.size:
        activity |= np.max(np.abs(background_matrix), axis=0) > max(
            1e-8,
            background_peak * 1e-3,
        )
    if not np.any(activity):
        return float(window_end)
    last_active = float(window_times[int(np.flatnonzero(activity)[-1])])
    padding = max(0.15, 0.05 * (window_end - window_start))
    return max(float(window_start) + 1e-6, min(float(window_end), last_active + padding))


def prepare_window_background_data(
    context: BackgroundComparisonContext,
    sequence: Any,
    *,
    window_start: float,
    window_end: float,
    model_style_map: Mapping[str, Mapping[str, Any]],
    style: BackgroundComparisonStyle,
) -> dict[str, Any]:
    past_sequence = sequence.get_subsequence(0, window_start, reset_t_nll_to_end=True)
    observed_sequence = sequence.get_subsequence(
        window_start,
        window_end,
        reset_t_nll_to_end=True,
    )
    times = sequence.time_series_times
    mask = (times >= window_start) & (times < window_end)
    window_times = times[mask]
    if window_times.numel() == 0:
        raise ValueError(f"No time-series points in window [{window_start}, {window_end}).")
    injection_times = sequence.time_series_times[mask].detach().cpu().numpy()
    injection_values = sequence.time_series[mask].reshape(-1).detach().cpu().numpy()
    window_times_np = window_times.detach().cpu().numpy()

    background_records: list[dict[str, Any]] = []
    count_records: list[dict[str, Any]] = []
    for index, (name, record) in enumerate(context.model_registry.items()):
        background = getattr(record["model"], "bg_model", None)
        if background is None:
            continue
        background.cache_batch(
            time_series=sequence.time_series.unsqueeze(0),
            time_series_times=sequence.time_series_times.unsqueeze(0),
        )
        intensity = background.intensity(
            ts_batch=background.ts_batch_cache,
            t_query=window_times.unsqueeze(0),
        ).reshape(-1)
        background_records.append(
            {
                "model": name,
                "intensity": intensity.detach().cpu().numpy(),
                "style": _model_style(name, index, model_style_map, style),
            }
        )
        forecast_count = background.forecast_count(window_start, window_end)
        count_records.append(
            {
                "model": name,
                "background_forecast_count": float(forecast_count.detach().cpu().item()),
            }
        )
    if not background_records:
        raise RuntimeError("No model with bg_model was available for window comparison.")
    background_matrix = np.vstack(
        [record["intensity"] for record in background_records]
    )
    display_end = focused_window_end(
        window_times_np,
        injection_times,
        injection_values,
        background_matrix,
        window_start=window_start,
        window_end=window_end,
    )
    return {
        "sequence": sequence,
        "past_sequence": past_sequence,
        "observed_sequence": observed_sequence,
        "time_values": window_times_np,
        "injection_values": injection_values,
        "injection_times": injection_times,
        "background_records": background_records,
        "background_counts": pd.DataFrame(count_records)
        .sort_values("background_forecast_count", ascending=False)
        .reset_index(drop=True),
        "seismicity_sequence": observed_sequence,
        "seismicity_start": float(window_start),
        "seismicity_end": float(window_end),
        "display_start": float(window_start),
        "display_end": display_end,
        "xlabel": catalog_time_label(context.metadata),
    }


def _plot_background_lines(axis, time_values, records) -> list[Any]:
    lines: list[Any] = []
    for index, record in enumerate(records):
        model_style = dict(record.get("style") or {})
        line = axis.plot(
            time_values,
            record["intensity"],
            color=model_style.get("color"),
            linestyle=model_style.get("linestyle", "-"),
            linewidth=model_style.get("linewidth", 1.5),
            alpha=model_style.get("alpha", 0.95),
            solid_capstyle="round",
            dash_capstyle="round",
            label=record["model"],
            zorder=3 + index * 0.01,
        )[0]
        lines.append(line)
    return lines


def plot_background_comparison(
    data: Mapping[str, Any],
    *,
    layout: BackgroundComparisonLayout,
    style: BackgroundComparisonStyle,
    figure_width: float | None = None,
) -> tuple[Any, list[Any]]:
    records = list(data["background_records"])
    background_count = len(records)
    background_panels = 1 if layout.overlay_background else background_count
    figure_width = float(figure_width or style.wide_width)
    if layout.panel_b_only:
        figure, background_axes = plt.subplots(
            background_panels,
            1,
            figsize=(style.full_width, max(1.45 * background_panels, 2.3)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        injection_axis = None
        seismicity_axis = None
        background_axes = background_axes.ravel().tolist()
    else:
        figure, axes = plt.subplots(
            background_panels + 2,
            1,
            figsize=(figure_width, 4.0 + background_panels),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0] * (background_panels + 2)},
        )
        injection_axis = axes[0]
        background_axes = list(axes[1:-1])
        seismicity_axis = axes[-1]
    figure.patch.set_facecolor("white")
    x_limits = (float(data["display_start"]), float(data["display_end"]))

    if injection_axis is not None:
        injection_axis.plot(
            data.get("injection_times", data["time_values"]),
            data["injection_values"],
            linewidth=0.95,
            color=style.injection_color,
            alpha=0.9,
        )
        _clean_axis(
            injection_axis,
            style,
            ylabel=INJECTION_YLABEL,
            xlim=x_limits,
            suppress_titles=layout.suppress_titles,
        )
        if layout.show_panel_labels:
            _add_panel_label(injection_axis, 0)

    if layout.overlay_background:
        background_axis = background_axes[0]
        _plot_background_lines(background_axis, data["time_values"], records)
        _clean_axis(
            background_axis,
            style,
            xlabel=data["xlabel"] if layout.panel_b_only else "",
            xlim=x_limits,
            suppress_titles=layout.suppress_titles,
        )
        background_axis.legend(
            loc="upper right",
            ncol=min(3, background_count),
            **style.legend_kwargs,
        )
        if layout.show_panel_labels:
            _add_panel_label(background_axis, 1)
    else:
        for index, (background_axis, record) in enumerate(zip(background_axes, records)):
            _plot_background_lines(background_axis, data["time_values"], [record])
            _clean_axis(
                background_axis,
                style,
                xlabel=(
                    data["xlabel"]
                    if layout.panel_b_only and index == background_count - 1
                    else ""
                ),
                xlim=x_limits,
                suppress_titles=layout.suppress_titles,
            )
            background_axis.legend(loc="upper right", **style.legend_kwargs)
            if layout.show_panel_labels:
                _add_panel_label(background_axis, 1 + index)

    if seismicity_axis is not None:
        vis.plot_intensity(
            data["seismicity_sequence"],
            dt=1 / 24,
            ax=seismicity_axis,
            color=style.seismicity_color,
            T0=data["seismicity_start"],
            T=data["seismicity_end"],
        )
        _clean_axis(
            seismicity_axis,
            style,
            xlabel=data["xlabel"],
            ylabel=SEISMICITY_YLABEL,
            xlim=x_limits,
            suppress_titles=layout.suppress_titles,
        )
        if layout.show_panel_labels:
            _add_panel_label(seismicity_axis, background_panels + 1)

    target_axis = background_axes[len(background_axes) // 2]
    target_axis.set_ylabel(BACKGROUND_YLABEL, labelpad=8)
    target_axis.yaxis.label.set_clip_on(False)
    _style_figure(figure, style, suppress_titles=layout.suppress_titles)
    return figure, background_axes


def _supports_dynamic_b_sampling(model: Any) -> bool:
    try:
        parameters = inspect.signature(model.sample).parameters
        supported = {"b_sampling", "updater", "updater_name", "updater_cfg"}
        if supported.intersection(parameters):
            return True
    except Exception:
        pass
    return hasattr(model, "_normalize_b_sampling_mode")


def _sampling_model(
    model: Any,
    *,
    magnitude_completeness: float,
    sampling_mode: str,
    updater_name: str,
    updater_config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, str]:
    if sampling_mode != "updater" or not _supports_dynamic_b_sampling(model):
        return model, "model"

    def updater_factory():
        return build_sampling_updater(
            model=model,
            mc=float(magnitude_completeness),
            updater_name=updater_name,
            updater_cfg=dict(updater_config),
            device=getattr(model, "device", device),
        )

    return (
        UpdaterSamplingWrapper(
            model,
            sampling_mode=sampling_mode,
            updater_factory=updater_factory,
        ),
        sampling_mode,
    )


def sample_registry_forecasts(
    context: BackgroundComparisonContext,
    *,
    past_sequence: Any,
    background_cache_sequence: Any,
    duration: float,
    num_samples: int,
    samples_per_batch: int,
    seed: int,
    shared_seed: bool,
    sampling_mode: str,
    updater_name: str,
    updater_config: Mapping[str, Any],
    device: Any,
) -> tuple[OrderedDict[str, list[Any]], pd.DataFrame]:
    past_sequence = past_sequence.to(device) if torch.cuda.is_available() else past_sequence
    forecasts_by_model: OrderedDict[str, list[Any]] = OrderedDict()
    summary_rows: list[dict[str, Any]] = []
    for index, (name, record) in enumerate(context.model_registry.items()):
        sampling_model, resolved_mode = _sampling_model(
            record["model"],
            magnitude_completeness=context.magnitude_completeness,
            sampling_mode=sampling_mode,
            updater_name=updater_name,
            updater_config=updater_config,
            device=device,
        )
        model_seed = int(seed) if shared_seed else int(seed) + index
        forecasts = sample_tpp_forecasts(
            model=sampling_model,
            past_seq=past_sequence,
            duration=duration,
            num_samples=int(num_samples),
            samples_per_batch=int(samples_per_batch),
            bg_cache_seq=background_cache_sequence,
            seed=model_seed,
        )
        forecasts = [
            forecast.cpu() if hasattr(forecast, "cpu") else forecast
            for forecast in forecasts
        ]
        forecasts_by_model[name] = forecasts
        zero_count = sum(
            int((torch.as_tensor(forecast.inter_times).cpu() == 0).any().item())
            for forecast in forecasts
        )
        counts = np.asarray([len(forecast) for forecast in forecasts], dtype=float)
        summary_rows.append(
            {
                "model": name,
                "b_sampling": resolved_mode,
                "seed": model_seed,
                "forecasts": len(forecasts),
                "mean_count": float(counts.mean()) if counts.size else np.nan,
                "median_count": float(np.median(counts)) if counts.size else np.nan,
                "zero_inter_time_forecasts": zero_count,
            }
        )
    return forecasts_by_model, pd.DataFrame(summary_rows)


def _style_existing_panel_labels(figure, *, fontsize: float = 10.0) -> None:
    for axis in figure.axes:
        for text in axis.texts:
            label = text.get_text().strip()
            if label.startswith("(") and label.endswith(")") and label[1:-1].isalpha():
                text.set_position((0.014, 0.965))
                text.set_ha("left")
                text.set_va("top")
                text.set_fontsize(fontsize)
                text.set_fontweight("normal")
                text.set_bbox(
                    {"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.45}
                )
                text.set_zorder(20)


def plot_multi_model_forecasts(
    sequence: Any,
    forecasts_by_model: Mapping[str, Sequence[Any]],
    *,
    xlabel: str,
    style: BackgroundComparisonStyle,
) -> tuple[Any, Any]:
    figure, axes_by_model = vis.visualize_trajectories_multi_model(
        sequence,
        forecasts_by_model,
        save_path=None,
        figsize=(10, 1.8 * len(forecasts_by_model)),
        xlabel=xlabel,
        reset_t_nll_to_end=True,
    )
    for _, (_, right_axis, _) in axes_by_model.items():
        for text in right_axis.texts:
            content = text.get_text()
            if content.startswith("Observed:") and "\n95%:" in content:
                text.set_position((0.52, 0.35))
                text.set_ha("left")
                text.set_va("top")
    _style_existing_panel_labels(figure)
    _style_figure(figure, style, suppress_titles=True)
    return figure, axes_by_model


def plot_primary_catalog_tests(
    forecasts: Sequence[Any],
    observed_sequence: Any,
    *,
    magnitude_completeness: float,
    style: BackgroundComparisonStyle,
    max_magnitude: float = 5.0,
    magnitude_step: float = 0.01,
) -> tuple[dict[str, Any], Any, Any]:
    observed_count = len(observed_sequence)
    number_result = catalog_tests.run_number_test_result(
        event_counts=[len(forecast) for forecast in forecasts],
        obs_count=observed_count,
        min_mw=magnitude_completeness,
        obs_name="observed",
        sim_name="Simulated",
        plot=False,
    )
    number_result.plot(
        plot_args={"figsize": (style.medium_width, 3.2), "tight_layout": False}
    )
    number_figure = style_current_figure(
        title="N-Test: Observed vs Simulated Event Counts"
    )
    number_figure.patch.set_facecolor("white")

    non_empty = [forecast for forecast in forecasts if len(forecast) > 0]
    config = {
        "min_mw": magnitude_completeness,
        "max_mw": max_magnitude,
        "dmw": magnitude_step,
    }
    count_distribution = functools.partial(
        catalog_tests.compute_magnitude_distribution,
        **config,
    )
    result_dict = catalog_tests.magnitude_test_from_counts(
        observed_catalog=observed_sequence,
        forecast_catalogs=forecasts,
        get_counts=lambda catalog: count_distribution(
            np.asarray(catalog.mag.cpu())
        )[1],
        verbose=False,
        debug=False,
    )
    magnitude_result = catalog_tests.run_magnitude_test_result(
        result_dict=result_dict,
        obs_catalog_repr="obs",
        obs_name="observed",
        sim_name="Simulated",
        plot=False,
    )
    magnitude_result.plot(
        plot_args={"figsize": (style.medium_width, 3.2), "tight_layout": False}
    )
    magnitude_figure = style_current_figure(
        title="M-Test: Magnitude Distribution Comparison"
    )
    magnitude_figure.patch.set_facecolor("white")

    count_values = np.asarray([len(forecast) for forecast in forecasts], dtype=float)
    summary = {
        "observed_count": observed_count,
        "forecast_count_mean": float(count_values.mean()) if count_values.size else np.nan,
        "empty_forecasts": len(forecasts) - len(non_empty),
        "non_empty_forecasts": len(non_empty),
    }
    if non_empty:
        summary["forecast_magnitude_min"] = min(
            float(forecast.mag.min().item()) for forecast in non_empty
        )
        summary["forecast_magnitude_max"] = max(
            float(forecast.mag.max().item()) for forecast in non_empty
        )
    if observed_count:
        summary["observed_magnitude_min"] = float(observed_sequence.mag.min().item())
        summary["observed_magnitude_max"] = float(observed_sequence.mag.max().item())
    return summary, number_figure, magnitude_figure


def plot_latent_background(
    model: Any,
    *,
    model_label: str,
    model_style: Mapping[str, Any],
    style: BackgroundComparisonStyle,
    device: Any,
    grid_size: int = 200,
) -> Any | None:
    model.eval()
    grid_device = getattr(model, "device", device)
    grid = torch.linspace(0, 1, int(grid_size), device=grid_device)
    distribution = None
    background = getattr(model, "bg_model", None)
    if background is not None and hasattr(background, "svgp"):
        with torch.no_grad():
            distribution = background.svgp(grid.unsqueeze(-1))
    elif background is not None and hasattr(background, "gp_model"):
        try:
            with torch.no_grad():
                distribution = background.gp_model(grid.unsqueeze(-1))
        except Exception:
            return None
    if distribution is None:
        return None

    mean = distribution.mean.detach().cpu().numpy()
    standard_deviation = np.sqrt(distribution.variance.detach().cpu().numpy())
    x = grid.detach().cpu().numpy().reshape(-1)
    if mean.ndim == 2:
        y = mean[:, 0]
        y_std = standard_deviation[:, 0]
    else:
        y = mean.reshape(-1)
        y_std = standard_deviation.reshape(-1)
    color = model_style.get("color", style.model_palette[0])
    figure, axis = plt.subplots(
        figsize=(style.medium_width, 2.8),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    axis.plot(x, y, color=color, linewidth=1.35, label=f"{model_label} latent mean")
    axis.fill_between(
        x,
        y - 2 * y_std,
        y + 2 * y_std,
        alpha=0.20,
        color=color,
        label="±2 std. dev.",
    )
    _clean_axis(
        axis,
        style,
        xlabel="Normalized time",
        ylabel="Latent value",
        title="Latent background process posterior",
        suppress_titles=False,
    )
    axis.legend(loc="best", **style.legend_kwargs)
    return figure


def resolve_sliding_cache_selection(
    sequence: Any,
    *,
    cache_directories: Sequence[str | Path],
    explicit_filename: str | None,
    duration: float,
    slide_step: float,
    quantiles: tuple[float, float],
    samples_per_batch: int,
    script_predict_b: bool,
    script_sampling_seed: int,
    prefix: str,
) -> SlidingCacheSelection:
    contracts = (
        ("script", script_predict_b, script_sampling_seed),
        ("script_legacy_seed", script_predict_b, None),
        ("notebook_legacy", None, None),
    )
    candidates: list[Path] = []
    for directory_like in cache_directories:
        directory = Path(directory_like)
        if explicit_filename:
            candidates.append(directory / explicit_filename)
        if directory.exists():
            candidates.extend(
                sorted(
                    directory.glob("sliding_window_cache*.npz"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
    unique_candidates = list(dict.fromkeys(candidates))
    for contract_name, predict_b, sampling_seed in contracts:
        metadata = build_sliding_cache_metadata(
            sequence,
            duration=duration,
            slide_step=slide_step,
            quantiles=quantiles,
            samples_per_batch=samples_per_batch,
            predict_b=predict_b,
            sampling_seed=sampling_seed,
        )
        for candidate in unique_candidates:
            if candidate.exists() and load_sliding_window_cache_if_compatible(
                candidate,
                metadata=metadata,
            ) is not None:
                return SlidingCacheSelection(
                    path=candidate,
                    contract_name=contract_name,
                    predict_b=predict_b,
                    sampling_seed=sampling_seed,
                    metadata=metadata,
                )

    contract_name, predict_b, sampling_seed = contracts[0]
    metadata = build_sliding_cache_metadata(
        sequence,
        duration=duration,
        slide_step=slide_step,
        quantiles=quantiles,
        samples_per_batch=samples_per_batch,
        predict_b=predict_b,
        sampling_seed=sampling_seed,
    )
    target_directory = Path(cache_directories[0])
    filename = explicit_filename or build_default_sliding_cache_filename(
        metadata,
        prefix=prefix,
    )
    return SlidingCacheSelection(
        path=target_directory / filename,
        contract_name=contract_name,
        predict_b=predict_b,
        sampling_seed=sampling_seed,
        metadata=metadata,
    )
