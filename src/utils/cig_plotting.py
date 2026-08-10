"""Loading, transforming, and plotting cumulative information-gain curves.

The functions in this module are intentionally notebook-agnostic.  A notebook
should define the experiment configuration and output paths; this module owns
the repeatable data validation, split handling, curve transformation, and
Matplotlib layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.tpp_experiments import load_tpp_catalog


VALID_PLOT_START_MODES = {"val", "test", "full", "nll"}
VALID_BOUNDARY_SOURCES = {"catalog", "checkpoint", "manual"}
_SPLIT_ALIASES = {"val": "val", "validation": "val", "test": "test"}


@dataclass
class CIGPlotConfig:
    """Display options for :func:`plot_cig_grid`."""

    start_mode: str = "val"
    reset_y_at_plot_start: bool = True
    reset_at_split_boundaries: bool = False
    reset_boundary_splits: tuple[str, ...] = ("val", "test")
    model_label_order: Sequence[str] | None = None
    exclude_labels: set[str] = field(default_factory=set)
    model_display_labels: Mapping[str, str] = field(default_factory=dict)
    model_style_map: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    direct_labels: bool = True
    show_legend: bool = False
    direct_label_fontsize: float = 7.2
    direct_label_min_gap_fraction: float = 0.115
    figsize: tuple[float, float] = (13.8, 5.0)
    share_y: bool = False
    ncols: int = 2

    def __post_init__(self) -> None:
        self.start_mode = normalize_plot_start_mode(self.start_mode)
        self.reset_boundary_splits = normalize_reset_splits(self.reset_boundary_splits)
        if self.ncols < 1:
            raise ValueError("ncols must be positive.")

    def display_label(self, label: str) -> str:
        return self.model_display_labels.get(label, str(label).replace("+", "-"))

    def model_style(self, label: str) -> dict[str, Any]:
        return dict(self.model_style_map.get(label, {"linewidth": 1.55}))


def normalize_plot_start_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in VALID_PLOT_START_MODES:
        raise ValueError(
            f"plot start mode must be one of {sorted(VALID_PLOT_START_MODES)}, got {mode!r}."
        )
    return mode


def format_start_label(mode: str) -> str:
    return {
        "val": "validation start",
        "test": "test start",
        "nll": "NLL-window start",
        "full": "curve start",
    }[normalize_plot_start_mode(mode)]


def panel_label(index: int) -> str:
    """Return spreadsheet-style panel labels: ``(a)`` ... ``(aa)``."""

    n = int(index) + 1
    if n < 1:
        raise ValueError("panel index must be non-negative.")
    chars: list[str] = []
    while n:
        n, remainder = divmod(n - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return f"({''.join(reversed(chars))})"


def metadata_boundary_days(metadata: Mapping[str, Any] | None, split: str) -> float | None:
    """Resolve a split boundary as days since catalog start.

    Relative ``*_start_t`` metadata is preferred.  Absolute timestamps are
    used as a fallback when ``start_ts`` and ``freq`` are available.
    """

    if not hasattr(metadata, "get"):
        return None
    split = _SPLIT_ALIASES.get(str(split).lower())
    if split is None:
        raise ValueError("split must be 'val'/'validation' or 'test'.")

    raw_time = metadata.get(f"{split}_start_t")
    frequency = metadata.get("freq")
    if raw_time is not None:
        if frequency is None:
            return float(raw_time)
        try:
            return float(raw_time) * float(pd.Timedelta(frequency) / pd.Timedelta("1D"))
        except Exception:
            return float(raw_time)

    start_timestamp = metadata.get("start_ts")
    boundary_timestamp = metadata.get(f"{split}_start_ts")
    if start_timestamp is None or boundary_timestamp is None or frequency is None:
        return None
    try:
        elapsed = pd.Timestamp(boundary_timestamp) - pd.Timestamp(start_timestamp)
        units = elapsed / pd.Timedelta(frequency)
        return float(units) * float(pd.Timedelta(frequency) / pd.Timedelta("1D"))
    except Exception:
        return None


def resolve_plot_start_days(
    metadata: Mapping[str, Any] | None,
    curve_df: pd.DataFrame,
    mode: str,
    *,
    strict: bool = True,
) -> float:
    """Resolve the left edge for a CIG plot.

    ``full`` and ``nll`` start at the first exported curve point.  ``val`` and
    ``test`` require the corresponding metadata boundary by default.  This
    strict behavior prevents a mislabeled split-relative plot from silently
    falling back to the full curve.
    """

    mode = normalize_plot_start_mode(mode)
    if "time_days" not in curve_df:
        raise KeyError("curve data is missing 'time_days'.")
    curve_start = float(curve_df["time_days"].min())
    curve_end = float(curve_df["time_days"].max())
    if mode in {"full", "nll"}:
        return curve_start

    boundary = metadata_boundary_days(metadata, mode)
    if boundary is None:
        if strict:
            raise ValueError(
                f"Missing {mode!r} split boundary metadata; cannot label this view as {mode!r}."
            )
        return curve_start
    boundary = float(boundary)
    if boundary < curve_start - 1e-9 or boundary > curve_end + 1e-9:
        raise ValueError(
            f"{mode!r} split boundary {boundary:.6f} lies outside the exported curve "
            f"range [{curve_start:.6f}, {curve_end:.6f}]."
        )
    return boundary


def resolve_plot_value_column(
    curve_df: pd.DataFrame,
    curve_path: Path,
    preferred: str,
    fallback: str,
) -> str:
    if preferred in curve_df.columns:
        return preferred
    if fallback in curve_df.columns:
        print(f"Warning: {curve_path.name} missing {preferred!r}; using fallback {fallback!r}.")
        return fallback
    raise KeyError(f"{curve_path} is missing both {preferred!r} and {fallback!r}.")


def _resolve_checkpoint_path(
    item: Mapping[str, Any],
    *,
    project_root: Path,
    model_spec_path: Path,
    model_label: str,
    overrides: Mapping[str, str | Path],
) -> Path:
    override = overrides.get(str(item["dataset"]))
    if override is not None:
        path = Path(override)
        return path if path.is_absolute() else project_root / path

    if not model_spec_path.exists():
        raise FileNotFoundError(f"Model spec file not found: {model_spec_path}")
    with model_spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    spec = payload.get("configs", {}).get(item["config_name"])
    if not isinstance(spec, dict) or not spec.get("models"):
        raise KeyError(f"No model list for {item['config_name']!r} in {model_spec_path}")
    selected = next(
        (model for model in spec["models"] if model.get("label") == model_label),
        None,
    )
    if selected is None:
        available = [model.get("label") for model in spec["models"]]
        raise KeyError(
            f"Model {model_label!r} is not configured for {item['config_name']!r}; "
            f"available={available}."
        )
    relative_path = selected.get("checkpoint_rel_path")
    if not relative_path:
        raise KeyError(
            f"Model {model_label!r} for {item['dataset']} has no checkpoint_rel_path."
        )
    path = Path(relative_path)
    return path if path.is_absolute() else project_root / path


def _checkpoint_catalog_config(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyperparameters = (
        checkpoint.get("hyperparameters", {}) if isinstance(checkpoint, dict) else {}
    )
    catalog_config = (
        hyperparameters.get("catalog_cfg", {}) if isinstance(hyperparameters, dict) else {}
    )
    if not isinstance(catalog_config, dict):
        raise TypeError(
            f"Checkpoint catalog_cfg must be a dict, got {type(catalog_config).__name__}."
        )
    return dict(catalog_config)


def load_cig_datasets(
    dataset_configs: Sequence[Mapping[str, Any]],
    *,
    project_root: str | Path,
    metric_export_dir: str | Path,
    split: str,
    sequence_index: int,
    plot_start_mode: str,
    boundary_source: str,
    checkpoint_model_spec_path: str | Path,
    checkpoint_model_label: str,
    checkpoint_path_overrides: Mapping[str, str | Path] | None = None,
    manual_split_boundaries: Mapping[str, Mapping[str, Any]] | None = None,
    preferred_value_column: str = "cum_log_likelihood_delta",
    fallback_value_column: str = "plot_value",
    expected_labels: Iterable[str] = (),
    excluded_labels: Iterable[str] = (),
    display_labels: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, list[str]]]:
    """Load validated CIG exports and matching split metadata."""

    project_root = Path(project_root)
    metric_export_dir = Path(metric_export_dir)
    model_spec_path = Path(checkpoint_model_spec_path)
    boundary_source = str(boundary_source).lower()
    if boundary_source not in VALID_BOUNDARY_SOURCES:
        raise ValueError(f"boundary_source must be one of {sorted(VALID_BOUNDARY_SOURCES)}.")
    plot_start_mode = normalize_plot_start_mode(plot_start_mode)
    overrides = dict(checkpoint_path_overrides or {})
    manual_boundaries = dict(manual_split_boundaries or {})
    display_labels = dict(display_labels or {})

    loaded: list[dict[str, Any]] = []
    for configured_item in dataset_configs:
        item = dict(configured_item)
        dataset = str(item["dataset"])
        curve_path = metric_export_dir / (
            f"cum_log_likelihood_compare_{dataset}_{split}_{sequence_index}_curves.csv"
        )
        if not curve_path.exists():
            raise FileNotFoundError(
                f"Missing curve CSV for {dataset}: {curve_path}\n"
                "Run notebooks/cumulative_likelihood_over_time.ipynb for this dataset first."
            )
        curve_df = pd.read_csv(curve_path)
        if curve_df.empty:
            raise ValueError(f"Curve CSV is empty: {curve_path}")
        value_column = resolve_plot_value_column(
            curve_df, curve_path, preferred_value_column, fallback_value_column
        )
        required_columns = {"label", "dataset", "time_days", value_column}
        missing_columns = sorted(required_columns - set(curve_df.columns))
        if missing_columns:
            raise KeyError(f"{curve_path} is missing required columns: {missing_columns}")
        curve_df["label"] = curve_df["label"].replace({"Oracle": "ORACLE"})

        checkpoint_path: Path | None = None
        catalog_config: dict[str, Any] = {}
        if boundary_source == "checkpoint":
            checkpoint_path = _resolve_checkpoint_path(
                item,
                project_root=project_root,
                model_spec_path=model_spec_path,
                model_label=checkpoint_model_label,
                overrides=overrides,
            )
            catalog_config = _checkpoint_catalog_config(checkpoint_path)

        catalog, registry_name, _ = load_tpp_catalog(
            dataset,
            base_dir=project_root / "data" / dataset,
            catalog_cfg=catalog_config,
        )
        metadata = dict(getattr(catalog, "metadata", {}) or {})
        if boundary_source == "manual":
            manual = manual_boundaries.get(dataset)
            if not isinstance(manual, Mapping):
                raise KeyError(f"manual_split_boundaries must define {dataset!r}.")
            allowed = {
                "train_start_t", "val_start_t", "test_start_t",
                "train_start_ts", "val_start_ts", "test_start_ts",
            }
            unknown = set(manual) - allowed
            if unknown:
                raise KeyError(f"Unsupported manual boundary keys for {dataset}: {sorted(unknown)}")
            metadata.update(manual)

        val_start = metadata_boundary_days(metadata, "val")
        test_start = metadata_boundary_days(metadata, "test")
        start_days = resolve_plot_start_days(metadata, curve_df, plot_start_mode)
        loaded.append(
            {
                **item,
                "curve_path": curve_path,
                "curve_df": curve_df,
                "plot_value_column": value_column,
                "meta": metadata,
                "registry_name": registry_name,
                "catalog_cfg": catalog_config,
                "checkpoint_path": checkpoint_path,
                "val_start_days": val_start,
                "test_start_days": test_start,
                "plot_start_days": start_days,
            }
        )

    expected = set(expected_labels)
    excluded = set(excluded_labels)
    missing_by_dataset: dict[str, list[str]] = {}
    summary_rows: list[dict[str, Any]] = []
    for item in loaded:
        curve_df = item["curve_df"]
        available = set(curve_df["label"].dropna().unique()) - excluded
        missing = sorted(expected - available)
        if missing:
            missing_by_dataset[item["dataset"]] = missing
        summary_rows.append(
            {
                "dataset": item["dataset"],
                "title": item["title"],
                "registry": item["registry_name"],
                "rows": len(curve_df),
                "labels": ", ".join(
                    display_labels.get(label, str(label).replace("+", "-"))
                    for label in curve_df["label"].drop_duplicates()
                ),
                "min_days": float(curve_df["time_days"].min()),
                "max_days": float(curve_df["time_days"].max()),
                "val_start_days": item["val_start_days"],
                "test_start_days": item["test_start_days"],
                "plot_start_days": item["plot_start_days"],
                "split_boundary_source": boundary_source,
                "checkpoint_path": (
                    str(item["checkpoint_path"]) if item["checkpoint_path"] else None
                ),
                "plot_value_column": item["plot_value_column"],
                "curve_path": str(item["curve_path"].relative_to(project_root)),
            }
        )
    return loaded, pd.DataFrame(summary_rows), missing_by_dataset


def normalize_reset_splits(split_names: Iterable[str]) -> tuple[str, ...]:
    if isinstance(split_names, str):
        split_names = (split_names,)
    normalized: list[str] = []
    for split_name in split_names:
        key = _SPLIT_ALIASES.get(str(split_name).lower())
        if key is None:
            raise ValueError(f"Unsupported CIG reset boundary split: {split_name}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def reset_boundary_items(item: Mapping[str, Any], config: CIGPlotConfig) -> list[tuple[str, float]]:
    if not config.reset_at_split_boundaries:
        return []
    boundary_map = {"val": item.get("val_start_days"), "test": item.get("test_start_days")}
    return [
        (split_name, float(boundary_map[split_name]))
        for split_name in config.reset_boundary_splits
        if boundary_map.get(split_name) is not None and np.isfinite(boundary_map[split_name])
    ]


def _step_value_at_or_before(times: np.ndarray, values: np.ndarray, query_time: float) -> float:
    if times.size == 0:
        return 0.0
    index = np.searchsorted(times, float(query_time), side="right") - 1
    return float(values[int(np.clip(index, 0, times.size - 1))])


def _insert_step_points(
    times: np.ndarray,
    values: np.ndarray,
    query_times: Iterable[float],
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).copy()
    values = np.asarray(values, dtype=float).copy()
    if times.size == 0:
        return times, values
    if np.any(np.diff(times) < -atol):
        raise ValueError("time_days must be sorted before inserting step points.")
    valid_query_times = (
        float(value)
        for value in query_times
        if value is not None and np.isfinite(value)
    )
    for query in sorted(valid_query_times):
        if query < times[0] - atol or query > times[-1] + atol:
            continue
        close_index = np.flatnonzero(np.isclose(times, query, rtol=0.0, atol=atol))
        if close_index.size:
            times[close_index[0]] = query
            continue
        insertion_index = int(np.searchsorted(times, query, side="right"))
        step_value = _step_value_at_or_before(times, values, query)
        times = np.insert(times, insertion_index, query)
        values = np.insert(values, insertion_index, step_value)
    return times, values


def prepare_label_curve(
    curve_df: pd.DataFrame,
    label: str,
    start_days: float,
    *,
    value_column: str,
    reset_y: bool = True,
    split_reset_items: Sequence[tuple[str, float]] = (),
) -> dict[str, np.ndarray | str] | None:
    """Crop one right-continuous step curve and apply display-only resets."""

    subset = curve_df[curve_df["label"] == label].sort_values("time_days")
    if subset.empty:
        return None
    times = subset["time_days"].to_numpy(dtype=float)
    values = subset[value_column].to_numpy(dtype=float)
    start_days = float(start_days)
    split_reset_items = tuple(
        (split_name, float(boundary))
        for split_name, boundary in split_reset_items
        if boundary is not None and np.isfinite(boundary)
    )
    times, values = _insert_step_points(
        times, values, [start_days, *(boundary for _, boundary in split_reset_items)]
    )
    value_at_start = _step_value_at_or_before(times, values, start_days)
    mask = times >= start_days - 1e-10
    cropped_times = times[mask]
    source_values = values[mask]
    if cropped_times.size == 0:
        raise ValueError(
            f"plot start {start_days} is after the final curve point for label {label!r}."
        )
    if cropped_times[0] > start_days:
        cropped_times = np.concatenate([[start_days], cropped_times])
        source_values = np.concatenate([[value_at_start], source_values])
    elif cropped_times[0] < start_days:
        cropped_times[0] = start_days

    offset = np.full_like(source_values, value_at_start if reset_y else 0.0, dtype=float)
    segment_start_days = np.full_like(source_values, start_days if reset_y else np.nan, dtype=float)
    for _, boundary in sorted(split_reset_items, key=lambda pair: pair[1]):
        if boundary < start_days - 1e-10 or boundary > cropped_times[-1] + 1e-10:
            continue
        boundary_value = _step_value_at_or_before(times, values, boundary)
        boundary_mask = cropped_times >= boundary - 1e-10
        offset[boundary_mask] = boundary_value
        segment_start_days[boundary_mask] = boundary

    return {
        "absolute_days": cropped_times,
        "days_since_plot_start": cropped_times - start_days,
        "source_value": source_values,
        "plot_value": source_values - offset,
        "offset": offset,
        "segment_start_days": segment_start_days,
        "value_column": value_column,
    }


def ordered_labels(curve_df: pd.DataFrame, config: CIGPlotConfig) -> list[str]:
    labels = [
        label for label in curve_df["label"].drop_duplicates().tolist()
        if label not in config.exclude_labels
    ]
    if config.model_label_order is None:
        return labels
    ordered = [label for label in config.model_label_order if label in labels]
    ordered.extend(label for label in labels if label not in ordered)
    return ordered


def add_direct_curve_labels(
    ax,
    label_curve_map: Mapping[str, tuple[np.ndarray, np.ndarray, Any]],
    *,
    display_label,
    fontsize: float,
    minimum_gap_fraction: float,
) -> None:
    if not label_curve_map:
        return
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    endpoints = [
        [label, float(x[-1]), float(y[-1]), line]
        for label, (x, y, line) in label_curve_map.items()
        if len(x)
    ]
    if not endpoints:
        return
    endpoints.sort(key=lambda endpoint: endpoint[2])
    label_low = y_min + 0.07 * y_span
    label_high = y_max - 0.07 * y_span
    available = max(label_high - label_low, 1e-12)
    minimum_gap = min(minimum_gap_fraction * y_span, available / max(1, len(endpoints) - 1))
    target_y = np.clip(np.array([endpoint[2] for endpoint in endpoints]), label_low, label_high)
    for index in range(1, len(target_y)):
        target_y[index] = max(target_y[index], target_y[index - 1] + minimum_gap)
    if target_y[-1] > label_high:
        target_y[-1] = label_high
        for index in range(len(target_y) - 2, -1, -1):
            target_y[index] = min(target_y[index], target_y[index + 1] - minimum_gap)
    tolerance = 1e-9 * y_span
    if target_y[0] < label_low - tolerance or np.any(np.diff(target_y) < minimum_gap - tolerance):
        raise RuntimeError("Direct-label layout could not satisfy the requested minimum spacing.")

    ax.set_xlim(x_min, x_max + 0.26 * x_span)
    text_x = x_max + 0.035 * x_span
    for (label, x_end, y_end, line), text_y in zip(endpoints, target_y):
        color = line.get_color()
        ax.plot(
            [x_end, text_x - 0.01 * x_span], [y_end, float(text_y)],
            color=color, linewidth=0.55, alpha=0.65, clip_on=False,
        )
        ax.text(
            text_x, float(text_y), display_label(label), color=color, fontsize=fontsize,
            va="center", ha="left", clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.15},
        )


def draw_split_markers(ax, item: Mapping[str, Any], *, label_y: float = 0.985) -> None:
    start = float(item["plot_start_days"])
    x_min, x_max = ax.get_xlim()
    split_boundaries = (
        ("Val", item.get("val_start_days")),
        ("Test", item.get("test_start_days")),
    )
    visible_boundaries = []
    for label, boundary in split_boundaries:
        if boundary is None:
            continue
        x = float(boundary) - start
        if x <= 1e-9 or x >= x_max:
            continue
        visible_boundaries.append((label, x))

    stagger_labels = (
        len(visible_boundaries) == 2
        and abs(visible_boundaries[1][1] - visible_boundaries[0][1])
        < 0.065 * max(x_max - x_min, 1e-12)
    )
    for index, (label, x) in enumerate(visible_boundaries):
        text_y = label_y - 0.11 * index if stagger_labels else label_y
        ax.axvline(x, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)
        ax.text(
            x, text_y, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
            zorder=4,
        )


def plot_cig_grid(
    items: Sequence[Mapping[str, Any]],
    *,
    config: CIGPlotConfig,
    phase_annotations: Mapping[str, Mapping[str, Any]] | None = None,
    phase_style=None,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Plot a grid of CIG curves and return the plotted long-form data."""

    from src.utils.injection_phases import DEFAULT_PHASE_RIBBON_STYLE, draw_phase_ribbon

    phase_annotations = phase_annotations or {}
    phase_style = phase_style or DEFAULT_PHASE_RIBBON_STYLE
    nrows = int(np.ceil(len(items) / config.ncols))
    fig, axes = plt.subplots(
        nrows, config.ncols, figsize=config.figsize, sharey=config.share_y, squeeze=False
    )
    flat_axes = axes.ravel()
    handles_by_label: dict[str, Any] = {}
    exported_rows: list[pd.DataFrame] = []

    for panel_index, (ax, item) in enumerate(zip(flat_axes, items)):
        curve_df = item["curve_df"]
        phase_annotation = phase_annotations.get(item["dataset"])
        label_curve_map: dict[str, tuple[np.ndarray, np.ndarray, Any]] = {}
        for label in ordered_labels(curve_df, config):
            prepared = prepare_label_curve(
                curve_df,
                label,
                item["plot_start_days"],
                value_column=item["plot_value_column"],
                reset_y=config.reset_y_at_plot_start,
                split_reset_items=reset_boundary_items(item, config),
            )
            if prepared is None:
                continue
            x = np.asarray(prepared["days_since_plot_start"])
            y = np.asarray(prepared["plot_value"])
            line = ax.step(x, y, where="post", label=label, **config.model_style(label))[0]
            handles_by_label.setdefault(label, line)
            label_curve_map[label] = (x, y, line)
            exported_rows.append(
                pd.DataFrame(
                    {
                        "dataset": item["dataset"],
                        "title": item["title"],
                        "label": label,
                        "absolute_days": prepared["absolute_days"],
                        "days_since_plot_start": prepared["days_since_plot_start"],
                        "source_value": prepared["source_value"],
                        "plot_value": prepared["plot_value"],
                        "offset": prepared["offset"],
                        "segment_start_days": prepared["segment_start_days"],
                    }
                )
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        title_padding = phase_style.title_padding if phase_annotation else 7
        annotation_y = phase_style.aux_label_y if phase_annotation else 0.99
        ax.set_title(item["title"], fontsize=11, pad=title_padding)
        ax.text(
            0.01, annotation_y, panel_label(panel_index), transform=ax.transAxes,
            va="top", ha="left", fontsize=11,
        )
        ax.set_xlabel(
            f"Days since {format_start_label(config.start_mode)}",
            fontsize=10,
            labelpad=phase_style.xlabel_padding,
        )
        ylabel = "CIG vs ETAS"
        if config.reset_at_split_boundaries:
            ylabel += " (split reset)"
        elif not config.reset_y_at_plot_start:
            ylabel += " (cumulative)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if config.direct_labels:
            add_direct_curve_labels(
                ax,
                label_curve_map,
                display_label=config.display_label,
                fontsize=config.direct_label_fontsize,
                minimum_gap_fraction=config.direct_label_min_gap_fraction,
            )
        draw_phase_ribbon(ax, item, phase_annotation, style=phase_style)
        draw_split_markers(ax, item, label_y=annotation_y)

    for ax in flat_axes[len(items):]:
        ax.axis("off")
    if config.show_legend:
        legend_labels = [
            label for label in (config.model_label_order or list(handles_by_label))
            if label in handles_by_label
        ]
        legend_labels.extend(label for label in handles_by_label if label not in legend_labels)
        fig.legend(
            [handles_by_label[label] for label in legend_labels],
            [config.display_label(label) for label in legend_labels],
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=max(1, len(legend_labels)), fontsize=9, frameon=False,
            columnspacing=1.0, handlelength=2.3,
        )
    bottom_margin = 0.09 if config.show_legend else 0.04
    fig.tight_layout(rect=(0.02, bottom_margin, 0.965, 0.96))
    plotted_data = pd.concat(exported_rows, ignore_index=True) if exported_rows else pd.DataFrame()
    return fig, axes, plotted_data


def cig_figure_stem(config: CIGPlotConfig) -> str:
    """Build a concise stem, appending suffixes only for non-default behavior."""

    stem = f"cig_from_{config.start_mode}"
    if not config.reset_y_at_plot_start:
        stem += "_absolute"
    if config.reset_at_split_boundaries:
        reset_splits = "-".join(config.reset_boundary_splits)
        stem += f"_reset_{reset_splits}"
    return stem
