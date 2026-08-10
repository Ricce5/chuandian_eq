"""Detect and render injection-operation phases on a shared absolute timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.textpath import TextPath

from src.utils.injection_data import extract_injection_count_series
from src.utils.tpp_experiments import load_tpp_catalog


@dataclass(frozen=True)
class PhaseRibbonStyle:
    band_ymin: float = 1.025
    band_ymax: float = 1.105
    active_band_color: str = "#dcece8"
    intermittent_band_color: str = "#edf3f1"
    post_band_color: str = "#eeeeee"
    band_edge_color: str = "#ffffff"
    band_linewidth: float = 0.45
    band_alpha: float = 0.90
    aux_label_y: float = 0.985
    title_padding: float = 30.0
    xlabel_padding: float = 4.0
    text_color: str = "#46514f"
    fontsize: float = 6.3
    label_padding_points: float = 1.5
    short_label_min_width_fraction: float = 0.035
    boundary_color: str = "#c4c4c4"
    boundary_linestyle: Any = (0, (2.5, 3.0))
    boundary_linewidth: float = 0.70
    boundary_alpha: float = 0.72
    show_boundary_lines: bool = True


DEFAULT_PHASE_RIBBON_STYLE = PhaseRibbonStyle()


# Dataset-specific semantics stay declarative; the detector implementations
# below are generic and independently testable.
DEFAULT_PANEL_PHASE_CONFIGS: dict[str, dict[str, Any]] = {
    "CB_HAB1a": {
        "anchor_key": "curve_start_days",
        "anchor_label": "curve start",
        "detector": "cb1a_full_campaigns",
        "labels": (
            "pre-injection", "early intermittent injection", "shut-in",
            "sustained injection", "intermittent injection", "reinjection",
        ),
        "display_labels": (
            "Pre-injection", "Early intermittent injection", "Shut-in",
            "Sustained injection", "Intermittent injection", "Re-injection",
        ),
        "short_display_labels": (
            "Pre-inj.", "Early intermittent", "Shut-in",
            "Sustained inj.", "Intermittent inj.", "Re-inj.",
        ),
        "active_threshold_fraction": 0.01,
        "main_inactive_min_days": 0.50,
        "late_search_fraction": 0.88,
        "late_window_days": 0.50,
        "late_active_fraction": 0.70,
        "late_mean_fraction": 0.25,
        "boundary_overrides_days_from_anchor": None,
        "boundary_basis": (
            "injection: first active sample above 1% of the full-window peak",
            "injection: start of the longest inactive gap",
            "injection: end of the longest inactive gap",
            "injection: first >=0.50-day inactive interval after main injection begins",
            "injection: first sustained late high-activity cluster",
        ),
    },
    "CB_HAB4": {
        "anchor_key": "curve_start_days",
        "anchor_label": "curve start",
        "detector": "campaign_with_shutin",
        "labels": (
            "pre-injection", "initial injection", "shut-in",
            "sustained injection", "post-injection",
        ),
        "display_labels": (
            "Pre-injection", "Initial injection", "Shut-in",
            "Sustained injection", "Post-injection",
        ),
        "short_display_labels": ("Pre", "Init.", "Shut-in", "Sustained inj.", "Post-inj."),
        "active_threshold_fraction": 0.01,
        "shutin_min_days": 0.50,
        "boundary_overrides_days_from_anchor": None,
        "boundary_basis": (
            "injection: first active sample above 1% of the full-window peak",
            "injection: start of the longest inactive gap",
            "injection: end of the longest inactive gap",
            "injection: first sample after the final active injection sample",
        ),
    },
    "PNR_1z": {
        "anchor_key": "curve_start_days",
        "anchor_label": "curve start",
        "detector": "split_pulse_campaigns",
        "labels": (
            "early pulse sequence", "shut-in", "renewed pulse phase one",
            "renewed pulse phase two", "final pulse", "post-injection",
        ),
        "display_labels": (
            "Early pulse sequence", "Shut-in", "Renewed pulses I",
            "Renewed pulses II", "Final pulse", "Post-injection",
        ),
        "short_display_labels": (
            "Early pulses", "Shut-in", "Pulse I", "Pulse II", "Final", "Post-inj.",
        ),
        "active_threshold_fraction": 0.01,
        "major_shutin_min_days": 10.0,
        "renewed_phase_count": 3,
        "boundary_overrides_days_from_anchor": None,
        "boundary_basis": (
            "injection: start of the longest full-window inactive gap",
            "injection: end of the longest full-window inactive gap",
            "injection: midpoint of the first selected renewed-campaign gap",
            "injection: midpoint of the second selected renewed-campaign gap",
            "injection: first sample after the final active injection sample",
        ),
        "full_view_groups": (
            {
                "labels": (
                    "renewed pulse phase one", "renewed pulse phase two", "final pulse",
                ),
                "display_label": "Renewed pulses",
            },
        ),
    },
    "St1-2018": {
        "anchor_key": "curve_start_days",
        "anchor_label": "curve start",
        "detector": "final_injection_stop",
        "labels": ("pulsed injection", "post-injection"),
        "display_labels": ("Pulsed injection", "Post-injection"),
        "short_display_labels": ("Injection", "Post-inj."),
        "active_threshold_fraction": 0.01,
        "boundary_overrides_days_from_anchor": None,
        "boundary_basis": ("injection: first sample after the final active injection sample",),
    },
}


def load_phase_source_series(item: Mapping[str, Any], *, data_root: str | Path) -> dict[str, Any]:
    dataset = str(item["dataset"])
    catalog, registry_name, _ = load_tpp_catalog(
        dataset,
        base_dir=Path(data_root) / dataset,
        catalog_cfg=item.get("catalog_cfg") or {},
        candidates=[f"{dataset}-Standard", dataset],
    )
    series = extract_injection_count_series(catalog.full_sequence, catalog.metadata)
    return {**series, "registry_name": registry_name}


def _phase_analysis_window(
    item: Mapping[str, Any], series: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if config["anchor_key"] == "curve_start_days":
        anchor_days = float(item["curve_df"]["time_days"].min())
    else:
        anchor_days = item.get(config["anchor_key"])
    if anchor_days is None or not np.isfinite(anchor_days):
        raise ValueError(f"Missing phase anchor {config['anchor_key']!r} for {item['dataset']}.")
    anchor_days = float(anchor_days)
    end_days = float(item["curve_df"]["time_days"].max())
    absolute_days = np.asarray(series["absolute_days"], dtype=float)
    mask = (absolute_days >= anchor_days) & (absolute_days <= end_days)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"No usable phase-analysis samples for {item['dataset']}.")
    return {
        "anchor_days": anchor_days,
        "end_days": end_days,
        "x": absolute_days[mask] - anchor_days,
        "injection": np.asarray(series["injection"], dtype=float)[mask],
        "counts": np.asarray(series["counts"], dtype=float)[mask],
    }


def _active_run_indices(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = np.diff(np.r_[False, np.asarray(active, dtype=bool), False].astype(int))
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1


def _injection_activity(data: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    x = np.asarray(data["x"], dtype=float)
    injection = np.nan_to_num(np.asarray(data["injection"], dtype=float), nan=0.0)
    peak = float(np.nanmax(injection))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("No positive injection values were found.")
    threshold = float(config["active_threshold_fraction"]) * peak
    active = injection > threshold
    run_starts, run_ends = _active_run_indices(active)
    if run_starts.size == 0:
        raise RuntimeError("No active injection samples were found.")
    sample_delta = float(np.nanmedian(np.diff(x)))
    analysis_end = float(data["end_days"] - data["anchor_days"])
    runs = tuple(
        {
            "start": float(x[start]),
            "end": float(min(x[end] + sample_delta, analysis_end)),
            "start_idx": int(start),
            "end_idx": int(end),
        }
        for start, end in zip(run_starts, run_ends)
    )
    gaps = tuple(
        {
            "start": runs[index]["end"],
            "end": runs[index + 1]["start"],
            "duration": runs[index + 1]["start"] - runs[index]["end"],
        }
        for index in range(len(runs) - 1)
    )
    return {
        "x": x,
        "injection": injection,
        "active": active,
        "peak_injection": peak,
        "active_threshold": threshold,
        "sample_dt": sample_delta,
        "analysis_end": analysis_end,
        "runs": runs,
        "gaps": gaps,
    }


def _largest_gap(activity: Mapping[str, Any], minimum_duration: float = 0.0) -> Mapping[str, float]:
    candidates = [gap for gap in activity["gaps"] if gap["duration"] >= float(minimum_duration)]
    if not candidates:
        raise RuntimeError("No inactive gap satisfies the requested duration.")
    return max(candidates, key=lambda gap: gap["duration"])


def _detect_cb1a_full_campaigns(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    activity = _injection_activity(data, config)
    major_gap = _largest_gap(activity)
    later_gaps = [
        gap for gap in activity["gaps"]
        if gap["start"] >= major_gap["end"]
        and gap["duration"] >= config["main_inactive_min_days"]
    ]
    if not later_gaps:
        raise RuntimeError("No sustained inactive interval follows CB1a main injection.")
    main_stop = min(later_gaps, key=lambda gap: gap["start"])["start"]

    window_size = max(2, int(np.ceil(config["late_window_days"] / activity["sample_dt"])))
    if window_size >= activity["x"].size:
        raise RuntimeError("Late-reactivation detection window is longer than the analysis data.")
    active_cumsum = np.r_[0, np.cumsum(activity["active"].astype(float))]
    injection_cumsum = np.r_[0.0, np.cumsum(np.maximum(activity["injection"], 0.0))]
    forward_active_fraction = (
        active_cumsum[window_size:] - active_cumsum[:-window_size]
    ) / window_size
    forward_mean = (
        injection_cumsum[window_size:] - injection_cumsum[:-window_size]
    ) / window_size
    window_x = activity["x"][: forward_active_fraction.size]
    search_start = max(main_stop, config["late_search_fraction"] * activity["analysis_end"])
    qualifies = (
        (window_x >= search_start)
        & (forward_active_fraction >= config["late_active_fraction"])
        & (forward_mean >= config["late_mean_fraction"] * activity["peak_injection"])
    )
    if not np.any(qualifies):
        raise RuntimeError("No sustained late-reactivation cluster was found.")
    qualifying_index = int(np.flatnonzero(qualifies)[0])
    active_in_window = np.flatnonzero(
        activity["active"][qualifying_index : qualifying_index + window_size]
    )
    reactivation = float(activity["x"][qualifying_index + int(active_in_window[0])])
    boundaries = (
        activity["runs"][0]["start"],
        major_gap["start"],
        major_gap["end"],
        float(main_stop),
        reactivation,
    )
    return boundaries, {
        "active_threshold": activity["active_threshold"],
        "peak_injection": activity["peak_injection"],
        "major_inactive_gap_days": major_gap["duration"],
        "late_window_days": config["late_window_days"],
    }


def _detect_campaign_with_shutin(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    activity = _injection_activity(data, config)
    shut_in = _largest_gap(activity, config["shutin_min_days"])
    final_stop = activity["runs"][-1]["end"]
    if final_stop >= activity["analysis_end"]:
        raise RuntimeError("Injection remains active through the end of the analysis window.")
    return (
        activity["runs"][0]["start"], shut_in["start"], shut_in["end"], final_stop,
    ), {
        "active_threshold": activity["active_threshold"],
        "peak_injection": activity["peak_injection"],
        "selected_shutin_days": shut_in["duration"],
    }


def _detect_split_pulse_campaigns(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    activity = _injection_activity(data, config)
    major_shut_in = _largest_gap(activity, config["major_shutin_min_days"])
    renewed_gap_count = int(config["renewed_phase_count"]) - 1
    renewed_gaps = [gap for gap in activity["gaps"] if gap["start"] >= major_shut_in["end"]]
    if len(renewed_gaps) < renewed_gap_count:
        raise RuntimeError("Too few gaps in the renewed pulse campaign.")
    selected = sorted(
        sorted(renewed_gaps, key=lambda gap: gap["duration"], reverse=True)[:renewed_gap_count],
        key=lambda gap: gap["start"],
    )
    renewed_boundaries = tuple(0.5 * (gap["start"] + gap["end"]) for gap in selected)
    final_stop = activity["runs"][-1]["end"]
    if final_stop >= activity["analysis_end"]:
        raise RuntimeError("Injection remains active through the end of the analysis window.")
    return (
        major_shut_in["start"], major_shut_in["end"], *renewed_boundaries, final_stop,
    ), {
        "active_threshold": activity["active_threshold"],
        "peak_injection": activity["peak_injection"],
        "major_shutin_days": major_shut_in["duration"],
        "renewed_selected_gap_days": tuple(gap["duration"] for gap in selected),
    }


def _detect_final_injection_stop(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    activity = _injection_activity(data, config)
    final_stop = activity["runs"][-1]["end"]
    if final_stop >= activity["analysis_end"]:
        raise RuntimeError("Injection remains active through the end of the analysis window.")
    return (final_stop,), {
        "active_threshold": activity["active_threshold"],
        "peak_injection": activity["peak_injection"],
    }


_DETECTORS = {
    "cb1a_full_campaigns": _detect_cb1a_full_campaigns,
    "campaign_with_shutin": _detect_campaign_with_shutin,
    "split_pulse_campaigns": _detect_split_pulse_campaigns,
    "final_injection_stop": _detect_final_injection_stop,
}


def detect_panel_phase_annotation(
    item: Mapping[str, Any],
    *,
    data_root: str | Path,
    phase_configs: Mapping[str, Mapping[str, Any]] = DEFAULT_PANEL_PHASE_CONFIGS,
) -> dict[str, Any] | None:
    config = phase_configs.get(str(item["dataset"]))
    if config is None:
        return None
    series = load_phase_source_series(item, data_root=data_root)
    data = _phase_analysis_window(item, series, config)
    override = config.get("boundary_overrides_days_from_anchor")
    if override is not None:
        boundaries = tuple(float(value) for value in override)
        evidence = {"override": True}
        boundary_basis = tuple("manual override from phase config" for _ in boundaries)
    else:
        detector = _DETECTORS.get(config["detector"])
        if detector is None:
            raise ValueError(f"Unsupported phase detector: {config['detector']}")
        boundaries, evidence = detector(data, config)
        boundary_basis = tuple(config["boundary_basis"])

    boundaries = tuple(float(value) for value in boundaries)
    if len(boundaries) != len(config["labels"]) - 1:
        raise ValueError("Phase detector returned the wrong number of boundaries.")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("Detected phase boundaries must be strictly increasing.")
    absolute_boundaries = tuple(data["anchor_days"] + value for value in boundaries)
    interval_edges = (data["anchor_days"], *absolute_boundaries, data["end_days"])
    intervals = tuple(
        {
            "label": label,
            "display_label": display_label,
            "short_display_label": short_label,
            "start_absolute_days": float(start),
            "end_absolute_days": float(end),
        }
        for label, display_label, short_label, start, end in zip(
            config["labels"],
            config["display_labels"],
            config.get("short_display_labels", config["display_labels"]),
            interval_edges[:-1],
            interval_edges[1:],
        )
    )
    return {
        "dataset": item["dataset"],
        "title": item["title"],
        "anchor_days": data["anchor_days"],
        "anchor_label": config["anchor_label"],
        "boundaries_from_anchor": boundaries,
        "boundary_absolute_days": absolute_boundaries,
        "boundary_basis": boundary_basis,
        "intervals": intervals,
        "evidence": evidence,
        "source": series["source"],
        "registry_name": series["registry_name"],
        "full_view_groups": config.get("full_view_groups", ()),
    }


def build_panel_phase_annotations(
    items: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path,
    phase_configs: Mapping[str, Mapping[str, Any]] = DEFAULT_PANEL_PHASE_CONFIGS,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    annotations: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for item in items:
        annotation = detect_panel_phase_annotation(
            item, data_root=data_root, phase_configs=phase_configs
        )
        if annotation is None:
            continue
        annotations[str(item["dataset"])] = annotation
        print(
            f"{item['title']} phase boundaries since {annotation['anchor_label']}: "
            + ", ".join(f"{value:.3f} d" for value in annotation["boundaries_from_anchor"])
        )
        for index, (relative_days, absolute_days, basis) in enumerate(
            zip(
                annotation["boundaries_from_anchor"],
                annotation["boundary_absolute_days"],
                annotation["boundary_basis"],
            )
        ):
            summary_rows.append(
                {
                    "dataset": item["title"],
                    "from_phase": annotation["intervals"][index]["label"],
                    "to_phase": annotation["intervals"][index + 1]["label"],
                    "days_since_anchor": relative_days,
                    "absolute_catalog_days": absolute_days,
                    "basis": basis,
                    "source": annotation["source"],
                    **annotation["evidence"],
                }
            )
    return annotations, pd.DataFrame(summary_rows)


def visible_phase_intervals(
    item: Mapping[str, Any], annotation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Crop absolute phase intervals to the current plot start/end."""

    plot_start = float(item["plot_start_days"])
    plot_end = float(item["curve_df"]["time_days"].max())
    visible: list[dict[str, Any]] = []
    for interval in annotation["intervals"]:
        visible_start = max(float(interval["start_absolute_days"]), plot_start)
        visible_end = min(float(interval["end_absolute_days"]), plot_end)
        if visible_end <= visible_start:
            continue
        visible.append(
            {
                **interval,
                "left_plot_days": visible_start - plot_start,
                "right_plot_days": visible_end - plot_start,
            }
        )
    return visible


def _phase_label_for_interval(ax, interval: Mapping[str, Any], style: dict[str, Any]) -> str:
    left_pixels = ax.transData.transform((interval["left_plot_days"], 0.0))[0]
    right_pixels = ax.transData.transform((interval["right_plot_days"], 0.0))[0]
    available_points = abs(right_pixels - left_pixels) * 72.0 / ax.figure.dpi
    available_points -= 2.0 * float(style["label_padding_points"])
    candidates = dict.fromkeys(
        [interval["display_label"], interval.get("short_display_label", interval["display_label"])]
    )
    for candidate in candidates:
        width = TextPath((0.0, 0.0), candidate, size=style["fontsize"]).get_extents().width
        if width <= available_points:
            return candidate
    x_min, x_max = ax.get_xlim()
    width_fraction = (
        interval["right_plot_days"] - interval["left_plot_days"]
    ) / max(float(x_max - x_min), 1e-12)
    if width_fraction >= style["short_label_min_width_fraction"]:
        return interval.get("short_display_label", interval["display_label"])
    return ""


def _phase_band_color(interval: Mapping[str, Any], style: Mapping[str, Any]) -> str:
    label = str(interval["label"]).lower()
    if any(token in label for token in ("pre-injection", "shut-in", "post-injection")):
        return style["post_band_color"]
    if "intermittent" in label:
        return style["intermittent_band_color"]
    return style["active_band_color"]


def draw_phase_ribbon(
    ax,
    item: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    *,
    style: PhaseRibbonStyle = DEFAULT_PHASE_RIBBON_STYLE,
) -> list[dict[str, Any]] | None:
    if annotation is None:
        return None
    style_dict = asdict(style)
    visible = visible_phase_intervals(item, annotation)
    if not visible:
        return None

    plot_start = float(item["plot_start_days"])
    data_xmax = max(float(item["curve_df"]["time_days"].max()) - plot_start, 0.0)
    band_y = 0.5 * (style.band_ymin + style.band_ymax)
    is_full_view = np.isclose(plot_start, annotation["anchor_days"], atol=1e-9)
    groups = annotation.get("full_view_groups", ()) if is_full_view else ()
    grouped_labels = {label for group in groups for label in group["labels"]}

    for interval in visible:
        left = interval["left_plot_days"]
        right = interval["right_plot_days"]
        color = _phase_band_color(interval, style_dict)
        ax.axvspan(
            left,
            right,
            ymin=style.band_ymin,
            ymax=style.band_ymax,
            facecolor=color,
            edgecolor=style.band_edge_color,
            linewidth=style.band_linewidth,
            alpha=style.band_alpha,
            clip_on=False,
            zorder=3.2,
        )
        label = "" if interval["label"] in grouped_labels else _phase_label_for_interval(
            ax, interval, style_dict
        )
        if label:
            ax.text(
                0.5 * (left + right),
                band_y,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=style.fontsize,
                color=style.text_color,
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.94, "pad": 0.18},
                clip_on=False,
                zorder=4,
            )

    for group in groups:
        grouped = [interval for interval in visible if interval["label"] in group["labels"]]
        if not grouped:
            continue
        left = min(interval["left_plot_days"] for interval in grouped)
        right = max(interval["right_plot_days"] for interval in grouped)
        color = _phase_band_color(grouped[0], style_dict)
        ax.text(
            0.5 * (left + right),
            band_y,
            group["display_label"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=style.fontsize,
            color=style.text_color,
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.94, "pad": 0.18},
            clip_on=False,
            zorder=4,
        )

    if style.show_boundary_lines:
        for absolute_boundary in annotation["boundary_absolute_days"]:
            x = float(absolute_boundary) - plot_start
            if 0.0 < x < data_xmax:
                ax.axvline(
                    x,
                    color=style.boundary_color,
                    linestyle=style.boundary_linestyle,
                    linewidth=style.boundary_linewidth,
                    alpha=style.boundary_alpha,
                    zorder=1.1,
                )
    return visible
