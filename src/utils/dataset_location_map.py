"""Dataset-location loading and Matplotlib/PyGMT map rendering helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class LocationMapConfig:
    data_root: Path
    use_pygmt: bool = True
    world_shapefile: Path | None = None
    draw_world_polygons: bool = True
    point_limit_per_dataset: int = 300
    location_fallbacks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    projected_crs: Mapping[str, str] = field(default_factory=dict)
    label_offsets: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    show_location_points: bool = True
    show_grid: bool = False
    pygmt_region: Sequence[float] = (-180, 180, -60, 85)
    pygmt_projection: str = "M16c"
    pygmt_title: str | None = None
    pygmt_frame: Sequence[str] = ("xa60f30", "ya30f15")
    pygmt_export_margin: str = "0.35c/0.14c/0.32c/0.06c"
    pygmt_display_width: int = 780
    pygmt_colors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        if self.world_shapefile is not None:
            self.world_shapefile = Path(self.world_shapefile)


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _pick_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {_normalize_column_name(column): column for column in dataframe.columns}
    for candidate in candidates:
        column = normalized.get(_normalize_column_name(candidate))
        if column is not None:
            return column
    return None


def dataset_location_files(dataset: str, data_root: str | Path) -> list[Path]:
    base = Path(data_root) / dataset
    return [
        base / "processed" / f"{dataset}_eq_processed.csv",
        base / "raw" / f"{dataset}_catalog.csv",
        base / "raw" / f"{dataset}_Eq.csv",
        base / "raw" / f"{dataset}_Traj.csv",
    ]


def lonlat_from_columns(
    dataframe: pd.DataFrame,
    *,
    dataset: str,
    source: str,
    projected_crs: Mapping[str, str] | None = None,
) -> pd.DataFrame | None:
    latitude_column = _pick_column(
        dataframe, ["latitude", "lat", "latitude deg", "lat deg", "Latitude (deg)", "Lat_deg"]
    )
    longitude_column = _pick_column(
        dataframe,
        ["longitude", "lon", "lng", "longitude deg", "lon deg", "Longitude (deg)", "Lon_deg"],
    )
    if latitude_column is None or longitude_column is None:
        return None
    latitude_series = pd.to_numeric(dataframe[latitude_column], errors="coerce")
    longitude_series = pd.to_numeric(dataframe[longitude_column], errors="coerce")
    valid = np.isfinite(latitude_series) & np.isfinite(longitude_series)
    if not valid.any():
        return None
    latitude = latitude_series[valid].to_numpy(dtype=float)
    longitude = longitude_series[valid].to_numpy(dtype=float)
    degree_mask = (np.abs(latitude) <= 90) & (np.abs(longitude) <= 180)
    if degree_mask.any():
        return pd.DataFrame(
            {"lat": latitude[degree_mask], "lon": longitude[degree_mask], "source": source}
        )

    crs = dict(projected_crs or {}).get(dataset)
    if crs is None:
        return None
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # Some induced-seismicity CSVs store northing/easting under lat/lon names.
        projected_longitude, projected_latitude = transformer.transform(longitude, latitude)
    except Exception as error:
        print(f"Could not transform projected coordinates for {dataset} ({crs}): {error}")
        return None
    transformed_mask = (
        np.isfinite(projected_latitude)
        & np.isfinite(projected_longitude)
        & (np.abs(projected_latitude) <= 90)
        & (np.abs(projected_longitude) <= 180)
    )
    if not transformed_mask.any():
        return None
    return pd.DataFrame(
        {
            "lat": projected_latitude[transformed_mask],
            "lon": projected_longitude[transformed_mask],
            "source": f"{source} transformed from {crs}",
        }
    )


def load_dataset_locations(dataset: str, config: LocationMapConfig) -> pd.DataFrame:
    for path in dataset_location_files(dataset, config.data_root):
        if not path.exists():
            continue
        try:
            dataframe = pd.read_csv(path)
        except Exception as error:
            print(f"Could not read {path}: {error}")
            continue
        points = lonlat_from_columns(
            dataframe,
            dataset=dataset,
            source=str(path.relative_to(config.data_root.parent)),
            projected_crs=config.projected_crs,
        )
        if points is not None and not points.empty:
            return points
    fallback = config.location_fallbacks.get(dataset)
    if fallback is None:
        raise ValueError(f"No finite location coordinates found for {dataset}.")
    return pd.DataFrame(
        {"lat": [fallback["lat"]], "lon": [fallback["lon"]], "source": [fallback["source"]]}
    )


def build_location_items(
    loaded_items: Sequence[Mapping[str, Any]], config: LocationMapConfig
) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for item in loaded_items:
        points = load_dataset_locations(str(item["dataset"]), config)
        center = points[["lon", "lat"]].median()
        locations.append(
            {
                "dataset": item["dataset"],
                "title": item["title"],
                "points": points,
                "center_lon": float(center["lon"]),
                "center_lat": float(center["lat"]),
                "source": points["source"].iloc[0],
            }
        )
    return locations


def location_summary(locations: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": location["dataset"],
                "title": location["title"],
                "center_lat": location["center_lat"],
                "center_lon": location["center_lon"],
                "n_location_points": len(location["points"]),
                "source": location["source"],
            }
            for location in locations
        ]
    )


def _sample_points(points: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(points) <= limit:
        return points
    return points.sample(n=limit, random_state=0).sort_index()


def _read_shapefile_polygons(shapefile: Path) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    with shapefile.open("rb") as handle:
        handle.seek(100)
        while True:
            record_header = handle.read(8)
            if len(record_header) < 8:
                break
            _, content_words = struct.unpack(">2i", record_header)
            content = handle.read(content_words * 2)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in {5, 15, 25, 31}:
                continue
            number_of_parts, number_of_points = struct.unpack("<2i", content[36:44])
            if number_of_parts <= 0 or number_of_points <= 0:
                continue
            parts_offset = 44
            points_offset = parts_offset + 4 * number_of_parts
            if len(content) < points_offset + 16 * number_of_points:
                continue
            part_starts = struct.unpack(
                f"<{number_of_parts}i", content[parts_offset:points_offset]
            )
            coordinates = np.frombuffer(
                content, dtype="<f8", count=number_of_points * 2, offset=points_offset
            ).reshape(number_of_points, 2)
            for index, start in enumerate(part_starts):
                end = part_starts[index + 1] if index + 1 < len(part_starts) else number_of_points
                part = coordinates[int(start) : int(end)]
                if len(part) >= 3:
                    polygons.append(part[:: max(1, len(part) // 450)].copy())
    return polygons


def _plot_world_background(ax, config: LocationMapConfig) -> str:
    ax.set_facecolor("#eef7ff")
    shapefile = config.world_shapefile
    if not config.draw_world_polygons or shapefile is None or not shapefile.exists():
        return "longitude/latitude grid only"
    try:
        from matplotlib.collections import PolyCollection

        polygons = _read_shapefile_polygons(shapefile)
        if polygons:
            ax.add_collection(
                PolyCollection(
                    polygons,
                    facecolor="#f3efe4",
                    edgecolor="0.62",
                    linewidths=0.28,
                    closed=True,
                    zorder=0,
                )
            )
            return "Natural Earth country polygons"
    except Exception as error:
        print(f"Could not draw world shapefile; using lon/lat grid: {error}")
    return "longitude/latitude grid only"


def plot_location_map_matplotlib(
    loaded_items: Sequence[Mapping[str, Any]], config: LocationMapConfig
):
    locations = build_location_items(loaded_items, config)
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    background_source = _plot_world_background(axis, config)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 85)
    axis.set_xlabel("Longitude (°)", fontsize=10)
    axis.set_ylabel("Latitude (°)", fontsize=10)
    if config.pygmt_title:
        axis.set_title(config.pygmt_title, fontsize=12, pad=8)
    axis.set_xticks(np.arange(-180, 181, 60))
    axis.set_yticks(np.arange(-60, 91, 30))
    axis.tick_params(axis="both", which="major", pad=6)
    if config.show_grid:
        axis.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, zorder=1)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, location in enumerate(locations):
        color = colors[index % len(colors)]
        points = _sample_points(location["points"], config.point_limit_per_dataset)
        if config.show_location_points:
            axis.scatter(
                points["lon"], points["lat"], s=8, color=color,
                alpha=0.22, linewidths=0, zorder=3,
            )
        axis.scatter(
            location["center_lon"], location["center_lat"], s=86, color=color,
            marker="*", edgecolor="black", linewidth=0.65, zorder=4, label=location["title"],
        )
        dx, dy = config.label_offsets.get(location["dataset"], (4.0, 4.0))
        axis.annotate(
            location["title"],
            xy=(location["center_lon"], location["center_lat"]),
            xytext=(location["center_lon"] + dx, location["center_lat"] + dy),
            textcoords="data",
            fontsize=9,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7, "alpha": 0.8},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
            zorder=5,
        )
    axis.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)
    axis.text(
        0.995, 0.02, f"Background: {background_source}", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=7.5, color="0.35",
    )
    figure.tight_layout()
    return figure, location_summary(locations)


def _require_pygmt():
    try:
        import pygmt
    except ImportError as error:
        raise ImportError(
            "PyGMT is required for use_pygmt=True; install it with conda-forge."
        ) from error
    return pygmt


def plot_location_map_pygmt(
    loaded_items: Sequence[Mapping[str, Any]], config: LocationMapConfig
):
    pygmt = _require_pygmt()
    locations = build_location_items(loaded_items, config)
    figure = pygmt.Figure()
    with pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        MAP_ANNOT_OFFSET_PRIMARY="6p",
    ):
        figure.coast(
            region=config.pygmt_region,
            projection=config.pygmt_projection,
            frame=config.pygmt_frame,
            land="#f3efe4",
            water="#eef7ff",
            shorelines="0.35p,gray40",
            borders="1/0.25p,gray65",
        )
        for location in locations:
            color = config.pygmt_colors.get(location["dataset"], "black")
            if config.show_location_points:
                points = _sample_points(location["points"], config.point_limit_per_dataset)
                figure.plot(
                    x=points["lon"].to_numpy(dtype=float),
                    y=points["lat"].to_numpy(dtype=float),
                    style="c0.035c",
                    fill=color,
                    pen="0p",
                    transparency=70,
                )
            figure.plot(
                x=[location["center_lon"]],
                y=[location["center_lat"]],
                style="a0.34c",
                fill=color,
                pen="0.45p,black",
                label=location["title"],
            )
            dx, dy = config.label_offsets.get(location["dataset"], (4.0, 4.0))
            label_lon = location["center_lon"] + dx
            label_lat = location["center_lat"] + dy
            figure.plot(
                x=[location["center_lon"], label_lon],
                y=[location["center_lat"], label_lat],
                pen=f"0.45p,{color}",
            )
            figure.text(
                x=label_lon,
                y=label_lat,
                text=location["title"],
                font=f"9p,Helvetica,{color}",
                justify="LM",
            )
        figure.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@15+p0.5p,gray40")
    return figure, location_summary(locations)


def plot_dataset_location_map(
    loaded_items: Sequence[Mapping[str, Any]], config: LocationMapConfig
):
    if config.use_pygmt:
        return plot_location_map_pygmt(loaded_items, config)
    return plot_location_map_matplotlib(loaded_items, config)


def save_location_map(
    figure,
    *,
    output_dir: str | Path,
    stem: str,
    config: LocationMapConfig,
    save_pdf: bool = True,
    save_png: bool = True,
    dpi: int = 300,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    for extension, enabled in (("pdf", save_pdf), ("png", save_png)):
        if not enabled:
            continue
        path = output_dir / f"{stem}.{extension}"
        if config.use_pygmt:
            figure.savefig(str(path), resize=f"+m{config.pygmt_export_margin}")
        else:
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths[extension] = path
    return paths


def show_location_map(figure, config: LocationMapConfig) -> None:
    if config.use_pygmt:
        figure.show(
            width=config.pygmt_display_width,
            resize=f"+m{config.pygmt_export_margin}",
        )
    else:
        plt.show()
