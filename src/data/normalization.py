import numpy as np

DEFAULT_MAG_MIN = 3.0
DEFAULT_MAG_MAX = 9.0


def validate_magnitude_bounds(mag_min: float, mag_max: float) -> tuple[float, float]:
    mag_min = float(mag_min)
    mag_max = float(mag_max)
    if not np.isfinite(mag_min) or not np.isfinite(mag_max):
        raise ValueError(f"mag_min/mag_max must be finite, got mag_min={mag_min}, mag_max={mag_max}")
    if mag_max <= mag_min:
        raise ValueError(f"mag_max must be greater than mag_min, got mag_min={mag_min}, mag_max={mag_max}")
    return mag_min, mag_max


def resolve_magnitude_bounds(args, default_min=DEFAULT_MAG_MIN, default_max=DEFAULT_MAG_MAX) -> tuple[float, float]:
    mag_min = getattr(args, "mag_min", default_min)
    mag_max = getattr(args, "mag_max", default_max)
    return validate_magnitude_bounds(mag_min, mag_max)


def normalize_magnitude_range(values, *, mag_min: float, mag_max: float):
    mag_min, mag_max = validate_magnitude_bounds(mag_min, mag_max)
    values_arr = np.asarray(values, dtype=float)
    return (values_arr - mag_min) / (mag_max - mag_min)


def inverse_normalize_magnitude_range(values, *, mag_min: float, mag_max: float):
    mag_min, mag_max = validate_magnitude_bounds(mag_min, mag_max)
    values_arr = np.asarray(values, dtype=float)
    return values_arr * (mag_max - mag_min) + mag_min
