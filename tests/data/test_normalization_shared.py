import numpy as np
import pytest

from src.data.normalization import (
    inverse_normalize_magnitude_range,
    normalize_magnitude_range,
    resolve_magnitude_bounds,
    validate_magnitude_bounds,
)


class _Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_validate_magnitude_bounds_ok():
    mag_min, mag_max = validate_magnitude_bounds(3.0, 9.0)
    assert np.isclose(mag_min, 3.0)
    assert np.isclose(mag_max, 9.0)


def test_validate_magnitude_bounds_invalid_order():
    with pytest.raises(ValueError):
        validate_magnitude_bounds(9.0, 3.0)


def test_normalize_and_inverse_magnitude_range_roundtrip():
    values = np.array([3.0, 4.5, 6.0, 9.0], dtype=float)
    norm = normalize_magnitude_range(values, mag_min=3.0, mag_max=9.0)
    back = inverse_normalize_magnitude_range(norm, mag_min=3.0, mag_max=9.0)
    assert np.allclose(back, values, atol=1e-8)


def test_resolve_magnitude_bounds_defaults():
    args = _Args()
    mag_min, mag_max = resolve_magnitude_bounds(args)
    assert np.isclose(mag_min, 3.0)
    assert np.isclose(mag_max, 9.0)
