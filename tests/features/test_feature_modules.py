import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.features.b_ll import estimate_b_value, log_likelihood_b
from src.features.seismic_features import (
    cal2jd,
    calculate_elapsed_times,
    calculate_magnitudes_and_features,
    calculate_seismic_change_rate,
    calculate_seismic_features,
    calculate_seismic_features_n,
)
from src.features.si_ll import log_lhood_comp


def _build_catalog(n=120):
    t = np.arange(n, dtype=float) * 5.0
    mag = 4.7 + 1.8 * (0.5 + 0.5 * np.sin(np.arange(n) / 6.0))
    return np.column_stack([t, mag])


def test_cal2jd_accepts_multiple_input_types():
    ref = cal2jd([2025, 2, 12, 10, 30, 15.5])
    dt = datetime(2025, 2, 12, 10, 30, 15, 500000)
    ts = pd.Timestamp("2025-02-12 10:30:15.500000")
    assert np.isclose(cal2jd(dt), ref)
    assert np.isclose(cal2jd(ts), ref)


def test_calculate_elapsed_times_uses_global_last_index():
    jd = np.array([1.0, 3.0, 10.0])
    mag = np.array([5.0, 6.1, 5.5])
    out = calculate_elapsed_times(jd, 11.0, mag, Mag_elaps=[6.0, 7.0])
    assert np.isclose(out[0], 8.0)
    assert np.isnan(out[1])


def test_calculate_magnitudes_and_features_basic_consistency():
    mags = np.array([5.0, 5.1, 5.2, 5.3])
    (
        _b_lsq,
        _a_lsq,
        _std_lsq,
        b_mlk,
        a_mlk,
        _std_mlk,
        _dm_lsq,
        _dm_mlk,
        _b_std_lsq,
        _b_std_mlk,
        num_mag_int,
    ) = calculate_magnitudes_and_features(mags, Mc=5.0, dMag=0.1)
    expected_b = np.log10(np.e) / (mags.mean() - 5.0)
    expected_a = np.log10(len(mags)) + expected_b * 5.0
    assert np.isclose(b_mlk, expected_b)
    assert np.isclose(a_mlk, expected_a)
    assert num_mag_int[0] == len(mags)
    assert np.all(np.diff(num_mag_int) <= 0)


def test_calculate_seismic_change_rate_empty_returns_nan():
    beta, zvalue = calculate_seismic_change_rate(np.array([]), Twindow=20, t=100)
    assert np.isnan(beta)
    assert np.isnan(zvalue)


def test_calculate_seismic_features_lmax_and_legacy_twindow():
    data = _build_catalog()
    kwargs = dict(Mc=4.7, Mf=5.8, Tfore=20, dt=10, dMag=0.1, Mag_elaps=[5.8, 6.0], L_max=13, context_len=1)
    feat1, num1 = calculate_seismic_features(data, Twindow=50, **kwargs)
    feat2, num2 = calculate_seismic_features(data, Twindow=[50], **kwargs)
    assert num1.shape[1] == 13
    assert num2.shape[1] == 13
    assert feat1.shape == feat2.shape
    assert np.allclose(feat1["t"].to_numpy(), feat2["t"].to_numpy())
    assert np.allclose(num1, num2, equal_nan=True)


def test_calculate_seismic_features_filters_invalid_t_array():
    data = _build_catalog()
    twindow = 50
    valid_t = data[20, 0]
    t_arr = np.array([0.0, valid_t, data[-1, 0] + 100.0])
    feat, num_mag = calculate_seismic_features(
        data,
        Mc=4.7,
        Mf=5.8,
        Twindow=twindow,
        Tfore=20,
        dt=10,
        t_arrary=t_arr,
        L_max=8,
    )
    assert len(feat) == 1
    assert np.isclose(feat["t"].iloc[0], valid_t)
    assert num_mag.shape == (1, 8)


def test_calculate_seismic_features_n_outputs_per_window():
    data = _build_catalog()
    results, num_mag_all = calculate_seismic_features_n(
        data,
        Mc=4.7,
        Mf=5.8,
        Twindow_list=[50, 30],
        Tfore=20,
        dt=10,
        dMag=0.1,
        Mag_elaps=[5.8],
    )
    assert set(results.keys()) == {50, 30}
    assert set(num_mag_all.keys()) == {50, 30}
    assert len(results[50]) == len(num_mag_all[50])
    assert len(results[30]) == len(num_mag_all[30])


def test_b_log_likelihood_and_estimate_guardrails():
    mags = np.array([5.0, 5.1, 5.2, 5.4, 5.6, 5.8])
    assert log_likelihood_b(-1.0, mags, Mc=4.7, Mmax=10.0) == -np.inf
    assert log_likelihood_b(1.0, mags, Mc=5.0, Mmax=4.9) == -np.inf
    b_hat, ll = estimate_b_value(mags, Mc=4.7, Mmax=10.0)
    assert np.isfinite(b_hat) and b_hat > 0
    assert np.isfinite(ll)
    b_empty, ll_empty = estimate_b_value([], Mc=4.7, Mmax=10.0)
    assert np.isnan(b_empty)
    assert np.isnan(ll_empty)


def test_log_lhood_comp_matches_manual_formula_and_no_mutation():
    rate = {
        "m_0": 4.7,
        "t_b_s": np.array([0.0, 1.0, 2.0, 3.0]),
        "dot_V_bs": np.array([1.0, 1.1, 1.2, 1.3]),
        "t_sbs": np.array([0.5, 1.5, 2.5]),
        "tot_V": 5.0,
        "data_magn": np.array([5.0, 5.2, 5.1]),
    }
    theta = np.array([0.2, 1.0])
    out = log_lhood_comp(theta, rate)

    af, b = theta
    n = len(rate["data_magn"])
    a1 = n * (af - b * rate["m_0"]) / np.log10(np.exp(1))
    dot_v_bs_ts = np.interp(rate["t_sbs"], rate["t_b_s"], rate["dot_V_bs"])
    k2 = np.sum(np.log(dot_v_bs_ts))
    a5 = -10 ** (af - b * rate["m_0"]) * rate["tot_V"]
    b_ll = log_likelihood_b(b, rate["data_magn"], rate["m_0"], 10)
    expected = -(a1 + k2 + a5 + b_ll) / 10000

    assert np.isclose(out, expected)
    assert "N" not in rate


def test_glob_rect_grid_and_wrapper_consistency():
    pytest.importorskip("pyproj")
    from src.features.glob_rect_grid import GlobRectGrid, glob_rect_grid

    lon1, lat1, area1, vec1 = glob_rect_grid([1, 1], [100, 102], [30, 32])
    lon2, lat2, area2, vec2 = GlobRectGrid([1, 1], [100, 102], [30, 32])

    assert lon1.shape == (2, 2)
    assert lat1.shape == (2, 2)
    assert area1.shape == (2, 2)
    assert vec1.shape == (4, 3)
    assert np.all(area1 > 0)
    assert np.allclose(lon1, lon2)
    assert np.allclose(lat1, lat2)
    assert np.allclose(area1, area2)
    assert np.allclose(vec1, vec2)


def test_glob_rect_grid_input_validation():
    pytest.importorskip("pyproj")
    from src.features.glob_rect_grid import glob_rect_grid

    with pytest.raises(ValueError):
        glob_rect_grid([1], [100, 102], [30, 32])
