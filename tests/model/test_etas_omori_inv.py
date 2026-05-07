import numpy as np

from src.models.tpp.etas import omori_int, omori_inv
from src.models.tpp.etas_zhuang import _omori_inv_np


def test_conditional_cdf_equals_global_cdf_affine_map():
    """F_cond(t) == (F(t)-F(T1))/(F(T2)-F(T1)) on [T1, T2]."""
    c = 0.1
    p = 1.2
    t_max = 10.0
    T1, T2 = 0.5, 3.0

    t = np.linspace(T1, T2, 128)

    F = lambda x: omori_int(0.0, x, c, p) / omori_int(0.0, t_max, c, p)
    lhs = (F(t) - F(T1)) / (F(T2) - F(T1))
    rhs = omori_int(T1, t, c, p) / omori_int(T1, T2, c, p)

    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_omori_inv_handles_p_equal_one():
    np.random.seed(123)
    samples = omori_inv(T1=0.2, T2=2.0, c=0.1, p=1.0, size=10_000, t_max=10.0)

    assert np.isfinite(samples).all()
    assert np.min(samples) >= 0.2 - 1e-12
    assert np.max(samples) <= 2.0 + 1e-12


def test_omori_inv_clips_to_tmax_when_t2_exceeds_cap():
    np.random.seed(123)
    samples = omori_inv(T1=0.0, T2=20.0, c=0.1, p=1.2, size=10_000, t_max=5.0)

    assert np.isfinite(samples).all()
    assert np.max(samples) <= 5.0 + 1e-12


def test_omori_inv_np_in_ogata_handles_p_equal_one_and_tmax_clip():
    np.random.seed(123)
    samples = _omori_inv_np(T1=0.2, T2=20.0, c=0.1, p=1.0, size=10_000, t_max=5.0)

    assert np.isfinite(samples).all()
    assert np.min(samples) >= 0.2 - 1e-12
    assert np.max(samples) <= 5.0 + 1e-12
