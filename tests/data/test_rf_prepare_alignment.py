import numpy as np
import pandas as pd

from src.data import event_loader
from src.data import event_pipeline
from src.data import preparation


class _Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_catalog_df():
    n = 160
    t = np.arange(n, dtype=float) * 2.0
    mag = 3.0 + 3.0 * (0.5 + 0.5 * np.sin(np.arange(n) / 5.0))
    lat = np.linspace(20.0, 21.0, num=len(t))
    lon = np.linspace(100.0, 101.0, num=len(t))
    dep = np.linspace(5.0, 10.0, num=len(t))
    dt = np.diff(np.concatenate(([t[0]], t)))
    return pd.DataFrame(
        {
            "t": t,
            "Magnitude": mag,
            "Latitude": lat,
            "Longitude": lon,
            "Depth": dep,
            "dt": dt,
            "dt_unfiltered": dt,
        }
    )


def test_rf_labels_follow_event_loader_rule(monkeypatch, tmp_path):
    df = _make_catalog_df()
    df_nl, _ = event_loader.normalize_df(df)
    samples, array_dict = event_loader.construct_samples_list(
        df,
        df_nl,
        Mc=3.0,
        Mf=5.5,
        Twindow=20.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
    )
    expected_labels, expected_valid = event_loader.build_classification_labels(array_dict, Mf=5.5)
    expected_y = expected_labels[expected_valid].astype(int)
    expected_t = np.array([sample["t"] for sample in samples], dtype=float)[expected_valid]

    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(
        event_pipeline,
        "save_or_load_data",
        lambda base_path, generate_fn, **kwargs: generate_fn(),
    )

    args = _Args(
        Mc=3.0,
        Mf=5.5,
        Twindow=20.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        split_by_time=True,
        time_order=("train", "val", "test"),
        dMag=0.1,
        Mag_elaps=[5.5],
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "beta", "zvalue", "T_elaps5.5"],
    )

    _df, features_df, _x_train, _x_val, _x_test, y_train, y_val, y_test = preparation.prepare_data_rf(
        args,
        str(tmp_path),
    )
    got_y = np.concatenate([y_train, y_val, y_test], axis=0)

    assert np.array_equal(got_y, expected_y)
    assert np.allclose(features_df["t"].to_numpy(), expected_t)


def test_shared_window_cache_builder_matches_direct_construct(monkeypatch, tmp_path):
    df = _make_catalog_df()
    df_nl, _ = event_loader.normalize_df(df)
    exp_samples, exp_array = event_loader.construct_samples_list(
        df,
        df_nl,
        Mc=3.0,
        Mf=5.5,
        Twindow=20.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
    )

    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(
        event_pipeline,
        "save_or_load_data",
        lambda base_path, generate_fn, **kwargs: generate_fn(),
    )

    args = _Args(Mc=3.0, Mf=5.5, Twindow=20.0, Tfore=8.0, dt=4.0, context_len=1)
    bundle = event_pipeline.load_event_windows_with_cache(
        str(tmp_path),
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=args.context_len,
        task_prefix="rf_classifier",
    )
    got_df, got_samples, got_array = bundle.df, bundle.samples_list, bundle.array_dict

    assert len(got_df) == len(df)
    assert len(got_samples) == len(exp_samples)
    assert np.allclose([s["t"] for s in got_samples], [s["t"] for s in exp_samples])
    for scope in ("history", "future", "context"):
        assert got_array[scope].keys() == exp_array[scope].keys()
        for key in got_array[scope]:
            assert len(got_array[scope][key]) == len(exp_array[scope][key])
            for a, b in zip(got_array[scope][key], exp_array[scope][key]):
                assert np.array_equal(a, b)


def test_event_window_cache_key_can_include_mag_elaps(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())

    captured = {}

    def _fake_save_or_load_data(base_path, generate_fn, **kwargs):
        captured.update(kwargs)
        return generate_fn()

    monkeypatch.setattr(event_pipeline, "save_or_load_data", _fake_save_or_load_data)

    event_pipeline.load_event_windows_with_cache(
        str(tmp_path),
        Mc=3.0,
        Mf=5.5,
        Twindow=20.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        task_prefix="rf_classifier",
        cache_key_extra={"Mag_elaps": [5.5, 6.0], "dMag": 0.1},
    )

    assert "Mag_elaps" in captured
    assert captured["Mag_elaps"] == [5.5, 6.0]
    assert "dMag" in captured
    assert captured["dMag"] == 0.1
