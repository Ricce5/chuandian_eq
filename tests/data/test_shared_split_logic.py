import numpy as np
import pandas as pd

from src.data import event_loader
from src.data import event_pipeline
from src.data import preparation
from src.train.regressor_train_step import get_root_dataset


class _Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_catalog_df(n=180):
    t = np.arange(n, dtype=float) * 2.0
    mag = 3.0 + 3.0 * (0.5 + 0.5 * np.sin(np.arange(n) / 7.0))
    lat = np.linspace(20.0, 22.0, num=n)
    lon = np.linspace(100.0, 102.0, num=n)
    dep = np.linspace(5.0, 12.0, num=n)
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


def test_prepare_data_rf_uses_event_loader_split(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(event_pipeline, "save_or_load_data", lambda base_path, generate_fn, **kwargs: generate_fn())

    called = {"count": 0}
    original_split = event_loader.split_dataset

    def _split_spy(*args, **kwargs):
        called["count"] += 1
        return original_split(*args, **kwargs)

    monkeypatch.setattr(event_loader, "split_dataset", _split_spy)

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
    preparation.prepare_data_rf(args, str(tmp_path))
    assert called["count"] >= 1


def test_prepare_data_lstm_uses_event_loader_split(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(event_pipeline, "save_or_load_data", lambda base_path, generate_fn, **kwargs: generate_fn())

    called = {"count": 0}
    original_split = event_loader.split_dataset

    def _split_spy(*args, **kwargs):
        called["count"] += 1
        return original_split(*args, **kwargs)

    monkeypatch.setattr(event_loader, "split_dataset", _split_spy)

    args = _Args(
        Mc=3.0,
        Mf=3.0,
        Twindow=24.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        split_by_time=True,
        time_order=("train", "val", "test"),
        batch_size=8,
        time_step=4,
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"],
    )
    preparation.prepare_data_lstm(args, str(tmp_path))
    assert called["count"] >= 1


def test_prepare_data_lstm_cache_key_includes_time_step_and_mag_elaps(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())

    captured = {}

    def _fake_save_or_load_data(base_path, generate_fn, **kwargs):
        captured.update(kwargs)
        return generate_fn()

    monkeypatch.setattr(event_pipeline, "save_or_load_data", _fake_save_or_load_data)

    args = _Args(
        Mc=3.0,
        Mf=3.0,
        Twindow=24.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        batch_size=8,
        time_step=5,
        lstm_feature_mode="external_sliding",
        Mag_elaps=[5.5, 6.0],
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"],
    )
    preparation.prepare_data_lstm(args, str(tmp_path))

    assert captured["sub_dir"] == "lstm_regressor"
    assert captured["prefix"] == "lstm_regressor"
    assert "feature_window" in captured
    assert captured["feature_window"] is None
    assert "feature_step" in captured
    assert captured["feature_step"] is None
    assert "time_step" in captured
    assert captured["time_step"] == 5
    assert "lstm_feature_mode" in captured
    assert captured["lstm_feature_mode"] == "external_sliding"
    assert "Mag_elaps" in captured
    assert captured["Mag_elaps"] == [5.5, 6.0]


def test_prepare_data_lstm_external_sliding_rejects_feature_step(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(event_pipeline, "save_or_load_data", lambda base_path, generate_fn, **kwargs: generate_fn())

    args = _Args(
        Mc=3.0,
        Mf=3.0,
        Twindow=24.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        batch_size=8,
        time_step=5,
        feature_step=2.0,
        lstm_feature_mode="external_sliding",
        Mag_elaps=[5.5],
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"],
    )

    with np.testing.assert_raises_regex(ValueError, "feature_step is not supported"):
        preparation.prepare_data_lstm(args, str(tmp_path))


def test_split_config_defaults():
    args = _Args()
    cfg = preparation._split_config(args)
    assert cfg["by_time"] is True
    assert np.isclose(cfg["train_ratio"], preparation.DEFAULT_TRAIN_RATIO)
    assert np.isclose(cfg["val_ratio"], preparation.DEFAULT_VAL_RATIO)
    assert cfg["time_order"] == ("train", "val", "test")


def test_prepare_data_lstm_label_normalization_matches_prepare_data_range(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(event_pipeline, "save_or_load_data", lambda base_path, generate_fn, **kwargs: generate_fn())

    args = _Args(
        Mc=3.0,
        Mf=3.0,
        Twindow=24.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        split_by_time=True,
        time_order=("train", "val", "test"),
        batch_size=16,
        time_step=4,
        mag_min=3.0,
        mag_max=9.0,
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"],
    )

    _meta, train_loader, _val_loader, _test_loader, _dataset = preparation.prepare_data_lstm(args, str(tmp_path))
    root_ds = get_root_dataset(train_loader)
    assert root_ds.label_norm_cfg["type"] == "fixed_range"
    assert np.isclose(root_ds.label_norm_cfg["mag_min"], 3.0)
    assert np.isclose(root_ds.label_norm_cfg["mag_max"], 9.0)

    y_train_norm = root_ds.y.numpy()
    y_train_denorm = root_ds.inverse_normalize_label(root_ds.y).numpy()
    y_train_renorm = (y_train_denorm - 3.0) / (9.0 - 3.0)
    assert np.allclose(y_train_norm, y_train_renorm, atol=1e-6)


def test_prepare_data_lstm_external_sliding_aligns_val_test_to_outer_split(monkeypatch, tmp_path):
    df = _make_catalog_df()
    monkeypatch.setattr(event_pipeline, "load_and_filter_catalog", lambda base_dir, Mc: df.copy())
    monkeypatch.setattr(event_pipeline, "save_or_load_data", lambda base_path, generate_fn, **kwargs: generate_fn())

    args = _Args(
        Mc=3.0,
        Mf=3.0,
        Twindow=24.0,
        Tfore=8.0,
        dt=4.0,
        context_len=1,
        split_by_time=True,
        time_order=("train", "val", "test"),
        batch_size=16,
        time_step=4,  # seq_len=4
        lstm_feature_mode="external_sliding",
        mag_min=3.0,
        mag_max=9.0,
        feature_cols=["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"],
    )

    features_meta, train_loader, val_loader, test_loader, _dataset = preparation.prepare_data_lstm(args, str(tmp_path))
    # feature meta should be NaN-filtered and aligned with dataset indices
    meta_t = np.asarray(features_meta["t"], dtype=float)
    root_train = get_root_dataset(train_loader)
    assert len(meta_t) == len(root_train)

    # Reconstruct outer window split boundaries
    df_nl, _ = event_loader.normalize_df(df)
    samples_list, _array_dict = event_loader.construct_samples_list(
        df,
        df_nl,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=args.context_len,
    )
    n_outer = len(samples_list)
    train_outer, val_outer, test_outer = preparation._build_outer_split_indices_for_lstm(
        total_outer_windows=n_outer,
        loader_module=event_loader,
        args=args,
    )
    val_min_t = samples_list[int(val_outer.min())]["t"]
    val_max_t = samples_list[int(val_outer.max())]["t"]
    test_min_t = samples_list[int(test_outer.min())]["t"]
    test_max_t = samples_list[int(test_outer.max())]["t"]

    val_subset = val_loader.dataset
    test_subset = test_loader.dataset
    val_t = meta_t[np.asarray(val_subset.indices, dtype=int)]
    test_t = meta_t[np.asarray(test_subset.indices, dtype=int)]

    # Validation/test sets align with outer split ranges exactly
    assert val_t.size > 0
    assert test_t.size > 0
    assert np.isclose(val_t.min(), val_min_t)
    assert np.isclose(val_t.max(), val_max_t)
    assert np.isclose(test_t.min(), test_min_t)
    assert np.isclose(test_t.max(), test_max_t)

    # Training set is allowed to be smaller (early windows consumed by seq_len-1 warmup)
    train_subset = train_loader.dataset
    train_t = meta_t[np.asarray(train_subset.indices, dtype=int)]
    train_min_t_expected = samples_list[0]["t"] + (args.time_step - 1) * args.dt
    assert train_t.size > 0
    assert train_t.min() >= train_min_t_expected - 1e-8
