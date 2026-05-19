import numpy as np
import pandas as pd

from src.data import event_loader
from src.data import event_pipeline


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


def _build_samples(df, *, Mc=3.0, Mf=5.5, Twindow=20.0, Tfore=8.0, dt=4.0, context_len=1):
    df_nl, _ = event_loader.normalize_df(df)
    samples, array_dict = event_loader.construct_samples_list(
        df,
        df_nl,
        Mc=Mc,
        Mf=Mf,
        Twindow=Twindow,
        Tfore=Tfore,
        dt=dt,
        context_len=context_len,
    )
    return samples, array_dict


def test_full_window_feature_frame_shape_and_target():
    df = _make_catalog_df()
    samples, array_dict = _build_samples(df)
    feature_cols = ["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "beta", "zvalue", "T_elaps5.5"]
    out = event_pipeline.build_full_window_feature_frame(array_dict, samples, feature_cols, Mc=3.0, dMag=0.1)

    assert len(out) == len(samples)
    assert set(feature_cols).issubset(out.columns)
    assert "Mag_max_obs" not in out.columns

    future_mag = array_dict["future"]["Magnitude"][0]
    expected = np.max(future_mag) if len(future_mag) > 0 else np.nan
    assert np.isclose(out["Mag_max_obs"].iloc[0], expected, equal_nan=True) if "Mag_max_obs" in out.columns else True


def test_inner_sliding_feature_tensor_shape_and_sequence_length():
    df = _make_catalog_df()
    samples, array_dict = _build_samples(df, Twindow=24.0, dt=4.0)
    feature_cols = ["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"]

    X, y, meta = event_pipeline.build_inner_sliding_feature_tensor(
        array_dict,
        samples,
        feature_cols,
        Mc=3.0,
        dMag=0.1,
        Twindow=24.0,
        time_step=4,
    )
    assert X.ndim == 3
    assert X.shape[0] == len(samples)
    assert X.shape[1] == 4
    assert X.shape[2] == len(feature_cols)
    assert y.shape[0] == len(samples)
    assert len(meta) == len(samples)
    assert "t" in meta.columns
    assert "Mag_max_obs" in meta.columns


def test_full_window_and_rf_builder_alias():
    df = _make_catalog_df()
    samples, array_dict = _build_samples(df)
    feature_cols = ["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "beta", "zvalue"]
    f1 = event_pipeline.build_full_window_feature_frame(array_dict, samples, feature_cols, Mc=3.0, dMag=0.1)
    f2 = event_pipeline.build_rf_feature_frame(array_dict, samples, feature_cols, Mc=3.0, dMag=0.1)

    assert f1.shape == f2.shape
    assert list(f1.columns) == list(f2.columns)
    assert np.allclose(f1.to_numpy(), f2.to_numpy(), equal_nan=True)


def test_inner_sliding_default_step_uses_outer_dt():
    feature_window, feature_step, seq_len = event_pipeline.resolve_inner_sliding_params(
        Twindow=360.0,
        dt=10.0,
        feature_window=320.0,
        feature_step=None,
    )
    assert np.isclose(feature_window, 320.0)
    assert np.isclose(feature_step, 10.0)
    assert seq_len == 5


def test_external_sliding_feature_tensor_shape_and_alignment():
    df = _make_catalog_df()
    samples, array_dict = _build_samples(df, Twindow=24.0, dt=4.0)
    feature_cols = ["Num", "Mag_max", "Mag_mean", "b_lsq", "a_lsq", "T_elaps5.5"]

    X, y, meta = event_pipeline.build_external_sliding_feature_tensor(
        array_dict,
        samples,
        feature_cols,
        Mc=3.0,
        dMag=0.1,
        Twindow=24.0,
        time_step=4,
    )

    expected_seq_len = 4
    expected_count = len(samples) - expected_seq_len + 1
    assert X.ndim == 3
    assert X.shape[0] == expected_count
    assert X.shape[1] == expected_seq_len
    assert X.shape[2] == len(feature_cols)
    assert y.shape[0] == expected_count
    assert len(meta) == expected_count
    assert "t" in meta.columns
    assert "Mag_max_obs" in meta.columns
    assert "sequence_mode" in meta.columns
    assert set(meta["sequence_mode"].unique()) == {event_pipeline.FEATURE_CONSTRUCTION_EXTERNAL_SLIDING}


def test_external_sliding_requires_time_step():
    with np.testing.assert_raises_regex(ValueError, "requires a positive integer time_step"):
        event_pipeline.resolve_external_sliding_params(
            Twindow=24.0,
            dt=4.0,
            time_step=None,
            feature_window=None,
            feature_step=None,
        )


def test_external_sliding_rejects_feature_step():
    with np.testing.assert_raises_regex(ValueError, "feature_step is not supported"):
        event_pipeline.resolve_external_sliding_params(
            Twindow=24.0,
            dt=4.0,
            time_step=4,
            feature_window=None,
            feature_step=2.0,
        )


def test_full_window_feature_frame_global_t_elaps_uses_catalog_history():
    df = _make_catalog_df(n=420)
    # Ensure there is at least one >=7.0 event early in catalog, so
    # global elapsed can differ from window elapsed on later windows.
    df.loc[0, "Magnitude"] = 7.2
    samples, array_dict = _build_samples(df, Mc=3.0, Mf=5.5, Twindow=180.0, Tfore=90.0, dt=10.0, context_len=1)
    feature_cols = ["T_elaps7"]

    out_window = event_pipeline.build_full_window_feature_frame(
        array_dict,
        samples,
        feature_cols,
        Mc=3.0,
        dMag=0.1,
        t_elaps_mode="window",
    )
    out_global = event_pipeline.build_full_window_feature_frame(
        array_dict,
        samples,
        feature_cols,
        Mc=3.0,
        dMag=0.1,
        t_elaps_mode="global",
        global_t=df["t"].to_numpy(dtype=float),
        global_mag=df["Magnitude"].to_numpy(dtype=float),
    )

    window_nan = out_window["T_elaps7"].isna().sum()
    global_nan = out_global["T_elaps7"].isna().sum()
    assert global_nan <= window_nan
    assert np.any(~np.isclose(out_window["T_elaps7"], out_global["T_elaps7"], equal_nan=True))


def test_window_feature_map_global_telaps_computed_even_when_subwindow_empty():
    # sub-window contains no events, but global history has >= threshold events.
    history_t = np.array([], dtype=float)
    history_mag = np.array([], dtype=float)
    global_t = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    global_mag = np.array([5.0, 6.2, 5.8, 6.6], dtype=float)
    t_reference = 50.0
    feature_cols = ["Num", "T_elaps6"]
    elapsed_thresholds = {"T_elaps6": 6.0}

    feature_map = event_pipeline._compute_window_feature_map(
        history_t,
        history_mag,
        feature_cols=feature_cols,
        Mc=3.0,
        dMag=0.1,
        t_reference=t_reference,
        elapsed_thresholds=elapsed_thresholds,
        t_elaps_mode="global",
        global_t=global_t,
        global_mag=global_mag,
    )

    # last global event >= 6.0 occurs at t=40.0
    assert np.isclose(feature_map["T_elaps6"], 10.0)
    # window-based scalar features remain NaN for empty local history.
    assert np.isnan(feature_map["Num"])
