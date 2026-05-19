import logging


logger = logging.getLogger(__name__)


def prepare_data_lstm_legacy(args, base_dir):
    import numpy as np
    import src.data.lstm_loader as loader
    import src.features.seismic_features as sf
    from src.data.preprocessing import load_and_filter_catalog
    from src.utils.file_utils import save_or_load_data

    def create_lstm_data(features, target, timestep):
        X, y = [], []
        for index in range(len(features) - timestep):
            X.append(features[index : index + timestep])
            y.append(target[index + timestep])
        return np.array(X), np.array(y)

    if not hasattr(args, "feature_cols"):
        raise AttributeError("args.feature_cols is required for prepare_data_lstm_legacy")

    df = load_and_filter_catalog(base_dir, Mc=args.Mc)

    def generate_seismic_data():
        features_df, num_mag = sf.calculate_seismic_features(
            df.to_numpy(),
            Mc=args.Mc,
            Mf=args.Mf,
            Twindow=args.Twindow,
            Tfore=args.Tfore,
            dt=args.dt,
            dMag=args.dMag,
            Mag_elaps=args.Mag_elaps,
            L_max=60,
            context_len=args.context_len,
        )
        return {
            "features_df": features_df,
            "num_mag": num_mag,
        }

    cached_data = save_or_load_data(
        base_path=base_dir,
        generate_fn=generate_seismic_data,
        sub_dir="processed/rf_classifier",
        prefix="rf",
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        dMag=args.dMag,
        Mag_elaps=args.Mag_elaps,
        context_len=args.context_len,
    )

    features_df = cached_data["features_df"]
    features_df_nl, scalars = loader.normalize_df(features_df)
    features = features_df_nl[args.feature_cols].values
    target = features_df_nl["Mag_max_obs"].values.copy()
    X, y = create_lstm_data(features, target, timestep=args.time_step)
    X, y = loader.clean_data(X, y)
    dataset, data_loaders = loader.split_dataset(
        X,
        y,
        by_time=args.split_by_time,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        time_order=getattr(args, "time_order", ("train", "val", "test")),
        scalars=scalars,
    )

    return features_df, data_loaders["train"], data_loaders["val"], data_loaders["test"], dataset
