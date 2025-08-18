import math
from src.utils.file_utils import save_or_load_data
from src.utils.catalog_utils import split_minibatches
from src.data.preprocessing import load_and_filter_catalog, calculate_catalog_statistics 
from  omegaconf import OmegaConf

def prepare_data(args, base_dir):
    import src.data.event_loader as loader
    df = load_and_filter_catalog(base_dir, Mc=args.Mc)
    statistics = calculate_catalog_statistics(df)
    args.stats = OmegaConf.create(statistics)
    df_nl,scalers = loader.normalize_df(df)
    args.task_type = getattr(args, "task_type", "classification")
    task_prefix_map = {
        "classification": "classifier",
        "regression": "regressor",
        "count": "counter"
    }
    task_prefix = task_prefix_map.get(args.task_type)
    def generate_data():
        samples_list, array_dict = loader.construct_samples_list(
            df, df_nl,
            Mc=args.Mc,
            Mf=args.Mf,
            Twindow=args.Twindow,
            Tfore=args.Tfore,
            dt=args.dt,
            context_len= getattr(args, 'context_len', 0),
        )
        return {
            "samples_list": samples_list,
            "array_dict": array_dict
        }
    cached = save_or_load_data(
        base_path=base_dir,
        generate_fn=generate_data,
        sub_dir=f"processed/{task_prefix}",
        prefix=task_prefix,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=getattr(args, 'context_len', 0),
    )
    array_dict = cached["array_dict"]

    dataset = loader.EventDataset(array_dict, args.Mf,task_type=args.task_type)
    train_set, val_set, test_set = loader.split_dataset(
        dataset,
        by_time=args.split_by_time,
        train_ratio=0.8,
        val_ratio=0.1,
        time_order=getattr(args, 'time_order', ('train', 'val', 'test')),
        task_type=args.task_type
    )
    if args.use_sampler and args.task_type == "classification":
        sampler = loader.get_balanced_sampler(train_set)
    else:
        sampler = None
    train_loader = loader.get_dataloader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler,task_type=args.task_type)
    val_loader = loader.get_dataloader(val_set, batch_size=args.batch_size, shuffle=False, task_type=args.task_type)
    test_loader = loader.get_dataloader(test_set, batch_size=args.batch_size, shuffle=False,task_type=args.task_type)
    return df, train_loader, val_loader, test_loader,dataset


def prepare_data_lstm(args, base_dir="data/CD2021"):
    def create_lstm_data(features, target, timestep):
        """
        划分数据集，生成特征数据和目标数据
        :param features: 特征数据（二维数组），数据集的所有特征列（去除目标列）
        :param target: 目标数据（数组），数据集的目标列
        :param timestep: 时间步长，用于生成每个样本的特征序列长度
        :return: 特征数据X和目标数据Y
        """
        X, y = [], []
        
        for index in range(len(features) - timestep):
            X.append(features[index: index + timestep])
            y.append(target[index + timestep])

        # 转换为NumPy数组
        X, y = np.array(X), np.array(y)
        return X, y

    import src.data.lstm_loader as loader
    import src.features.seismic_features as sf
    from src.data.preprocessing import load_and_filter_catalog
    from src.data.data_utils import get_split_indices
    from src.utils.file_utils import save_or_load_data
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

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
            "num_mag": num_mag
        }
    # 加载或生成处理过的数据
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
        context_len=args.context_len
    )

    features_df = cached_data["features_df"]
    features_df_nl,scalars = loader.normalize_df(features_df)
    num_mag = cached_data["num_mag"]
    features = features_df_nl[args.feature_cols].values
    target = features_df_nl['Mag_max_obs'].values.copy()
    X, y = create_lstm_data(features, target, timestep=args.time_step) 
    X,y = loader.clean_data(X,y)
    dataset,data_loaders = loader.split_dataset(
        X, y,
        by_time=args.split_by_time,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        time_order=getattr(args, 'time_order', ('train', 'val', 'test')),
        scalars=scalars
    )

    return features_df,data_loaders['train'], data_loaders['val'], data_loaders['test'], dataset


def prepare_data_tpp(args, base_dir,use_double_precision=False):
    import os
    import torch

    from src.data.tpp_dataset import TppDataset
    from src.data.sequence import EventSequence
    import src.data.catalog as catalog
    import src.catalogs as catalogs

    # Earthquake datasets
    root_dir = os.path.join(base_dir, 'raw')
    dat_files = [f for f in os.listdir(root_dir) if f.endswith('.dat')]

    if len(dat_files) == 1:
        file_path = os.path.join(root_dir, dat_files[0])
    else:
        file_path = None

    catalog_ds_class = catalog.Catalog.by_name(f"{args.dataset}-Standard")
    print(f"Using catalog dataset class: {catalog_ds_class}")

    catalog_ds = catalog_ds_class(root_dir=root_dir,catalog_file=file_path,)
    if use_double_precision:
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.double()
        args.precision = 64
    else:
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.float()
        args.precision = 32
    args.num_events_train = sum(seq.num_nll_events for seq in catalog_ds.train)
    args.num_events_val = sum(seq.num_nll_events for seq in catalog_ds.val)
    print(f"Number of training events: {args.num_events_train}")
    print(f"Number of validation events: {args.num_events_val}")
    if getattr(args, 'minibatch_training', True):
       print("Splitting into minibatches")
       catalog_ds = split_minibatches(catalog_ds,300,40000) 

    train_loader = catalog_ds.train.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        pad_token_id=getattr(args, 'pad_token_id', None),
    )
    val_loader = catalog_ds.val.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        pad_token_id=getattr(args, 'pad_token_id', None),
    )
    test_loader = catalog_ds.test.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        pad_token_id=getattr(args, 'pad_token_id', None),
    )
    args.tau_mean = torch.cat([seq.inter_times[:-1] for seq in catalog_ds.train]).mean().item()
    args.tau_min = torch.cat([seq.inter_times[:-1] for seq in catalog_ds.train]).min().item()
    args.tau_max = torch.cat([seq.inter_times[:-1] for seq in catalog_ds.train]).max().item()
    args.tau_q05 = torch.cat([seq.inter_times[:-1] for seq in catalog_ds.train]).quantile(0.5).item()
    args.tau_q025 = torch.cat([seq.inter_times[:-1] for seq in catalog_ds.train]).quantile(0.025).item()
    args.mag_mean = torch.cat([seq.mag for seq in catalog_ds.train]).mean().item()
    args.time_max = torch.max(torch.tensor([seq.t_end for seq in catalog_ds.train])).item()
    args.time_mean = torch.cat([seq.arrival_times[:-1] for seq in catalog_ds.train]).mean().item()
    args.mag_completeness = catalog_ds.metadata["mag_completeness"]
    if "richter_b" in catalog_ds.metadata:
        # Use ground truth value, if available
        args.richter_b_mle = catalog_ds.metadata["richter_b"]
    else:
        mag_roundoff_error = catalog_ds.metadata.get("mag_roundoff_error", 0.0)
        args.richter_b_mle = math.log10(math.exp(1)) / (
            args.mag_mean - args.mag_completeness + 0.5 * mag_roundoff_error
        )

    return catalog_ds.full_sequence, train_loader, val_loader, test_loader, catalog_ds

  