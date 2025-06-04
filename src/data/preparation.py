from src.utils.file_utils import save_or_load_data
from src.data.preprocessing import load_and_filter_catalog


def prepare_data(args, base_dir="data/CD2021"):
    import src.data.event_loader as loader
    df = load_and_filter_catalog(base_dir, Mc=args.Mc)
    df_nl = loader.normalize_df(df)
    args.task_type = getattr(args, "task_type", "classification")
    task_prefix_map = {
        "classification": "classifier",
        "regression": "regressor",
        "count": "counter"
    }
    task_prefix = task_prefix_map.get(args.task_type)
    def generate_data():
        samples_list, array_dict = loader.get_list(
            df, df_nl,
            Mc=args.Mc,
            Mf=args.Mf,
            Twindow=args.Twindow,
            Tfore=args.Tfore,
            dt=args.dt,
            context_len=args.context_len
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
        context_len=args.context_len,
    )
    array_dict = cached["array_dict"]

    dataset = loader.EventDataset(array_dict, args.Mf)
    train_set, val_set, test_set = loader.split_dataset(
        dataset,
        by_time=args.split_by_time,
        train_ratio=0.8,
        val_ratio=0.1,
        time_order=getattr(args, 'time_order', ('train', 'val', 'test')),
    )
    if args.use_sampler and args.task_type == "classification":
        sampler = loader.get_balanced_sampler(train_set)
    else:
        sampler = None
    train_loader = loader.get_dataloader(train_set, batch_size=args.batch_size, shuffle=True, sampler=sampler)
    val_loader = loader.get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = loader.get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)
    return df, train_loader, val_loader, test_loader



