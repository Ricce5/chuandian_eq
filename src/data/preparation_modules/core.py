from omegaconf import OmegaConf

from src.data.event_pipeline import load_event_windows_for_task
from src.data.normalization import resolve_magnitude_bounds
from src.data.preprocessing import calculate_catalog_statistics

from .common import _loader_runtime_kwargs, _maybe_apply_train_subset, _split_with_event_loader


def prepare_data(args, base_dir):
    import src.data.event_loader as loader

    args.task_type = getattr(args, "task_type", "classification")
    mag_min, mag_max = resolve_magnitude_bounds(args)
    event_bundle = load_event_windows_for_task(
        base_dir=base_dir,
        task_type=args.task_type,
        Mc=args.Mc,
        Mf=args.Mf,
        Twindow=args.Twindow,
        Tfore=args.Tfore,
        dt=args.dt,
        context_len=getattr(args, 'context_len', 0),
    )
    df = event_bundle.df
    statistics = calculate_catalog_statistics(df)
    args.stats = OmegaConf.create(statistics)

    dataset = loader.EventDataset(
        event_bundle.array_dict,
        args.Mf,
        task_type=args.task_type,
        mag_min=mag_min,
        mag_max=mag_max,
    )
    train_set, val_set, test_set = _split_with_event_loader(
        dataset,
        task_type=args.task_type,
        loader_module=loader,
        args=args,
    )
    train_set = _maybe_apply_train_subset(
        train_set,
        args,
        args.task_type,
        loader_module=loader,
    )
    if getattr(args, 'use_sampler', False) and args.task_type == "classification":
        sampler = loader.get_balanced_sampler(train_set, seed=getattr(args, "seed", 0))
    else:
        sampler = None

    loader_kwargs = _loader_runtime_kwargs(args)
    train_loader = loader.get_dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        task_type=args.task_type,
        **loader_kwargs,
    )
    val_loader = loader.get_dataloader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        task_type=args.task_type,
        **loader_kwargs,
    )
    test_loader = loader.get_dataloader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        task_type=args.task_type,
        **loader_kwargs,
    )
    return df, train_loader, val_loader, test_loader, dataset
