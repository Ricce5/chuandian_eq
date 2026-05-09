import logging
import math
from functools import partial

from src.utils.catalog_utils import split_minibatches, split_sequence

logger = logging.getLogger(__name__)


def _drop_tpp_sequence_events(seq, drop_prob: float, min_nll_events: int):
    if not hasattr(seq, "drop_events") or drop_prob <= 0:
        return seq
    return seq.drop_events(drop_prob=drop_prob, min_nll_events=min_nll_events)


def _maybe_wrap_tpp_train_dataset(dataset, args):
    drop_prob = float(getattr(args, "event_drop_prob", 0.0) or 0.0)
    if drop_prob <= 0.0:
        return dataset

    min_nll_events = int(getattr(args, "event_drop_min_nll_events", 1))
    logger.info(
        "Applying train-time TPP event dropping with prob=%.4f and min_nll_events=%s",
        drop_prob,
        min_nll_events,
    )
    from src.data.tpp_dataset import TppDataset

    return TppDataset(
        dataset.sequences,
        sequence_transform=partial(
            _drop_tpp_sequence_events,
            drop_prob=drop_prob,
            min_nll_events=min_nll_events,
        ),
    )


def _auto_configure_global_bg_time_bounds(args, stats_source):
    """Populate bg_model_cfg global time bounds when global normalization is enabled."""
    bg_cfg = getattr(args, "bg_model_cfg", None)
    if bg_cfg is None:
        return

    time_normalization = str(getattr(bg_cfg, "time_normalization", "per_sequence")).strip().lower()
    if time_normalization != "global":
        return

    has_min = getattr(bg_cfg, "global_time_min", None) is not None
    has_max = getattr(bg_cfg, "global_time_max", None) is not None
    if has_min and has_max:
        return
    if has_min != has_max:
        raise ValueError(
            "bg_model_cfg.global_time_min/global_time_max must be both set or both unset "
            "when time_normalization='global'."
        )

    time_min = min(float(seq.t_start) for seq in stats_source)
    time_max = max(float(seq.t_end) for seq in stats_source)
    bg_cfg.global_time_min = float(time_min)
    bg_cfg.global_time_max = float(time_max)
    logger.info(
        "Auto-configured bg_model_cfg global time bounds: "
        "global_time_min=%.6f, global_time_max=%.6f",
        bg_cfg.global_time_min,
        bg_cfg.global_time_max,
    )


def prepare_data_tpp(args, base_dir):
    import torch
    from src.data.tpp_dataset import TppDataset
    import src.data.catalog as catalog
    import src.catalogs as catalogs # ensure catalogs are registered
    from src.catalogs.pathing import build_tpp_catalog_init_kwargs

    catalog_ds_class = catalog.Catalog.by_name(f"{args.dataset}-Standard")
    catalog_cfg = getattr(args, 'catalog_cfg', {})
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=catalog_ds_class,
        base_dir=base_dir,
        catalog_cfg=catalog_cfg,
    )

    catalog_ds = catalog_ds_class(**init_kwargs)

    use_all_for_train = getattr(args, "use_all_data", False)

    stats_source = catalog_ds.train
    if use_all_for_train:
        stats_source = TppDataset(
            catalog_ds.train.sequences
            + catalog_ds.val.sequences
            + catalog_ds.test.sequences
        )
    _auto_configure_global_bg_time_bounds(args, stats_source)

    args.tau_mean = torch.cat([seq.inter_times[:-1] for seq in stats_source]).mean().item()
    args.time_min = torch.min(torch.tensor([seq.t_start for seq in stats_source])).item()
    args.tau_min = torch.cat([seq.inter_times[:-1] for seq in stats_source]).min().item()
    args.tau_max = torch.cat([seq.inter_times[:-1] for seq in stats_source]).max().item()
    args.tau_q05 = torch.cat([seq.inter_times[:-1] for seq in stats_source]).quantile(0.5).item()
    args.tau_q025 = torch.cat([seq.inter_times[:-1] for seq in stats_source]).quantile(0.025).item()
    args.mag_mean = torch.cat([seq.mag for seq in stats_source]).mean().item()
    args.time_max = torch.max(torch.tensor([seq.t_end for seq in stats_source])).item()
    args.time_mean = torch.cat([seq.arrival_times[:-1] for seq in stats_source]).mean().item()
    args.mag_completeness = catalog_ds.metadata["mag_completeness"]

    config_b = getattr(args, "richter_b", None)

    if config_b is not None:
        args.richter_b_mle = float(config_b)
    elif "richter_b" in catalog_ds.metadata:
        args.richter_b_mle = catalog_ds.metadata["richter_b"]
    else:
        mag_roundoff_error = catalog_ds.metadata.get("mag_roundoff_error", 0.0)
        args.richter_b_mle = math.log10(math.exp(1)) / (
            args.mag_mean - args.mag_completeness + 0.5 * mag_roundoff_error
        )

    if getattr(args, 'use_b_updater', False):
        from src.data.bayesian_b_updater import BayesianGRBUpdater
        b_updater = BayesianGRBUpdater(
            Mc=args.mag_completeness,
            **args.b_updater_cfg,
            mag_key="mag",
            write_back=True,
        )
        catalog_ds.set_b_updater(b_updater)
        catalog_ds.estimate_gr_b()
        logger.info(
            "Using Bayesian GR b-value updater with delta=%s, a0=%s, init b=%.4f, mag_completeness=%s",
            b_updater.delta,
            b_updater.a0,
            b_updater.init_b_target,
            b_updater.Mc,
        )

    if getattr(args, 'use_double_precision', False):
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.double()
        catalog_ds.full_sequence = catalog_ds.full_sequence.double()
        args.precision = 64
    else:
        for cat in (catalog_ds.train, catalog_ds.val, catalog_ds.test):
            for seq in cat:
                seq.float()
        catalog_ds.full_sequence = catalog_ds.full_sequence.float()
        args.precision = 32

    if use_all_for_train:
        logger.info("TPP pretrain mode: using all splits as training, disabling val/test loaders.")
        if getattr(args, 'minibatch_training', False):
            max_events = getattr(args, 'max_seq_len', 2000)
            mean_nll_events = getattr(args, 'mean_nll_events', 300)
            train_dataset = split_sequence(catalog_ds.full_sequence, mean_nll_events, max_events)
        else:
            train_dataset = TppDataset([catalog_ds.full_sequence])
        args.num_events_train = sum(seq.num_nll_events for seq in train_dataset)
        train_dataset = _maybe_wrap_tpp_train_dataset(train_dataset, args)
        train_loader = train_dataset.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            pad_token_id=getattr(args, 'pad_token_id', None),
        )
        val_loader = None
        test_loader = None
        args.num_events_val = 0
    else:
        args.num_events_train = sum(seq.num_nll_events for seq in catalog_ds.train)
        args.num_events_val = sum(seq.num_nll_events for seq in catalog_ds.val)
        args.num_events_test = sum(seq.num_nll_events for seq in catalog_ds.test)
        logger.info("Number of training events: %s", args.num_events_train)
        logger.info("Number of validation events: %s", args.num_events_val)
        logger.info("Number of test events: %s", args.num_events_test)
        if getattr(args, 'minibatch_training', False):
            logger.info("Splitting into minibatches")
            max_events = getattr(args, 'max_seq_len', 2000)
            mean_nll_events = getattr(args, 'mean_nll_events', 300)
            catalog_ds = split_minibatches(catalog_ds, mean_nll_events, max_events)

        train_dataset = _maybe_wrap_tpp_train_dataset(catalog_ds.train, args)
        train_loader = train_dataset.get_dataloader(
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

    return catalog_ds.full_sequence, train_loader, val_loader, test_loader, catalog_ds
