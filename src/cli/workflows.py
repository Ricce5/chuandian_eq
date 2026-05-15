import logging
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

import src.train.config_setup as config_setup
import src.train.trainer as trainer
from .common import build_metrics_filename, build_test_checkpoint_path, resolve_test_threshold, save_metrics
from .data_factory import get_model_and_data

LOGGER = logging.getLogger(__name__)


def _safe_event_count(args, key):
    value = getattr(args, key, 0)
    if value is None:
        return 0
    return int(value)


def load_resume_checkpoint(args, device):
    resume_path = getattr(args, "resume_path", None)
    if not resume_path:
        LOGGER.info("No resume path provided. Training will start from scratch.")
        return None
    if not os.path.exists(resume_path):
        LOGGER.warning("Resume path '%s' not found. Training will start from scratch.", resume_path)
        return None

    LOGGER.info("Loading checkpoint from: %s", resume_path)
    return torch.load(resume_path, map_location=device, weights_only=False)


def run_train(args_cli, args, device):
    config_path = f"{args.save_dir}/config.yaml"
    src_cfg = os.path.abspath(str(args_cli.config))
    dst_cfg = os.path.abspath(config_path)
    if src_cfg != dst_cfg:
        shutil.copy(args_cli.config, config_path)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))
    try:
        _train_step, _df, train_loader, val_loader, _test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
        checkpoint = load_resume_checkpoint(args, device)

        model, criterion, optimizer, scheduler, args = config_setup.setup_config(
            args,
            device,
            train_dataloader=train_loader,
            checkpoint=checkpoint,
            restore_weights=(checkpoint is not None),
        )

        trainer.train_and_save(
            args=args,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=args.save_dir,
            device=device,
            index=1,
            writer=writer,
            start_epoch=int(getattr(args, "start_epoch", 0)),
            best_val_loss=float(getattr(args, "best_val_loss", float("inf"))),
        )
    finally:
        writer.close()


def run_test(args_cli, args, device):
    checkpoint_path = build_test_checkpoint_path(args_cli, args.save_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = config_setup.load_args_from_checkpoint(args, checkpoint)
    args.minibatch_training = False
    args.load_specific_parts = None
    args.use_sampler = False
    args.model = args.model.lower()
    if getattr(args, "task_type", None) == "tpp" and getattr(args, "event_drop_prob", 0.0):
        LOGGER.info("Disabling train-time TPP event dropping during evaluation.")
        args.event_drop_prob = 0.0

    train_step, _df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
    model, criterion, optimizer, scheduler, args = config_setup.setup_config(
        args,
        device,
        train_dataloader=train_loader,
        checkpoint=checkpoint,
        restore_weights=True,
    )

    if args.task_type == "tpp":
        test_nll_kwargs = {
            "reduction": getattr(args, "loss_reduction", None),
        }
        _results, metrics = train_step.test(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_dir=args.save_dir,
            nll_kwargs=test_nll_kwargs,
        )
        metrics["num_events_train"] = _safe_event_count(args, "num_events_train")
        metrics["num_events_val"] = _safe_event_count(args, "num_events_val")
        metrics["num_events_test"] = _safe_event_count(args, "num_events_test")
    else:
        test_kwargs = {
            "model": model,
            "criterion": criterion,
            "data_loader": test_loader,
            "device": device,
            "save_dir": args.save_dir,
        }
        if getattr(args, "task_type", None) == "classification":
            test_kwargs["threshold"] = resolve_test_threshold(args_cli, checkpoint)

        test_loss, metrics = train_step.test(**test_kwargs)
        LOGGER.info("Test loss: %s", test_loss)
        train_step.visualize_results(model, train_loader, val_loader, test_loader, device, args.save_dir)

    metrics_name = build_metrics_filename(args_cli)
    save_metrics(args.save_dir, metrics_name, metrics)
    torch.cuda.empty_cache()
