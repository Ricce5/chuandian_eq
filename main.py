# %%
import argparse
import importlib
import json
import logging
import os
import shutil

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from config import config_loader
from src.data.preparation import prepare_data, prepare_data_lstm, prepare_data_tpp
from src.models.builders import ModelBuilder
import src.train.config_setup as config_setup
import src.train.trainer as trainer
from src.train.model_routing import get_model_family, get_train_step_module
from src.utils.file_utils import create_save_dir, find_latest_model_path
from src.utils.logging_utils import setup_logging
from src.utils.utils import set_seed

# torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(__name__)


def _get_data_preparation_fn(model_name):
    model_name = model_name.lower()
    if model_name == "lstm":
        return prepare_data_lstm

    family = get_model_family(model_name)
    if family in {"classifier", "regressor"}:
        return prepare_data
    return prepare_data_tpp


def get_model_and_data(args, base_path, _device=None):
    model_name = args.model.lower()
    train_step_module = get_train_step_module(model_name)
    train_step = importlib.import_module(f"src.train.{train_step_module}")
    data_func = _get_data_preparation_fn(model_name)

    df, train_loader, val_loader, test_loader, _ = data_func(args, base_path)
    return train_step, df, train_loader, val_loader, test_loader


def _load_resume_checkpoint(args, device):
    resume_path = getattr(args, "resume_path", None)
    if not resume_path:
        LOGGER.info("No resume path provided. Training will start from scratch.")
        return None
    if not os.path.exists(resume_path):
        LOGGER.warning("Resume path '%s' not found. Training will start from scratch.", resume_path)
        return None

    LOGGER.info("Loading checkpoint from: %s", resume_path)
    return torch.load(resume_path, map_location=device, weights_only=False)


def _build_test_checkpoint_path(args_cli, save_dir):
    if args_cli.ckpt_select == "epoch":
        if args_cli.ckpt_epoch is None:
            raise ValueError('When --ckpt_select is "epoch", --ckpt_epoch must be provided')
        return f"{save_dir}/epoch_{args_cli.ckpt_epoch}_model_{args_cli.trial_index}.pth"
    return f"{save_dir}/{args_cli.ckpt_select}_model_{args_cli.trial_index}.pth"


def _resolve_test_threshold(args_cli, checkpoint):
    cli_thresh = getattr(args_cli, "threshold", None)
    if cli_thresh is not None:
        LOGGER.info("Using threshold %s from CLI argument for testing.", cli_thresh)
        return cli_thresh

    if getattr(args_cli, "no_val_threshold", False):
        return None

    if isinstance(checkpoint, dict):
        val_metrics = checkpoint.get("val_metrics", {})
        if isinstance(val_metrics, dict) and "threshold" in val_metrics:
            threshold = val_metrics["threshold"]
            LOGGER.info("Using threshold %s from checkpoint val_metrics for testing.", threshold)
            return threshold
    return None


def _build_metrics_filename(args_cli):
    if args_cli.ckpt_select == "epoch":
        return f"metrics_test_epoch_{args_cli.ckpt_epoch}_{args_cli.trial_index}.json"
    return f"metrics_test_{args_cli.ckpt_select}_{args_cli.trial_index}.json"


def objective(trial, args):
    args.epochs = 120
    args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # args.scheduler_factor = trial.suggest_uniform('scheduler_factor', 0.1, 0.9)
    # args.scheduler_patience = trial.suggest_int('scheduler_patience', 2, 6)
    # args.scheduler_threshold = trial.suggest_float('scheduler_threshold', 1e-5, 1e-3, log=True)
    # args.scheduler_min_lr = trial.suggest_float('scheduler_min_lr', 1e-6, 1e-4, log=True)
    args.warnup_ratio = trial.suggest_uniform("warnup_ratio", 0, 0.2)
    args.scheduler_type = trial.suggest_categorical("scheduler_type", ["hf_cosine", "cosine", "hf_linear", "hf_constant"])

    shutil.copy(args_cli.config, f"{args.save_dir}/config.yaml")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard", f"trial_{trial.number}"))

    _train_step, _df, train_loader, val_loader, _test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
    model, criterion, optimizer, scheduler, args = config_setup.setup_config(args, device, train_loader)

    LOGGER.info("Trial %s", trial.number)

    val_loss, _best_metrics = trainer.train_and_save(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        device=device,
        index=trial.number,
        writer=writer,
    )
    torch.cuda.empty_cache()
    return val_loss


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test", "optuna"], default="train", help="Run mode: train or test")
    parser.add_argument("--model", type=str, choices=ModelBuilder.list_available(), required=True, help="Model name")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to load checkpoint for test mode")
    parser.add_argument("--trial_index", type=int, default=1, help="Index of the trial for optuna")
    parser.add_argument(
        "--ckpt_select",
        type=str,
        choices=["best", "last", "epoch"],
        default="best",
        help="Which checkpoint to use in test mode (best, last, or epoch)",
    )
    parser.add_argument("--ckpt_epoch", type=int, default=None, help="Epoch number to load when --ckpt_select epoch")
    parser.add_argument("--threshold", type=float, default=None, help="If provided, use this threshold for classification test (overrides checkpoint)")
    parser.add_argument("--no_val_threshold", action="store_true", default=False, help="Do not use threshold stored in checkpoint val_metrics")

    args_cli = parser.parse_args()
    if args_cli.config is None:
        args_cli.config = f"config/{args_cli.model}.yaml"

    args = config_loader.load_args_from_yaml(args_cli.config)
    assert args.model.lower() == args_cli.model.lower(), "Model name in config must match command line argument"
    args.model = args.model.lower()

    seed = getattr(args, "seed", 0)
    set_seed(seed)

    # Save directory: create or load based on mode
    if args_cli.mode in {"train", "optuna"}:
        args.save_dir = args_cli.checkpoint_dir or create_save_dir(base_dir="checkpoints", model_name=args.model)
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = args_cli.checkpoint_dir or find_latest_model_path(args_cli.model)

    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_id}" if args.cuda else "cpu")
    setup_logging(level=getattr(args, "log_level", None), log_file=os.path.join(args.save_dir, "run.log"))

    if args_cli.mode == "train":
        config_path = f"{args.save_dir}/config.yaml"
        shutil.copy(args_cli.config, config_path)
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

        _train_step, _df, train_loader, val_loader, _test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
        checkpoint = _load_resume_checkpoint(args, device)

        model, criterion, optimizer, scheduler, args = config_setup.setup_config(
            args,
            device,
            train_dataloader=train_loader,
            checkpoint=checkpoint,
            restore_weights=(checkpoint is not None),
        )

        _val_loss, _metrics = trainer.train_and_save(
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
        )

    elif args_cli.mode == "test":
        checkpoint_path = _build_test_checkpoint_path(args_cli, args.save_dir)
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
            _results, metrics = train_step.test(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                save_dir=args.save_dir,
            )
            metrics["num_events_train"] = args.num_events_train
            metrics["num_events_val"] = args.num_events_val
            metrics["num_events_test"] = args.num_events_test
        else:
            test_kwargs = {
                "model": model,
                "criterion": criterion,
                "data_loader": test_loader,
                "device": device,
                "save_dir": args.save_dir,
            }
            if getattr(args, "task_type", None) == "classification":
                test_kwargs["threshold"] = _resolve_test_threshold(args_cli, checkpoint)

            test_loss, metrics = train_step.test(**test_kwargs)
            LOGGER.info("Test loss: %s", test_loss)
            train_step.visualize_results(model, train_loader, val_loader, test_loader, device, args.save_dir)

        metrics_name = _build_metrics_filename(args_cli)
        with open(os.path.join(args.save_dir, metrics_name), "w") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved test metrics to %s", os.path.join(args.save_dir, metrics_name))
        LOGGER.info("Metrics: %s", metrics)
        torch.cuda.empty_cache()

    elif args_cli.mode == "optuna":
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=10)

        LOGGER.info("Best trial:")
        LOGGER.info("%s", study.best_trial.params)
