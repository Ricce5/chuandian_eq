import importlib
import logging
import os

import torch
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel

from .model_routing import get_train_step_module, is_tpp_family

logger = logging.getLogger(__name__)


def step_scheduler(scheduler, event: str, *, val_loss=None):
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

    mode_map = {
        "step": "step",
        "update": "step",
        "epoch": "epoch",
        "epoch_end": "epoch",
    }
    if event not in mode_map:
        raise ValueError("event must be one of {'step','update','epoch','epoch_end'}")
    mode = mode_map[event]

    if isinstance(scheduler, ReduceLROnPlateau):
        if mode == "epoch":
            if val_loss is None:
                raise ValueError("ReduceLROnPlateau requires a validation loss at epoch end.")
            scheduler.step(val_loss)
    elif isinstance(scheduler, LRScheduler) or hasattr(scheduler, "step"):
        if mode == "step":
            scheduler.step()
    else:
        raise ValueError(f"Unsupported scheduler type: {type(scheduler)}")


def _resolve_train_validate(model_name):
    module_name = get_train_step_module(model_name)
    module = importlib.import_module(f".{module_name}", package=__package__)
    return module.train, module.validate


def _parse_save_epochs(raw_save_epochs):
    if raw_save_epochs is None:
        return set()

    if isinstance(raw_save_epochs, str):
        try:
            return {int(x.strip()) for x in raw_save_epochs.split(",") if x.strip()}
        except Exception:
            return set()

    try:
        return {int(x) for x in raw_save_epochs}
    except Exception:
        return set()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )
    logger.info("Checkpoint saved at %s", path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def train_and_save(
    args,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    save_dir,
    device,
    index=1,
    writer=None,
):
    train, validate = _resolve_train_validate(args.model)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_interrupted_{index}.pth")
    save_epoch_set = _parse_save_epochs(getattr(args, "save_epochs", None))

    best_val_loss = float("inf")
    start_epoch = 0
    epoch = start_epoch

    accumulation_steps = getattr(args, "accumulation_steps", 1)
    use_ema = getattr(args, "use_ema", False)
    ema_decay = getattr(args, "ema_decay", 0.999)

    train_metrics = {}
    val_metrics = {}

    if use_ema:
        def _ema_avg_fn(ema_p, p, n):
            return ema_p * ema_decay + p * (1.0 - ema_decay)

        ema_model = AveragedModel(model, avg_fn=_ema_avg_fn).to(device)
        logger.info("Using EMA with decay %s", ema_decay)
    else:
        ema_model = None

    if os.path.exists(checkpoint_path):
        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info("Epoch %s", epoch + 1)

            train_kwargs = {}
            if is_tpp_family(args.model):
                train_kwargs["use_amp"] = getattr(args, "use_amp", False)

            train_loss, train_metrics = train(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                accumulation_steps,
                ema_model=ema_model,
                **train_kwargs,
            )
            val_loss, val_metrics = validate(
                val_loader,
                ema_model if use_ema and ema_model is not None else model,
                criterion,
                device,
            )

            step_scheduler(scheduler, event="epoch", val_loss=val_loss)

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
                for key, value in train_metrics.items():
                    writer.add_scalar(f"Metric/train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f"Metric/val/{key}", value, epoch)

            current_state_dict = ema_model.module.state_dict() if use_ema and ema_model is not None else model.state_dict()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, f"best_model_{index}.pth")
                torch.save(
                    {
                        "model_state_dict": current_state_dict,
                        "val_loss": best_val_loss,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "hyperparameters": OmegaConf.to_container(args, resolve=True),
                    },
                    best_model_path,
                )
                logger.info("New best validation loss: %.4f, saved to %s", best_val_loss, best_model_path)

            last_model_path = os.path.join(save_dir, f"last_model_{index}.pth")
            save_data_last = {
                "model_state_dict": current_state_dict,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "hyperparameters": OmegaConf.to_container(args, resolve=True),
            }
            torch.save(save_data_last, last_model_path)
            logger.info("Last model at epoch %s saved to %s", epoch + 1, last_model_path)

            current_epoch = epoch + 1
            if current_epoch in save_epoch_set:
                epoch_model_path = os.path.join(save_dir, f"epoch_{current_epoch}_model_{index}.pth")
                save_data_epoch = dict(save_data_last)
                save_data_epoch["epoch"] = current_epoch
                torch.save(save_data_epoch, epoch_model_path)
                logger.info("Saved epoch %s snapshot to %s", current_epoch, epoch_model_path)

    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving current state...")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_loss)

    return best_val_loss, {"train_metrics": train_metrics, "val_metrics": val_metrics}
