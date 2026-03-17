import logging

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR, StepLR
from transformers import get_constant_schedule_with_warmup

from src.models.builders import ModelBuilder
from src.utils.binary_focal_loss import FocalLossWrapper

from .scheduler import CosineWithWarmupFloor, LinearWithWarmupFloor, NoOpScheduler, WarmupLinearDecay

logger = logging.getLogger(__name__)

def _prune_to_schema(src, schema):
    """
    Recursively prune `src`, keeping only the keys/structure present in `schema`.
    - Dict: Retain only the keys present in `schema` and recursively prune the sub-items.
    - List: Use the 0th element of `schema` as a template for recursive pruning; if `schema` is an empty list, return an empty list directly.
    - Atomic value: Return `src` as is.
    """
    # Dict matching
    if isinstance(schema, DictConfig) and isinstance(src, DictConfig):
        out = OmegaConf.create({})
        for k in schema.keys():
            if k in src:
                out[k] = _prune_to_schema(src[k], schema[k])
        return out

    # List matching
    if isinstance(schema, ListConfig) and isinstance(src, ListConfig):
        if len(schema) == 0:
            return OmegaConf.create([])
        template = schema[0]
        return OmegaConf.create([_prune_to_schema(v, template) for v in src])
    return src

def load_args_from_checkpoint(cfg, checkpoint):
    if 'hyperparameters' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")

    restored_cfg = OmegaConf.create(checkpoint['hyperparameters'])

    if cfg is not None:
        cfg_pruned = _prune_to_schema(cfg, restored_cfg)
        final_cfg = OmegaConf.merge(cfg_pruned, restored_cfg)
    else:
        final_cfg = restored_cfg
    # OmegaConf.set_struct(final_cfg, True)
    return final_cfg


def freeze_model_parts(model, freeze_keywords=None, allowed_names=None, exclude_keywords=None):
    """
    Freeze only the parameters that match the specified keywords and are within the allowed_names (the set of loaded parameters).
    Optional exclude_keywords can be used to exclude certain submodules (e.g., 'linear', 'fc1', 'fc2', etc.).
    input:
        model: nn.Module
        freeze_keywords: List[str]  # param names containing any of these keywords will be frozen
        allowed_names: Optional[Set[str]]  # select from these parameter names only
        exclude_keywords: List[str]  # parameters containing any of these keywords will not be frozen
    """
    if freeze_keywords is None:
        freeze_keywords = []
    if exclude_keywords is None:
        exclude_keywords = []

    allowed_names = set(allowed_names) if allowed_names is not None else None

    for name, param in model.named_parameters():
        if allowed_names is not None and name not in allowed_names:
            continue
        if any(k in name for k in freeze_keywords) and not any(e in name for e in exclude_keywords):
            param.requires_grad = False
            logger.info("Froze parameter: %s", name)

def load_model_weights(model, checkpoint_state_dict, load_specific_parts=None):
    """
    Load model weights; return loaded_names: Set[str], indicating the parameter names that were actually loaded into model_state_dict.
    """
    model_state_dict = model.state_dict()
    loaded_names = set()

    logger.info("load_specific_parts: %s", load_specific_parts)
    if load_specific_parts is not None:
        for name, param in checkpoint_state_dict.items():
            if any(keyword in name for keyword in load_specific_parts) and name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
                    loaded_names.add(name)  
                    logger.info("Loaded part: %s", name)
                else:
                    logger.warning(
                        "Shape mismatch for %s (checkpoint: %s, model: %s)",
                        name,
                        param.shape,
                        model_state_dict[name].shape,
                    )
            elif name not in model_state_dict:
                logger.warning("%s not found in model", name)
    else:
        load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
        logger.info("Checkpoint loaded: %s", load_result)

        if load_result.missing_keys:
            logger.warning("Missing keys (not loaded in the model): %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys (present in checkpoint but not in model): %s", load_result.unexpected_keys)
        loaded_names = set(model_state_dict.keys()) - set(load_result.missing_keys)

    model.load_state_dict(model_state_dict)
    return loaded_names  


def load_model_from_checkpoint(model, checkpoint, freeze_parts=None, load_specific_parts=None,
                               exclude_freeze_parts=None):
    """
    Load model weights from checkpoint and optionally freeze: only freeze the "loaded" parts this time
    """
    loaded_names = load_model_weights(model, checkpoint['model_state_dict'], load_specific_parts=load_specific_parts)

    if 'hyperparameters' in checkpoint:
        logger.info("hyperparameters: %s", checkpoint['hyperparameters'])
    if 'train_metrics' in checkpoint:
        logger.info("train_metrics: %s", checkpoint['train_metrics'])
    if 'val_metrics' in checkpoint:
        logger.info("val_metrics: %s", checkpoint['val_metrics'])

    if freeze_parts:
        freeze_model_parts(
            model,
            freeze_keywords=freeze_parts,
            allowed_names=loaded_names,                # only freeze what was actually loaded
            exclude_keywords=exclude_freeze_parts or []
        )

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    return model, start_epoch, best_val_loss


def setup_config(args, device, train_dataloader=None, checkpoint=None, restore_weights=True):
    """
    Initialize the model, optimizer, and scheduler (supports loading from a checkpoint for training or testing).
    """
    model_builder = ModelBuilder.by_name(args.model)()
    model = model_builder(args, device)

    args.start_epoch = 0
    args.best_val_loss = float('inf')

    if restore_weights and checkpoint is not None:
        model, args.start_epoch, args.best_val_loss = load_model_from_checkpoint(
            model,
            checkpoint,
            freeze_parts=getattr(args, 'freeze_parts', None),
            load_specific_parts=getattr(args, 'load_specific_parts', None),
            exclude_freeze_parts=getattr(args, 'exclude_freeze_parts', None)
        )

    # ------------------------------------------------------------------
    # Criterion
    # ------------------------------------------------------------------
    criterion_name = getattr(args, 'criterion_name', None)
    criterion_cfg = getattr(args, 'criterion_cfg', {})

    if args.task_type == "classification":
        if criterion_name is None:
            criterion_name = 'bce'
        if criterion_name == 'bce':
            criterion = nn.BCEWithLogitsLoss(**criterion_cfg)
        elif criterion_name == 'focal':
            criterion = FocalLossWrapper(**criterion_cfg)
        else:
            raise ValueError(f"Unsupported criterion_name for classification: {criterion_name}")
        logger.info("Using classification criterion: %s", criterion_name)
        logger.info("Criterion config: %s", criterion_cfg)

    elif args.task_type == "regression":
        if criterion_name is None:
            criterion_name = 'mse'
        if criterion_name == 'mse':
            criterion = nn.MSELoss(**criterion_cfg)
        elif criterion_name == 'mae':
            criterion = nn.L1Loss(**criterion_cfg)
        elif criterion_name == 'huber':
            criterion = nn.HuberLoss(**criterion_cfg)
        elif criterion_name == 'smooth_l1':
            criterion = nn.SmoothL1Loss(**criterion_cfg)
        else:
            raise ValueError(f"Unsupported criterion_name for regression: {criterion_name}")
        logger.info("Using regression criterion: %s", criterion_name)
        logger.info("Criterion config: %s", criterion_cfg)

    elif args.task_type == "count":
        criterion = nn.PoissonNLLLoss(**criterion_cfg)
        logger.info("Using count criterion: PoissonNLLLoss")
        logger.info("Criterion config: %s", criterion_cfg)

    elif args.task_type == "tpp":
        criterion = None

    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")

    # ------------------------------------------------------------------
    # Optimizer param groups
    # ------------------------------------------------------------------
    trainable_params = [
        (name, param) for name, param in model.named_parameters()
        if param.requires_grad
    ]

    param_groups = []
    handled = set()

    def build_no_decay_names(model_obj):
        no_decay = set()
        for pname, param in model_obj.named_parameters():
            if pname.endswith(".bias") or param.ndim == 1:
                no_decay.add(pname)
        return no_decay

    no_decay_names = build_no_decay_names(model)

    encoder_lr = getattr(args, "encoder_learning_rate", None)
    encoder_keywords = getattr(args, "encoder_param_keywords", ["encoder"])

    def add_group_named(params_with_names, lr):
        if not params_with_names:
            return
        decay_params = [p for n, p in params_with_names if n not in no_decay_names]
        no_decay_params = [p for n, p in params_with_names if n in no_decay_names]

        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": lr,
                "weight_decay": args.weight_decay
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0
            })

    # Encoder params (optional different LR)
    if encoder_lr is not None:
        encoder_params = [
            (name, param) for name, param in trainable_params
            if any(keyword in name for keyword in encoder_keywords)
        ]
        logger.info("encoder_param_keywords: %s", encoder_keywords)
        logger.info("encoder_lr matches: %s", [name for name, _ in encoder_params])
        handled.update(id(p) for _, p in encoder_params)
        add_group_named(encoder_params, encoder_lr)

    # Background model params (optional)
    bg_lr = getattr(args, "bg_learning_rate", None)
    if hasattr(model, "bg_model") and model.bg_model is not None and bg_lr is not None:
        bg_params = [
            (name, param) for name, param in trainable_params
            if name.startswith("bg_model.") and id(param) not in handled
        ]
        handled.update(id(p) for _, p in bg_params)
        add_group_named(bg_params, bg_lr)

    # Remaining params
    remaining_params = [
        (name, param) for name, param in trainable_params
        if id(param) not in handled
    ]
    add_group_named(remaining_params, args.learning_rate)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.99)
    )

    scheduler = get_scheduler(
        args.scheduler_type,
        optimizer,
        args,
        train_dataloader=train_dataloader
    )

    return model, criterion, optimizer, scheduler, args




def get_scheduler(scheduler_type, optimizer, args, train_dataloader=None):
    """
    Return the configured learning-rate scheduler.
    """
    warmup_ratio = getattr(args, "warmup_ratio", 0.1)
    try:
        warmup_ratio = float(warmup_ratio)
    except (TypeError, ValueError):
        warmup_ratio = 0.1

    accumulation_steps = max(1, int(getattr(args, "accumulation_steps", 1)))

    def _get_total_steps():
        if train_dataloader is None:
            raise ValueError(
                f"Scheduler '{scheduler_type}' requires train_dataloader to compute total training steps."
            )
        return max(1, len(train_dataloader) * args.epochs // accumulation_steps)

    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=args.scheduler_threshold,
            min_lr=args.scheduler_min_lr,
        )

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.scheduler_min_lr,
        )

    if scheduler_type in {"hf_cosine", "hf_linear", "hf_constant", "step_warmup", "warmup_linear_decay"}:
        total_steps = _get_total_steps()
        warmup_steps = int(warmup_ratio * total_steps)

    if scheduler_type == "hf_cosine":
        return CosineWithWarmupFloor(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr=args.scheduler_min_lr,
        )

    if scheduler_type == "hf_linear":
        return LinearWithWarmupFloor(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr=args.scheduler_min_lr,
        )

    if scheduler_type == "hf_constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    if scheduler_type == "step_warmup":
        step_size = max(1, int(args.step_lr_step_size_ratio * total_steps))
        gamma = args.step_lr_gamma
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        if warmup_steps <= 0:
            return step_scheduler

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_steps],
        )

    if scheduler_type == "warmup_linear_decay":
        return WarmupLinearDecay(
            optimizer,
            base_lr=args.learning_rate,
            min_lr=args.scheduler_min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    if scheduler_type == "none":
        return NoOpScheduler(optimizer)

    raise ValueError(
        "Invalid scheduler type. Choose from 'plateau', 'cosine', 'hf_cosine', "
        "'hf_linear', 'hf_constant', 'step_warmup', 'warmup_linear_decay', 'none'."
    )


def load_and_prepare_model(checkpoint_path, device, compile=True):
    check_point = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = load_args_from_checkpoint(None, check_point)
    model_builder = ModelBuilder.by_name(args.model.lower())()
    model = model_builder(args, device)
    model, _, _ = load_model_from_checkpoint(model, check_point)
    if compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()
    return model, args
