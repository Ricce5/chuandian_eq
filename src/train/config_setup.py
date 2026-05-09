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


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _log_name_preview(title, names, level=logging.INFO, limit=8):
    names = list(names)
    if not names:
        return
    preview = names[:limit]
    suffix = "" if len(names) <= limit else f" ... (+{len(names) - limit} more)"
    logger.log(level, "%s: %s%s", title, preview, suffix)


def _audit_keyword_coverage(context, keywords, population_names, positive_names, limit=8):
    keywords = list(keywords or [])
    if not keywords:
        return

    population = set(population_names)
    positive = set(positive_names)
    for keyword in keywords:
        target = {name for name in population if keyword in name}
        hit = {name for name in positive if keyword in name}
        missing = sorted(target - hit)
        logger.info(
            "%s [%s]: target=%d, hit=%d, missing=%d, complete=%s",
            context,
            keyword,
            len(target),
            len(hit),
            len(missing),
            not missing,
        )
        if missing:
            _log_name_preview(
                f"{context} missing preview [{keyword}]",
                missing,
                level=logging.WARNING,
                limit=limit,
            )


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


def freeze_model_parts(model, freeze_keywords=None, allowed_names=None, exclude_keywords=None, log_each=False):
    """
    Freeze parameters that match freeze_keywords and optional allowed_names / exclude_keywords filters.
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
    frozen_names = []

    for name, param in model.named_parameters():
        if allowed_names is not None and name not in allowed_names:
            continue
        if any(k in name for k in freeze_keywords) and not any(e in name for e in exclude_keywords):
            param.requires_grad = False
            frozen_names.append(name)
            if log_each:
                logger.info("Froze parameter: %s", name)

    return frozen_names


def apply_freeze_policy(
    model,
    freeze_parts=None,
    exclude_freeze_parts=None,
    freeze_loaded_only=True,
    loaded_names=None,
    context="setup",
):
    freeze_parts = list(freeze_parts or [])
    exclude_freeze_parts = list(exclude_freeze_parts or [])
    freeze_loaded_only = _to_bool(freeze_loaded_only, default=True)

    if not freeze_parts:
        logger.info("Freeze skipped (%s): freeze_parts is empty.", context)
        return []

    allow_loaded_scope = freeze_loaded_only and loaded_names is not None
    if freeze_loaded_only and loaded_names is None:
        logger.warning(
            "freeze_loaded_only=True but no loaded parameter scope provided (%s); "
            "falling back to freezing all matched parameters.",
            context,
        )

    frozen_names = freeze_model_parts(
        model,
        freeze_keywords=freeze_parts,
        allowed_names=loaded_names if allow_loaded_scope else None,
        exclude_keywords=exclude_freeze_parts,
    )

    mode = "loaded_only" if allow_loaded_scope else "all_matched"
    logger.info(
        "Freeze summary (%s): mode=%s, freeze_parts=%s, exclude_freeze_parts=%s, frozen=%d",
        context,
        mode,
        freeze_parts,
        exclude_freeze_parts,
        len(frozen_names),
    )
    if allow_loaded_scope:
        logger.info("Freeze scope (%s): loaded parameter count=%d", context, len(loaded_names))

    if frozen_names:
        _log_name_preview(f"Frozen parameter preview ({context})", frozen_names, level=logging.INFO, limit=10)
    else:
        logger.warning(
            "Freeze matched nothing (%s). Check freeze_parts / exclude_freeze_parts keywords.",
            context,
        )

    return frozen_names

def load_model_weights(model, checkpoint_state_dict, load_specific_parts=None):
    """
    Load model weights; return loaded_names: Set[str], indicating the parameter names that were actually loaded into model_state_dict.
    """
    model_state_dict = model.state_dict()
    loaded_names = set()
    shape_mismatch_names = []
    not_found_names = []

    logger.info("Loading model weights: load_specific_parts=%s", load_specific_parts)
    if load_specific_parts is not None:
        target_model_names = {
            name for name in model_state_dict
            if any(keyword in name for keyword in load_specific_parts)
        }
        ckpt_matched_in_model_names = {
            name for name in checkpoint_state_dict
            if any(keyword in name for keyword in load_specific_parts) and name in model_state_dict
        }

        for name, param in checkpoint_state_dict.items():
            if any(keyword in name for keyword in load_specific_parts) and name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
                    loaded_names.add(name)
                else:
                    shape_mismatch_names.append(name)
            elif any(keyword in name for keyword in load_specific_parts) and name not in model_state_dict:
                not_found_names.append(name)

        missing_in_checkpoint_names = sorted(target_model_names - ckpt_matched_in_model_names)
        missing_after_load_names = sorted(target_model_names - loaded_names)
        out_of_scope_loaded_names = sorted(loaded_names - target_model_names)

        logger.info(
            "Selective load summary: requested_parts=%s, target_in_model=%d, matched_in_checkpoint=%d, "
            "loaded=%d, missing_after_load=%d, shape_mismatch=%d, missing_in_model=%d, out_of_scope_loaded=%d",
            load_specific_parts,
            len(target_model_names),
            len(ckpt_matched_in_model_names),
            len(loaded_names),
            len(missing_after_load_names),
            len(shape_mismatch_names),
            len(not_found_names),
            len(out_of_scope_loaded_names),
        )

        _audit_keyword_coverage(
            "Selective load coverage",
            load_specific_parts,
            target_model_names,
            loaded_names,
        )

        if missing_in_checkpoint_names:
            _log_name_preview(
                "Checkpoint missing target parameter preview",
                missing_in_checkpoint_names,
                level=logging.WARNING,
                limit=10,
            )
        if missing_after_load_names:
            _log_name_preview(
                "Target not-loaded parameter preview",
                missing_after_load_names,
                level=logging.WARNING,
                limit=10,
            )
        if shape_mismatch_names:
            _log_name_preview("Shape mismatch parameter preview", shape_mismatch_names, level=logging.WARNING, limit=10)
        if not_found_names:
            _log_name_preview("Missing-in-model parameter preview", not_found_names, level=logging.WARNING, limit=10)
        if out_of_scope_loaded_names:
            _log_name_preview(
                "Out-of-scope loaded parameter preview",
                out_of_scope_loaded_names,
                level=logging.ERROR,
                limit=10,
            )
        if loaded_names:
            _log_name_preview("Loaded parameter preview", loaded_names, level=logging.INFO, limit=10)
    else:
        load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
        logger.info("Checkpoint loaded: %s", load_result)

        if load_result.missing_keys:
            logger.warning("Missing keys (not loaded in the model): %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys (present in checkpoint but not in model): %s", load_result.unexpected_keys)
        loaded_names = set(model_state_dict.keys()) - set(load_result.missing_keys)
        logger.info(
            "Full load summary: loaded=%d, missing=%d, unexpected=%d",
            len(loaded_names),
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    model.load_state_dict(model_state_dict)
    return loaded_names


def load_model_from_checkpoint(model, checkpoint, freeze_parts=None, load_specific_parts=None,
                               exclude_freeze_parts=None, freeze_loaded_only=True):
    """
    Load model weights from checkpoint and optionally freeze model parts.

    Args:
        freeze_loaded_only:
            - True: freeze only parameters that were actually loaded from checkpoint.
            - False: freeze all parameters matching freeze_parts, including randomly initialized ones.
    """
    loaded_names = load_model_weights(
        model,
        checkpoint['model_state_dict'],
        load_specific_parts=load_specific_parts,
    )

    if 'hyperparameters' in checkpoint:
        logger.info("Checkpoint hyperparameters loaded.")
    if 'train_metrics' in checkpoint:
        logger.info("Checkpoint train metrics available.")
    if 'val_metrics' in checkpoint:
        logger.info("Checkpoint val metrics available.")

    apply_freeze_policy(
        model,
        freeze_parts=freeze_parts,
        exclude_freeze_parts=exclude_freeze_parts,
        freeze_loaded_only=freeze_loaded_only,
        loaded_names=loaded_names,
        context="checkpoint_restore",
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

    # Optional data-uncertainty augmentation knobs used by training steps.
    setattr(model, 'mag_noise_scale', getattr(args, 'mag_noise_scale', 0.0))
    setattr(model, 'mag_noise_type', getattr(args, 'mag_noise_type', 'gaussian'))

    args.start_epoch = 0
    args.best_val_loss = float('inf')

    freeze_parts = getattr(args, 'freeze_parts', None)
    load_specific_parts = getattr(args, 'load_specific_parts', None)
    exclude_freeze_parts = getattr(args, 'exclude_freeze_parts', None)
    freeze_loaded_only = _to_bool(getattr(args, 'freeze_loaded_only', True), default=True)

    if restore_weights and checkpoint is not None:
        logger.info("Restoring model from checkpoint.")
        model, args.start_epoch, args.best_val_loss = load_model_from_checkpoint(
            model,
            checkpoint,
            freeze_parts=freeze_parts,
            load_specific_parts=load_specific_parts,
            exclude_freeze_parts=exclude_freeze_parts,
            freeze_loaded_only=freeze_loaded_only,
        )
    else:
        logger.info("No checkpoint restore requested; initializing model from scratch.")
        apply_freeze_policy(
            model,
            freeze_parts=freeze_parts,
            exclude_freeze_parts=exclude_freeze_parts,
            freeze_loaded_only=freeze_loaded_only,
            loaded_names=None,
            context="fresh_init",
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
        trainable_names = [name for name, _ in trainable_params]
        encoder_params = [
            (name, param) for name, param in trainable_params
            if any(keyword in name for keyword in encoder_keywords)
        ]
        logger.info("encoder_param_keywords: %s", encoder_keywords)
        encoder_param_names = [name for name, _ in encoder_params]
        logger.info(
            "encoder_lr grouping summary: lr=%s, matched=%d",
            encoder_lr,
            len(encoder_param_names),
        )
        _audit_keyword_coverage(
            "Encoder LR coverage",
            encoder_keywords,
            trainable_names,
            encoder_param_names,
        )
        _log_name_preview("encoder_lr matched preview", encoder_param_names, level=logging.INFO, limit=10)
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

    if restore_weights and checkpoint is not None:
        resume_optimizer = bool(getattr(args, "resume_optimizer", False))
        resume_scheduler = bool(getattr(args, "resume_scheduler", False))

        if not (resume_optimizer or resume_scheduler):
            args.start_epoch = 0
            args.best_val_loss = float("inf")
            logger.info(
                "Not resuming optimizer/scheduler state; reset training state to "
                "start_epoch=0, best_val_loss=inf."
            )

        if resume_scheduler:
            scheduler_state = checkpoint.get("scheduler_state_dict", None)
            if scheduler_state is None:
                logger.warning("resume_scheduler=True but checkpoint has no 'scheduler_state_dict'.")
            else:
                scheduler.load_state_dict(scheduler_state)
                logger.info("Loaded scheduler state from resume checkpoint.")

        if resume_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict", None)
            if optimizer_state is None:
                logger.warning("resume_optimizer=True but checkpoint has no 'optimizer_state_dict'.")
            else:
                optimizer.load_state_dict(optimizer_state)
                logger.info("Loaded optimizer state from resume checkpoint.")

        if resume_optimizer or resume_scheduler:
            args.start_epoch = int(checkpoint.get("epoch", args.start_epoch))
            args.best_val_loss = float(
                checkpoint.get(
                    "best_val_loss",
                    checkpoint.get("val_loss", args.best_val_loss),
                )
            )
            logger.info(
                "Resume training state enabled: start_epoch=%s, best_val_loss=%.6f",
                args.start_epoch,
                args.best_val_loss,
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
        assert args.scheduler_patience is not None, "scheduler_patience must be set for ReduceLROnPlateau"
        assert args.scheduler_threshold is not None, "scheduler_threshold must be set for ReduceLROnPlateau"
        assert args.scheduler_min_lr is not None, "scheduler_min_lr must be set for ReduceLROnPlateau"
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
