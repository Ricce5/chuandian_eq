import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from .scheduler import WarmupLinearDecay
from torch import nn


def load_args_from_checkpoint(args, checkpoint):
    """
    从 checkpoint 中恢复超参数并更新 args。
    """
    if 'hyperparameters' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")
    
    restored_args = checkpoint['hyperparameters']
    for key, value in restored_args.items():
        setattr(args, key, value)
    
    return args

def setup_config(args, device, model_class, train_dataloader=None, checkpoint=None, restore_weights=False):
    """
    根据 args 配置初始化模型、损失函数、优化器、调度器等训练组件
    """

    if restore_weights:
        if checkpoint is None:
            raise ValueError("Checkpoint must be provided if restore_weights is True.")
        args = load_args_from_checkpoint(args, checkpoint)
        model = model_class(args, device=device)
        load_result = model.load_state_dict(checkpoint['model_state_dict'])
        print("hyperparameters:", checkpoint['hyperparameters'])
        print("train_metrics:", checkpoint['train_metrics'])
        print("val_metrics:", checkpoint['val_metrics'])
        print("Checkpoint loaded:", load_result)
    else:
        model = model_class(args, device=device)
        
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas = (0.9, 0.99)
    )

    scheduler = get_scheduler(args.scheduler_type, optimizer, args, train_dataloader=train_dataloader)

    return model, criterion, optimizer, scheduler, args



def get_scheduler(scheduler_type, optimizer, args, train_dataloader=None):
    """
    根据参数返回对应的学习率调度器
    支持 PyTorch 和 Hugging Face 的调度器
    """
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=args.scheduler_threshold,
            min_lr=args.scheduler_min_lr
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.scheduler_min_lr
        )
    elif scheduler_type == "hf_cosine":
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "hf_linear":
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "hf_constant":
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == "step_warmup":
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        step_size = int(args.step_lr_step_size_ratio*total_steps)  # e.g., every N steps to decay
        gamma = args.step_lr_gamma               # decay factor

        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_type == "warmup_linear_decay":
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        return WarmupLinearDecay(
            optimizer,
            base_lr=args.learning_rate,
            min_lr=args.scheduler_min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )   

    else:
        raise ValueError("Invalid scheduler type. Choose from 'plateau', 'cosine', 'hf_cosine', 'hf_linear', 'hf_constant'.")