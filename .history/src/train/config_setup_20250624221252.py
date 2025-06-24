import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from .scheduler import WarmupLinearDecay, NoOpScheduler
from torch import nn
import os


def load_args_from_checkpoint(args, checkpoint):
    """
    从 checkpoint 中恢复超参数并更新 args。(处理 checkpoint 中参数与 config 中参数不一致的情况)
    """
    if 'hyperparameters' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")
    
    restored_args = checkpoint['hyperparameters']
    for key, value in restored_args.items():
        setattr(args, key, value)
    
    return args

def freeze_model_parts(model, freeze_keywords=None):
    if freeze_keywords is None:
        freeze_keywords = []

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in freeze_keywords):
            param.requires_grad = False
            print(f"Froze parameter: {name}")


def load_model_from_checkpoint(model, checkpoint, freeze_parts=None):
    """
    从 checkpoint 中加载模型权重和参数，返回加载后的模型及辅助信息
    """
    load_result = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print("Checkpoint loaded:", load_result)

    if 'hyperparameters' in checkpoint:
        print("hyperparameters:", checkpoint['hyperparameters'])
    if 'train_metrics' in checkpoint:
        print("train_metrics:", checkpoint['train_metrics'])
    if 'val_metrics' in checkpoint:
        print("val_metrics:", checkpoint['val_metrics'])

    if freeze_parts:
        freeze_model_parts(model, freeze_keywords=freeze_parts)

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    return model, start_epoch, best_val_loss


def setup_config(args, device,train_dataloader=None, checkpoint=None, restore_weights=True):
    """
    初始化模型、优化器、调度器（支持从 checkpoint 加载训练或测试模型）
    """
    # model = model_class(args, device=device)
    from src.models.builders import ModelBuilder
    model_builder = ModelBuilder.by_name(args.model)()
    model = model_builder(args, device)
    # from src.models.Models import classifier
    # model = classifier(args, device=device)

    # 默认值
    args.start_epoch = 0
    args.best_val_loss = float('inf')

    if restore_weights and checkpoint is not None:
        model, args.start_epoch, args.best_val_loss = load_model_from_checkpoint(
            model, checkpoint, freeze_parts=getattr(args, 'freeze_parts', None)
        )

    # 损失函数
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    elif args.task_type == "count":
        criterion = nn.PoissonNLLLoss(log_input=False, full=False)
    elif args.task_type == "tpp":
        criterion = None
    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99)
    )

    scheduler = get_scheduler(args.scheduler_type, optimizer, args, train_dataloader=train_dataloader)

    return model, criterion, optimizer, scheduler, args



def get_scheduler(scheduler_type, optimizer, args, train_dataloader=None):
    """
    根据参数返回对应的学习率调度器
    支持 PyTorch 和 Hugging Face 的调度器
    """
    if args.warmup_ratio is not None:
        accumulation_steps = getattr(args, 'accumulation_steps', 1)
        total_steps = len(train_dataloader) * args.epochs // accumulation_steps
        warmup_steps = int(args.warmup_ratio * total_steps)

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
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "hf_linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "hf_constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == "step_warmup":
        step_size = int(args.step_lr_step_size_ratio * total_steps)  # 每N个步骤衰减一次
        gamma = args.step_lr_gamma  # 衰减因子

        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_type == "warmup_linear_decay":
        return WarmupLinearDecay(
            optimizer,
            base_lr=args.learning_rate,
            min_lr=args.scheduler_min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )   
    elif scheduler_type == "none":
         return NoOpScheduler(optimizer)
    else:
        raise ValueError("Invalid scheduler type. Choose from 'plateau', 'cosine', 'hf_cosine', 'hf_linear', 'hf_constant'.")
