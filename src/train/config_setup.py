import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from .scheduler import WarmupLinearDecay, NoOpScheduler
from torch import nn
import argparse
from argparse import Namespace
import os
from omegaconf import DictConfig, ListConfig, OmegaConf
import typing

def _prune_to_schema(src, schema):
    """
    递归裁剪 src，只保留 schema 中存在的键/结构。
    - Dict：仅保留 schema 有的键，并对子项递归裁剪
    - List：按 schema 的第 0 个元素作为模板递归裁剪；若 schema 为空列表，则直接返回空列表
    - 原子值：直接返回 src
    """
    # Dict 匹配
    if isinstance(schema, DictConfig) and isinstance(src, DictConfig):
        out = OmegaConf.create({})
        for k in schema.keys():
            if k in src:
                out[k] = _prune_to_schema(src[k], schema[k])
        return out

    # List 匹配
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



def freeze_model_parts(model, freeze_keywords=None):
    if freeze_keywords is None:
        freeze_keywords = []

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in freeze_keywords):
            param.requires_grad = False
            print(f"Froze parameter: {name}")

def load_model_weights(model, checkpoint_state_dict, load_specific_parts=None):
    """
    加载模型权重
    
    参数:
        model (nn.Module): 目标模型
        checkpoint_state_dict (dict): 从检查点加载的权重
        load_specific_parts (list, optional): 需要加载的部分权重的关键字列表。默认为None，加载整个模型。
    
    返回:
        None
    """
    model_state_dict = model.state_dict()  # 获取模型当前的state_dict
    print(f"load_specific_parts: {load_specific_parts}")  # 调试信息
    if load_specific_parts is not None:
        # 选择性加载权重到指定部分
        for name, param in checkpoint_state_dict.items():
            if any(keyword in name for keyword in load_specific_parts) and name in model_state_dict:
                # 检查形状是否匹配
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
                    print(f"Loaded part: {name}")
                else:
                    print(f"Warning: Shape mismatch for {name} (checkpoint: {param.shape}, model: {model_state_dict[name].shape})")
            elif name not in model_state_dict:
                print(f"Warning: {name} not found in model!")
    else:
        # 默认加载整个模型
        load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
        print(f"Checkpoint loaded: {load_result}")

        # 检查哪些权重没有被加载
        if load_result.missing_keys:
            print(f"Warning: Missing keys (not loaded in the model): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys (present in checkpoint but not in model): {load_result.unexpected_keys}")

    # 最终赋值模型的state_dict
    model.load_state_dict(model_state_dict)




def load_model_from_checkpoint(model, checkpoint, freeze_parts=None, load_specific_parts=None):
    """
    从 checkpoint 中加载模型权重和参数，返回加载后的模型及辅助信息
    """
    load_model_weights(model, checkpoint['model_state_dict'], load_specific_parts=load_specific_parts)

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
 

    # 默认值
    args.start_epoch = 0
    args.best_val_loss = float('inf')

    if restore_weights and checkpoint is not None:
        model, args.start_epoch, args.best_val_loss = load_model_from_checkpoint(
            model, checkpoint, freeze_parts=getattr(args, 'freeze_parts', None) ,load_specific_parts=getattr(args, 'load_specific_parts', None)
        )

    # 损失函数
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    elif args.task_type == "regression":
        # criterion = nn.MSELoss()
        criterion = nn.HuberLoss(delta=0.5)
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
    warmup_ratio = getattr(args, 'warmup_ratio', 0.1) 
    try:
        warmup_ratio = float(warmup_ratio)
    except ValueError:
        warmup_ratio = 0.1
        
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    total_steps = len(train_dataloader) * args.epochs // accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)

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
