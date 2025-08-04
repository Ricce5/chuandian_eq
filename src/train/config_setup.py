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

def clean_for_omegaconf(d):
    # 递归清理输入，去除不支持的 Union 类型字段和复杂对象
    if isinstance(d, (DictConfig, dict)):
        new_d = {}
        for k, v in d.items():
            # 跳过Union类型字段（类型信息在 __origin__）
            if hasattr(v, '__origin__') and v.__origin__ is typing.Union:
                continue
            # 递归清理子字段
            if isinstance(v, (DictConfig, dict, ListConfig, list)):
                new_d[k] = clean_for_omegaconf(v)
            # 对于类实例，转成 dict 再清理
            elif hasattr(v, '__dict__'):
                new_d[k] = clean_for_omegaconf(vars(v))
            else:
                new_d[k] = v
        return new_d
    elif isinstance(d, (ListConfig, list)):
        return [clean_for_omegaconf(x) for x in d]
    else:
        return d
    
def dict_to_namespace(d):
    """递归地将字典转成 argparse.Namespace"""
    ns = argparse.Namespace()
    for k, v in d.items():
       setattr(ns, k, v)
    return ns

def load_args_from_checkpoint(args, checkpoint):
    if 'hyperparameters' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")

    restored_args = checkpoint['hyperparameters']
    if isinstance(restored_args, (DictConfig, ListConfig)):
        restored_args = OmegaConf.to_container(restored_args, resolve=True)

    restored_args_clean = clean_for_omegaconf(restored_args)
    base_dict = vars(args) if args is not None and not isinstance(args, dict) else (args or {})

    merged_dict = dict(base_dict)
    merged_dict.update(restored_args_clean)

    merged_namespace = dict_to_namespace(merged_dict)
    return merged_namespace


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
