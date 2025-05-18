import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler



def step_scheduler(scheduler, event='epoch', val_loss=None):
    from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

    # 只在每个 epoch 时，传入 val_loss
    if event == 'epoch':
        if isinstance(scheduler, ReduceLROnPlateau):
            if val_loss is None:
                raise ValueError("ReduceLROnPlateau scheduler requires val_loss at epoch end.")
            scheduler.step(val_loss)
        elif isinstance(scheduler, LRScheduler):
            scheduler.step()
    # 每个 step 时调用
    elif event == 'batch':
        if isinstance(scheduler, LRScheduler) and isinstance(scheduler, (
            torch.optim.lr_scheduler.LambdaLR, 
            torch.optim.lr_scheduler.CyclicLR, 
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.MultiplicativeLR,
            torch.optim.lr_scheduler.LinearLR,
            torch.optim.lr_scheduler.ConstantLR)):
            scheduler.step()
        elif not isinstance(scheduler, LRScheduler) and hasattr(scheduler, 'step'):
            # 对非 _LRScheduler 的调度器，如 Hugging Face 的 schedulers
            scheduler.step()






def train_and_save(args, model, criterion, optimizer, scheduler, train_loader,
                   val_loader, save_dir, device, index=1, writer=None):
    
    if args.model == "Classifier":
        from .classifier_train_step import train, validate
    else:
        raise ValueError(f"Unsupported model class: {args.model}")
    print(args.model)
    best_val_loss = float('inf')  
    best_model_wts = None  
    os.makedirs(save_dir, exist_ok=True) 

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # 训练模型
        train_loss, train_metrics = train(train_loader, model, criterion, optimizer,scheduler, device)

        # 验证模型
        val_loss, val_metrics = validate(val_loader, model, criterion, device)

        # 更新学习率调度器
        step_scheduler(scheduler, event='epoch', val_loss=val_loss)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for k, v in train_metrics.items():
                writer.add_scalar(f"Metric/train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"Metric/val/{k}", v, epoch)
        # 计算最佳验证损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()  # 保存最佳模型的权重

            # 保存模型参数和超参数
            save_data = {
                'model_state_dict': best_model_wts,
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,  # 保存训练阶段的指标
                'val_metrics': val_metrics,      # 保存验证阶段的指标
                'hyperparameters': vars(args),   # 保存超参数到文件
            }

            file_path = os.path.join(save_dir, f'best_model_{index}.pth') # {model}_{dataset}_ep{epoch}_val{val_loss:.4f}.pth
            torch.save(save_data, file_path)
            print(f"New best validation loss: {best_val_loss:.4f}, saving model weights and hyperparameters to {file_path}.")

    # 返回最佳验证损失和验证阶段的指标
    return best_val_loss, {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }


