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





def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))

def train_and_save(args, model, criterion, optimizer, scheduler, train_loader,
                   val_loader, save_dir, device, index=1, writer=None):

    if args.model in ["Classifier", "Classifier_STM"]:
        from .classifier_train_step import train, validate
    else:
        raise ValueError(f"Unsupported model class: {args.model}")

    print(args.model)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_interrupted_{index}.pth')
    best_val_loss = float('inf')
    best_model_wts = None
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")

            train_loss, train_metrics = train(train_loader, model, criterion, optimizer, scheduler, device)
            val_loss, val_metrics = validate(val_loader, model, criterion, device)

            step_scheduler(scheduler, event='epoch', val_loss=val_loss)

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                for k, v in train_metrics.items():
                    writer.add_scalar(f"Metric/train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"Metric/val/{k}", v, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = model.state_dict()

                save_data = {
                    'model_state_dict': best_model_wts,
                    'val_loss': best_val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'hyperparameters': vars(args),
                }

                best_model_path = os.path.join(save_dir, f'best_model_{index}.pth')
                torch.save(save_data, best_model_path)
                print(f"New best validation loss: {best_val_loss:.4f}, saved to {best_model_path}")

            last_model_path = os.path.join(save_dir, f'last_model_{index}.pth')
            save_data_last = {
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'hyperparameters': vars(args),
            }

            torch.save(save_data_last, last_model_path)
            print(f"Last model at epoch {epoch+1} saved to {last_model_path}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_loss)

    return best_val_loss, {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }


