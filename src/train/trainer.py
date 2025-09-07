import os
import torch
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel




def step_scheduler(scheduler, event='epoch', val_loss=None):
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

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
            torch.optim.lr_scheduler.ConstantLR,
            torch.optim.lr_scheduler.SequentialLR)):
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

    classifier_models = [
        "classifier", "classifier_stm", "classifier_se", "clf_attnpl", "clf_attnpl_t",
        "classifier_tm_s", "clf_tm_attnpl", "clf_tm_attnpl_t", "classifier_stm_s",
        "clf_tm_cv_attnpl_t", "clf_mixer_attnpl_t"
    ]
    regressor_models = ["regressor", "lstm", "reg_attnpl", "reg_mixer_attnpl_t"]
    tpp_models = ["thp", "rtpp", "mtpp", "thp_deltat", "mhp", "btpp","etas"]
    tpp_m_models = ["mixer_tpp"]
    print(args.model)
    if args.model in classifier_models:
        from .classifier_train_step import train, validate
    elif args.model in regressor_models:
        from .regressor_train_step import train, validate
    elif args.model in tpp_models:
        from .tpp_train_step import train, validate
    elif args.model in tpp_m_models:
        from .tpp_m_train_step import train, validate
    else:
        raise ValueError(f"Unsupported model class: {args.model}")

    print(args.model)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_interrupted_{index}.pth')
    best_val_loss = float('inf')
    best_model_wts = None
    start_epoch = 0
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    train_metrics = {}
    val_metrics = {}   
    
    use_ema = getattr(args, 'use_ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
   
    if use_ema:
        def _ema_avg_fn(ema_p, p, n):  # 把超参透传进来
            return ema_p * ema_decay + p * (1.0 - ema_decay)
        ema_model = AveragedModel(model, avg_fn=_ema_avg_fn).to(device)
        print(f"Using EMA with decay {ema_decay}")
    else:
         ema_model = None

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
     
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")

            train_loss, train_metrics = train(train_loader, model, criterion, optimizer, scheduler, device,accumulation_steps, ema_model=ema_model)
            val_loss, val_metrics = validate(val_loader, 
                                              ema_model if use_ema and ema_model is not None else model,
                                            criterion, device)

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
                best_model_wts = ema_model.module.state_dict() if use_ema and ema_model is not None else model.state_dict()

                save_data = {
                    'model_state_dict': best_model_wts,
                    'val_loss': best_val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'hyperparameters': OmegaConf.to_container(args, resolve=True),
                }

                best_model_path = os.path.join(save_dir, f'best_model_{index}.pth')
                torch.save(save_data, best_model_path)
                print(f"New best validation loss: {best_val_loss:.4f}, saved to {best_model_path}")

            last_model_path = os.path.join(save_dir, f'last_model_{index}.pth')
            save_data_last = {
                'model_state_dict': ema_model.module.state_dict() if use_ema and ema_model is not None else model.state_dict(),
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'hyperparameters': OmegaConf.to_container(args, resolve=True),
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


