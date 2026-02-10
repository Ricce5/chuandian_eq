# %%
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import importlib
from config import config_loader
from src.utils.utils import set_seed
from src.utils.file_utils import create_save_dir,find_latest_model_path
from src.data.preparation import prepare_data,prepare_data_lstm,prepare_data_tpp
import src.train.config_setup as config_setup 
import src.train.trainer as trainer
from src.models.builders import ModelBuilder
import shutil
import os
import json
import yaml
import optuna
# torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


def get_model_and_data(args, base_path, device):
    # Map model names to their train step and data preparation functions
    model_map = {
        "classifier": ("src.train.classifier_train_step", prepare_data),
        "classifier_stm": ("src.train.classifier_train_step", prepare_data),
        "classifier_tm_s": ("src.train.classifier_train_step", prepare_data),
        "clf_tm_attnpl": ("src.train.classifier_train_step", prepare_data),
        "clf_tm_attnpl_t": ("src.train.classifier_train_step", prepare_data),
        "clf_tm_cv_attnpl_t": ("src.train.classifier_train_step", prepare_data),
        "classifier_stm_s": ("src.train.classifier_train_step", prepare_data),
        "classifier_se": ("src.train.classifier_train_step", prepare_data),
        "clf_attnpl": ("src.train.classifier_train_step", prepare_data),
        "clf_attnpl_t": ("src.train.classifier_train_step", prepare_data),
        "clf_mixer_attnpl_t": ("src.train.classifier_train_step", prepare_data),
        "reg_attnpl": ("src.train.regressor_train_step", prepare_data),
        "reg_mixer_attnpl_t": ("src.train.regressor_train_step", prepare_data),
        "lstm": ("src.train.regressor_train_step", prepare_data_lstm),
        "thp": ("src.train.tpp_train_step", prepare_data_tpp),
        "thp_deltat": ("src.train.tpp_train_step", prepare_data_tpp),
        "rtpp": ("src.train.tpp_train_step", prepare_data_tpp),
        "nhpp": ("src.train.tpp_train_step", prepare_data_tpp),
        "etas": ("src.train.tpp_train_step", prepare_data_tpp),
        "mtpp": ("src.train.tpp_train_step", prepare_data_tpp),
        "mhp": ("src.train.tpp_train_step", prepare_data_tpp),
        "btpp": ("src.train.tpp_train_step", prepare_data_tpp),
        "mixer_tpp": ("src.train.tpp_m_train_step", prepare_data_tpp),
    }
    model_type = args.model
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {', '.join(model_map.keys())}.")
    train_step_module, data_func = model_map[model_type]
    train_step = importlib.import_module(train_step_module)
    df, train_loader, val_loader, test_loader, _ = data_func(args, base_path)
    return train_step, df, train_loader, val_loader, test_loader


def objective(trial,args):
    args.epochs = 120
    args.learning_rate =trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    # args.scheduler_factor = trial.suggest_uniform('scheduler_factor', 0.1, 0.9)
    # args.scheduler_patience = trial.suggest_int('scheduler_patience', 2, 6)
    # args.scheduler_threshold = trial.suggest_float('scheduler_threshold', 1e-5, 1e-3, log=True)
    # args.scheduler_min_lr = trial.suggest_float('scheduler_min_lr', 1e-6, 1e-4, log=True)
    args.warnup_ratio = trial.suggest_uniform('warnup_ratio', 0, 0.2)
    args.scheduler_type = trial.suggest_categorical('scheduler_type', ['hf_cosine', 'cosine','hf_linear', 'hf_constant'])

    shutil.copy(args_cli.config, f"{args.save_dir}/config.yaml")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard", f"trial_{trial.number}"))

    train_step, df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
    model, criterion, optimizer, scheduler, args = config_setup.setup_config(args, device,train_loader)

    print(f'Trial {trial.number}')
    
    val_loss, best_metrics = trainer.train_and_save(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        device=device,
        index=trial.number,
        writer=writer,
        )
    torch.cuda.empty_cache()
    return val_loss



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test','optuna'], default='train', help='Run mode: train or test')
    parser.add_argument('--model', type=str, choices=ModelBuilder.list_available(),
                         required=True, help='Model name')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,help='Directory to load checkpoint for test mode')
    parser.add_argument('--trial_index', type=int, default=1, help='Index of the trial for optuna')
    parser.add_argument('--ckpt_select', type=str, choices=['best', 'last', 'epoch'], default='best',
                    help='Which checkpoint to use in test mode (best, last, or epoch)')
    parser.add_argument('--ckpt_epoch', type=int, default=None, help='Epoch number to load when --ckpt_select epoch')
    parser.add_argument('--threshold', type=float, default=None, help='If provided, use this threshold for classification test (overrides checkpoint)')
    parser.add_argument('--no_val_threshold', action='store_true', default=False, help='Do not use threshold stored in checkpoint val_metrics')


    args_cli = parser.parse_args()
    if args_cli.config is None:
        args_cli.config = f"config/{args_cli.model}.yaml"

    args = config_loader.load_args_from_yaml(args_cli.config)
    assert args.model.lower() == args_cli.model.lower(), "Model name in config must match command line argument"
    seed = getattr(args, 'seed', 0)
    set_seed(seed)

    # Save directory: create or load based on mode
    if args_cli.mode in ["train", "optuna"]:
        args.save_dir = args_cli.checkpoint_dir or create_save_dir(base_dir="checkpoints", model_name=args.model)
    else:
        args.save_dir =args_cli.checkpoint_dir or find_latest_model_path(args_cli.model)

    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_id}" if args.cuda else "cpu")




    if args_cli.mode == "train":
        config_path =  f"{args.save_dir}/config.yaml"
        shutil.copy(args_cli.config, config_path)
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

        train_step,  df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)

        resume_path = getattr(args, 'resume_path', None)
        if resume_path is None:
            print("[INFO] No resume path provided. Training will start from scratch.")
            checkpoint = None
        elif not os.path.exists(resume_path):
            print(f"[WARNING] Resume path '{resume_path}' not found. Training will start from scratch.")
            checkpoint = None
        else:
            print(f"[INFO] Loading checkpoint from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        model, criterion, optimizer, scheduler, args = config_setup.setup_config(
            args, device,train_dataloader=train_loader,
            checkpoint=checkpoint, restore_weights=(checkpoint is not None)
)
        
        val_loss, metrics = trainer.train_and_save(
            args=args,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=args.save_dir,
            device=device,
            index=1,
            writer=writer,
        )
    elif args_cli.mode == "test":
        if args_cli.ckpt_select == 'epoch':
            if args_cli.ckpt_epoch is None:
                raise ValueError('When --ckpt_select is "epoch", --ckpt_epoch must be provided')
            checkpoint_path = f"{args.save_dir}/epoch_{args_cli.ckpt_epoch}_model_{args_cli.trial_index}.pth"
        else:
            checkpoint_path = f"{args.save_dir}/{args_cli.ckpt_select}_model_{args_cli.trial_index}.pth"  #  last/best
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = config_setup.load_args_from_checkpoint(args, checkpoint)
        args.minibatch_training = False
        args.load_specific_parts = None
        args.use_sampler = False
        args.model = args.model.lower()
        train_step, df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
        model, criterion, optimizer, scheduler, args = config_setup.setup_config(
            args, device,train_loader,
            checkpoint=checkpoint, restore_weights=True
        )
        if args.task_type != "tpp":
            # Determine threshold only for classification task
            if getattr(args, 'task_type', None) == 'classification':
                # Priority: CLI --threshold > checkpoint val_metrics (unless --no_val_threshold) > None
                cli_thresh = getattr(args_cli, 'threshold', None)
                if cli_thresh is not None:
                    threshold = cli_thresh
                    logging.info(f"Using threshold {threshold} from CLI argument for testing.")
                else:
                    threshold = None
                    if not getattr(args_cli, 'no_val_threshold', False):
                        if checkpoint is not None and isinstance(checkpoint, dict):
                            val_metrics = checkpoint.get('val_metrics', {})
                            if isinstance(val_metrics, dict) and 'threshold' in val_metrics:
                                threshold = val_metrics.get('threshold')
                                logging.info(f"Using threshold {threshold} from checkpoint val_metrics for testing.")

                test_loss, metrics = train_step.test(
                    model=model,
                    criterion=criterion,
                    data_loader=test_loader,  # Test data loader
                    device=device,
                    save_dir=args.save_dir,
                    threshold=threshold,
                )
            elif getattr(args, 'task_type', None) == 'regression':
                # For non-classification (e.g., regression), do not pass threshold
                test_loss, metrics = train_step.test(
                    model=model,
                    criterion=criterion,
                    data_loader=test_loader,  # Test data loader
                    device=device,
                    save_dir=args.save_dir,
                )
            print("Test loss:", test_loss)
            train_step.visualize_results(model,train_loader, val_loader, test_loader, device,args.save_dir)
        else:
            results, metrics = train_step.test(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader, 
                device=device,
                save_dir=args.save_dir,
            )
            metrics["num_events_train"] = args.num_events_train
            metrics["num_events_val"] = args.num_events_val
        
        # name metrics file to reflect selected checkpoint (include epoch if provided)
        if args_cli.ckpt_select == 'epoch':
            metrics_name = f"metrics_test_epoch_{args_cli.ckpt_epoch}_{args_cli.trial_index}.json"
        else:
            metrics_name = f"metrics_test_{args_cli.ckpt_select}_{args_cli.trial_index}.json"
        with open(os.path.join(args.save_dir, metrics_name), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved test metrics to {os.path.join(args.save_dir, metrics_name)}")

        print("Metrics:", metrics)
        torch.cuda.empty_cache()

    elif args_cli.mode == "optuna":
        study = optuna.create_study(direction="minimize")  
        study.optimize(lambda trial: objective(trial, args), n_trials=10)


        print("Best trial:")
        print(study.best_trial.params)