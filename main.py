# %%
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import importlib
import numpy as np
from config import config_loader
from src.utils.utils import set_seed
from src.utils.file_utils import create_save_dir,find_latest_model_path
from src.data.preparation import prepare_data
import src.train.config_setup as config_setup 
import src.train.trainer as trainer
import src.models.Models
import shutil
import os
import json
import yaml
import optuna



def get_model_and_data(args, base_path, device):
    model_type = args.model
    supported_models = {
        "Classifier": {
            "train_step_module": "src.train.classifier_train_step",
            "model_class": "Classifier",
            "data_func": "prepare_data",
        },
        "Classifier_STM": {
            "train_step_module": "src.train.classifier_train_step",
            "model_class": "Classifier_STM",
            "data_func": "prepare_data",
        },
         "Classifier_SE": {
            "train_step_module": "src.train.classifier_train_step",
            "model_class": "Classifier_SE",
            "data_func": "prepare_data",
        },
        "ClfAttnPl": {
            "train_step_module": "src.train.classifier_train_step",
            "model_class": "ClfAttnPl",
            "data_func": "prepare_data",
        },
        "ClfAttnPl_T": {
            "train_step_module": "src.train.classifier_train_step",
            "model_class": "ClfAttnPl_T",
            "data_func": "prepare_data",
        },


    }
    if model_type not in supported_models:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {', '.join(supported_models.keys())}.")
    model_info = supported_models[model_type]
    train_step = __import__(model_info["train_step_module"], fromlist=[''])
    model_class = getattr(src.models.Models, model_info["model_class"])
    data_func = globals()[model_info["data_func"]]
    df, train_loader, val_loader, test_loader,dataset,scalars = data_func(args, base_path)
    return train_step, model_class, df, train_loader, val_loader, test_loader


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

    train_step, model_class, df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
    model, criterion, optimizer, scheduler, args = config_setup.setup_config(args, device, model_class,train_loader)

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
    # 返回验证集的损失作为优化目标
    return val_loss



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test','optuna'], default='train', help='Run mode: train or test')
    parser.add_argument('--model', type=str, choices=['Classifier','Classifier_STM','Classifier_SE','ClfAttnPl','ClfAttnPl_T'], required=True, help='Model name')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,help='Directory to load checkpoint for test mode')
    parser.add_argument('--trial_index', type=int, default=1, help='Index of the trial for optuna')


    
    args_cli = parser.parse_args()
    if args_cli.config is None:
        args_cli.config = f"config/{args_cli.model}.yaml"

   
    set_seed(0)

    args = config_loader.load_args_from_yaml(args_cli.config)

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

        train_step, model_class, df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
        model, criterion, optimizer, scheduler, args = config_setup.setup_config(args, device, model_class,train_loader)
        
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
        checkpoint_path = f"{args.save_dir}/best_model_{args_cli.trial_index}.pth"  #  last/best
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = config_setup.load_args_from_checkpoint(args, checkpoint)
        args.use_sampler = False
        train_step, model_class, df, train_loader, val_loader, test_loader = get_model_and_data(args, f"data/{args.dataset}", device)
        model, criterion, optimizer, scheduler, args = config_setup.setup_config(
            args, device, model_class,train_loader,
            checkpoint=checkpoint, restore_weights=True
        )

   
        test_loss, metrics = train_step.test(
            model=model,
            criterion=criterion,
            data_loader=test_loader,  # Test data loader
            device=device,
            save_dir=args.save_dir,
        )

        train_step.visualize_results(model,train_loader, val_loader, test_loader, device,args.save_dir)

        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("Test loss:", test_loss)
        print("Metrics:", metrics)
        torch.cuda.empty_cache()

    elif args_cli.mode == "optuna":
        study = optuna.create_study(direction="minimize")  
        study.optimize(lambda trial: objective(trial, args), n_trials=10)


        print("Best trial:")
        print(study.best_trial.params)