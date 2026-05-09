import json
import logging
import math
import os

import optuna
import torch
from omegaconf import OmegaConf
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter

import src.train.config_setup as config_setup
import src.train.trainer as trainer
from src.utils.utils import set_seed
from .data_factory import get_model_and_data

LOGGER = logging.getLogger(__name__)


def cfg_to_dict(node):
    if node is None:
        return {}
    if isinstance(node, dict):
        return node
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    try:
        return dict(node)
    except Exception:
        return {}


def clone_args(args):
    return OmegaConf.create(OmegaConf.to_container(args, resolve=True))


def suggest_from_spec(trial, name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Search spec for '{name}' must be a dict, got: {type(spec)}")

    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step", None)
        step = None if step is None else float(step)
        log_scale = bool(spec.get("log", False))
        if step is not None and log_scale:
            raise ValueError(f"Spec '{name}' cannot use both step and log=True.")
        return trial.suggest_float(name, low, high, step=step, log=log_scale)

    if spec_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log_scale = bool(spec.get("log", False))
        return trial.suggest_int(name, low, high, step=step, log=log_scale)

    if spec_type == "categorical":
        choices = list(spec.get("choices", []))
        if len(choices) == 0:
            raise ValueError(f"Categorical spec '{name}' must provide non-empty choices.")
        return trial.suggest_categorical(name, choices)

    raise ValueError(
        f"Unsupported search spec type for '{name}': {spec_type}. "
        "Use one of: float, int, categorical."
    )


def _set_nested_value(root, dotted_name, value):
    if "." not in dotted_name:
        setattr(root, dotted_name, value)
        return

    parts = dotted_name.split(".")
    node = root
    for part in parts[:-1]:
        child = getattr(node, part, None)
        if child is None:
            child = OmegaConf.create({})
            setattr(node, part, child)
        node = child
    setattr(node, parts[-1], value)


def apply_trial_search_space(trial, args_trial, search_space):
    sampled = {}
    for param_name, spec in search_space.items():
        value = suggest_from_spec(trial, param_name, spec)
        _set_nested_value(args_trial, param_name, value)
        sampled[param_name] = value
    return sampled


def metric_value(metrics, metric_name):
    if metric_name in metrics:
        return metrics[metric_name]
    key_lower = str(metric_name).lower()
    for key, value in metrics.items():
        if str(key).lower() == key_lower:
            return value
    raise KeyError(f"Metric '{metric_name}' not found in metrics: {list(metrics.keys())}")


def build_objective_score(metrics, metric_weights):
    score = 0.0
    detail = {}
    for metric_name, weight in metric_weights.items():
        value = float(metric_value(metrics, metric_name))
        weight = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"Metric '{metric_name}' is not finite: {value}")
        score += weight * value
        detail[metric_name] = value
    return float(score), detail


def resolve_optuna_profile_name(args, args_cli, optuna_cfg):
    profiles = cfg_to_dict(optuna_cfg.get("profiles", {}))
    if not profiles:
        return None

    requested = getattr(args_cli, "optuna_profile", None)
    if requested:
        if requested not in profiles:
            raise ValueError(f"Unknown optuna profile '{requested}'. Available: {list(profiles.keys())}")
        return requested

    default_profile = optuna_cfg.get("default_profile", None)
    if default_profile and default_profile in profiles:
        return default_profile

    t_elaps_mode = str(getattr(args, "t_elaps_mode", "")).strip().lower()
    if t_elaps_mode in profiles:
        return t_elaps_mode

    if len(profiles) == 1:
        return next(iter(profiles.keys()))

    raise ValueError(
        "Multiple optuna profiles are configured but none selected. "
        "Please pass --optuna_profile (e.g. global/window) or set optuna.default_profile in config."
    )


def resolve_optuna_storage(storage_cfg, save_dir):
    if not storage_cfg:
        return None
    if "://" in str(storage_cfg):
        return str(storage_cfg)
    storage_path = str(storage_cfg)
    if not os.path.isabs(storage_path):
        storage_path = os.path.join(save_dir, storage_path)
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    return f"sqlite:///{storage_path}"


def create_optuna_sampler(optuna_cfg, sampler_seed):
    sampler_name = str(optuna_cfg.get("sampler", "tpe")).strip().lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=sampler_seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=sampler_seed)
    raise ValueError(f"Unsupported optuna sampler: {sampler_name}. Use 'tpe' or 'random'.")


def export_optuna_study_summary(study, output_path, *, profile_name=None, top_k=5):
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    completed_trials.sort(key=lambda item: float(item.value))
    top_trials = completed_trials[: max(1, int(top_k))]
    payload = {
        "study_name": study.study_name,
        "profile": profile_name,
        "direction": str(study.direction),
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed_trials),
        "best_trial": {
            "number": study.best_trial.number,
            "value": float(study.best_trial.value),
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
        "top_trials": [
            {
                "number": trial.number,
                "value": float(trial.value),
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in top_trials
        ],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def objective_lstm_dtw_rmse(trial, args_base, args_cli, device, optuna_cfg, profile_name=None):
    args_trial = clone_args(args_base)
    if profile_name is None:
        profile_cfg = {}
    else:
        profile_cfg = cfg_to_dict(optuna_cfg.get("profiles", {}).get(profile_name, {}))

    if "t_elaps_mode" in profile_cfg:
        args_trial.t_elaps_mode = profile_cfg["t_elaps_mode"]

    n_epochs = int(optuna_cfg.get("epochs", getattr(args_trial, "epochs", 120)))
    args_trial.epochs = n_epochs

    trial_seed_base = int(optuna_cfg.get("trial_seed_base", getattr(args_trial, "seed", 0)))
    args_trial.seed = int(trial_seed_base + trial.number)
    set_seed(
        args_trial.seed,
        deterministic=bool(getattr(args_trial, "deterministic", True)),
        deterministic_warn_only=bool(getattr(args_trial, "deterministic_warn_only", False)),
        use_deterministic_algorithms=bool(getattr(args_trial, "use_deterministic_algorithms", False)),
    )

    metric_weights = cfg_to_dict(optuna_cfg.get("metric_weights", {"RMSE": 1.0, "DTW": 1.0}))
    search_space = cfg_to_dict(profile_cfg.get("params", {}))
    if not search_space:
        search_space = cfg_to_dict(optuna_cfg.get("search_space", {}))
    if not search_space:
        raise ValueError("Optuna search space is empty. Please define optuna.search_space or optuna.profiles.<name>.params.")

    sampled_params = apply_trial_search_space(trial, args_trial, search_space)

    profile_suffix = f"_{profile_name}" if profile_name else ""
    trial_dir = os.path.join(args_base.save_dir, "optuna_trials", f"trial_{trial.number:04d}{profile_suffix}")
    os.makedirs(trial_dir, exist_ok=True)
    args_trial.save_dir = trial_dir
    trial_cfg = clone_args(args_trial)
    trial_cfg.optuna_trial_meta = {
        "trial_number": int(trial.number),
        "profile": profile_name,
        "sampled_params": sampled_params,
        "objective_metric_weights": metric_weights,
    }
    OmegaConf.save(trial_cfg, os.path.join(trial_dir, "config_base.yaml"))

    writer = SummaryWriter(log_dir=os.path.join(args_base.save_dir, "tensorboard", f"trial_{trial.number}{profile_suffix}"))
    try:
        train_step, _df, train_loader, val_loader, _test_loader = get_model_and_data(
            args_trial,
            f"data/{args_trial.dataset}",
            device,
        )

        model, criterion, optimizer, scheduler, args_trial = config_setup.setup_config(
            args_trial,
            device,
            train_dataloader=train_loader,
            checkpoint=None,
            restore_weights=False,
        )

        LOGGER.info(
            "Optuna trial %s | profile=%s | sampled=%s",
            trial.number,
            profile_name,
            sampled_params,
        )

        trainer.train_and_save(
            args=args_trial,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=trial_dir,
            device=device,
            index=1,
            writer=writer,
        )

        best_ckpt_path = os.path.join(trial_dir, "best_model_1.pth")
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        _val_eval_loss, val_metrics = train_step.test(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            save_dir=trial_dir,
        )

        score, score_detail = build_objective_score(val_metrics, metric_weights)
        trial.set_user_attr("val_metrics", val_metrics)
        trial.set_user_attr("score_detail", score_detail)
        trial.set_user_attr("profile", profile_name)
        trial.set_user_attr("trial_dir", trial_dir)

        with open(os.path.join(trial_dir, "metrics_optuna_val.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "profile": profile_name,
                    "sampled_params": sampled_params,
                    "metric_weights": metric_weights,
                    "objective_score": score,
                    "val_metrics": val_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        LOGGER.info(
            "Optuna trial %s done | profile=%s | score=%.6f | RMSE=%.6f | DTW=%.6f | DTW_normalized=%.6f",
            trial.number,
            profile_name,
            score,
            float(metric_value(val_metrics, "RMSE")),
            float(metric_value(val_metrics, "DTW")),
            float(metric_value(val_metrics, "DTW_normalized")),
        )
        return score
    finally:
        writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_profile_names(args, args_cli, optuna_cfg):
    profiles = cfg_to_dict(optuna_cfg.get("profiles", {}))
    if args_cli.optuna_all_profiles:
        if not profiles:
            return [None]
        return list(profiles.keys())
    return [resolve_optuna_profile_name(args, args_cli, optuna_cfg)]


def run_optuna(args_cli, args, device):
    optuna_cfg = cfg_to_dict(getattr(args, "optuna", {}))
    objective_name = str(optuna_cfg.get("objective", "lstm_dtw_rmse")).strip().lower()
    if objective_name != "lstm_dtw_rmse":
        raise ValueError(
            f"Unsupported optuna objective '{objective_name}'. "
            "Currently supported: 'lstm_dtw_rmse'."
        )
    if args.model != "lstm":
        raise ValueError("optuna objective 'lstm_dtw_rmse' currently supports model='lstm' only.")

    profile_names = _resolve_profile_names(args, args_cli, optuna_cfg)

    sampler_seed = args_cli.optuna_sampler_seed
    if sampler_seed is None:
        sampler_seed = optuna_cfg.get("sampler_seed", getattr(args, "seed", 0))
    sampler_seed = int(sampler_seed)
    n_trials_cfg = optuna_cfg.get("n_trials", 10)
    n_trials = int(args_cli.optuna_trials if args_cli.optuna_trials is not None else n_trials_cfg)
    top_k = int(optuna_cfg.get("report_top_k", 5))

    base_storage_cfg = args_cli.optuna_storage or optuna_cfg.get("storage", None)
    study_summaries = []
    for profile_name in profile_names:
        storage = resolve_optuna_storage(base_storage_cfg, args.save_dir)
        profile_suffix = profile_name if profile_name else "default"
        default_study_name = f"{args.model}_{profile_suffix}_dtw_rmse"
        study_name = args_cli.optuna_study_name or optuna_cfg.get("study_name", default_study_name)
        if len(profile_names) > 1 and study_name == (args_cli.optuna_study_name or optuna_cfg.get("study_name", None)):
            study_name = f"{study_name}_{profile_suffix}"

        sampler = create_optuna_sampler(optuna_cfg, sampler_seed)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage),
            direction="minimize",
            sampler=sampler,
        )
        LOGGER.info(
            "Starting Optuna | profile=%s | n_trials=%s | study=%s | storage=%s",
            profile_name,
            n_trials,
            study.study_name,
            storage,
        )
        study.optimize(
            lambda trial, _profile=profile_name: objective_lstm_dtw_rmse(
                trial,
                args,
                args_cli,
                device,
                optuna_cfg,
                profile_name=_profile,
            ),
            n_trials=n_trials,
        )

        summary_path = os.path.join(args.save_dir, f"optuna_summary_{profile_suffix}.json")
        summary = export_optuna_study_summary(
            study,
            summary_path,
            profile_name=profile_name,
            top_k=top_k,
        )
        LOGGER.info("Optuna profile=%s best value=%.6f params=%s", profile_name, summary["best_trial"]["value"], summary["best_trial"]["params"])
        study_summaries.append(summary)

    with open(os.path.join(args.save_dir, "optuna_summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(study_summaries, f, ensure_ascii=False, indent=2)

