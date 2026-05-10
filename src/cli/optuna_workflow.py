"""Optuna workflow helpers for CLI-driven hyperparameter optimization.

This module contains utility functions and the end-to-end execution flow used
by the `--mode optuna` entrypoint, including search-space sampling, objective
construction, profile resolution, study creation, and summary export.
"""

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
DEFAULT_OBJECTIVE = "weighted_metrics"
SUPPORTED_OBJECTIVES = {"weighted_metrics", "val_loss"}
SUPPORTED_DIRECTIONS = {"minimize", "maximize"}


def cfg_to_dict(node):
    """Convert config-like objects into a plain Python dict.

    Args:
        node: A dict, OmegaConf node, or mapping-like object.

    Returns:
        A plain dict. Returns an empty dict when conversion is not possible.
    """
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


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clone_args(args):
    """Deep-clone an OmegaConf args object into a new mutable config."""
    return OmegaConf.create(OmegaConf.to_container(args, resolve=True))


def suggest_from_spec(trial, name, spec):
    """Sample one hyperparameter value from a search specification.

    Supports `float`, `int`, and `categorical` spec types using Optuna's
    suggestion APIs.

    Args:
        trial: Optuna trial object.
        name: Parameter name used in the trial.
        spec: Search-space specification dictionary.

    Returns:
        The sampled parameter value.
    """
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
    """Set a possibly dotted attribute path on an OmegaConf-like object."""
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
    """Apply all search-space specs to `args_trial` and return sampled values.

    Args:
        trial: Optuna trial object.
        args_trial: Trial-local args/config object to mutate.
        search_space: Mapping from dotted parameter name to spec dictionary.

    Returns:
        A dict of sampled parameter values keyed by parameter name.
    """
    sampled = {}
    for param_name, spec in search_space.items():
        value = suggest_from_spec(trial, param_name, spec)
        _set_nested_value(args_trial, param_name, value)
        sampled[param_name] = value
    return sampled


def metric_value(metrics, metric_name):
    """Read a metric value from a metric dict with case-insensitive fallback."""
    if metric_name in metrics:
        return metrics[metric_name]
    key_lower = str(metric_name).lower()
    for key, value in metrics.items():
        if str(key).lower() == key_lower:
            return value
    raise KeyError(f"Metric '{metric_name}' not found in metrics: {list(metrics.keys())}")


def build_objective_score(metrics, metric_weights):
    """Compute weighted objective score from validation metrics.

    Args:
        metrics: Metric dictionary from validation.
        metric_weights: Mapping of metric name to weight.

    Returns:
        Tuple of `(score, detail)` where `detail` contains used metric values.
    """
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
    """Resolve the effective Optuna profile name for current run.

    Resolution order:
    1) `--optuna_profile` from CLI
    2) `optuna.default_profile` from config
    3) `args.t_elaps_mode` when matched in profile names
    4) the only profile when exactly one profile exists
    """
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
    """Resolve Optuna storage config to a URI.

    Returns:
        - `None` when storage is not configured.
        - Original URI when `storage_cfg` already looks like a URI.
        - `sqlite:///...` URI for local file paths (absolute or save_dir-relative).
    """
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
    """Create an Optuna sampler from config."""
    sampler_name = str(optuna_cfg.get("sampler", "tpe")).strip().lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=sampler_seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=sampler_seed)
    raise ValueError(f"Unsupported optuna sampler: {sampler_name}. Use 'tpe' or 'random'.")


def export_optuna_study_summary(study, output_path, *, profile_name=None, top_k=5):
    """Export a compact JSON summary of an Optuna study.

    The summary contains best trial metadata, top-K completed trials, and basic
    study statistics.
    """
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed_trials.sort(key=lambda item: float(item.value), reverse=not is_minimize)
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
    _save_json(output_path, payload)
    return payload


def _eval_val_metrics_for_optuna(train_step, args_trial, model, criterion, val_loader, device, trial_dir):
    task_type = str(getattr(args_trial, "task_type", "")).strip().lower()
    if task_type == "tpp":
        test_nll_kwargs = {
            "reduction": getattr(args_trial, "loss_reduction", None),
        }
        _results, metrics = train_step.test(
            model=model,
            criterion=criterion,
            train_loader=None,
            val_loader=val_loader,
            test_loader=None,
            device=device,
            save_dir=trial_dir,
            nll_kwargs=test_nll_kwargs,
        )
        return metrics

    test_kwargs = {
        "model": model,
        "criterion": criterion,
        "data_loader": val_loader,
        "device": device,
        "save_dir": trial_dir,
    }
    if task_type == "classification":
        test_kwargs["threshold"] = None
    _val_eval_loss, metrics = train_step.test(**test_kwargs)
    return metrics


def _resolve_profile_cfg(optuna_cfg, profile_name):
    if profile_name is None:
        return {}
    return cfg_to_dict(optuna_cfg.get("profiles", {}).get(profile_name, {}))


def _resolve_metric_weights(optuna_cfg):
    metric_weights = cfg_to_dict(optuna_cfg.get("metric_weights", {}))
    if not metric_weights:
        raise ValueError(
            "Optuna metric_weights is empty. "
            "Please define optuna.metric_weights, e.g. {auc: -1.0} or {RMSE: 1.0}."
        )
    return metric_weights


def _resolve_search_space(optuna_cfg, profile_cfg):
    search_space = cfg_to_dict(profile_cfg.get("params", {}))
    if search_space:
        return search_space
    search_space = cfg_to_dict(optuna_cfg.get("search_space", {}))
    if search_space:
        return search_space
    raise ValueError("Optuna search space is empty. Please define optuna.search_space or optuna.profiles.<name>.params.")


def _resolve_objective_and_direction(optuna_cfg):
    objective_name = str(optuna_cfg.get("objective", DEFAULT_OBJECTIVE)).strip().lower()
    if objective_name not in SUPPORTED_OBJECTIVES:
        supported_text = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(
            f"Unsupported optuna objective '{objective_name}'. "
            f"Currently supported: {supported_text}."
        )
    direction = str(optuna_cfg.get("direction", "minimize")).strip().lower()
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"Unsupported optuna direction '{direction}'. Use 'minimize' or 'maximize'.")
    return objective_name, direction


def _resolve_optuna_runtime(args_cli, args, optuna_cfg):
    objective_name, direction = _resolve_objective_and_direction(optuna_cfg)
    profile_names = _resolve_profile_names(args, args_cli, optuna_cfg)
    sampler_seed = args_cli.optuna_sampler_seed
    if sampler_seed is None:
        sampler_seed = optuna_cfg.get("sampler_seed", getattr(args, "seed", 0))

    n_trials_cfg = optuna_cfg.get("n_trials", 10)
    n_trials = int(args_cli.optuna_trials if args_cli.optuna_trials is not None else n_trials_cfg)
    top_k = int(optuna_cfg.get("report_top_k", 5))
    base_storage_cfg = args_cli.optuna_storage or optuna_cfg.get("storage", None)
    return {
        "objective_name": objective_name,
        "direction": direction,
        "profile_names": profile_names,
        "sampler_seed": int(sampler_seed),
        "n_trials": n_trials,
        "top_k": top_k,
        "base_storage_cfg": base_storage_cfg,
    }


def _build_study_name(args_cli, optuna_cfg, model_name, profile_name, objective_name, multi_profile):
    profile_suffix = profile_name or "default"
    configured_name = args_cli.optuna_study_name or optuna_cfg.get("study_name")
    if configured_name:
        return f"{configured_name}_{profile_suffix}" if multi_profile else configured_name
    return f"{model_name}_{profile_suffix}_{objective_name}"


def objective_weighted_metrics(trial, args_base, args_cli, device, optuna_cfg, profile_name=None):
    """Generic Optuna objective using weighted validation metrics.

    Trains one trial with sampled hyperparameters, restores the best
    checkpoint, evaluates validation metrics, and returns weighted score.
    """
    args_trial = clone_args(args_base)
    profile_cfg = _resolve_profile_cfg(optuna_cfg, profile_name)

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

    objective_name, _direction = _resolve_objective_and_direction(optuna_cfg)
    metric_weights = _resolve_metric_weights(optuna_cfg) if objective_name == "weighted_metrics" else None
    search_space = _resolve_search_space(optuna_cfg, profile_cfg)

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

        best_val_loss, train_summary = trainer.train_and_save(
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

        if objective_name == "val_loss":
            score = float(best_val_loss)
            score_detail = {"val_loss": score}
            val_metrics = cfg_to_dict(train_summary.get("val_metrics", {})) if isinstance(train_summary, dict) else {}
            trial.set_user_attr("val_loss", score)
        else:
            best_ckpt_path = os.path.join(trial_dir, "best_model_1.pth")
            if not os.path.exists(best_ckpt_path):
                raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])

            val_metrics = _eval_val_metrics_for_optuna(
                train_step=train_step,
                args_trial=args_trial,
                model=model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                trial_dir=trial_dir,
            )
            score, score_detail = build_objective_score(val_metrics, metric_weights)
            trial.set_user_attr("val_metrics", val_metrics)

        trial.set_user_attr("score_detail", score_detail)
        trial.set_user_attr("profile", profile_name)
        trial.set_user_attr("trial_dir", trial_dir)

        _save_json(
            os.path.join(trial_dir, "metrics_optuna_val.json"),
            {
                "profile": profile_name,
                "sampled_params": sampled_params,
                "objective": objective_name,
                "metric_weights": metric_weights,
                "objective_score": score,
                "best_val_loss": float(best_val_loss),
                "val_metrics": val_metrics,
            },
        )

        LOGGER.info("Optuna trial %s done | profile=%s | score=%.6f | detail=%s", trial.number, profile_name, score, score_detail)
        return score
    finally:
        writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_profile_names(args, args_cli, optuna_cfg):
    """Resolve profile execution list for a single- or all-profile run."""
    profiles = cfg_to_dict(optuna_cfg.get("profiles", {}))
    if args_cli.optuna_all_profiles:
        if not profiles:
            return [None]
        return list(profiles.keys())
    return [resolve_optuna_profile_name(args, args_cli, optuna_cfg)]


def run_optuna(args_cli, args, device):
    """Run Optuna studies according to runtime config and CLI overrides.

    Supports single profile (`--optuna_profile`) or all configured profiles
    (`--optuna_all_profiles`), writes per-profile summaries, and also exports an
    aggregated summary file under `args.save_dir`.
    """
    optuna_cfg = cfg_to_dict(getattr(args, "optuna", {}))
    runtime = _resolve_optuna_runtime(args_cli, args, optuna_cfg)
    objective_name = runtime["objective_name"]
    direction = runtime["direction"]
    profile_names = runtime["profile_names"]
    sampler_seed = runtime["sampler_seed"]
    n_trials = runtime["n_trials"]
    top_k = runtime["top_k"]
    base_storage_cfg = runtime["base_storage_cfg"]
    multi_profile = len(profile_names) > 1

    study_summaries = []
    for profile_name in profile_names:
        storage = resolve_optuna_storage(base_storage_cfg, args.save_dir)
        profile_suffix = profile_name or "default"
        study_name = _build_study_name(
            args_cli=args_cli,
            optuna_cfg=optuna_cfg,
            model_name=args.model,
            profile_name=profile_name,
            objective_name=objective_name,
            multi_profile=multi_profile,
        )

        sampler = create_optuna_sampler(optuna_cfg, sampler_seed)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage),
            direction=direction,
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
            lambda trial, _profile=profile_name: objective_weighted_metrics(
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

    _save_json(os.path.join(args.save_dir, "optuna_summary_all.json"), study_summaries)
