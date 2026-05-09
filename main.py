import logging

from src.cli.common import parse_and_prepare_runtime
from src.cli.data_factory import get_model_and_data
from src.cli.optuna_workflow import apply_trial_search_space, resolve_optuna_profile_name, run_optuna
from src.cli.workflows import run_test, run_train

LOGGER = logging.getLogger(__name__)

# Backward compatibility exports (used by existing scripts/tests).
_apply_trial_search_space = apply_trial_search_space
_resolve_optuna_profile_name = resolve_optuna_profile_name


def main(argv=None):
    runtime = parse_and_prepare_runtime(argv)
    args_cli = runtime.args_cli
    args = runtime.args
    device = runtime.device

    if args_cli.mode == "train":
        run_train(args_cli, args, device)
    elif args_cli.mode == "test":
        run_test(args_cli, args, device)
    elif args_cli.mode == "optuna":
        run_optuna(args_cli, args, device)
    else:
        raise ValueError(f"Unsupported mode: {args_cli.mode}")


if __name__ == "__main__":
    main()

