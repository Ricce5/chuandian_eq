#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from automation import build_profiles_runner_parser, run_optuna_profiles


def main():
    parser = build_profiles_runner_parser(
        description="One-click run LSTM Optuna for global/window profiles and aggregate best results.",
        default_model="lstm",
        default_config="config/lstm.yaml",
        default_out_dir="tmp/lstm_optuna_profiles",
    )
    args = parser.parse_args()
    run_optuna_profiles(args)


if __name__ == "__main__":
    main()
