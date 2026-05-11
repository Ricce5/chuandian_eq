#!/usr/bin/env python3
from optuna_profiles_common import build_profiles_runner_parser, run_optuna_profiles


def main():
    parser = build_profiles_runner_parser(
        description=(
            "One-click run reg_mixer_attnpl_t Optuna across configured profiles "
            "and aggregate best results."
        ),
        default_model="reg_mixer_attnpl_t",
        default_config="config/reg_mixer_attnpl_t.yaml",
        default_out_dir="tmp/reg_mixer_attnpl_t_optuna_profiles",
    )
    args = parser.parse_args()
    run_optuna_profiles(args)


if __name__ == "__main__":
    main()
