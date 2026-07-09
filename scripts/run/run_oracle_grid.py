#!/usr/bin/env python3
"""Oracle grid runner.

This entrypoint reuses the shared rtpp_v2 grid runner implementation while
providing Oracle-specific defaults.
"""

from pathlib import Path
import runpy
import sys


DEFAULT_EXP_CONFIG = "config/experiments/oracle_multi_dataset.yaml"
DEFAULT_MODEL = "oracle"
DEFAULT_BASE_CONFIG = "config/oracle.yaml"


def _has_option(argv: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(item == option or item.startswith(prefix) for item in argv)


def _inject_default_args(argv: list[str]) -> list[str]:
    if any(item in {"-h", "--help"} for item in argv):
        return argv

    out = list(argv)
    if not _has_option(out, "--model"):
        out.extend(["--model", DEFAULT_MODEL])
    if not _has_option(out, "--base_config"):
        out.extend(["--base_config", DEFAULT_BASE_CONFIG])
    if not _has_option(out, "--exp_config"):
        out.extend(["--exp_config", DEFAULT_EXP_CONFIG])
    return out


def main() -> None:
    current_file = Path(__file__).resolve()
    scripts_root = current_file.parents[1]
    repo_root = scripts_root.parent
    target = scripts_root / "run/run_rtpp_v2_grid.py"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    sys.argv = [str(target), *_inject_default_args(sys.argv[1:])]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
