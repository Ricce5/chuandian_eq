import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "run_sliding_window_forecast_for_seeds.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_sliding_window_forecast_for_seeds",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SEED_WRAPPER = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SEED_WRAPPER
SCRIPT_SPEC.loader.exec_module(SEED_WRAPPER)


def _args() -> Namespace:
    return Namespace(
        python="python",
        checkpoint_file="best_model_1.pth",
        cache_filename_template=None,
        metrics_filename="metrics.json",
        metrics_only=False,
    )


def test_build_command_forwards_seed_from_run_name():
    command = SEED_WRAPPER.build_command(
        args=_args(),
        script_path=Path("scripts/run_sliding_window_forecast.py"),
        run_dir=Path("/tmp/demo_seed_2"),
        forwarded_args=["--eval-range", "test"],
        device=None,
    )

    assert command[-4:] == ["--seed", "2", "--eval-range", "test"]


def test_build_command_preserves_explicit_seed_override():
    command = SEED_WRAPPER.build_command(
        args=_args(),
        script_path=Path("scripts/run_sliding_window_forecast.py"),
        run_dir=Path("/tmp/demo_seed_2"),
        forwarded_args=["--seed", "17"],
        device=None,
    )

    assert command.count("--seed") == 1
    assert command[-2:] == ["--seed", "17"]
