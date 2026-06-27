#!/usr/bin/env python3
"""Run sliding-window forecasts for selected seed run directories."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_RUNS_DIR = "experiments/etas_multi_ds_bg_norm_0.2/runs"
SEED_PATTERN = re.compile(r"(?:^|_)seed_(\d+)(?:$|_)")


def resolve_project_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing 'src' directory.")


PROJECT_ROOT = resolve_project_root()


def resolve_path(path_value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find run directories under --runs-dir whose names contain selected "
            "seed ids, then call scripts/run_sliding_window_forecast.py for each."
        )
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Parent directory containing per-run checkpoint folders.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed ids to run, e.g. --seeds 0 1 2.",
    )
    parser.add_argument(
        "--run-name-regex",
        default=None,
        help="Optional regex filter applied to run directory names.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="best_model_1.pth",
        help="Checkpoint filename inside each run directory.",
    )
    parser.add_argument(
        "--script",
        default="scripts/run_sliding_window_forecast.py",
        help="Sliding-window script to execute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the sliding-window script.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of run directories to process concurrently. Default: 1. "
            "Use cautiously on GPU workloads."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Optional comma-separated device list assigned round-robin to jobs, "
            "e.g. cuda:0,cuda:1 or cpu. Adds --device to child commands unless "
            "already forwarded."
        ),
    )
    parser.add_argument(
        "--cache-filename-template",
        default=None,
        help=(
            "Optional cache filename template passed to child commands. "
            "Available fields: {run}, {seed}. By default the child script uses "
            "its own notebook-compatible cache filename."
        ),
    )
    parser.add_argument(
        "--metrics-filename",
        default="sliding_window_eval_metrics.json",
        help=(
            "Metrics JSON filename used by --skip-existing-ok. Also forwarded "
            "to child commands unless already provided after '--'."
        ),
    )
    parser.add_argument(
        "--skip-existing-ok",
        action="store_true",
        help="Skip run directories whose metrics JSON already has status='ok'.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Forward --no-save-plots to child commands for faster metrics-only runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining seeds if one command fails.",
    )
    parser.add_argument(
        "sliding_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args forwarded to run_sliding_window_forecast.py. "
            "Use '--' before forwarded args."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class RunJob:
    index: int
    total: int
    run_dir: Path
    command: list[str]


def normalize_forwarded_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_devices(devices_arg: str | None) -> list[str]:
    if devices_arg is None:
        return []
    return [item.strip() for item in devices_arg.split(",") if item.strip()]


def has_forwarded_option(forwarded_args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(item == option or item.startswith(prefix) for item in forwarded_args)


def get_forwarded_option_value(
    forwarded_args: list[str],
    option: str,
    default: str,
) -> str:
    prefix = f"{option}="
    for index, item in enumerate(forwarded_args):
        if item.startswith(prefix):
            return item[len(prefix):]
        if item == option and index + 1 < len(forwarded_args):
            return forwarded_args[index + 1]
    return default


def extract_seed(run_name: str) -> int | None:
    match = SEED_PATTERN.search(run_name)
    if match is None:
        return None
    return int(match.group(1))


def discover_run_dirs(
    runs_dir: Path,
    *,
    seeds: set[int],
    run_name_regex: str | None,
    checkpoint_file: str,
) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Runs path is not a directory: {runs_dir}")

    name_filter = re.compile(run_name_regex) if run_name_regex else None
    run_dirs: list[Path] = []
    for candidate in sorted(runs_dir.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        seed = extract_seed(candidate.name)
        if seed is None or seed not in seeds:
            continue
        if name_filter is not None and name_filter.search(candidate.name) is None:
            continue
        if not (candidate / checkpoint_file).exists():
            print(
                f"Skip missing checkpoint: {candidate / checkpoint_file}",
                file=sys.stderr,
            )
            continue
        run_dirs.append(candidate)
    return run_dirs


def quote_command(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def format_cache_filename(template: str, run_dir: Path) -> str:
    seed = extract_seed(run_dir.name)
    return template.format(run=run_dir.name, seed="" if seed is None else seed)


def metrics_status_is_ok(run_dir: Path, metrics_filename: str) -> bool:
    metrics_path = Path(metrics_filename)
    if not metrics_path.is_absolute():
        metrics_path = run_dir / metrics_path
    if not metrics_path.exists():
        return False

    try:
        with metrics_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception:
        return False

    return isinstance(payload, dict) and payload.get("status") == "ok"


def build_command(
    *,
    args: argparse.Namespace,
    script_path: Path,
    run_dir: Path,
    forwarded_args: list[str],
    device: str | None,
) -> list[str]:
    command = [
        args.python,
        str(script_path),
        "--checkpoint-dir",
        str(run_dir),
        "--checkpoint-file",
        args.checkpoint_file,
        "--output-dir",
        str(run_dir),
    ]

    if args.cache_filename_template:
        command.extend(
            [
                "--cache-filename",
                format_cache_filename(args.cache_filename_template, run_dir),
            ]
        )

    if not has_forwarded_option(forwarded_args, "--metrics-filename"):
        command.extend(["--metrics-filename", args.metrics_filename])

    if (
        args.metrics_only
        and not has_forwarded_option(forwarded_args, "--save-plots")
        and not has_forwarded_option(forwarded_args, "--no-save-plots")
    ):
        command.append("--no-save-plots")

    if device is not None and not has_forwarded_option(forwarded_args, "--device"):
        command.extend(["--device", device])

    command.extend(forwarded_args)
    return command


def run_job(job: RunJob) -> int:
    print(f"[{job.index}/{job.total}] START {job.run_dir.name}", flush=True)
    print(quote_command(job.command), flush=True)
    completed = subprocess.run(job.command, cwd=PROJECT_ROOT, check=False)
    status = "OK" if completed.returncode == 0 else f"FAIL exit={completed.returncode}"
    print(f"[{job.index}/{job.total}] {status} {job.run_dir.name}", flush=True)
    return completed.returncode


def main() -> int:
    args = parse_args()
    runs_dir = resolve_path(args.runs_dir)
    script_path = resolve_path(args.script)
    forwarded_args = normalize_forwarded_args(args.sliding_args)
    selected_seeds = set(args.seeds)
    devices = parse_devices(args.devices)
    jobs = max(1, int(args.jobs))
    metrics_filename = get_forwarded_option_value(
        forwarded_args,
        "--metrics-filename",
        args.metrics_filename,
    )

    if not script_path.exists():
        raise FileNotFoundError(f"Sliding-window script not found: {script_path}")

    run_dirs = discover_run_dirs(
        runs_dir,
        seeds=selected_seeds,
        run_name_regex=args.run_name_regex,
        checkpoint_file=args.checkpoint_file,
    )
    if not run_dirs:
        print(
            f"No matching runs found in {runs_dir} for seeds={sorted(selected_seeds)}.",
            file=sys.stderr,
        )
        return 1

    skipped_existing = 0
    if args.skip_existing_ok:
        filtered_run_dirs = []
        for run_dir in run_dirs:
            if metrics_status_is_ok(run_dir, metrics_filename):
                skipped_existing += 1
                print(f"Skip existing ok metrics: {run_dir.name}")
            else:
                filtered_run_dirs.append(run_dir)
        run_dirs = filtered_run_dirs

    print(f"Found {len(run_dirs)} matching run(s).")
    if skipped_existing:
        print(f"Skipped {skipped_existing} already-completed run(s).")
    if not run_dirs:
        return 0

    if jobs > 1 and not args.dry_run and not args.continue_on_error:
        print(
            "Parallel mode note: already-started jobs will finish before exit "
            "when a failure occurs.",
            file=sys.stderr,
        )

    run_jobs = [
        RunJob(
            index=index,
            total=len(run_dirs),
            run_dir=run_dir,
            command=build_command(
                args=args,
                script_path=script_path,
                run_dir=run_dir,
                forwarded_args=forwarded_args,
                device=devices[(index - 1) % len(devices)] if devices else None,
            ),
        )
        for index, run_dir in enumerate(run_dirs, start=1)
    ]

    failures: list[tuple[Path, int]] = []

    if args.dry_run:
        for job in run_jobs:
            print(f"[{job.index}/{job.total}] {job.run_dir.name}")
            print(quote_command(job.command))
        return 0

    if jobs == 1:
        for job in run_jobs:
            returncode = run_job(job)
            if returncode != 0:
                failures.append((job.run_dir, returncode))
                if not args.continue_on_error:
                    break
    else:
        pending_jobs = list(run_jobs)
        running = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            while pending_jobs or running:
                while pending_jobs and len(running) < jobs:
                    job = pending_jobs.pop(0)
                    running[executor.submit(run_job, job)] = job

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    job = running.pop(future)
                    try:
                        returncode = future.result()
                    except Exception as exc:
                        print(
                            f"[{job.index}/{job.total}] FAIL {job.run_dir.name}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        returncode = 1

                    if returncode != 0:
                        failures.append((job.run_dir, returncode))
                        if not args.continue_on_error:
                            pending_jobs.clear()

    if failures:
        print("Failed run(s):", file=sys.stderr)
        for run_dir, returncode in failures:
            print(f"  - {run_dir} (exit {returncode})", file=sys.stderr)
        return failures[0][1]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
