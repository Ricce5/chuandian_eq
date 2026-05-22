#!/usr/bin/env python3
"""Backward-compatible entrypoint.

This file delegates to: `summarize/summarize_clf_mf_tf_grid.py`.
"""

from pathlib import Path
import runpy
import sys


def _delegate() -> None:
    current_file = Path(__file__).resolve()
    scripts_root = current_file.parent
    repo_root = scripts_root.parent
    target = scripts_root / "summarize/summarize_clf_mf_tf_grid.py"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    _delegate()
