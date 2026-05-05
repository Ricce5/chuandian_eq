"""Runtime-oriented utility helpers.

This module collects project-root resolution and model-wrapper helpers that are
commonly used in notebooks and scripts.
"""

from pathlib import Path

__all__ = [
    "resolve_project_root",
    "unwrap_compiled_model",
]


def resolve_project_root() -> Path:
    """Find project root by locating the nearest directory containing ``src/``."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing 'src' directory.")


def unwrap_compiled_model(model):
    """Return original model when ``torch.compile`` wrapper is present."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model

