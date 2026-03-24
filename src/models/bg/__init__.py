"""Background model registry entrypoint."""

from .base import BGModel


def _safe_import(module_name: str) -> None:
    try:
        __import__(f"{__name__}.{module_name}")
    except ModuleNotFoundError:
        # Some background models rely on optional acceleration libraries.
        # Keep the package importable so models without those dependencies
        # remain usable and testable.
        return


for _module_name in (
    "conv_mlp",
    "gp_latent_bg",
    "kernel",
    "latent_bg",
    "mamba",
    "ncde",
    "proportional",
    "rnn",
    "ssm",
    "stochastic_ssm",
):
    _safe_import(_module_name)

__all__ = ["BGModel"]
