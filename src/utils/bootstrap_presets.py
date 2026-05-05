"""Reusable bootstrap presets for paired multi-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.bootstrap_ci import BootstrapConfig, SamplingMethod

__all__ = ["PairedBootstrapPreset"]


@dataclass(frozen=True)
class PairedBootstrapPreset:
    """Preset bundle for paired bootstrap in comparison experiments.

    Parameters
    ----------
    sampling:
        Resampling mode, ``"iid"`` or ``"block"``.
    ci:
        Percentile interval, e.g. ``(2.5, 97.5)``.
    n_resamples_curve:
        Number of resamples used for curve CI (ROC/PR envelopes).
    n_resamples_metric:
        Number of resamples used for scalar metrics (AUC/AP/etc.).
    block_size:
        Block length used only when ``sampling="block"``.
    circular_block:
        Whether block bootstrap uses circular/wrap-around indexing.
    """

    sampling: SamplingMethod = "iid"
    ci: tuple[float, float] = (2.5, 97.5)
    n_resamples_curve: int = 400
    n_resamples_metric: int = 800
    block_size: int | None = 24
    circular_block: bool = True

    def __post_init__(self) -> None:
        if self.n_resamples_curve <= 0:
            raise ValueError("n_resamples_curve must be positive.")
        if self.n_resamples_metric <= 0:
            raise ValueError("n_resamples_metric must be positive.")

        low, high = self.ci
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(f"Invalid ci percentile bounds: {self.ci!r}")

        if self.block_size is not None and self.block_size <= 0:
            raise ValueError("block_size must be positive when provided.")

    def _effective_block_size(self) -> int | None:
        if self.sampling != "block":
            return None
        return self.block_size

    def curve_config(self, *, seed: int | None) -> BootstrapConfig:
        """Build :class:`BootstrapConfig` for curve bootstrap."""

        return BootstrapConfig(
            n_resamples=self.n_resamples_curve,
            ci=self.ci,
            seed=seed,
            sampling=self.sampling,
            block_size=self._effective_block_size(),
            circular_block=self.circular_block,
        )

    def metric_config(self, *, seed: int | None) -> BootstrapConfig:
        """Build :class:`BootstrapConfig` for metric bootstrap."""

        return BootstrapConfig(
            n_resamples=self.n_resamples_metric,
            ci=self.ci,
            seed=seed,
            sampling=self.sampling,
            block_size=self._effective_block_size(),
            circular_block=self.circular_block,
        )
