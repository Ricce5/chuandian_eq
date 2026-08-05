"""Synthetic induced-seismicity experiments for ETAS background models.

The functions in this module keep the causal order explicit:

``virtual injection -> ETAS background immigrants -> triggered descendants``.

They are designed for *model-recovery* experiments.  Event provenance labels
are exact only for a synthetic catalog generated with
``ETAS.sample(return_event_metadata=True)``; on an observed catalog the
returned background/triggered values are model responsibilities, not observed
causal labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence as TypingSequence

import numpy as np
import torch

from src.data import Batch, Sequence, TppDataset
from src.utils.catalog_utils import train_val_test_split_sequence_float


def moving_block_bootstrap_injection(
    injection: np.ndarray | TypingSequence[float],
    *,
    target_length: int | None = None,
    block_size: int = 24,
    random_state: int | None = None,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Generate an exogenous injection scenario by resampling contiguous blocks.

    Block resampling preserves within-block pulse shape and local autocorrelation,
    unlike iid resampling.  It does not use the earthquake catalog and therefore
    cannot introduce an artificial earthquake-to-injection feedback path.

    Args:
        injection: One-dimensional injection-rate series, or a ``(T, F)``
            multi-well series.
        target_length: Number of output samples. Defaults to the input length.
        block_size: Number of consecutive samples per sampled block.
        random_state: Optional NumPy random seed.
        clip_nonnegative: Clamp output to zero after resampling.
    """
    values = np.asarray(injection, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("injection must have shape (T,) or (T, F) with T > 0.")
    if not np.isfinite(values).all():
        raise ValueError("injection contains non-finite values.")

    n_samples = int(values.shape[0])
    target_length = n_samples if target_length is None else int(target_length)
    block_size = int(block_size)
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if not 1 <= block_size <= n_samples:
        raise ValueError(
            f"block_size must be in [1, {n_samples}], got {block_size}."
        )

    rng = np.random.default_rng(random_state)
    pieces: list[np.ndarray] = []
    produced = 0
    max_start = n_samples - block_size
    while produced < target_length:
        start = int(rng.integers(0, max_start + 1))
        block = values[start : start + block_size]
        pieces.append(block)
        produced += block_size

    output = np.concatenate(pieces, axis=0)[:target_length].copy()
    if clip_nonnegative:
        output = np.maximum(output, 0.0)
    return output[:, 0] if was_vector else output


def normalize_injection_with_reference(
    raw_injection: np.ndarray | TypingSequence[float],
    norm_stats: Mapping[str, Any],
    *,
    field: str = "inj_rate",
) -> np.ndarray:
    """Apply fixed reference min-max statistics without mutating a catalog.

    ``Catalog.normalize_fields`` estimates and overwrites min/max statistics.
    That is appropriate while constructing an observed catalog but invalid for a
    recovery experiment: every virtual scenario must be seen on the scale used
    to train the original model.
    """
    if field not in norm_stats:
        raise KeyError(f"No normalization statistics for field {field!r}.")
    stats = norm_stats[field]
    try:
        min_value = float(stats["min"])
        max_value = float(stats["max"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"norm_stats[{field!r}] must contain scalar 'min' and 'max' values."
        ) from exc

    raw = np.asarray(raw_injection, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("raw_injection contains non-finite values.")
    return (raw - min_value) / max(max_value - min_value, 1e-8)


def _as_single_feature_injection(
    values: np.ndarray | TypingSequence[float],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(f"{name} must have shape (T,) or (T, 1).")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must have at least two time samples.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


@torch.inference_mode()
def simulate_etas_virtual_catalog(
    generator: Any,
    *,
    raw_injection: np.ndarray | TypingSequence[float],
    injection_times: np.ndarray | TypingSequence[float],
    norm_stats: Mapping[str, Any] | None = None,
    normalization_field: str = "inj_rate",
    random_state: int = 123,
    max_length: int | None = 50_000,
    t_nll_start: float | None = None,
) -> Sequence:
    """Generate one labelled ETAS catalog under a virtual injection scenario.

    ``generator`` must be an :class:`~src.models.tpp.etas.ETAS` model fitted
    with a background model.  The background model cache is intentionally
    replaced with the supplied scenario before sampling.
    """
    if getattr(generator, "bg_model", None) is None:
        raise ValueError(
            "simulate_etas_virtual_catalog requires an ETAS generator with bg_model."
        )
    if not hasattr(generator.bg_model, "cache_batch"):
        raise TypeError("generator.bg_model must implement cache_batch().")

    raw = _as_single_feature_injection(raw_injection, name="raw_injection")
    if norm_stats is None:
        normalized = raw.copy()
    else:
        normalized = _as_single_feature_injection(
            normalize_injection_with_reference(
                raw,
                norm_stats,
                field=normalization_field,
            ),
            name="normalized_injection",
        )

    times = np.asarray(injection_times, dtype=np.float64).reshape(-1)
    if times.size != raw.shape[0]:
        raise ValueError(
            "injection_times and raw_injection must have the same number of samples."
        )
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("injection_times must be finite and strictly increasing.")

    t_start = float(times[0])
    t_end = float(times[-1])
    if t_nll_start is None:
        t_nll_start = t_start
    if not t_start <= float(t_nll_start) <= t_end:
        raise ValueError("t_nll_start must lie within the injection time range.")

    time_series = torch.as_tensor(normalized, dtype=torch.float32)
    time_series_times = torch.as_tensor(times, dtype=torch.float32)
    generator.bg_model.cache_batch(
        time_series.unsqueeze(0),
        time_series_times.unsqueeze(0),
        cache_lambda=True,
    )
    sampled = generator.sample(
        batch_size=1,
        t_start=t_start,
        duration=t_end - t_start,
        random_state=int(random_state),
        max_length=max_length,
        n_jobs=1,
        return_sequences=True,
        return_event_metadata=True,
    )[0]

    return Sequence(
        inter_times=sampled.inter_times,
        t_start=sampled.t_start,
        t_nll_start=float(t_nll_start),
        mag=sampled.mag,
        etas_source=sampled.etas_source,
        etas_parent_index=sampled.etas_parent_index,
        etas_generation=sampled.etas_generation,
        time_series=time_series,
        time_series_times=time_series_times,
        raw_time_series=torch.as_tensor(raw, dtype=torch.float32),
        raw_time_series_times=time_series_times,
    )


def save_virtual_induced_catalog(
    output_dir: str | Path,
    sequence: Sequence,
    *,
    train_start: float | None = None,
    val_start: float,
    test_start: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist a virtual full/train/val/test TPP dataset plus metadata.

    The output is intentionally simple: ``full_sequence.pt``, ``train.pt``,
    ``val.pt``, ``test.pt`` and ``metadata.pt``.  It can be loaded with
    :class:`~src.data.tpp_dataset.TppDataset` without regenerating source data.
    Existing artifact files are never overwritten. ``metadata`` must include
    ``mag_completeness`` (normally ``float(generator.M_c)``), so a recovered
    ETAS model uses the same magnitude reference as its generator.
    """
    user_metadata = dict(metadata or {})
    if "mag_completeness" not in user_metadata:
        raise ValueError(
            "metadata must include mag_completeness, usually float(generator.M_c)."
        )

    root = Path(output_dir).expanduser().resolve()
    val_start = float(val_start)
    test_start = float(test_start)
    train_start = sequence.t_start if train_start is None else float(train_start)
    if not sequence.t_start <= train_start <= val_start <= test_start <= sequence.t_end:
        raise ValueError("Require t_start <= train_start <= val_start <= test_start <= t_end.")

    paths = {
        "full_sequence": root / "full_sequence.pt",
        "train": root / "train.pt",
        "val": root / "val.pt",
        "test": root / "test.pt",
        "metadata": root / "metadata.pt",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing virtual artifacts: {names}")

    root.mkdir(parents=True, exist_ok=True)
    train_seq, val_seq, test_seq = train_val_test_split_sequence_float(
        sequence,
        start_ts=float(sequence.t_start),
        train_start_ts=train_start,
        val_start_ts=val_start,
        test_start_ts=test_start,
    )
    TppDataset([sequence]).save_to_disk(paths["full_sequence"])
    TppDataset([train_seq]).save_to_disk(paths["train"])
    TppDataset([val_seq]).save_to_disk(paths["val"])
    TppDataset([test_seq]).save_to_disk(paths["test"])

    saved_metadata = {
        "name": "VirtualInducedETAS",
        "schema_version": 1,
        "mag_roundoff_error": 0.0,
        "freq": "1D",
        "t_start": float(sequence.t_start),
        "t_end": float(sequence.t_end),
        "train_start": train_start,
        "val_start": val_start,
        "test_start": test_start,
        "has_etas_simulation_truth": all(
            key in sequence
            for key in ("etas_source", "etas_parent_index", "etas_generation")
        ),
    }
    saved_metadata.update(user_metadata)
    torch.save(saved_metadata, paths["metadata"])
    return paths


def _resolve_device(model: Any, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("device must be provided for a model without parameters.") from exc


def _validate_query_times(t_query: np.ndarray | TypingSequence[float]) -> np.ndarray:
    times = np.asarray(t_query, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("t_query must contain at least one time.")
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0.0):
        raise ValueError("t_query must be finite and non-decreasing.")
    return times


def _trapezoid_integral(values: np.ndarray, times: np.ndarray) -> float:
    """Integrate a one-dimensional trajectory without NumPy-version coupling."""
    if times.size < 2:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(times)))


@torch.inference_mode()
def decompose_etas_intensity(
    model: Any,
    sequence: Sequence,
    t_query: np.ndarray | TypingSequence[float],
    *,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate ETAS constant, injection and triggered intensity components.

    In this repository ``ETAS.h_intensity`` is ``mu + trigger``.  The helper
    therefore subtracts ``mu`` before reporting the pure triggered component;
    ``background`` is consistently defined as ``mu + injection``.
    """
    if not hasattr(model, "h_intensity"):
        raise TypeError("model must expose ETAS-style h_intensity().")
    times = _validate_query_times(t_query)
    model_device = _resolve_device(model, device)
    batch = Batch.from_list([sequence]).to(model_device)
    query = torch.as_tensor(
        times,
        dtype=batch.arrival_times.dtype,
        device=model_device,
    ).unsqueeze(0)

    h_plus_mu = model.h_intensity(batch, t_query=query)[0]
    mu = model.mu.to(device=model_device, dtype=h_plus_mu.dtype)
    constant = torch.ones_like(h_plus_mu) * mu
    trigger = (h_plus_mu - constant).clamp_min(0.0)

    if getattr(model, "bg_model", None) is None:
        injection = torch.zeros_like(trigger)
    else:
        injection = model.bg_model.intensity(batch, t_query=query)[0].clamp_min(0.0)
    background = constant + injection
    total = background + trigger

    return {
        "time": times.copy(),
        "constant": constant.detach().cpu().numpy(),
        "injection": injection.detach().cpu().numpy(),
        "background": background.detach().cpu().numpy(),
        "trigger": trigger.detach().cpu().numpy(),
        "total": total.detach().cpu().numpy(),
    }


def etas_event_responsibilities(
    model: Any,
    sequence: Sequence,
    *,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Return probabilistic background/triggered attribution at catalog events."""
    event_times = sequence.arrival_times.detach().cpu().numpy()
    if event_times.size == 0:
        return {
            "event_time": event_times.astype(np.float64),
            "background_probability": np.empty(0, dtype=np.float64),
            "trigger_probability": np.empty(0, dtype=np.float64),
        }
    parts = decompose_etas_intensity(model, sequence, event_times, device=device)
    total = np.maximum(parts["total"], 1e-12)
    result = {
        "event_time": event_times.astype(np.float64, copy=True),
        "background_probability": parts["background"] / total,
        "trigger_probability": parts["trigger"] / total,
    }
    if "etas_source" in sequence:
        result["simulation_source"] = sequence.etas_source.detach().cpu().numpy()
    if "etas_parent_index" in sequence:
        result["simulation_parent_index"] = sequence.etas_parent_index.detach().cpu().numpy()
    return result


def compare_etas_components(
    original_model: Any,
    fitted_model: Any,
    sequence: Sequence,
    t_query: np.ndarray | TypingSequence[float],
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Compare original and fitted ETAS components on the same forcing/history.

    The same ``sequence`` and query grid are passed to both models.  This is
    crucial: comparing each model on a separately simulated history confounds
    parameter-recovery error with different realized aftershock cascades.
    """
    times = _validate_query_times(t_query)
    original = decompose_etas_intensity(
        original_model,
        sequence,
        times,
        device=device,
    )
    fitted = decompose_etas_intensity(
        fitted_model,
        sequence,
        times,
        device=device,
    )

    metrics: dict[str, float] = {}
    for component in ("constant", "injection", "background", "trigger", "total"):
        delta = fitted[component] - original[component]
        metrics[f"{component}_mae"] = float(np.mean(np.abs(delta)))
        metrics[f"{component}_rmse"] = float(np.sqrt(np.mean(np.square(delta))))
        metrics[f"{component}_expected_count_original"] = _trapezoid_integral(
            original[component],
            times,
        )
        metrics[f"{component}_expected_count_fitted"] = _trapezoid_integral(
            fitted[component],
            times,
        )

    return {
        "time": times.copy(),
        "original": original,
        "fitted": fitted,
        "metrics": metrics,
    }


__all__ = [
    "compare_etas_components",
    "decompose_etas_intensity",
    "etas_event_responsibilities",
    "moving_block_bootstrap_injection",
    "normalize_injection_with_reference",
    "save_virtual_induced_catalog",
    "simulate_etas_virtual_catalog",
]
