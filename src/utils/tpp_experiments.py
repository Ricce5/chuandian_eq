from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import src.data.catalog as catalog

from src.utils.catalog_pathing import build_tpp_catalog_init_kwargs
from src.utils.utils import set_seed


def _cache_background_sequence(model: Any, bg_cache_seq: Any | None) -> None:
    if bg_cache_seq is None:
        return
    bg_model = getattr(model, "bg_model", None)
    if bg_model is None:
        return
    time_series = getattr(bg_cache_seq, "time_series", None)
    time_series_times = getattr(bg_cache_seq, "time_series_times", None)
    if time_series is None or time_series_times is None:
        return
    bg_model.cache_batch(
        time_series=time_series.unsqueeze(0),
        time_series_times=time_series_times.unsqueeze(0),
    )


def resolve_registered_catalog_class(
    dataset_name: str,
    *,
    candidates: Sequence[str] | None = None,
) -> tuple[str, type]:
    """Resolve a registered catalog class from common dataset name candidates."""
    search_names = list(candidates) if candidates is not None else [
        f"{dataset_name}-Standard",
        dataset_name,
    ]
    for registry_name in search_names:
        try:
            return registry_name, catalog.Catalog.by_name(registry_name)
        except RuntimeError:
            continue

    raise RuntimeError(
        f"No catalog class registered for dataset '{dataset_name}'. "
        f"Tried candidates={search_names}."
    )


def load_tpp_catalog(
    dataset_name: str,
    *,
    base_dir: str | Path,
    catalog_cfg: Mapping[str, Any] | None = None,
    candidates: Sequence[str] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Instantiate a TPP catalog and return it with registry name and init kwargs."""
    dataset_dir = Path(base_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    registry_name, catalog_cls = resolve_registered_catalog_class(
        dataset_name,
        candidates=candidates,
    )
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=catalog_cls,
        base_dir=dataset_dir,
        catalog_cfg=catalog_cfg,
    )
    return catalog_cls(**init_kwargs), registry_name, init_kwargs


def sample_tpp_forecasts(
    model: Any,
    past_seq: Any,
    *,
    duration: float,
    num_samples: int,
    samples_per_batch: int,
    bg_cache_seq: Any | None = None,
    seed: int | None = None,
    sample_max_length: int | None = None,
    predict_b: bool | None = None,
    verbose: bool | None = None,
) -> list[Any]:
    """Sample forecast sequences while adapting to model-specific sample kwargs."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive.")

    if seed is not None:
        set_seed(seed)
    model.eval()
    _cache_background_sequence(model, bg_cache_seq)

    try:
        sample_sig = inspect.signature(model.sample).parameters
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sample_sig.values()
        )
    except Exception:
        sample_sig = {}
        accepts_var_kwargs = False

    def _supports(name: str) -> bool:
        return accepts_var_kwargs or name in sample_sig

    static_kwargs: dict[str, Any] = {}
    if _supports("return_sequences"):
        static_kwargs["return_sequences"] = True
    if verbose is not None and _supports("verbose"):
        static_kwargs["verbose"] = verbose
    if predict_b is not None and _supports("predict_b"):
        static_kwargs["predict_b"] = predict_b
    if bg_cache_seq is not None and _supports("bg_cache_seq"):
        static_kwargs["bg_cache_seq"] = bg_cache_seq
    if seed is not None and _supports("random_state"):
        static_kwargs["random_state"] = int(seed)
    if sample_max_length is not None:
        if _supports("max_length"):
            static_kwargs["max_length"] = int(sample_max_length)
        if _supports("max_sample_len"):
            static_kwargs["max_sample_len"] = int(sample_max_length)

    all_forecasts: list[Any] = []
    remaining = int(num_samples)
    while remaining > 0:
        cur_batch_size = min(samples_per_batch, remaining)
        sample_kwargs = {
            "batch_size": cur_batch_size,
            "duration": duration,
            "past_seq": past_seq,
            **static_kwargs,
        }
        batch_forecasts = model.sample(**sample_kwargs)
        all_forecasts.extend(batch_forecasts)
        remaining -= cur_batch_size

    return all_forecasts
