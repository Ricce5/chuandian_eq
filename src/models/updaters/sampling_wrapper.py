from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import torch

from .base import BValueUpdaterBase


def _infer_model_dtype(model: Any) -> torch.dtype:
    """Infer dtype from model parameters; fallback to default dtype."""
    try:
        return next(model.parameters()).dtype
    except Exception:
        return torch.get_default_dtype()


def _infer_model_device(
    model: Any,
    device: Optional[torch.device | str] = None,
) -> torch.device:
    target_device = getattr(model, "device", None) if device is None else device
    if target_device is None:
        return torch.device("cpu")
    return torch.device(target_device)


def _infer_mag_completeness(model: Any) -> float | None:
    if not hasattr(model, "mag_completeness"):
        return None
    mc = getattr(model, "mag_completeness")
    if isinstance(mc, torch.Tensor):
        if mc.numel() != 1:
            raise ValueError("model.mag_completeness must be scalar when used for updater factory.")
        return float(mc.detach().cpu().item())
    return float(mc)


def _normalize_updater_config(
    updater_name: Optional[str],
    updater_cfg: Optional[dict[str, Any]],
) -> tuple[Optional[str], dict[str, Any]]:
    cfg = dict(updater_cfg or {})
    resolved_name = updater_name
    if resolved_name is None:
        raw_name = cfg.pop("name", None)
        if raw_name is None:
            raw_name = cfg.pop("type", None)
        if raw_name is not None:
            resolved_name = str(raw_name)
    return resolved_name, cfg


def _select_supported_kwargs(constructor, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(constructor)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return kwargs
    valid = set(sig.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in valid}


def _call_model_updater_hook(
    model: Any,
    *,
    updater_name: Optional[str] = None,
    updater_cfg: Optional[dict[str, Any]] = None,
):
    hook = getattr(model, "_build_sampling_b_updater", None)
    if not callable(hook):
        return None

    sig = inspect.signature(hook)
    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    hook_kwargs: dict[str, Any] = {}
    if accepts_kwargs or "updater_name" in params:
        hook_kwargs["updater_name"] = updater_name
    if accepts_kwargs or "updater_cfg" in params:
        hook_kwargs["updater_cfg"] = dict(updater_cfg or {})

    if hook_kwargs:
        try:
            return hook(**hook_kwargs)
        except TypeError:
            return hook()
    return hook()


def build_sampling_updater(
    model: Any,
    *,
    mc: float | None = None,
    updater_name: Optional[str] = None,
    updater_cfg: Optional[dict[str, Any]] = None,
    device: Optional[torch.device | str] = None,
) -> BValueUpdaterBase:
    """Build a sampling-time updater instance via the updater registry."""
    resolved_name, cfg = _normalize_updater_config(updater_name, updater_cfg)
    resolved_name = resolved_name or getattr(model, "b_updater_name", None)
    if resolved_name is None:
        raise ValueError("No updater_name provided for sampling updater.")

    inferred_mc = _infer_mag_completeness(model) if mc is None else float(mc)
    init_kwargs: dict[str, Any] = {
        "mag_key": "mag",
        "write_back": False,
        "dtype": _infer_model_dtype(model),
        "device": _infer_model_device(model, device=device),
    }
    if inferred_mc is not None:
        init_kwargs["Mc"] = inferred_mc
    init_kwargs.update(cfg)

    constructor = BValueUpdaterBase.by_name(resolved_name)
    final_kwargs = _select_supported_kwargs(constructor, init_kwargs)
    return constructor(**final_kwargs)


def resolve_sampling_updater(
    model: Any,
    *,
    updater: Any = None,
    updater_name: Optional[str] = None,
    updater_cfg: Optional[dict[str, Any]] = None,
    device: Optional[torch.device | str] = None,
    prefer_model_hook: bool = True,
):
    """Resolve updater instance from explicit object, model hook, or registry."""
    if updater is not None:
        return updater

    resolved_name, cfg = _normalize_updater_config(updater_name, updater_cfg)

    if prefer_model_hook:
        updater_from_hook = _call_model_updater_hook(
            model,
            updater_name=resolved_name,
            updater_cfg=cfg,
        )
        if updater_from_hook is not None:
            return updater_from_hook

    has_updater_spec = resolved_name is not None or bool(cfg) or getattr(model, "b_updater_name", None) is not None
    if not has_updater_spec:
        return None

    return build_sampling_updater(
        model=model,
        updater_name=resolved_name,
        updater_cfg=cfg,
        device=device,
    )


def make_sampling_updater(
    model: Any,
    *,
    mc: float,
    updater_cfg: Optional[dict[str, Any]] = None,
    device: Optional[torch.device | str] = None,
    updater_name: Optional[str] = None,
) -> BValueUpdaterBase:
    """Backward-compatible alias for `build_sampling_updater`."""
    return build_sampling_updater(
        model=model,
        mc=mc,
        updater_name=updater_name,
        updater_cfg=updater_cfg,
        device=device,
    )


class UpdaterSamplingWrapper:
    """Wrap a model and inject updater kwargs into ``sample`` when needed."""

    def __init__(
        self,
        model: Any,
        *,
        sampling_mode: str = "model",
        updater_factory: Optional[Callable[[], Any]] = None,
        updater_name: Optional[str] = None,
        updater_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.b_sampling = str(sampling_mode)
        self.updater_factory = updater_factory
        self.updater_name = updater_name
        self.updater_cfg = dict(updater_cfg or {})

    def sample(
        self,
        batch_size: int,
        duration: float,
        past_seq=None,
        return_sequences: bool = False,
        **kwargs,
    ):
        if self.b_sampling != "model":
            kwargs.setdefault("b_sampling", self.b_sampling)

            if kwargs.get("updater") is None:
                if self.updater_factory is not None:
                    updater = self.updater_factory()
                    kwargs["updater"] = updater
                elif self.updater_name is not None or self.updater_cfg:
                    if self.updater_name is not None:
                        kwargs.setdefault("updater_name", self.updater_name)
                    if self.updater_cfg:
                        kwargs.setdefault("updater_cfg", dict(self.updater_cfg))

        kwargs = _select_supported_kwargs(self.model.sample, kwargs)
        return self.model.sample(
            batch_size=batch_size,
            duration=duration,
            past_seq=past_seq,
            return_sequences=return_sequences,
            **kwargs,
        )

    def __getattr__(self, name: str):
        return getattr(self.model, name)


__all__ = [
    "UpdaterSamplingWrapper",
    "_infer_model_dtype",
    "build_sampling_updater",
    "make_sampling_updater",
    "resolve_sampling_updater",
]
