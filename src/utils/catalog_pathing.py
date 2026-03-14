from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.utils.file_utils import build_catalog_root_dir


def to_serializable_ts(ts: Any) -> Any:
    """Convert pandas timestamps to JSON-safe strings."""
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return ts


def resolve_dataset_data_dir(
    root_dir: str | Path,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve canonical dataset root ``data/<dataset>`` from runtime paths."""
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    root = Path(root_dir).expanduser().resolve()
    if root.name in {"raw", "processed", "catalogs"}:
        return root.parent
    if root.parent.name == "catalogs":
        return root.parent.parent
    return root


def resolve_catalogs_dir(dataset_dir: str | Path) -> Path:
    """Resolve canonical catalog cache dir ``data/<dataset>/catalogs``."""
    return Path(dataset_dir).expanduser().resolve() / "catalogs"


def migrate_legacy_catalog_artifacts(
    base_catalogs_dir: str | Path,
    target_dir: str | Path,
) -> None:
    """Copy legacy non-hashed catalog artifacts into hash cache dir once."""
    base = Path(base_catalogs_dir).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    if not base.exists() or base == target:
        return

    target.mkdir(parents=True, exist_ok=True)
    patterns = (
        "full_sequence.pt",
        "metadata.pt",
        "norm_stats.pt",
        "catalog.csv",
        "catalog_*.csv",
        "train.pt",
        "val.pt",
        "test.pt",
    )
    for pattern in patterns:
        for src in base.glob(pattern):
            if not src.is_file():
                continue
            dst = target / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)


def resolve_source_file(
    dataset_dir: str | Path,
    default_filename: str,
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve source file path from canonical raw/processed dirs."""
    if explicit_path is not None:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {p}")
        return p

    ds = Path(dataset_dir).expanduser().resolve()
    candidates = [
        ds / "raw" / default_filename,
        ds / "processed" / default_filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Source file not found: {default_filename}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def build_hashed_catalog_root(
    root_dir: str | Path,
    catalog_cfg: Mapping[str, Any],
    *,
    migrate_legacy: bool = True,
) -> Path:
    """Build hashed cache directory from config and return absolute path."""
    sub_root_dir, _ = build_catalog_root_dir(root_dir, catalog_cfg)
    hashed_root = Path(sub_root_dir).expanduser().resolve()
    if migrate_legacy:
        migrate_legacy_catalog_artifacts(root_dir, hashed_root)
    return hashed_root


def refresh_cached_metadata(
    root_dir: str | Path,
    metadata: Mapping[str, Any],
    required_files: Sequence[str],
) -> None:
    """When legacy cache files exist, align metadata to current schema."""
    root = Path(root_dir).expanduser().resolve()
    if all((root / fname).exists() for fname in required_files):
        torch.save(dict(metadata), root / "metadata.pt")


def build_tpp_catalog_init_kwargs(
    catalog_ds_class: type,
    dataset_name: str,
    base_dir: str | Path,
    catalog_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic init kwargs for catalog constructors."""
    _ = dataset_name  # kept for backward-compatible API
    dataset_dir = resolve_dataset_data_dir(root_dir=base_dir, data_dir=None)
    cfg = dict(catalog_cfg or {})

    init_sig = inspect.signature(catalog_ds_class.__init__).parameters
    supports_data_dir = "data_dir" in init_sig
    is_grouped_family = any(
        base.__name__ == "InducedTripletGroupedCatalog"
        for base in catalog_ds_class.__mro__
    )

    root_dir = Path(cfg.get("root_dir", resolve_catalogs_dir(dataset_dir))).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs: dict[str, Any] = {"root_dir": str(root_dir), **cfg}

    if supports_data_dir and "data_dir" not in init_kwargs:
        if not is_grouped_family:
            init_kwargs["data_dir"] = str(dataset_dir)

    return init_kwargs

