#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .analysis_common import (
    find_metrics_file,
    load_summary_rows,
    lookup_metric_value,
    read_json,
    resolve_run_dir,
    to_float,
)


SEED_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")


def split_seed_suffix(run_name: str) -> tuple[str, Optional[int]]:
    text = str(run_name or "")
    matched = SEED_SUFFIX_PATTERN.match(text)
    if not matched:
        return text, None
    base = matched.group("base")
    try:
        seed_value = int(matched.group("seed"))
    except (TypeError, ValueError):
        seed_value = None
    return base, seed_value


def infer_variant_name_from_run(run_name: str | None, *, strip_tf_prefix: bool = False) -> str | None:
    if not run_name:
        return None
    text = str(run_name)
    marker = "_seed_"
    if marker in text:
        text = text.split(marker, 1)[0]
    if strip_tf_prefix:
        matched = re.match(r"^tf\d+_(.+)$", text)
        if matched:
            return matched.group(1)
    return text


def dedupe_summary_rows(summary_rows: Sequence[Dict]) -> List[Dict]:
    deduped: Dict[str, Dict] = {}
    extras: List[Dict] = []
    for item in summary_rows:
        run_name = item.get("run")
        if not run_name:
            extras.append(item)
            continue
        deduped[str(run_name)] = item
    return list(deduped.values()) + extras


def load_json_mapping(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return {}
    loaded = read_json(path)
    return loaded if isinstance(loaded, dict) else {}


def status_ok_from_summary(item: Dict) -> int:
    return int((item.get("train_returncode") in (None, 0)) and (item.get("test_returncode") in (None, 0)))


def collect_per_run_summary_rows(
    exp_dir: Path,
    metric_keys: Sequence[str],
    *,
    ckpt_select: str = "best",
    row_extra_builder: Callable[..., Optional[Dict]],
    sort_key_fn=None,
) -> List[Dict]:
    summary_rows = dedupe_summary_rows(load_summary_rows(exp_dir))

    rows: List[Dict] = []
    for item in summary_rows:
        run_name = item.get("run")
        if not run_name:
            continue

        run_name = str(run_name)
        run_dir = resolve_run_dir(exp_dir, run_name, item.get("run_dir"), ckpt_select="auto")
        metrics_path = find_metrics_file(run_dir, ckpt_select=ckpt_select)
        metrics = load_json_mapping(metrics_path)

        row = {
            "run": run_name,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path else "",
            "status_ok": status_ok_from_summary(item),
            "metrics_found": int(bool(metrics)),
        }
        extra_fields = row_extra_builder(
            item=item,
            run_name=run_name,
            run_dir=run_dir,
            metrics_path=metrics_path,
            metrics=metrics,
        )
        if extra_fields:
            row.update(extra_fields)

        for key in metric_keys:
            row[key] = to_float(lookup_metric_value(metrics, key))
        rows.append(row)

    if sort_key_fn is not None:
        rows.sort(key=sort_key_fn)
    return rows


def build_tf_mf_group_key(row: Dict):
    tf = row.get("Tfore")
    mf = row.get("Mf")
    if tf is None or mf is None:
        return None
    return int(tf), float(mf)


def sort_tf_mf_group_key(item):
    (tf, mf), _ = item
    return tf, mf


def build_variant_group_key(row: Dict, *, strip_tf_prefix: bool = False):
    variant = row.get("variant_name")
    if variant:
        return str(variant)
    return infer_variant_name_from_run(row.get("run"), strip_tf_prefix=strip_tf_prefix)


def build_variant_or_matrix_group_key(row: Dict):
    variant_name = str(row.get("variant_name") or "").strip()
    if variant_name:
        return "variant", variant_name
    attn_layer_idx = row.get("attn_layer_idx")
    load_strategy = row.get("load_strategy")
    if attn_layer_idx is None or load_strategy is None:
        return None
    return "matrix", f"attn_l{int(attn_layer_idx)}__{str(load_strategy)}"


def sort_group_type_name_key(item):
    (group_type, group_name), _ = item
    return group_type, group_name


def resolve_cfg_path(run_dir: Path, cfg_path_raw) -> Path | None:
    candidates: List[Path] = []
    if cfg_path_raw:
        raw = Path(str(cfg_path_raw)).expanduser()
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((run_dir / raw).resolve())

    candidates.extend([run_dir / "config_input.yaml", run_dir / "config.yaml"])

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def resolve_tf_mf_group_mode(rows: Sequence[Dict], mode_raw: str, variant_key_getter) -> str:
    mode = str(mode_raw).strip().lower()
    if mode in {"tf_mf", "variant"}:
        return mode
    if mode != "auto":
        raise ValueError("--group_by must be one of: auto, tf_mf, variant")

    combos = set()
    variants = set()
    for row in rows:
        tf = row.get("Tfore")
        mf = row.get("Mf")
        if tf is not None and mf is not None:
            combos.add((int(tf), float(mf)))
        variant_key = variant_key_getter(row)
        if variant_key is not None:
            variants.add(variant_key)

    if len(combos) <= 1 and len(variants) > 1:
        return "variant"
    return "tf_mf"


def safe_mapping(value) -> Dict:
    return value if isinstance(value, dict) else {}


def is_single_window_variant_grid(rows: Sequence[Dict]) -> bool:
    combos = set()
    for row in rows:
        tf = row.get("Tfore")
        mf = row.get("Mf")
        if tf is not None and mf is not None:
            combos.add((int(tf), float(mf)))
    variant_names = {row.get("variant_name") for row in rows if row.get("variant_name")}
    return len(combos) == 1 and len(variant_names) > 1
