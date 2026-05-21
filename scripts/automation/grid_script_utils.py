#!/usr/bin/env python3
import re
from typing import Dict, Optional

import yaml


def normalize_run_name(raw: str, fallback: str) -> str:
    text = str(raw or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^0-9a-zA-Z._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def build_tf_mf_seed_variant_name(tf: int, mf: float, seed: int) -> str:
    mf_str = str(mf).replace(".", "p")
    return f"tf_{tf}_mf_{mf_str}_seed_{seed}"


def build_attn_load_seed_variant_name(attn_layer: int, load_strategy: str, seed: int) -> str:
    return f"attn_l{attn_layer}_load_{load_strategy}_seed_{seed}"


def parse_set_by_tf(raw) -> Dict[int, Dict]:
    if raw is None:
        return {}

    if isinstance(raw, dict):
        loaded = raw
    else:
        text = str(raw).strip()
        if not text:
            return {}
        loaded = yaml.safe_load(text)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("set_by_tf must be a mapping: {Tfore: {dotted.key: value, ...}, ...}")

    out: Dict[int, Dict] = {}
    for tf_raw, overrides in loaded.items():
        tf = int(tf_raw)
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise ValueError(
                f"set_by_tf[{tf}] must be a mapping of config overrides, got: {type(overrides)}"
            )
        out[tf] = dict(overrides)
    return out


def expand_grid_points(
    exp_cfg: dict,
    *,
    forward_keys,
    reserved_keys,
    default_name_prefix: str = "grid",
) -> Optional[list[dict]]:
    if not isinstance(exp_cfg, dict):
        return None

    grid_cfg = exp_cfg.get("grid")
    if grid_cfg is None:
        return None
    if not isinstance(grid_cfg, dict):
        raise ValueError("exp config key `grid` must be a mapping/object.")

    points = grid_cfg.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("exp config key `grid.points` must be a non-empty list.")

    name_prefix = str(grid_cfg.get("name_prefix", default_name_prefix)).strip()
    common = grid_cfg.get("common", {})
    if common is None:
        common = {}
    if not isinstance(common, dict):
        raise ValueError("exp config key `grid.common` must be a mapping/object.")

    common_overrides = common.get("overrides", {})
    if common_overrides is None:
        common_overrides = {}
    if not isinstance(common_overrides, dict):
        raise ValueError("exp config key `grid.common.overrides` must be a mapping/object.")

    reserved_keys_set = set(reserved_keys)
    forward_keys_list = list(forward_keys)

    expanded = []
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"grid.points[{idx}] must be a mapping/object.")

        point_overrides = point.get("overrides", {})
        if point_overrides is None:
            point_overrides = {}
        if not isinstance(point_overrides, dict):
            raise ValueError(f"grid.points[{idx}].overrides must be a mapping/object.")

        point_name_raw = point.get("name", f"{idx + 1:02d}")
        point_name = normalize_run_name(point_name_raw, fallback=f"{idx + 1:02d}")
        if name_prefix:
            variant_name = normalize_run_name(
                f"{name_prefix}_{point_name}",
                fallback=f"{name_prefix}_{idx + 1:02d}",
            )
        else:
            variant_name = point_name

        merged = {"name": variant_name}
        for key in forward_keys_list:
            if key in common:
                merged[key] = common[key]
            if key in point:
                merged[key] = point[key]

        shorthand_overrides = {
            key: value
            for key, value in point.items()
            if key not in reserved_keys_set
        }
        merged_overrides = {}
        merged_overrides.update(common_overrides)
        merged_overrides.update(shorthand_overrides)
        merged_overrides.update(point_overrides)
        merged["overrides"] = merged_overrides

        expanded.append(merged)

    return expanded


def resolve_point_seeds(point: dict, default_seeds: list[int]) -> list[int]:
    if "seeds" in point and point.get("seeds") is not None:
        raw = point.get("seeds")
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw]
        items = [x.strip() for x in str(raw).split(",") if x.strip()]
        return [int(x) for x in items]
    if "seed" in point and point.get("seed") is not None:
        return [int(point.get("seed"))]
    return [int(x) for x in default_seeds]


def parse_load_strategy_item(raw: str):
    item = raw.strip()
    if not item:
        raise ValueError("Empty load strategy item.")
    if ":" not in item:
        raise ValueError(f"Invalid load strategy item: {raw!r}. Expected name:value")
    name, value = item.split(":", 1)
    name = name.strip()
    value = value.strip().lower()
    if value in {"none", "null"}:
        parsed = None
    elif value in {"input_layer0_layer1", "input_l0_l1", "layer01_input", "l01_input", "three"}:
        parsed = "input_layer0_layer1"
    elif value in {"input_layer", "input", "input_proj", "inproj"}:
        parsed = "input_layer"
    elif value in {"layer0", "l0"}:
        parsed = "layer0"
    elif value in {"layer1", "l1"}:
        parsed = "layer1"
    elif value in {"layer0_input", "l0_input", "first", "first_layer"}:
        parsed = "layer0_input"
    elif value in {"layer1_input", "l1_input", "second", "second_layer"}:
        parsed = "layer1_input"
    elif value in {"encoder"}:
        parsed = "encoder"
    else:
        raise ValueError(
            f"Unsupported load strategy value: {value!r}. "
            "Use one of: none, input_layer0_layer1, input_layer, layer0, layer1, layer0_input, layer1_input, encoder."
        )
    return name, parsed


def parse_load_strategies(raw: str):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise ValueError("No load strategies provided.")
    parsed = {}
    for item in items:
        name, value = parse_load_strategy_item(item)
        parsed[name] = value
    return parsed


def parse_single_strategy_value(raw: str):
    _, parsed = parse_load_strategy_item(f"variant:{raw}")
    return parsed


def resolve_variant_load_strategy(raw_value, load_strategy_map: dict):
    if raw_value is None:
        return "base", "__KEEP_BASE__"
    text = str(raw_value).strip()
    if not text:
        return "base", "__KEEP_BASE__"
    text_lower = text.lower()
    if text_lower in {"base", "keep_base", "inherit"}:
        return "base", "__KEEP_BASE__"
    if text in load_strategy_map:
        return text, load_strategy_map[text]
    parsed = parse_single_strategy_value(text)
    if parsed is None:
        return "none", None
    alias_name = text.lower().replace(":", "_")
    alias_name = re.sub(r"[^0-9a-zA-Z._-]+", "_", alias_name)
    return alias_name, parsed


def apply_encoder_load_strategy(cfg, strategy_value, pretrain_resume_path):
    cfg.pop("load_specific_parts", None)
    cfg.pop("encoder_param_keywords", None)
    cfg.pop("resume_path", None)
    if strategy_value is None:
        return
    if not pretrain_resume_path:
        raise ValueError(
            "Selected load_strategy requires pretrained checkpoint, but base config has empty resume_path. "
            "Set resume_path in base config or use load_strategy=none/base."
        )
    if strategy_value == "encoder":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = ["encoder"]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "input_layer0_layer1":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.input_proj",
            "base_model.encoder.layers.0",
            "base_model.encoder.layers.1",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "input_layer":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer0":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.0",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer1":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.1",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer0_input":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.0",
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    if strategy_value == "layer1_input":
        cfg.resume_path = pretrain_resume_path
        cfg.load_specific_parts = [
            "base_model.encoder.layers.1",
            "base_model.encoder.input_proj",
        ]
        cfg.encoder_param_keywords = list(cfg.load_specific_parts)
        return
    raise ValueError(f"Unknown load strategy value: {strategy_value!r}")
