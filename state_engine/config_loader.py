"""Config loader utilities for symbol-specific overrides."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PHASE_D_FILTER_FIELDS = (
    "sessions_in",
    "state_age_min",
    "state_age_max",
    "dist_vwap_atr_min",
    "dist_vwap_atr_max",
    "breakmag_min",
    "breakmag_max",
    "reentry_min",
    "reentry_max",
    "margin_min",
    "margin_max",
)


def deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides onto defaults."""
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file and validate minimal schema.
    If path is relative, resolve it against project root.
    """
    config_path = Path(path)

    # --- NEW: resolve relative paths against project root ---
    if not config_path.is_absolute():
        # state_engine/config_loader.py -> project root = parents[1]
        project_root = Path(__file__).resolve().parents[1]
        config_path = (project_root / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    yaml_spec = importlib.util.find_spec("yaml")
    has_yaml = yaml_spec is not None

    if suffix in {".yaml", ".yml"}:
        if not has_yaml:
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "PyYAML not installed; use a .json config or install PyYAML."
                ) from exc
        else:
            import yaml
            data = yaml.safe_load(raw_text) or {}

    elif suffix == ".json":
        data = json.loads(raw_text)

    else:
        # fallback: try yaml first, then json
        if has_yaml:
            import yaml
            data = yaml.safe_load(raw_text) or {}
        else:
            data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    _validate_config(data)
    normalize_phase_d_config(data)
    return data


def normalize_phase_d_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize Phase D look_for configs so filters are always populated."""
    phase_d = config.get("phase_d")
    if not isinstance(phase_d, dict):
        return config
    look_fors = phase_d.get("look_fors")
    if not isinstance(look_fors, dict):
        return config

    normalized: dict[str, Any] = {}
    for look_for_name, rule_cfg in look_fors.items():
        if not isinstance(rule_cfg, dict):
            normalized[look_for_name] = rule_cfg
            continue
        filters_cfg = rule_cfg.get("filters")
        filters: dict[str, Any] = deepcopy(filters_cfg) if isinstance(filters_cfg, dict) else {}
        for key in PHASE_D_FILTER_FIELDS:
            if key in rule_cfg and rule_cfg[key] is not None:
                filters[key] = rule_cfg[key]
        rule_copy = deepcopy(rule_cfg)
        rule_copy["filters"] = filters
        normalized[look_for_name] = rule_copy

    phase_d["look_fors"] = normalized
    return config


__all__ = [
    "PHASE_D_FILTER_FIELDS",
    "deep_merge",
    "load_config",
    "normalize_phase_d_config",
]

def _validate_config(config: dict[str, Any]) -> None:
    if "symbol" in config:
        raise ValueError("Config must not define 'symbol'; provide symbol via CLI.")

    event_cfg = config.get("event_scorer", {})
    if event_cfg is None:
        return
    if not isinstance(event_cfg, dict):
        raise ValueError("event_scorer must be a mapping.")

    _validate_float_range(event_cfg, "train_ratio", min_value=0.0, max_value=1.0, strict_min=True)
    _validate_positive_int(event_cfg, "k_bars")
    _validate_tf_int_map(event_cfg, "k_bars_by_tf")
    _validate_positive_float(event_cfg, "reward_r")
    _validate_positive_float(event_cfg, "sl_mult")
    _validate_float_range(event_cfg, "r_thr", min_value=-10.0, max_value=10.0)
    _validate_string(event_cfg, "score_tf")
    _validate_string(event_cfg, "allow_tf")

    tie_break = event_cfg.get("tie_break")
    if tie_break is not None and tie_break not in {"distance", "worst"}:
        raise ValueError("tie_break must be 'distance' or 'worst'.")

    include_transition = event_cfg.get("include_transition")
    if include_transition is not None and not isinstance(include_transition, bool):
        raise ValueError("include_transition must be boolean.")

    meta_policy = event_cfg.get("meta_policy", {})
    if meta_policy is not None:
        if not isinstance(meta_policy, dict):
            raise ValueError("meta_policy must be a mapping.")
        enabled = meta_policy.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("meta_policy.enabled must be boolean.")
        _validate_float_range(meta_policy, "meta_margin_min", min_value=0.0, max_value=1.0)
        _validate_float_range(meta_policy, "meta_margin_max", min_value=0.0, max_value=1.0)
        if (
            meta_policy.get("meta_margin_min") is not None
            and meta_policy.get("meta_margin_max") is not None
            and meta_policy["meta_margin_min"] >= meta_policy["meta_margin_max"]
        ):
            raise ValueError("meta_margin_min must be < meta_margin_max.")

    family_training = event_cfg.get("family_training", {})
    if family_training is not None:
        if not isinstance(family_training, dict):
            raise ValueError("family_training must be a mapping.")
        _validate_positive_int(family_training, "min_samples_train")

    thresholds = event_cfg.get("decision_thresholds", {})
    _validate_decision_thresholds(thresholds)

    modes = event_cfg.get("modes", {})
    if modes is not None:
        if not isinstance(modes, dict):
            raise ValueError("modes must be a mapping.")
        for mode_name, mode_cfg in modes.items():
            if not isinstance(mode_cfg, dict):
                raise ValueError(f"mode '{mode_name}' must be a mapping.")
            _validate_decision_thresholds(mode_cfg.get("decision_thresholds", {}))

    research_cfg = event_cfg.get("research", {})
    if research_cfg is not None:
        if not isinstance(research_cfg, dict):
            raise ValueError("research must be a mapping.")
        enabled = research_cfg.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("research.enabled must be boolean.")
        if "d1_anchor_hour" in research_cfg:
            d1_anchor_hour = research_cfg.get("d1_anchor_hour")
            if (
                not isinstance(d1_anchor_hour, int)
                or isinstance(d1_anchor_hour, bool)
                or not 0 <= d1_anchor_hour <= 23
            ):
                raise ValueError("research.d1_anchor_hour must be an integer in range [0, 23]")
        features = research_cfg.get("features", {})
        if features is not None:
            if not isinstance(features, dict):
                raise ValueError("research.features must be a mapping.")
            _validate_bool(features, "session_bucket")
            _validate_bool(features, "hour_bucket")
            _validate_bool(features, "trend_context_D1")
            _validate_bool(features, "vol_context")
        k_bars_grid = research_cfg.get("k_bars_grid")
        if k_bars_grid is not None:
            if not isinstance(k_bars_grid, list):
                raise ValueError("research.k_bars_grid must be a list.")
            for value in k_bars_grid:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("research.k_bars_grid must contain positive integers.")
        _validate_tf_int_list_map(research_cfg, "k_bars_grid_by_tf")
        diagnostics = research_cfg.get("diagnostics", {})
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                raise ValueError("research.diagnostics must be a mapping.")
            _validate_float_range(diagnostics, "max_family_concentration", min_value=0.0, max_value=1.0)
            _validate_float_range(diagnostics, "min_temporal_dispersion", min_value=0.0, max_value=1.0)
            _validate_float_range(diagnostics, "max_score_tail_slope", min_value=0.0)
            _validate_positive_int(diagnostics, "min_calib_samples")


def _validate_float_range(
    data: dict[str, Any],
    key: str,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> None:
    if key not in data:
        return
    value = data[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    numeric_value = float(value)
    if min_value is not None:
        if strict_min and numeric_value <= min_value:
            raise ValueError(f"{key} must be > {min_value}.")
        if not strict_min and numeric_value < min_value:
            raise ValueError(f"{key} must be >= {min_value}.")
    if max_value is not None and numeric_value > max_value:
        raise ValueError(f"{key} must be <= {max_value}.")


def _validate_positive_float(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    value = data[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    if float(value) <= 0:
        raise ValueError(f"{key} must be > 0.")


def _validate_positive_int(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    value = data[key]
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value <= 0:
        raise ValueError(f"{key} must be > 0.")


def _validate_string(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")


def _validate_tf_int_map(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    mapping = data[key]
    if not isinstance(mapping, dict):
        raise ValueError(f"{key} must be a mapping.")
    for tf_key, value in mapping.items():
        if not isinstance(tf_key, str) or not tf_key.strip():
            raise ValueError(f"{key} keys must be non-empty strings.")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} values must be positive integers.")


def _validate_tf_int_list_map(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    mapping = data[key]
    if not isinstance(mapping, dict):
        raise ValueError(f"{key} must be a mapping.")
    for tf_key, values in mapping.items():
        if not isinstance(tf_key, str) or not tf_key.strip():
            raise ValueError(f"{key} keys must be non-empty strings.")
        if not isinstance(values, list):
            raise ValueError(f"{key} values must be lists of positive integers.")
        for value in values:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} values must be lists of positive integers.")


def _validate_decision_thresholds(thresholds: Any) -> None:
    if thresholds is None:
        return
    if not isinstance(thresholds, dict):
        raise ValueError("decision_thresholds must be a mapping.")
    _validate_positive_int(thresholds, "n_min")
    _validate_float_range(thresholds, "winrate_min", min_value=0.0, max_value=1.0)
    _validate_float_range(thresholds, "r_mean_min")
    if "p10_min" in thresholds and thresholds["p10_min"] is not None:
        _validate_float_range(thresholds, "p10_min")


def _validate_bool(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean.")
