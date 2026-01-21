import json
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_LOADER_PATH = PROJECT_ROOT / "state_engine" / "config_loader.py"
spec = importlib.util.spec_from_file_location("config_loader", CONFIG_LOADER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load config_loader module spec.")
config_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_loader)

import pytest

deep_merge = config_loader.deep_merge
load_config = config_loader.load_config


def test_deep_merge_overrides_nested_values() -> None:
    defaults = {
        "event_scorer": {
            "train_ratio": 0.8,
            "meta_policy": {"enabled": True, "meta_margin_min": 0.1, "meta_margin_max": 0.95},
        },
    }
    overrides = {
        "event_scorer": {"meta_policy": {"meta_margin_min": 0.2}},
    }

    merged = deep_merge(defaults, overrides)

    assert merged["event_scorer"]["train_ratio"] == 0.8
    assert merged["event_scorer"]["meta_policy"]["meta_margin_min"] == 0.2
    assert merged["event_scorer"]["meta_policy"]["meta_margin_max"] == 0.95


def test_load_config_rejects_symbol_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"symbol": "XAUUSD", "event_scorer": {"train_ratio": 0.8}}))

    with pytest.raises(ValueError, match="symbol"):
        load_config(config_path)


def test_load_config_validates_meta_margin_bounds(tmp_path: Path) -> None:
    config = {
        "event_scorer": {"meta_policy": {"meta_margin_min": 0.9, "meta_margin_max": 0.5}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="meta_margin_min"):
        load_config(config_path)


def test_load_config_allows_null_p10_min(tmp_path: Path) -> None:
    config = {
        "event_scorer": {
            "decision_thresholds": {"n_min": 100, "winrate_min": 0.5, "r_mean_min": 0.01, "p10_min": None}
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loaded = load_config(config_path)

    assert loaded["event_scorer"]["decision_thresholds"]["p10_min"] is None


def test_load_config_allows_research_block(tmp_path: Path) -> None:
    config = {
        "event_scorer": {
            "research": {
                "enabled": True,
                "features": {"session_bucket": True, "hour_bucket": True, "trend_context_D1": False},
                "k_bars_grid": [12, 24],
                "diagnostics": {"max_family_concentration": 0.6, "min_temporal_dispersion": 0.3},
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loaded = load_config(config_path)

    assert loaded["event_scorer"]["research"]["enabled"] is True


@pytest.mark.parametrize("anchor_hour", [0, 23])
def test_load_config_accepts_valid_d1_anchor_hour(tmp_path: Path, anchor_hour: int) -> None:
    config = {
        "event_scorer": {"research": {"d1_anchor_hour": anchor_hour}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loaded = load_config(config_path)

    assert loaded["event_scorer"]["research"]["d1_anchor_hour"] == anchor_hour


@pytest.mark.parametrize("anchor_hour", [-1, 24, "3", None])
def test_load_config_rejects_invalid_d1_anchor_hour(tmp_path: Path, anchor_hour: object) -> None:
    config = {
        "event_scorer": {"research": {"d1_anchor_hour": anchor_hour}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="research.d1_anchor_hour must be an integer in range \\[0, 23\\]"):
        load_config(config_path)
