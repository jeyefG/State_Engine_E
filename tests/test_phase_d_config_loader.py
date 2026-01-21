import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_LOADER_PATH = PROJECT_ROOT / "state_engine" / "config_loader.py"
spec = importlib.util.spec_from_file_location("config_loader", CONFIG_LOADER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load config_loader module spec.")
config_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_loader)
load_config = config_loader.load_config


def test_phase_d_look_for_filters_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "symbol.yaml"
    config_payload = {
        "phase_d": {
            "enabled": True,
            "look_fors": {
                "LOOK_FOR_example": {
                    "enabled": True,
                    "base_state": "TREND",
                    "require_all": False,
                    "sessions_in": ["LONDON", "NY"],
                    "state_age_min": 2,
                    "state_age_max": 5,
                    "dist_vwap_atr_min": 0.5,
                    "dist_vwap_atr_max": 1.5,
                    "breakmag_min": 0.25,
                    "breakmag_max": 1.0,
                    "reentry_min": 1.0,
                    "reentry_max": 3.0,
                    "margin_min": 0.1,
                    "margin_max": 0.9,
                }
            },
        }
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    loaded = load_config(config_path)
    filters = loaded["phase_d"]["look_fors"]["LOOK_FOR_example"]["filters"]

    assert filters
    assert filters["sessions_in"] == ["LONDON", "NY"]
    assert filters["state_age_min"] == 2
    assert filters["state_age_max"] == 5
    assert filters["dist_vwap_atr_min"] == 0.5
    assert filters["dist_vwap_atr_max"] == 1.5
    assert filters["breakmag_min"] == 0.25
    assert filters["breakmag_max"] == 1.0
    assert filters["reentry_min"] == 1.0
    assert filters["reentry_max"] == 3.0
    assert filters["margin_min"] == 0.1
    assert filters["margin_max"] == 0.9
