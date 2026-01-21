from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("pandas")

from state_engine.quality import validate_quality_config_dict


def test_quality_config_rejects_forbidden_sections() -> None:
    with pytest.raises(ValueError, match="state_engine"):
        validate_quality_config_dict({"state_engine": {"foo": 1}})


def test_quality_config_rejects_unknown_sections() -> None:
    with pytest.raises(ValueError, match="Unknown quality config sections"):
        validate_quality_config_dict({"thresholds": {}, "extra": {"foo": 1}})


def test_quality_config_accepts_numeric_sections() -> None:
    validate_quality_config_dict(
        {
            "thresholds": {"er_low": 0.1, "break_low": 0.2},
            "windows": {"range_slope_k": 5},
            "scoring": {"score_min": 0.7, "score_margin": 0.2},
        }
    )
