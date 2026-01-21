import logging

import pytest

from state_engine.pipeline_phase_d import validate_look_for_context_requirements


def test_validate_look_for_context_requirements_missing_columns_raises() -> None:
    cfg = {
        "phase_d": {
            "enabled": True,
            "look_fors": {
                "LOOK_FOR_transition_failure": {
                    "enabled": True,
                    "base_state": "transition",
                    "filters": {"sessions_in": ["NY"]},
                }
            },
        },
    }
    available = {"BreakMag", "ReentryCount"}
    with pytest.raises(ValueError, match="missing columns"):
        validate_look_for_context_requirements(cfg, available, logger=logging.getLogger("test"))


def test_validate_look_for_context_requirements_ok() -> None:
    cfg = {
        "phase_d": {
            "enabled": True,
            "look_fors": {
                "LOOK_FOR_transition_failure": {
                    "enabled": True,
                    "base_state": "transition",
                    "filters": {
                        "sessions_in": ["NY"],
                        "state_age_max": 2,
                        "dist_vwap_atr_max": 1.5,
                    },
                }
            },
        },
    }
    available = {"ctx_session_bucket", "ctx_state_age", "ctx_dist_vwap_atr", "BreakMag", "ReentryCount"}
    validate_look_for_context_requirements(cfg, available, logger=logging.getLogger("test"))
