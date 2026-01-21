import logging

import pandas as pd
import pytest

from scripts.train_event_scorer import (
    _apply_event_allow_gates,
    _force_phase_e_allow_identity,
    _required_allow_by_family,
)
from state_engine.pipeline_phase_d import (
    validate_look_for_columns,
    validate_look_for_context_requirements,
)


def test_look_for_filters_missing_dist_vwap_atr_raises() -> None:
    cfg = {
        "phase_d": {
            "enabled": True,
            "look_fors": {
                "LOOK_FOR_balance_vwap_proximity": {
                    "enabled": True,
                    "base_state": "balance",
                    "filters": {"dist_vwap_atr_min": 0.1},
                }
            },
        },
    }
    available = {"BreakMag", "ReentryCount"}
    with pytest.raises(ValueError, match="dist_vwap_atr"):
        validate_look_for_context_requirements(cfg, available, logger=logging.getLogger("test"))


def test_validate_look_for_columns_missing_required_raises() -> None:
    ctx_df = pd.DataFrame({"LOOK_FOR_balance_fade": [1, 0]})
    with pytest.raises(ValueError, match="Missing LOOK_FOR columns"):
        validate_look_for_columns(
            ctx_df,
            ["LOOK_FOR_transition_failure"],
            logger=logging.getLogger("test"),
        )


def test_validate_look_for_columns_non_binary_or_nan_raises() -> None:
    ctx_df = pd.DataFrame({"LOOK_FOR_balance_fade": [1, float("nan"), 2]})
    with pytest.raises(ValueError, match="LOOK_FOR column LOOK_FOR_balance_fade"):
        validate_look_for_columns(
            ctx_df,
            ["LOOK_FOR_balance_fade"],
            logger=logging.getLogger("test"),
        )


def test_phase_e_allow_identity_matches_required_allow() -> None:
    allow_map = _required_allow_by_family()
    events = pd.DataFrame(
        {
            "family_id": [
                "E_BALANCE_FADE",
                "E_TREND_PULLBACK",
                "E_TRANSITION_FAILURE",
                "E_TREND_CONTINUATION",
            ],
            "LOOK_FOR_balance_fade": [1, 0, 0, 0],
            "LOOK_FOR_trend_pullback": [0, 1, 0, 0],
            "LOOK_FOR_transition_failure": [0, 0, 1, 0],
            "LOOK_FOR_trend_continuation": [0, 0, 0, 1],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="5min"),
    )
    gated = _apply_event_allow_gates(
        events,
        required_allow_by_family=allow_map,
        logger=logging.getLogger("test"),
    )
    enforced = _force_phase_e_allow_identity(
        gated,
        required_allow_by_family=allow_map,
        logger=logging.getLogger("test"),
    )
    assert set(enforced["allow_id"]) == set(allow_map.values())
    assert not enforced["allow_id"].astype(str).str.contains(",", na=False).any()
