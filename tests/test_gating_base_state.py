import pandas as pd
import pytest

from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels


def test_gating_requires_base_state_for_custom_allow() -> None:
    outputs = pd.DataFrame(
        {
            "state_hat": [StateLabels.BALANCE, StateLabels.TREND],
            "margin": [0.2, 0.3],
        }
    )
    config_meta = {
        "phase_d": {
            "enabled": True,
            "look_fors": {
                "LOOK_FOR_custom_context": {
                    "enabled": True,
                    "filters": {"sessions_in": ["LONDON"]},
                }
            },
        },
    }

    policy = GatingPolicy()
    with pytest.raises(ValueError, match="base_state"):
        policy.apply(outputs, config_meta=config_meta)
