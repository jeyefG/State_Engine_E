from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pd = pytest.importorskip("pandas")

from state_engine.labels import StateLabels
from state_engine.quality import (
    QualityConfig,
    QualityScoring,
    QualityThresholds,
    QualityWindows,
    assign_quality_labels,
)


def test_quality_assigner_labels_expected_states() -> None:
    config = QualityConfig(
        thresholds=QualityThresholds(
            er_low=0.2,
            er_mid=0.4,
            er_high=0.7,
            netmove_low=0.2,
            netmove_mid=0.5,
            netmove_high=0.9,
            break_low=0.2,
            break_mid=0.5,
            break_high=0.8,
            reentry_min=2.0,
            reentry_mid=3.0,
            reentry_high=4.0,
            reentry_low=1.0,
            reentry_max_for_stable=3.0,
            inside_mid=0.4,
            inside_high=0.6,
            range_slope_min=-0.01,
            swing_mid=2.0,
            swing_high=3.0,
            close_mid_low=0.4,
            close_mid_high=0.6,
        ),
        windows=QualityWindows(range_slope_k=3, transition_failed_lookback=3),
        scoring=QualityScoring(score_min=0.7, score_margin=0.1),
    )

    features = pd.DataFrame(
        {
            "ER": [0.1, 0.9, 0.3],
            "NetMove": [0.1, 1.0, 0.4],
            "Range_W": [1.0, 1.0, 1.0],
            "BreakMag": [0.1, 0.9, 0.3],
            "ReentryCount": [3.0, 0.5, 5.0],
            "InsideBarsRatio": [0.2, 0.2, 0.5],
            "SwingHighCount": [1.0, 1.0, 2.0],
            "SwingLowCount": [1.0, 1.0, 2.0],
            "CloseLocation": [0.5, 0.8, 0.5],
        }
    )
    states = pd.Series(
        [StateLabels.BALANCE, StateLabels.TREND, StateLabels.TRANSITION],
        index=features.index,
    )

    labels, warnings = assign_quality_labels(states, features, config)

    assert warnings == []
    assert labels.iloc[0] == "BALANCE_STABLE"
    assert labels.iloc[1] == "TREND_STRONG"
    assert labels.iloc[2] == "TRANSITION_NOISY"
