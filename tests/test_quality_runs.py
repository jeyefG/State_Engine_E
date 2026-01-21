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
    assign_quality_labels,
    build_quality_diagnostics,
    format_quality_diagnostics,
)


def test_quality_assigner_missing_features_unclassified() -> None:
    config = QualityConfig()
    features = pd.DataFrame(
        {
            "ER": [0.1],
            "NetMove": [0.1],
            "Range_W": [1.0],
            "ReentryCount": [2.0],
            "InsideBarsRatio": [0.4],
            "SwingHighCount": [1.0],
            "SwingLowCount": [1.0],
            "CloseLocation": [0.5],
        }
    )
    states = pd.Series([StateLabels.BALANCE], index=features.index)

    labels, warnings = assign_quality_labels(states, features, config)

    assert labels.iloc[0] == "UNCLASSIFIED"
    assert any("missing_features" in warning for warning in warnings)


def test_quality_assigner_optional_close_location() -> None:
    config = QualityConfig()
    features = pd.DataFrame(
        {
            "ER": [0.1],
            "NetMove": [0.1],
            "Range_W": [1.0],
            "BreakMag": [0.1],
            "ReentryCount": [3.0],
            "InsideBarsRatio": [0.4],
            "SwingHighCount": [1.0],
            "SwingLowCount": [1.0],
        }
    )
    states = pd.Series([StateLabels.BALANCE], index=features.index)

    labels, warnings = assign_quality_labels(states, features, config)

    assert labels.iloc[0] == "BALANCE_STABLE"
    assert any("missing_optional_features" in warning for warning in warnings)


def test_quality_diagnostics_flags_degenerate_labels() -> None:
    states = pd.Series([StateLabels.BALANCE] * 5)
    labels = pd.Series(["BALANCE_STABLE"] * 5)
    diagnostics = build_quality_diagnostics(states, labels, {"thresholds.er_low": "defaults"}, [])

    assert any("degenerate_label" in warning for warning in diagnostics.warnings)


def test_quality_config_effective_outputs_overrides_only() -> None:
    states = pd.Series([StateLabels.BALANCE])
    labels = pd.Series(["BALANCE_STABLE"])
    diagnostics = build_quality_diagnostics(states, labels, {"thresholds.er_low": "defaults"}, [])
    lines = format_quality_diagnostics(diagnostics)

    assert any("none | defaults" in line for line in lines)

    diagnostics_override = build_quality_diagnostics(
        states,
        labels,
        {"thresholds.er_low": "defaults", "thresholds.er_high": "symbol"},
        [],
    )
    override_lines = format_quality_diagnostics(diagnostics_override)

    assert any("thresholds.er_high | symbol" in line for line in override_lines)
