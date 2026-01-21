import pandas as pd

from scripts.train_state_engine import (
    _build_look_for_coverage_table,
    _build_confidence_summary,
    _build_context_features_audit,
)
from state_engine.labels import StateLabels


def test_phase_d_outputs_do_not_include_outcome_columns() -> None:
    outputs = pd.DataFrame(
        {
            "state_hat": [StateLabels.BALANCE, StateLabels.TREND, StateLabels.TRANSITION],
            "margin": [0.2, 0.3, 0.1],
        }
    )
    gating = pd.DataFrame(
        {
            "LOOK_FOR_balance_fade": [1, 0, 0],
            "LOOK_FOR_trend_pullback": [0, 1, 0],
        }
    )
    ctx_features = pd.DataFrame(
        {
            "ctx_session_bucket": ["ASIA", "LONDON", "NY"],
            "ctx_state_age": [1, 2, 3],
            "ctx_dist_vwap_atr": [0.5, 1.2, 0.8],
        }
    )

    allow_coverage = _build_look_for_coverage_table(gating, outputs)
    context_audit = _build_context_features_audit(ctx_features)
    confidence_summary = _build_confidence_summary(outputs)

    forbidden = {"ret_struct", "future_high", "future_low"}
    for df in (allow_coverage, context_audit, confidence_summary):
        assert forbidden.isdisjoint(set(df.columns))


def test_allow_coverage_table_has_counts_only() -> None:
    outputs = pd.DataFrame(
        {
            "state_hat": [StateLabels.BALANCE, StateLabels.BALANCE, StateLabels.TREND],
            "quality_label": ["ok", "ok", "warn"],
            "margin": [0.2, 0.1, 0.3],
        }
    )
    gating = pd.DataFrame(
        {
            "LOOK_FOR_balance_fade": [1, 1, 0],
            "LOOK_FOR_trend_pullback": [0, 0, 1],
        }
    )

    coverage = _build_look_for_coverage_table(gating, outputs)
    assert set(coverage.columns) == {
        "look_for_rule",
        "n",
        "pct_total",
        "state_counts",
        "quality_counts",
    }
