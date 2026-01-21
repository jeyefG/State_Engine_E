import argparse

import numpy as np
import pandas as pd
import pandas.testing as pdt

from scripts.train_event_scorer import _build_training_diagnostic_report, _make_margin_bin_rank


def _thresholds() -> argparse.Namespace:
    return argparse.Namespace(
        decision_n_min=10,
        decision_r_mean_min=0.0,
        decision_winrate_min=0.5,
        decision_p10_min=-1.0,
    )


def test_diagnostic_report_determinism_seeded() -> None:
    rng = np.random.default_rng(123)
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    events_diag = pd.DataFrame(
        {
            "state_hat_H1": rng.choice(["BALANCE", "TREND", "TRANSITION"], size=n),
            "allow_id": rng.choice(
                ["LOOK_FOR_balance_fade", "LOOK_FOR_trend_pullback", "LOOK_FOR_transition_failure"], size=n
            ),
            "margin": rng.uniform(0.05, 0.95, size=n),
            "r": rng.normal(0.0, 0.5, size=n),
            "win": rng.integers(0, 2, size=n),
        },
        index=idx,
    )
    events_diag["margin_bin"] = _make_margin_bin_rank(events_diag["margin"])
    df_m5_ctx = pd.DataFrame({"atr_14": np.full(n, 1.0)}, index=idx)

    report_a = _build_training_diagnostic_report(
        events_diag,
        df_m5_ctx,
        thresholds=_thresholds(),
        state_col="state_hat_H1",
        phase_e=False,
    )
    report_b = _build_training_diagnostic_report(
        events_diag,
        df_m5_ctx,
        thresholds=_thresholds(),
        state_col="state_hat_H1",
        phase_e=False,
    )

    for key in report_a:
        pdt.assert_frame_equal(report_a[key], report_b[key])


def test_causality_merge_asof_shift_unchanged() -> None:
    h1_idx = pd.date_range("2025-01-01", periods=10, freq="h")
    m5_idx = pd.date_range("2025-01-01", periods=120, freq="5min")
    df_h1 = pd.DataFrame(
        {
            "state_hat": np.arange(len(h1_idx)),
            "margin": np.linspace(0.1, 0.9, len(h1_idx)),
            "LOOK_FOR_trend_pullback": 1,
        },
        index=h1_idx,
    )
    df_m5 = pd.DataFrame(
        {
            "open": np.linspace(100, 110, len(m5_idx)),
            "high": np.linspace(100.5, 110.5, len(m5_idx)),
            "low": np.linspace(99.5, 109.5, len(m5_idx)),
            "close": np.linspace(100, 110, len(m5_idx)),
        },
        index=m5_idx,
    )
    h1_ctx = df_h1.shift(1)
    merged = pd.merge_asof(
        df_m5.reset_index().rename(columns={"index": "time"}),
        h1_ctx.reset_index().rename(columns={"index": "time"}),
        on="time",
        direction="backward",
    ).set_index("time")
    merged["ret_1"] = merged["close"].pct_change().shift(1)
    cutoff = merged.index[len(merged) // 2]

    mutated = merged.copy()
    mutated.loc[mutated.index > cutoff, "close"] *= 1.5
    mutated["ret_1"] = mutated["close"].pct_change().shift(1)

    before_mask = merged.index <= cutoff
    pdt.assert_series_equal(
        merged.loc[before_mask, "ret_1"],
        mutated.loc[before_mask, "ret_1"],
    )


def test_coverage_sanity_index_match_and_atr() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="5min")
    df_m5_ctx = pd.DataFrame({"atr_14": np.ones(len(idx))}, index=idx)
    events_idx = idx[:10]
    events_diag = pd.DataFrame(
        {
            "state_hat_H1": ["BALANCE"] * len(events_idx),
            "allow_id": ["LOOK_FOR_balance_fade"] * len(events_idx),
            "margin": np.linspace(0.1, 0.9, len(events_idx)),
            "r": np.linspace(-0.2, 0.3, len(events_idx)),
            "win": np.ones(len(events_idx)),
        },
        index=events_idx,
    )
    events_diag["margin_bin"] = _make_margin_bin_rank(events_diag["margin"])

    report = _build_training_diagnostic_report(
        events_diag,
        df_m5_ctx,
        thresholds=_thresholds(),
        state_col="state_hat_H1",
        phase_e=False,
    )
    coverage_global = report["coverage_global"]
    assert float(coverage_global.loc[0, "events_index_match_pct"]) == 1.0
    assert not df_m5_ctx.loc[events_diag.index, "atr_14"].isna().any()
