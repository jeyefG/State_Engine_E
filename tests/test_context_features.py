import pandas as pd

from state_engine.context_features import build_context_features


def test_context_features_non_xau_has_required_columns() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="H")
    ohlcv = pd.DataFrame(
        {
            "high": [1.1, 1.2, 1.15, 1.18],
            "low": [1.0, 1.05, 1.07, 1.1],
            "close": [1.05, 1.1, 1.12, 1.15],
        },
        index=index,
    )
    outputs = pd.DataFrame({"state_hat": [0, 0, 1, 1]}, index=index)

    ctx = build_context_features(
        ohlcv,
        outputs,
        symbol="EURUSD",
        timeframe="H2",
    )

    assert "ctx_session_bucket" in ctx.columns
    assert "ctx_state_age" in ctx.columns
    assert "ctx_dist_vwap_atr" in ctx.columns
    assert ctx["ctx_dist_vwap_atr"].isna().all()
    assert "ctx_vwap_source" in ctx.columns
    assert "ctx_vwap_anchor" in ctx.columns
    assert (ctx["ctx_vwap_source"] == "missing").all()
    assert (ctx["ctx_vwap_anchor"] == "missing").all()
