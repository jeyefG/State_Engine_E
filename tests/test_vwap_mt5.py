import numpy as np
import pandas as pd

from state_engine.vwap import compute_vwap_mt5_daily


def test_mt5_vwap_daily_reset() -> None:
    idx = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-02 00:00", "2024-01-02 00:05"]
    )
    df = pd.DataFrame(
        {
            "high": [101, 101, 201, 201],
            "low": [99, 99, 199, 199],
            "close": [100, 100, 200, 200],
            "tick_volume": [10, 10, 10, 10],
        },
        index=idx,
    )
    vwap = compute_vwap_mt5_daily(df)
    assert vwap.iloc[2] == 200


def test_mt5_vwap_uses_tick_volume_when_real_zero() -> None:
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "real_volume": [0, 0, 0],
            "tick_volume": [10, 20, 30],
        },
        index=idx,
    )
    vwap = compute_vwap_mt5_daily(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    expected = (typical * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()
    assert np.allclose(vwap.to_numpy(), expected.to_numpy(), equal_nan=True)
