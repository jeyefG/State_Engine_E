import pandas as pd

from state_engine.session import get_session_bucket


def test_fx_bucket_utc_naive() -> None:
    ts = pd.Timestamp("2025-01-01 05:30")
    assert get_session_bucket(ts, "EURUSD") == "ASIA"


def test_us500_bucket_ny_time_from_utc() -> None:
    ts = pd.Timestamp("2025-01-02 14:00", tz="UTC")
    assert get_session_bucket(ts, "US500") == "PRE_MARKET"


def test_metals_bucket_ny_time_from_utc() -> None:
    ts = pd.Timestamp("2025-01-02 22:30", tz="UTC")
    assert get_session_bucket(ts, "XAUUSD") == "ROLLOVER"


def test_unknown_bucket_fallback() -> None:
    assert get_session_bucket(pd.NaT, "EURUSD") == "UNKNOWN"
