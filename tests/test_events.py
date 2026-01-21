import pandas as pd

from state_engine.events import EventDetectionConfig, EventExtractor, EventFamily, EventType
from state_engine.vwap import compute_vwap_mt5_daily


def _make_m5_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="5min")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [101, 102, 101, 101, 101, 101],
            "low": [99, 98, 99, 99, 99, 99],
            "close": [100, 101, 100, 100, 100, 100],
            "vwap": [100, 100, 100, 100, 100, 100],
            "volume": [10, 12, 11, 13, 10, 9],
        },
        index=idx,
    )


def test_events_schema_has_required_columns() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")

    required = {
        "symbol",
        "ts",
        "event_type",
        "family_id",
        "event_id",
        "dist_to_vwap",
        "abs_dist_to_vwap",
        "vwap_slope",
        "range_1",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "atr_14",
        "dist_to_vwap_atr",
        "volume_rel",
        "is_touch_exact",
        "is_rejection",
        "hour",
        "minute",
    }
    assert required.issubset(events.columns)


def test_events_no_leakage_features_ignore_future_changes() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")
    event_ts = events["ts"].iloc[0]
    baseline = events.loc[events["ts"] == event_ts, "dist_to_vwap"].iloc[0]

    df_future = df_m5.copy()
    df_future.loc[df_future.index[-1], "close"] = 999
    df_future.loc[df_future.index[-1], "vwap"] = 999

    events_future = extractor.extract(df_future, symbol="TEST")
    updated = events_future.loc[events_future["ts"] == event_ts, "dist_to_vwap"].iloc[0]

    assert baseline == updated


def test_detects_near_vwap_event() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")

    assert (events["event_type"] == EventType.NEAR_VWAP.value).any()


def test_events_have_atr_14_after_warmup() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="5min")
    df_m5 = pd.DataFrame(
        {
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.0] * 30,
            "vwap": [100.0] * 30,
            "volume": [10.0] * 30,
        },
        index=idx,
    )

    extractor = EventExtractor()
    events = extractor.extract(df_m5, symbol="TEST")
    assert "atr_14" in events.columns

    warm_idx = idx[14]
    warm_events = events.loc[events["ts"] >= warm_idx]
    assert not warm_events.empty
    assert warm_events["atr_14"].notna().all()


def test_event_timestamp_matches_m5_index() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")
    assert not events.empty
    assert events.index[0] in df_m5.index


def test_vwap_daily_reset() -> None:
    idx = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-02 00:00", "2024-01-02 00:05"]
    )
    df_m5 = pd.DataFrame(
        {
            "open": [100, 100, 200, 200],
            "high": [101, 101, 201, 201],
            "low": [99, 99, 199, 199],
            "close": [100, 100, 200, 200],
            "tick_volume": [10, 10, 10, 10],
        },
        index=idx,
    )
    vwap = compute_vwap_mt5_daily(df_m5)
    assert vwap.iloc[2] == 200


def test_family_id_from_allow() -> None:
    df_m5 = _make_m5_df()
    df_m5["LOOK_FOR_balance_fade"] = 1
    df_m5["state_hat_H1"] = 0
    df_m5["margin_H1"] = 0.2
    config = EventDetectionConfig(near_vwap_mode="continuous")
    extractor = EventExtractor(config=config)

    events = extractor.extract(df_m5, symbol="TEST")

    assert (events["family_id"] == EventFamily.BALANCE_FADE.value).any()


def test_detect_touch_and_rejection_increases_supply() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    df_m5 = pd.DataFrame(
        {
            "open": [99.0] * 10,
            "high": [101.0] * 10,
            "low": [98.0] * 10,
            "close": [101.0] * 10,
            "vwap": [100.0] * 10,
            "volume": [10.0] * 10,
        },
        index=idx,
    )
    config_near_only = EventDetectionConfig(
        near_vwap_mode="enter",
        near_vwap_cooldown_bars=0,
        touch_mode="off",
        rejection_mode="off",
    )
    config_touch_reject = EventDetectionConfig(
        near_vwap_mode="enter",
        near_vwap_cooldown_bars=0,
        touch_mode="on",
        rejection_mode="on",
    )
    events_near_only = EventExtractor(config=config_near_only).extract(df_m5, symbol="TEST")
    events_touch_reject = EventExtractor(config=config_touch_reject).extract(df_m5, symbol="TEST")

    assert len(events_touch_reject) > len(events_near_only)
