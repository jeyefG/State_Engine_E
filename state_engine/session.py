"""Session bucket utilities for M5 contextual features."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

US_INDEX_SYMBOLS = {"US500", "NAS100"}
METAL_SYMBOLS = {"XAUUSD", "XAGUSD"}

FX_BUCKETS = ("ASIA", "LONDON_AM", "LONDON_NY", "NY_AM", "NY_PM", "ROLLOVER")
US_INDEX_BUCKETS = ("PRE_MARKET", "CASH_OPEN", "MID_DAY", "POWER_HOUR", "AFTER_HOURS")
METAL_BUCKETS = ("ASIA", "LONDON_OPEN", "NY_OPEN", "NY_MID", "NY_CLOSE", "ROLLOVER")
SESSION_BUCKETS = (
    "ASIA",
    "LONDON_AM",
    "LONDON_NY",
    "NY_AM",
    "NY_PM",
    "ROLLOVER",
    "PRE_MARKET",
    "CASH_OPEN",
    "MID_DAY",
    "POWER_HOUR",
    "AFTER_HOURS",
    "LONDON_OPEN",
    "NY_OPEN",
    "NY_MID",
    "NY_CLOSE",
    "UNKNOWN",
)


def _normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    return "".join(ch for ch in str(symbol).upper() if ch.isalnum())


def _symbol_matches(symbol: str | None, candidates: set[str]) -> bool:
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return False
    return any(normalized.startswith(candidate) for candidate in candidates)


def _localize_timestamp(ts: pd.Timestamp, tz_name: str) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(tz_name))


def _minutes_since_midnight(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _in_window(
    minutes: int,
    start: int,
    end: int,
    *,
    inclusive_end: bool = True,
) -> bool:
    if start <= end:
        return start <= minutes <= end if inclusive_end else start <= minutes < end
    if inclusive_end:
        return minutes >= start or minutes <= end
    return minutes >= start or minutes < end


def _bucket_fx(minutes: int) -> str:
    if _in_window(minutes, 0, 6 * 60 + 59):
        return "ASIA"
    if _in_window(minutes, 7 * 60, 10 * 60 + 59):
        return "LONDON_AM"
    if _in_window(minutes, 11 * 60, 13 * 60 + 59):
        return "LONDON_NY"
    if _in_window(minutes, 14 * 60, 16 * 60 + 59):
        return "NY_AM"
    if _in_window(minutes, 17 * 60, 20 * 60 + 59):
        return "NY_PM"
    if _in_window(minutes, 21 * 60, 23 * 60 + 59):
        return "ROLLOVER"
    return "UNKNOWN"


def _bucket_us_index(minutes: int) -> str:
    if _in_window(minutes, 8 * 60, 9 * 60 + 30, inclusive_end=False):
        return "PRE_MARKET"
    if _in_window(minutes, 9 * 60 + 30, 10 * 60 + 30, inclusive_end=False):
        return "CASH_OPEN"
    if _in_window(minutes, 10 * 60 + 30, 14 * 60 + 30, inclusive_end=False):
        return "MID_DAY"
    if _in_window(minutes, 14 * 60 + 30, 15 * 60 + 30, inclusive_end=False):
        return "POWER_HOUR"
    return "AFTER_HOURS"


def _bucket_metals(minutes: int) -> str:
    if _in_window(minutes, 19 * 60, 2 * 60 + 59):
        return "ASIA"
    if _in_window(minutes, 3 * 60, 8 * 60 + 29):
        return "LONDON_OPEN"
    if _in_window(minutes, 8 * 60 + 30, 10 * 60 + 59):
        return "NY_OPEN"
    if _in_window(minutes, 11 * 60, 13 * 60 + 59):
        return "NY_MID"
    if _in_window(minutes, 14 * 60, 16 * 60 + 59):
        return "NY_CLOSE"
    if _in_window(minutes, 17 * 60, 18 * 60 + 59):
        return "ROLLOVER"
    return "UNKNOWN"


def get_session_bucket(ts: pd.Timestamp, symbol: str | None) -> str:
    """Return session bucket based on timestamp and symbol mapping."""
    try:
        timestamp = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if pd.isna(timestamp):
        return "UNKNOWN"

    if _symbol_matches(symbol, US_INDEX_SYMBOLS):
        local_ts = _localize_timestamp(timestamp, "America/New_York")
        return _bucket_us_index(_minutes_since_midnight(local_ts))
    if _symbol_matches(symbol, METAL_SYMBOLS):
        local_ts = _localize_timestamp(timestamp, "America/New_York")
        return _bucket_metals(_minutes_since_midnight(local_ts))

    local_ts = _localize_timestamp(timestamp, "UTC")
    return _bucket_fx(_minutes_since_midnight(local_ts))


__all__ = [
    "FX_BUCKETS",
    "US_INDEX_BUCKETS",
    "METAL_BUCKETS",
    "SESSION_BUCKETS",
    "get_session_bucket",
]
