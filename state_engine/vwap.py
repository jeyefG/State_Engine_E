"""VWAP utilities aligned with MT5 (daily reset, typical price, volume fallback)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)


def _resolve_time_series(
    df: pd.DataFrame,
    *,
    time_col: str | None,
    tz_server: str | None,
) -> pd.Series:
    if time_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("VWAP requires datetime index or time column.")
        ts = pd.Series(df.index, index=df.index)
    else:
        ts = pd.to_datetime(df[time_col])
    if tz_server and ts.dt.tz is not None:
        ts = ts.dt.tz_convert(tz_server)
    return ts


def _resolve_volume_columns(
    df: pd.DataFrame,
    *,
    real_vol_col: str,
    tick_vol_col: str,
) -> tuple[str | None, str | None]:
    real_col = real_vol_col if real_vol_col in df.columns else None
    tick_col = tick_vol_col if tick_vol_col in df.columns else None
    if real_col is None and "real_volume" in df.columns:
        real_col = "real_volume"
    if tick_col is None and "tick_volume" in df.columns:
        tick_col = "tick_volume"
    return real_col, tick_col


def compute_vwap_mt5_daily(
    df: pd.DataFrame,
    *,
    time_col: str | None = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    real_vol_col: str = "volume",
    tick_vol_col: str = "tick_volume",
    tz_server: str | None = None,
) -> pd.Series:
    """Compute MT5-style daily VWAP (reset at server midnight)."""
    required = {high_col, low_col, close_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns for VWAP: {sorted(missing)}")

    ts = _resolve_time_series(df, time_col=time_col, tz_server=tz_server)
    real_col, tick_col = _resolve_volume_columns(
        df,
        real_vol_col=real_vol_col,
        tick_vol_col=tick_vol_col,
    )
    metadata: dict[str, object] = {
        "vwap_mode": "mt5_daily",
        "vwap_reset_mode_effective": "mt5_daily",
        "vwap_price_columns": (high_col, low_col, close_col),
        "vwap_real_volume_col": real_col,
        "vwap_tick_volume_col": tick_col,
        "vwap_time_col": time_col or "index",
        "vwap_tz_server": tz_server,
        "vwap_volume_source": None,
        "vwap_invalid_reason": None,
    }

    if real_col is None and tick_col is None:
        _LOGGER.warning("VWAP mt5_daily missing both volume columns: %s/%s", real_vol_col, tick_vol_col)
        metadata["vwap_invalid_reason"] = "missing_volume_column"
        vwap = pd.Series(np.nan, index=df.index, name="vwap")
        vwap.attrs.update(metadata)
        return vwap

    real = pd.to_numeric(df[real_col], errors="coerce") if real_col else None
    tick = pd.to_numeric(df[tick_col], errors="coerce") if tick_col else None

    if real is None:
        volume = tick
        metadata["vwap_volume_source"] = tick_col
    elif tick is None:
        volume = real
        metadata["vwap_volume_source"] = real_col
    else:
        volume = pd.Series(np.where(real > 0, real, tick), index=df.index)
        metadata["vwap_volume_source"] = f"{real_col}>0_else_{tick_col}"

    volume = pd.to_numeric(volume, errors="coerce").fillna(0.0).astype(float)
    if (volume > 0).sum() == 0:
        metadata["vwap_invalid_reason"] = "no_positive_volume"
        vwap = pd.Series(np.nan, index=df.index, name="vwap")
        vwap.attrs.update(metadata)
        return vwap

    order = np.arange(len(df))
    sorted_df = df.assign(_ts=ts.to_numpy(), _order=order).sort_values(
        ["_ts", "_order"], kind="mergesort"
    )
    sorted_ts = pd.to_datetime(sorted_df["_ts"])
    day_id = sorted_ts.dt.normalize().to_numpy()
    high = pd.to_numeric(sorted_df[high_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(sorted_df[low_col], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(sorted_df[close_col], errors="coerce").to_numpy(dtype=float)
    vol_sorted = volume.reindex(sorted_df.index).to_numpy(dtype=float)

    typical = (high + low + close) / 3.0
    vwap_values = np.full(len(sorted_df), np.nan, dtype=float)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = None
    for i, day in enumerate(day_id):
        if prev_day is None or day != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day
        vol_i = vol_sorted[i]
        if not np.isfinite(vol_i) or vol_i <= 0:
            vol_i = 0.0
        tp_i = typical[i]
        if not np.isfinite(tp_i):
            tp_i = 0.0
            vol_i = 0.0
        cum_pv += tp_i * vol_i
        cum_vol += vol_i
        vwap_values[i] = cum_pv / cum_vol if cum_vol > 0 else np.nan

    sorted_df = sorted_df.assign(_vwap=vwap_values)
    sorted_back = sorted_df.sort_values("_order", kind="mergesort")
    vwap = pd.Series(sorted_back["_vwap"].to_numpy(), index=df.index, name="vwap")
    vwap.attrs.update(metadata)
    return vwap


def mt5_day_id(
    df: pd.DataFrame,
    *,
    time_col: str | None = None,
    tz_server: str | None = None,
) -> pd.Series:
    ts = _resolve_time_series(df, time_col=time_col, tz_server=tz_server)
    return pd.to_datetime(ts).dt.normalize()


__all__ = ["compute_vwap_mt5_daily", "mt5_day_id"]
