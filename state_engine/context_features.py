"""Contextual pseudo-features for H2 gating filters."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd


_SESSION_BUCKETS = ("ASIA", "LONDON", "NY", "NY_PM")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextFeatureConfig:
    """Configuration for contextual pseudo-features."""

    max_state_age: int = 12
    atr_window: int = 14
    vwap_window: int = 50
    vwap_reset_mode: str | None = None
    vwap_day_anchor: str | None = None
    eps: float = 1e-9


def _normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    return "".join(ch for ch in str(symbol).upper() if ch.isalnum())


def _normalize_timeframe(timeframe: str | None) -> str:
    if timeframe is None:
        return ""
    return str(timeframe).upper()


def _resolve_vwap_day_anchor(symbol: str | None, anchor: str | None) -> str:
    if anchor:
        return anchor
    normalized_symbol = _normalize_symbol(symbol)
    if normalized_symbol.startswith("XAUUSD"):
        return "ny_rollover_17"
    return "server_midnight"


def is_context_enabled(symbol: str | None, timeframe: str | None) -> bool:
    """Return True when context features are enabled."""
    _normalize_symbol(symbol)
    _normalize_timeframe(timeframe)
    return True


def _hour_bucket(hour: int) -> str:
    if 0 <= hour <= 6:
        return "ASIA"
    if 7 <= hour <= 12:
        return "LONDON"
    if 13 <= hour <= 17:
        return "NY"
    return "NY_PM"


def compute_session_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Map timestamps to ASIA/LONDON/NY/NY_PM buckets using index hour.

    The hour is taken directly from the provided index (no timezone conversion).
    """
    hours = pd.Series(index.hour, index=index)
    return hours.map(_hour_bucket).rename("ctx_session_bucket")


def compute_state_age(state_hat: pd.Series, *, max_age: int) -> pd.Series:
    """Count consecutive bars in the same base state (1-based, clipped)."""
    state_series = state_hat.copy()
    change = state_series.ne(state_series.shift(1)) | state_series.isna()
    groups = change.cumsum()
    age = state_series.groupby(groups).cumcount() + 1
    age = age.where(state_series.notna()).clip(upper=max_age)
    return age.rename("ctx_state_age")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _resolve_vwap_column(ohlcv: pd.DataFrame) -> str | None:
    for col in ("vwap", "VWAP"):
        if col in ohlcv.columns:
            return col
    return None


def _resolve_vwap_series(
    ohlcv: pd.DataFrame,
    *,
    vwap_window: int,
    reset_mode: str | None,
    vwap_day_anchor: str,
) -> pd.Series | None:
    vwap_col = _resolve_vwap_column(ohlcv)
    if vwap_col is not None:
        provided = pd.to_numeric(ohlcv[vwap_col], errors="coerce")
        if provided.notna().any():
            return provided

    _LOGGER.warning(
        "VWAP not provided in ohlcv; computing day-anchor VWAP (anchor=%s, window=%s).",
        vwap_day_anchor,
        vwap_window,
    )
    _LOGGER.debug("ohlcv.columns=%s", list(ohlcv.columns))
    vwap = _compute_standard_vwap(
        ohlcv,
        day_anchor=vwap_day_anchor,
        vwap_window=vwap_window,
    )
    if vwap is None:
        return None
    return pd.to_numeric(vwap, errors="coerce")


def _resolve_volume_series(
    ohlcv: pd.DataFrame,
) -> tuple[pd.Series | None, str, float, float]:
    real_volume = None
    if "real_volume" in ohlcv.columns:
        real_volume = pd.to_numeric(ohlcv["real_volume"], errors="coerce")
    tick_volume = None
    if "tick_volume" in ohlcv.columns:
        tick_volume = pd.to_numeric(ohlcv["tick_volume"], errors="coerce")
    if tick_volume is None and real_volume is None:
        return None, "missing", 100.0, 100.0
    if real_volume is None:
        vol_raw = tick_volume
        source = "tick_volume"
    elif tick_volume is None:
        vol_raw = real_volume
        source = "real_volume"
    else:
        real_mask = real_volume > 0
        vol_raw = pd.Series(
            np.where(real_mask, real_volume, tick_volume),
            index=ohlcv.index,
        )
        if real_mask.any() and (~real_mask).any():
            source = "mixed"
        elif real_mask.any():
            source = "real_volume"
        else:
            source = "tick_volume"
    pre_zero_pct = float((vol_raw.fillna(0).eq(0)).mean() * 100.0)
    vol = vol_raw.astype(float).clip(lower=1.0)
    post_zero_pct = float(vol.eq(0).mean() * 100.0)
    return vol, source, pre_zero_pct, post_zero_pct


def _day_id(index: pd.DatetimeIndex, anchor: str) -> pd.Series:
    if anchor == "server_midnight":
        day = index.normalize()
        return pd.Series(day, index=index)
    if anchor == "utc_midnight":
        if index.tz is None:
            localized = index.tz_localize("UTC")
        else:
            localized = index.tz_convert("UTC")
        day = localized.normalize().tz_localize(None)
        return pd.Series(day, index=index)
    if anchor == "ny_rollover_17":
        if index.tz is None:
            localized = index.tz_localize("UTC")
        else:
            localized = index.tz_convert("UTC")
        ny_time = localized.tz_convert("America/New_York")
        day = (ny_time - pd.Timedelta(hours=17)).normalize().tz_localize(None)
        return pd.Series(day, index=index)
    raise ValueError(f"Unknown vwap_day_anchor: {anchor}")


def _compute_standard_vwap(
    ohlcv: pd.DataFrame,
    *,
    day_anchor: str,
    vwap_window: int,
) -> pd.Series | None:
    vol, _, _, _ = _resolve_volume_series(ohlcv)
    if vol is None:
        _LOGGER.warning("VWAP missing for ctx_dist_vwap_atr; skipping (no volume column).")
        return None

    vwap_win = max(int(vwap_window), 1)
    day_id = _day_id(ohlcv.index, day_anchor)

    df = pd.DataFrame(
        {
            "high": pd.to_numeric(ohlcv["high"], errors="coerce"),
            "low": pd.to_numeric(ohlcv["low"], errors="coerce"),
            "close": pd.to_numeric(ohlcv["close"], errors="coerce"),
            "_vol": vol,
        },
        index=ohlcv.index,
    )

    def _compute_group(group: pd.DataFrame) -> pd.Series:
        hh = group["high"].rolling(vwap_win, min_periods=1).max()
        ll = group["low"].rolling(vwap_win, min_periods=1).min()
        hv = group["_vol"].rolling(vwap_win, min_periods=1).max().clip(lower=1.0)
        pivot_price = (hh + ll + group["close"]) / 3.0
        cum_vol = hv.cumsum()
        cum_pv = (pivot_price * hv).cumsum()
        return cum_pv / cum_vol

    return df.groupby(day_id, sort=False, group_keys=False).apply(_compute_group).rename(
        "vwap"
    )


def _resolve_atr_series(ohlcv: pd.DataFrame, *, window: int) -> pd.Series:
    for col in ohlcv.columns:
        if col.lower() == "atr_h2":
            return pd.to_numeric(ohlcv[col], errors="coerce")
    return _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], window)


def log_vwap_validation(
    df: pd.DataFrame,
    vwap: pd.Series | None,
    atr: pd.Series | None,
    group_key: pd.Series | None,
    day_id: pd.Series | None,
    source: str,
    anchor: str | None,
    window: int,
) -> None:
    """Log sanity checks for VWAP-derived distances."""
    _, vol_source, vol_zero_pct_pre, vol_zero_pct_post = _resolve_volume_series(df)
    if vwap is None:
        _LOGGER.info(
            "VWAP_VALIDATION source=%s anchor=%s window=%s vol_source=%s "
            "vol_zero_pct_pre_clip=%.2f vol_zero_pct_post_clip=%.2f "
            "vwap_nan_pct=100.00 atr_missing",
            source,
            anchor,
            window,
            vol_source,
            vol_zero_pct_pre,
            vol_zero_pct_post,
        )
        return

    vwap_nan_pct = vwap.isna().mean() * 100.0
    if atr is None or atr.isna().all():
        _LOGGER.info(
            "VWAP_VALIDATION source=%s anchor=%s window=%s vol_source=%s "
            "vol_zero_pct_pre_clip=%.2f vol_zero_pct_post_clip=%.2f "
            "vwap_nan_pct=%.2f atr_missing",
            source,
            anchor,
            window,
            vol_source,
            vol_zero_pct_pre,
            vol_zero_pct_post,
            vwap_nan_pct,
        )
        return

    close = pd.to_numeric(df["close"], errors="coerce")
    denom = np.maximum(atr.to_numpy(), np.finfo(float).eps)
    dist = (close - vwap).abs() / denom
    dist_nan_pct = pd.Series(dist, index=close.index).isna().mean() * 100.0
    dist_p50, dist_p90, dist_p99, dist_max = np.nanquantile(
        dist, [0.5, 0.9, 0.99, 1.0]
    )
    _LOGGER.info(
        "VWAP_VALIDATION source=%s anchor=%s window=%s vol_source=%s "
        "vol_zero_pct_pre_clip=%.2f vol_zero_pct_post_clip=%.2f "
        "vwap_nan_pct=%.2f dist_nan_pct=%.2f dist_p50=%.4f dist_p90=%.4f "
        "dist_p99=%.4f dist_max=%.4f",
        source,
        anchor,
        window,
        vol_source,
        vol_zero_pct_pre,
        vol_zero_pct_post,
        vwap_nan_pct,
        dist_nan_pct,
        dist_p50,
        dist_p90,
        dist_p99,
        dist_max,
    )
    if vwap_nan_pct > 5.0 or dist_nan_pct > 5.0:
        if vol_zero_pct_pre > 0.0:
            cause = "volume_zero_pre_clip"
        elif vol_source == "missing":
            cause = "missing_volume"
        else:
            cause = "vwap_nan_regression"
        _LOGGER.warning(
            "VWAP_INVALID source=%s anchor=%s window=%s vwap_nan_pct=%.2f "
            "dist_nan_pct=%.2f cause=%s",
            source,
            anchor,
            window,
            vwap_nan_pct,
            dist_nan_pct,
            cause,
        )

    if day_id is not None:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        day_low = low.groupby(day_id).transform("min")
        day_high = high.groupby(day_id).transform("max")
        outside = (vwap < day_low) | (vwap > day_high)
        per_day = outside.groupby(day_id).mean()
        pct_outside = float(per_day.mean() * 100.0) if not per_day.empty else 0.0
        tp = (high + low + close) / 3.0
        first_mask = day_id.ne(day_id.shift(1))
        vwap_first = vwap[first_mask]
        tp_first = tp[first_mask]
        atr_first = atr[first_mask]
        denom_first = np.maximum(atr_first.to_numpy(), np.finfo(float).eps)
        dist_first = (vwap_first - tp_first).abs() / denom_first
        if len(dist_first):
            p50_first, p90_first = np.nanquantile(dist_first, [0.5, 0.9])
        else:
            p50_first, p90_first = 0.0, 0.0
        _LOGGER.info(
            "VWAP_DAY_VALIDATION anchor=%s pct_outside_day_range=%.2f "
            "abs_vwap_first_tp_first_atr_p50=%.4f abs_vwap_first_tp_first_atr_p90=%.4f",
            anchor,
            pct_outside,
            p50_first,
            p90_first,
        )

    if group_key is None:
        return

    dist_series = pd.Series(dist, index=close.index)
    summary_rows = []
    for bucket, values in dist_series.groupby(group_key):
        values = values.dropna()
        if values.empty:
            continue
        summary_rows.append(
            {
                "bucket": bucket,
                "count": int(values.shape[0]),
                "p50": float(values.quantile(0.5)),
                "p90": float(values.quantile(0.9)),
            }
        )
    if not summary_rows:
        return

    summary = pd.DataFrame(summary_rows).sort_values("count", ascending=False).head(4)
    for _, row in summary.iterrows():
        _LOGGER.info(
            "VWAP_BUCKET_VALIDATION bucket=%s count=%d dist_p50=%.4f dist_p90=%.4f",
            row["bucket"],
            row["count"],
            row["p50"],
            row["p90"],
        )


def compute_dist_vwap_atr(
    ohlcv: pd.DataFrame,
    *,
    atr_window: int,
    vwap_window: int = 50,
    vwap_reset_mode: str | None = None,
    vwap_day_anchor: str | None = None,
    eps: float,
) -> pd.Series:
    """Compute absolute distance to VWAP normalized by ATR."""
    anchor = _resolve_vwap_day_anchor(None, vwap_day_anchor)
    vwap = _resolve_vwap_series(
        ohlcv,
        vwap_window=vwap_window,
        reset_mode=vwap_reset_mode,
        vwap_day_anchor=anchor,
    )
    if vwap is None:
        return pd.Series(np.nan, index=ohlcv.index, name="ctx_dist_vwap_atr")
    atr = _resolve_atr_series(ohlcv, window=atr_window)
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    denom = np.maximum(atr.to_numpy(), eps)
    dist = (close - vwap).abs() / denom
    return pd.Series(dist, index=ohlcv.index, name="ctx_dist_vwap_atr")


def build_context_features(
    ohlcv: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    symbol: str | None,
    timeframe: str | None,
    config: ContextFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build ctx_* features after state_hat/margin are available."""
    cfg = config or ContextFeatureConfig(vwap_day_anchor=_resolve_vwap_day_anchor(symbol, None))
    is_context_enabled(symbol, timeframe)

    session_bucket = compute_session_bucket(outputs.index)
    state_age = compute_state_age(outputs["state_hat"], max_age=cfg.max_state_age)
    anchor = _resolve_vwap_day_anchor(symbol, cfg.vwap_day_anchor)
    day_id = _day_id(ohlcv.index, anchor)
    vwap = _resolve_vwap_series(
        ohlcv,
        vwap_window=cfg.vwap_window,
        reset_mode=cfg.vwap_reset_mode,
        vwap_day_anchor=anchor,
    )
    atr = _resolve_atr_series(ohlcv, window=cfg.atr_window)
    if vwap is None:
        dist_vwap_atr = pd.Series(np.nan, index=ohlcv.index, name="ctx_dist_vwap_atr")
        vwap_source = "missing"
        vwap_anchor = "missing"
    else:
        close = pd.to_numeric(ohlcv["close"], errors="coerce")
        denom = np.maximum(atr.to_numpy(), cfg.eps)
        dist = (close - vwap).abs() / denom
        dist_vwap_atr = pd.Series(dist, index=ohlcv.index, name="ctx_dist_vwap_atr")
        vwap_col = _resolve_vwap_column(ohlcv)
        provided_valid = False
        if vwap_col is not None:
            provided_valid = pd.to_numeric(ohlcv[vwap_col], errors="coerce").notna().any()
        vwap_source = "provided" if provided_valid else "day_anchor_std"
        vwap_anchor = anchor
    dist_vwap_atr = dist_vwap_atr.reindex(outputs.index)
    vwap_source = pd.Series(vwap_source, index=outputs.index, name="ctx_vwap_source")
    vwap_anchor = pd.Series(vwap_anchor, index=outputs.index, name="ctx_vwap_anchor")
    log_vwap_validation(
        ohlcv,
        vwap,
        atr,
        session_bucket.reindex(ohlcv.index),
        day_id,
        vwap_source.iloc[0],
        anchor,
        cfg.vwap_window,
    )

    ctx_features = pd.concat(
        [session_bucket, state_age, dist_vwap_atr, vwap_source, vwap_anchor], axis=1
    )
    non_null_pct = (ctx_features.notna().mean() * 100.0).to_dict()
    _LOGGER.info(
        "CTX_FEATURES_NON_NULL_PCT %s",
        {key: round(value, 2) for key, value in non_null_pct.items()},
    )
    return ctx_features


__all__ = [
    "ContextFeatureConfig",
    "is_context_enabled",
    "compute_session_bucket",
    "compute_state_age",
    "compute_dist_vwap_atr",
    "build_context_features",
]
