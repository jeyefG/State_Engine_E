"""Event detection and feature extraction for the Event Scorer (Layer A)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import logging

import numpy as np
import pandas as pd

from state_engine.vwap import compute_vwap_mt5_daily

EPS = 1e-9
BASE_SCORE_TF = "M5"
_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe).upper()


def _timeframe_minutes(timeframe: str) -> int:
    key = _normalize_timeframe(timeframe)
    if key not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe for event scaling: {timeframe}")
    return _TIMEFRAME_MINUTES[key]


def _scale_window(value: int, score_tf: str, base_tf: str = BASE_SCORE_TF) -> int:
    base_minutes = _timeframe_minutes(base_tf)
    score_minutes = _timeframe_minutes(score_tf)
    scaled = max(1, int(round(value * (base_minutes / score_minutes))))
    return scaled


class EventType(str, Enum):
    TOUCH_VWAP = "E_TOUCH_VWAP"
    REJECTION_VWAP = "E_REJECTION_VWAP"
    NEAR_VWAP = "E_NEAR_VWAP"


class EventFamily(str, Enum):
    BALANCE_FADE = "E_BALANCE_FADE"
    BALANCE_REVERT = "E_BALANCE_REVERT"
    TRANSITION_TEST = "E_TRANSITION_TEST"
    TRANSITION_FAILURE = "E_TRANSITION_FAILURE"
    TREND_PULLBACK = "E_TREND_PULLBACK"
    TREND_CONTINUATION = "E_TREND_CONTINUATION"
    GENERIC_VWAP = "E_GENERIC_VWAP"


def required_allow_by_family() -> dict[str, str]:
    return {
        EventFamily.BALANCE_FADE.value: "LOOK_FOR_balance_fade",
        EventFamily.TRANSITION_FAILURE.value: "LOOK_FOR_transition_failure",
        EventFamily.TREND_PULLBACK.value: "LOOK_FOR_trend_pullback",
        EventFamily.TREND_CONTINUATION.value: "LOOK_FOR_trend_continuation",
    }


@dataclass(frozen=True)
class EventDetectionConfig:
    score_tf: str = BASE_SCORE_TF
    balance_window: int = 12
    balance_edge: float = 0.20
    balance_min_range_atr: float = 0.8
    balance_soft_edge: float = 0.35
    balance_soft_range_mult: float = 0.5
    transition_window: int = 12
    transition_near_edge_atr: float = 0.25
    trend_window: int = 6
    trend_pullback_edge: float = 0.30
    trend_momentum_atr_thr: float = 0.3
    trend_pullback_momentum_mult: float = 0.5
    trend_pullback_edge_mult: float = 0.6
    trend_continuation_comp_window: int = 6
    trend_continuation_break_window: int = 6
    trend_continuation_comp_thr: float = 1.2
    trend_continuation_momentum_window: int = 3
    trend_continuation_comp_mult: float = 1.5
    trend_continuation_break_atr: float = 0.25
    vwap_slope_window: int = 5
    volume_rel_window: int = 20
    vwap_window: int = 50
    near_vwap_atr: float = 0.35
    near_vwap_cooldown_bars: int = 3
    vwap_reset_mode: str | None = None
    vwap_session_cut_hour: int = 22
    vwap_tz_server: str | None = None
    near_vwap_mode: str = "enter"
    touch_mode: str = "on"
    rejection_mode: str = "on"

    def for_timeframe(self, score_tf: str, *, base_tf: str = BASE_SCORE_TF) -> "EventDetectionConfig":
        score_tf_norm = _normalize_timeframe(score_tf)
        base_tf_norm = _normalize_timeframe(base_tf)
        if score_tf_norm == base_tf_norm:
            return replace(self, score_tf=score_tf_norm)
        return replace(
            self,
            score_tf=score_tf_norm,
            balance_window=_scale_window(self.balance_window, score_tf_norm, base_tf_norm),
            transition_window=_scale_window(self.transition_window, score_tf_norm, base_tf_norm),
            trend_window=_scale_window(self.trend_window, score_tf_norm, base_tf_norm),
            trend_continuation_comp_window=_scale_window(
                self.trend_continuation_comp_window,
                score_tf_norm,
                base_tf_norm,
            ),
            trend_continuation_break_window=_scale_window(
                self.trend_continuation_break_window,
                score_tf_norm,
                base_tf_norm,
            ),
            trend_continuation_momentum_window=_scale_window(
                self.trend_continuation_momentum_window,
                score_tf_norm,
                base_tf_norm,
            ),
            vwap_slope_window=_scale_window(self.vwap_slope_window, score_tf_norm, base_tf_norm),
            volume_rel_window=_scale_window(self.volume_rel_window, score_tf_norm, base_tf_norm),
            vwap_window=_scale_window(self.vwap_window, score_tf_norm, base_tf_norm),
            near_vwap_cooldown_bars=_scale_window(self.near_vwap_cooldown_bars, score_tf_norm, base_tf_norm),
        )


class EventExtractor:
    """Layer A extractor: detect events and compute event features without leakage."""

    def __init__(self, *, eps: float = EPS, config: EventDetectionConfig | None = None) -> None:
        self.eps = eps
        self.config = config or EventDetectionConfig()
        self.logger = logging.getLogger(__name__)

    def extract(self, df_m5: pd.DataFrame, symbol: str, *, vwap_col: str = "vwap") -> pd.DataFrame:
        df = df_m5.copy()
        vwap_metadata: dict[str, object] = {}
        if vwap_col not in df.columns:
            time_col = None
            if not isinstance(df.index, pd.DatetimeIndex):
                if "ts" in df.columns:
                    time_col = "ts"
                elif "time" in df.columns:
                    time_col = "time"
            self.logger.info("VWAP not provided; computing mt5_daily VWAP from OHLC.")
            vwap_series = compute_vwap_mt5_daily(
                df,
                time_col=time_col,
                high_col="high",
                low_col="low",
                close_col="close",
                real_vol_col="volume",
                tick_vol_col="tick_volume",
                tz_server=self.config.vwap_tz_server,
            )
            df[vwap_col] = vwap_series.to_numpy()
            vwap_metadata.update(vwap_series.attrs)
        else:
            vwap_metadata["vwap_reset_mode_effective"] = "provided"
            vwap_metadata["vwap_volume_source"] = "provided"

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

        ts = _extract_timestamp(df)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(ts)
            ts = pd.Series(df.index, index=df.index)

        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        vwap = df[vwap_col]
        vwap_valid_mask = vwap.notna()
        vwap_valid_pct = float(vwap_valid_mask.mean() * 100.0) if len(vwap_valid_mask) else 0.0
        vwap_invalid_reason = vwap_metadata.get("vwap_invalid_reason")
        if vwap_invalid_reason is None and not vwap_valid_mask.any():
            vwap_invalid_reason = "vwap_all_nan"
        vwap_metadata["vwap_valid_pct"] = vwap_valid_pct
        if vwap_invalid_reason is not None:
            vwap_metadata["vwap_invalid_reason"] = vwap_invalid_reason
        self.logger.info(
            "VWAP validity: valid_pct=%.2f%% invalid_reason=%s volume_source=%s",
            vwap_valid_pct,
            vwap_metadata.get("vwap_invalid_reason"),
            vwap_metadata.get("vwap_volume_source"),
        )
        if not vwap_valid_mask.all():
            invalid_bars = int((~vwap_valid_mask).sum())
            self.logger.info("VWAP invalid bars=%s (excluded from VWAP event checks)", invalid_bars)
            vwap_metadata["vwap_invalid_bars"] = invalid_bars

        dist_to_vwap = close - vwap
        abs_dist_to_vwap = dist_to_vwap.abs()
        vwap_slope_window = max(int(self.config.vwap_slope_window), 1)
        vwap_slope = (vwap - vwap.shift(vwap_slope_window)) / float(vwap_slope_window)
        range_1 = (high - low)
        body_ratio = (close - open_).abs() / (range_1 + self.eps)
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        upper_wick_ratio = upper_wick / (range_1 + self.eps)
        lower_wick_ratio = lower_wick / (range_1 + self.eps)

        atr_14 = _ensure_atr_14(df)

        self._log_vwap_telemetry(
            close=close,
            vwap=vwap,
            atr=atr_14,
        )

        dist_to_vwap_atr = dist_to_vwap / (atr_14 + self.eps)

        if "volume" in df.columns:
            volume_window = max(int(self.config.volume_rel_window), 1)
            volume_rel = df["volume"] / df["volume"].rolling(volume_window).mean()
        else:
            volume_rel = pd.Series(np.nan, index=df.index)

        is_touch_exact = (low <= vwap) & (high >= vwap)
        is_rejection = is_touch_exact & (
            ((close > vwap) & (open_ <= vwap)) | ((close < vwap) & (open_ >= vwap))
        )

        features = pd.DataFrame(
            {
                "dist_to_vwap": dist_to_vwap,
                "abs_dist_to_vwap": abs_dist_to_vwap,
                "vwap_slope": vwap_slope,
                "range_1": range_1,
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "atr_14": atr_14,
                "dist_to_vwap_atr": dist_to_vwap_atr,
                "volume_rel": volume_rel,
                "is_touch_exact": is_touch_exact.astype(int),
                "is_rejection": is_rejection.astype(int),
                "hour": ts.dt.hour,
                "minute": ts.dt.minute,
            },
            index=df.index,
        )

        threshold = self.config.near_vwap_atr
        dist_to_vwap_atr_abs = dist_to_vwap_atr.abs()
        dist_quantiles = dist_to_vwap_atr_abs.quantile([0.1, 0.5, 0.9, 0.99]).to_dict()
        self.logger.info(
            "VWAP sanity: threshold=%.3f dist_to_vwap_atr_abs_q=%s",
            threshold,
            {k: round(v, 4) for k, v in dist_quantiles.items()},
        )
        near_mask = dist_to_vwap_atr_abs <= threshold
        prev_outside = (dist_to_vwap_atr_abs.shift(1) > threshold).fillna(True)
        enter_mask = near_mask & prev_outside
        near_mode = self.config.near_vwap_mode
        if near_mode not in {"enter", "continuous"}:
            raise ValueError("near_vwap_mode must be 'enter' or 'continuous'")
        event_near_mask = enter_mask if near_mode == "enter" else near_mask
        cooldown = max(int(self.config.near_vwap_cooldown_bars), 0)
        if cooldown > 0:
            recent_enter = (
                event_near_mask.shift(1)
                .rolling(cooldown, min_periods=1)
                .max()
                .fillna(False)
                .gt(0)
            )
            before_count = int(event_near_mask.sum())
            event_near_mask = event_near_mask & ~recent_enter
            filtered_count = before_count - int(event_near_mask.sum())
        else:
            filtered_count = 0

        self.logger.info(
            "E_NEAR_VWAP mode=%s count=%s cooldown=%s filtered=%s",
            near_mode,
            int(event_near_mask.sum()),
            cooldown,
            filtered_count,
        )

        context_cols = [
            col
            for col in df.columns
            if col.startswith("LOOK_FOR_")
            or col.startswith("state_hat_")
            or col.startswith("margin_")
            or col.startswith("ctx_")
            or col in {"state_hat_ctx", "margin_ctx"}
        ]
        context = df[context_cols] if context_cols else None

        rejection_mode = self.config.rejection_mode
        if rejection_mode not in {"on", "off"}:
            raise ValueError("rejection_mode must be 'on' or 'off'")
        touch_mode = self.config.touch_mode
        if touch_mode not in {"on", "off"}:
            raise ValueError("touch_mode must be 'on' or 'off'")

        rejection_mask = is_rejection if rejection_mode == "on" else pd.Series(False, index=df.index)
        touch_mask = (
            is_touch_exact & ~rejection_mask if touch_mode == "on" else pd.Series(False, index=df.index)
        )
        near_mask_final = event_near_mask & ~rejection_mask & ~touch_mask

        events = []
        if rejection_mode == "on":
            events.append(_build_events(features, context, symbol, rejection_mask, EventType.REJECTION_VWAP))
        if touch_mode == "on":
            events.append(_build_events(features, context, symbol, touch_mask, EventType.TOUCH_VWAP))
        events.append(_build_events(features, context, symbol, near_mask_final, EventType.NEAR_VWAP))

        events_df = pd.concat([frame for frame in events if not frame.empty], axis=0)
        if events_df.empty:
            empty = pd.DataFrame(
                columns=[
                    "symbol",
                    "ts",
                    "event_type",
                    "family_id",
                    "side",
                    "event_id",
                    *features.columns,
                    "event_features",
                ]
            )
            empty.index.name = "ts"
            if vwap_metadata:
                empty.attrs.update(vwap_metadata)
            return empty

        allow_cols = [col for col in events_df.columns if col.startswith("LOOK_FOR_")]
        events_df["family_id"] = _infer_family_id(events_df, allow_cols)

        events_df.index.name = "ts"
        events_df = events_df.sort_index()
        events_df["event_id"] = range(1, len(events_df) + 1)
        events_df["ts"] = events_df.index
        self.logger.info(
            "Event context columns selected: count=%s cols=%s",
            len(context_cols),
            context_cols,
        )
        if vwap_metadata:
            events_df.attrs.update(vwap_metadata)
        return events_df

    def _log_vwap_telemetry(
        self,
        *,
        close: pd.Series,
        vwap: pd.Series,
        atr: pd.Series,
    ) -> None:
        vwap_nan_pct = float(vwap.isna().mean() * 100.0) if len(vwap) else 0.0
        vwap_min = float(vwap.min()) if vwap.notna().any() else float("nan")
        vwap_max = float(vwap.max()) if vwap.notna().any() else float("nan")
        self.logger.info(
            "VWAP mt5_daily telemetry: mode=mt5_daily nan_pct=%.2f vwap_min=%.5f vwap_max=%.5f",
            vwap_nan_pct,
            vwap_min,
            vwap_max,
        )
        if atr is not None and atr.notna().any():
            denom = np.maximum(atr.to_numpy(), self.eps)
            dist = (close - vwap).abs().to_numpy() / denom
            dist_series = pd.Series(dist, index=close.index)
            quantiles = dist_series.quantile([0.1, 0.5, 0.9, 0.99]).to_dict()
            self.logger.info(
                "VWAP mt5_daily sanity: abs_dist_to_vwap_atr_q=%s",
                {k: round(v, 4) for k, v in quantiles.items()},
            )
        else:
            returns = close.pct_change()
            scale = float(returns.std()) if returns.notna().any() else np.nan
            if not np.isnan(scale) and scale > 0:
                dist = (close - vwap).abs() / scale
                quantiles = dist.quantile([0.1, 0.5, 0.9, 0.99]).to_dict()
                self.logger.info(
                    "VWAP mt5_daily sanity: abs_dist_to_vwap_stdret_q=%s",
                    {k: round(v, 4) for k, v in quantiles.items()},
                )


def _compute_vwap(
    df: pd.DataFrame,
    *,
    vwap_col: str,
    reset_mode: str,
    vwap_window: int,
) -> dict[str, object]:
    """
    VWAP fallback usando el método estándar por sesión:
      - price = (high + low + close) / 3
      - vwap = sum(price * volume) / sum(volume)

    reset_mode:
      - "cumulative": una sola acumulación en todo el DF
      - "daily": resetea por día calendario (según timestamp extraído)
      - "session": resetea por session_id/session
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns for VWAP: {sorted(missing)}")

    if reset_mode not in {"daily", "session", "cumulative"}:
        raise ValueError("vwap_reset_mode must be 'daily', 'session', or 'cumulative'")

    metadata: dict[str, object] = {
        "vwap_volume_source": None,
        "vwap_invalid_reason": None,
    }

    volume_col = None
    if "real_volume" in df.columns and pd.to_numeric(df["real_volume"], errors="coerce").notna().any():
        volume_col = "real_volume"
    elif "tick_volume" in df.columns and pd.to_numeric(df["tick_volume"], errors="coerce").notna().any():
        volume_col = "tick_volume"

    if volume_col is None:
        metadata["vwap_invalid_reason"] = "missing_volume_column"
        return {
            "vwap": pd.Series(np.nan, index=df.index, name=vwap_col),
            "metadata": metadata,
        }

    volume = pd.to_numeric(df[volume_col], errors="coerce").astype(float)
    if (volume > 0).sum() == 0:
        metadata["vwap_volume_source"] = volume_col
        metadata["vwap_invalid_reason"] = "no_positive_volume"
        return {
            "vwap": pd.Series(np.nan, index=df.index, name=vwap_col),
            "metadata": metadata,
        }
    metadata["vwap_volume_source"] = volume_col

    def _compute_group(g: pd.DataFrame) -> pd.Series:
        price = (g["high"] + g["low"] + g["close"]) / 3.0
        vol = pd.to_numeric(g[volume_col], errors="coerce").astype(float)
        pv = (price * vol).where(vol > 0, 0.0)
        cum_pv = pv.cumsum()
        cum_vol = vol.where(vol > 0, 0.0).cumsum()
        vwap = cum_pv / cum_vol.replace(0, np.nan)
        vwap.name = vwap_col
        return vwap

    if reset_mode == "cumulative":
        return {"vwap": _compute_group(df), "metadata": metadata}

    if reset_mode == "session":
        session_col = None
        for col in ("session_id", "session"):
            if col in df.columns:
                session_col = col
                break
        if session_col is None:
            raise ValueError("vwap_reset_mode 'session' requires session_id/session column")
        group_key = df[session_col]
    else:
        ts = _extract_timestamp(df)
        group_key = pd.to_datetime(ts).dt.date

    # groupby con sort=False para respetar el orden original
    vwap = df.groupby(group_key, sort=False, group_keys=False).apply(_compute_group)
    vwap.name = vwap_col
    return {"vwap": vwap, "metadata": metadata}


def detect_events(df_m5_ctx: pd.DataFrame, config: EventDetectionConfig | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper around EventExtractor.extract."""
    symbol = "UNKNOWN"
    if "symbol" in df_m5_ctx.columns and not df_m5_ctx["symbol"].empty:
        symbol = str(df_m5_ctx["symbol"].iloc[0])
    extractor = EventExtractor(config=config)
    return extractor.extract(df_m5_ctx, symbol=symbol)


def label_events(
    events_df: pd.DataFrame,
    df_m5: pd.DataFrame,
    k_bars: int,
    reward_r: float,
    sl_mult: float,
    atr_window: int = 14,
    r_thr: float = 0.0,
    tie_break: str = "distance",
    clip_mtm: bool = True,
) -> pd.DataFrame:
    """Label events using a triple-barrier proxy with continuous outcome."""
    if events_df.empty:
        return events_df.copy()

    required = {"open", "high", "low", "close"}
    missing = required - set(df_m5.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns in df_m5: {sorted(missing)}")

    df = events_df.copy().sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    df_m5 = df_m5.copy().sort_index()
    if df_m5.index.has_duplicates:
        df_m5 = df_m5[~df_m5.index.duplicated(keep="last")]

    atr_short = _atr(df_m5["high"], df_m5["low"], df_m5["close"], atr_window)

    if tie_break not in {"worst", "distance"}:
        raise ValueError("tie_break must be 'worst' or 'distance'")

    entry_prices: list[float | None] = []
    sl_prices: list[float | None] = []
    tp_prices: list[float | None] = []
    labels: list[int | None] = []
    r_outcomes: list[float | None] = []
    r_mean_k: list[float | None] = []

    indexer = df_m5.index.get_indexer(df.index)

    for i, (ts, row) in enumerate(df.iterrows()):
        idx = indexer[i]
        if idx == -1:
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        entry_idx = idx + 1
        if entry_idx >= len(df_m5.index):
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        entry_price = float(df_m5["open"].iloc[entry_idx])
        atr_value = atr_short.iloc[idx]
        if pd.isna(atr_value):
            entry_prices.append(entry_price)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        sl_proxy = float(atr_value * sl_mult)
        if row["side"] == "long":
            sl = entry_price - sl_proxy
            tp = entry_price + reward_r * sl_proxy
            direction = 1.0
        else:
            sl = entry_price + sl_proxy
            tp = entry_price - reward_r * sl_proxy
            direction = -1.0

        outcome: float | None = None
        end_idx = min(entry_idx + k_bars, len(df_m5.index))
        future_slice = df_m5.iloc[entry_idx:end_idx]
        for future_idx, (high_val, low_val) in enumerate(
            zip(future_slice["high"].to_numpy(), future_slice["low"].to_numpy()),
            start=entry_idx,
        ):
            if row["side"] == "long":
                hit_sl = low_val <= sl
                hit_tp = high_val >= tp
            else:
                hit_sl = high_val >= sl
                hit_tp = low_val <= tp

            if hit_sl and hit_tp:
                if tie_break == "worst":
                    outcome = -1.0
                else:
                    open_price = float(df_m5["open"].iloc[future_idx])
                    sl_dist = abs(open_price - sl)
                    tp_dist = abs(tp - open_price)
                    outcome = -1.0 if sl_dist <= tp_dist else reward_r
                break
            if hit_sl:
                outcome = -1.0
                break
            if hit_tp:
                outcome = reward_r
                break

        if outcome is None:
            end_idx_mtm = min(entry_idx + k_bars - 1, len(df_m5.index) - 1)
            close_end = float(df_m5["close"].iloc[end_idx_mtm])
            sl_dist = sl_proxy if sl_proxy != 0 else np.nan
            outcome = direction * (close_end - entry_price) / sl_dist
            if clip_mtm and not pd.isna(outcome):
                outcome = float(np.clip(outcome, -1.0, reward_r))

        if sl_proxy != 0 and not future_slice.empty:
            mtm_series = direction * (future_slice["close"] - entry_price) / sl_proxy
            r_mean_k.append(float(mtm_series.mean()))
        else:
            r_mean_k.append(None)

        entry_prices.append(entry_price)
        sl_prices.append(sl)
        tp_prices.append(tp)
        if outcome is None or pd.isna(outcome):
            r_outcomes.append(None)
            labels.append(None)
        else:
            r_outcomes.append(float(outcome))
            labels.append(int(outcome > r_thr))

    df["entry_price"] = entry_prices
    df["sl_price"] = sl_prices
    df["tp_price"] = tp_prices
    df["label"] = labels
    df["r_outcome"] = r_outcomes
    df["r_mean_k"] = r_mean_k
    return df


def _extract_timestamp(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"])
    if "time" in df.columns:
        return pd.to_datetime(df["time"])
    raise ValueError("Missing datetime index or 'ts'/'time' column for timestamps")


def _build_events(
    features: pd.DataFrame,
    context: pd.DataFrame | None,
    symbol: str,
    mask: pd.Series,
    event_type: EventType,
) -> pd.DataFrame:
    if not mask.any():
        return pd.DataFrame()

    subset = features.loc[mask].copy()
    if context is not None:
        subset = subset.join(context.loc[mask], how="left")
    subset.insert(0, "symbol", symbol)
    subset.insert(1, "ts", subset.index)
    subset.insert(2, "event_type", event_type.value)
    subset.insert(3, "family_id", EventFamily.GENERIC_VWAP.value)
    subset.insert(4, "side", np.where(subset["dist_to_vwap"] >= 0, "long", "short"))
    subset["event_id"] = np.nan
    subset["event_features"] = subset[features.columns].to_dict(orient="records")
    subset.index.name = "ts"
    return subset


def _ensure_atr_14(df: pd.DataFrame) -> pd.Series:
    for col in ("atr_14", "atr", "ATR", "atr14"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").astype(float)
            series = series.rename("atr_14")
            return series

    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing required OHLC columns for ATR: {col}")

    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        sorted_df = df.sort_index()
        atr = _atr(sorted_df["high"], sorted_df["low"], sorted_df["close"], 14)
        atr = atr.reindex(df.index)
    else:
        atr = _atr(df["high"], df["low"], df["close"], 14)
    return atr.rename("atr_14")


def _resolve_vwap_reset_mode(df: pd.DataFrame, mode: str | None) -> str:
    if mode is not None:
        return mode
    if any(col in df.columns for col in ("session_id", "session")):
        return "session"
    return "daily"


def _infer_family_id(events_df: pd.DataFrame, allow_cols: list[str]) -> pd.Series:
    if not allow_cols:
        return pd.Series(EventFamily.GENERIC_VWAP.value, index=events_df.index)
    allow_active = events_df[allow_cols].fillna(0).astype(int)
    family = pd.Series(EventFamily.GENERIC_VWAP.value, index=events_df.index)
    for family_value, allow_col in required_allow_by_family().items():
        if allow_col not in allow_cols:
            continue
        mask = allow_active[allow_col].eq(1) & family.eq(EventFamily.GENERIC_VWAP.value)
        family.loc[mask] = family_value
    return family


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


__all__ = [
    "EventType",
    "EventExtractor",
    "EventFamily",
    "EventDetectionConfig",
    "required_allow_by_family",
    "detect_events",
    "label_events",
]
