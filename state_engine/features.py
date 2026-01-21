"""Feature engineering for the PA-first State Engine.

All features are calculated using only information available at or before time t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for PA-based features."""

    window: int = 24
    recent_window: int = 8
    enable_slopes: bool = False
    include_er_netmove: bool = False
    allow_label_proxies: bool = False    


class FeatureEngineer:
    """Compute PA-native features for State Engine classification."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with engineered features.

        Parameters
        ----------
        ohlcv:
            DataFrame with columns: ["open", "high", "low", "close", "volume"].
        """
        self._validate_input(ohlcv)
        cfg = self.config

        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        atr_w = self._atr(high, low, close, cfg.window)
        atr_n = self._atr(high, low, close, cfg.recent_window)

        displacement = (close - close.shift(cfg.window)).abs()
        path_raw = close.diff().abs().rolling(cfg.window).sum()
        efficiency = displacement / path_raw.replace(0, np.nan)

        net_move = displacement / atr_w.replace(0, np.nan)
        path = path_raw / atr_w.replace(0, np.nan)

        # Use prior window (up to t-1) to avoid range always containing current close
        range_high = high.shift(1).rolling(cfg.window).max()
        range_low = low.shift(1).rolling(cfg.window).min()
        range_width = (range_high - range_low)
        range_w = range_width / atr_w.replace(0, np.nan)

        close_location = self._close_location(close, range_low, range_width)
        break_mag = self._break_magnitude(close, range_low, range_high, atr_n)

        reentry = self._reentry_count(
            close,
            range_low,
            range_high,
            cfg.recent_window,
        )
        inside_ratio = self._inside_bars_ratio(high, low, cfg.recent_window)
        swing_high, swing_low = self._swing_counts(high, low, cfg.window)

        atr_ratio = atr_n / atr_w.replace(0, np.nan)

        features = pd.DataFrame(
            {
                "NetMove": net_move,
                "Path": path,
                "ER": efficiency,
                "Range_W": range_w,
                "CloseLocation": close_location,
                "BreakMag": break_mag,
                "ReentryCount": reentry,
                "InsideBarsRatio": inside_ratio,
                "SwingHighCount": swing_high,
                "SwingLowCount": swing_low,
                "ATR_Ratio": atr_ratio,
            },
            index=ohlcv.index,
        )

        if cfg.enable_slopes:
            features["ERSlope"] = self._slope(efficiency, cfg.window)
            features["RangeSlope"] = self._slope(range_w, cfg.window)

        return features

    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        names = [
            "Path",
            "Range_W",
            "CloseLocation",
            "BreakMag",
            "ReentryCount",
            "InsideBarsRatio",
            "SwingHighCount",
            "SwingLowCount",
            "ATR_Ratio",
        ]
        if self.config.include_er_netmove:
            names = ["NetMove", "ER", *names]
        if self.config.enable_slopes:
            names.extend(["ERSlope", "RangeSlope"])
        return names

    def training_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return features aligned with the configured training set."""
        names = self.feature_names()
        if not self.config.allow_label_proxies:
            # Avoid definition leakage: labeler uses ER/NetMove to build y.
            names = [n for n in names if n not in ("ER", "NetMove")]

        missing = [name for name in names if name not in features.columns]
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
        return features[names]

    @staticmethod
    def _validate_input(ohlcv: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
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

    @staticmethod
    def _close_location(
        close: pd.Series,
        range_low: pd.Series,
        range_width: pd.Series,
    ) -> pd.Series:
        location = (close - range_low) / range_width.replace(0, np.nan)
        return location.fillna(0.5)

    @staticmethod
    def _break_magnitude(
        close: pd.Series,
        range_low: pd.Series,
        range_high: pd.Series,
        atr_n: pd.Series,
    ) -> pd.Series:
        clamped = close.clip(lower=range_low, upper=range_high)
        return (close - clamped).abs() / atr_n.replace(0, np.nan)

    @staticmethod
    def _reentry_count(
        close: pd.Series,
        range_low: pd.Series,
        range_high: pd.Series,
        window: int,
    ) -> pd.Series:
        inside = (close >= range_low) & (close <= range_high)
        outside = ~inside
        reentry = outside.shift(1) & inside
        return reentry.rolling(window).sum()

    @staticmethod
    def _inside_bars_ratio(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        inside = (high <= prev_high) & (low >= prev_low)
        return inside.rolling(window).mean()

    @staticmethod
    def _swing_counts(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_high_2 = high.shift(2)
        prev_low_2 = low.shift(2)
        pivot_high = (prev_high > prev_high_2) & (prev_high > high)
        pivot_low = (prev_low < prev_low_2) & (prev_low < low)
        swing_high = pivot_high.astype(int).rolling(window).sum().fillna(0.0)
        swing_low = pivot_low.astype(int).rolling(window).sum().fillna(0.0)
        return swing_high, swing_low

    @staticmethod
    def _slope(series: pd.Series, window: int) -> pd.Series:
        return (series - series.shift(window)) / window


__all__ = ["FeatureConfig", "FeatureEngineer"]

