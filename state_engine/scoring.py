"""Event scorer (ML) for M5 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .events import EventFamily
from .labels import StateLabels
from .session import SESSION_BUCKETS, get_session_bucket


@dataclass(frozen=True)
class EventScorerConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = -1
    random_state: int = 42
    calibration_method: str = "sigmoid"


class FeatureBuilder:
    """Build M5 features for event scoring."""

    def __init__(
        self,
        atr_window: int = 14,
        micro_window: int = 6,
        *,
        context_tf: str = "H1",
        context_prefix: str | None = None,
    ) -> None:
        self.atr_window = atr_window
        self.micro_window = micro_window
        self.context_tf = str(context_tf).upper()
        self.context_prefix = context_prefix
        self.logger = logging.getLogger(__name__)

    def _resolve_context_columns(self, df_m5_ctx: pd.DataFrame) -> tuple[str, str]:
        if self.context_prefix:
            state_col = f"{self.context_prefix}state_hat"
            margin_col = f"{self.context_prefix}margin"
            if state_col in df_m5_ctx.columns and margin_col in df_m5_ctx.columns:
                return state_col, margin_col
        state_col = f"state_hat_{self.context_tf}"
        margin_col = f"margin_{self.context_tf}"
        if state_col in df_m5_ctx.columns and margin_col in df_m5_ctx.columns:
            return state_col, margin_col
        if "state_hat_ctx" in df_m5_ctx.columns and "margin_ctx" in df_m5_ctx.columns:
            return "state_hat_ctx", "margin_ctx"
        relevant_cols = [
            col
            for col in df_m5_ctx.columns
            if col.startswith("state_hat_") or col.startswith("margin_") or col.startswith("LOOK_FOR_")
        ]
        expected = [
            state_col,
            margin_col,
            "state_hat_ctx",
            "margin_ctx",
        ]
        raise ValueError(
            "Missing context columns for feature building. "
            f"Expected one of: {expected}. "
            f"Available relevant columns: {relevant_cols}"
        )

    def validate_context_metadata(self, metadata: dict[str, Any] | None) -> None:
        if not isinstance(metadata, dict):
            return
        meta_tf = metadata.get("context_tf") or metadata.get("allow_tf")
        if meta_tf is None:
            return
        meta_tf_norm = str(meta_tf).upper()
        if meta_tf_norm != self.context_tf:
            self.logger.warning(
                "Context TF mismatch: FeatureBuilder=%s scorer_metadata=%s",
                self.context_tf,
                meta_tf_norm,
            )

    def build(self, df_m5_ctx: pd.DataFrame) -> pd.DataFrame:
        state_col, margin_col = self._resolve_context_columns(df_m5_ctx)
        required = {"open", "high", "low", "close", state_col, margin_col}
        missing = required - set(df_m5_ctx.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature building: {sorted(missing)}")

        df = df_m5_ctx.copy()
        if "pf_session_bucket" not in df.columns:
            symbol_series = df.get("symbol", pd.Series("UNKNOWN", index=df.index))
            session_bucket = [
                get_session_bucket(ts, symbol)
                for ts, symbol in zip(df.index, symbol_series)
            ]
            df["pf_session_bucket"] = session_bucket
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        returns_1 = close.pct_change()
        returns_3 = close.pct_change(3)
        realized_vol = returns_1.rolling(self.micro_window).std()
        atr_short = _atr(high, low, close, self.atr_window)

        range_ = (high - low)
        body = (close - open_).abs()
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        range_safe = range_.replace(0, np.nan)

        body_ratio = body / range_safe
        upper_ratio = upper_wick / range_safe
        lower_ratio = lower_wick / range_safe

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        overlap = (np.minimum(high, prev_high) - np.maximum(low, prev_low)).clip(lower=0)
        overlap_ratio = overlap / range_safe

        chop_proxy = (range_.rolling(self.micro_window).sum()) / (
            high.rolling(self.micro_window).max() - low.rolling(self.micro_window).min()
        ).replace(0, np.nan)

        roc = close.diff(self.micro_window) / close.shift(self.micro_window)
        range_atr = range_ / atr_short.replace(0, np.nan)
        recent_high = high.rolling(self.micro_window).max()
        recent_low = low.rolling(self.micro_window).min()
        dist_to_high = (recent_high - close) / atr_short.replace(0, np.nan)
        dist_to_low = (close - recent_low) / atr_short.replace(0, np.nan)
        mom_atr = close.diff(self.micro_window) / atr_short.replace(0, np.nan)

        balance_window = 12
        trend_window = 6
        continuation_comp_window = 6
        continuation_break_window = 6

        balance_high = high.rolling(balance_window).max()
        balance_low = low.rolling(balance_window).min()
        balance_width = (balance_high - balance_low).replace(0, np.nan)
        range_width_atr = balance_width / atr_short.replace(0, np.nan)
        location_0_1 = (close - balance_low) / balance_width

        trend_high = high.rolling(trend_window).max()
        trend_low = low.rolling(trend_window).min()
        trend_width = (trend_high - trend_low).replace(0, np.nan)
        trend_location = (close - trend_low) / trend_width
        momentum_norm = (close - close.shift(trend_window)) / atr_short.replace(0, np.nan)
        pullback_depth_0_1 = np.where(momentum_norm >= 0, 1.0 - trend_location, trend_location)

        comp_range = high.rolling(continuation_comp_window).max() - low.rolling(continuation_comp_window).min()
        compression_ratio = comp_range / atr_short.replace(0, np.nan)
        breakout_high = high.shift(1).rolling(continuation_break_window).max()
        breakout_low = low.shift(1).rolling(continuation_break_window).min()
        breakout_dist_long = (breakout_high - close) / atr_short.replace(0, np.nan)
        breakout_dist_short = (close - breakout_low) / atr_short.replace(0, np.nan)
        breakout_dist_atr = np.where(momentum_norm >= 0, breakout_dist_long, breakout_dist_short)

        hours = df.index.hour.to_series(index=df.index)
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)

        features = pd.DataFrame(
            {
                "ret_1": returns_1,
                "ret_3": returns_3,
                "realized_vol": realized_vol,
                "atr_short": atr_short,
                "range": range_,
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_ratio,
                "lower_wick_ratio": lower_ratio,
                "overlap_ratio": overlap_ratio,
                "chop_proxy": chop_proxy,
                "roc": roc,
                "range_atr": range_atr,
                "dist_to_high_atr": dist_to_high,
                "dist_to_low_atr": dist_to_low,
                "mom_atr": mom_atr,
                "range_width_atr": range_width_atr,
                "location_0_1": location_0_1,
                "momentum_norm": momentum_norm,
                "compression_ratio": compression_ratio,
                "breakout_dist_atr": breakout_dist_atr,
                "pullback_depth_0_1": pullback_depth_0_1,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "margin_ctx": df[margin_col],
            },
            index=df.index,
        )

        state_one_hot = pd.get_dummies(df[state_col].map(_state_name), prefix="state")
        for state_name in ("balance", "transition", "trend"):
            col = f"state_{state_name}"
            if col not in state_one_hot.columns:
                state_one_hot[col] = 0.0
        allow_cols = [col for col in df.columns if col.startswith("LOOK_FOR_")]
        allow_features = df[allow_cols].astype(float) if allow_cols else pd.DataFrame(index=df.index)

        session_one_hot = pd.get_dummies(df["pf_session_bucket"], prefix="pf_session_bucket")
        for bucket in SESSION_BUCKETS:
            col = f"pf_session_bucket_{bucket}"
            if col not in session_one_hot.columns:
                session_one_hot[col] = 0.0

        features = pd.concat([features, state_one_hot, allow_features, session_one_hot], axis=1)
        return features

    def add_family_features(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.DataFrame:
        family_one_hot = pd.get_dummies(family_ids, prefix="family")
        for family in EventFamily:
            col = f"family_{family.value}"
            if col not in family_one_hot.columns:
                family_one_hot[col] = 0.0
        return pd.concat([features, family_one_hot], axis=1)


class EventScorer:
    """LightGBM scorer with probability calibration."""

    def __init__(self, config: EventScorerConfig | None = None) -> None:
        self.config = config or EventScorerConfig()
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: CalibratedClassifierCV | None = None
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        calib_features: pd.DataFrame | None = None,
        calib_labels: pd.Series | None = None,
    ) -> None:
        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        model.fit(features, labels)
        self._model = model

        if calib_features is not None and calib_labels is not None and not calib_labels.empty:
            class_counts = calib_labels.value_counts()
            if calib_labels.nunique() < 2 or class_counts.min() < 2:
                self._calibrator = None
            else:
                cv_folds = min(5, int(class_counts.min()))
                calibrator = CalibratedClassifierCV(
                    FrozenEstimator(model),
                    method=self.config.calibration_method,
                    cv=cv_folds,
                )
                calibrator.fit(calib_features, calib_labels)
                self._calibrator = calibrator
        else:
            self._calibrator = None

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        if self._calibrator is not None:
            proba = self._calibrator.predict_proba(features)[:, 1]
        else:
            proba = self._model.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name="edge_score")

    def predict_score(self, features: pd.DataFrame) -> pd.Series:
        return self.predict_proba(features)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "calibrator": self._calibrator,
            "metadata": {
                "config": asdict(self.config),
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict):
            raise ValueError("Invalid scorer payload; expected dict.")
        self._model = payload.get("model")
        self._calibrator = payload.get("calibrator")
        self.metadata = payload.get("metadata", {})

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("EventScorer must be fit or loaded before use.")


class EventScorerBundle:
    """Family-specific scorer bundle with a compatible interface."""

    global_key = "__global__"

    def __init__(self, config: EventScorerConfig | None = None) -> None:
        self.config = config or EventScorerConfig()
        self.scorers: dict[str, EventScorer] = {}
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        family_ids: pd.Series,
        calib_features: pd.DataFrame | None = None,
        calib_labels: pd.Series | None = None,
        calib_family_ids: pd.Series | None = None,
    ) -> None:
        families = family_ids.dropna().unique()
        for family in families:
            family_mask = family_ids == family
            if family_mask.sum() == 0:
                continue
            scorer = EventScorer(self.config)
            calib_X = None
            calib_y = None
            if calib_features is not None and calib_labels is not None and calib_family_ids is not None:
                calib_mask = calib_family_ids == family
                if calib_mask.any():
                    calib_X = calib_features.loc[calib_mask]
                    calib_y = calib_labels.loc[calib_mask]
            scorer.fit(features.loc[family_mask], labels.loc[family_mask], calib_X, calib_y)
            self.scorers[str(family)] = scorer

    def predict_proba(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.Series:
        scores = pd.Series(index=features.index, dtype=float, name="edge_score")
        for family, scorer in self.scorers.items():
            if family == self.global_key:
                continue
            mask = family_ids == family
            if mask.any():
                scores.loc[mask] = scorer.predict_proba(features.loc[mask]).values
        global_scorer = self.scorers.get(self.global_key)
        if global_scorer is not None:
            missing = scores.isna()
            if missing.any():
                scores.loc[missing] = global_scorer.predict_proba(features.loc[missing]).values
        return scores

    def predict_score(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.Series:
        return self.predict_proba(features, family_ids)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scorers": self.scorers,
            "metadata": {
                "config": asdict(self.config),
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict):
            raise ValueError("Invalid scorer payload; expected dict.")
        if "scorers" in payload:
            self.scorers = payload.get("scorers", {})
            self.metadata = payload.get("metadata", {})
            return
        if "model" in payload:
            scorer = EventScorer(self.config)
            scorer._model = payload.get("model")
            scorer._calibrator = payload.get("calibrator")
            scorer.metadata = payload.get("metadata", {})
            self.scorers = {self.global_key: scorer}
            self.metadata = scorer.metadata
            return
        raise ValueError("Invalid scorer payload; expected scorers or model.")


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


def _state_name(value: int | float) -> str:
    try:
        label = StateLabels(int(value))
    except Exception:
        return "unknown"
    if label == StateLabels.BALANCE:
        return "balance"
    if label == StateLabels.TRANSITION:
        return "transition"
    return "trend"


__all__ = [
    "EventScorerConfig",
    "FeatureBuilder",
    "EventScorer",
    "EventScorerBundle",
    "EventFamily",
]
