"""LightGBM model wrapper for the State Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from .labels import StateLabels


@dataclass(frozen=True)
class StateEngineModelConfig:
    """Configuration for the LightGBM model."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = -1
    random_state: int = 42
    class_weight: str | dict[int, float] | None = None


class StateEngineModel:
    """Train and run a LightGBM multiclass classifier."""

    def __init__(self, config: StateEngineModelConfig | None = None) -> None:
        self.config = config or StateEngineModelConfig()
        self._model: lgb.LGBMClassifier | None = None
        self.metadata: dict[str, Any] = {}

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the LightGBM model."""
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(StateLabels),
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
        )
        model.fit(features, labels)
        self._model = model

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities as a DataFrame."""
        self._require_fitted_model()
        probs = self._model.predict_proba(features)
        return pd.DataFrame(
            probs,
            columns=["P(balance)", "P(transition)", "P(trend)"],
            index=features.index,
        )

    def predict_outputs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict state, margin, and probabilities for reporting."""
        probas = self.predict_proba(features)
        top1 = probas.max(axis=1)
        top2 = probas.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
        margin = top1 - top2
        state_hat = probas.idxmax(axis=1).map(
            {
                "P(balance)": StateLabels.BALANCE,
                "P(transition)": StateLabels.TRANSITION,
                "P(trend)": StateLabels.TREND,
            }
        )
        return pd.DataFrame(
            {
                "state_hat": state_hat,
                "margin": margin,
                "P(balance)": probas["P(balance)"],
                "P(transition)": probas["P(transition)"],
                "P(trend)": probas["P(trend)"],
            },
            index=features.index,
        )

    def predict_state(self, features: pd.DataFrame) -> pd.Series:
        """Predict most likely state label."""
        self._require_fitted_model()
        preds = self._model.predict(features)
        return pd.Series(preds, index=features.index)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Persist model to disk."""
        self._require_fitted_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "metadata": {
                "model_config": asdict(self.config),
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self._model = payload["model"]
            self.metadata = payload.get("metadata", {})
        elif isinstance(payload, lgb.LGBMClassifier):
            self._model = payload
            self.metadata = {}
        else:
            raise ValueError("Unrecognized model payload; expected dict or LGBMClassifier.")

    def feature_importances(self) -> pd.Series:
        """Return feature importances from the fitted model."""
        self._require_fitted_model()
        importances = self._model.feature_importances_
        feature_names = self._model.feature_name_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def _require_fitted_model(self) -> None:
        if self._model is None:
            raise RuntimeError("StateEngineModel must be fit or loaded before use.")


__all__ = ["StateEngineModel", "StateEngineModelConfig"]
