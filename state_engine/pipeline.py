"""Pipeline utilities to build datasets for the State Engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FeatureConfig, FeatureEngineer
from .labels import StateLabeler


@dataclass(frozen=True)
class DatasetArtifacts:
    features: pd.DataFrame
    labels: pd.Series
    full_features: pd.DataFrame


class DatasetBuilder:
    """Create feature/label datasets for the State Engine."""

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        labeler: StateLabeler | None = None,
    ) -> None:
        self.feature_engineer = FeatureEngineer(feature_config)
        self.labeler = labeler or StateLabeler()

    def build(self, ohlcv: pd.DataFrame) -> DatasetArtifacts:
        """Compute features and labels from OHLCV data."""
        full_features = self.feature_engineer.compute_features(ohlcv)
        labels = self.labeler.label(full_features)
        features = self.feature_engineer.training_features(full_features)
        return DatasetArtifacts(features=features, labels=labels, full_features=full_features)


__all__ = ["DatasetArtifacts", "DatasetBuilder"]
