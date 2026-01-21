"""State Engine package for PA-first market state classification."""

from .backtest import BacktestConfig, Signal, Trade, run_backtest
from .events import EventDetectionConfig, EventFamily, detect_events, label_events
from .features import FeatureConfig, FeatureEngineer
from .gating import GatingPolicy, PhaseDConfig
from .labels import StateLabeler, StateLabels
from .model import StateEngineModel, StateEngineModelConfig
from .mt5_connector import MT5Connector
from .scoring import EventScorer, EventScorerBundle, EventScorerConfig, FeatureBuilder
from .sweep import run_param_sweep

__all__ = [
    "BacktestConfig",
    "Signal",
    "Trade",
    "run_backtest",
    "EventDetectionConfig",
    "EventFamily",
    "detect_events",
    "label_events",
    "FeatureConfig",
    "FeatureEngineer",
    "GatingPolicy",
    "PhaseDConfig",
    "StateLabeler",
    "StateLabels",
    "StateEngineModel",
    "StateEngineModelConfig",
    "MT5Connector",
    "EventScorer",
    "EventScorerBundle",
    "EventScorerConfig",
    "FeatureBuilder",
    "run_param_sweep",
]
