"""Quality layer for descriptive regime assessment (Phase C)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from state_engine.config_loader import deep_merge, load_config
from state_engine.labels import StateLabels


QUALITY_LABEL_UNCLASSIFIED = "UNCLASSIFIED"

QUALITY_LABELS_BY_STATE: dict[StateLabels, list[str]] = {
    StateLabels.BALANCE: [
        "BALANCE_STABLE",
        "BALANCE_COMPRESSING",
        "BALANCE_LEAKING",
        QUALITY_LABEL_UNCLASSIFIED,
    ],
    StateLabels.TREND: [
        "TREND_STRONG",
        "TREND_GRINDING",
        "TREND_FRAGILE",
        QUALITY_LABEL_UNCLASSIFIED,
    ],
    StateLabels.TRANSITION: [
        "TRANSITION_NOISY",
        "TRANSITION_FAILED",
        QUALITY_LABEL_UNCLASSIFIED,
    ],
}

ALL_QUALITY_LABELS: list[str] = sorted(
    {label for labels in QUALITY_LABELS_BY_STATE.values() for label in labels}
)


@dataclass(frozen=True)
class QualityThresholds:
    er_low: float = 0.2
    er_mid: float = 0.4
    er_high: float = 0.65
    netmove_low: float = 0.3
    netmove_mid: float = 0.6
    netmove_high: float = 1.0
    break_low: float = 0.25
    break_mid: float = 0.6
    break_high: float = 1.0
    reentry_min: float = 2.0
    reentry_mid: float = 3.0
    reentry_high: float = 5.0
    reentry_low: float = 1.0
    reentry_max_for_stable: float = 3.0
    inside_mid: float = 0.4
    inside_high: float = 0.6
    range_slope_min: float = -0.01
    swing_mid: float = 3.0
    swing_high: float = 5.0
    close_mid_low: float = 0.4
    close_mid_high: float = 0.6


@dataclass(frozen=True)
class QualityWindows:
    range_slope_k: int = 6
    transition_failed_lookback: int = 5


@dataclass(frozen=True)
class QualityScoring:
    score_min: float = 0.7
    score_margin: float = 0.15


@dataclass(frozen=True)
class QualityConfig:
    thresholds: QualityThresholds = QualityThresholds()
    windows: QualityWindows = QualityWindows()
    scoring: QualityScoring = QualityScoring()


@dataclass(frozen=True)
class QualityDiagnostics:
    config_sources: dict[str, str]
    distribution_rows: list[dict[str, Any]]
    persistence_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    warnings: list[str]


def default_quality_config_dict() -> dict[str, Any]:
    return {
        "thresholds": asdict(QualityThresholds()),
        "windows": asdict(QualityWindows()),
        "scoring": asdict(QualityScoring()),
    }


def validate_quality_config_dict(config: dict[str, Any]) -> None:
    for forbidden in ("state_engine", "gating", "event_scorer"):
        if forbidden in config:
            raise ValueError(f"Quality config must not include '{forbidden}'.")

    allowed_sections = {"thresholds", "windows", "scoring", "diagnostic_tables"}
    unknown_sections = set(config.keys()) - allowed_sections
    if unknown_sections:
        raise ValueError(f"Unknown quality config sections: {sorted(unknown_sections)}")

    for section, values in config.items():
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Quality config section '{section}' must be a mapping.")
        if section == "diagnostic_tables":
            _validate_diagnostic_tables(values)
            continue
        for key, value in values.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Quality config '{section}.{key}' must be numeric.")


def _validate_diagnostic_tables(diagnostic_cfg: dict[str, Any]) -> None:
    allowed = {"n_min", "delta_max", "top_k", "bins"}
    unknown_keys = set(diagnostic_cfg.keys()) - allowed
    if unknown_keys:
        raise ValueError(
            "Quality config 'diagnostic_tables' unknown keys: "
            f"{sorted(unknown_keys)}"
        )
    for key in ("n_min", "top_k", "delta_max"):
        if key in diagnostic_cfg and diagnostic_cfg[key] is not None:
            if not isinstance(diagnostic_cfg[key], (int, float)):
                raise ValueError(f"Quality config 'diagnostic_tables.{key}' must be numeric.")
    bins = diagnostic_cfg.get("bins")
    if bins is None:
        return
    if not isinstance(bins, dict):
        raise ValueError("Quality config 'diagnostic_tables.bins' must be a mapping.")
    for bin_key, values in bins.items():
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                "Quality config 'diagnostic_tables.bins.%s' must be a list." % bin_key
            )
        for value in values:
            if not isinstance(value, (int, float, str)):
                raise ValueError(
                    "Quality config 'diagnostic_tables.bins.%s' values must be numeric."
                    % bin_key
                )


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flattened[f"{section}.{key}"] = value
        else:
            flattened[section] = values
    return flattened


def _merge_with_sources(
    base: dict[str, Any],
    sources: dict[str, str],
    overrides: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    merged = deep_merge(base, overrides)
    flattened = _flatten_config(overrides)
    for key in flattened:
        sources[key] = source_name
    return merged


def _config_from_dict(config: dict[str, Any]) -> QualityConfig:
    thresholds = QualityThresholds(**config.get("thresholds", {}))
    windows = QualityWindows(**config.get("windows", {}))
    scoring = QualityScoring(**config.get("scoring", {}))
    return QualityConfig(thresholds=thresholds, windows=windows, scoring=scoring)


def load_quality_config(
    symbol: str,
    cli_path: str | Path | None = None,
) -> tuple[QualityConfig, dict[str, str], list[str]]:
    project_root = Path(__file__).resolve().parents[1]
    template_path = project_root / "configs" / "quality" / "_template.yaml"
    symbol_path = project_root / "configs" / "quality" / f"{symbol}.yaml"

    warnings: list[str] = []

    defaults = default_quality_config_dict()
    sources = {key: "defaults" for key in _flatten_config(defaults)}

    merged = defaults
    for path, name in ((template_path, "template"), (symbol_path, "symbol")):
        if path.exists():
            loaded = load_config(path)
            validate_quality_config_dict(loaded)
            merged = _merge_with_sources(merged, sources, loaded, name)
            missing_keys = set(_flatten_config(defaults)) - set(_flatten_config(loaded))
            if missing_keys:
                warnings.append(f"config_incomplete:{name} missing={sorted(missing_keys)}")
        else:
            warnings.append(f"config_missing:{name} path={path}")

    if cli_path is not None:
        loaded = load_config(cli_path)
        validate_quality_config_dict(loaded)
        merged = _merge_with_sources(merged, sources, loaded, "cli")
        missing_keys = set(_flatten_config(defaults)) - set(_flatten_config(loaded))
        if missing_keys:
            warnings.append(f"config_incomplete:cli missing={sorted(missing_keys)}")

    config = _config_from_dict(merged)
    return config, sources, warnings


def assign_quality_labels(
    states: pd.Series,
    features: pd.DataFrame,
    config: QualityConfig,
) -> tuple[pd.Series, list[str]]:
    required = {
        "ER",
        "NetMove",
        "Range_W",
        "BreakMag",
        "ReentryCount",
        "InsideBarsRatio",
    }
    warnings: list[str] = []

    missing = set(required) - set(features.columns)
    missing_optional: set[str] = set()
    has_close_location = "CloseLocation" in features.columns
    if not has_close_location:
        missing_optional.add("CloseLocation")
    swing_count = None
    if "SwingCount" in features.columns:
        swing_count = features["SwingCount"]
    elif {"SwingHighCount", "SwingLowCount"}.issubset(features.columns):
        swing_count = features["SwingHighCount"] + features["SwingLowCount"]
    else:
        missing.add("SwingCount")

    if missing:
        warnings.append(f"missing_features:{sorted(missing)}")
        return pd.Series(QUALITY_LABEL_UNCLASSIFIED, index=states.index), warnings
    if missing_optional:
        warnings.append(f"missing_optional_features:{sorted(missing_optional)}")

    er = features["ER"]
    net_move = features["NetMove"].abs()
    range_w = features["Range_W"]
    break_mag = features["BreakMag"]
    reentry = features["ReentryCount"]
    inside_ratio = features["InsideBarsRatio"]
    swing_count = swing_count.reindex(features.index)

    range_slope = _rolling_slope(range_w, config.windows.range_slope_k)
    if has_close_location:
        close_location = features["CloseLocation"]
        close_mid = close_location.between(
            config.thresholds.close_mid_low,
            config.thresholds.close_mid_high,
        )
        close_mid_recent = (
            close_mid.rolling(config.windows.transition_failed_lookback)
            .max()
            .fillna(0)
            .astype(bool)
        )
    else:
        close_mid = pd.Series(0.0, index=features.index)
        close_mid_recent = pd.Series(0.0, index=features.index)

    scores = pd.DataFrame(index=features.index)

    balance_stable_components = [
        _score_leq(er, config.thresholds.er_low),
        _score_leq(net_move, config.thresholds.netmove_low),
        _score_geq(reentry, config.thresholds.reentry_min),
    ]
    if has_close_location:
        balance_stable_components.append(close_mid.astype(float))
    scores["BALANCE_STABLE"] = _score_avg(balance_stable_components)

    scores["BALANCE_COMPRESSING"] = _score_avg(
        [
            _score_leq(range_slope, config.thresholds.range_slope_min),
            _score_geq(inside_ratio, config.thresholds.inside_high),
            _score_leq(break_mag, config.thresholds.break_low),
        ]
    )

    scores["BALANCE_LEAKING"] = _score_avg(
        [
            _score_leq(er, config.thresholds.er_mid),
            _score_or(
                _score_geq(break_mag, config.thresholds.break_mid),
                _score_geq(net_move, config.thresholds.netmove_mid),
            ),
            _score_leq(reentry, config.thresholds.reentry_max_for_stable),
        ]
    )

    scores["TREND_STRONG"] = _score_avg(
        [
            _score_geq(er, config.thresholds.er_high),
            _score_geq(net_move, config.thresholds.netmove_high),
            _score_leq(reentry, config.thresholds.reentry_low),
        ]
    )

    scores["TREND_GRINDING"] = _score_avg(
        [
            _score_between(er, config.thresholds.er_mid, config.thresholds.er_high),
            _score_geq(reentry, config.thresholds.reentry_mid),
            _score_geq(swing_count, config.thresholds.swing_mid),
            _score_leq(net_move, config.thresholds.netmove_high),
        ]
    )

    scores["TREND_FRAGILE"] = _score_avg(
        [
            _score_leq(er, config.thresholds.er_mid),
            _score_geq(reentry, config.thresholds.reentry_high),
            _score_or(
                _score_geq(inside_ratio, config.thresholds.inside_high),
                _score_leq(break_mag, config.thresholds.break_low),
            ),
        ]
    )

    scores["TRANSITION_NOISY"] = _score_avg(
        [
            _score_geq(reentry, config.thresholds.reentry_high),
            _score_geq(swing_count, config.thresholds.swing_high),
            _score_geq(inside_ratio, config.thresholds.inside_mid),
            _score_leq(break_mag, config.thresholds.break_mid),
        ]
    )

    transition_failed_components = [
        _score_between(break_mag, config.thresholds.break_mid, config.thresholds.break_high),
        _score_geq(reentry, config.thresholds.reentry_mid),
    ]
    if has_close_location:
        transition_failed_components.append(close_mid_recent.astype(float))
    scores["TRANSITION_FAILED"] = _score_avg(transition_failed_components)

    labels = pd.Series(QUALITY_LABEL_UNCLASSIFIED, index=features.index, dtype=object)

    for state, label_list in QUALITY_LABELS_BY_STATE.items():
        state_mask = states == state
        candidate_scores = scores.loc[state_mask, label_list[:-1]]
        assigned = _select_label(
            candidate_scores,
            config.scoring.score_min,
            config.scoring.score_margin,
        )
        labels.loc[state_mask] = assigned

    return labels, warnings


def build_quality_diagnostics(
    states: pd.Series,
    quality_labels: pd.Series,
    config_sources: dict[str, str],
    warnings: Iterable[str],
) -> QualityDiagnostics:
    distribution_rows: list[dict[str, Any]] = []
    for state, labels in QUALITY_LABELS_BY_STATE.items():
        mask = states == state
        state_labels = quality_labels.loc[mask]
        total = len(state_labels)
        for label in labels:
            count = int((state_labels == label).sum())
            pct = (count / total) * 100 if total else 0.0
            distribution_rows.append(
                {"state": state.name, "label": label, "count": count, "pct": pct}
            )

    persistence_rows: list[dict[str, Any]] = []
    run_lengths = _run_lengths(quality_labels)
    for label in ALL_QUALITY_LABELS:
        lengths = run_lengths.get(label, [])
        if lengths:
            median = float(np.median(lengths))
            p90 = float(np.percentile(lengths, 90))
        else:
            median = 0.0
            p90 = 0.0
        persistence_rows.append({"label": label, "median": median, "p90": p90})

    split_rows: list[dict[str, Any]] = []
    ordered = pd.DataFrame(
        {"state": states, "quality": quality_labels},
        index=quality_labels.index,
    ).sort_index()
    splits = np.array_split(ordered, 3) if len(ordered) else []
    split_names = ["early", "mid", "late"]
    for name, split in zip(split_names, splits):
        for state in QUALITY_LABELS_BY_STATE:
            state_mask = split["state"] == state
            state_total = int(state_mask.sum())
            if state_total == 0:
                split_rows.append(
                    {
                        "split": name,
                        "state": state.name,
                        "pct_unclassified": 0.0,
                        "top_label": QUALITY_LABEL_UNCLASSIFIED,
                        "top_pct": 0.0,
                    }
                )
                continue
            subset = split.loc[state_mask, "quality"]
            unclassified_pct = float((subset == QUALITY_LABEL_UNCLASSIFIED).mean() * 100)
            classified = subset[subset != QUALITY_LABEL_UNCLASSIFIED]
            if classified.empty:
                top_label = QUALITY_LABEL_UNCLASSIFIED
                top_pct = 0.0
            else:
                top_label = classified.value_counts().idxmax()
                top_pct = float((classified == top_label).mean() * 100)
            split_rows.append(
                {
                    "split": name,
                    "state": state.name,
                    "pct_unclassified": unclassified_pct,
                    "top_label": top_label,
                    "top_pct": top_pct,
                }
            )

    warnings_list = list(warnings)
    overall_counts = quality_labels.value_counts(dropna=False)
    total = len(quality_labels)
    for label, count in overall_counts.items():
        pct = (count / total) * 100 if total else 0.0
        if pct < 1.0 or pct > 80.0:
            warnings_list.append(f"degenerate_label:{label} pct={pct:.2f}")

    return QualityDiagnostics(
        config_sources=config_sources,
        distribution_rows=distribution_rows,
        persistence_rows=persistence_rows,
        split_rows=split_rows,
        warnings=warnings_list,
    )


def format_quality_diagnostics(diagnostics: QualityDiagnostics) -> list[str]:
    lines: list[str] = []

    config_rows = []
    for key in sorted(diagnostics.config_sources):
        source = diagnostics.config_sources[key]
        if source != "defaults":
            config_rows.append([key, source])
    if not config_rows:
        config_rows = [["none", "defaults"]]
    lines.extend(_format_table("QUALITY_CONFIG_EFFECTIVE", ["key", "source"], config_rows))

    dist_rows = [
        [row["state"], row["label"], row["count"], f"{row['pct']:.2f}%"]
        for row in diagnostics.distribution_rows
    ]
    lines.extend(
        _format_table("QUALITY_DISTRIBUTION", ["state", "label", "count", "pct"], dist_rows)
    )

    persistence_rows = [
        [row["label"], f"{row['median']:.2f}", f"{row['p90']:.2f}"]
        for row in diagnostics.persistence_rows
    ]
    lines.extend(
        _format_table("QUALITY_PERSISTENCE", ["label", "median", "p90"], persistence_rows)
    )

    split_rows = [
        [
            row["split"],
            row["state"],
            f"{row['pct_unclassified']:.2f}%",
            row["top_label"],
            f"{row['top_pct']:.2f}%",
        ]
        for row in diagnostics.split_rows
    ]
    lines.extend(
        _format_table(
            "QUALITY_SPLITS",
            ["split", "state", "pct_unclassified", "top_label", "top_pct"],
            split_rows,
        )
    )

    warning_rows = [[warning] for warning in diagnostics.warnings] or [["none"]]
    lines.extend(_format_table("QUALITY_WARNINGS", ["warning"], warning_rows))

    return lines


def _format_table(title: str, columns: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [title]
    header = " | ".join(columns)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(" | ".join(str(item) for item in row))
    return lines


def _select_label(scores: pd.DataFrame, score_min: float, score_margin: float) -> pd.Series:
    if scores.empty:
        return pd.Series(QUALITY_LABEL_UNCLASSIFIED, index=scores.index)
    score_values = scores.to_numpy(dtype=float)
    best_idx = np.argmax(score_values, axis=1)
    best_scores = np.max(score_values, axis=1)
    second_best = np.partition(score_values, -2, axis=1)[:, -2] if scores.shape[1] > 1 else 0.0
    margin_ok = best_scores - second_best >= score_margin
    min_ok = best_scores >= score_min
    selected_labels = scores.columns.to_numpy()[best_idx]
    labels = np.where(min_ok & margin_ok, selected_labels, QUALITY_LABEL_UNCLASSIFIED)
    return pd.Series(labels, index=scores.index, dtype=object)


def _score_avg(scores: Iterable[pd.Series]) -> pd.Series:
    stacked = pd.concat(scores, axis=1).astype(float)
    return stacked.mean(axis=1).fillna(0.0).clip(0.0, 1.0)


def _score_or(score_a: pd.Series, score_b: pd.Series) -> pd.Series:
    combined = pd.concat([score_a, score_b], axis=1).max(axis=1)
    return combined.fillna(0.0).clip(0.0, 1.0)


def _score_leq(series: pd.Series, threshold: float) -> pd.Series:
    scale = max(abs(threshold), 1e-6)
    return (1 - ((series - threshold) / scale).clip(lower=0.0)).clip(0.0, 1.0)


def _score_geq(series: pd.Series, threshold: float) -> pd.Series:
    scale = max(abs(threshold), 1e-6)
    return (1 - ((threshold - series) / scale).clip(lower=0.0)).clip(0.0, 1.0)


def _score_between(series: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        return (series == low).astype(float)
    center = (low + high) / 2.0
    half = (high - low) / 2.0
    dist = (series - center).abs()
    return (1 - (dist / half)).clip(0.0, 1.0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 2)
    idx = np.arange(window)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        coef = np.polyfit(idx, values, 1)[0]
        return float(coef)

    return series.rolling(window).apply(slope, raw=True)


def _run_lengths(series: pd.Series) -> dict[str, list[int]]:
    lengths: dict[str, list[int]] = {label: [] for label in ALL_QUALITY_LABELS}
    if series.empty:
        return lengths

    current_label = series.iloc[0]
    run = 1
    for label in series.iloc[1:]:
        if label == current_label:
            run += 1
        else:
            lengths.setdefault(current_label, []).append(run)
            current_label = label
            run = 1
    lengths.setdefault(current_label, []).append(run)
    return lengths


__all__ = [
    "QUALITY_LABEL_UNCLASSIFIED",
    "QUALITY_LABELS_BY_STATE",
    "QualityConfig",
    "QualityDiagnostics",
    "assign_quality_labels",
    "build_quality_diagnostics",
    "format_quality_diagnostics",
    "load_quality_config",
    "validate_quality_config_dict",
]
