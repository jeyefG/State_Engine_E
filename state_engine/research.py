"""Research exploration utilities for Event Scorer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchVariant:
    variant_id: str
    variant_type: str
    k_bars: int
    n_min: int
    winrate_min: float
    r_mean_min: float
    p10_min: float | None


def _format_float(value: float | None, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return "na"
    formatted = f"{value:.{precision}f}"
    return formatted.rstrip("0").rstrip(".")


def _build_variant_id(variant_type: str, k_bars: int, thresholds: dict[str, float | None]) -> str:
    n_min = thresholds.get("n_min")
    winrate_min = thresholds.get("winrate_min")
    r_mean_min = thresholds.get("r_mean_min")
    p10_min = thresholds.get("p10_min")
    return (
        f"{variant_type}_n{int(n_min)}"
        f"_rm{_format_float(r_mean_min)}"
        f"_p10{_format_float(p10_min)}"
        f"_wr{_format_float(winrate_min)}"
        f"_k{k_bars}"
    )


def _grid_values(grid: dict[str, Iterable[float]], key: str, fallback: float | int | None) -> list[float | int | None]:
    if key in grid and isinstance(grid[key], Iterable):
        values = list(grid[key])
        if values:
            return values
    return [fallback]


_EXPLORATION_KIND_ALIASES = {
    "kbars_sweep": "kbars_only",
    "thresholds_sweep": "thresholds_only",
}


def normalize_exploration_kind(kind: str) -> str:
    return _EXPLORATION_KIND_ALIASES.get(kind, kind)


def generate_research_variants(
    *,
    kind: str,
    base_k_bars: int,
    k_bars_grid: list[int] | None,
    base_thresholds: dict[str, float | int | None],
    thresholds_grid: dict[str, Iterable[float]] | None,
    seed: int,
    max_variants: int | None,
) -> list[ResearchVariant]:
    kind = normalize_exploration_kind(kind)
    if kind not in {"thresholds_only", "kbars_only"}:
        raise ValueError(f"Unsupported research exploration kind: {kind}")

    thresholds_grid = thresholds_grid or {}
    variants: list[ResearchVariant] = []

    if kind == "thresholds_only":
        n_values = _grid_values(thresholds_grid, "n_min", base_thresholds.get("n_min"))
        winrate_values = _grid_values(thresholds_grid, "winrate_min", base_thresholds.get("winrate_min"))
        r_mean_values = _grid_values(thresholds_grid, "r_mean_min", base_thresholds.get("r_mean_min"))
        p10_values = _grid_values(thresholds_grid, "p10_min", base_thresholds.get("p10_min"))
        for n_min, winrate_min, r_mean_min, p10_min in product(
            n_values,
            winrate_values,
            r_mean_values,
            p10_values,
        ):
            thresholds = {
                "n_min": int(n_min) if n_min is not None else 0,
                "winrate_min": float(winrate_min) if winrate_min is not None else 0.0,
                "r_mean_min": float(r_mean_min) if r_mean_min is not None else 0.0,
                "p10_min": float(p10_min) if p10_min is not None else None,
            }
            variant_id = _build_variant_id("thr", base_k_bars, thresholds)
            variants.append(
                ResearchVariant(
                    variant_id=variant_id,
                    variant_type="thresholds_only",
                    k_bars=int(base_k_bars),
                    n_min=thresholds["n_min"],
                    winrate_min=thresholds["winrate_min"],
                    r_mean_min=thresholds["r_mean_min"],
                    p10_min=thresholds["p10_min"],
                )
            )
    else:
        k_values = k_bars_grid if k_bars_grid else [base_k_bars]
        thresholds = {
            "n_min": int(base_thresholds.get("n_min", 0)),
            "winrate_min": float(base_thresholds.get("winrate_min", 0.0)),
            "r_mean_min": float(base_thresholds.get("r_mean_min", 0.0)),
            "p10_min": base_thresholds.get("p10_min"),
        }
        for k_bars in k_values:
            variant_id = _build_variant_id("kbar", int(k_bars), thresholds)
            variants.append(
                ResearchVariant(
                    variant_id=variant_id,
                    variant_type="kbars_only",
                    k_bars=int(k_bars),
                    n_min=thresholds["n_min"],
                    winrate_min=thresholds["winrate_min"],
                    r_mean_min=thresholds["r_mean_min"],
                    p10_min=thresholds["p10_min"],
                )
            )

    if max_variants is not None and max_variants > 0 and len(variants) > max_variants:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(variants), size=max_variants, replace=False)
        variants = [variants[idx] for idx in sorted(chosen)]

    return variants


def _topk_indices(scores: pd.Series, k: int) -> pd.Index:
    if scores.empty:
        return pd.Index([])
    k_eff = min(k, len(scores))
    if k_eff == 0:
        return pd.Index([])
    if scores.nunique(dropna=False) <= 1:
        return scores.sort_index().head(k_eff).index
    return scores.nlargest(k_eff).index


def _lift_at_k(scores: pd.Series, labels_: pd.Series, k: int) -> float:
    base_rate = float(labels_.mean()) if not labels_.empty else float("nan")
    if base_rate == 0 or scores.empty:
        return float("nan")
    top_idx = _topk_indices(scores, k)
    if top_idx.empty:
        return float("nan")
    return float(labels_.loc[top_idx].mean()) / base_rate


def _score_tail_slope(scores: pd.Series, min_points: int = 20) -> float | None:
    if scores.empty or len(scores) < min_points:
        return None
    p90 = float(scores.quantile(0.9))
    p99 = float(scores.quantile(0.99))
    if np.isnan(p90) or np.isnan(p99):
        return None
    return (p99 - p90) / 0.09


def _temporal_dispersion(
    timestamps: pd.DatetimeIndex,
    min_calib_samples: int,
) -> float:
    if timestamps.empty:
        return 0.0
    buckets = timestamps.to_period("M")
    counts = pd.Series(1, index=buckets).groupby(level=0).sum()
    total = len(counts)
    if total == 0:
        return 0.0
    hits = int((counts >= min_calib_samples).sum())
    return hits / total


def _family_concentration(scores: pd.Series, families: pd.Series, k: int = 10) -> float:
    if scores.empty or families.empty:
        return float("nan")
    top_idx = _topk_indices(scores, k)
    if top_idx.empty:
        return float("nan")
    family_counts = families.loc[top_idx].value_counts()
    if family_counts.empty:
        return float("nan")
    return float(family_counts.max()) / len(top_idx)


def _research_status(
    *,
    qualified: bool,
    temporal_dispersion: float,
    family_concentration: float,
    min_temporal_dispersion: float,
    max_family_concentration: float,
    relaxed_thresholds: bool,
) -> str:
    overfit = qualified and (
        (pd.notna(family_concentration) and family_concentration > max_family_concentration)
        or (relaxed_thresholds and temporal_dispersion < min_temporal_dispersion)
    )
    if overfit:
        return "RESEARCH_OVERFIT_SUSPECT"
    if temporal_dispersion < min_temporal_dispersion or not qualified:
        return "RESEARCH_UNSTABLE"
    if qualified and family_concentration <= max_family_concentration and temporal_dispersion >= min_temporal_dispersion:
        return "RESEARCH_OK"
    return "RESEARCH_UNSTABLE"


def evaluate_research_variants(
    *,
    variants: list[ResearchVariant],
    scores: pd.Series | None,
    labels: pd.Series,
    r_outcome: pd.Series,
    families: pd.Series,
    timestamps: pd.DatetimeIndex,
    train_count: int,
    calib_count: int,
    allow_rate: float | None,
    diagnostics_cfg: dict,
    baseline_thresholds: dict[str, float | int | None],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    min_calib_samples = int(diagnostics_cfg.get("min_calib_samples", 200))
    min_temporal_dispersion = float(diagnostics_cfg.get("min_temporal_dispersion", 0.3))
    max_family_concentration = float(diagnostics_cfg.get("max_family_concentration", 0.6))

    scores = scores if scores is not None else pd.Series([], dtype=float)

    n_total = int(calib_count)
    winrate = float(labels.mean()) if n_total else float("nan")
    r_mean = float(r_outcome.mean()) if n_total else float("nan")
    p10 = float(r_outcome.quantile(0.1)) if n_total else float("nan")
    lift10 = _lift_at_k(scores, labels, 10)
    lift20 = _lift_at_k(scores, labels, 20)
    lift50 = _lift_at_k(scores, labels, 50)
    spearman = scores.corr(r_outcome, method="spearman") if len(scores) else float("nan")
    family_concentration_top10 = _family_concentration(scores, families, k=10)
    temporal_dispersion = _temporal_dispersion(timestamps, min_calib_samples)
    score_tail_slope = _score_tail_slope(scores, min_points=20)

    for variant in variants:
        reasons: list[str] = []
        if pd.isna(winrate) or pd.isna(r_mean) or pd.isna(p10):
            reasons.append("DIAGNOSTIC_FAIL")
        if n_total < variant.n_min:
            reasons.append("N_TOO_LOW")
        if winrate < variant.winrate_min:
            reasons.append("WINRATE_TOO_LOW")
        if r_mean < variant.r_mean_min:
            reasons.append("R_MEAN_TOO_LOW")
        if variant.p10_min is not None and p10 < variant.p10_min:
            reasons.append("P10_TOO_LOW")
        if score_tail_slope is None:
            reasons.append("TAIL_SLOPE_NA")

        qualified = len([reason for reason in reasons if reason not in {"TAIL_SLOPE_NA"}]) == 0
        baseline_n_min = baseline_thresholds.get("n_min")
        baseline_n_min = int(baseline_n_min) if baseline_n_min is not None else int(variant.n_min)
        baseline_p10 = baseline_thresholds.get("p10_min")
        relaxed_thresholds = (
            variant.n_min < baseline_n_min
            or (baseline_p10 is not None and (variant.p10_min is None or variant.p10_min < baseline_p10))
        )
        research_status = _research_status(
            qualified=qualified,
            temporal_dispersion=temporal_dispersion,
            family_concentration=family_concentration_top10,
            min_temporal_dispersion=min_temporal_dispersion,
            max_family_concentration=max_family_concentration,
            relaxed_thresholds=relaxed_thresholds,
        )

        rows.append(
            {
                "variant_id": variant.variant_id,
                "variant_type": variant.variant_type,
                "k_bars": int(variant.k_bars),
                "n_min": int(variant.n_min),
                "winrate_min": float(variant.winrate_min),
                "r_mean_min": float(variant.r_mean_min),
                "p10_min": float(variant.p10_min) if variant.p10_min is not None else None,
                "n_total": n_total,
                "n_train": int(train_count),
                "n_test": int(calib_count),
                "allow_rate": float(allow_rate) if allow_rate is not None else None,
                "winrate": winrate,
                "r_mean": r_mean,
                "p10": p10,
                "lift10": lift10,
                "lift20": lift20,
                "lift50": lift50,
                "spearman": spearman,
                "family_concentration_top10": family_concentration_top10,
                "temporal_dispersion": temporal_dispersion,
                "score_tail_slope": score_tail_slope,
                "qualified": qualified,
                "fail_reason": "|".join(reasons),
                "research_status": research_status,
            }
        )

    columns = [
        "variant_id",
        "variant_type",
        "k_bars",
        "n_min",
        "winrate_min",
        "r_mean_min",
        "p10_min",
        "n_total",
        "n_train",
        "n_test",
        "allow_rate",
        "winrate",
        "r_mean",
        "p10",
        "lift10",
        "lift20",
        "lift50",
        "spearman",
        "family_concentration_top10",
        "temporal_dispersion",
        "score_tail_slope",
        "qualified",
        "fail_reason",
        "research_status",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_family_variant_report(
    *,
    variants: list[ResearchVariant],
    scores: pd.Series | None,
    labels: pd.Series,
    r_outcome: pd.Series,
    families: pd.Series,
) -> pd.DataFrame:
    if not variants:
        return pd.DataFrame(
            columns=[
                "variant_id",
                "family",
                "n",
                "winrate",
                "r_mean",
                "p10",
                "lift10",
                "lift20",
                "qualified",
                "fail_reason",
            ]
        )
    scores = scores if scores is not None else pd.Series([], dtype=float)
    rows: list[dict[str, object]] = []
    for variant in variants:
        for family_id, fam_labels in labels.groupby(families, observed=True):
            fam_scores = scores.loc[fam_labels.index] if not scores.empty else pd.Series([], dtype=float)
            fam_r = r_outcome.loc[fam_labels.index]
            n_total = int(len(fam_labels))
            winrate = float(fam_labels.mean()) if n_total else float("nan")
            r_mean = float(fam_r.mean()) if n_total else float("nan")
            p10 = float(fam_r.quantile(0.1)) if n_total else float("nan")
            lift10 = _lift_at_k(fam_scores, fam_labels, 10)
            lift20 = _lift_at_k(fam_scores, fam_labels, 20)
            reasons: list[str] = []
            if pd.isna(winrate) or pd.isna(r_mean) or pd.isna(p10):
                reasons.append("DIAGNOSTIC_FAIL")
            if n_total < variant.n_min:
                reasons.append("N_TOO_LOW")
            if winrate < variant.winrate_min:
                reasons.append("WINRATE_TOO_LOW")
            if r_mean < variant.r_mean_min:
                reasons.append("R_MEAN_TOO_LOW")
            if variant.p10_min is not None and p10 < variant.p10_min:
                reasons.append("P10_TOO_LOW")
            qualified = len(reasons) == 0
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "family": family_id,
                    "n": n_total,
                    "winrate": winrate,
                    "r_mean": r_mean,
                    "p10": p10,
                    "lift10": lift10,
                    "lift20": lift20,
                    "qualified": qualified,
                    "fail_reason": "|".join(reasons),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ResearchVariant",
    "generate_research_variants",
    "evaluate_research_variants",
    "build_family_variant_report",
]
