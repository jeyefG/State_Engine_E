"""Walk-forward utilities for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calib_start: pd.Timestamp
    calib_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walkforward_splits(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    calib_days: int,
    test_days: int,
    step_days: int,
) -> list[WalkForwardSplit]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts >= end_ts:
        return []
    if train_days <= 0 or calib_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("Walk-forward windows must be positive.")

    train_delta = pd.Timedelta(days=train_days)
    calib_delta = pd.Timedelta(days=calib_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    splits: list[WalkForwardSplit] = []
    cursor = start_ts
    idx = 0
    while True:
        train_start = cursor
        train_end = train_start + train_delta
        calib_start = train_end
        calib_end = calib_start + calib_delta
        test_start = calib_end
        test_end = test_start + test_delta
        if test_end > end_ts:
            break
        splits.append(
            WalkForwardSplit(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                calib_start=calib_start,
                calib_end=calib_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        idx += 1
        cursor = cursor + step_delta
    return splits


def margin_bins(series: pd.Series, q: int = 3) -> pd.Series:
    bins = pd.Series(index=series.index, dtype="object")
    non_na = series.dropna()
    if non_na.empty:
        return bins
    try:
        bins.loc[non_na.index] = pd.qcut(non_na, q=q, duplicates="drop")
    except ValueError:
        rank_pct = non_na.rank(pct=True)
        bins.loc[non_na.index] = pd.cut(rank_pct, bins=q, include_lowest=True)
    return bins


def apply_edge_ablation(
    events: pd.DataFrame,
    scores: pd.Series,
    ablation: str,
    seed: int = 7,
    margin_bin_q: int = 3,
) -> pd.Series:
    if ablation == "real":
        return scores
    if ablation == "constant":
        return pd.Series(0.5, index=scores.index, name="edge_score")
    if ablation != "shuffle":
        raise ValueError("ablation must be one of: real, shuffle, constant")

    rng = np.random.default_rng(seed)
    bins = margin_bins(events["margin_H1"], q=margin_bin_q)
    shuffled = scores.copy()
    grouped = events.assign(margin_bin=bins).groupby(["family_id", "state_hat_H1", "margin_bin"], dropna=False)
    for _, group in grouped:
        if group.empty:
            continue
        idx = group.index
        if len(idx) < 2:
            shuffled.loc[idx] = scores.loc[idx]
            continue
        shuffled.loc[idx] = rng.permutation(scores.loc[idx].to_numpy())
    return shuffled


__all__ = [
    "WalkForwardSplit",
    "generate_walkforward_splits",
    "margin_bins",
    "apply_edge_ablation",
]
