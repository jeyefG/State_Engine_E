"""Lightweight parameter sweep utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class SweepResult:
    params: dict[str, float]
    metrics: dict[str, float]


def run_param_sweep(
    pipeline_fn: Callable[[dict[str, float]], dict[str, float]],
    grid: dict[str, Iterable[float]],
) -> pd.DataFrame:
    """Run a lightweight parameter sweep and return results DataFrame."""
    keys = list(grid.keys())
    combos = list(product(*[grid[key] for key in keys]))
    results: list[dict[str, float]] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        metrics = pipeline_fn(params)
        results.append({**params, **metrics})
    return pd.DataFrame(results)


__all__ = ["SweepResult", "run_param_sweep"]
