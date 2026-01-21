"""Phase E (outcome-based) metrics for the State Engine."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .labels import StateLabels


def _bucketize(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    if series is None or series.empty or not series.notna().any():
        return pd.Series("NA", index=series.index if series is not None else None)
    bucketed = pd.cut(series.astype(float), bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype(object).fillna("NA")


def _split_time_terciles(index: pd.DatetimeIndex) -> list[pd.Series]:
    index_ns = index.view("i8")
    t33 = np.quantile(index_ns, 0.33)
    t66 = np.quantile(index_ns, 0.66)
    return [
        index_ns <= t33,
        (index_ns > t33) & (index_ns <= t66),
        index_ns > t66,
    ]


def _format_bin_edge(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == -float("inf"):
        return "-inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_range_labels(bins: list[float]) -> list[str]:
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == -float("inf"):
            labels.append(f"<={_format_bin_edge(right)}")
        elif right == float("inf"):
            labels.append(f">{_format_bin_edge(left)}")
        else:
            labels.append(f"{_format_bin_edge(left)}-{_format_bin_edge(right)}")
    return labels


def _coerce_bins(
    override: list[float] | tuple[float, ...] | None,
    fallback: tuple[float, ...],
    fallback_labels: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if not override:
        return fallback, fallback_labels
    bins = tuple(float(x) for x in override)
    labels = tuple(_build_range_labels(list(bins)))
    return bins, labels


def _format_rescue_table(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str],
    ordered_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns=column_map)
    available_cols = [col for col in ordered_columns if col in df.columns]
    df = df[available_cols].copy()
    float_cols = [col for col in ["ev", "p10", "p50", "p90", "delta", "pct_state"] if col in df.columns]
    if float_cols:
        df[float_cols] = df[float_cols].round(6)
    return df


def _export_rescue_csv(
    df: pd.DataFrame,
    *,
    title: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = title.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_title}.csv"
    df.to_csv(output_path, index=False)
    logger.info("rescue_scan export_csv=%s rows=%s", output_path, len(df))


def compute_ret_struct(
    ohlcv: pd.DataFrame,
    *,
    k_bars: int,
    output_index: pd.Index,
) -> pd.Series:
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    future_high = high.shift(-1).rolling(window=k_bars, min_periods=k_bars).max().shift(-(k_bars - 1))
    future_low = low.shift(-1).rolling(window=k_bars, min_periods=k_bars).min().shift(-(k_bars - 1))
    up_move = np.log(future_high / close)
    down_move = np.log(close / future_low)
    return (up_move - down_move).rename("ret_struct").reindex(output_index)


def _build_ev_extended_table(
    ev_frame: pd.DataFrame,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    ctx_features: pd.DataFrame,
    min_hvc_samples: int,
    logger: logging.Logger,
    warnings_state: dict[str, bool],
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    base = ev_frame[["ret_struct", "state_hat"]].copy()
    state_name = base["state_hat"].map(lambda v: StateLabels(v).name if not pd.isna(v) else "NA")
    base["state"] = state_name
    if "quality_label" in outputs.columns:
        base["quality_label"] = outputs["quality_label"].reindex(base.index).fillna("NA")
    else:
        base["quality_label"] = "NA"
    if "ctx_session_bucket" in ctx_features.columns and ctx_features["ctx_session_bucket"].notna().any():
        base["ctx_session_bucket"] = (
            ctx_features["ctx_session_bucket"].reindex(base.index).fillna("NA")
        )
    else:
        if not warnings_state.get("missing_session_bucket", False):
            logger.warning(
                "ctx_session_bucket missing; fallback to timestamp hour bucket (tz-naive)."
            )
            warnings_state["missing_session_bucket"] = True
        hours = pd.Series(base.index.hour, index=base.index)
        base["ctx_session_bucket"] = hours.map(
            lambda hour: "ASIA"
            if 0 <= hour <= 6
            else "LONDON"
            if 7 <= hour <= 12
            else "NY"
            if 13 <= hour <= 17
            else "NY_PM"
        )
    if "ctx_state_age" in ctx_features.columns and ctx_features["ctx_state_age"].notna().any():
        base["ctx_state_age_bucket"] = _bucketize(
            ctx_features["ctx_state_age"].reindex(base.index),
            bins=[0, 2, 5, 10, np.inf],
            labels=["0-2", "3-5", "6-10", "11+"],
        )
    else:
        base["ctx_state_age_bucket"] = "NA"
    if "ctx_dist_vwap_atr" in ctx_features.columns and ctx_features["ctx_dist_vwap_atr"].notna().any():
        base["ctx_dist_vwap_atr_bucket"] = _bucketize(
            ctx_features["ctx_dist_vwap_atr"].reindex(base.index),
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["<=0.5", "0.5-1", "1-2", ">2"],
        )
    else:
        base["ctx_dist_vwap_atr_bucket"] = "NA"

    allow_cols = [col for col in gating.columns if col.startswith("LOOK_FOR_")]
    if not allow_cols:
        if not warnings_state.get("missing_allow_cols", False):
            logger.warning("No LOOK_FOR_* columns found; skipping EV extended allow conditioning.")
            warnings_state["missing_allow_cols"] = True
        coverage = {
            "total_rows_used": int(len(base)),
            "total_rows_allowed": 0,
            "rows_before_explode": 0,
            "rows_after_explode": 0,
        }
        return pd.DataFrame(), coverage, pd.DataFrame()
    allow_df = gating[allow_cols].reindex(base.index).fillna(False).astype(bool)
    allow_rules = allow_df.apply(
        lambda row: [col for col, val in row.items() if bool(val)],
        axis=1,
    )
    allowed_mask = allow_rules.map(bool)
    before_explode_rows = int(allowed_mask.sum())
    exploded = base.loc[allowed_mask].assign(allow_rule=allow_rules.loc[allowed_mask]).explode(
        "allow_rule"
    )
    after_explode_rows = int(len(exploded))
    coverage = {
        "total_rows_used": int(len(base)),
        "total_rows_allowed": int(before_explode_rows),
        "rows_before_explode": before_explode_rows,
        "rows_after_explode": after_explode_rows,
    }
    if exploded.empty:
        return pd.DataFrame(), coverage, pd.DataFrame()

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    state_means = exploded.groupby("state")["ret_struct"].mean()

    grouped = exploded.groupby(group_cols)["ret_struct"]
    summary = grouped.agg(
        n_samples="count",
        ev_mean="mean",
        ev_p10=lambda s: s.quantile(0.1),
        ev_p50="median",
        winrate=lambda s: float((s > 0).mean() * 100.0),
    ).reset_index()
    summary["uplift_vs_state"] = summary.apply(
        lambda row: row["ev_mean"] - state_means.get(row["state"], np.nan),
        axis=1,
    )
    summary = summary[summary["n_samples"] >= min_hvc_samples].reset_index(drop=True)
    summary = summary[
        [
            "state",
            "allow_rule",
            "quality_label",
            "ctx_session_bucket",
            "ctx_state_age_bucket",
            "ctx_dist_vwap_atr_bucket",
            "n_samples",
            "ev_mean",
            "ev_p10",
            "ev_p50",
            "winrate",
            "uplift_vs_state",
        ]
    ]
    return summary, coverage, exploded


def _build_ev_extended_stability(
    exploded: pd.DataFrame,
    ev_extended: pd.DataFrame,
    ev_min_split_samples: int,
) -> pd.DataFrame:
    if exploded.empty or ev_extended.empty:
        return pd.DataFrame()

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    valid_keys = set(tuple(row) for row in ev_extended[group_cols].to_numpy())
    index = exploded.sort_index().index
    split_masks = _split_time_terciles(index)
    uplifts_by_group: dict[tuple[Any, ...], list[float]] = {}
    for mask in split_masks:
        split = exploded.loc[mask]
        if split.empty:
            continue
        state_counts = split.groupby("state")["ret_struct"].count()
        state_means = split.groupby("state")["ret_struct"].mean()
        group_counts = split.groupby(group_cols)["ret_struct"].count()
        group_means = split.groupby(group_cols)["ret_struct"].mean()
        for key, ev_mean in group_means.items():
            state_key = key[0]
            if key not in valid_keys:
                continue
            if (
                state_key not in state_means.index
                or state_counts.get(state_key, 0) < ev_min_split_samples
                or group_counts.get(key, 0) < ev_min_split_samples
            ):
                continue
            uplift = ev_mean - state_means.loc[state_key]
            uplifts_by_group.setdefault(key, []).append(float(uplift))

    rows: list[dict[str, Any]] = []
    for key, uplifts in uplifts_by_group.items():
        uplifts_arr = np.array(uplifts, dtype=float)
        rows.append(
            {
                "state": key[0],
                "quality_label": key[1],
                "allow_rule": key[2],
                "ctx_session_bucket": key[3],
                "ctx_state_age_bucket": key[4],
                "ctx_dist_vwap_atr_bucket": key[5],
                "n_splits_present": int(len(uplifts_arr)),
                "uplift_mean": float(np.mean(uplifts_arr)) if len(uplifts_arr) else np.nan,
                "uplift_std": float(np.std(uplifts_arr)) if len(uplifts_arr) else np.nan,
                "pct_splits_uplift_pos": float(np.mean(uplifts_arr > 0) * 100.0)
                if len(uplifts_arr)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _rescue_scan_tables(
    df_outputs: pd.DataFrame,
    *,
    target_state: str,
    state_col_candidates: tuple[str, ...] = ("state", "state_base"),
    quality_col: str = "quality",
    time_col: str = "time",
    split_col: str = "split",
    top_k: int = 12,
    n_min: int = 30,
    delta_max: float = 0.15,
    age_bins: tuple[float, ...] = (-float("inf"), 2, 5, 10, float("inf")),
    age_labels: tuple[str, ...] = ("0-2", "3-5", "6-10", "11+"),
    dist_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.0, 2.0, float("inf")),
    dist_labels: tuple[str, ...] = ("<=0.5", "0.5-1", "1-2", ">2"),
    atr_ratio_bins: tuple[float, ...] = (-float("inf"), 0.75, 1.0, 1.25, 1.5, float("inf")),
    atr_ratio_labels: tuple[str, ...] = ("<=0.75", "0.75-1", "1-1.25", "1.25-1.5", ">1.5"),
    break_mag_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.0, 1.5, 2.5, float("inf")),
    break_mag_labels: tuple[str, ...] = ("<=0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"),
    reentry_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.5, 2.5, 4.5, float("inf")),
    reentry_labels: tuple[str, ...] = ("0", "1", "2", "3-4", "5+"),
    confidence_cols: tuple[str, ...] = ("score_margin", "margin", "confidence"),
    logger: logging.Logger | None = None,
    emit_table: Callable[[str, pd.DataFrame, int | None], None] | None = None,
    rescue_output_dir: Path | None = None,
) -> None:
    logger = logger or logging.getLogger(__name__)
    emit_table = emit_table or (lambda _title, _df, _max: None)
    rescue_output_dir = rescue_output_dir or (Path("state_engine") / "models" / "rescue")

    def _emit(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
        _export_rescue_csv(df, title=title, output_dir=rescue_output_dir, logger=logger)
        emit_table(title, df, max_rows)

    state_col = next((col for col in state_col_candidates if col in df_outputs.columns), None)
    if state_col is None:
        logger.warning(
            "rescue_scan target=%s skipped: missing state column (candidates=%s)",
            target_state,
            state_col_candidates,
        )
        return

    total_rows = len(df_outputs)
    state_df = df_outputs.loc[df_outputs[state_col] == target_state].copy()
    n_state = len(state_df)
    pct_total = (n_state / total_rows * 100.0) if total_rows else 0.0

    if "session_bucket" in state_df.columns:
        state_df["session_bucket"] = state_df["session_bucket"].fillna("UNKNOWN")
    else:
        logger.info("rescue_scan target=%s session_bucket missing; using UNKNOWN", target_state)
        state_df["session_bucket"] = "UNKNOWN"

    if "state_age" in state_df.columns:
        state_df["state_age_bucket"] = pd.cut(
            pd.to_numeric(state_df["state_age"], errors="coerce"),
            bins=list(age_bins),
            labels=list(age_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        logger.info("rescue_scan target=%s state_age missing; using MISSING", target_state)
        state_df["state_age_bucket"] = "MISSING"

    if "dist_vwap_atr" in state_df.columns:
        state_df["dist_vwap_atr_bucket"] = pd.cut(
            pd.to_numeric(state_df["dist_vwap_atr"], errors="coerce"),
            bins=list(dist_bins),
            labels=list(dist_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        logger.info("rescue_scan target=%s dist_vwap_atr missing; using MISSING", target_state)
        state_df["dist_vwap_atr_bucket"] = "MISSING"

    if "ATR_Ratio" in state_df.columns and state_df["ATR_Ratio"].notna().any():
        state_df["atr_ratio_bucket"] = pd.cut(
            pd.to_numeric(state_df["ATR_Ratio"], errors="coerce"),
            bins=list(atr_ratio_bins),
            labels=list(atr_ratio_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["atr_ratio_bucket"] = "MISSING"

    if "BreakMag" in state_df.columns and state_df["BreakMag"].notna().any():
        state_df["breakmag_bucket"] = pd.cut(
            pd.to_numeric(state_df["BreakMag"], errors="coerce"),
            bins=list(break_mag_bins),
            labels=list(break_mag_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["breakmag_bucket"] = "MISSING"

    if "ReentryCount" in state_df.columns and state_df["ReentryCount"].notna().any():
        state_df["reentry_bucket"] = pd.cut(
            pd.to_numeric(state_df["ReentryCount"], errors="coerce"),
            bins=list(reentry_bins),
            labels=list(reentry_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["reentry_bucket"] = "MISSING"

    composition_rows = [
        {
            "bucket": "TOTAL_STATE",
            "n": n_state,
            "pct_total": pct_total,
            "pct_state": 100.0 if n_state else 0.0,
        }
    ]
    if quality_col in state_df.columns:
        quality_counts = state_df[quality_col].fillna("NA").astype(str).value_counts()
        top_quality = quality_counts.head(top_k)
        for label, count in top_quality.items():
            composition_rows.append(
                {
                    "bucket": f"quality:{label}",
                    "n": int(count),
                    "pct_total": (count / total_rows * 100.0) if total_rows else 0.0,
                    "pct_state": (count / n_state * 100.0) if n_state else 0.0,
                }
            )
        if len(quality_counts) > top_k:
            other_count = int(quality_counts.iloc[top_k:].sum())
            composition_rows.append(
                {
                    "bucket": "quality:OTHER",
                    "n": other_count,
                    "pct_total": (other_count / total_rows * 100.0) if total_rows else 0.0,
                    "pct_state": (other_count / n_state * 100.0) if n_state else 0.0,
                }
            )
    else:
        logger.info("rescue_scan target=%s skipped quality (missing %s)", target_state, quality_col)

    composition_df = pd.DataFrame(composition_rows)
    _emit(f"{target_state}_COMPOSITION", composition_df)

    if "ret_struct" not in state_df.columns:
        logger.info("rescue_scan target=%s skipped extended grid (missing ret_struct)", target_state)
        return

    state_df["ret_struct"] = pd.to_numeric(state_df["ret_struct"], errors="coerce")
    state_df_ev = state_df.dropna(subset=["ret_struct"]).copy()
    n_state_ev = len(state_df_ev)
    state_ev_mean = float(state_df_ev["ret_struct"].mean()) if n_state_ev else np.nan
    min_split_samples = max(10, int(n_min / 3))

    axes_map = {
        "BALANCE": [
            "session_bucket",
            "state_age_bucket",
            "dist_vwap_atr_bucket",
            "atr_ratio_bucket",
        ],
        "TRANSITION": [
            "session_bucket",
            "breakmag_bucket",
            "reentry_bucket",
            "state_age_bucket",
            "dist_vwap_atr_bucket",
        ],
    }
    candidate_axes = axes_map.get(
        target_state,
        ["session_bucket", "state_age_bucket", "dist_vwap_atr_bucket"],
    )
    candidate_axes = [
        axis
        for axis in candidate_axes
        if axis in state_df_ev.columns and state_df_ev[axis].notna().any()
    ]
    if not candidate_axes:
        logger.info("rescue_scan target=%s skipped extended grid (no axes)", target_state)
        return

    if len(candidate_axes) <= 3:
        axis_combos = [tuple(candidate_axes)]
    else:
        axis_combos = list(itertools.combinations(candidate_axes, 3))

    time_index: pd.DatetimeIndex | None = None
    if time_col in state_df_ev.columns:
        time_index = pd.DatetimeIndex(pd.to_datetime(state_df_ev[time_col], errors="coerce"))
    elif isinstance(state_df_ev.index, pd.DatetimeIndex):
        time_index = state_df_ev.index

    stability_counts: dict[tuple[Any, ...], int] = {}
    if time_index is not None and n_state_ev:
        split_masks = _split_time_terciles(time_index)
        for mask in split_masks:
            split = state_df_ev.loc[mask]
            split_state_n = int(split["ret_struct"].count())
            if split_state_n < min_split_samples:
                continue
            for combo in axis_combos:
                grouped = split.groupby(list(combo))["ret_struct"].agg(["count", "mean"])
                for key, row in grouped.iterrows():
                    if int(row["count"]) < min_split_samples:
                        continue
                    bucket_key = (",".join(combo),) + (key if isinstance(key, tuple) else (key,))
                    stability_counts[bucket_key] = stability_counts.get(bucket_key, 0) + 1

    rows: list[pd.DataFrame] = []
    for combo in axis_combos:
        grouped = state_df_ev.groupby(list(combo))["ret_struct"]
        summary = grouped.agg(
            n_samples="count",
            ev_mean="mean",
            p10=lambda s: s.quantile(0.10),
            p50=lambda s: s.quantile(0.50),
            p90=lambda s: s.quantile(0.90),
        ).reset_index()
        summary["pct_state"] = (summary["n_samples"] / n_state_ev * 100.0) if n_state_ev else 0.0
        summary["delta_vs_state"] = summary["ev_mean"] - state_ev_mean
        summary["axes"] = ",".join(combo)
        if stability_counts:
            def _stability_row(row: pd.Series) -> int:
                key = (summary["axes"].iloc[0],) + tuple(row[col] for col in combo)
                return stability_counts.get(key, 0)
            summary["stability_splits"] = summary.apply(_stability_row, axis=1)
        else:
            summary["stability_splits"] = 0
        summary = summary[
            ["axes", *combo, "n_samples", "pct_state", "ev_mean", "p10", "p50", "p90", "delta_vs_state", "stability_splits"]
        ]
        rows.append(summary)

    grid_extended = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not grid_extended.empty:
        grid_extended = grid_extended.sort_values(
            by=["pct_state", "n_samples"], ascending=[False, False]
        )
    column_map = {
        "axes": "axis",
        "session_bucket": "ses",
        "state_age_bucket": "age",
        "dist_vwap_atr_bucket": "dvwap",
        "breakmag_bucket": "breakmag",
        "reentry_bucket": "reentry",
        "atr_ratio_bucket": "atr_bucket",
        "n_samples": "n",
        "pct_state": "pct_state",
        "ev_mean": "ev",
        "delta_vs_state": "delta",
        "stability_splits": "stable_n",
    }
    if target_state == "TRANSITION":
        ordered_columns = [
            "axis",
            "ses",
            "age",
            "dvwap",
            "breakmag",
            "reentry",
            "n",
            "pct_state",
            "ev",
            "p10",
            "p50",
            "p90",
            "delta",
            "stable_n",
        ]
    else:
        ordered_columns = [
            "axis",
            "ses",
            "age",
            "dvwap",
            "n",
            "pct_state",
            "ev",
            "p10",
            "p50",
            "p90",
            "delta",
            "stable_n",
            "atr_bucket",
        ]
    grid_display = _format_rescue_table(
        grid_extended.sort_values(by=["n_samples", "ev_mean"], ascending=[False, False]),
        column_map=column_map,
        ordered_columns=ordered_columns,
    )
    _emit(f"{target_state}_RESCUE_GRID_EXTENDED", grid_display, max_rows=30)

    candidates = grid_extended.loc[
        (grid_extended["n_samples"] >= n_min)
        & (grid_extended["delta_vs_state"].abs() <= delta_max)
    ].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(
            by=["pct_state", "n_samples"], ascending=[False, False]
        )
    else:
        logger.info("rescue_scan target=%s no candidates meet filters", target_state)
    candidates_display = _format_rescue_table(
        candidates,
        column_map=column_map,
        ordered_columns=ordered_columns,
    )
    _emit(f"{target_state}_TOP_CANDIDATES", candidates_display, max_rows=top_k)

    if not grid_extended.empty:
        decision_df = grid_extended.copy()
        def _decision(row: pd.Series) -> str:
            if row["n_samples"] >= n_min and row["ev_mean"] > 0 and row["p10"] > 0 and abs(row["delta_vs_state"]) <= delta_max:
                return "KEEP"
            if row["n_samples"] >= n_min and (row["ev_mean"] > 0 or row["p10"] > 0):
                return "REVIEW"
            return "REJECT"
        decision_df["decision"] = decision_df.apply(_decision, axis=1)
        decision_df = decision_df.loc[
            (decision_df["n_samples"] >= n_min)
            & (decision_df["delta_vs_state"].abs() <= delta_max)
        ]
        decision_df = decision_df.sort_values(by=["n_samples", "pct_state"], ascending=[False, False])
    else:
        decision_df = pd.DataFrame(
            {"decision": ["no data"]},
        )
    decision_display = _format_rescue_table(
        decision_df,
        column_map=column_map,
        ordered_columns=[*ordered_columns, "decision"],
    )
    _emit(f"{target_state}_DECISION_TABLE", decision_display, max_rows=top_k)

    group_cols = [
        col
        for col in ["session_bucket", "state_age_bucket", "dist_vwap_atr_bucket"]
        if col in state_df.columns
    ]
    conf_col = next((col for col in confidence_cols if col in state_df.columns), None)
    if conf_col is None:
        logger.info("rescue_scan target=%s skipped confidence summary (missing)", target_state)
        return
    state_df[conf_col] = pd.to_numeric(state_df[conf_col], errors="coerce")
    if not state_df[conf_col].notna().any():
        logger.info("rescue_scan target=%s skipped confidence summary (empty)", target_state)
        return

    conf_summary = (
        state_df.groupby(group_cols, dropna=False)[conf_col]
        .agg(
            mean="mean",
            p10=lambda s: s.quantile(0.10),
            p50=lambda s: s.quantile(0.50),
            p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )
    _emit(f"{target_state}_CONFIDENCE_SUMMARY", conf_summary)


def run_phase_e_reporting(
    *,
    ohlcv: pd.DataFrame,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    ctx_features: pd.DataFrame,
    full_features: pd.DataFrame,
    label_order: list[StateLabels],
    ev_min_split_samples: int,
    diagnose_rescue_scans: bool,
    rescue_params: dict[str, Any],
    logger: logging.Logger,
    emit_table: Callable[[str, pd.DataFrame, int | None], None],
    warnings_state: dict[str, bool],
    is_verbose: bool,
    is_debug: bool,
    symbol: str,
) -> None:
    logger.warning("PHASE E METRICS ENABLED: outcome-based telemetry active.")
    ev_k_bars = rescue_params.get("ev_k_bars", 2)
    min_hvc_samples = rescue_params.get("min_hvc_samples", 100)

    ret_struct = compute_ret_struct(ohlcv, k_bars=ev_k_bars, output_index=outputs.index)
    ev_frame = pd.DataFrame(
        {
            "ret_struct": ret_struct,
            "state_hat": outputs["state_hat"],
        },
        index=outputs.index,
    ).join(gating, how="left")
    ev_frame = ev_frame.dropna(subset=["ret_struct"])

    ev_base = float(ev_frame["ret_struct"].mean()) if not ev_frame.empty else 0.0
    ev_by_state: list[dict[str, Any]] = []
    for label in label_order:
        mask_state = ev_frame["state_hat"] == label
        state_ret = ev_frame.loc[mask_state, "ret_struct"]
        ev_by_state.append(
            {
                "state": label.name,
                "n_samples": int(mask_state.sum()),
                "ev": float(state_ret.mean()) if not state_ret.empty else 0.0,
            }
        )

    ev_extended_table, ev_extended_coverage, ev_extended_exploded = _build_ev_extended_table(
        ev_frame=ev_frame,
        outputs=outputs,
        gating=gating,
        ctx_features=ctx_features,
        min_hvc_samples=min_hvc_samples,
        logger=logger,
        warnings_state=warnings_state,
    )
    ev_extended_stability = _build_ev_extended_stability(
        ev_extended_exploded,
        ev_extended_table,
        ev_min_split_samples,
    )

    ev_base_df = pd.DataFrame(
        [
            {
                "EV_BASE": ev_base,
                "k_bars": ev_k_bars,
                "n_samples": len(ev_frame),
            }
        ]
    )
    emit_table("EV_BASE", ev_base_df, None)

    ev_state_df = pd.DataFrame(ev_by_state)
    emit_table("EV_state", ev_state_df, None)

    if not ev_extended_exploded.empty:
        ev_hvc_grouped = (
            ev_extended_exploded.groupby(["state", "allow_rule"])["ret_struct"]
            .agg(n_samples="count", ev_mean="mean")
            .reset_index()
        )
        ev_state_map = {
            row["state"]: row["ev"] for row in ev_by_state
        }
        ev_hvc_grouped["uplift_vs_state"] = ev_hvc_grouped.apply(
            lambda row: row["ev_mean"] - ev_state_map.get(row["state"], np.nan),
            axis=1,
        )
        ev_hvc_grouped = ev_hvc_grouped[ev_hvc_grouped["n_samples"] >= min_hvc_samples]
    else:
        ev_hvc_grouped = pd.DataFrame(columns=["state", "allow_rule", "n_samples", "ev_mean", "uplift_vs_state"])
    emit_table("EV_HVC (state, allow)", ev_hvc_grouped, None)

    if diagnose_rescue_scans:
        diagnostic_cfg = rescue_params.get("diagnostic_cfg", {})
        diagnostic_bins = diagnostic_cfg.get("bins", {})
        if diagnostic_bins is None or not isinstance(diagnostic_bins, dict):
            diagnostic_bins = {}
        rescue_top_k = int(diagnostic_cfg.get("top_k", rescue_params.get("rescue_top_k", 12)))
        rescue_n_min = int(diagnostic_cfg.get("n_min", rescue_params.get("rescue_n_min", 30)))
        rescue_delta_max = float(diagnostic_cfg.get("delta_max", rescue_params.get("rescue_delta_max", 0.15)))
        age_bins, age_labels = _coerce_bins(
            diagnostic_bins.get("state_age"),
            (-float("inf"), 2, 5, 10, float("inf")),
            ("0-2", "3-5", "6-10", "11+"),
        )
        dist_bins, dist_labels = _coerce_bins(
            diagnostic_bins.get("dist_vwap_atr"),
            (-float("inf"), 0.5, 1.0, 2.0, float("inf")),
            ("<=0.5", "0.5-1", "1-2", ">2"),
        )
        breakmag_bins, breakmag_labels = _coerce_bins(
            diagnostic_bins.get("breakmag"),
            (-float("inf"), 0.5, 1.0, 1.5, 2.5, float("inf")),
            ("<=0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"),
        )
        reentry_bins, reentry_labels = _coerce_bins(
            diagnostic_bins.get("reentry"),
            (-float("inf"), 0.5, 1.5, 2.5, 4.5, float("inf")),
            ("0", "1", "2", "3-4", "5+"),
        )
        df_outputs = outputs.copy()
        df_outputs["state"] = outputs["state_hat"].map(
            lambda v: StateLabels(v).name if not pd.isna(v) else "NA"
        )
        df_outputs["ret_struct"] = ev_frame["ret_struct"].reindex(outputs.index)
        if "ctx_session_bucket" in ctx_features.columns:
            df_outputs["session_bucket"] = ctx_features["ctx_session_bucket"]
        if "ctx_state_age" in ctx_features.columns:
            df_outputs["state_age"] = ctx_features["ctx_state_age"]
        if "ctx_dist_vwap_atr" in ctx_features.columns:
            df_outputs["dist_vwap_atr"] = ctx_features["ctx_dist_vwap_atr"]
        for col in ("ATR_Ratio", "BreakMag", "ReentryCount"):
            if col in full_features.columns:
                df_outputs[col] = full_features[col].reindex(outputs.index)
        df_outputs["time"] = df_outputs.index
        _rescue_scan_tables(
            df_outputs,
            target_state="BALANCE",
            quality_col="quality_label",
            top_k=rescue_top_k,
            n_min=rescue_n_min,
            delta_max=rescue_delta_max,
            age_bins=age_bins,
            age_labels=age_labels,
            dist_bins=dist_bins,
            dist_labels=dist_labels,
            break_mag_bins=breakmag_bins,
            break_mag_labels=breakmag_labels,
            reentry_bins=reentry_bins,
            reentry_labels=reentry_labels,
            logger=logger,
            emit_table=emit_table,
            rescue_output_dir=Path("state_engine") / "models" / "rescue",
        )
        _rescue_scan_tables(
            df_outputs,
            target_state="TRANSITION",
            quality_col="quality_label",
            top_k=rescue_top_k,
            n_min=rescue_n_min,
            delta_max=rescue_delta_max,
            age_bins=age_bins,
            age_labels=age_labels,
            dist_bins=dist_bins,
            dist_labels=dist_labels,
            break_mag_bins=breakmag_bins,
            break_mag_labels=breakmag_labels,
            reentry_bins=reentry_bins,
            reentry_labels=reentry_labels,
            logger=logger,
            emit_table=emit_table,
            rescue_output_dir=Path("state_engine") / "models" / "rescue",
        )

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    if not ev_extended_stability.empty:
        stability_ranked = ev_extended_stability.merge(
            ev_extended_table[group_cols + ["n_samples"]],
            on=group_cols,
            how="left",
        )
        stability_ranked["n_splits_present"] = stability_ranked["n_splits_present"].fillna(0)
        stability_ranked["pct_splits_uplift_pos"] = stability_ranked["pct_splits_uplift_pos"].fillna(0.0)
        stability_ranked["uplift_mean"] = stability_ranked["uplift_mean"].fillna(0.0)
        stability_ranked["n_samples"] = stability_ranked["n_samples"].fillna(0)
        stability_ranked = stability_ranked.sort_values(
            by=[
                "n_splits_present",
                "pct_splits_uplift_pos",
                "uplift_mean",
                "n_samples",
            ],
            ascending=[False, False, False, False],
        )
        top_stability = stability_ranked.head(20)
        top_extended = ev_extended_table.merge(
            stability_ranked[group_cols],
            on=group_cols,
            how="inner",
        )
        top_extended = top_extended.set_index(group_cols).loc[
            top_stability.set_index(group_cols).index
        ].reset_index()
    else:
        top_stability = ev_extended_stability
        top_extended = ev_extended_table.head(20)

    if not ev_extended_table.empty:
        emit_table("EV_EXTENDED_TABLE (top 20)", top_extended, 20)
        if not top_stability.empty:
            emit_table("EV_EXTENDED_STABILITY (top 20)", top_stability, 20)

    coverage_rows = [
        {
            "total_rows_used": ev_extended_coverage.get("total_rows_used", 0),
            "total_rows_allowed": ev_extended_coverage.get("total_rows_allowed", 0),
            "rows_before_explode": ev_extended_coverage.get("rows_before_explode", 0),
            "rows_after_explode": ev_extended_coverage.get("rows_after_explode", 0),
            "coverage_before_explode_pct": (
                ev_extended_coverage.get("rows_before_explode", 0)
                / max(ev_extended_coverage.get("total_rows_used", 1), 1)
                * 100.0
            ),
            "coverage_after_explode_pct": (
                ev_extended_coverage.get("rows_after_explode", 0)
                / max(ev_extended_coverage.get("total_rows_used", 1), 1)
                * 100.0
            ),
        }
    ]
    emit_table("COVERAGE_SUMMARY", pd.DataFrame(coverage_rows), None)

    if is_debug:
        state_cov_df = pd.DataFrame(
            ev_frame["state_hat"].map(lambda v: StateLabels(v).name).value_counts()
        ).reset_index()
        state_cov_df.columns = ["state_hat", "count"]
        emit_table("COVERAGE_STATE", state_cov_df, None)

        if "quality_label" in outputs.columns:
            quality_cov_df = outputs.loc[ev_frame.index, "quality_label"].value_counts().reset_index()
            quality_cov_df.columns = ["quality_label", "count"]
            emit_table("COVERAGE_QUALITY", quality_cov_df, None)

        if not ev_extended_exploded.empty:
            allow_cov_df = ev_extended_exploded["allow_rule"].value_counts().reset_index()
            allow_cov_df.columns = ["allow_rule", "count"]
            emit_table("COVERAGE_LOOK_FOR", allow_cov_df, None)

        if "ctx_session_bucket" in ctx_features.columns:
            session_cov_df = ctx_features.loc[ev_frame.index, "ctx_session_bucket"].value_counts().reset_index()
            session_cov_df.columns = ["ctx_session_bucket", "count"]
            emit_table("COVERAGE_SESSION", session_cov_df, None)

    if is_verbose:
        if "ctx_dist_vwap_atr" in ctx_features.columns:
            dist_series = pd.to_numeric(
                ctx_features.loc[ev_frame.index, "ctx_dist_vwap_atr"], errors="coerce"
            )
            vwap_summary = pd.DataFrame(
                [
                    {
                        "nan_pct": float(dist_series.isna().mean() * 100.0),
                        "dist_nan_pct": float(dist_series.isna().mean() * 100.0),
                        "dist_p50": float(dist_series.quantile(0.5)) if dist_series.notna().any() else np.nan,
                        "dist_p90": float(dist_series.quantile(0.9)) if dist_series.notna().any() else np.nan,
                    }
                ]
            )
            emit_table("VWAP_VALIDATION (summary)", vwap_summary, None)

    logger.info("PHASE E telemetry complete for symbol=%s", symbol)
