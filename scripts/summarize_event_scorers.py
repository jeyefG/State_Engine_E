#!/usr/bin/env python3
"""Summarize Event Scorer artifacts across symbols."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRIC_FIELDS = {
    "auc": {"auc"},
    "lift20": {"lift20"},
    "r_mean20": {"rmean20"},
    "spearman": {"spearman"},
}


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(df: pd.DataFrame, targets: set[str]) -> str | None:
    for col in df.columns:
        if _normalize_column(col) in targets:
            return col
    return None


def _parse_symbol(path: Path, prefix: str, suffix: str) -> str | None:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def _select_row_by_scope(df: pd.DataFrame, scope_values: set[str]) -> pd.Series | None:
    scope_cols = [col for col in df.columns if _normalize_column(col) in {"scope", "block", "dataset"}]
    model_cols = [col for col in df.columns if _normalize_column(col) == "model"]

    scoped = pd.DataFrame()
    for col in scope_cols:
        mask = df[col].astype(str).str.upper().isin(scope_values)
        if mask.any():
            scoped = df.loc[mask]
            break

    candidates = scoped if not scoped.empty else df
    if not model_cols:
        return candidates.iloc[0] if not candidates.empty else None
    model_col = model_cols[0]
    scorer_rows = candidates[candidates[model_col].astype(str).str.upper() == "SCORER"]
    if not scorer_rows.empty:
        return scorer_rows.iloc[0]
    return candidates.iloc[0] if not candidates.empty else None


def _extract_metrics(metrics_path: Path | None) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {
        "events_total_post_meta": None,
        "events_per_day": None,
        "unique_days": None,
        "auc_no_meta": None,
        "auc_meta": None,
        "lift20_meta": None,
        "r_mean20_meta": None,
        "spearman_meta": None,
        "recommendation": None,
        "split_warning_hits_global": None,
    }
    if metrics_path is None or not metrics_path.exists():
        return result

    df = pd.read_csv(metrics_path)
    if df.empty:
        return result

    normalized_cols = {_normalize_column(col): col for col in df.columns}
    for key in ("events_total_post_meta", "events_per_day", "unique_days", "split_warning_hits_global"):
        col = normalized_cols.get(_normalize_column(key))
        if col is not None:
            result[key] = df[col].iloc[0]

    rec_col = normalized_cols.get("recommendation")
    if rec_col is not None:
        result["recommendation"] = df[rec_col].iloc[0]

    direct_map = {
        "auc_no_meta": "aucnometa",
        "auc_meta": "aucmeta",
        "lift20_meta": "lift20meta",
        "r_mean20_meta": "rmean20meta",
        "spearman_meta": "spearmanmeta",
    }
    for dest, norm_key in direct_map.items():
        if norm_key in normalized_cols:
            result[dest] = df[normalized_cols[norm_key]].iloc[0]

    if all(result[key] is None for key in ("auc_no_meta", "auc_meta", "lift20_meta", "r_mean20_meta", "spearman_meta")):
        meta_row = _select_row_by_scope(df, {"META", "POST_META"})
        no_meta_row = _select_row_by_scope(df, {"NO_META", "PRE_META"})
        if meta_row is not None:
            result["auc_meta"] = meta_row.get(_find_column(df, METRIC_FIELDS["auc"]))
            result["lift20_meta"] = meta_row.get(_find_column(df, METRIC_FIELDS["lift20"]))
            result["r_mean20_meta"] = meta_row.get(_find_column(df, METRIC_FIELDS["r_mean20"]))
            result["spearman_meta"] = meta_row.get(_find_column(df, METRIC_FIELDS["spearman"]))
        if no_meta_row is not None:
            result["auc_no_meta"] = no_meta_row.get(_find_column(df, METRIC_FIELDS["auc"]))

    return result


def _extract_family_summary(family_path: Path | None) -> dict[str, int | str | None]:
    result: dict[str, int | str | None] = {"families_trained": None}
    if family_path is None or not family_path.exists():
        return result
    df = pd.read_csv(family_path)
    if df.empty:
        result["families_trained"] = 0
        return result
    status_col = _find_column(df, {"status"})
    family_col = _find_column(df, {"familyid", "family"})
    if status_col is not None:
        trained = df[status_col].astype(str).str.upper() == "TRAINED"
        result["families_trained"] = int(trained.sum())
        return result
    if family_col is not None:
        result["families_trained"] = int(df[family_col].nunique())
    return result


def _extract_calib_summary(calib_path: Path | None) -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {
        "best_regime_id": None,
        "worst_regime_id": None,
        "any_low_samples_flag": None,
    }
    if calib_path is None or not calib_path.exists():
        return result
    df = pd.read_csv(calib_path)
    if df.empty:
        return result

    regime_col = _find_column(df, {"regimeid", "regime"})
    if regime_col is not None:
        result["best_regime_id"] = df[regime_col].iloc[0]
        result["worst_regime_id"] = df[regime_col].iloc[-1]

    flag_col = _find_column(df, {"flag", "flags"})
    if flag_col is not None:
        flag_series = df[flag_col].astype(str)
        result["any_low_samples_flag"] = bool(flag_series.str.contains("LOW_SAMPLES", na=False).any())

    return result


def _extract_summary_json(summary_path: Path | None) -> dict[str, object | None]:
    fields = {
        "symbol": None,
        "start": None,
        "end": None,
        "h1_cutoff": None,
        "m5_cutoff": None,
        "events_total_post_meta": None,
        "events_per_day": None,
        "unique_days": None,
        "auc_no_meta": None,
        "auc_meta": None,
        "lift20_meta": None,
        "r_mean20_meta": None,
        "spearman_meta": None,
        "families_trained": None,
        "best_regime_id": None,
        "worst_regime_id": None,
        "recommendation": None,
        "split_warning_hits_global": None,
        "allow_active_pct": None,
        "feature_count": None,
        "min_samples_train": None,
    }
    if summary_path is None or not summary_path.exists():
        return fields
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in fields:
        if key in payload:
            fields[key] = payload.get(key)
    return fields


def _format_float(value: float | int | None, decimals: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{decimals}f}"


def _format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(value)}"


def _prefer(primary: object | None, fallback: object | None) -> object | None:
    if primary is None or (isinstance(primary, float) and pd.isna(primary)):
        return fallback
    return primary


def _build_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(row.get(col, "")))
    header = " ".join(col.ljust(widths[col]) for col in columns)
    lines = [header]
    for row in rows:
        lines.append(" ".join(row.get(col, "").ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def _collect_symbols(models_dir: Path) -> dict[str, dict[str, Path]]:
    patterns = {
        "metrics": "metrics_*_event_scorer.csv",
        "family": "family_summary_*_event_scorer.csv",
        "calib": "calib_top_scored_*_event_scorer.csv",
        "summary": "summary_*_event_scorer.json",
    }
    symbols: dict[str, dict[str, Path]] = {}
    for key, pattern in patterns.items():
        for path in models_dir.glob(pattern):
            symbol = _parse_symbol(path, pattern.split("*")[0], pattern.split("*")[1])
            if symbol is None:
                continue
            symbols.setdefault(symbol, {})[key] = path
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Event Scorer metrics across symbols.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--models-dir", type=Path, default=repo_root / "state_engine" / "models")
    parser.add_argument("--sort", default="r_mean20_meta")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    symbols = _collect_symbols(args.models_dir)
    rows: list[dict[str, object]] = []

    for symbol in sorted(symbols.keys()):
        artifacts = symbols[symbol]
        summary = _extract_summary_json(artifacts.get("summary"))
        metrics = _extract_metrics(artifacts.get("metrics"))
        families = _extract_family_summary(artifacts.get("family"))
        calib = _extract_calib_summary(artifacts.get("calib"))

        row = {
            "symbol": symbol,
            "events_total_post_meta": _prefer(summary["events_total_post_meta"], metrics["events_total_post_meta"]),
            "events_per_day": _prefer(summary["events_per_day"], metrics["events_per_day"]),
            "unique_days": _prefer(summary["unique_days"], metrics["unique_days"]),
            "families_trained": _prefer(summary["families_trained"], families["families_trained"]),
            "auc_no_meta": _prefer(summary["auc_no_meta"], metrics["auc_no_meta"]),
            "auc_meta": _prefer(summary["auc_meta"], metrics["auc_meta"]),
            "lift20_meta": _prefer(summary["lift20_meta"], metrics["lift20_meta"]),
            "r_mean20_meta": _prefer(summary["r_mean20_meta"], metrics["r_mean20_meta"]),
            "spearman_meta": _prefer(summary["spearman_meta"], metrics["spearman_meta"]),
            "best_regime_id": _prefer(summary["best_regime_id"], calib["best_regime_id"]),
            "worst_regime_id": _prefer(summary["worst_regime_id"], calib["worst_regime_id"]),
            "recommendation": _prefer(summary["recommendation"], metrics["recommendation"]),
            "any_low_samples_flag": calib["any_low_samples_flag"],
            "split_warning_hits_global": _prefer(
                summary["split_warning_hits_global"],
                metrics["split_warning_hits_global"],
            ),
        }

        missing_fields = [
            key
            for key, value in row.items()
            if key != "symbol" and (value is None or (isinstance(value, float) and pd.isna(value)))
        ]
        row["missing_fields"] = ",".join(missing_fields) if missing_fields else ""
        rows.append(row)

    if not rows:
        print("EVENT SCORER | SUMMARY (0 symbols)")
        return

    df = pd.DataFrame(rows)
    sort_key = args.sort
    if sort_key in df.columns:
        df = df.sort_values([sort_key, "auc_meta"], ascending=[False, False], na_position="last")

    if args.top is not None:
        df = df.head(args.top)

    console_rows = []
    for _, row in df.iterrows():
        console_rows.append(
            {
                "symbol": str(row["symbol"]),
                "events": _format_int(row["events_total_post_meta"]),
                "/day": _format_float(row["events_per_day"], 1),
                "aucM": _format_float(row["auc_meta"], 3),
                "lift20M": _format_float(row["lift20_meta"], 3),
                "r20M": _format_float(row["r_mean20_meta"], 2),
                "sprM": _format_float(row["spearman_meta"], 3),
                "families": _format_int(row["families_trained"]),
                "recommendation": "NA" if pd.isna(row["recommendation"]) else str(row["recommendation"]),
            }
        )

    print(f"EVENT SCORER | SUMMARY ({len(df)} symbols)")
    print(_build_table(console_rows, ["symbol", "events", "/day", "aucM", "lift20M", "r20M", "sprM", "families", "recommendation"]))

    expected_symbols = {
        path.name[: -len("_event_scorer.pkl")]
        for path in args.models_dir.glob("*_event_scorer.pkl")
    }
    missing_summaries = expected_symbols - set(symbols.keys())
    if missing_summaries:
        print(f"MISSING_SUMMARIES={len(missing_summaries)}")

    output_dir = args.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "event_scorer_summary.csv"
    json_path = output_dir / "event_scorer_summary.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)


if __name__ == "__main__":
    main()
