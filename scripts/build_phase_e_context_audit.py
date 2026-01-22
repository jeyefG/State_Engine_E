"""Build Phase E context audit + broadcast coverage summary (descriptive)."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_engine.config_loader import load_config
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline_phase_d import build_context_bundle
from state_engine.quality import (
    QUALITY_LABEL_UNCLASSIFIED,
    assign_quality_labels,
    load_quality_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase E context audit + intraday coverage summary (descriptive)."
    )
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. XAUUSD.mg)")
    parser.add_argument("--start", required=True, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument(
        "--context-tf",
        required=True,
        choices=["H1", "H2"],
        help="Timeframe base del State Engine (H1 o H2)",
    )
    parser.add_argument(
        "--score-tf",
        required=True,
        choices=["M5", "M15"],
        help="Timeframe intradía para broadcast causal (M5 o M15)",
    )
    parser.add_argument(
        "--window-hours",
        required=True,
        type=int,
        help="Ventana temporal (horas) para features/quality en TF base",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "state_engine" / "models"),
        help="Ruta base de modelos .pkl (el archivo se resuelve por símbolo)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Ruta de salida CSV (phase_e_audit_*.csv)",
    )
    parser.add_argument(
        "--cache-parquet",
        default=None,
        help="Ruta opcional para cache del merge (parquet)",
    )
    parser.add_argument(
        "--fail-on-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abortar si falla auditoría (default: true)",
    )
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("phase_e_context_audit")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def load_symbol_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "symbols" / f"{symbol}.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("symbol config load failed for %s: %s", symbol, exc)
        return {}
    return config if isinstance(config, dict) else {}


def normalize_state_labels(state_hat: pd.Series) -> pd.Series:
    mapping = {
        StateLabels.BALANCE: "BALANCE",
        StateLabels.TRANSITION: "TRANSITION",
        StateLabels.TREND: "TREND",
        int(StateLabels.BALANCE): "BALANCE",
        int(StateLabels.TRANSITION): "TRANSITION",
        int(StateLabels.TREND): "TREND",
        "balance": "BALANCE",
        "transition": "TRANSITION",
        "trend": "TREND",
        "BALANCE": "BALANCE",
        "TRANSITION": "TRANSITION",
        "TREND": "TREND",
    }
    normalized = state_hat.map(mapping)
    return normalized.fillna("UNKNOWN")


def look_for_base_state_map(symbol_cfg: dict | None) -> dict[str, str]:
    if not isinstance(symbol_cfg, dict):
        return {}
    phase_d = symbol_cfg.get("phase_d")
    if not isinstance(phase_d, dict):
        return {}
    look_for_cfg = phase_d.get("look_fors")
    if not isinstance(look_for_cfg, dict):
        return {}
    base_state_map: dict[str, str] = {}
    for look_for_name, rule_cfg in look_for_cfg.items():
        if not isinstance(rule_cfg, dict):
            continue
        base_state = rule_cfg.get("base_state") or rule_cfg.get("anchor_state")
        if base_state is None:
            continue
        base_state_map[look_for_name] = str(base_state).strip().lower()
    return base_state_map


def infer_base_state_from_name(look_for_name: str) -> str | None:
    name = look_for_name.lower()
    for candidate in ("balance", "transition", "trend"):
        if f"look_for_{candidate}" in name:
            return candidate
    return None


def ensure_quality_label_full(ctx_df: pd.DataFrame, base_state: pd.Series) -> pd.Series:
    if "quality_label_full" in ctx_df.columns:
        return ctx_df["quality_label_full"].astype(str)
    if "quality_label" in ctx_df.columns:
        quality_series = ctx_df["quality_label"].astype(str)
    else:
        quality_series = pd.Series(QUALITY_LABEL_UNCLASSIFIED, index=ctx_df.index, dtype=str)
    return quality_series.where(
        (quality_series.notna()) & (quality_series != QUALITY_LABEL_UNCLASSIFIED),
        base_state + f"_{QUALITY_LABEL_UNCLASSIFIED}",
    )


def audit_phase_d_base(
    ctx_df: pd.DataFrame,
    base_state: pd.Series,
    *,
    symbol_cfg: dict | None,
    logger: logging.Logger,
) -> list[str]:
    failures: list[str] = []

    if "state_hat" not in ctx_df.columns:
        failures.append("missing state_hat in base context")
    if "margin" not in ctx_df.columns:
        failures.append("missing margin in base context")

    if not ctx_df.empty:
        first_state = ctx_df["state_hat"].iloc[0] if "state_hat" in ctx_df.columns else None
        first_margin = ctx_df["margin"].iloc[0] if "margin" in ctx_df.columns else None
        if pd.isna(first_state):
            failures.append("state_hat is NaN in first base row")
        if pd.isna(first_margin):
            failures.append("margin is NaN in first base row")

    look_for_cols = [col for col in ctx_df.columns if col.startswith("LOOK_FOR_")]
    base_state_map = look_for_base_state_map(symbol_cfg)
    look_for_mismatches: dict[str, int] = {}
    for look_for_name in look_for_cols:
        expected = base_state_map.get(look_for_name) or infer_base_state_from_name(look_for_name)
        if expected is None or expected == "any":
            continue
        expected_norm = expected.strip().upper()
        active = ctx_df[look_for_name] == 1
        known = base_state != "UNKNOWN"
        mismatches = active & known & (base_state != expected_norm)
        mismatch_count = int(mismatches.sum())
        if mismatch_count:
            look_for_mismatches[look_for_name] = mismatch_count
    if look_for_mismatches:
        detail = "; ".join(
            f"{name} mismatches={count}" for name, count in sorted(look_for_mismatches.items())
        )
        failures.append(f"LOOK_FOR mismatches vs base_state: {detail}")

    quality_label_full = ensure_quality_label_full(ctx_df, base_state)
    quality_prefix = quality_label_full.str.extract(
        r"^(BALANCE|TRANSITION|TREND)", expand=False
    )
    known = base_state != "UNKNOWN"
    mismatches = known & quality_prefix.notna() & (quality_prefix != base_state)
    pct_bad = float(mismatches.mean()) if len(mismatches) else 0.0
    if pct_bad > 0.0:
        failures.append(f"quality/base_state mismatches pct_bad={pct_bad * 100:.3f}%")

    if failures:
        for failure in failures:
            logger.error("AUDIT_FAIL %s", failure)
    else:
        logger.info("AUDIT_OK Phase D base context")
    return failures


def broadcast_causal(
    ctx_df: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    *,
    logger: logging.Logger,
) -> pd.DataFrame:
    ctx = ctx_df.copy().sort_index()
    score = ohlcv_score.copy().sort_index()
    if getattr(ctx.index, "tz", None) is not None:
        ctx.index = ctx.index.tz_localize(None)
    if getattr(score.index, "tz", None) is not None:
        score.index = score.index.tz_localize(None)
    ctx = ctx.reset_index().rename(columns={ctx.index.name or "index": "time"})
    score = score.reset_index().rename(columns={score.index.name or "index": "time"})
    merged = pd.merge_asof(score, ctx, on="time", direction="backward", allow_exact_matches=True)
    merged = merged.set_index("time")
    look_for_cols = [col for col in merged.columns if col.startswith("LOOK_FOR_")]
    if look_for_cols:
        merged[look_for_cols] = merged[look_for_cols].fillna(0).astype(int)
    missing_ctx = merged[["state_hat", "margin"]].isna().mean()
    if (missing_ctx > 0.25).any():
        logger.warning("High missing context after broadcast: %s", missing_ctx.to_dict())
    return merged


def summarize_coverage(
    merged: pd.DataFrame,
    *,
    symbol: str,
    score_tf: str,
) -> pd.DataFrame:
    look_for_cols = [col for col in merged.columns if col.startswith("LOOK_FOR_")]
    summaries: list[pd.DataFrame] = []

    if not look_for_cols:
        merged = merged.copy()
        merged["LOOK_FOR_NONE"] = 1
        look_for_cols = ["LOOK_FOR_NONE"]

    if "base_state" not in merged.columns:
        merged = merged.copy()
        merged["base_state"] = normalize_state_labels(merged["state_hat"])
    if "quality_label_full" not in merged.columns:
        merged = merged.copy()
        merged["quality_label_full"] = ensure_quality_label_full(
            merged,
            normalize_state_labels(merged["state_hat"]),
        )

    for look_for_name in look_for_cols:
        segment = merged[merged[look_for_name] == 1].copy()
        if segment.empty:
            continue
        segment["symbol"] = symbol
        segment["score_tf"] = score_tf
        segment["look_for_rule"] = look_for_name
        segment["event_day"] = segment.index.normalize()

        group_cols = [
            "symbol",
            "score_tf",
            "base_state",
            "quality_label_full",
            "look_for_rule",
        ]
        agg_map = {
            "event_day": "nunique",
            "margin": "mean",
        }
        if "ctx_dist_vwap_atr" in segment.columns:
            segment["nan_ctx_dist_vwap_atr"] = segment["ctx_dist_vwap_atr"].isna()
            agg_map["nan_ctx_dist_vwap_atr"] = "mean"

        grouped = segment.groupby(group_cols).agg(agg_map).reset_index()
        event_counts = segment.groupby(group_cols).size().reset_index(name="n_events")
        grouped = grouped.merge(event_counts, on=group_cols, how="left")
        grouped = grouped.rename(
            columns={
                "event_day": "n_days",
                "margin": "avg_margin",
                "nan_ctx_dist_vwap_atr": "nan_rate_ctx_dist_vwap_atr",
            }
        )
        grouped["events_per_day"] = grouped["n_events"] / grouped["n_days"]
        if "nan_rate_ctx_dist_vwap_atr" not in grouped.columns:
            grouped["nan_rate_ctx_dist_vwap_atr"] = pd.NA
        summaries.append(grouped)

    if not summaries:
        return pd.DataFrame(
            columns=[
                "symbol",
                "score_tf",
                "base_state",
                "quality_label_full",
                "look_for_rule",
                "n_events",
                "n_days",
                "events_per_day",
                "avg_margin",
                "nan_rate_ctx_dist_vwap_atr",
            ]
        )
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values("n_events", ascending=False)
    return summary


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    symbol = args.symbol
    context_tf = args.context_tf.upper()
    score_tf = args.score_tf.upper()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    bars_per_hour = {"H1": 1, "H2": 0.5}
    derived_bars = math.ceil(args.window_hours * bars_per_hour[context_tf])
    if derived_bars < 4:
        raise ValueError(f"Context window too small: {derived_bars} bars (min=4).")

    logger.info(
        "window_hours=%s context_tf=%s score_tf=%s derived_bars=%s",
        args.window_hours,
        context_tf,
        score_tf,
        derived_bars,
    )

    connector = MT5Connector()
    logger.info("download base start symbol=%s tf=%s", symbol, context_tf)
    ohlcv_ctx = connector.obtener_ohlcv(symbol, context_tf, start, end)
    logger.info("download base done rows=%s", len(ohlcv_ctx))

    model_root = Path(args.model_path)
    if model_root.is_dir() or model_root.suffix.lower() != ".pkl":
        model_path = model_root / f"{safe_filename(symbol)}_state_engine.pkl"
    else:
        model_path = model_root
    logger.info("predict start model_path=%s", model_path)
    model = StateEngineModel()
    model.load(model_path)

    feature_engineer = FeatureEngineer(FeatureConfig(window=derived_bars))

    logger.info("quality start")
    quality_config, _, quality_warnings = load_quality_config(symbol)
    if quality_warnings:
        logger.warning("quality config warnings: %s", quality_warnings)
    full_features = feature_engineer.compute_features(ohlcv_ctx)
    features = feature_engineer.training_features(full_features)
    outputs = model.predict_outputs(features)
    quality_labels, quality_assign_warnings = assign_quality_labels(
        outputs["state_hat"],
        full_features.reindex(outputs.index),
        quality_config,
    )
    if quality_assign_warnings:
        logger.warning("quality assign warnings: %s", quality_assign_warnings)
    logger.info("quality done")

    logger.info("build Phase D base context")
    symbol_cfg = load_symbol_config(symbol, logger)
    gating_policy = GatingPolicy()
    context_bundle = build_context_bundle(
        symbol=symbol,
        context_tf=context_tf,
        score_tf=context_tf,
        ohlcv_ctx=ohlcv_ctx,
        ohlcv_score=ohlcv_ctx,
        state_model=model,
        feature_engineer=feature_engineer,
        gating_policy=gating_policy,
        symbol_cfg=symbol_cfg,
        phase_e=False,
        logger=logger,
        quality_labels=quality_labels,
        causal_shift=False,
    )
    ctx_df_base = context_bundle.ctx_df.copy()

    base_state = normalize_state_labels(ctx_df_base["state_hat"])
    ctx_df_base["base_state"] = base_state
    ctx_df_base["quality_label_full"] = ensure_quality_label_full(ctx_df_base, base_state)

    failures = audit_phase_d_base(
        ctx_df_base,
        base_state,
        symbol_cfg=symbol_cfg,
        logger=logger,
    )
    if failures and args.fail_on_audit:
        raise SystemExit("Audit failed; aborting due to --fail-on-audit.")

    logger.info("download score start symbol=%s tf=%s", symbol, score_tf)
    ohlcv_score = connector.obtener_ohlcv(symbol, score_tf, start, end)
    logger.info("download score done rows=%s", len(ohlcv_score))

    logger.info("broadcast causal context")
    merged = broadcast_causal(ctx_df_base, ohlcv_score, logger=logger)
    merged["base_state"] = normalize_state_labels(merged["state_hat"])
    merged["quality_label_full"] = ensure_quality_label_full(
        merged,
        merged["base_state"],
    )

    if args.cache_parquet:
        cache_path = Path(args.cache_parquet)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.reset_index().to_parquet(cache_path, index=False)
        logger.info("cache parquet written path=%s", cache_path)

    summary = summarize_coverage(merged, symbol=symbol, score_tf=score_tf)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info("summary export done path=%s rows=%s", output_path, len(summary))


if __name__ == "__main__":
    main()
