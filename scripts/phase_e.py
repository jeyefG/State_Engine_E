"""Build canonical Phase D context at the State Engine base timeframe and continuing with Fase E"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

import sys

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_engine.config_loader import load_config
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline_phase_d import audit_phase_d_contamination, build_context_bundle
from state_engine.pipeline_phase_d import _merge_context_score as merge_context_intraday
from state_engine.pipeline_phase_d import _normalize_state_labels
from state_engine.quality import (
    assign_quality_labels,
    default_quality_config_dict,
    validate_quality_config_dict,
    _config_from_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase D context (TF base).")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. XAUUSD.mg)")
    parser.add_argument("--start", required=True, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument(
        "--timeframe",
        required=True,
        choices=["H1", "H2"],
        help="Timeframe base del State Engine (H1 o H2)",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "state_engine" / "models"),
        help="Ruta base de modelos .pkl (el archivo se resuelve por símbolo)",
    )
    parser.add_argument(
        "--window-hours",
        required=True,
        type=int,
        help="Ventana temporal (horas) para features/quality en TF base",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "phase_d_context"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--phase-e",
        action="store_true",
        help="Continuar con Phase E usando ctx_df en memoria",
    )
    parser.add_argument(
        "--score-tf",
        default=None,
        help="Timeframe intradía para Phase E (ej. M5/M15).",
    )
    parser.add_argument(
        "--out-phase-e",
        default=None,
        help="Ruta base de salida para Phase E (sin sufijo).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta base de salida compartida (Phase E usa sufijos _universe/_lookfor/_coverage).",
    )
    parser.add_argument(
        "--fail-on-audit",
        action="store_true",
        help="Abortar si falla la auditoría canónica de Phase D.",
    )
    parser.add_argument(
        "--debug-allow-missing-quality",
        action="store_true",
        help="DEBUG: permitir quality_labels=None (no recomendado).",
    )
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("phase_d_context")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_symbol_config(symbol: str, logger: logging.Logger) -> dict:
    config_path = PROJECT_ROOT / "configs" / "symbols" / f"{symbol}.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("symbol config load failed for %s: %s", symbol, exc)
        return {}
    return config if isinstance(config, dict) else {}


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _deep_merge_dicts(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_quality_config(symbol_cfg: dict, logger: logging.Logger):
    defaults = default_quality_config_dict()
    quality_cfg = None
    if isinstance(symbol_cfg, dict):
        phase_c_cfg = symbol_cfg.get("phase_c")
        if isinstance(phase_c_cfg, dict) and isinstance(phase_c_cfg.get("quality"), dict):
            quality_cfg = phase_c_cfg.get("quality")
        elif isinstance(symbol_cfg.get("quality"), dict):
            quality_cfg = symbol_cfg.get("quality")
    if quality_cfg is None:
        logger.info("quality config: using defaults (no symbol override).")
        merged = defaults
    else:
        validate_quality_config_dict(quality_cfg)
        merged = _deep_merge_dicts(defaults, quality_cfg)
    return _config_from_dict(merged)


def _run_phase_d_audit(logger: logging.Logger, *, fail_on_audit: bool) -> None:
    audit_handler = logging.Handler()
    audit_handler.setLevel(logging.ERROR)
    audit_records: list[logging.LogRecord] = []

    def _capture(record: logging.LogRecord) -> None:
        audit_records.append(record)

    audit_handler.emit = _capture  # type: ignore[assignment]
    logger.addHandler(audit_handler)
    try:
        audit_phase_d_contamination(logger=logger)
    finally:
        logger.removeHandler(audit_handler)
    audit_failed = any("AUDIT_FAIL" in record.getMessage() for record in audit_records)
    if audit_failed and fail_on_audit:
        logger.error("Phase D audit failed; aborting (--fail-on-audit).")
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    symbol = args.symbol
    context_tf = args.timeframe.upper()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    bars_per_hour = {"H1": 1, "H2": 0.5}
    derived_bars = math.ceil(args.window_hours * bars_per_hour[context_tf])
    if derived_bars < 4:
        raise ValueError(f"Context window too small: {derived_bars} bars (min=4).")
    logger.info(
        "window_hours=%s context_tf=%s derived_bars=%s",
        args.window_hours,
        context_tf,
        derived_bars,
    )

    _run_phase_d_audit(logger, fail_on_audit=args.fail_on_audit)

    logger.info("download start symbol=%s tf=%s", symbol, context_tf)
    connector = MT5Connector()
    ohlcv_ctx = connector.obtener_ohlcv(symbol, context_tf, start, end)
    logger.info("download done rows=%s", len(ohlcv_ctx))

    model_root = Path(args.model_path)
    if model_root.is_dir() or model_root.suffix.lower() != ".pkl":
        model_path = model_root / f"{safe_filename(symbol)}_state_engine.pkl"
    else:
        model_path = model_root
    logger.info("predict start model_path=%s", model_path)
    model = StateEngineModel()
    model.load(model_path)
    feature_engineer = FeatureEngineer(FeatureConfig(window=derived_bars))
    full_features = feature_engineer.compute_features(ohlcv_ctx)
    features = feature_engineer.training_features(full_features)
    outputs = model.predict_outputs(features)
    logger.info("predict done rows=%s", len(outputs))

    logger.info("quality start")
    symbol_cfg = load_symbol_config(symbol, logger)
    if args.debug_allow_missing_quality:
        logger.warning("DEBUG: quality_labels=None (Phase C skipped).")
        quality_labels = None
    else:
        quality_config = _resolve_quality_config(symbol_cfg, logger)
        quality_labels, quality_assign_warnings = assign_quality_labels(
            outputs["state_hat"],
            full_features.reindex(outputs.index),
            quality_config,
        )
        if quality_assign_warnings:
            logger.warning("quality assign warnings: %s", quality_assign_warnings)
    logger.info("quality done")

    logger.info("look_for start")
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
    )
    ctx_df = context_bundle.ctx_df.copy()
    # CONSUMIBLE Phase E = ctx_df
    logger.info("look_for done")

    if (quality_labels is None) and (not args.debug_allow_missing_quality):
        logger.error("quality_labels=None no permitido en flujo principal (Phase C faltante).")
        raise SystemExit(2)

    if (not args.debug_allow_missing_quality) and ctx_df["quality_label_full"].nunique() <= 3:
        logger.error(
            "quality_label_full degenerado (nunique<=3). Phase C no aplicado correctamente."
        )
        raise SystemExit(2)

    # AQUÍ TERMINA Phase D
    # ctx_df ES EL CONSUMIBLE

    look_for_cols = [col for col in ctx_df.columns if col.startswith("LOOK_FOR_")]
    if look_for_cols:
        ctx_df[look_for_cols] = ctx_df[look_for_cols].fillna(0).astype(int)
    else:
        ctx_df["LOOK_FOR_NONE"] = 1
        look_for_cols = ["LOOK_FOR_NONE"]

    ctx_df["symbol"] = symbol
    index_name = ctx_df.index.name or "time"
    ctx_df = ctx_df.reset_index().rename(columns={index_name: "time"})

    ordered_cols = ["time", "symbol", "state_hat", "quality_label"]
    ordered_cols += sorted(look_for_cols)
    optional_cols = []
    if "margin" in ctx_df.columns:
        optional_cols.append("margin")
    optional_cols += sorted(col for col in ctx_df.columns if col.startswith("ctx_"))
    remaining_cols = [
        col
        for col in ctx_df.columns
        if col not in set(ordered_cols + optional_cols)
    ]
    ordered_cols += optional_cols + remaining_cols
    ctx_df = ctx_df.loc[:, ordered_cols]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"phase_d_context_base_{safe_filename(symbol)}_"
        f"{context_tf}_{args.start}_{args.end}.parquet"
    )
    output_path = output_dir / filename
    # ARTEFACTO AUDITORÍA Phase D (no es el consumible de Phase E; el consumible es ctx_df en memoria)
    # parquet/csv ES SOLO AUDITORÍA
    logger.info("export start path=%s", output_path)
    ctx_df.to_parquet(output_path, index=False)
    logger.info("export done rows=%s", len(ctx_df))

    # AQUÍ COMIENZA Phase E
    # ctx_df ES EL CONSUMIBLE

    if args.phase_e:
        score_tf = args.score_tf
        if score_tf is None:
            score_tf = (
                symbol_cfg.get("event_scorer", {}).get("score_tf") if isinstance(symbol_cfg, dict) else None
            )
        if score_tf is None:
            score_tf = "M5"
        score_tf = str(score_tf).upper()

        logger.info("Phase E download start symbol=%s tf=%s", symbol, score_tf)
        ohlcv_score = connector.obtener_ohlcv(symbol, score_tf, start, end)
        logger.info("Phase E download done rows=%s", len(ohlcv_score))

        df_score_ctx = merge_context_intraday(
            context_bundle.ctx_df.copy(),
            ohlcv_score,
            context_tf=context_tf,
            logger=logger,
        )

        state_col = f"state_hat_{context_tf}"
        if state_col in df_score_ctx.columns:
            base_state = _normalize_state_labels(df_score_ctx[state_col])
        else:
            base_state = _normalize_state_labels(df_score_ctx["state_hat"])
        df_score_ctx["base_state"] = base_state

        look_for_cols = [col for col in df_score_ctx.columns if col.startswith("LOOK_FOR_")]
        if look_for_cols:
            df_score_ctx[look_for_cols] = df_score_ctx[look_for_cols].fillna(0).astype(int)
        any_lookfor = df_score_ctx[look_for_cols].any(axis=1) if look_for_cols else pd.Series(False, index=df_score_ctx.index)

        logger.info("Phase E df_score_ctx rows=%s", len(df_score_ctx))
        logger.info("Phase E base_state value_counts:\n%s", df_score_ctx["base_state"].value_counts().to_string())
        logger.info(
            "Phase E quality_label_full value_counts (top10):\n%s",
            df_score_ctx["quality_label_full"].value_counts().head(10).to_string(),
        )
        logger.info("Phase E pct_any_lookfor=%.2f%%", float(any_lookfor.mean() * 100.0))
        distinct_pairs = (
            df_score_ctx[["base_state", "quality_label_full"]]
            .drop_duplicates()
            .sort_values(["base_state", "quality_label_full"])
        )
        logger.info(
            "Phase E distinct (base_state, quality_label_full) pairs=%s\n%s",
            len(distinct_pairs),
            distinct_pairs.to_string(index=False),
        )

        universe_events = df_score_ctx.copy()
        lookfor_events = df_score_ctx.loc[any_lookfor].copy()

        coverage_df = pd.DataFrame()
        if look_for_cols:
            total_rows = len(df_score_ctx)
            covered_rows = int(any_lookfor.sum())
            coverage_df = pd.DataFrame(
                [
                    {
                        "total_rows": total_rows,
                        "rows_with_look_for": covered_rows,
                        "coverage_pct": (covered_rows / max(total_rows, 1)) * 100.0,
                    }
                ]
            )

        output_base = args.out_phase_e or args.out
        if output_base is None:
            output_base = str(
                output_dir
                / f"phase_e_context_{safe_filename(symbol)}_{score_tf}_{args.start}_{args.end}"
            )

        universe_path = Path(f"{output_base}_universe.csv").resolve()
        lookfor_path = Path(f"{output_base}_lookfor.csv").resolve()
        coverage_path = Path(f"{output_base}_coverage.csv").resolve()

        universe_events.to_csv(universe_path, index=False)
        lookfor_events.to_csv(lookfor_path, index=False)
        if not coverage_df.empty:
            coverage_df.to_csv(coverage_path, index=False)

        logger.info("Phase E export universe=%s", universe_path)
        logger.info("Phase E export lookfor=%s", lookfor_path)
        if not coverage_df.empty:
            logger.info("Phase E export coverage=%s", coverage_path)


if __name__ == "__main__":
    main()

# Ejemplo CLI:
# python scripts/build_phase_d_context.py --symbol XAUUSD.mg --start 2024-01-01 --end 2024-02-01 --timeframe H1 --window-hours 48 --phase-e --score-tf M5 --out-phase-e outputs/phase_e_XAUUSD
