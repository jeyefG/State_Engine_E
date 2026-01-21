"""Build canonical Phase D context at the State Engine base timeframe."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

from state_engine.config_loader import load_config
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline_phase_d import build_context_bundle
from state_engine.quality import (
    QUALITY_LABEL_UNCLASSIFIED,
    assign_quality_labels,
    load_quality_config,
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
        required=True,
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
    quality_config, _, quality_warnings = load_quality_config(symbol)
    if quality_warnings:
        logger.warning("quality config warnings: %s", quality_warnings)
    quality_labels, quality_assign_warnings = assign_quality_labels(
        outputs["state_hat"],
        full_features.reindex(outputs.index),
        quality_config,
    )
    if quality_assign_warnings:
        logger.warning("quality assign warnings: %s", quality_assign_warnings)
    logger.info("quality done")

    logger.info("look_for start")
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
    )
    ctx_df = context_bundle.ctx_df.copy()
    logger.info("look_for done")

    quality_aligned = quality_labels.reindex(ctx_df.index)
    ctx_df["quality_label"] = quality_aligned.fillna(QUALITY_LABEL_UNCLASSIFIED)

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
    logger.info("export start path=%s", output_path)
    ctx_df.to_parquet(output_path, index=False)
    logger.info("export done rows=%s", len(ctx_df))


if __name__ == "__main__":
    main()
