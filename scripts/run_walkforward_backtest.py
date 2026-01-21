"""Walk-forward evaluation for the Event Scorer (end-to-end pipeline)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from state_engine.backtest import BacktestConfig, Signal, run_backtest
from state_engine.events import EventFamily, detect_events, label_events
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.scoring import EventScorer, EventScorerBundle, EventScorerConfig, FeatureBuilder
from state_engine.walkforward import WalkForwardSplit, apply_edge_ablation, generate_walkforward_splits
from state_engine.config_loader import load_config
from state_engine.context_features import build_context_features
from state_engine.pipeline_phase_d import validate_look_for_context_requirements


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for the Event Scorer.")
    parser.add_argument("--symbol", default="EURUSD", help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--train-days", type=int, default=120, help="Días en train")
    parser.add_argument("--calib-days", type=int, default=30, help="Días en calibración")
    parser.add_argument("--test-days", type=int, default=30, help="Días en test")
    parser.add_argument("--step-days", type=int, default=30, help="Paso (rolling) entre folds")
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--model-dir", type=Path, default=Path(PROJECT_ROOT / "state_engine" / "models"))
    parser.add_argument("--output-dir", type=Path, default=Path(PROJECT_ROOT / "state_engine" / "models" / "walkforward"), help="Directorio base para resultados")
    parser.add_argument("--edge-threshold", type=float, default=0.6, help="Threshold global edge_score")
    parser.add_argument("--max-holding-bars", type=int, default=24, help="Máximo de velas M5 por trade")
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador ATR para SL")
    parser.add_argument("--k-bars", type=int, default=24, help="Ventana futura K para etiquetas")
    parser.add_argument("--r-thr", type=float, default=0.0, help="Umbral para label binario basado en r_outcome")
    parser.add_argument("--tie-break", default="distance", choices=["distance", "worst"], help="Tie-break TP/SL")
    parser.add_argument("--fee", type=float, default=0.0, help="Fee por trade (en unidades de precio)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage en unidades de precio")
    parser.add_argument("--ablation", default="real", choices=["real", "shuffle", "constant"], help="Ablation mode")
    parser.add_argument("--seed", type=int, default=7, help="Seed para ablations")
    parser.add_argument("--min-family-samples", type=int, default=200, help="Mínimo de muestras por familia")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("walkforward")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_symbol_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "symbols" / f"{symbol}.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("symbol config load failed for %s: %s", symbol, exc)
        return {}
    if not isinstance(config, dict):
        return {}
    return config


def _json_default(obj: object) -> str | float | int:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def build_h1_context(
    ohlcv_h1: pd.DataFrame,
    state_model: StateEngineModel,
    feature_engineer: FeatureEngineer,
    gating: GatingPolicy,
    symbol_cfg: dict | None,
    symbol: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    full_features = feature_engineer.compute_features(ohlcv_h1)
    features = feature_engineer.training_features(full_features)
    outputs = state_model.predict_outputs(features)
    ctx_features = build_context_features(
        ohlcv_h1,
        outputs,
        symbol=symbol,
        timeframe="H1",
    )
    features_for_gating = full_features.join(ctx_features, how="left").reindex(outputs.index)
    validate_look_for_context_requirements(
        symbol_cfg,
        set(features_for_gating.columns) | set(outputs.columns),
        logger=logger,
    )
    allows = gating.apply(outputs, features=features_for_gating, config_meta=symbol_cfg, logger=logger)
    ctx_cols = [col for col in outputs.columns if col.startswith("ctx_")]
    ctx = pd.concat([outputs[["state_hat", "margin", *ctx_cols]], ctx_features, allows], axis=1)
    return ctx.shift(1)


def merge_h1_m5(ctx_h1: pd.DataFrame, ohlcv_m5: pd.DataFrame) -> pd.DataFrame:
    h1 = ctx_h1.copy().sort_index()
    m5 = ohlcv_m5.copy().sort_index()
    if getattr(h1.index, "tz", None) is not None:
        h1.index = h1.index.tz_localize(None)
    if getattr(m5.index, "tz", None) is not None:
        m5.index = m5.index.tz_localize(None)
    h1 = h1.reset_index().rename(columns={h1.index.name or "index": "time"})
    m5 = m5.reset_index().rename(columns={m5.index.name or "index": "time"})
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    merged = merged.rename(columns={"state_hat": "state_hat_H1", "margin": "margin_H1"})
    return merged


def slice_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index < end)]


def build_signals(
    events: pd.DataFrame,
    df_m5: pd.DataFrame,
    edge_threshold: float,
    reward_r: float,
    sl_mult: float,
    atr_window: int = 14,
) -> list[Signal]:
    if events.empty:
        return []

    atr_short = _atr(df_m5["high"], df_m5["low"], df_m5["close"], atr_window)
    signals: list[Signal] = []
    for ts, row in events.iterrows():
        if row["edge_score"] < edge_threshold:
            continue
        if ts not in df_m5.index:
            continue
        idx = df_m5.index.get_loc(ts)
        entry_idx = idx + 1
        if entry_idx >= len(df_m5.index):
            continue
        entry_time = df_m5.index[entry_idx]
        entry_price = float(df_m5["open"].iloc[entry_idx])
        atr_value = atr_short.iloc[idx]
        if pd.isna(atr_value):
            continue
        sl_proxy = float(atr_value * sl_mult)
        trigger_required = row["family_id"] != EventFamily.TREND_CONTINUATION.value
        if row["side"] == "long":
            sl = entry_price - sl_proxy
            tp = entry_price + reward_r * sl_proxy
            if trigger_required:
                trigger_price = row["high"]
                triggered = df_m5["high"].iloc[entry_idx] >= trigger_price
            else:
                triggered = True
        else:
            sl = entry_price + sl_proxy
            tp = entry_price - reward_r * sl_proxy
            if trigger_required:
                trigger_price = row["low"]
                triggered = df_m5["low"].iloc[entry_idx] <= trigger_price
            else:
                triggered = True
        if trigger_required and not triggered:
            continue
        if entry_time <= ts:
            continue
        signals.append(
            Signal(
                signal_time=ts,
                entry_time=entry_time,
                family_id=row["family_id"],
                side=row["side"],
                entry_price=entry_price,
                sl_price=sl,
                tp_price=tp,
                edge_score=float(row["edge_score"]),
            )
        )
    return signals


def _align_training_data(
    labeled_events: pd.DataFrame,
    event_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    labeled = labeled_events.dropna(subset=["label"])
    if labeled.index.has_duplicates:
        labeled = labeled[~labeled.index.duplicated(keep="last")]
    if labeled.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object)
    features = event_features
    if features.index.has_duplicates:
        features = features[~features.index.duplicated(keep="last")]
    features = features.reindex(labeled.index)
    combined = pd.concat([features, labeled[["label", "family_id"]]], axis=1)
    combined = combined.dropna()
    if combined.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object)
    X = combined.drop(columns=["label", "family_id"])
    y = combined["label"].astype(int)
    fam = combined["family_id"].astype(str)
    return X, y, fam


def _train_scorer_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fam_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    fam_calib: pd.Series,
    min_family_samples: int,
) -> EventScorerBundle | None:
    if X_train.empty or y_train.nunique() < 2:
        return None
    config = EventScorerConfig()
    scorer_bundle = EventScorerBundle(config)
    global_scorer = EventScorer(config)
    global_scorer.fit(X_train, y_train, X_calib if not X_calib.empty else None, y_calib if not y_calib.empty else None)
    scorer_bundle.scorers[scorer_bundle.global_key] = global_scorer

    for family_id in sorted(fam_train.unique()):
        train_mask = fam_train == family_id
        calib_mask = fam_calib == family_id
        if train_mask.sum() < min_family_samples:
            continue
        if y_train.loc[train_mask].nunique() < 2:
            continue
        if calib_mask.any() and y_calib.loc[calib_mask].nunique() < 2:
            continue
        scorer = EventScorer(config)
        calib_X = X_calib.loc[calib_mask] if calib_mask.any() else None
        calib_y = y_calib.loc[calib_mask] if calib_mask.any() else None
        scorer.fit(X_train.loc[train_mask], y_train.loc[train_mask], calib_X, calib_y)
        scorer_bundle.scorers[str(family_id)] = scorer
    return scorer_bundle


def _edge_stats(scores: pd.Series) -> dict[str, float]:
    if scores.empty:
        return {"p50": 0.0, "p90": 0.0, "mean": 0.0}
    return {
        "p50": float(scores.quantile(0.5)),
        "p90": float(scores.quantile(0.9)),
        "mean": float(scores.mean()),
    }


def _metrics_block(
    trades_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        "global": metrics.get("global", {}),
        "by_family": {k: v for k, v in metrics.items() if k.startswith("family:")},
    }


def _save_fold_outputs(
    fold_dir: Path,
    events: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    prefix: str,
) -> dict[str, str]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    events_path = fold_dir / "events.csv"
    signals_path = fold_dir / f"{prefix}signals.csv"
    trades_path = fold_dir / f"{prefix}trades.csv"
    equity_path = fold_dir / f"{prefix}equity.csv"
    events.to_csv(events_path, index=True)
    signals_df.to_csv(signals_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=True)
    return {
        "events": str(events_path),
        "signals": str(signals_path),
        "trades": str(trades_path),
        "equity": str(equity_path),
    }


def run_walkforward(symbol: str, args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    def _safe_symbol(sym: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    state_model_path = args.state_model
    if state_model_path is None:
        state_model_path = args.model_dir / f"{_safe_symbol(symbol)}_state_engine.pkl"
    if not state_model_path.exists():
        raise FileNotFoundError(f"State model not found: {state_model_path}")

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)
    ohlcv_h1 = connector.obtener_h1(symbol, fecha_inicio, fecha_fin)
    ohlcv_m5 = connector.obtener_m5(symbol, fecha_inicio, fecha_fin)
    ohlcv_m5["symbol"] = symbol

    server_now = connector.server_now(symbol).tz_localize(None)
    ohlcv_h1 = ohlcv_h1[ohlcv_h1.index < server_now.floor("h")]
    ohlcv_m5 = ohlcv_m5[ohlcv_m5.index < server_now.floor("5min")]

    state_model = StateEngineModel()
    state_model.load(state_model_path)

    feature_engineer = FeatureEngineer(FeatureConfig())
    symbol_cfg = load_symbol_config(symbol, logger)
    gating = GatingPolicy()
    ctx_h1 = build_h1_context(
        ohlcv_h1,
        state_model,
        feature_engineer,
        gating,
        symbol_cfg,
        args.symbol,
        logger,
    )
    df_m5_ctx = merge_h1_m5(ctx_h1, ohlcv_m5)
    df_m5_ctx = df_m5_ctx.dropna(subset=["state_hat_H1", "margin_H1"])

    events_all = detect_events(df_m5_ctx)
    if events_all.empty:
        logger.warning("No events detected for %s; exiting.", symbol)
        return pd.DataFrame()

    feature_builder = FeatureBuilder()
    base_features = feature_builder.build(df_m5_ctx)

    splits = generate_walkforward_splits(
        start=pd.to_datetime(args.start),
        end=pd.to_datetime(args.end),
        train_days=args.train_days,
        calib_days=args.calib_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    if not splits:
        logger.warning("No folds generated; check date range.")
        return pd.DataFrame()

    output_base = args.output_dir / _safe_symbol(symbol)
    output_base.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for split in splits:
        logger.info(
            "Fold %s: train=%s->%s calib=%s->%s test=%s->%s",
            split.index,
            split.train_start,
            split.train_end,
            split.calib_start,
            split.calib_end,
            split.test_start,
            split.test_end,
        )

        events_train = slice_time(events_all, split.train_start, split.train_end)
        events_calib = slice_time(events_all, split.calib_start, split.calib_end)
        events_test = slice_time(events_all, split.test_start, split.test_end)

        df_m5_train = slice_time(ohlcv_m5, split.train_start, split.train_end)
        df_m5_calib = slice_time(ohlcv_m5, split.calib_start, split.calib_end)
        df_m5_test = slice_time(ohlcv_m5, split.test_start, split.test_end)

        labeled_train = label_events(
            events_train,
            df_m5_train,
            k_bars=args.k_bars,
            reward_r=args.reward_r,
            sl_mult=args.sl_mult,
            atr_window=14,
            r_thr=args.r_thr,
            tie_break=args.tie_break,
        )
        labeled_calib = label_events(
            events_calib,
            df_m5_calib,
            k_bars=args.k_bars,
            reward_r=args.reward_r,
            sl_mult=args.sl_mult,
            atr_window=14,
            r_thr=args.r_thr,
            tie_break=args.tie_break,
        )

        features_train = feature_builder.add_family_features(base_features.reindex(events_train.index), events_train["family_id"])
        features_calib = feature_builder.add_family_features(base_features.reindex(events_calib.index), events_calib["family_id"])
        X_train, y_train, fam_train = _align_training_data(labeled_train, features_train)
        X_calib, y_calib, fam_calib = _align_training_data(labeled_calib, features_calib)

        scorer_bundle = _train_scorer_bundle(
            X_train,
            y_train,
            fam_train,
            X_calib,
            y_calib,
            fam_calib,
            min_family_samples=args.min_family_samples,
        )

        events_test = events_test.copy()
        if scorer_bundle is None:
            logger.warning("Fold %s: insufficient data for training; using fallback edge_score=0.5", split.index)
            edge_scores = pd.Series(0.5, index=events_test.index, name="edge_score")
        else:
            features_test = feature_builder.add_family_features(
                base_features.reindex(events_test.index),
                events_test["family_id"],
            )
            edge_scores = scorer_bundle.predict_proba(features_test, events_test["family_id"])

        events_test["edge_score"] = edge_scores
        baseline_scores = apply_edge_ablation(events_test, edge_scores, args.ablation, seed=args.seed)
        events_test["edge_score_baseline"] = baseline_scores

        signals = build_signals(events_test, df_m5_test, args.edge_threshold, args.reward_r, args.sl_mult)
        config = BacktestConfig(
            allow_overlap=False,
            max_holding_bars=args.max_holding_bars,
            fee_per_trade=args.fee,
            slippage=args.slippage,
        )
        trades_df, equity_df, metrics = run_backtest(df_m5_test, signals, config)

        baseline_events = events_test.copy()
        baseline_events["edge_score"] = baseline_events["edge_score_baseline"]
        baseline_signals = build_signals(baseline_events, df_m5_test, args.edge_threshold, args.reward_r, args.sl_mult)
        baseline_trades_df, baseline_equity_df, baseline_metrics = run_backtest(df_m5_test, baseline_signals, config)

        fold_dir = output_base / f"fold_{split.index}"
        paths = _save_fold_outputs(
            fold_dir,
            events_test,
            pd.DataFrame([s.__dict__ for s in signals]),
            trades_df,
            equity_df,
            prefix="",
        )
        baseline_paths = _save_fold_outputs(
            fold_dir,
            events_test,
            pd.DataFrame([s.__dict__ for s in baseline_signals]),
            baseline_trades_df,
            baseline_equity_df,
            prefix="baseline_",
        )

        fold_metrics = {
            "fold": split.index,
            "symbol": symbol,
            "ranges": asdict(split),
            "counts": {
                "events": int(len(events_test)),
                "signals": int(len(signals)),
                "trades": int(len(trades_df)),
                "baseline_signals": int(len(baseline_signals)),
                "baseline_trades": int(len(baseline_trades_df)),
            },
            "edge_stats": {
                "scorer": _edge_stats(events_test["edge_score"]),
                "baseline": _edge_stats(events_test["edge_score_baseline"]),
            },
            "metrics": {
                "scorer": _metrics_block(trades_df, metrics),
                "baseline": _metrics_block(baseline_trades_df, baseline_metrics),
            },
            "paths": {
                "scorer": paths,
                "baseline": baseline_paths,
            },
            "ablation": args.ablation,
        }
        metrics_path = fold_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(fold_metrics, handle, indent=2, default=_json_default)

        summary_rows.append(
            {
                "symbol": symbol,
                "fold": split.index,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "calib_start": split.calib_start,
                "calib_end": split.calib_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "events": len(events_test),
                "signals": len(signals),
                "trades": len(trades_df),
                "baseline_signals": len(baseline_signals),
                "baseline_trades": len(baseline_trades_df),
                "edge_p50": fold_metrics["edge_stats"]["scorer"]["p50"],
                "edge_p90": fold_metrics["edge_stats"]["scorer"]["p90"],
                "baseline_edge_p50": fold_metrics["edge_stats"]["baseline"]["p50"],
                "baseline_edge_p90": fold_metrics["edge_stats"]["baseline"]["p90"],
                "expectancy": metrics.get("global", {}).get("expectancy", 0.0),
                "profit_factor": metrics.get("global", {}).get("profit_factor", 0.0),
                "max_drawdown": metrics.get("global", {}).get("max_drawdown", 0.0),
                "baseline_expectancy": baseline_metrics.get("global", {}).get("expectancy", 0.0),
                "baseline_profit_factor": baseline_metrics.get("global", {}).get("profit_factor", 0.0),
                "baseline_max_drawdown": baseline_metrics.get("global", {}).get("max_drawdown", 0.0),
                "ablation": args.ablation,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_base / "summary_folds.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)
    return summary_df


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    run_walkforward(args.symbol, args, logger)


if __name__ == "__main__":
    main()
