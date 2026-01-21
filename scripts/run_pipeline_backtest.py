"""Run end-to-end pipeline: State Engine -> Events -> Scorer -> Signals -> Backtest."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from state_engine.backtest import BacktestConfig, Signal, run_backtest
from state_engine.events import EventFamily, detect_events
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.scoring import EventScorerBundle, FeatureBuilder
from state_engine.sweep import run_param_sweep
from state_engine.config_loader import load_config
from state_engine.context_features import build_context_features
from state_engine.pipeline_phase_d import validate_look_for_context_requirements


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Run pipeline backtest.")
    parser.add_argument("--symbol", default="EURUSD", help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--scorer-model", type=Path, default=None, help="Ruta del modelo Event Scorer (pkl)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para modelos",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(PROJECT_ROOT / "state_engine" / "models" / "backtests"), help="Directorio base para resultados")
    parser.add_argument("--edge-threshold", type=float, default=0.6, help="Threshold global edge_score")
    parser.add_argument("--max-holding-bars", type=int, default=24, help="Máximo de velas M5 por trade")
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador ATR para SL")
    parser.add_argument("--fee", type=float, default=0.0, help="Fee por trade (en unidades de precio)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage en unidades de precio")
    parser.add_argument(
        "--spread-per-trade",
        type=float,
        default=None,
        help="Spread por trade (precio). Se aplica como slippage adicional de spread/2",
    )
    parser.add_argument("--sweep", action="store_true", help="Ejecutar sweep de parámetros")
    parser.add_argument("--sweep-thresholds", type=float, nargs="*", default=[0.5, 0.6, 0.7], help="Edge thresholds")
    parser.add_argument("--sweep-k", type=int, nargs="*", default=[12, 24], help="Max holding bars")
    parser.add_argument("--sweep-r", type=float, nargs="*", default=[0.8, 1.0, 1.2], help="R múltiplo TP")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("pipeline")
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
    look_fors = gating.apply(outputs, features=features_for_gating, config_meta=symbol_cfg, logger=logger)
    ctx_cols = [col for col in outputs.columns if col.startswith("ctx_")]
    ctx = pd.concat([outputs[["state_hat", "margin", *ctx_cols]], ctx_features, look_fors], axis=1)
    return ctx.shift(1)


def merge_h1_m5(ctx_h1: pd.DataFrame, ohlcv_m5: pd.DataFrame) -> pd.DataFrame:
    h1 = ctx_h1.reset_index().rename(columns={"time": "time"}).sort_values("time")
    m5 = ohlcv_m5.reset_index().rename(columns={"time": "time"}).sort_values("time")
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    merged = merged.rename(columns={"state_hat": "state_hat_H1", "margin": "margin_H1"})
    return merged


def allow_constant_within_hour(df_m5_ctx: pd.DataFrame, look_for_cols: list[str]) -> bool:
    if not look_for_cols:
        return True
    grouped = df_m5_ctx[look_for_cols].groupby(df_m5_ctx.index.floor("h"))
    diffs = grouped.nunique().max()
    return bool((diffs <= 1).all())


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


def print_section(title: str, lines: list[str]) -> None:
    print(f"\n[{title}]")
    for line in lines:
        print(f"- {line}")


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)

    def _safe_symbol(sym: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    state_model_path = args.state_model
    if state_model_path is None:
        state_model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_state_engine.pkl"
    scorer_model_path = args.scorer_model
    if scorer_model_path is None:
        scorer_model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_event_scorer.pkl"

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)
    ohlcv_h1 = connector.obtener_h1(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5 = connector.obtener_m5(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5["symbol"] = args.symbol

    server_now = connector.server_now(args.symbol).tz_localize(None)
    ohlcv_h1 = ohlcv_h1[ohlcv_h1.index < server_now.floor("h")]
    ohlcv_m5 = ohlcv_m5[ohlcv_m5.index < server_now.floor("5min")]

    if not state_model_path.exists():
        raise FileNotFoundError(f"State model not found: {state_model_path}")

    state_model = StateEngineModel()
    state_model.load(state_model_path)

    feature_engineer = FeatureEngineer(FeatureConfig())
    symbol_cfg = load_symbol_config(args.symbol, logger)
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
    allowed_states = {StateLabels.BALANCE, StateLabels.TREND}
    df_m5_ctx = df_m5_ctx[df_m5_ctx["state_hat_H1"].isin(allowed_states)]

    look_for_cols = [col for col in df_m5_ctx.columns if col.startswith("LOOK_FOR_")]
    if not allow_constant_within_hour(df_m5_ctx, look_for_cols):
        logger.warning("LOOK_FOR_* changed within an hour. Check H1->M5 bridge.")

    events = detect_events(df_m5_ctx)
    allowed_families = {EventFamily.BALANCE_FADE.value, EventFamily.TREND_PULLBACK.value}
    events = events[events["family_id"].isin(allowed_families)]
    if events.empty:
        logger.warning("No events detected; exiting.")
        return

    feature_builder = FeatureBuilder()
    features_all = feature_builder.build(df_m5_ctx)
    event_features = features_all.reindex(events.index)
    event_features = feature_builder.add_family_features(event_features, events["family_id"])

    use_fallback = False
    if scorer_model_path.exists():
        scorer = EventScorerBundle()
        scorer.load(scorer_model_path)
        edge_scores = scorer.predict_proba(event_features, events["family_id"])
    else:
        use_fallback = True
        edge_scores = pd.Series(0.5, index=event_features.index, name="edge_score")

    events = events.copy()
    events["edge_score"] = edge_scores

    signals = build_signals(events, ohlcv_m5, args.edge_threshold, args.reward_r, args.sl_mult)
    spread_half = (args.spread_per_trade or 0.0) / 2.0
    effective_slippage = args.slippage + spread_half
    config = BacktestConfig(
        allow_overlap=False,
        max_holding_bars=args.max_holding_bars,
        fee_per_trade=args.fee,
        slippage=effective_slippage,
    )

    trades_df, equity_df, metrics = run_backtest(ohlcv_m5, signals, config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.csv"
    signals_path = output_dir / "signals.csv"
    trades_path = output_dir / "trades.csv"
    equity_path = output_dir / "equity.csv"

    events.to_csv(events_path, index=True)
    signals_df = pd.DataFrame([s.__dict__ for s in signals])
    signals_df.to_csv(signals_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=True)

    print_section(
        "DATA",
        [
            f"H1 rows={len(ohlcv_h1)} M5 rows={len(ohlcv_m5)}",
            f"range={ohlcv_m5.index.min()} -> {ohlcv_m5.index.max()}",
        ],
    )
    margin_series = df_m5_ctx["margin_H1"].dropna()
    print_section(
        "STATE ENGINE H1",
        [
            f"state_hat distribution={df_m5_ctx['state_hat_H1'].value_counts().to_dict()}",
            f"margin p50={margin_series.quantile(0.5):.3f} p90={margin_series.quantile(0.9):.3f}",
        ],
    )
    look_for_stats = {col: float(df_m5_ctx[col].mean()) for col in look_for_cols}
    print_section(
        "LOOK_FOR",
        [
            f"LOOK_FOR rates={look_for_stats}",
            f"tail={df_m5_ctx[look_for_cols].tail(5).to_dict(orient='records')}",
        ],
    )
    events_by_family = events["family_id"].value_counts().to_dict()
    events_by_side = events["side"].value_counts().to_dict()
    daily_rate = events.groupby(events.index.date).size().mean()
    print_section(
        "EVENTS",
        [
            f"count_by_family={events_by_family}",
            f"count_by_side={events_by_side}",
            f"avg_daily_events={daily_rate:.2f}",
        ],
    )
    scorer_stats = {
        "mean": float(events["edge_score"].mean()),
        "p50": float(events["edge_score"].quantile(0.5)),
        "p90": float(events["edge_score"].quantile(0.9)),
    }
    print_section(
        "SCORER",
        [
            f"edge_score stats={scorer_stats}",
            f"fallback={'YES' if use_fallback else 'NO'}",
        ],
    )
    if signals_df.empty or "edge_score" not in signals_df.columns:
        top_signals = []
    else:
        top_signals = signals_df.sort_values("edge_score", ascending=False).head(10).to_dict(orient="records")
    print_section(
        "SIGNALS",
        [
            f"count_by_family={signals_df['family_id'].value_counts().to_dict() if not signals_df.empty else {}}",
            f"top10={top_signals}",
        ],
    )
    metrics_by_family = {key: value for key, value in metrics.items() if key.startswith("family:")}
    metrics_by_edge_bin = {key: value for key, value in metrics.items() if key.startswith("edge_bin:")}
    if trades_df.empty or "pnl" not in trades_df.columns:
        top_trades: list[dict[str, object]] = []
        worst_trades: list[dict[str, object]] = []
    else:
        top_trades = trades_df.sort_values("pnl", ascending=False).head(10).to_dict(orient="records")
        worst_trades = trades_df.sort_values("pnl", ascending=True).head(10).to_dict(orient="records")
    print_section(
        "BACKTEST",
        [
            f"global={metrics.get('global', {})}",
            f"by_family={metrics_by_family}",
            f"by_edge_bin={metrics_by_edge_bin}",
            f"top10_trades={top_trades}",
            f"worst10_trades={worst_trades}",
        ],
    )
    if not equity_df.empty:
        monthly_equity = equity_df["equity"].resample("M").last().to_dict()
    else:
        monthly_equity = {}
    print_section("EQUITY_MONTHLY", [f"equity_monthly={monthly_equity}"])
    print_section(
        "SAVE",
        [
            f"events={events_path}",
            f"signals={signals_path}",
            f"trades={trades_path}",
            f"equity={equity_path}",
        ],
    )

    if args.sweep:
        logger.info("Running param sweep...")

        def pipeline_fn(params: dict[str, float]) -> dict[str, float]:
            threshold = float(params["edge_threshold"])
            max_hold = int(params["max_holding_bars"])
            reward_r = float(params["reward_r"])
            signals_local = build_signals(events, ohlcv_m5, threshold, reward_r, args.sl_mult)
            cfg = BacktestConfig(
                allow_overlap=False,
                max_holding_bars=max_hold,
                fee_per_trade=args.fee,
                slippage=effective_slippage,
            )
            trades_df_local, _, metrics_local = run_backtest(ohlcv_m5, signals_local, cfg)
            global_metrics = metrics_local.get("global", {})
            return {
                "profit_factor": float(global_metrics.get("profit_factor", 0.0)),
                "max_drawdown": float(global_metrics.get("max_drawdown", 0.0)),
                "expectancy": float(global_metrics.get("expectancy", 0.0)),
                "n_trades": float(global_metrics.get("n_trades", 0.0)),
            }

        grid = {
            "edge_threshold": args.sweep_thresholds,
            "max_holding_bars": args.sweep_k,
            "reward_r": args.sweep_r,
        }
        sweep_results = run_param_sweep(pipeline_fn, grid)
        sweep_path = output_dir / "sweep_results.csv"
        sweep_results.to_csv(sweep_path, index=False)
        top10 = sweep_results.sort_values(["profit_factor", "max_drawdown"], ascending=[False, True]).head(10)
        bottom10 = sweep_results.sort_values(["profit_factor", "max_drawdown"], ascending=[True, False]).head(10)
        print_section("SWEEP", [f"top10={top10.to_dict(orient='records')}", f"bottom10={bottom10.to_dict(orient='records')}"])
        print_section("SAVE", [f"sweep_results={sweep_path}"])


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


if __name__ == "__main__":
    main()
