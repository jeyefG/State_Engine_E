"""Watchdog for State Engine opportunities on new context bars."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import pandas as pd

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_engine.features import FeatureConfig
from state_engine.events import detect_events
from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline import DatasetBuilder
from state_engine.scoring import EventScorerBundle, FeatureBuilder


LABEL_ORDER = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]
_TIMEFRAME_FLOOR_MAP = {
    "M1": "1min",
    "M2": "2min",
    "M3": "3min",
    "M4": "4min",
    "M5": "5min",
    "M6": "6min",
    "M10": "10min",
    "M12": "12min",
    "M15": "15min",
    "M20": "20min",
    "M30": "30min",
    "H1": "1h",
    "H2": "2h",
    "H3": "3h",
    "H4": "4h",
    "H6": "6h",
    "H8": "8h",
    "H12": "12h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "MS",
}


def try_import_rich() -> dict[str, Any] | None:
    try:
        from rich.console import Console

        return {"Console": Console}
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watchdog for State Engine LOOK_FOR_* signals.")
    parser.add_argument(
        "--symbols",
        required=True,
        help="Símbolos MT5 separados por coma (ej. EURUSD,XAUUSD)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=str(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio donde se encuentran los modelos entrenados.",
    )
    parser.add_argument(
        "--model-template",
        default="{symbol}_state_engine.pkl",
        help="Plantilla del nombre de modelo. Usa {symbol} ya sanitizado.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=45,
        help="Días de historial (context TF) para calcular features.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Intervalo de polling en segundos para nuevas velas del contexto.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecuta un solo ciclo y termina.",
    )
    parser.add_argument(
        "--scorer-dir",
        type=Path,
        default=None,
        help=(
            "Directorio donde se encuentran los modelos del Event Scorer "
            "(default: --model-dir o PROJECT_ROOT/models)."
        ),
    )
    parser.add_argument(
        "--scorer-template",
        default="{symbol}_event_scorer.pkl",
        help="Plantilla del nombre del scorer. Usa {symbol} ya sanitizado.",
    )
    parser.add_argument(
        "--m5-lookback-min",
        type=int,
        default=180,
        help="Ventana M5 a evaluar hacia atrás desde el cutoff (minutos).",
    )
    parser.add_argument(
        "--top-events",
        type=int,
        default=5,
        help="Número máximo de eventos M5 a mostrar en el ranking.",
    )
    parser.add_argument(
        "--min-edge-score",
        type=float,
        default=None,
        help="Umbral mínimo para mostrar eventos en el ranking.",
    )
    parser.add_argument(
        "--context-tf",
        default=None,
        help="Timeframe del contexto State/LOOK_FOR (default: metadata del State Engine).",
    )
    parser.add_argument(
        "--phase-e",
        action="store_true",
        help="Modo Fase E (telemetría sin thresholds).",
    )
    return parser.parse_args()


def safe_symbol(sym: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)


def feature_config_from_metadata(metadata: dict[str, Any]) -> FeatureConfig:
    raw = metadata.get("feature_config") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return FeatureConfig()
    allowed = {key for key in FeatureConfig.__dataclass_fields__}
    config = {key: value for key, value in raw.items() if key in allowed}
    return FeatureConfig(**config)


def class_distribution(labels: pd.Series) -> list[dict[str, Any]]:
    total = len(labels)
    result = []
    for label in LABEL_ORDER:
        count = int((labels == int(label)).sum())
        pct = (count / total) * 100 if total else 0.0
        result.append({"label": label.name, "count": count, "pct": pct})
    return result


def load_model(symbol: str, model_dir: Path, template: str) -> tuple[StateEngineModel, Path]:
    path = model_dir / template.format(symbol=safe_symbol(symbol))
    model = StateEngineModel()
    model.load(path)
    return model, path


def resolve_context_tf(requested: str | None, model: StateEngineModel) -> tuple[str, str]:
    meta_tf = model.metadata.get("timeframe") if isinstance(model.metadata, dict) else None
    if requested:
        return str(requested).upper(), "override(--context-tf)"
    if meta_tf:
        return str(meta_tf).upper(), "metadata"
    return "H1", "default"


def _timeframe_floor_freq(timeframe: str) -> str:
    tf = str(timeframe).upper()
    if tf not in _TIMEFRAME_FLOOR_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_FLOOR_MAP[tf]


def load_event_scorer(
    symbol: str,
    scorer_dir: Path,
    template: str,
) -> tuple[EventScorerBundle | None, Path]:
    path = scorer_dir / template.format(symbol=safe_symbol(symbol))
    scorer = EventScorerBundle()
    if not path.exists():
        return None, path
    scorer.load(path)
    return scorer, path


def fetch_m5(
    connector: MT5Connector,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    return connector.obtener_m5(symbol, start, end)


def build_m5_context(
    df_m5: pd.DataFrame,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    *,
    symbol: str | None = None,
    context_tf: str = "H1",
) -> pd.DataFrame:
    if symbol is not None and "symbol" not in df_m5.columns:
        df_m5 = df_m5.copy()
        df_m5["symbol"] = symbol
    ctx_cols = [col for col in outputs.columns if col.startswith("ctx_")]
    context_tf_norm = str(context_tf).upper()
    h1_ctx = outputs[["state_hat", "margin", *ctx_cols]].rename(
        columns={"state_hat": f"state_hat_{context_tf_norm}", "margin": f"margin_{context_tf_norm}"}
    )
    h1_ctx = h1_ctx.join(gating).shift(1).sort_index()
    m5 = df_m5.sort_index().reset_index()
    h1 = h1_ctx.reset_index()
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    allow_cols = [col for col in gating.columns if col in merged.columns]
    if allow_cols:
        merged[allow_cols] = merged[allow_cols].fillna(0).astype(int)
    return merged


def _entry_proxy(events_df: pd.DataFrame, df_m5: pd.DataFrame) -> pd.DataFrame:
    event_idx = df_m5.index.get_indexer(events_df.index)
    next_times = []
    next_prices = []
    for pos in event_idx:
        if pos == -1 or pos + 1 >= len(df_m5.index):
            next_times.append(None)
            next_prices.append(None)
            continue
        next_times.append(df_m5.index[pos + 1])
        next_prices.append(float(df_m5["open"].iloc[pos + 1]))
    enriched = events_df.copy()
    enriched["entry_proxy_time"] = next_times
    enriched["entry_proxy_price"] = next_prices
    return enriched


def align_features(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    aligned = features.join(labels.rename("label"), how="inner")
    aligned = aligned.dropna()
    return aligned


def render_summary(
    *,
    symbol: str,
    model_path: Path,
    metadata: dict[str, Any],
    labels: pd.Series,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    context_tf: str,
    last_bar_ts: pd.Timestamp,
    server_now: pd.Timestamp,
    console: Any | None,
) -> None:
    allow_any = gating.any(axis=1)
    look_for_coverage_rate = float(allow_any.mean()) if len(gating) else 0.0

    last_idx = outputs.index.max()
    last_allow = bool(allow_any.loc[last_idx]) if last_idx in allow_any.index else False
    last_state_hat = outputs.loc[last_idx, "state_hat"] if last_idx in outputs.index else None
    last_margin = float(outputs.loc[last_idx, "margin"]) if last_idx in outputs.index else None
    last_rules = [col for col in gating.columns if bool(gating.loc[last_idx, col])] if last_idx in gating.index else []

    label_dist = class_distribution(labels)
    baseline = max(label_dist, key=lambda r: r["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "NA"
    baseline_pct = baseline["pct"] if baseline else 0.0

    bar_age_minutes = (server_now - last_bar_ts).total_seconds() / 60.0
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    tick_age_min_utc = (now_utc - server_now).total_seconds() / 60.0

    trained_start = metadata.get("start")
    trained_end = metadata.get("end")
    n_samples = metadata.get("n_samples")
    n_train = metadata.get("n_train")
    n_test = metadata.get("n_test")
    state_label = StateLabels(int(last_state_hat)).name if last_state_hat is not None else "NA"
    margin_value = f"{last_margin:.4f}" if last_margin is not None else "NA"

    if console:
        console.print()
        console.print("[bold]=== State Engine Summary ===[/bold]")
        console.print(f"[cyan]Symbol:[/cyan] {symbol}")
        if trained_start and trained_end:
            console.print(f"[cyan]Period:[/cyan] {trained_start} -> {trained_end}")
        if n_samples is not None and n_train is not None and n_test is not None:
            console.print(f"[cyan]Samples:[/cyan] {n_samples} (train={n_train}, test={n_test})")
        console.print(f"[cyan]Baseline:[/cyan] {baseline_label} ({baseline_pct:.2f}%)")
        console.print(
            f"[cyan]Look-for coverage:[/cyan] {look_for_coverage_rate*100:.2f}%"
        )
        console.print(
            f"[cyan]Last {context_tf} bar used:[/cyan] {last_bar_ts} | age_min={bar_age_minutes:.2f}"
        )
        console.print(
            f"[cyan]Server now (tick):[/cyan] {server_now} | tick_age_min_vs_utc={tick_age_min_utc:.2f}"
        )
        console.print(
            f"[cyan]Last bar look_for:[/cyan] any={last_allow} | state_hat={state_label} | margin={margin_value}"
        )
        console.print(f"[cyan]Last bar rules fired:[/cyan] {last_rules if last_rules else '[]'}")
        console.print(f"[cyan]Model saved:[/cyan] {model_path}")
        return

    print()
    print("=== State Engine Summary ===")
    print(f"Symbol: {symbol}")
    if trained_start and trained_end:
        print(f"Period: {trained_start} -> {trained_end}")
    if n_samples is not None and n_train is not None and n_test is not None:
        print(f"Samples: {n_samples} (train={n_train}, test={n_test})")
    print(f"Baseline: {baseline_label} ({baseline_pct:.2f}%)")
    print(f"Look-for coverage rate: {look_for_coverage_rate*100:.2f}%")
    print(f"Last {context_tf} bar used: {last_bar_ts} | age_min={bar_age_minutes:.2f}")
    print(f"Server now (tick): {server_now} | tick_age_min_vs_utc={tick_age_min_utc:.2f}")
    print(f"Last bar look_for: any={last_allow} | state_hat={state_label} | margin={margin_value}")
    print(f"Last bar rules fired: {last_rules if last_rules else '[]'}")
    print(f"Model saved: {model_path}")


def main() -> None:
    args = parse_args()
    symbols = [sym.strip() for sym in args.symbols.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("Debe especificar al menos un símbolo en --symbols.")

    scorer_dir = args.scorer_dir or args.model_dir or (PROJECT_ROOT / "models")

    rich_modules = try_import_rich()
    console = rich_modules["Console"]() if rich_modules else None
    connector = MT5Connector()
    models: dict[str, StateEngineModel] = {}
    model_paths: dict[str, Path] = {}
    feature_configs: dict[str, FeatureConfig] = {}
    scorers: dict[str, EventScorerBundle | None] = {}
    scorer_paths: dict[str, Path] = {}

    def _signal_handler(sig: int, frame: Any) -> None:
        stop_event.set()

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        context_tfs: dict[str, str] = {}
        feature_builders: dict[str, FeatureBuilder] = {}
        for symbol in symbols:
            model, path = load_model(symbol, args.model_dir, args.model_template)
            models[symbol] = model
            model_paths[symbol] = path
            feature_configs[symbol] = feature_config_from_metadata(model.metadata)
            context_tf, context_source = resolve_context_tf(args.context_tf, model)
            context_tfs[symbol] = context_tf
            feature_builders[symbol] = FeatureBuilder(context_tf=context_tf)
            print(f"[watchdog] symbol={symbol} context_tf resolved={context_tf} source={context_source}")
            scorer, scorer_path = load_event_scorer(symbol, scorer_dir, args.scorer_template)
            scorers[symbol] = scorer
            scorer_paths[symbol] = scorer_path
            if scorer is not None:
                feature_builders[symbol].validate_context_metadata(scorer.metadata)

        last_seen: dict[str, pd.Timestamp] = {}

        while True:
            if stop_event.is_set():
                break
            for symbol in symbols:
                server_now = connector.server_now(symbol).tz_localize(None)
                context_tf = context_tfs.get(symbol, "H1")
                cutoff = server_now.floor(_timeframe_floor_freq(context_tf))
                start = cutoff - timedelta(days=args.lookback_days)
                end = cutoff + timedelta(days=1)

                ohlcv = connector.obtener_ohlcv(symbol, context_tf, start, end)
                ohlcv = ohlcv[ohlcv.index < cutoff]
                if ohlcv.empty:
                    continue

                last_bar_ts = pd.Timestamp(ohlcv.index.max()).tz_localize(None)
                if last_seen.get(symbol) == last_bar_ts:
                    continue

                builder = DatasetBuilder(feature_config=feature_configs[symbol])
                artifacts = builder.build(ohlcv)
                aligned = align_features(artifacts.features, artifacts.labels)
                if aligned.empty:
                    last_seen[symbol] = last_bar_ts
                    continue

                features = aligned.drop(columns=["label"])
                labels = aligned["label"].astype(int)
                full_features = artifacts.full_features.loc[features.index]

                outputs = models[symbol].predict_outputs(features)
                gating_policy = GatingPolicy()
                gating = gating_policy.apply(outputs, full_features)

                last_idx = outputs.index.max()
                allow_any = gating.any(axis=1)
                last_allow = bool(allow_any.loc[last_idx]) if last_idx in allow_any.index else False
                render_summary(
                    symbol=symbol,
                    model_path=model_paths[symbol],
                    metadata=models[symbol].metadata,
                    labels=labels,
                    outputs=outputs,
                    gating=gating,
                    context_tf=context_tf,
                    last_bar_ts=last_bar_ts,
                    server_now=server_now,
                    console=console,
                )

                if last_allow:
                    snapshot_start = cutoff - timedelta(minutes=args.m5_lookback_min)
                    try:
                        df_m5 = fetch_m5(connector, symbol, snapshot_start, cutoff)
                        df_m5 = df_m5[df_m5.index <= cutoff]
                    except Exception as exc:
                        message = f"Warning: M5 fetch failed for {symbol}: {exc}"
                        if console:
                            console.print(f"[yellow]{message}[/yellow]")
                        else:
                            print(message)
                        last_seen[symbol] = last_bar_ts
                        continue

                    if df_m5.empty:
                        message = f"Warning: no M5 data for {symbol} in window."
                        if console:
                            console.print(f"[yellow]{message}[/yellow]")
                        else:
                            print(message)
                        last_seen[symbol] = last_bar_ts
                        continue

                    df_m5_ctx = build_m5_context(
                        df_m5,
                        outputs,
                        gating,
                        symbol=symbol,
                        context_tf=context_tf,
                    )
                    try:
                        events_df = detect_events(df_m5_ctx)
                    except Exception as exc:
                        message = f"Warning: event detection failed for {symbol}: {exc}"
                        if console:
                            console.print(f"[yellow]{message}[/yellow]")
                        else:
                            print(message)
                        last_seen[symbol] = last_bar_ts
                        continue

                    event_counts = events_df["family_id"].value_counts().to_dict()
                    lines = [
                        "=== Event Scorer (M5) Snapshot ===",
                        f"Window: {snapshot_start} -> {cutoff}",
                        f"Events detected: total={len(events_df)} | by_family={event_counts}",
                    ]
                    if events_df.empty:
                        lines.append("no events detected in M5 window.")
                    else:
                        scorer = scorers.get(symbol)
                        if scorer is None:
                            lines.append("scorer not available – cannot rank opportunities.")
                        else:
                            try:
                                feature_builder = feature_builders[symbol]
                                base_features = feature_builder.build(df_m5_ctx)
                                event_features = base_features.loc[events_df.index]
                                event_features = feature_builder.add_family_features(
                                    event_features, events_df["family_id"]
                                )
                                edge_scores = scorer.predict_proba(event_features, events_df["family_id"])
                                ranked = events_df.copy()
                                ranked["edge_score"] = edge_scores
                                ranked = _entry_proxy(ranked, df_m5)
                                if args.phase_e:
                                    lines.append(
                                        "phase_e=ON: min_edge_score ignored (telemetry only)."
                                    )
                                elif args.min_edge_score is not None:
                                    ranked = ranked[ranked["edge_score"] >= args.min_edge_score]
                                ranked = ranked.sort_values("edge_score", ascending=False)
                                lines.append("top_events is presentation-only (no decisions).")
                                lines.append("Top events (sorted by edge_score desc):")
                                for idx, (ts, row) in enumerate(
                                    ranked.head(args.top_events).iterrows(), start=1
                                ):
                                    side = str(row.get("side", "")).upper()
                                    edge = row.get("edge_score")
                                    edge_value = f"{edge:.2f}" if pd.notna(edge) else "NA"
                                    entry_time = row.get("entry_proxy_time")
                                    entry_price = row.get("entry_proxy_price")
                                    lines.append(
                                        f"{idx}) ts={ts} | family={row.get('family_id')} | "
                                        f"side={side} | edge={edge_value}"
                                    )
                                    lines.append(
                                        f"   entry_proxy_time={entry_time} | "
                                        f"entry_proxy_price={entry_price}"
                                    )
                                if ranked.empty:
                                    lines.append("no events above edge_score threshold.")
                            except Exception as exc:
                                lines.append(f"scorer error – cannot rank opportunities: {exc}")

                    for line in lines:
                        if console:
                            console.print(line)
                        else:
                            print(line)

                last_seen[symbol] = last_bar_ts

            if args.once:
                break
            if stop_event.wait(timeout=args.poll_seconds):
                break
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()
