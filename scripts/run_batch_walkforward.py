"""Batch runner for walk-forward Event Scorer evaluation."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.run_walkforward_backtest import run_walkforward, setup_logging


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Batch walk-forward evaluation.")
    parser.add_argument("--symbols", required=True, help="Lista de símbolos separados por coma")
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


def _symbols_list(raw: str) -> list[str]:
    return [sym.strip() for sym in raw.split(",") if sym.strip()]


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    symbols = _symbols_list(args.symbols)
    if not symbols:
        raise ValueError("No symbols provided.")

    all_rows: list[pd.DataFrame] = []
    for symbol in symbols:
        logger.info("Running walk-forward for %s", symbol)
        summary_df = run_walkforward(symbol, args, logger)
        if not summary_df.empty:
            all_rows.append(summary_df)

    if not all_rows:
        logger.warning("No summaries produced.")
        return

    summary_all = pd.concat(all_rows, ignore_index=True)
    summary_path = args.output_dir / "summary_all_symbols.csv"
    summary_all.to_csv(summary_path, index=False)
    logger.info("Saved batch summary to %s", summary_path)


if __name__ == "__main__":
    main()
