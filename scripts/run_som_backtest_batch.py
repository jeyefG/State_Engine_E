"""Run SOM backtests for a predefined list of symbols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_pipeline_backtest.py"

SYMBOLS = [
    "US500",
    "EURUSD",
    "GBPUSD",
    "XAUUSD",
    "XAGUSD",
    "AUDUSD",
    "USDCLP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOM backtests in batch.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models" / "backtests"),
        help="Directorio base para resultados (se crea outputs/<symbol>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime los comandos a ejecutar sin correrlos.",
    )
    args, extra = parser.parse_known_args()
    args.extra = extra
    return args


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    extra_args = list(args.extra)

    filtered_extra: list[str] = []
    skip_next = False
    for idx, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--symbol":
            skip_next = True
            continue
        if arg.startswith("--symbol="):
            continue
        if arg == "--output-dir":
            skip_next = True
            continue
        if arg.startswith("--output-dir="):
            continue
        filtered_extra.append(arg)

    for symbol in SYMBOLS:
        output_dir = output_root / symbol
        cmd = [
            sys.executable,
            str(RUN_SCRIPT),
            "--symbol",
            symbol,
            "--output-dir",
            str(output_dir),
            *filtered_extra,
        ]
        print("Running:", " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
