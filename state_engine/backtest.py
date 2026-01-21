"""Deterministic backtester for M5 signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    family_id: str
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    edge_score: float


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    family_id: str
    side: str
    entry_price: float
    exit_price: float
    edge_score: float
    outcome: str
    pnl: float
    holding_bars: int


@dataclass(frozen=True)
class BacktestConfig:
    allow_overlap: bool = False
    max_holding_bars: int = 24
    fee_per_trade: float = 0.0
    slippage: float = 0.0


def run_backtest(
    df_m5: pd.DataFrame,
    signals: Iterable[Signal],
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Run a deterministic backtest and return trades, equity curve, metrics."""
    cfg = config or BacktestConfig()
    signals_sorted = sorted(signals, key=lambda s: s.entry_time)
    trades: list[Trade] = []
    equity: list[dict[str, float]] = []
    open_until: pd.Timestamp | None = None

    df = df_m5.copy()
    for signal in signals_sorted:
        if signal.entry_time not in df.index:
            continue
        if not cfg.allow_overlap and open_until is not None and signal.entry_time <= open_until:
            continue

        entry_idx = df.index.get_loc(signal.entry_time)
        entry_price = _apply_slippage(signal.entry_price, signal.side, cfg.slippage, entry=True)
        sl_price = signal.sl_price
        tp_price = signal.tp_price

        exit_price = entry_price
        exit_time = signal.entry_time
        outcome = "timeout"

        for offset in range(cfg.max_holding_bars):
            idx = entry_idx + offset
            if idx >= len(df.index):
                break
            high = df["high"].iloc[idx]
            low = df["low"].iloc[idx]
            if signal.side == "long":
                hit_sl = low <= sl_price
                hit_tp = high >= tp_price
            else:
                hit_sl = high >= sl_price
                hit_tp = low <= tp_price

            if hit_sl and hit_tp:
                outcome = "sl"
                exit_price = sl_price
                exit_time = df.index[idx]
                break
            if hit_sl:
                outcome = "sl"
                exit_price = sl_price
                exit_time = df.index[idx]
                break
            if hit_tp:
                outcome = "tp"
                exit_price = tp_price
                exit_time = df.index[idx]
                break

            exit_time = df.index[idx]

        exit_price = _apply_slippage(exit_price, signal.side, cfg.slippage, entry=False)
        pnl = _pnl(entry_price, exit_price, signal.side) - cfg.fee_per_trade
        holding_bars = df.index.get_loc(exit_time) - entry_idx + 1

        trades.append(
            Trade(
                entry_time=signal.entry_time,
                exit_time=exit_time,
                family_id=signal.family_id,
                side=signal.side,
                entry_price=entry_price,
                exit_price=exit_price,
                edge_score=signal.edge_score,
                outcome=outcome,
                pnl=pnl,
                holding_bars=holding_bars,
            )
        )
        open_until = exit_time

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = _build_equity_curve(trades_df)
    metrics = compute_metrics(trades_df)
    return trades_df, equity_df, metrics


def compute_metrics(trades_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute metrics globally, per family, and per edge_score bin."""
    metrics: dict[str, dict[str, float]] = {}
    metrics["global"] = _metrics_block(trades_df)

    if not trades_df.empty:
        for family_id, group in trades_df.groupby("family_id"):
            metrics[f"family:{family_id}"] = _metrics_block(group)

        bins = pd.cut(trades_df["edge_score"], bins=[0, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        for bin_label, group in trades_df.groupby(bins):
            metrics[f"edge_bin:{bin_label}"] = _metrics_block(group)

    return metrics


def _metrics_block(trades: pd.DataFrame) -> dict[str, float]:
    n_trades = float(len(trades))
    if n_trades == 0:
        return {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0.0,
            "exposure": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0
    win_rate = float(len(wins) / n_trades)
    expectancy = float(trades["pnl"].mean())
    profit_factor = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if not losses.empty else float("inf")
    max_dd, max_dd_duration = _max_drawdown(trades)
    exposure = float(trades["holding_bars"].sum()) if "holding_bars" in trades else 0.0
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration,
        "exposure": exposure,
    }


def _build_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["time", "equity"]).set_index("time")
    trades_df = trades_df.sort_values("exit_time")
    equity = trades_df["pnl"].cumsum()
    return pd.DataFrame({"time": trades_df["exit_time"], "equity": equity}).set_index("time")


def _max_drawdown(trades_df: pd.DataFrame) -> tuple[float, float]:
    if trades_df.empty:
        return 0.0, 0.0
    equity = trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())
    duration = float((drawdown != 0).astype(int).groupby((drawdown == 0).cumsum()).sum().max())
    return max_dd, duration


def _pnl(entry_price: float, exit_price: float, side: str) -> float:
    if side == "long":
        return exit_price - entry_price
    return entry_price - exit_price


def _apply_slippage(price: float, side: str, slippage: float, entry: bool) -> float:
    if slippage == 0:
        return price
    if entry:
        return price + slippage if side == "long" else price - slippage
    return price - slippage if side == "long" else price + slippage


__all__ = ["Signal", "Trade", "BacktestConfig", "run_backtest", "compute_metrics"]
