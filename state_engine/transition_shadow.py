"""Shadow-mode descriptive analysis for TRANSITION state edge."""

from __future__ import annotations

import math

import pandas as pd

from state_engine.labels import StateLabels


def _state_label(value: object) -> str:
    try:
        return StateLabels(int(value)).name
    except Exception:
        return str(value)


def _format_value(value: float | int) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metrics_block(df: pd.DataFrame, pnl_col: str) -> dict[str, float]:
    if df.empty:
        return {
            "trades": 0.0,
            "winrate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "pnl_std": 0.0,
        }
    pnl = df[pnl_col]
    return {
        "trades": float(len(pnl)),
        "winrate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "pnl_std": float(pnl.std(ddof=0)),
    }


def print_transition_shadow_report(
    df: pd.DataFrame,
    *,
    state_col: str = "state_hat",
    margin_col: str = "margin_H1",
    pnl_col: str = "pnl",
    transition_value: int | str = StateLabels.TRANSITION,
) -> None:
    """Print shadow-mode metrics for TRANSITION edge analysis.

    Expects df to include columns: state_hat, margin_H1, pnl (or overrides).
    """
    if state_col not in df.columns:
        raise KeyError(f"Missing required column: {state_col}")
    if margin_col not in df.columns:
        raise KeyError(f"Missing required column: {margin_col}")
    if pnl_col not in df.columns:
        raise KeyError(f"Missing required column: {pnl_col}")

    df_local = df.copy()
    df_local["_state_label"] = df_local[state_col].map(_state_label)

    print("\n[Baseline por estado]")
    for state_label, group in df_local.groupby("_state_label"):
        metrics = _metrics_block(group, pnl_col)
        print(
            "- {state}: trades={trades} | winrate={winrate} | avg_pnl={avg_pnl} | "
            "total_pnl={total_pnl} | pnl_std={pnl_std}".format(
                state=state_label,
                trades=_format_value(metrics["trades"]),
                winrate=_format_value(metrics["winrate"]),
                avg_pnl=_format_value(metrics["avg_pnl"]),
                total_pnl=_format_value(metrics["total_pnl"]),
                pnl_std=_format_value(metrics["pnl_std"]),
            )
        )

    transition_label = _state_label(transition_value)
    transition_mask = df_local["_state_label"] == transition_label
    transition_df = df_local.loc[transition_mask]
    transition_metrics = _metrics_block(transition_df, pnl_col)

    print("\n[TRANSITION completo]")
    print(
        "- trades={trades} | winrate={winrate} | avg_pnl={avg_pnl} | total_pnl={total_pnl} | "
        "pnl_std={pnl_std}".format(
            trades=_format_value(transition_metrics["trades"]),
            winrate=_format_value(transition_metrics["winrate"]),
            avg_pnl=_format_value(transition_metrics["avg_pnl"]),
            total_pnl=_format_value(transition_metrics["total_pnl"]),
            pnl_std=_format_value(transition_metrics["pnl_std"]),
        )
    )

    margin_abs = df_local[margin_col].abs()
    margin_threshold = float(margin_abs.quantile(0.8)) if not margin_abs.empty else float("nan")
    conditioned_df = transition_df.loc[transition_df[margin_col].abs() >= margin_threshold]

    print("\n[TRANSITION condicionado | abs(margin_H1) >= p80]")
    print(
        "- count={count} | mean={mean} | sum={sum} | std={std}".format(
            count=_format_value(len(conditioned_df)),
            mean=_format_value(conditioned_df[pnl_col].mean() if not conditioned_df.empty else 0.0),
            sum=_format_value(conditioned_df[pnl_col].sum() if not conditioned_df.empty else 0.0),
            std=_format_value(conditioned_df[pnl_col].std(ddof=0) if not conditioned_df.empty else 0.0),
        )
    )

    transition_avg = transition_metrics["avg_pnl"]
    transition_std = transition_metrics["pnl_std"]
    conditioned_avg = float(conditioned_df[pnl_col].mean()) if not conditioned_df.empty else 0.0
    conditioned_std = float(conditioned_df[pnl_col].std(ddof=0)) if not conditioned_df.empty else 0.0
    avg_delta = conditioned_avg - transition_avg
    std_delta = conditioned_std - transition_std
    freq_loss = 0.0
    if transition_metrics["trades"] > 0:
        freq_loss = 1.0 - (len(conditioned_df) / transition_metrics["trades"])

    print("\n[Comparación directa]")
    print(
        "- avg_pnl: full={full} | conditioned={cond} | delta={delta}".format(
            full=_format_value(transition_avg),
            cond=_format_value(conditioned_avg),
            delta=_format_value(avg_delta),
        )
    )
    print(
        "- pnl_std: full={full} | conditioned={cond} | delta={delta}".format(
            full=_format_value(transition_std),
            cond=_format_value(conditioned_std),
            delta=_format_value(std_delta),
        )
    )
    print(f"- pérdida de frecuencia: {freq_loss * 100:.2f}%")

    total_count = len(df_local)
    transition_pct = (len(transition_df) / total_count * 100.0) if total_count else 0.0
    subset_pct = (len(conditioned_df) / len(transition_df) * 100.0) if len(transition_df) else 0.0

    print("\n[Frecuencia relativa]")
    print(f"- % TRANSITION sobre total: {transition_pct:.2f}%")
    print(f"- % subset condicionado sobre TRANSITION: {subset_pct:.2f}%")

    print("\n[Conclusión]")
    if conditioned_df.empty or conditioned_avg <= 0 or conditioned_avg <= transition_avg:
        print("A) No existe edge rescatable → TRANSITION se congela definitivamente como no operable.")
    else:
        print(
            "B) Existe edge solo en un subset raro y bien definido → TRANSITION sigue no operable, "
            "pero se documenta una regla especial futura (no se implementa ahora)."
        )


__all__ = ["print_transition_shadow_report"]
