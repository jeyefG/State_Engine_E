"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
import ast
import json
import importlib
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any

import numpy as np
import pandas as pd

from state_engine.features import FeatureConfig
from state_engine.context_features import build_context_features
from state_engine.config_loader import load_config
from state_engine.gating import GatingPolicy
from state_engine.pipeline_phase_d import validate_look_for_context_requirements
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel, StateEngineModelConfig
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline import DatasetBuilder
from state_engine.quality import (
    assign_quality_labels,
    build_quality_diagnostics,
    load_quality_config,
)




def parse_args() -> argparse.Namespace:
    
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument(
        "--start",
        required=True,
        help="Fecha inicio (YYYY-MM-DD) para descarga de velas",
    )
    parser.add_argument(
        "--end",
        default = default_end,
        help="Fecha fin (YYYY-MM-DD) para descarga de velas",
    )
    parser.add_argument(
        "--timeframe",
        choices=["M30", "H1", "H2", "H4"],
        default="H1",
        help="Timeframe de velas para el State Engine (M30, H1, H2, H4).",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Ventana temporal total (en horas) usada como contexto estructural.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Model output path. Si no se indica, se genera automáticamente desde el símbolo.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para guardar el modelo cuando --model-out no se especifica.",
    )
    parser.add_argument(
        "--log-level",
        choices=["minimal", "verbose", "debug"],
        default="minimal",
        help="Nivel de reporte en consola (minimal, verbose, debug).",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, WARNING)",
    )
    parser.add_argument("--min-samples", type=int, default=2000, help="Minimum samples required to train")
    parser.add_argument(
        "--ev-min-split-samples",
        type=int,
        default=50,
        help="Minimum samples per split for EV stability",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio (0-1)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")
    parser.add_argument("--report-out", type=Path, help="Optional report output path (.json)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' in LightGBM")
    parser.add_argument("--quality", action="store_true", help="Enable Quality Layer labeling (Phase C)")
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=None,
        help="Optional Quality Layer config path (YAML/JSON)",
    )
    parser.add_argument(
        "--diagnose-rescue-scans",
        action="store_true",
        help="Run diagnostic BALANCE/TRANSITION rescue scans (telemetry only).",
    )
    parser.add_argument(
        "--rescue-n-min",
        type=int,
        default=30,
        help="Minimum samples per rescue scan candidate bucket.",
    )
    parser.add_argument(
        "--rescue-delta-max",
        type=float,
        default=0.15,
        help="Max abs(delta_vs_state) for rescue scan candidates.",
    )
    parser.add_argument(
        "--rescue-top-k",
        type=int,
        default=12,
        help="Top-k rows to display for rescue scan tables.",
    )
    parser.add_argument(
        "--enable-phase-e-metrics",
        action="store_true",
        help="Enable Phase E outcome-based metrics (experimental, opt-in).",
    )
    return parser.parse_args()


def try_import_rich() -> dict[str, Any] | None:
    try:
        from rich.console import Console
        from rich.table import Table

        return {"Console": Console, "Table": Table}
    except Exception:
        return None


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("state_engine")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def class_distribution(labels: np.ndarray, label_order: list[StateLabels]) -> list[dict[str, Any]]:
    total = len(labels)
    result = []
    for label in label_order:
        count = int(np.sum(labels == label))
        pct = (count / total) * 100 if total else 0.0
        result.append({"label": label.name, "count": count, "pct": pct})
    return result


def confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray, label_order: list[StateLabels]) -> np.ndarray:
    matrix = np.zeros((len(label_order), len(label_order)), dtype=int)
    for i, actual in enumerate(label_order):
        for j, predicted in enumerate(label_order):
            matrix[i, j] = int(np.sum((labels_true == actual) & (labels_pred == predicted)))
    return matrix


def f1_macro(matrix: np.ndarray) -> float:
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def percentiles(series: pd.Series, qs: list[float]) -> dict[str, float | None]:
    """Return selected percentiles for a numeric Series.

    Parameters
    ----------
    series:
        Numeric series (non-numeric coerced to NaN).
    qs:
        Percentiles in [0, 100], e.g. [0, 50, 90, 95, 99, 100].

    Returns
    -------
    dict mapping percentile -> value (or None if empty after dropping NaN).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {str(q): None for q in qs}
    qv = s.quantile([q / 100.0 for q in qs]).to_dict()
    return {str(q): float(qv[q / 100.0]) for q in qs}


def render_table(
    console: Any,
    table_class: Any,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    table = table_class(title=title)
    for col in columns:
        table.add_column(str(col))
    for row in rows:
        table.add_row(*[str(x) for x in row])
    console.print(table)


REPORT_LEVELS = {"minimal": 0, "verbose": 1, "debug": 2}


def _build_context_features_audit(ctx_features: pd.DataFrame) -> pd.DataFrame:
    audit_cols = [
        "ctx_session_bucket",
        "ctx_state_age",
        "ctx_dist_vwap_atr",
    ]
    extra_cols = [col for col in ctx_features.columns if col.startswith("ctx_") and col not in audit_cols]
    all_cols = audit_cols + extra_cols
    rows: list[dict[str, Any]] = []
    for col in all_cols:
        if col not in ctx_features.columns:
            rows.append(
                {
                    "column": col,
                    "present": False,
                    "non_null_pct": 0.0,
                    "nan_pct": 100.0,
                    "min": None,
                    "max": None,
                }
            )
            continue
        series = ctx_features[col]
        non_null_pct = float(series.notna().mean() * 100.0) if len(series) else 0.0
        nan_pct = 100.0 - non_null_pct
        numeric = pd.to_numeric(series, errors="coerce")
        min_val = float(numeric.min()) if numeric.notna().any() else None
        max_val = float(numeric.max()) if numeric.notna().any() else None
        rows.append(
            {
                "column": col,
                "present": True,
                "non_null_pct": non_null_pct,
                "nan_pct": nan_pct,
                "min": min_val,
                "max": max_val,
            }
        )
    return pd.DataFrame(rows)


def _build_confidence_summary(outputs: pd.DataFrame) -> pd.DataFrame:
    if "margin" not in outputs.columns:
        return pd.DataFrame()
    summary_cols = ["state_hat"]
    if "quality_label" in outputs.columns:
        summary_cols.append("quality_label")
    grouped = outputs.groupby(summary_cols)["margin"]
    summary = grouped.agg(
        n_samples="count",
        mean="mean",
        p10=lambda s: s.quantile(0.10),
        p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90),
    ).reset_index()
    summary["state_hat"] = summary["state_hat"].map(
        lambda v: StateLabels(v).name if not pd.isna(v) else "NA"
    )
    return summary


def _build_look_for_coverage_table(
    look_for_df: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    look_for_cols = [col for col in look_for_df.columns if col.startswith("LOOK_FOR_")]
    total_rows = len(look_for_df)
    rows: list[dict[str, Any]] = []
    for look_for_name in look_for_cols:
        mask = look_for_df[look_for_name].astype(bool)
        n_allow = int(mask.sum())
        pct_total = (n_allow / total_rows * 100.0) if total_rows else 0.0
        state_counts = (
            outputs.loc[mask, "state_hat"]
            .map(lambda v: StateLabels(v).name if not pd.isna(v) else "NA")
            .value_counts()
            .to_dict()
        )
        quality_counts = {}
        if "quality_label" in outputs.columns:
            quality_counts = outputs.loc[mask, "quality_label"].fillna("NA").astype(str).value_counts().to_dict()
        rows.append(
            {
                "look_for_rule": look_for_name,
                "n": n_allow,
                "pct_total": pct_total,
                "state_counts": state_counts,
                "quality_counts": quality_counts,
            }
        )
    return pd.DataFrame(rows)


def _build_time_splits(index: pd.Index) -> dict[str, pd.Series]:
    if len(index) == 0:
        return {}
    index_series = pd.Series(index)
    q1 = index_series.quantile(1 / 3)
    q2 = index_series.quantile(2 / 3)
    return {
        "early": index <= q1,
        "mid": (index > q1) & (index <= q2),
        "late": index > q2,
    }


def _build_coverage_by_split(
    look_for_df: pd.DataFrame,
    outputs: pd.DataFrame,
    look_for_cfg: dict[str, Any],
) -> pd.DataFrame:
    from state_engine.gating import _base_state_mask

    look_for_cols = [col for col in look_for_df.columns if col.startswith("LOOK_FOR_")]
    splits = _build_time_splits(outputs.index)
    if not look_for_cols or not splits:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    state_hat = outputs["state_hat"]
    for split_name, split_mask in splits.items():
        n_total = int(split_mask.sum())
        for look_for_name in look_for_cols:
            rule_cfg = look_for_cfg.get(look_for_name, {}) if isinstance(look_for_cfg, dict) else {}
            base_state = None
            if isinstance(rule_cfg, dict):
                base_state = rule_cfg.get("base_state") or rule_cfg.get("anchor_state")
            if base_state is None:
                continue
            base_mask = _base_state_mask(state_hat, base_state, outputs.index)
            n_base_state = int((base_mask & split_mask).sum())
            n_hits = int((look_for_df[look_for_name].astype(bool) & split_mask).sum())
            pct_total = (n_hits / n_total) if n_total else 0.0
            pct_of_base = (n_hits / n_base_state) if n_base_state else float("nan")
            rows.append(
                {
                    "split": split_name,
                    "look_for": look_for_name,
                    "base_state": base_state,
                    "n": n_hits,
                    "pct_total": pct_total,
                    "n_base_state": n_base_state,
                    "pct_of_base_state": pct_of_base,
                }
            )
    return pd.DataFrame(rows)


def _build_look_for_jaccard(look_for_df: pd.DataFrame) -> pd.DataFrame:
    look_for_cols = [col for col in look_for_df.columns if col.startswith("LOOK_FOR_")]
    if len(look_for_cols) < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for idx_a, col_a in enumerate(look_for_cols):
        mask_a = look_for_df[col_a].astype(bool)
        for col_b in look_for_cols[idx_a + 1 :]:
            mask_b = look_for_df[col_b].astype(bool)
            inter = int((mask_a & mask_b).sum())
            union = int((mask_a | mask_b).sum())
            jaccard = (inter / union) if union else float("nan")
            rows.append(
                {
                    "A": col_a,
                    "B": col_b,
                    "jaccard": jaccard,
                    "inter": inter,
                    "union": union,
                }
            )
    if not rows:
        return pd.DataFrame()
    jaccard_df = pd.DataFrame(rows)
    jaccard_df = jaccard_df.sort_values(by="jaccard", ascending=False, na_position="last")
    return jaccard_df.head(20).reset_index(drop=True)


def _load_diagnostic_table_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = Path("configs") / "symbols" / f"{symbol}.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("diagnostic_tables config load failed for %s: %s", symbol, exc)
        return {}
    diagnostic_cfg = config.get("diagnostic_tables", {})
    if diagnostic_cfg is None or not isinstance(diagnostic_cfg, dict):
        return {}
    return diagnostic_cfg


def _load_symbol_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "symbols" / f"{symbol}.yaml"
    logger.info("SYMBOL_CONFIG_PATH symbol=%s path=%s exists=%s", symbol, config_path, config_path.exists())
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


def _look_for_filter_config_rows(symbol_config: dict[str, Any], look_for_rule: str) -> list[dict[str, str]]:
    phase_d = symbol_config.get("phase_d", {})
    if not isinstance(phase_d, dict):
        return []
    look_for_cfg = phase_d.get("look_fors", {})
    if not isinstance(look_for_cfg, dict):
        return []
    rule_cfg = look_for_cfg.get(look_for_rule, {})
    if not isinstance(rule_cfg, dict):
        return []
    return [{"key": f"{look_for_rule}.{key}", "source": "symbol"} for key in sorted(rule_cfg.keys())]


def _look_for_context_filter_counts(
    look_for_context_frame: pd.DataFrame,
    look_for_rule: str,
    filters_cfg: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    if look_for_rule not in look_for_context_frame.columns:
        logger.warning("phase_d.look_fors.%s missing in look_for_df; skipping counts.", look_for_rule)
        return pd.DataFrame()

    base_allow = look_for_context_frame[look_for_rule].astype(bool)
    total_base = int(base_allow.sum())
    if total_base == 0:
        return pd.DataFrame(
            [
                {
                    "filter": "base_allow",
                    "total_base": total_base,
                    "pass": 0,
                    "fail": 0,
                    "pass_pct": 0.0,
                    "fail_pct": 0.0,
                    "notes": "no base allow rows",
                }
            ]
        )

    def _mask_from_col(col_name: str, mask_fn, label: str) -> dict[str, Any]:
        if col_name not in look_for_context_frame.columns:
            return {
                "filter": label,
                "total_base": total_base,
                "pass": 0,
                "fail": total_base,
                "pass_pct": 0.0,
                "fail_pct": 100.0,
                "notes": f"missing {col_name}",
            }
        series = look_for_context_frame[col_name]
        mask = mask_fn(series)
        pass_count = int((base_allow & mask).sum())
        fail_count = total_base - pass_count
        pass_pct = (pass_count / total_base) * 100.0 if total_base else 0.0
        fail_pct = (fail_count / total_base) * 100.0 if total_base else 0.0
        return {
            "filter": label,
            "total_base": total_base,
            "pass": pass_count,
            "fail": fail_count,
            "pass_pct": pass_pct,
            "fail_pct": fail_pct,
            "notes": "",
        }

    rows: list[dict[str, Any]] = []
    sessions_in = filters_cfg.get("sessions_in")
    if sessions_in is not None:
        allowed = {str(val) for val in sessions_in}
        rows.append(
            _mask_from_col(
                "ctx_session_bucket",
                lambda s: s.astype(str).isin(allowed),
                f"sessions_in={sorted(allowed)}",
            )
        )
    state_age_min = filters_cfg.get("state_age_min")
    state_age_max = filters_cfg.get("state_age_max")
    if state_age_min is not None or state_age_max is not None:
        def _state_age_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if state_age_min is not None:
                mask &= numeric >= float(state_age_min)
            if state_age_max is not None:
                mask &= numeric <= float(state_age_max)
            return mask

        label = f"state_age[{state_age_min},{state_age_max}]"
        rows.append(_mask_from_col("ctx_state_age", _state_age_mask, label))

    dist_vwap_atr_min = filters_cfg.get("dist_vwap_atr_min")
    dist_vwap_atr_max = filters_cfg.get("dist_vwap_atr_max")
    if dist_vwap_atr_min is not None or dist_vwap_atr_max is not None:
        def _dist_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if dist_vwap_atr_min is not None:
                mask &= numeric >= float(dist_vwap_atr_min)
            if dist_vwap_atr_max is not None:
                mask &= numeric <= float(dist_vwap_atr_max)
            return mask

        label = f"dist_vwap_atr[{dist_vwap_atr_min},{dist_vwap_atr_max}]"
        rows.append(_mask_from_col("ctx_dist_vwap_atr", _dist_mask, label))

    breakmag_min = filters_cfg.get("breakmag_min")
    breakmag_max = filters_cfg.get("breakmag_max")
    if breakmag_min is not None or breakmag_max is not None:
        def _breakmag_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if breakmag_min is not None:
                mask &= numeric >= float(breakmag_min)
            if breakmag_max is not None:
                mask &= numeric <= float(breakmag_max)
            return mask

        label = f"breakmag[{breakmag_min},{breakmag_max}]"
        rows.append(_mask_from_col("BreakMag", _breakmag_mask, label))

    reentry_min = filters_cfg.get("reentry_min")
    reentry_max = filters_cfg.get("reentry_max")
    if reentry_min is not None or reentry_max is not None:
        def _reentry_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if reentry_min is not None:
                mask &= numeric >= float(reentry_min)
            if reentry_max is not None:
                mask &= numeric <= float(reentry_max)
            return mask

        label = f"reentry[{reentry_min},{reentry_max}]"
        rows.append(_mask_from_col("ReentryCount", _reentry_mask, label))

    margin_min = filters_cfg.get("margin_min")
    margin_max = filters_cfg.get("margin_max")
    if margin_min is not None or margin_max is not None:
        def _margin_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if margin_min is not None:
                mask &= numeric >= float(margin_min)
            if margin_max is not None:
                mask &= numeric <= float(margin_max)
            return mask

        label = f"margin[{margin_min},{margin_max}]"
        rows.append(_mask_from_col("margin", _margin_mask, label))

    return pd.DataFrame(rows)

def _print_block(
    title: str,
    df: pd.DataFrame,
    *,
    console: Any | None,
    table_class: Any | None,
    logger: logging.Logger,
    max_rows: int | None = None,
) -> None:
    if max_rows is not None:
        df = df.head(max_rows)
    if console and table_class:
        render_table(console, table_class, title, df.columns.tolist(), df.values.tolist())
    else:
        logger.info(
            "%s\n%s",
            title,
            df.to_string(index=False, max_colwidth=None, line_width=2000),
        )


def main() -> None:
    args = parse_args()
    
    def _safe_symbol(sym: str) -> str:
    # deja letras/números/._- y reemplaza lo demás por _
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    if args.model_out is None:
        sym = _safe_symbol(args.symbol)
        args.model_out = args.model_dir / f"{sym}_state_engine.pkl"
        
    rich_modules = try_import_rich() if not args.no_rich else None
    logger = setup_logging(args.logging_level)
    report_rank = REPORT_LEVELS[args.log_level]
    is_verbose = report_rank >= REPORT_LEVELS["verbose"]
    is_debug = report_rank >= REPORT_LEVELS["debug"]
    warnings_state: dict[str, bool] = {}
    logging.getLogger("state_engine.context_features").setLevel(logging.WARNING)

    console = None
    use_rich = bool(rich_modules)
    if use_rich and rich_modules:
        console = rich_modules["Console"]()

    bars_per_hour = {
        "M30": 2,
        "H1": 1,
        "H2": 0.5,
        "H4": 0.25,
    }
    derived_bars = math.ceil(args.window_hours * bars_per_hour[args.timeframe])
    if derived_bars < 4:
        raise ValueError(f"Context window too small: {derived_bars} bars (min=4).")
    logger.info(
        "timeframe=%s window_hours=%s derived_bars=%s",
        args.timeframe,
        args.window_hours,
        derived_bars,
    )

    timeframe_floor = {
        "M30": "30min",
        "H1": "h",
        "H2": "2h",
        "H4": "4h",
    }[args.timeframe]

    def step(name: str) -> float:
        logger.info("stage=%s", name)
        return datetime.now(tz=timezone.utc).timestamp()

    def step_done(start_ts: float) -> float:
        return datetime.now(tz=timezone.utc).timestamp() - start_ts

    stage_start = step("descarga_tf")
    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv = connector.obtener_ohlcv(args.symbol, args.timeframe, fecha_inicio, fecha_fin)
    
    # 1) Hora servidor MT5 (inferida desde último tick)
    server_now = connector.server_now(args.symbol).tz_localize(None)
    
    # 2) Guardrail correcto: usar solo velas cerradas según hora servidor
    #    La vela en formación empieza en server_now.floor(timeframe)
    cutoff = server_now.floor(timeframe_floor)
    ohlcv = ohlcv[ohlcv.index < cutoff]
    
    # 3) Timestamp última vela usada
    last_bar_ts = pd.Timestamp(ohlcv.index.max()).tz_localize(None)
    
    # 4) Edad en minutos (server_now - last_bar_ts)
    bar_age_minutes = (server_now - last_bar_ts).total_seconds() / 60.0
    
    # 5) Diagnóstico: si el tick está viejo, esto será grande
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    tick_age_min_utc = (now_utc - server_now).total_seconds() / 60.0

    elapsed_download = step_done(stage_start)
    logger.info("download_rows=%s elapsed=%.2fs", len(ohlcv), elapsed_download)

    stage_start = step("build_dataset")
    feature_config = FeatureConfig(window=derived_bars)
    builder = DatasetBuilder(feature_config=feature_config)
    artifacts = builder.build(ohlcv)
    full_features = artifacts.full_features
    features_raw = artifacts.features
    labels_raw = artifacts.labels
    elapsed_build = step_done(stage_start)
    logger.info(
        "features_raw=%s labels_raw=%s elapsed=%.2fs",
        len(features_raw),
        len(labels_raw),
        elapsed_build,
    )

    stage_start = step("align_and_clean")
    aligned = features_raw.join(labels_raw.rename("label"), how="inner")
    dropped_nan = int(aligned.isna().any(axis=1).sum())
    aligned = aligned.dropna()
    features = aligned.drop(columns=["label"])
    labels = aligned["label"].astype(int)
    elapsed_clean = step_done(stage_start)
    logger.info("aligned_samples=%s dropped_nan=%s elapsed=%.2fs", len(aligned), dropped_nan, elapsed_clean)
    logger.info("n_features=%s feature_names=%s", features.shape[1], features.columns.tolist())

    if len(features) < args.min_samples:
        raise RuntimeError(f"Not enough samples to train: {len(features)} < {args.min_samples}")

    stage_start = step("split")
    n_total = len(features)
    n_train = int(n_total * args.split_ratio)
    features_train = features.iloc[:n_train]
    labels_train = labels.iloc[:n_train]
    features_test = features.iloc[n_train:]
    labels_test = labels.iloc[n_train:]
    elapsed_split = step_done(stage_start)
    logger.info(
        "n_train=%s n_test=%s split_ratio=%.2f elapsed=%.2fs",
        len(features_train),
        len(features_test),
        args.split_ratio,
        elapsed_split,
    )

    label_order = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]

    stage_start = step("train_model")
    model_config = StateEngineModelConfig(
        class_weight="balanced" if args.class_weight_balanced else None,
    )
    model = StateEngineModel(model_config)
    model.fit(features_train, labels_train)
    elapsed_train = step_done(stage_start)
    logger.info("train_elapsed=%.2fs", elapsed_train)

    stage_start = step("evaluate")
    preds_test = model.predict_state(features_test).to_numpy()
    labels_test_np = labels_test.to_numpy()
    matrix = confusion_matrix(labels_test_np, preds_test, label_order)
    acc = float(np.mean(preds_test == labels_test_np)) if len(labels_test_np) else 0.0
    f1 = f1_macro(matrix)
    elapsed_eval = step_done(stage_start)
    logger.info("accuracy=%.4f f1_macro=%.4f elapsed=%.2fs", acc, f1, elapsed_eval)

    stage_start = step("predict_outputs")
    outputs = model.predict_outputs(features)
    elapsed_outputs = step_done(stage_start)
    logger.info("outputs_rows=%s elapsed=%.2fs", len(outputs), elapsed_outputs)

    quality_diagnostics = None
    quality_warnings = []
    if args.quality:
        quality_config, quality_sources, quality_warnings = load_quality_config(
            args.symbol,
            args.quality_config,
        )
        quality_features = full_features.reindex(outputs.index)
        quality_labels, quality_warnings_assign = assign_quality_labels(
            outputs["state_hat"],
            quality_features,
            quality_config,
        )
        outputs["quality_label"] = quality_labels
        quality_diagnostics = build_quality_diagnostics(
            outputs["state_hat"],
            quality_labels,
            quality_sources,
            [*quality_warnings, *quality_warnings_assign],
        )

    ctx_features = build_context_features(
        ohlcv,
        outputs,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if ctx_features is None:
        ctx_features = pd.DataFrame(index=outputs.index)
    else:
        ctx_features = ctx_features.reindex(outputs.index)
    if is_debug:
        logger.info("ctx_features_cols=%s", list(ctx_features.columns))
        logger.info("ctx_features_nonnull_tail=\n%s", ctx_features.tail(5).to_string())
    ctx_cols = [col for col in ctx_features.columns if col.startswith("ctx_")]
    if ctx_features.empty or not ctx_cols:
        logger.warning(
            "ctx_features empty or missing ctx_* columns; look_for filters will be inactive."
        )

    symbol_config = _load_symbol_config(args.symbol, logger)
    gating_config_meta = symbol_config

    # Extra reporting helpers
    state_hat_dist = class_distribution(outputs["state_hat"].to_numpy(), label_order)
    q_list = [0, 50, 75, 90, 95, 99, 100]
    breakmag_p = percentiles(full_features["BreakMag"], q_list) if "BreakMag" in full_features.columns else {str(q): None for q in q_list}
    reentry_p = percentiles(full_features["ReentryCount"], q_list) if "ReentryCount" in full_features.columns else {str(q): None for q in q_list}

    stage_start = step("phase_d")
    logger.info("phase_d_module=%s", GatingPolicy.__module__)
    gating_mod = importlib.import_module(GatingPolicy.__module__)
    logger.info("phase_d_file=%s", getattr(gating_mod, "__file__", None))
    logger.info("context_features_module=%s", build_context_features.__module__)
    context_mod = importlib.import_module(build_context_features.__module__)
    logger.info("context_features_file=%s", getattr(context_mod, "__file__", None))
    gating_policy = GatingPolicy()
    features_for_gating = full_features.join(ctx_features, how="left").reindex(outputs.index)
    validate_look_for_context_requirements(
        symbol_config,
        set(features_for_gating.columns) | set(outputs.columns),
        logger=logger,
    )
    gating = gating_policy.apply(
        outputs,
        features_for_gating,
        logger=logger,
        symbol=args.symbol,
        config_meta=gating_config_meta,
    )
    look_for_cols = list(gating.columns)
    look_for_any = gating.any(axis=1) if look_for_cols else pd.Series(False, index=outputs.index)
    look_for_context_frame = pd.DataFrame(index=outputs.index).join(gating[look_for_cols], how="left")

    def _attach_ctx_column(name: str, candidates: list[str]) -> None:
        series = None
        for candidate in candidates:
            if candidate in ctx_features.columns:
                series = ctx_features[candidate]
                break
            if candidate in outputs.columns:
                series = outputs[candidate]
                break
            if candidate in full_features.columns:
                series = full_features[candidate]
                break
        if series is not None:
            look_for_context_frame[name] = series.reindex(outputs.index)

    _attach_ctx_column("ctx_session_bucket", ["ctx_session_bucket"])
    _attach_ctx_column("ctx_state_age", ["ctx_state_age"])
    _attach_ctx_column("ctx_dist_vwap_atr", ["ctx_dist_vwap_atr"])

    phase_e_enabled = bool(args.enable_phase_e_metrics)
    if phase_e_enabled:
        logger.warning("PHASE E METRICS ENABLED: outcome-based telemetry active.")
    else:
        logger.info("PHASE D ONLY: outcome metrics disabled.")
    
    # --- "Última vela" para reporting
    last_idx = outputs.index.max()
    
    # LOOK_FOR final (cualquier regla true)
    last_allow = bool(look_for_any.loc[last_idx]) if last_idx in look_for_any.index else False

    look_for_coverage_rate = float(look_for_any.mean()) if len(gating) else 0.0
    elapsed_gating = step_done(stage_start)
    logger.info("look_for_coverage_rate=%.2f%% elapsed=%.2fs", look_for_coverage_rate * 100, elapsed_gating)

    if is_debug:
        table_class = rich_modules["Table"] if use_rich and console else None

        def _emit_table_debug(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
            _print_block(
                title,
                df,
                console=console if use_rich else None,
                table_class=table_class,
                logger=logger,
                max_rows=max_rows,
            )

        if ctx_cols:
            allow_cols = [col for col in gating.columns if col.startswith("LOOK_FOR_")]
            debug_frame = (
                outputs[["state_hat", "margin"]]
                .join(ctx_features[ctx_cols], how="left")
                .join(full_features[["BreakMag", "ReentryCount"]], how="left")
                .join(gating[allow_cols], how="left")
            )
            logger.info("[CTX DIAGNOSTIC TABLE] last_rows=10\n%s", debug_frame.tail(10).to_string())
        else:
            logger.warning("ctx_features missing ctx_* columns; skipping ctx diagnostic table.")

        phase_d_cfg = symbol_config.get("phase_d", {}) if isinstance(symbol_config, dict) else {}
        look_for_cfg = phase_d_cfg.get("look_fors", {}) if isinstance(phase_d_cfg, dict) else {}
        transition_cfg = look_for_cfg.get("LOOK_FOR_transition_failure", {}) if isinstance(look_for_cfg, dict) else {}
        look_for_config_rows = _look_for_filter_config_rows(symbol_config, "LOOK_FOR_transition_failure")
        if look_for_config_rows:
            look_for_config_df = pd.DataFrame(look_for_config_rows)
            _emit_table_debug("LOOK_FOR_CONFIG_EFFECTIVE", look_for_config_df)
        if isinstance(transition_cfg, dict) and transition_cfg:
            look_for_filter_counts = _look_for_context_filter_counts(
                look_for_context_frame,
                "LOOK_FOR_transition_failure",
                transition_cfg.get("filters", {}),
                logger,
            )
            if not look_for_filter_counts.empty:
                _emit_table_debug("LOOK_FOR_transition_context_counts", look_for_filter_counts)

    stage_start = step("save_model")
    metadata = {
        "feature_names_used": features.columns.tolist(),
        "label_names": {label.name: int(label) for label in StateLabels},
        "classes": [label.name for label in StateLabels],
        "feature_config": asdict(feature_config),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "timeframe": args.timeframe,
        "window_hours": args.window_hours,
        "derived_bars": derived_bars,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "accuracy": acc,
        "f1_macro": f1,
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }
    model.save(args.model_out, metadata=metadata)
    elapsed_save = step_done(stage_start)
    logger.info("model_path=%s elapsed=%.2fs", str(args.model_out), elapsed_save)

    # Baseline (majority class)
    label_dist = class_distribution(labels.to_numpy(), label_order)
    baseline = max(label_dist, key=lambda r: r["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "NA"
    baseline_pct = baseline["pct"] if baseline else 0.0
    logger.info("baseline_state=%s baseline_pct=%.2f", baseline_label, baseline_pct)

    importances = model.feature_importances().head(15)

    # Guardrail detail counts (kept as in your existing report)
    transition_filters = {}
    if isinstance(symbol_config, dict):
        phase_d_cfg = symbol_config.get("phase_d", {})
        look_for_cfg = phase_d_cfg.get("look_fors", {}) if isinstance(phase_d_cfg, dict) else {}
        transition_cfg = look_for_cfg.get("LOOK_FOR_transition_failure", {}) if isinstance(look_for_cfg, dict) else {}
        if isinstance(transition_cfg, dict):
            transition_filters = transition_cfg.get("filters", {}) if isinstance(transition_cfg.get("filters"), dict) else {}
    margin_min = transition_filters.get("margin_min")
    breakmag_min = transition_filters.get("breakmag_min")
    reentry_min = transition_filters.get("reentry_min")
    transition_rule = outputs["margin"] >= float(margin_min) if margin_min is not None else pd.Series(False, index=outputs.index)
    transition_break = full_features["BreakMag"] >= float(breakmag_min) if breakmag_min is not None else pd.Series(False, index=outputs.index)
    transition_reentry = full_features["ReentryCount"] >= float(reentry_min) if reentry_min is not None else pd.Series(False, index=outputs.index)

    table_class = rich_modules["Table"] if use_rich and console else None

    def _emit_table(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
        _print_block(
            title,
            df,
            console=console if use_rich else None,
            table_class=table_class if use_rich else None,
            logger=logger,
            max_rows=max_rows,
        )

    composition_state_df = pd.DataFrame(state_hat_dist)
    composition_quality_df = pd.DataFrame()
    if "quality_label" in outputs.columns:
        quality_counts = outputs["quality_label"].fillna("NA").astype(str).value_counts()
        composition_quality_df = quality_counts.rename_axis("quality_label").reset_index(
            name="count"
        )
        composition_quality_df["pct"] = (
            composition_quality_df["count"] / max(len(outputs), 1) * 100.0
        )

    confidence_summary_df = _build_confidence_summary(outputs)
    context_audit_df = _build_context_features_audit(ctx_features)
    look_for_coverage_df = _build_look_for_coverage_table(gating, outputs)
    phase_d_cfg = symbol_config.get("phase_d", {}) if isinstance(symbol_config, dict) else {}
    look_for_cfg = phase_d_cfg.get("look_fors", {}) if isinstance(phase_d_cfg, dict) else {}
    coverage_by_split_df = _build_coverage_by_split(gating, outputs, look_for_cfg)
    look_for_jaccard_df = _build_look_for_jaccard(gating)
    look_for_values = pd.unique(gating.to_numpy().ravel()) if len(gating.columns) else []
    look_for_values_set = {int(val) for val in look_for_values if pd.notna(val)}
    is_binary = look_for_values_set.issubset({0, 1})
    logger.info(
        "PHASE_D_LOOK_FOR_SANITY total_look_fors=%s total_rows=%s is_binary=%s values=%s",
        len(gating.columns),
        len(gating),
        is_binary,
        sorted(look_for_values_set),
    )

    _emit_table("PHASE_D_COMPOSITION_STATE", composition_state_df)
    if not composition_quality_df.empty:
        _emit_table("PHASE_D_COMPOSITION_QUALITY", composition_quality_df)
    if not confidence_summary_df.empty:
        _emit_table("PHASE_D_CONFIDENCE_SUMMARY", confidence_summary_df)
    _emit_table("PHASE_D_CONTEXT_FEATURES_AUDIT", context_audit_df)
    if not look_for_coverage_df.empty:
        _emit_table("PHASE_D_LOOK_FOR_COVERAGE", look_for_coverage_df)
    if not coverage_by_split_df.empty:
        _emit_table("PHASE_D_COVERAGE_BY_SPLIT", coverage_by_split_df)
    if not look_for_jaccard_df.empty:
        _emit_table("PHASE_D_LOOK_FOR_JACCARD_TOP", look_for_jaccard_df)

    if is_verbose or is_debug:
        summary_lines = [
            f"Symbol: {args.symbol}",
            f"Period: {args.start} -> {args.end}",
            f"Timeframe: {args.timeframe}",
            f"Window: {args.window_hours}h (~{derived_bars} bars)",
            f"Samples: {len(features)} (train={len(features_train)}, test={len(features_test)})",
            f"Baseline: {baseline_label} ({baseline_pct:.2f}%)",
            f"Look-for coverage rate: {look_for_coverage_rate*100:.2f}%",
        ]

        if console and use_rich:
            console.print("=== State Engine Training Summary ===")
            for line in summary_lines:
                console.print(line)
        else:
            logger.info("=== State Engine Training Summary ===\n%s", "\n".join(summary_lines))

    if is_verbose:
        state_hat_df = pd.DataFrame(state_hat_dist)
        _emit_table("Distribución state_hat (predicción)", state_hat_df)

        if quality_diagnostics is not None:
            quality_dist_df = pd.DataFrame(quality_diagnostics.distribution_rows)
            _emit_table("QUALITY_DISTRIBUTION", quality_dist_df)
            quality_splits_df = pd.DataFrame(quality_diagnostics.split_rows)
            _emit_table("QUALITY_SPLITS", quality_splits_df)

            incomplete_warnings = [w for w in quality_warnings if w.startswith("config_incomplete")]
            missing_total = 0
            for warning in incomplete_warnings:
                if "missing=" in warning:
                    missing_str = warning.split("missing=", 1)[1]
                    try:
                        missing_list = ast.literal_eval(missing_str)
                    except (ValueError, SyntaxError):
                        missing_list = []
                    missing_total += len(missing_list)
            quality_summary = pd.DataFrame(
                [
                    {
                        "config_incomplete": bool(incomplete_warnings),
                        "missing_count": missing_total,
                    }
                ]
            )
            _emit_table("QUALITY_CONFIG_EFFECTIVE (summary)", quality_summary)

    if is_debug and quality_diagnostics is not None:
        quality_config_df = pd.DataFrame(
            [
                {"key": key, "source": src}
                for key, src in sorted(quality_diagnostics.config_sources.items())
            ]
        )
        _emit_table("QUALITY_CONFIG_EFFECTIVE", quality_config_df)
        quality_persist_df = pd.DataFrame(quality_diagnostics.persistence_rows)
        _emit_table("QUALITY_PERSISTENCE", quality_persist_df)
        quality_warn_df = pd.DataFrame({"warning": quality_diagnostics.warnings or ["none"]})
        _emit_table("QUALITY_WARNINGS", quality_warn_df)

    if phase_e_enabled:
        from state_engine.phase_e_metrics import run_phase_e_reporting

        diagnostic_cfg = _load_diagnostic_table_config(args.symbol, logger)
        rescue_params = {
            "ev_k_bars": 2,
            "min_hvc_samples": 100,
            "rescue_top_k": args.rescue_top_k,
            "rescue_n_min": args.rescue_n_min,
            "rescue_delta_max": args.rescue_delta_max,
            "diagnostic_cfg": diagnostic_cfg,
        }
        run_phase_e_reporting(
            ohlcv=ohlcv,
            outputs=outputs,
            gating=gating,
            ctx_features=ctx_features,
            full_features=full_features,
            label_order=label_order,
            ev_min_split_samples=args.ev_min_split_samples,
            diagnose_rescue_scans=args.diagnose_rescue_scans,
            rescue_params=rescue_params,
            logger=logger,
            emit_table=_emit_table,
            warnings_state=warnings_state,
            is_verbose=is_verbose,
            is_debug=is_debug,
            symbol=args.symbol,
        )
    elif args.diagnose_rescue_scans:
        logger.warning("diagnose_rescue_scans ignored (Phase D only). Enable --enable-phase-e-metrics to use.")

    if is_debug:
        class_dist_df = pd.DataFrame(label_dist)
        _emit_table("Distribución de clases", class_dist_df)
        percentiles_df = pd.DataFrame(
            [
                {"metric": "BreakMag", **breakmag_p},
                {"metric": "ReentryCount", **reentry_p},
            ]
        )
        _emit_table("Percentiles (BreakMag / ReentryCount)", percentiles_df)
        metrics_df = pd.DataFrame([{"accuracy": acc, "f1_macro": f1}])
        _emit_table("Métricas (Test)", metrics_df)
        confusion_df = pd.DataFrame(
            matrix,
            columns=[l.name for l in label_order],
            index=[l.name for l in label_order],
        ).reset_index().rename(columns={"index": "Actual"})
        _emit_table("Matriz de confusión", confusion_df)
        importances_df = importances.reset_index()
        importances_df.columns = ["feature", "importance"]
        _emit_table("Top features (importance)", importances_df)
        look_for_summary_df = pd.DataFrame(
            [
                {
                    "look_for_rule": col,
                    "count": int(gating[col].sum()),
                    "pct": float(gating[col].mean() * 100.0) if len(gating) else 0.0,
                }
                for col in gating.columns
            ]
        )
        _emit_table("Resumen look_for", look_for_summary_df)
        transition_df = pd.DataFrame(
            [
                {"condition": "margin>=min", "count": int(transition_rule.sum())},
                {
                    "condition": "margin+breakmag",
                    "count": int((transition_rule & transition_break).sum()),
                },
                {
                    "condition": "margin+breakmag+reentry",
                    "count": int(
                        (transition_rule & transition_break & transition_reentry).sum()
                    ),
                },
            ]
        )
        _emit_table("Detalles transición (guardrails)", transition_df)

    # Optional: JSON report
    report_payload = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "baseline": {"state": baseline_label, "pct": baseline_pct},
        "metrics": {"accuracy": acc, "f1_macro": f1},
        "confusion_matrix": {
            "labels": [label.name for label in label_order],
            "matrix": matrix.tolist(),
        },
        "class_distribution": label_dist,
        "state_hat_distribution": state_hat_dist,
        "phase_d_outputs": {
            "composition_state": composition_state_df.to_dict(orient="records"),
            "composition_quality": composition_quality_df.to_dict(orient="records"),
            "confidence_summary": confidence_summary_df.to_dict(orient="records"),
            "context_features_audit": context_audit_df.to_dict(orient="records"),
            "look_for_coverage_table": look_for_coverage_df.to_dict(orient="records"),
        },
        "guardrail_percentiles": {
            "quantiles": q_list,
            "BreakMag": breakmag_p,
            "ReentryCount": reentry_p,
        },
        "feature_importances": importances.to_dict(),
        "phase_d": {
            "look_for_coverage_rate": look_for_coverage_rate,
        },
        "model_path": str(args.model_out),
        "metadata": metadata,
        "training": {"class_weight": model_config.class_weight},
        "phase_e_metrics_enabled": phase_e_enabled,
        "timings": {
            "download": elapsed_download,
            "build_dataset": elapsed_build,
            "align_and_clean": elapsed_clean,
            "split": elapsed_split,
            "train": elapsed_train,
            "evaluate": elapsed_eval,
            "predict_outputs": elapsed_outputs,
            "gating": elapsed_gating,
            "save_model": elapsed_save,
        },
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
