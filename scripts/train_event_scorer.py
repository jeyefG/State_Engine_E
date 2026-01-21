"""Train the Event Scorer model from MT5 data.

The scorer uses a triple-barrier continuous outcome (r_outcome) and reports
ranking metrics like lift@K to gauge whether top-ranked events outperform the
base rate. lift@K = precision@K / base_rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import textwrap

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import roc_auc_score

from state_engine.events import (
    EventDetectionConfig,
    EventFamily,
    detect_events,
    label_events,
)
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.labels import StateLabels
from state_engine.research import (
    build_family_variant_report,
    evaluate_research_variants,
    generate_research_variants,
    normalize_exploration_kind,
)
from state_engine.scoring import EventScorer, EventScorerBundle, EventScorerConfig, FeatureBuilder
from state_engine.session import SESSION_BUCKETS, get_session_bucket
from state_engine.config_loader import deep_merge, load_config
from state_engine.vwap import compute_vwap_mt5_daily, mt5_day_id
from state_engine.pipeline_phase_d import build_context_bundle


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train Event Scorer model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD) para descarga score/allow")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD) para descarga score/allow")
    parser.add_argument("--score-tf", default="M5", help="Timeframe para scoring (ej. M5, M15)")
    parser.add_argument("--allow-tf", default="H1", help="Timeframe legado para contexto allow (deprecated)")
    parser.add_argument(
        "--context-tf",
        default=None,
        help="Timeframe del contexto State/LOOK_FOR (ej. H2). Default: metadata del State Engine o allow-tf.",
    )
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--model-out", type=Path, default=None, help="Ruta de salida para Event Scorer")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para modelos",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio train/calibración")
    parser.add_argument("--k-bars", type=int, default=24, help="Ventana futura K para etiquetas")
    parser.add_argument(
        "--horizon-min",
        type=float,
        default=None,
        help="Horizonte temporal (minutos). Si se define, k_bars = horizon_min / score_tf_minutes.",
    )
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP proxy")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador de ATR para SL proxy")
    parser.add_argument("--r-thr", type=float, default=0.0, help="Umbral para label binario basado en r_outcome")
    parser.add_argument("--tie-break", default="distance", choices=["distance", "worst"], help="Tie-break TP/SL")
    parser.add_argument(
        "--meta-policy",
        default="on",
        choices=["on", "off"],
        help="Activar meta policy (gating superior) para regímenes operables",
    )
    parser.add_argument(
        "--force-meta-policy",
        action="store_true",
        help="Forzar meta policy incluso en research (solo si meta_policy=on).",
    )
    parser.add_argument(
        "--include-transition",
        default="on",
        choices=["on", "off"],
        help="Incluir eventos en régimen TRANSITION (default on)",
    )
    parser.add_argument("--meta-margin-min", type=float, default=0.10, help="Margen mínimo H1 para meta policy")
    parser.add_argument("--meta-margin-max", type=float, default=0.95, help="Margen máximo H1 para meta policy")
    parser.add_argument("--decision-n-min", type=int, default=200, help="N mínimo para decision tradeable")
    parser.add_argument("--decision-r-mean-min", type=float, default=0.02, help="r_mean mínimo para decision tradeable")
    parser.add_argument("--decision-winrate-min", type=float, default=0.52, help="Winrate mínimo para decision tradeable")
    parser.add_argument("--decision-p10-min", type=float, default=-0.2, help="p10 mínimo para decision tradeable")
    parser.add_argument("--fallback-min-samples", type=int, default=300, help="Mínimo de muestras post-meta para metrics")
    parser.add_argument("--config", default="on", choices=["on", "off"], help="Activar config YAML por símbolo")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "research", "production"],
        help="Modo de thresholds para reportes diagnósticos",
    )
    parser.add_argument(
        "--phase-e",
        action="store_true",
        help="Habilitar modo Fase E (telemetría sin decisiones ni thresholds).",
    )
    parser.add_argument(
        "--telemetry",
        default="screen",
        choices=["screen", "triage", "files"],
        help="Nivel de telemetría (screen, triage, files).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    parser.add_argument(
        "--vwap-check-date",
        default=None,
        help="Fecha (YYYY-MM-DD) para generar PNG de VWAP check.",
    )
    parser.add_argument(
        "--vwap-tz-server",
        default=None,
        help="Timezone del servidor MT5 para reset diario (ej. Europe/Athens).",
    )
    return parser.parse_args()


def _default_symbol_config(args: argparse.Namespace) -> dict:
    return {
        "event_scorer": {
            "score_tf": args.score_tf,
            "allow_tf": args.allow_tf,
            "train_ratio": args.train_ratio,
            "k_bars": args.k_bars,
            "reward_r": args.reward_r,
            "sl_mult": args.sl_mult,
            "r_thr": args.r_thr,
            "tie_break": args.tie_break,
            "include_transition": args.include_transition == "on",
            "meta_policy": {
                "enabled": args.meta_policy == "on",
                "meta_margin_min": args.meta_margin_min,
                "meta_margin_max": args.meta_margin_max,
            },
            "family_training": {
                "min_samples_train": 200,
            },
            "decision_thresholds": {
                "n_min": args.decision_n_min,
                "winrate_min": args.decision_winrate_min,
                "r_mean_min": args.decision_r_mean_min,
                "p10_min": args.decision_p10_min,
            },
            "research": {
                "enabled": False,
                "d1_anchor_hour": 0,
                "features": {
                    "session_bucket": False,
                    "hour_bucket": False,
                    "trend_context_D1": False,
                    "vol_context": False,
                },
                "meta_policy_enabled": False,
                "k_bars_grid": [args.k_bars],
                "exploration": {
                    "enabled": False,
                    "kind": "thresholds_only",
                    "seed": 7,
                    "max_variants": 25,
                    "decision_thresholds_grid": {
                        "n_min": [args.decision_n_min],
                        "winrate_min": [args.decision_winrate_min],
                        "r_mean_min": [args.decision_r_mean_min],
                        "p10_min": [args.decision_p10_min],
                    },
                },
                "diagnostics": {
                    "max_family_concentration": 0.6,
                    "min_temporal_dispersion": 0.3,
                    "min_calib_samples": 200,
                },
            },
        },
    }


def _resolve_decision_thresholds(config: dict, mode: str) -> dict:
    event_cfg = config.get("event_scorer", {}) if isinstance(config.get("event_scorer"), dict) else {}
    base_thresholds = event_cfg.get("decision_thresholds", {}) or {}
    mode_thresholds = (
        event_cfg.get("modes", {}).get(mode, {}).get("decision_thresholds", {}) if isinstance(event_cfg.get("modes"), dict) else {}
    )
    return deep_merge(base_thresholds, mode_thresholds or {})


def _resolve_research_config(config: dict, mode: str) -> dict:
    research_cfg = config.get("research", {}) if isinstance(config.get("research"), dict) else {}
    if not research_cfg:
        event_cfg = config.get("event_scorer", {}) if isinstance(config.get("event_scorer"), dict) else {}
        legacy_cfg = event_cfg.get("research", {}) if isinstance(event_cfg.get("research"), dict) else {}
        if legacy_cfg:
            research_cfg = legacy_cfg
    event_cfg = config.get("event_scorer", {}) if isinstance(config.get("event_scorer"), dict) else {}
    mode_cfg = {}
    if isinstance(event_cfg.get("modes"), dict):
        mode_cfg = event_cfg.get("modes", {}).get(mode, {})
    if isinstance(mode_cfg, dict):
        for key in (
            "enabled",
            "features",
            "diagnostics",
            "k_bars_grid",
            "k_bars_grid_by_tf",
            "exploration",
            "meta_policy_enabled",
        ):
            if key in mode_cfg:
                research_cfg[key] = mode_cfg.get(key)
    enabled = bool(research_cfg.get("enabled", False))
    features = research_cfg.get("features", {}) if isinstance(research_cfg.get("features"), dict) else {}
    diagnostics = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
    k_bars_grid = research_cfg.get("k_bars_grid")
    return {
        "enabled": enabled,
        "features": features,
        "diagnostics": diagnostics,
        "k_bars_grid": k_bars_grid,
        "k_bars_grid_by_tf": research_cfg.get("k_bars_grid_by_tf"),
        "d1_anchor_hour": research_cfg.get("d1_anchor_hour", 0),
        "exploration": research_cfg.get("exploration", {}) if isinstance(research_cfg.get("exploration"), dict) else {},
        "meta_policy_enabled": bool(research_cfg.get("meta_policy_enabled", False)),
    }


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("event_scorer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _required_allow_by_family() -> dict[str, str]:
    return {
        EventFamily.BALANCE_FADE.value: "LOOK_FOR_balance_fade",
        EventFamily.TRANSITION_FAILURE.value: "LOOK_FOR_transition_failure",
        EventFamily.TREND_PULLBACK.value: "LOOK_FOR_trend_pullback",
        EventFamily.TREND_CONTINUATION.value: "LOOK_FOR_trend_continuation",
    }

def _apply_event_allow_gates(
    events_df: pd.DataFrame,
    *,
    required_allow_by_family: dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    required_allow = events_df["family_id"].map(required_allow_by_family)
    missing_family_mask = required_allow.isna()
    if missing_family_mask.any():
        missing_families = sorted(set(events_df.loc[missing_family_mask, "family_id"]))
        logger.warning(
            "Phase E allow gate: dropping events with unmapped families=%s",
            missing_families,
        )
        events_df = events_df.loc[~missing_family_mask]
        if events_df.empty:
            return events_df
        required_allow = required_allow.loc[events_df.index]
    required_allow_values = sorted(set(required_allow_by_family.values()))
    missing_cols = sorted(set(required_allow_values) - set(events_df.columns))
    if missing_cols:
        raise ValueError(f"Missing required LOOK_FOR columns for Phase E gating: {missing_cols}")
    events = events_df.copy()
    events["required_allow"] = required_allow
    allow_values = pd.Series(0, index=events.index)
    for allow_name in required_allow_values:
        mask = required_allow.eq(allow_name)
        if not mask.any():
            continue
        allow_values.loc[mask] = pd.to_numeric(events.loc[mask, allow_name], errors="coerce").fillna(0).astype(int)
    keep_mask = allow_values.eq(1)
    dropped = events.loc[~keep_mask]
    if len(dropped):
        dropped_counts = dropped["required_allow"].value_counts().to_dict()
        logger.info(
            "Phase E allow gate: dropped=%s total=%s breakdown=%s",
            len(dropped),
            len(events),
            dropped_counts,
        )
    else:
        logger.info("Phase E allow gate: dropped=0 total=%s", len(events))
    coverage_rows = []
    for allow_name in required_allow_values:
        total = int(required_allow.eq(allow_name).sum())
        kept = int((keep_mask & required_allow.eq(allow_name)).sum())
        coverage_pct = (kept / total * 100.0) if total else 0.0
        coverage_rows.append(
            {"allow": allow_name, "events_total": total, "events_kept": kept, "coverage_pct": coverage_pct}
        )
    coverage_df = pd.DataFrame(coverage_rows)
    logger.info("Phase E allow coverage:\n%s", coverage_df.to_string(index=False))
    return events.loc[keep_mask].drop(columns=["required_allow"])


def _force_phase_e_allow_identity(
    events_df: pd.DataFrame,
    *,
    required_allow_by_family: dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    if events_df.empty:
        logger.info("Phase E allow identity forced: no events to process.")
        return events_df
    events = events_df.copy()
    required_allow = events["family_id"].map(required_allow_by_family)
    if required_allow.isna().any():
        missing = sorted(set(events.loc[required_allow.isna(), "family_id"]))
        raise ValueError(f"Phase E allow identity: missing required_allow mapping for families={missing}")
    events["required_allow"] = required_allow
    events["allow_id"] = required_allow
    allowed_set = set(required_allow_by_family.values())
    invalid_allow = sorted(set(events["allow_id"]) - allowed_set)
    if invalid_allow:
        raise ValueError(f"Phase E allow identity: unexpected allow_id values={invalid_allow}")
    if events["allow_id"].astype(str).str.contains(",", na=False).any():
        raise ValueError("Phase E allow identity: allow_id contains comma-separated combos.")
    if (events["allow_id"] == "LOOK_FOR_none").any():
        raise ValueError("Phase E allow identity: LOOK_FOR_none is not permitted.")
    allow_cols = [col for col in events.columns if col.startswith("LOOK_FOR_")]
    if allow_cols:
        multi_allow_count = int((events[allow_cols].fillna(0).sum(axis=1) >= 2).sum())
        logger.info(
            "Phase E allow identity forced: allow_id == required_allow | events=%s multi_allow_events=%s allow_cols=%s",
            len(events),
            multi_allow_count,
            allow_cols,
        )
    else:
        logger.info(
            "Phase E allow identity forced: allow_id == required_allow | events=%s no_allow_columns",
            len(events),
        )
    allow_counts = events["allow_id"].value_counts().to_dict()
    logger.info("Phase E allow identity summary: %s", allow_counts)
    return events


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


def _ensure_atr_14(df: pd.DataFrame) -> pd.Series:
    for col in ("atr_14", "atr", "ATR", "atr14"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").astype(float)
            return series.rename("atr_14")
    return _atr(df["high"], df["low"], df["close"], 14).rename("atr_14")


def _reset_index_for_export(df: pd.DataFrame) -> pd.DataFrame:
    index_name = df.index.name or "index"
    if index_name in df.columns:
        return df.reset_index(drop=True)
    return df.reset_index()


def _with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _safe_symbol(sym: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)


def _output_prefix(symbol: str, score_tf: str, run_id: str) -> str:
    return f"{_safe_symbol(symbol)}_{_safe_symbol(score_tf)}_{run_id}"


def _attach_output_metadata(
    df: pd.DataFrame,
    *,
    run_id: str,
    symbol: str,
    allow_tf: str,
    score_tf: str,
    context_tf: str | None = None,
    config_path: Path | None,
    mode: str,
    config_hash: str | None = None,
    prompt_version: str | None = None,
) -> pd.DataFrame:
    output = df.copy()
    metadata = [
        ("run_id", run_id),
        ("symbol_effective", symbol),
        ("symbol", symbol),
        ("context_tf", context_tf),
        ("allow_tf", allow_tf),
        ("score_tf", score_tf),
        ("mode", mode),
        ("config_path", str(config_path) if config_path is not None else None),
        ("config_hash", config_hash),
        ("prompt_version", prompt_version),
    ]
    for key, value in reversed(metadata):
        if key in output.columns:
            output[key] = value
        else:
            output.insert(0, key, value)
    return output


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe).upper()


def _resolve_context_tf(
    *,
    requested: str | None,
    model_metadata: dict[str, object],
    logger: logging.Logger,
) -> str:
    meta_tf = None
    if isinstance(model_metadata, dict):
        meta_tf = model_metadata.get("context_tf") or model_metadata.get("timeframe")
    candidates = [requested, meta_tf]
    for candidate in candidates:
        if candidate:
            resolved = _normalize_timeframe(str(candidate))
            logger.info(
                "context_tf resolved to %s (requested=%s meta=%s)",
                resolved,
                requested,
                meta_tf,
            )
            return resolved
    resolved = "H1"
    logger.info("context_tf fallback=%s (requested=%s meta=%s)", resolved, requested, meta_tf)
    return resolved


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


def _timeframe_floor_freq(timeframe: str) -> str:
    tf = _normalize_timeframe(timeframe)
    if tf not in _TIMEFRAME_FLOOR_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_FLOOR_MAP[tf]


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = _normalize_timeframe(timeframe)
    if tf.startswith("M") and tf[1:].isdigit():
        return int(tf[1:])
    if tf.startswith("H") and tf[1:].isdigit():
        return int(tf[1:]) * 60
    if tf.startswith("D") and tf[1:].isdigit():
        return int(tf[1:]) * 24 * 60
    if tf.startswith("W") and tf[1:].isdigit():
        return int(tf[1:]) * 7 * 24 * 60
    if tf.startswith("MN") and tf[2:].isdigit():
        return int(tf[2:]) * 30 * 24 * 60
    raise ValueError(f"Unsupported timeframe for horizon conversion: {timeframe}")


def _resolve_k_bars_by_tf(event_cfg: dict, score_tf: str, fallback: int) -> int:
    k_bars_by_tf = event_cfg.get("k_bars_by_tf")
    if isinstance(k_bars_by_tf, dict):
        target = _normalize_timeframe(score_tf)
        for key, value in k_bars_by_tf.items():
            if _normalize_timeframe(str(key)) == target and isinstance(value, int):
                return value
    k_bars = event_cfg.get("k_bars", fallback)
    return k_bars if isinstance(k_bars, int) else fallback


def _resolve_k_bars_grid_by_tf(research_cfg: dict, score_tf: str) -> list[int] | None:
    k_bars_grid_by_tf = research_cfg.get("k_bars_grid_by_tf")
    if isinstance(k_bars_grid_by_tf, dict):
        target = _normalize_timeframe(score_tf)
        for key, value in k_bars_grid_by_tf.items():
            if _normalize_timeframe(str(key)) == target and isinstance(value, list):
                return value
    return research_cfg.get("k_bars_grid")


def _mode_suffix(mode: str) -> str:
    mapping = {
        "baseline": "_baseline",
        "production": "_prod",
        "research": "_reas",
    }
    return mapping.get(mode, "_baseline")


def _format_interval(interval: pd.Interval | str) -> str:
    if isinstance(interval, pd.Interval):
        left = f"{interval.left:.2f}"
        right = f"{interval.right:.2f}"
        return f"m({left}-{right}]"
    if isinstance(interval, str):
        return f"m{interval}"
    return "mNA"


def _margin_bins(series: pd.Series, q: int = 3) -> pd.Series:
    bins = pd.Series(index=series.index, dtype="object")
    non_na = series.dropna()
    if non_na.empty:
        return bins
    try:
        bins.loc[non_na.index] = pd.qcut(non_na, q=q, duplicates="drop")
    except ValueError:
        rank_pct = non_na.rank(pct=True)
        bins.loc[non_na.index] = pd.cut(rank_pct, bins=q, include_lowest=True)
    return bins


def _make_margin_bin_rank(series: pd.Series) -> pd.Series:
    bins = pd.Series(index=series.index, dtype="object")
    non_na = series.dropna()
    if non_na.empty:
        return bins
    try:
        cuts = pd.qcut(non_na, q=3, duplicates="drop")
    except ValueError:
        cuts = pd.Series(index=non_na.index, dtype="object")
    if not isinstance(cuts, pd.Series) or cuts.isna().all():
        p33, p66 = non_na.quantile([1 / 3, 2 / 3])
        cuts = pd.cut(
            non_na,
            bins=[-np.inf, p33, p66, np.inf],
            include_lowest=True,
        )
    categories = list(pd.Series(cuts).cat.categories)
    mapping = {cat: f"m{idx}" for idx, cat in enumerate(categories)}
    bins.loc[non_na.index] = pd.Series(cuts).map(mapping)
    return bins


def _state_label(value: int | float) -> str:
    try:
        return StateLabels(int(value)).name
    except Exception:
        return "UNKNOWN"


def _vwap_zone(dist_to_vwap_atr: pd.Series) -> pd.Series:
    dist_abs = dist_to_vwap_atr.abs()
    zones = np.select(
        [
            dist_abs < 0.35,
            dist_abs < 0.80,
            dist_abs < 1.60,
        ],
        ["CENTER", "INNER", "OUTER"],
        default="EXTREME",
    )
    zones = pd.Series(zones, index=dist_to_vwap_atr.index)
    zones = zones.where(dist_to_vwap_atr.notna(), other="UNKNOWN")
    return zones


def _resolve_vwap_report_mode(
    events_df: pd.DataFrame,
    df_score_ctx: pd.DataFrame,
    event_config: EventDetectionConfig,
) -> str:
    attrs_mode = events_df.attrs.get("vwap_mode")
    if attrs_mode is None:
        attrs_mode = events_df.attrs.get("vwap_reset_mode_effective")
    if attrs_mode is None and "vwap_reset_mode_effective" in events_df.columns and not events_df.empty:
        attrs_mode = events_df["vwap_reset_mode_effective"].iloc[0]
    if attrs_mode is not None:
        return str(attrs_mode)
    if "vwap" in df_score_ctx.columns:
        return "provided (no metadata)"
    return "mt5_daily (no metadata)"


def _write_vwap_diagnostic_plot(
    ohlcv_score: pd.DataFrame,
    symbol: str,
    event_config: EventDetectionConfig,
    logger: logging.Logger,
    *,
    check_date: str | None = None,
) -> Path | None:
    if ohlcv_score.empty:
        logger.warning("VWAP diagnostic plot skipped: empty OHLCV input.")
        return None
    df = ohlcv_score.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df.index))
    vwap_series = compute_vwap_mt5_daily(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        real_vol_col="volume",
        tick_vol_col="tick_volume",
        tz_server=event_config.vwap_tz_server,
    )
    if vwap_series is None or vwap_series.empty:
        logger.warning("VWAP diagnostic plot skipped: compute_vwap returned no series.")
        return None
    df["vwap"] = vwap_series
    day_id = mt5_day_id(df, tz_server=event_config.vwap_tz_server)
    if day_id.empty:
        logger.warning("VWAP diagnostic plot skipped: no session key available.")
        return None
    if check_date:
        target_day = pd.to_datetime(check_date).date()
        day_mask = day_id.dt.date == target_day
        if not day_mask.any():
            logger.warning("VWAP diagnostic plot skipped: date %s not found.", check_date)
            return None
        session_label = str(target_day)
    else:
        last_day = day_id.iloc[-1]
        day_mask = day_id == last_day
        session_label = str(last_day.date() if hasattr(last_day, "date") else last_day)
    session_df = df.loc[day_mask]
    if session_df.empty:
        logger.warning("VWAP diagnostic plot skipped: last session empty.")
        return None
    diagnostics_dir = PROJECT_ROOT / "outputs" / "vwap_check"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    file_label = str(session_label).replace("/", "-").replace(" ", "_")
    output_path = diagnostics_dir / f"{_safe_symbol(symbol)}_{file_label}.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("VWAP diagnostic plot skipped (matplotlib missing): %s", exc)
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(session_df.index, session_df["close"], label="Close", color="#1f77b4")
    if "vwap" in session_df.columns:
        ax.plot(session_df.index, session_df["vwap"], label="VWAP", color="#ff7f0e")
    if {"high", "low"} <= set(session_df.columns):
        ax.fill_between(
            session_df.index,
            session_df["low"],
            session_df["high"],
            color="#1f77b4",
            alpha=0.1,
            label="High/Low",
        )
    ax.set_title(f"{symbol} VWAP check | session={session_label} | mode=mt5_daily")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("VWAP diagnostic plot saved: %s", output_path)
    mean_abs_diff = (session_df["close"] - session_df["vwap"]).abs().mean()
    range_mean = (session_df["high"] - session_df["low"]).mean()
    ratio = float(mean_abs_diff / range_mean) if range_mean and not np.isnan(range_mean) else float("nan")
    logger.info(
        "VWAP diagnostic ratio: mean_abs_diff=%.6f range_mean=%.6f ratio=%.6f",
        mean_abs_diff,
        range_mean,
        ratio,
    )
    return output_path

def _value_state(
    overlap_ratio: pd.Series,
    compression_ratio: pd.Series,
    range_width_atr: pd.Series,
) -> pd.Series:
    valid = overlap_ratio.notna() & compression_ratio.notna() & range_width_atr.notna()

    at_value = (overlap_ratio >= 0.55) & (compression_ratio >= 0.55)
    far_value = (overlap_ratio <= 0.25) & (range_width_atr >= 1.20)

    states = np.select(
        [at_value, far_value],
        ["AT_VALUE", "FAR_FROM_VALUE"],
        default="NEAR_VALUE",
    )
    states = pd.Series(states, index=overlap_ratio.index)
    states = states.where(valid, other="UNKNOWN")
    return states

def _expansion_state(
    range_atr: pd.Series,
    atr_ratio: pd.Series,
    breakout_dist_atr: pd.Series,
) -> pd.Series:
    valid = range_atr.notna() & atr_ratio.notna() & breakout_dist_atr.notna()
    strong = (range_atr >= 1.6) & (breakout_dist_atr >= 0.6) & (atr_ratio >= 1.1)
    none = (range_atr < 1.0) & (breakout_dist_atr < 0.25) & (atr_ratio <= 1.0)
    states = np.select(
        [strong, none],
        ["STRONG", "NONE"],
        default="MILD",
    )
    states = pd.Series(states, index=range_atr.index)
    states = states.where(valid, other="UNKNOWN")
    return states


PSEUDO_FEATURES = {
    "vwap_zone": ["CENTER", "INNER", "OUTER", "EXTREME", "UNKNOWN"],
    "value_state": ["AT_VALUE", "NEAR_VALUE", "FAR_FROM_VALUE", "UNKNOWN"],
    "expansion_state": ["NONE", "MILD", "STRONG", "UNKNOWN"],
}


def _encode_pseudo_features(pseudo_features: pd.DataFrame) -> pd.DataFrame:
    encoded_parts: list[pd.DataFrame] = []
    for feature_name, categories in PSEUDO_FEATURES.items():
        series = pseudo_features[feature_name].astype("category")
        series = series.cat.set_categories(categories)
        dummies = pd.get_dummies(series, prefix=f"pf_{feature_name}")
        for category in categories:
            col = f"pf_{feature_name}_{category}"
            if col not in dummies.columns:
                dummies[col] = 0
        encoded_parts.append(dummies)
    return pd.concat(encoded_parts, axis=1)


def _build_pseudo_features(
    events_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    atr_ratio: pd.Series,
) -> pd.DataFrame:
    if "dist_to_vwap_atr" in events_df.columns:
        dist_to_vwap_atr = pd.to_numeric(events_df["dist_to_vwap_atr"], errors="coerce")
    else:
        dist_to_vwap_atr = pd.Series(np.nan, index=events_df.index)
    overlap_ratio = pd.to_numeric(feature_df.get("overlap_ratio", pd.Series(np.nan, index=events_df.index)), errors="coerce")
    compression_ratio = pd.to_numeric(
        feature_df.get("compression_ratio", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    range_width_atr = pd.to_numeric(
        feature_df.get("range_width_atr", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    range_atr = pd.to_numeric(feature_df.get("range_atr", pd.Series(np.nan, index=events_df.index)), errors="coerce")
    breakout_dist_atr = pd.to_numeric(
        feature_df.get("breakout_dist_atr", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    atr_ratio_event = atr_ratio.reindex(feature_df.index)
    pseudo = pd.DataFrame(
        {
            "vwap_zone": _vwap_zone(dist_to_vwap_atr),
            "value_state": _value_state(overlap_ratio, compression_ratio, range_width_atr),
            "expansion_state": _expansion_state(range_atr, atr_ratio_event, breakout_dist_atr),
        },
        index=events_df.index,
    )
    return pseudo


def _format_state_mix(counts: pd.Series, state_order: list[str]) -> list[str]:
    total = counts.sum()
    lines = []
    for state in state_order:
        count = int(counts.get(state, 0))
        pct = (count / total * 100.0) if total else 0.0
        lines.append(f"{state}={count} ({pct:.2f}%)")
    return lines


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{value:.2f}%"


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator in (None, 0) or pd.isna(denominator):
        return float("nan")
    return float(numerator) / float(denominator)


def _vwap_validity_summary(
    ohlcv_score: pd.DataFrame,
    event_config: EventDetectionConfig,
) -> dict[str, float]:
    if ohlcv_score.empty:
        return {"valid_pct_all": float("nan"), "valid_pct_last_session": float("nan")}
    df = ohlcv_score.copy()
    vwap_series = compute_vwap_mt5_daily(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        real_vol_col="volume",
        tick_vol_col="tick_volume",
        tz_server=event_config.vwap_tz_server,
    )
    if vwap_series is None or vwap_series.empty:
        return {"valid_pct_all": float("nan"), "valid_pct_last_session": float("nan")}
    vwap_valid_mask = vwap_series.notna()
    valid_pct_all = float(vwap_valid_mask.mean() * 100.0) if len(vwap_valid_mask) else float("nan")
    day_id = mt5_day_id(df, tz_server=event_config.vwap_tz_server)
    if day_id.empty:
        return {"valid_pct_all": valid_pct_all, "valid_pct_last_session": float("nan")}
    last_session = day_id.iloc[-1]
    session_mask = day_id == last_session
    valid_pct_last = float(vwap_valid_mask[session_mask].mean() * 100.0) if session_mask.any() else float("nan")
    return {"valid_pct_all": valid_pct_all, "valid_pct_last_session": valid_pct_last}

def _conditional_edge_regime_table(
    events_diag: pd.DataFrame,
    pseudo_col: str,
    min_n: int,
    *,
    state_col: str,
) -> pd.DataFrame:
    columns = [
        state_col,
        "margin_bin",
        "allow_id",
        "family_id",
        pseudo_col,
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty or pseudo_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)

    grouped = events_diag.groupby(
        [state_col, "margin_bin", "allow_id", "family_id", pseudo_col],
        observed=True,
    )

    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()

    summary = summary[summary["n"] >= min_n]
    summary = summary.sort_values(
        ["r_mean", "winrate", "n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary[columns]

def _meta_policy_mask(
    events_df: pd.DataFrame,
    allow_cols: list[str],
    margin_min: float,
    margin_max: float,
    *,
    margin_col: str,
) -> pd.Series:
    allow_active = (
        events_df[allow_cols].fillna(0).sum(axis=1) > 0
        if allow_cols
        else pd.Series(False, index=events_df.index)
    )
    margin_ok = events_df[margin_col].between(margin_min, margin_max, inclusive="both")
    return allow_active & margin_ok


def _resolve_meta_policy_effective(
    args: argparse.Namespace,
    *,
    research_enabled: bool,
    research_cfg: dict,
) -> tuple[bool, str | None]:
    meta_policy_on = args.meta_policy == "on"
    reason = None
    if meta_policy_on and args.phase_e:
        meta_policy_on = False
        reason = "phase_e"
    if meta_policy_on and research_enabled:
        meta_policy_enabled = bool(research_cfg.get("meta_policy_enabled", False))
        if not (meta_policy_enabled or args.force_meta_policy):
            meta_policy_on = False
            reason = "research_disabled"
    return meta_policy_on, reason


def _coverage_table(
    events_diag: pd.DataFrame,
    df_score_ctx: pd.DataFrame | None,
    *,
    state_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns_global = [
        "events_total_post_meta",
        "events_per_day",
        "unique_days",
        "state_balance_pct",
        "state_trend_pct",
        "state_transition_pct",
        "margin_bin_m0_pct",
        "margin_bin_m1_pct",
        "margin_bin_m2_pct",
        "events_index_match_pct",
    ]
    columns_by = [state_col, "margin_bin", "events_count", "events_pct"]
    if events_diag.empty:
        empty_global = pd.DataFrame([{col: 0.0 for col in columns_global}])
        empty_global["events_index_match_pct"] = float("nan") if df_score_ctx is None else float("nan")
        empty_by = pd.DataFrame(columns=columns_by)
        return empty_global, empty_by
    total = len(events_diag)
    unique_days = events_diag.index.normalize().nunique()
    events_per_day = total / unique_days if unique_days else 0.0
    state_counts = events_diag[state_col].value_counts()
    margin_counts = events_diag["margin_bin"].value_counts()
    index_match = (
        float(events_diag.index.isin(df_score_ctx.index).mean()) if df_score_ctx is not None else float("nan")
    )
    global_row = {
        "events_total_post_meta": total,
        "events_per_day": events_per_day,
        "unique_days": unique_days,
        "state_balance_pct": (state_counts.get("BALANCE", 0) / total * 100.0) if total else 0.0,
        "state_trend_pct": (state_counts.get("TREND", 0) / total * 100.0) if total else 0.0,
        "state_transition_pct": (state_counts.get("TRANSITION", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m0_pct": (margin_counts.get("m0", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m1_pct": (margin_counts.get("m1", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m2_pct": (margin_counts.get("m2", 0) / total * 100.0) if total else 0.0,
        "events_index_match_pct": index_match,
    }
    coverage_global = pd.DataFrame([global_row])[columns_global]
    by_state = events_diag.groupby([state_col, "margin_bin"], observed=True).size().reset_index(name="events_count")
    by_state["events_pct"] = (by_state["events_count"] / total * 100.0) if total else 0.0
    by_state = by_state.sort_values([state_col, "margin_bin"]).reset_index(drop=True)
    return coverage_global, by_state[columns_by]


def _regime_edge_table(events_diag: pd.DataFrame, *, state_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        state_col,
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    grouped = events_diag.groupby([state_col, "margin_bin", "allow_id"], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    summary = summary.sort_values([state_col, "margin_bin", "allow_id"]).reset_index(drop=True)
    ranked_desc = summary.sort_values(
        ["r_mean", "n", "winrate", state_col, "margin_bin", "allow_id"],
        ascending=[False, False, False, True, True, True],
    )
    ranked_asc = summary.sort_values(
        ["r_mean", "n", "winrate", state_col, "margin_bin", "allow_id"],
        ascending=[True, False, False, True, True, True],
    )
    top10 = ranked_desc.head(10)
    bottom10 = ranked_asc.head(10)
    ranked = pd.concat([top10, bottom10], ignore_index=True)
    return summary[columns], ranked[columns]


def _temporal_splits() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30 23:59:59")),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31 23:59:59")),
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:59:59")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")),
        ("2026YTD", pd.Timestamp("2026-01-01"), pd.Timestamp.max),
    ]


def _stability_table(events_diag: pd.DataFrame, *, state_col: str) -> pd.DataFrame:
    columns = [
        "split",
        state_col,
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    for split_name, start_ts, end_ts in _temporal_splits():
        mask = (events_diag.index >= start_ts) & (events_diag.index <= end_ts)
        split_events = events_diag.loc[mask]
        if split_events.empty:
            continue
        grouped = split_events.groupby([state_col, "margin_bin", "allow_id"], observed=True)
        summary = grouped.agg(
            n=("win", "size"),
            winrate=("win", "mean"),
            r_mean=("r", "mean"),
            r_median=("r", "median"),
            p10=("r", lambda s: s.quantile(0.1)),
            p90=("r", lambda s: s.quantile(0.9)),
        ).reset_index()
        summary.insert(0, "split", split_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(["split", state_col, "margin_bin", "allow_id"]).reset_index(drop=True)
    return table[columns]


def _decision_table(events_diag: pd.DataFrame, thresholds: argparse.Namespace, *, state_col: str) -> pd.DataFrame:
    columns = [
        state_col,
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
        "decision",
        "decision_reason",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby([state_col, "margin_bin", "allow_id"], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    decisions = []
    for _, row in summary.iterrows():
        reasons = []
        if row["n"] < thresholds.decision_n_min:
            reasons.append(f"n<{thresholds.decision_n_min}")
        if row["r_mean"] < thresholds.decision_r_mean_min:
            reasons.append(f"r_mean<{thresholds.decision_r_mean_min}")
        if row["winrate"] < thresholds.decision_winrate_min:
            reasons.append(f"winrate<{thresholds.decision_winrate_min}")
        if thresholds.decision_p10_min is not None and row["p10"] < thresholds.decision_p10_min:
            reasons.append(f"p10<{thresholds.decision_p10_min}")
        decision = "TRADEABLE" if not reasons else "NO_TRADEABLE"
        decisions.append(
            {
                "decision": decision,
                "decision_reason": "|".join(reasons),
            }
        )
    decision_df = pd.DataFrame(decisions)
    summary = pd.concat([summary, decision_df], axis=1)
    summary = summary.sort_values([state_col, "margin_bin", "allow_id"]).reset_index(drop=True)
    return summary[columns]


def _session_bucket_series(
    events_index: pd.DatetimeIndex,
    symbol: str,
    df_score_ctx: pd.DataFrame | None,
) -> pd.Series:
    if df_score_ctx is not None and "pf_session_bucket" in df_score_ctx.columns:
        return df_score_ctx["pf_session_bucket"].reindex(events_index)
    return pd.Series(
        [get_session_bucket(ts, symbol) for ts in events_index],
        index=events_index,
        name="pf_session_bucket",
    )


def _build_research_summary_from_grid(grid_results: pd.DataFrame, *, phase_e: bool) -> dict[str, object]:
    if grid_results.empty:
        return {
            "qualified_variants": 0,
            "top_5_variants_by_r_mean": [],
            "verdict": "TELEMETRY_ONLY" if phase_e else "NO LOCAL EDGE DETECTED",
        }
    qualified_count = (
        0 if phase_e else int(grid_results["qualified"].sum()) if "qualified" in grid_results.columns else 0
    )
    top_rows = (
        grid_results.sort_values(["r_mean", "winrate", "n_total"], ascending=[False, False, False])
        .head(5)
        .reset_index(drop=True)
    )
    top_rows = top_rows[
        [
            "variant_id",
            "variant_type",
            "k_bars",
            "n_min",
            "winrate_min",
            "r_mean_min",
            "p10_min",
            "n_total",
            "winrate",
            "r_mean",
            "p10",
        ]
    ]
    if phase_e:
        verdict = "TELEMETRY_ONLY"
    elif qualified_count > 0:
        verdict = "LOCAL EDGE DETECTED — CANDIDATE FOR PROD SPECIALIZATION"
    else:
        verdict = "NO LOCAL EDGE DETECTED"
    return {
        "qualified_variants": qualified_count,
        "top_5_variants_by_r_mean": top_rows.to_dict(orient="records"),
        "verdict": verdict,
    }


def _empty_research_variant_report(
    variants: list,
    reason: str,
) -> pd.DataFrame:
    rows = []
    for variant in variants:
        rows.append(
            {
                "variant_id": variant.variant_id,
                "variant_type": variant.variant_type,
                "k_bars": int(variant.k_bars),
                "n_min": int(variant.n_min),
                "winrate_min": float(variant.winrate_min),
                "r_mean_min": float(variant.r_mean_min),
                "p10_min": float(variant.p10_min) if variant.p10_min is not None else None,
                "n_total": 0,
                "n_train": 0,
                "n_test": 0,
                "allow_rate": None,
                "winrate": np.nan,
                "r_mean": np.nan,
                "p10": np.nan,
                "lift10": np.nan,
                "lift20": np.nan,
                "lift50": np.nan,
                "spearman": np.nan,
                "family_concentration_top10": np.nan,
                "temporal_dispersion": np.nan,
                "score_tail_slope": None,
                "qualified": False,
                "fail_reason": reason,
                "research_status": "RESEARCH_UNSTABLE",
            }
        )
    return pd.DataFrame(rows)


def _persist_research_outputs(
    model_dir: Path,
    symbol: str,
    allow_tf: str,
    score_tf: str,
    context_tf: str,
    run_id: str,
    config_path: Path | None,
    grid_results: pd.DataFrame,
    family_results: pd.DataFrame | None,
    summary_payload: dict[str, object],
    research_cfg: dict,
    config_hash: str | None,
    prompt_version: str | None,
    baseline_thresholds: dict[str, float | int | None],
    baseline_thresholds_source: str,
    mode_suffix: str,
    research_mode: bool,
    mode: str,
    phase_e: bool,
) -> None:
    if not research_mode:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = _output_prefix(symbol, score_tf, run_id)
    grid_path = model_dir / f"research_grid_results_{output_prefix}{mode_suffix}.csv"
    grid_export = grid_results
    family_export = family_results
    if phase_e:
        if "qualified" in grid_export.columns:
            grid_export = grid_export.rename(columns={"qualified": "qualified_telemetry"})
        if family_export is not None and "qualified" in family_export.columns:
            family_export = family_export.rename(columns={"qualified": "qualified_telemetry"})

    grid_output = _attach_output_metadata(
        grid_export,
        run_id=run_id,
        symbol=symbol,
        allow_tf=allow_tf,
        score_tf=score_tf,
        context_tf=context_tf,
            config_path=config_path,
            mode=mode,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
    grid_output.to_csv(grid_path, index=False)
    if family_export is not None:
        families_path = model_dir / f"research_grid_families_{output_prefix}{mode_suffix}.csv"
        families_output = _attach_output_metadata(
            family_export,
            run_id=run_id,
            symbol=symbol,
            allow_tf=allow_tf,
            score_tf=score_tf,
            context_tf=context_tf,
            config_path=config_path,
            mode=mode,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        families_output.to_csv(families_path, index=False)
    summary_path = model_dir / f"research_summary_{output_prefix}{mode_suffix}.json"
    payload = {
        "enabled": bool(research_cfg.get("enabled", False)),
        "run_id": run_id,
        "symbol": symbol,
        "allow_tf": allow_tf,
        "score_tf": score_tf,
        "mode": mode,
        "config_path": str(config_path) if config_path is not None else None,
        "qualified_variants": summary_payload.get("qualified_variants", 0),
        "top_5_variants_by_r_mean": summary_payload.get("top_5_variants_by_r_mean", []),
        "verdict": summary_payload.get("verdict", "NO LOCAL EDGE DETECTED"),
        "d1_anchor_hour": research_cfg.get("d1_anchor_hour", 0),
        "features_enabled": research_cfg.get("features", {}),
        "k_bars_grid": research_cfg.get("k_bars_grid"),
        "exploration": research_cfg.get("exploration", {}),
        "baseline_thresholds_source": baseline_thresholds_source,
        "baseline_thresholds": baseline_thresholds,
        "config_hash": config_hash,
        "prompt_version": prompt_version,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _session_conditional_edge_table(
    events_diag: pd.DataFrame,
    thresholds: argparse.Namespace,
    *,
    state_col: str,
    phase_e: bool,
) -> pd.DataFrame:
    columns = [
        "family_id",
        state_col,
        "margin_bin",
        "pf_session_bucket",
        "n",
        "winrate",
        "r_mean",
        "p10",
        "decision_reason",
    ]
    if events_diag.empty or "pf_session_bucket" not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(
        ["family_id", state_col, "margin_bin", "pf_session_bucket"],
        observed=True,
    )
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        p10=("r", lambda s: s.quantile(0.1)),
    ).reset_index()
    if phase_e:
        summary["decision_reason"] = ""
    else:
        decision_reasons = []
        for _, row in summary.iterrows():
            reasons = []
            if row["n"] < thresholds.decision_n_min:
                reasons.append(f"n<{thresholds.decision_n_min}")
            if row["r_mean"] < thresholds.decision_r_mean_min:
                reasons.append(f"r_mean<{thresholds.decision_r_mean_min}")
            if row["winrate"] < thresholds.decision_winrate_min:
                reasons.append(f"winrate<{thresholds.decision_winrate_min}")
            if thresholds.decision_p10_min is not None and row["p10"] < thresholds.decision_p10_min:
                reasons.append(f"p10<{thresholds.decision_p10_min}")
            decision_reasons.append("|".join(reasons))
        summary["decision_reason"] = decision_reasons
    summary = summary.sort_values(
        ["family_id", state_col, "margin_bin", "pf_session_bucket"],
    ).reset_index(drop=True)
    return summary[columns]


def _format_table_block(title: str, df_or_dict: pd.DataFrame | dict) -> str:
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    else:
        df = df_or_dict
    if df.empty:
        body = "(no rows)"
    else:
        body = df.to_string(index=False)
    return f"{title}\n{body}"


def _apply_fallback_guardrails(events_total_post_meta: int, fallback_min_samples: int) -> bool:
    return events_total_post_meta < fallback_min_samples


def _print_research_summary_block(
    session_edge: pd.DataFrame,
    *,
    state_col: str,
    phase_e: bool,
) -> dict[str, object]:
    min_n = 150
    min_r_mean = 0.05
    min_winrate = 0.55
    if phase_e or session_edge.empty:
        qualifying = pd.DataFrame()
    else:
        qualifying = session_edge[
            (session_edge["n"] >= min_n)
            & (session_edge["r_mean"] >= min_r_mean)
            & (session_edge["winrate"] >= min_winrate)
        ]
    qualifying_count = int(len(qualifying))
    if session_edge.empty:
        top_rows = pd.DataFrame()
    else:
        top_rows = (
            session_edge.sort_values(
                ["r_mean", "winrate", "n"],
                ascending=[False, False, False],
            )
            .head(5)
            .reset_index(drop=True)
        )
    verdict = "TELEMETRY_ONLY" if phase_e else (
        "LOCAL EDGE DETECTED — CANDIDATE FOR PROD SPECIALIZATION"
        if qualifying_count > 0
        else "NO LOCAL EDGE DETECTED"
    )
    return {
        "qualified_session_buckets": qualifying_count,
        "top_5_cells_by_r_mean": top_rows.to_dict(orient="records"),
        "verdict": verdict,
    }


def _save_model_if_ready(
    is_fallback: bool,
    scorer: EventScorerBundle,
    path: Path,
    metadata: dict[str, object],
) -> bool:
    if is_fallback:
        return False
    scorer.save(path, metadata=metadata)
    return True


def _build_allow_id(events_df: pd.DataFrame, allow_cols: list[str]) -> pd.Series:
    if not allow_cols:
        return pd.Series("LOOK_FOR_none", index=events_df.index)
    allow_active = events_df[allow_cols].fillna(0).astype(int)
    allow_id = allow_active.apply(
        lambda row: ",".join(sorted([col for col, val in row.items() if val == 1])) or "LOOK_FOR_none",
        axis=1,
    )
    return allow_id


def _redistribution_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["family_id", feature_col, "n", "pct_of_family"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = (
        events_diag.groupby(["family_id", feature_col], observed=True)
        .size()
        .reset_index(name="n")
    )
    grouped["pct_of_family"] = (
        grouped["n"] / grouped.groupby("family_id")["n"].transform("sum") * 100.0
    ).round(4)
    grouped = grouped.sort_values(["family_id", feature_col]).reset_index(drop=True)
    return grouped[columns]


def _conditional_edge_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["family_id", feature_col, "n", "winrate", "r_mean", "p10", "p90"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(["family_id", feature_col], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    summary = summary.sort_values(["family_id", feature_col]).reset_index(drop=True)
    return summary[columns]


def _pseudo_temporal_splits() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:59:59")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")),
        ("2026YTD", pd.Timestamp("2026-01-01"), pd.Timestamp.max),
    ]


def _pseudo_stability_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["split", "family_id", feature_col, "n", "winrate", "r_mean", "p10", "p90"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    for split_name, start_ts, end_ts in _pseudo_temporal_splits():
        mask = (events_diag.index >= start_ts) & (events_diag.index <= end_ts)
        split_events = events_diag.loc[mask]
        if split_events.empty:
            continue
        grouped = split_events.groupby(["family_id", feature_col], observed=True)
        summary = grouped.agg(
            n=("win", "size"),
            winrate=("win", "mean"),
            r_mean=("r", "mean"),
            p10=("r", lambda s: s.quantile(0.1)),
            p90=("r", lambda s: s.quantile(0.9)),
        ).reset_index()
        summary.insert(0, "split", split_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(["split", "family_id", feature_col]).reset_index(drop=True)
    return table[columns]


def _session_bucket_distribution(df_score_ctx: pd.DataFrame) -> pd.DataFrame:
    if df_score_ctx.empty:
        return pd.DataFrame(columns=["symbol", "pf_session_bucket", "n", "pct"])
    df = df_score_ctx.copy()
    if "pf_session_bucket" not in df.columns:
        symbol_series = df.get("symbol", pd.Series("UNKNOWN", index=df.index))
        df["pf_session_bucket"] = [
            get_session_bucket(ts, symbol)
            for ts, symbol in zip(df.index, symbol_series)
        ]
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    counts = df.groupby(["symbol", "pf_session_bucket"]).size().reset_index(name="n")
    counts["pct"] = (
        counts["n"] / counts.groupby("symbol")["n"].transform("sum") * 100.0
    ).round(2)
    counts = counts.sort_values(["symbol", "n"], ascending=[True, False]).reset_index(drop=True)
    return counts[["symbol", "pf_session_bucket", "n", "pct"]]


def _merge_asof_features(m5_index: pd.DatetimeIndex, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=m5_index)
    if getattr(m5_index, "tz", None) is not None:
        m5_index = m5_index.tz_localize(None)
    m5_df = pd.DataFrame({"time": m5_index})
    feats = features.copy()
    if getattr(feats.index, "tz", None) is not None:
        feats.index = feats.index.tz_localize(None)
    feats = feats.reset_index().rename(columns={feats.index.name or "index": "time"})
    merged = pd.merge_asof(m5_df, feats.sort_values("time"), on="time", direction="backward")
    return merged.set_index("time")


def _build_research_context_features(
    df_score_ctx: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    symbol: str,
    research_cfg: dict,
) -> pd.DataFrame:
    feature_flags = research_cfg.get("features", {}) if isinstance(research_cfg.get("features"), dict) else {}
    logger = logging.getLogger("event_scorer")
    raw_anchor_hour = research_cfg.get("d1_anchor_hour", 0)
    is_valid_anchor_hour = (
        isinstance(raw_anchor_hour, (int, np.integer))
        and not isinstance(raw_anchor_hour, bool)
        and 0 <= raw_anchor_hour <= 23
    )
    try:
        parsed_anchor_hour = int(raw_anchor_hour)
    except (TypeError, ValueError):
        parsed_anchor_hour = 0
        logger.warning(
            "Invalid research.d1_anchor_hour=%r; defaulting to 0 and normalizing to [0, 23].",
            raw_anchor_hour,
        )
    else:
        if not is_valid_anchor_hour:
            logger.warning(
                "Invalid research.d1_anchor_hour=%r; normalizing to [0, 23] via modulo.",
                raw_anchor_hour,
            )
    d1_anchor_hour = parsed_anchor_hour if is_valid_anchor_hour else parsed_anchor_hour % 24
    parts: list[pd.DataFrame] = []
    index = df_score_ctx.index

    if feature_flags.get("session_bucket"):
        session_series = pd.Series(
            [get_session_bucket(ts, symbol) for ts in index],
            index=index,
            name="pf_session_bucket_research",
        )
        session_one_hot = pd.get_dummies(session_series, prefix="pf_session_bucket_research")
        for bucket in SESSION_BUCKETS:
            col = f"pf_session_bucket_research_{bucket}"
            if col not in session_one_hot.columns:
                session_one_hot[col] = 0
        parts.append(session_one_hot)

    if feature_flags.get("hour_bucket"):
        hours = pd.Series(index=index, data=index.hour, name="pf_hour_bucket")
        hour_one_hot = pd.get_dummies(hours, prefix="pf_hour_bucket")
        for hour in range(24):
            col = f"pf_hour_bucket_{hour}"
            if col not in hour_one_hot.columns:
                hour_one_hot[col] = 0
        parts.append(hour_one_hot)

    needs_d1 = feature_flags.get("trend_context_D1") or feature_flags.get("vol_context")
    if needs_d1 and not ohlcv_h1.empty:
        h1 = ohlcv_h1.copy()
        if getattr(h1.index, "tz", None) is not None:
            h1.index = h1.index.tz_localize(None)
        if not h1.index.empty:
            daily_min = h1.index.to_series().groupby(h1.index.normalize()).min()
            aligned_ratio = float((daily_min.dt.hour == (d1_anchor_hour % 24)).mean())
            if aligned_ratio < 0.8:
                logger.warning(
                    "Research D1 resample anchor may be misaligned (anchor_hour=%s, aligned_days=%.1f%%). "
                    "Consider setting research.d1_anchor_hour or verifying server TZ.",
                    d1_anchor_hour,
                    aligned_ratio * 100.0,
                )
        if d1_anchor_hour:
            h1 = h1.copy()
            h1.index = h1.index - pd.Timedelta(hours=d1_anchor_hour)
        h1_atr = _atr(h1["high"], h1["low"], h1["close"], 14).rename("atr_h1")
        h1_features = pd.DataFrame(
            {
                "h1_close": h1["close"],
                "atr_h1": h1_atr,
            },
            index=h1.index,
        ).shift(1)
        h1_merged = _merge_asof_features(index, h1_features)

        d1 = (
            h1[["open", "high", "low", "close"]]
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        if d1_anchor_hour:
            d1 = d1.copy()
            d1.index = d1.index + pd.Timedelta(hours=d1_anchor_hour)
        d1_ema200 = d1["close"].ewm(span=200, min_periods=200).mean().rename("ema200")
        d1_atr = _atr(d1["high"], d1["low"], d1["close"], 14).rename("atr_d1")
        d1_features = pd.DataFrame(
            {
                "d1_close": d1["close"],
                "d1_high": d1["high"],
                "d1_low": d1["low"],
                "d1_ema200": d1_ema200,
                "atr_d1": d1_atr,
            },
            index=d1.index,
        ).shift(1)
        d1_merged = _merge_asof_features(index, d1_features)

        if feature_flags.get("trend_context_D1"):
            ema_dist = (d1_merged["d1_close"] - d1_merged["d1_ema200"]) / d1_merged["atr_d1"].replace(0, np.nan)
            ema_side = np.where(ema_dist >= 0, "ABOVE", "BELOW")
            ema_side = pd.Series(ema_side, index=index).where(ema_dist.notna(), other="UNKNOWN")
            ema_side_one_hot = pd.get_dummies(ema_side, prefix="pf_d1_ema200_side")
            for side in ["ABOVE", "BELOW", "UNKNOWN"]:
                col = f"pf_d1_ema200_side_{side}"
                if col not in ema_side_one_hot.columns:
                    ema_side_one_hot[col] = 0
            dist_to_high = (d1_merged["d1_high"] - df_score_ctx["close"]) / d1_merged["atr_d1"].replace(0, np.nan)
            dist_to_low = (df_score_ctx["close"] - d1_merged["d1_low"]) / d1_merged["atr_d1"].replace(0, np.nan)
            trend_features = pd.DataFrame(
                {
                    "pf_d1_ema200_dist": ema_dist,
                    "pf_d1_dist_to_high_atr": dist_to_high,
                    "pf_d1_dist_to_low_atr": dist_to_low,
                },
                index=index,
            )
            parts.append(pd.concat([trend_features, ema_side_one_hot], axis=1))

        if feature_flags.get("vol_context"):
            atr_h1_norm = h1_merged["atr_h1"] / h1_merged["h1_close"].replace(0, np.nan)
            atr_h1_to_d1 = h1_merged["atr_h1"] / d1_merged["atr_d1"].replace(0, np.nan)
            vol_features = pd.DataFrame(
                {
                    "pf_atr_h1_norm": atr_h1_norm,
                    "pf_atr_h1_to_d1": atr_h1_to_d1,
                },
                index=index,
            )
            parts.append(vol_features)

    if not parts:
        return pd.DataFrame(index=index)
    return pd.concat(parts, axis=1)


def _score_shape_diagnostics(
    scores: pd.Series,
    families: pd.Series,
    timestamps: pd.DatetimeIndex,
    k_values: list[int],
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "k",
                "edge_decay",
                "temporal_dispersion",
                "family_concentration",
                "score_tail_slope",
            ]
        )

    def _edge_decay(top_scores: pd.Series) -> float:
        if top_scores.empty:
            return float("nan")
        top_sorted = top_scores.sort_values(ascending=False)
        top_first = float(top_sorted.iloc[0])
        top_last = float(top_sorted.iloc[-1])
        if top_first == 0:
            return float("nan")
        return (top_first - top_last) / abs(top_first)

    unique_days_total = timestamps.normalize().nunique() if len(timestamps) else 0
    p90 = float(scores.quantile(0.9))
    p99 = float(scores.quantile(0.99))
    score_tail_slope = (p99 - p90) / 0.09 if not np.isnan(p90) and not np.isnan(p99) else float("nan")

    rows = []
    for k in k_values:
        k_eff = min(k, len(scores))
        top_idx = scores.nlargest(k_eff).index
        top_scores = scores.loc[top_idx]
        top_families = families.loc[top_idx]
        top_days = top_idx.normalize().nunique() if len(top_idx) else 0
        temporal_dispersion = (top_days / unique_days_total) if unique_days_total else float("nan")
        family_counts = top_families.value_counts()
        family_concentration = (
            float(family_counts.max()) / k_eff if k_eff and not family_counts.empty else float("nan")
        )
        rows.append(
            {
                "k": k,
                "edge_decay": _edge_decay(top_scores),
                "temporal_dispersion": temporal_dispersion,
                "family_concentration": family_concentration,
                "score_tail_slope": score_tail_slope,
            }
        )
    return pd.DataFrame(rows)


def _research_guardrails(
    score_shape: pd.DataFrame,
    train_metrics: dict[str, float] | None,
    calib_metrics: dict[str, float] | None,
    diagnostics_cfg: dict,
    calib_samples: int,
) -> pd.DataFrame:
    if score_shape.empty:
        return pd.DataFrame(columns=["k", "status", "reasons"])
    max_family_concentration = diagnostics_cfg.get("max_family_concentration", 0.6)
    min_temporal_dispersion = diagnostics_cfg.get("min_temporal_dispersion", 0.3)
    max_score_tail_slope = diagnostics_cfg.get("max_score_tail_slope", 2.5)
    min_calib_samples = diagnostics_cfg.get("min_calib_samples", 200)
    train_lift = train_metrics.get("lift@20") if train_metrics else None
    calib_lift = calib_metrics.get("lift@20") if calib_metrics else None
    rows = []

    def _has_reason(reasons_list: list[str], prefix: str) -> bool:
        return any(reason.startswith(prefix) for reason in reasons_list)

    for _, row in score_shape.iterrows():
        reasons = []
        if calib_samples < min_calib_samples:
            reasons.append(f"LOW_CALIB_SAMPLES<{min_calib_samples}")
        if pd.notna(row["family_concentration"]) and row["family_concentration"] > max_family_concentration:
            reasons.append("FAMILY_CONCENTRATION_HIGH")
        if pd.notna(row["temporal_dispersion"]) and row["temporal_dispersion"] < min_temporal_dispersion:
            reasons.append("TEMPORAL_DISPERSION_LOW")
        if pd.notna(row["score_tail_slope"]) and row["score_tail_slope"] > max_score_tail_slope:
            reasons.append("SCORE_TAIL_STEEP")
        if train_lift is not None and calib_lift is not None:
            if (train_lift - calib_lift) > 0.2 and calib_lift < 1.05:
                reasons.append("TRAIN_IMPROVEMENT_ONLY")
        status = "RESEARCH_OK"
        if any(reason in {"SCORE_TAIL_STEEP", "TRAIN_IMPROVEMENT_ONLY"} for reason in reasons):
            status = "RESEARCH_OVERFIT_SUSPECT"
        elif _has_reason(reasons, "LOW_CALIB_SAMPLES") or any(
            reason in {"FAMILY_CONCENTRATION_HIGH", "TEMPORAL_DISPERSION_LOW"} for reason in reasons
        ):
            status = "RESEARCH_UNSTABLE"
        rows.append({"k": int(row["k"]), "status": status, "reasons": "|".join(reasons)})
    return pd.DataFrame(rows)


def _build_training_diagnostic_report(
    events_diag: pd.DataFrame,
    df_score_ctx: pd.DataFrame | None,
    thresholds: argparse.Namespace,
    *,
    state_col: str,
    phase_e: bool,
) -> dict[str, pd.DataFrame]:
    report = {}
    coverage_global, coverage_by = _coverage_table(events_diag, df_score_ctx, state_col=state_col)
    regime_full, regime_ranked = _regime_edge_table(events_diag, state_col=state_col)
    stability = _stability_table(events_diag, state_col=state_col)
    decision = _decision_table(events_diag, thresholds, state_col=state_col) if not phase_e else pd.DataFrame()
    session_edge = _session_conditional_edge_table(events_diag, thresholds, state_col=state_col, phase_e=phase_e)
    base_frames = [coverage_global, coverage_by, regime_full, regime_ranked, stability, session_edge]
    if not phase_e:
        base_frames.append(decision)
    report["coverage_global"] = coverage_global
    report["coverage_by_state_margin"] = coverage_by
    report["regime_edge_full"] = regime_full
    report["regime_edge_ranked"] = regime_ranked
    report["stability"] = stability
    if not phase_e:
        report["decision"] = decision
    report["session_conditional_edge"] = session_edge
    for pseudo_col in PSEUDO_FEATURES.keys():
        redistribution = _redistribution_table(events_diag, pseudo_col)
        conditional = _conditional_edge_table(events_diag, pseudo_col)
        stability_pseudo = _pseudo_stability_table(events_diag, pseudo_col)
        conditional_regime = _conditional_edge_regime_table(
            events_diag,
            pseudo_col,
            min_n=0 if phase_e else thresholds.decision_n_min,
            state_col=state_col,
        )
        report[f"conditional_edge_regime_{pseudo_col}"] = conditional_regime
        base_frames.append(conditional_regime)
        report[f"redistribution_{pseudo_col}"] = redistribution
        report[f"conditional_edge_{pseudo_col}"] = conditional
        report[f"stability_{pseudo_col}"] = stability_pseudo
        base_frames.extend([redistribution, conditional, stability_pseudo])
    for frame in base_frames:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_cols] = frame[numeric_cols].round(4)
    if df_score_ctx is not None:
        report["session_bucket_distribution"] = _session_bucket_distribution(df_score_ctx)
    return report


def _persist_diagnostic_tables(
    *,
    diagnostic_report: dict[str, pd.DataFrame],
    supply_funnel: pd.DataFrame,
    model_dir: Path,
    output_prefix: str,
    output_suffix: str,
    mode_suffix: str,
    run_id: str,
    symbol: str,
    allow_tf: str,
    score_tf: str,
    context_tf: str,
    config_path: Path | None,
    mode: str,
    config_hash: str | None,
    prompt_version: str | None,
    logger: logging.Logger,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "coverage_global": diagnostic_report.get("coverage_global", pd.DataFrame()),
        "coverage_by_state_margin": diagnostic_report.get("coverage_by_state_margin", pd.DataFrame()),
        "regime_edge_full": diagnostic_report.get("regime_edge_full", pd.DataFrame()),
        "regime_edge_ranked": diagnostic_report.get("regime_edge_ranked", pd.DataFrame()),
        "stability": diagnostic_report.get("stability", pd.DataFrame()),
        "session_conditional_edge": diagnostic_report.get("session_conditional_edge", pd.DataFrame()),
        "session_bucket_distribution": diagnostic_report.get("session_bucket_distribution", pd.DataFrame()),
        "supply_funnel": supply_funnel,
    }
    for name, frame in tables.items():
        output_path = _with_suffix(
            model_dir / f"{name}_{output_prefix}_event_scorer{mode_suffix}.csv",
            output_suffix,
        )
        output = _attach_output_metadata(
            frame,
            run_id=run_id,
            symbol=symbol,
            allow_tf=allow_tf,
            score_tf=score_tf,
            context_tf=context_tf,
            config_path=config_path,
            mode=mode,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        output.to_csv(output_path, index=False)
        logger.info("%s_out=%s", name, output_path)


def _baseline_state_summary(events_df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for state_name in ["BALANCE", "TREND", "TRANSITION"]:
        state_mask = events_df["state_label"] == state_name if "state_label" in events_df.columns else []
        state_events = events_df.loc[state_mask] if len(state_mask) else pd.DataFrame()
        n_events = int(len(state_events))
        r_mean = float(state_events["r_outcome"].mean()) if n_events else float("nan")
        summary[state_name] = {"n": n_events, "r_mean": r_mean}
    return summary


def _scorer_vs_baseline_summary(metrics_df: pd.DataFrame, *, phase_e: bool) -> dict[str, float | str]:
    if metrics_df.empty or "model" not in metrics_df.columns:
        return {
            "delta_r_mean@20": float("nan"),
            "lift@20_ratio": float("nan"),
            "spearman": float("nan"),
            "verdict": "NO_DATA",
        }
    scorer_row = metrics_df.loc[metrics_df["model"] == "SCORER"]
    baseline_row = metrics_df.loc[metrics_df["model"] == "BASELINE"]
    scorer = scorer_row.iloc[0] if not scorer_row.empty else metrics_df.iloc[0]
    baseline = baseline_row.iloc[0] if not baseline_row.empty else None
    delta_r = float(scorer.get("r_mean@20", float("nan")))
    if baseline is not None:
        delta_r = float(scorer.get("r_mean@20", float("nan"))) - float(baseline.get("r_mean@20", float("nan")))
    lift_ratio = _safe_ratio(
        float(scorer.get("lift@20", float("nan"))),
        float(baseline.get("lift@20", float("nan"))) if baseline is not None else None,
    )
    spearman = float(scorer.get("spearman", float("nan")))
    if phase_e:
        verdict = "TELEMETRY_ONLY"
    elif pd.notna(delta_r) and pd.notna(lift_ratio) and pd.notna(spearman):
        verdict = "EDGE" if (delta_r > 0 and lift_ratio >= 1.0 and spearman > 0) else "NO_EDGE"
    else:
        verdict = "NO_DATA"
    return {
        "delta_r_mean@20": delta_r,
        "lift@20_ratio": lift_ratio,
        "spearman": spearman,
        "verdict": verdict,
    }


def _best_worst_regimes(regime_df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    empty = {"regime_id": "NA", "n": 0, "delta_r_mean@20": float("nan"), "flag": "NA"}
    if regime_df.empty:
        return empty, empty
    best_idx = regime_df["delta_r_mean@20"].idxmax()
    worst_idx = regime_df["delta_r_mean@20"].idxmin()
    best_row = regime_df.loc[best_idx]
    worst_row = regime_df.loc[worst_idx]
    best = {
        "regime_id": best_row["regime_id"],
        "n": int(best_row.get("samples_calib", 0)),
        "delta_r_mean@20": float(best_row.get("delta_r_mean@20", float("nan"))),
        "flag": best_row.get("flag", "NA"),
    }
    worst = {
        "regime_id": worst_row["regime_id"],
        "n": int(worst_row.get("samples_calib", 0)),
        "delta_r_mean@20": float(worst_row.get("delta_r_mean@20", float("nan"))),
        "flag": worst_row.get("flag", "NA"),
    }
    return best, worst


def _emit_screen_telemetry(
    *,
    args: argparse.Namespace,
    run_id: str,
    mode: str,
    k_bars: int,
    allow_cutoff: pd.Timestamp,
    score_cutoff: pd.Timestamp,
    supply_funnel: dict[str, int],
    vwap_summary: dict[str, float],
    baseline_summary: dict[str, dict[str, float | int]],
    scorer_summary: dict[str, float | str],
    best_regime: dict[str, object],
    worst_regime: dict[str, object],
) -> None:
    header = (
        f"RUN {run_id} | symbol={args.symbol} mode={mode} "
        f"score_tf={args.score_tf} context_tf={args.context_tf} k_bars={k_bars} "
        f"start={args.start} end={args.end} "
        f"cutoffs:ctx={allow_cutoff} score={score_cutoff}"
    )
    print(header)
    funnel_line = (
        "Funnel: score_total={score_total} after_merge={after_merge} after_ctx_dropna={after_ctx_dropna} "
        "events_detected={events_detected} events_labeled={events_labeled} "
        "events_post_state_filter={events_post_state_filter} events_post_meta={events_post_meta}"
    ).format(**supply_funnel)
    print(funnel_line)
    vwap_valid_all = vwap_summary.get("valid_pct_all")
    vwap_valid_last = vwap_summary.get("valid_pct_last_session")
    warn_flag = " WARN" if pd.notna(vwap_valid_last) and vwap_valid_last < 95 else ""
    print(
        "VWAP validity: valid_pct_all={all_pct} valid_pct_last_session={last_pct}{warn}".format(
            all_pct=_format_pct(vwap_valid_all),
            last_pct=_format_pct(vwap_valid_last),
            warn=warn_flag,
        )
    )
    baseline_line = "Baseline r_mean (post-filter): "
    parts = []
    for state_name in ["BALANCE", "TREND", "TRANSITION"]:
        stats = baseline_summary.get(state_name, {"n": 0, "r_mean": float("nan")})
        parts.append(
            f"{state_name} n={stats['n']} r_mean={_format_metric(float(stats['r_mean']))}"
        )
    print(baseline_line + " | ".join(parts))
    print(
        "Scorer vs baseline: delta_r_mean@20={delta} lift@20_ratio={lift} spearman={spear} verdict={verdict}".format(
            delta=_format_metric(float(scorer_summary.get("delta_r_mean@20", float("nan")))),
            lift=_format_metric(float(scorer_summary.get("lift@20_ratio", float("nan")))),
            spear=_format_metric(float(scorer_summary.get("spearman", float("nan")))),
            verdict=scorer_summary.get("verdict", "NA"),
        )
    )
    print(
        "Best regime: {regime} n={n} delta_r_mean@20={delta} flag={flag}".format(
            regime=best_regime.get("regime_id", "NA"),
            n=best_regime.get("n", 0),
            delta=_format_metric(float(best_regime.get("delta_r_mean@20", float("nan")))),
            flag=best_regime.get("flag", "NA"),
        )
    )
    print(
        "Worst regime: {regime} n={n} delta_r_mean@20={delta} flag={flag}".format(
            regime=worst_regime.get("regime_id", "NA"),
            n=worst_regime.get("n", 0),
            delta=_format_metric(float(worst_regime.get("delta_r_mean@20", float("nan")))),
            flag=worst_regime.get("flag", "NA"),
        )
    )


def _emit_triage_telemetry(
    *,
    args: argparse.Namespace,
    run_id: str,
    mode: str,
    k_bars: int,
    allow_cutoff: pd.Timestamp,
    score_cutoff: pd.Timestamp,
    top_regimes: pd.DataFrame,
    bottom_regimes: pd.DataFrame,
    inverted_count: int,
    evaluated_regimes: int,
    stability_summary: dict[str, dict[str, dict[str, float | int]]],
    family_status_counts: dict[str, int],
) -> None:
    header = (
        f"RUN {run_id} | symbol={args.symbol} mode={mode} "
        f"score_tf={args.score_tf} context_tf={args.context_tf} k_bars={k_bars} "
        f"start={args.start} end={args.end} "
        f"cutoffs:ctx={allow_cutoff} score={score_cutoff}"
    )
    print(header)
    print("Top 5 regimes by delta_r_mean@20:")
    if top_regimes.empty:
        print("(no regimes)")
    else:
        for _, row in top_regimes.iterrows():
            print(
                f"+ {row['regime_id']} n={int(row['samples_calib'])} "
                f"delta_r_mean@20={_format_metric(float(row['delta_r_mean@20']))} flag={row['flag']}"
            )
    print("Bottom 5 regimes by delta_r_mean@20:")
    if bottom_regimes.empty:
        print("(no regimes)")
    else:
        for _, row in bottom_regimes.iterrows():
            print(
                f"- {row['regime_id']} n={int(row['samples_calib'])} "
                f"delta_r_mean@20={_format_metric(float(row['delta_r_mean@20']))} flag={row['flag']}"
            )
    print(f"Inverted ranks: {inverted_count}/{evaluated_regimes}")
    print("Stability (top 5 regimes):")
    if not stability_summary:
        print("(no stability data)")
    else:
        for regime_id, splits in stability_summary.items():
            parts = []
            for split_name in ["2025H1", "2025H2", "2026YTD"]:
                split_stats = splits.get(split_name, {})
                n_val = int(split_stats.get("n", 0))
                r_mean = _format_metric(float(split_stats.get("r_mean@20", float("nan"))))
                parts.append(f"{split_name} n={n_val} r_mean@20={r_mean}")
            print(f"* {regime_id} | " + " | ".join(parts))
    status_line = "Family training status: " + " | ".join(
        f"{status}={count}" for status, count in family_status_counts.items()
    )
    print(status_line)


def _run_training_for_k(
    args: argparse.Namespace,
    k_bars: int,
    output_suffix: str,
    output_prefix: str,
    run_id: str,
    config_path: Path | None,
    detected_events: pd.DataFrame,
    event_config: EventDetectionConfig,
    df_score_ctx: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    features_all: pd.DataFrame,
    atr_ratio: pd.Series,
    feature_builder: FeatureBuilder,
    score_total: int,
    score_ctx_merged: int,
    score_ctx_dropna: int,
    allow_cutoff: pd.Timestamp,
    score_cutoff: pd.Timestamp,
    vwap_summary: dict[str, float],
    min_samples_train: int,
    seed: int,
    scorer_out_base: Path,
    mode_suffix: str,
    mode_effective: str,
    research_mode: bool,
    research_cfg: dict,
    research_variants: list,
    baseline_thresholds: dict[str, float | int | None],
    baseline_thresholds_source: str,
    config_hash: str | None,
    prompt_version: str | None,
    state_model_path: Path,
    state_model_metadata: dict[str, object],
    context_nan_pct: dict[str, float],
    event_counts: dict[str, dict[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame | None] | None:
    logger = logging.getLogger("event_scorer")
    research_enabled = research_mode and bool(research_cfg.get("enabled", False))
    meta_policy_effective, meta_policy_reason = _resolve_meta_policy_effective(
        args,
        research_enabled=research_enabled,
        research_cfg=research_cfg,
    )
    logger.debug(
        "Train config | symbol=%s start=%s end=%s k_bars=%s reward_r=%s sl_mult=%s r_thr=%s meta_policy=%s include_transition=%s",
        args.symbol,
        args.start,
        args.end,
        k_bars,
        args.reward_r,
        args.sl_mult,
        args.r_thr,
        args.meta_policy,
        args.include_transition,
    )

    def _base_summary_payload(verdict: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "score_tf": args.score_tf,
            "allow_tf": args.allow_tf,
            "context_tf": args.context_tf,
            "state_model_path": str(state_model_path),
            "state_model_metadata_timeframe": (
                state_model_metadata.get("timeframe") if isinstance(state_model_metadata, dict) else None
            ),
            "config_path": str(config_path) if config_path is not None else None,
            "config_hash": config_hash,
            "prompt_version": prompt_version,
            "allow_cutoff": allow_cutoff,
            "context_cutoff": allow_cutoff,
            "m5_cutoff": score_cutoff,
            "score_cutoff": score_cutoff,
            "k_bars": k_bars,
            "horizon_min": args.horizon_min,
            "reward_r": args.reward_r,
            "sl_mult": args.sl_mult,
            "r_thr": args.r_thr,
            "tie_break": args.tie_break,
            "meta_policy": {
                "enabled": args.meta_policy == "on",
                "meta_margin_min": args.meta_margin_min,
                "meta_margin_max": args.meta_margin_max,
            },
            "meta_policy_effective": meta_policy_effective,
            "phase_e": args.phase_e,
            "verdict": verdict,
            "event_counts": event_counts,
            "context_columns": {
                "state_col": args.context_state_col,
                "margin_col": args.context_margin_col,
            },
            "context_nan_pct": context_nan_pct,
        }

    def _emit_telemetry(
        *,
        supply_funnel_payload: dict[str, int],
        baseline_summary: dict[str, dict[str, float | int]],
        scorer_summary: dict[str, float | str],
        regime_df: pd.DataFrame,
        family_summary: pd.DataFrame,
        scores: pd.Series | None = None,
        r_outcome: pd.Series | None = None,
        regime_series: pd.Series | None = None,
    ) -> None:
        if args.telemetry == "files":
            return
        best_regime, worst_regime = _best_worst_regimes(regime_df)
        if args.telemetry == "screen":
            _emit_screen_telemetry(
                args=args,
                run_id=run_id,
                mode=mode_effective,
                k_bars=k_bars,
                allow_cutoff=allow_cutoff,
                score_cutoff=score_cutoff,
                supply_funnel=supply_funnel_payload,
                vwap_summary=vwap_summary,
                baseline_summary=baseline_summary,
                scorer_summary=scorer_summary,
                best_regime=best_regime,
                worst_regime=worst_regime,
            )
            return
        top_regimes = (
            regime_df.sort_values("delta_r_mean@20", ascending=False).head(5)
            if not regime_df.empty
            else pd.DataFrame()
        )
        bottom_regimes = (
            regime_df.sort_values("delta_r_mean@20", ascending=True).head(5)
            if not regime_df.empty
            else pd.DataFrame()
        )
        inverted_count = int(regime_df["flag"].str.contains("RANK_INVERTED", na=False).sum()) if not regime_df.empty else 0
        evaluated_regimes = int(len(regime_df))
        stability_summary: dict[str, dict[str, dict[str, float | int]]] = {}
        if scores is not None and r_outcome is not None and regime_series is not None and not top_regimes.empty:
            for regime_id in top_regimes["regime_id"]:
                split_stats: dict[str, dict[str, float | int]] = {}
                for split_name, start_ts, end_ts in _pseudo_temporal_splits():
                    split_mask = (scores.index >= start_ts) & (scores.index <= end_ts)
                    regime_mask = regime_series == regime_id
                    idx = scores.index[split_mask & regime_mask]
                    n_samples = int(len(idx))
                    r_mean_20 = mean_r_topk(scores.loc[idx], r_outcome.loc[idx], 20) if n_samples else float("nan")
                    split_stats[split_name] = {"n": n_samples, "r_mean@20": r_mean_20}
                stability_summary[str(regime_id)] = split_stats
        status_counts = (
            family_summary["status"].value_counts().to_dict()
            if "status" in family_summary.columns and not family_summary.empty
            else {}
        )
        _emit_triage_telemetry(
            args=args,
            run_id=run_id,
            mode=mode_effective,
            k_bars=k_bars,
            allow_cutoff=allow_cutoff,
            score_cutoff=score_cutoff,
            top_regimes=top_regimes,
            bottom_regimes=bottom_regimes,
            inverted_count=inverted_count,
            evaluated_regimes=evaluated_regimes,
            stability_summary=stability_summary,
            family_status_counts=status_counts,
        )

    events = label_events(
        detected_events.copy(),
        ohlcv_score,
        k_bars,
        args.reward_r,
        args.sl_mult,
        r_thr=args.r_thr,
        tie_break=args.tie_break,
    )
    labeled_events = events.copy()
    labeled_total = len(labeled_events)
    dropna_mask = labeled_events[["label", "r_outcome"]].isna().any(axis=1)
    dropna_count = int(dropna_mask.sum())
    events = labeled_events.dropna(subset=["label", "r_outcome"])
    events = events.sort_index()

    if events.empty:
        logger.warning("No labeled events after filtering.")
        events_diag = pd.DataFrame(
            columns=[args.context_state_col, "allow_id", "margin", "r", "win", "margin_bin"]
        )
        events_diag.index = pd.DatetimeIndex([])
        is_fallback = _apply_fallback_guardrails(0, args.fallback_min_samples)
        if is_fallback:
            logger.warning(
                "Fallback diagnostics: events_total_post_meta=0 < fallback_min_samples=%s",
                args.fallback_min_samples,
            )
        diagnostic_report = _build_training_diagnostic_report(
            events_diag,
            df_score_ctx,
            thresholds=args,
            state_col=args.context_state_col,
            phase_e=args.phase_e,
        )
        supply_funnel = pd.DataFrame(
            [
                {
                    "score_total": score_total,
                    "after_merge": score_ctx_merged,
                    "after_dropna_ctx": score_ctx_dropna,
                    "events_detected": len(detected_events),
                    "events_labeled": labeled_total,
                    "events_post_state_filter": 0,
                    "events_post_meta": 0,
                }
            ]
        )
        supply_funnel_payload = {
            "score_total": score_total,
            "after_merge": score_ctx_merged,
            "after_ctx_dropna": score_ctx_dropna,
            "events_detected": len(detected_events),
            "events_labeled": labeled_total,
            "events_post_state_filter": 0,
            "events_post_meta": 0,
        }
        _persist_diagnostic_tables(
            diagnostic_report=diagnostic_report,
            supply_funnel=supply_funnel,
            model_dir=args.model_dir,
            output_prefix=output_prefix,
            output_suffix=output_suffix,
            mode_suffix=mode_suffix,
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
            logger=logger,
        )
        summary_payload = _base_summary_payload(
            "TELEMETRY_ONLY" if args.phase_e else "NO_LABELED_EVENTS"
        )
        summary_path = _with_suffix(
            args.model_dir / f"summary_{output_prefix}_event_scorer.json",
            output_suffix,
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)
        logger.info("summary_out=%s", summary_path)
        _emit_telemetry(
            supply_funnel_payload=supply_funnel_payload,
            baseline_summary=_baseline_state_summary(pd.DataFrame()),
            scorer_summary=_scorer_vs_baseline_summary(pd.DataFrame(), phase_e=args.phase_e),
            regime_df=pd.DataFrame(),
            family_summary=pd.DataFrame(),
        )
        if research_mode:
            _print_research_summary_block(
                diagnostic_report.get("session_conditional_edge", pd.DataFrame()),
                state_col=args.context_state_col,
                phase_e=args.phase_e,
            )
        if research_mode and research_variants:
            diagnostics_cfg = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
            report = evaluate_research_variants(
                variants=research_variants,
                scores=pd.Series([], dtype=float),
                labels=pd.Series([], dtype=float),
                r_outcome=pd.Series([], dtype=float),
                families=pd.Series([], dtype="object"),
                timestamps=pd.DatetimeIndex([]),
                train_count=0,
                calib_count=0,
                allow_rate=None,
                diagnostics_cfg=diagnostics_cfg,
                baseline_thresholds=baseline_thresholds,
            )
            family_report = build_family_variant_report(
                variants=research_variants,
                scores=pd.Series([], dtype=float),
                labels=pd.Series([], dtype=float),
                r_outcome=pd.Series([], dtype=float),
                families=pd.Series([], dtype="object"),
            )
            return report, family_report
        return None
    logger.info("Labeled events by family:\n%s", events["family_id"].value_counts().to_string())
    logger.info("Labeled events by type:\n%s", events["event_type"].value_counts().to_string())

    events_all = events.copy()
    allow_cols = [col for col in events_all.columns if col.startswith("LOOK_FOR_")]
    if not allow_cols:
        logger.warning("No LOOK_FOR_* columns found on events; meta policy will be empty.")
    margin_bins = _margin_bins(events_all[args.context_margin_col], q=3)
    margin_bin_label = margin_bins.map(_format_interval)
    state_label = events_all[args.context_state_col].map(_state_label)
    events_all["state_label"] = state_label
    events_all["margin_bin"] = margin_bin_label
    if args.phase_e:
        events_all = _force_phase_e_allow_identity(
            events_all,
            required_allow_by_family=required_allow_map,
            logger=logger,
        )
    else:
        events_all["allow_id"] = _build_allow_id(events_all, allow_cols)
    events_all["regime_id"] = (
        state_label.astype(str)
        + "|"
        + margin_bin_label.fillna("mNA")
        + "|"
        + events_all["allow_id"]
        + "|"
        + events_all["family_id"].astype(str)
        + "|"
        + events_all["event_type"].astype(str)
    )

    include_transition = args.include_transition == "on"
    events_state_filtered = (
        events_all if include_transition else events_all.loc[events_all["state_label"] != "TRANSITION"]
    )
    if not include_transition:
        logger.info(
            "STATE FILTER | include_transition=off removed=%s kept=%s",
            len(events_all) - len(events_state_filtered),
            len(events_state_filtered),
        )

    event_features_all = features_all.reindex(events_state_filtered.index)
    pseudo_features = _build_pseudo_features(events_state_filtered, event_features_all, atr_ratio)
    events_state_filtered = events_state_filtered.join(pseudo_features)
    pseudo_encoded = _encode_pseudo_features(pseudo_features)
    event_features_all = pd.concat([event_features_all, pseudo_encoded], axis=1)

    state_counts_events = events_state_filtered["state_label"].value_counts()
    logger.debug("State mix (post filter): %s", state_counts_events.to_dict())

    vwap_mode = _resolve_vwap_report_mode(detected_events, df_score_ctx, event_config)
    vwap_real_col = detected_events.attrs.get("vwap_real_volume_col")
    vwap_tick_col = detected_events.attrs.get("vwap_tick_volume_col")
    vwap_price_cols = detected_events.attrs.get("vwap_price_columns")
    dist_abs = events_all["dist_to_vwap_atr"].abs()
    vwap_quantiles = dist_abs.quantile([0.1, 0.5, 0.9, 0.99]).to_dict() if not dist_abs.empty else {}
    logger.debug(
        "VWAP sanity | mode=%s near_vwap_atr=%s price_cols=%s real_vol_col=%s tick_vol_col=%s dist_q=%s",
        vwap_mode,
        event_config.near_vwap_atr,
        vwap_price_cols,
        vwap_real_col,
        vwap_tick_col,
        {k: round(v, 4) for k, v in vwap_quantiles.items()} if vwap_quantiles else {},
    )

    meta_policy_on = meta_policy_effective
    if meta_policy_reason == "phase_e":
        logger.info("phase_e=ON -> meta_policy disabled (telemetry only)")
    elif meta_policy_reason == "research_disabled":
        logger.info("research=ON -> meta_policy disabled (telemetry only)")
    allow_active_series = (
        events_state_filtered[allow_cols].fillna(0).sum(axis=1) > 0
        if allow_cols
        else pd.Series(False, index=events_state_filtered.index)
    )
    allow_active_pct = float(allow_active_series.mean() * 100) if len(allow_active_series) else 0.0
    allow_id_top = events_state_filtered["allow_id"].value_counts().head(10)
    logger.debug("Allow sanity | allow_active_pct=%.2f allow_id_top=%s", allow_active_pct, allow_id_top.to_dict())
    margin_ok_series = events_state_filtered[args.context_margin_col].between(
        args.meta_margin_min,
        args.meta_margin_max,
        inclusive="both",
    )
    meta_mask = _meta_policy_mask(
        events_state_filtered,
        allow_cols,
        args.meta_margin_min,
        args.meta_margin_max,
        margin_col=args.context_margin_col,
    )
    events_meta = events_state_filtered.loc[meta_mask].copy()

    meta_keep_count = int(meta_mask.sum()) if len(meta_mask) else 0
    logger.info(
        "INFO | META | meta_policy=%s allow_active_count=%s margin_ok_count=%s meta_keep_count=%s",
        "ON" if meta_policy_on else "OFF",
        int(allow_active_series.sum()),
        int(margin_ok_series.sum()),
        meta_keep_count,
    )

    if meta_policy_on and events_meta.empty:
        logger.warning("Meta policy filtered all events; continuing with fallback diagnostics.")

    if meta_policy_on:
        kept_pct = (
            (len(events_meta) / len(events_state_filtered)) * 100 if len(events_state_filtered) else 0.0
        )
        logger.info(
            "INFO | META | before_rows=%s after_rows=%s kept_pct=%.2f",
            len(events_state_filtered),
            len(events_meta),
            kept_pct,
        )

        def _mix_table(series: pd.Series, title: str) -> pd.DataFrame:
            before_counts = series.value_counts(dropna=False)
            after_counts = series.loc[events_meta.index].value_counts(dropna=False)
            table = pd.DataFrame(
                {
                    "id": before_counts.index,
                    "before_count": before_counts.values,
                    "before_pct": (before_counts / before_counts.sum()).mul(100).round(2).values,
                }
            )
            table["after_count"] = table["id"].map(after_counts).fillna(0).astype(int)
            table["after_pct"] = (table["after_count"] / max(after_counts.sum(), 1)).mul(100).round(2)
            logger.info("INFO | META | %s before/after:\n%s", title, table.to_string(index=False))
            return table

        _mix_table(events_state_filtered["family_id"], "family mix")
        _mix_table(events_state_filtered["state_label"], "state mix")
        _mix_table(events_state_filtered["margin_bin"].fillna("mNA"), "margin bins")

    events_for_training = events_meta if meta_policy_on else events_state_filtered
    events = events_for_training
    logger.debug(
        "Supply funnel | score_total=%s after_merge=%s after_ctx_dropna=%s events_detected=%s "
        "events_labeled=%s events_post_state_filter=%s events_post_meta=%s",
        score_total,
        score_ctx_merged,
        score_ctx_dropna,
        len(detected_events),
        labeled_total,
        len(events_state_filtered),
        len(events_for_training),
    )
    required_columns = {"state_label", "allow_id", args.context_margin_col, "r_outcome", "label"}
    missing_columns = sorted(required_columns - set(events_for_training.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in events_for_training: {missing_columns}")
    if not isinstance(events_for_training.index, pd.DatetimeIndex):
        raise ValueError("events_for_training index must be DatetimeIndex")
    events_diag = pd.DataFrame(
        {
            args.context_state_col: events_for_training["state_label"],
            "allow_id": events_for_training["allow_id"],
            "margin": events_for_training[args.context_margin_col],
            "r": events_for_training["r_outcome"],
            "win": events_for_training["label"].astype(int),
            "family_id": events_for_training["family_id"],
            "vwap_zone": events_for_training["vwap_zone"],
            "value_state": events_for_training["value_state"],
            "expansion_state": events_for_training["expansion_state"],
            "pf_session_bucket": _session_bucket_series(
                events_for_training.index,
                args.symbol,
                df_score_ctx,
            ),
        },
        index=events_for_training.index,
    )
    events_diag["margin_bin"] = _make_margin_bin_rank(events_diag["margin"])
    events_total_post_meta = len(events_for_training)
    fallback_reasons = []
    if _apply_fallback_guardrails(events_total_post_meta, args.fallback_min_samples):
        fallback_reasons.append(f"events<{args.fallback_min_samples}")
    if events_for_training["label"].nunique() < 2:
        fallback_reasons.append("single_class_labels")
    is_fallback = bool(fallback_reasons)
    if is_fallback:
        logger.warning(
            "Fallback diagnostics: events_total_post_meta=%s reasons=%s",
            events_total_post_meta,
            ",".join(fallback_reasons),
        )
    diagnostic_report = _build_training_diagnostic_report(
        events_diag,
        df_score_ctx,
        thresholds=args,
        state_col=args.context_state_col,
        phase_e=args.phase_e,
    )
    supply_funnel = pd.DataFrame(
        [
            {
                "score_total": score_total,
                "after_merge": score_ctx_merged,
                "after_dropna_ctx": score_ctx_dropna,
                "events_detected": len(detected_events),
                "events_labeled": labeled_total,
                "events_post_state_filter": len(events_state_filtered),
                "events_post_meta": len(events_for_training),
            }
        ]
    )
    _persist_diagnostic_tables(
        diagnostic_report=diagnostic_report,
        supply_funnel=supply_funnel,
        model_dir=args.model_dir,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        mode_suffix=mode_suffix,
        run_id=run_id,
        symbol=args.symbol,
        allow_tf=args.allow_tf,
        score_tf=args.score_tf,
        context_tf=args.context_tf,
        config_path=config_path,
        mode=mode_effective,
        config_hash=config_hash,
        prompt_version=prompt_version,
        logger=logger,
    )

    if is_fallback:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        detected_path = _with_suffix(
            args.model_dir / f"events_detected_{output_prefix}.csv",
            output_suffix,
        )
        labeled_path = _with_suffix(
            args.model_dir / f"events_labeled_{output_prefix}.csv",
            output_suffix,
        )
        detected_output = _attach_output_metadata(
            _reset_index_for_export(detected_events),
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        labeled_output = _attach_output_metadata(
            _reset_index_for_export(labeled_events),
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        detected_output.to_csv(detected_path, index=False)
        labeled_output.to_csv(labeled_path, index=False)
        logger.info("fallback_events_detected_out=%s", detected_path)
        logger.info("fallback_events_labeled_out=%s", labeled_path)
        if not events_for_training.empty:
            r_values = events_for_training["r_outcome"]
            summary = {
                "base_rate": float(events_for_training["label"].mean()),
                "winrate": float(events_for_training["label"].mean()),
                "r_mean_all": float(r_values.mean()),
                "p10": float(r_values.quantile(0.1)),
                "p90": float(r_values.quantile(0.9)),
            }
            logger.debug("Fallback metrics: %s", summary)
        summary_payload = _base_summary_payload(
            "TELEMETRY_ONLY" if args.phase_e else "FALLBACK_DIAGNOSTIC"
        )
        summary_path = _with_suffix(
            args.model_dir / f"summary_{output_prefix}_event_scorer.json",
            output_suffix,
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)
        logger.info("summary_out=%s", summary_path)
        supply_funnel_payload = {
            "score_total": score_total,
            "after_merge": score_ctx_merged,
            "after_ctx_dropna": score_ctx_dropna,
            "events_detected": len(detected_events),
            "events_labeled": labeled_total,
            "events_post_state_filter": len(events_state_filtered),
            "events_post_meta": len(events_for_training),
        }
        _emit_telemetry(
            supply_funnel_payload=supply_funnel_payload,
            baseline_summary=_baseline_state_summary(events_state_filtered),
            scorer_summary=_scorer_vs_baseline_summary(pd.DataFrame(), phase_e=args.phase_e),
            regime_df=pd.DataFrame(),
            family_summary=pd.DataFrame(),
        )
        if research_mode and research_variants:
            diagnostics_cfg = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
            report = evaluate_research_variants(
                variants=research_variants,
                scores=pd.Series([], dtype=float),
                labels=pd.Series([], dtype=float),
                r_outcome=pd.Series([], dtype=float),
                families=pd.Series([], dtype="object"),
                timestamps=pd.DatetimeIndex([]),
                train_count=0,
                calib_count=0,
                allow_rate=None,
                diagnostics_cfg=diagnostics_cfg,
                baseline_thresholds=baseline_thresholds,
            )
            family_report = build_family_variant_report(
                variants=research_variants,
                scores=pd.Series([], dtype=float),
                labels=pd.Series([], dtype=float),
                r_outcome=pd.Series([], dtype=float),
                families=pd.Series([], dtype="object"),
            )
            return report, family_report
        return None

    event_features_all = feature_builder.add_family_features(
        event_features_all,
        events_state_filtered["family_id"],
    )

    def _split_dataset(events_df: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        labels_series = events_df["label"].astype(int)
        split_idx = int(len(events_df) * args.train_ratio)
        return {
            "X_train": feature_df.loc[events_df.index].iloc[:split_idx],
            "y_train": labels_series.iloc[:split_idx],
            "X_calib": feature_df.loc[events_df.index].iloc[split_idx:],
            "y_calib": labels_series.iloc[split_idx:],
            "r_train": events_df["r_outcome"].iloc[:split_idx],
            "r_calib": events_df["r_outcome"].iloc[split_idx:],
            "fam_train": events_df["family_id"].iloc[:split_idx],
            "fam_calib": events_df["family_id"].iloc[split_idx:],
            "regime_calib": events_df["regime_id"].iloc[split_idx:],
        }

    dataset_main = _split_dataset(events_for_training, event_features_all)
    dataset_no_meta = _split_dataset(events_state_filtered, event_features_all)

    X_train = dataset_main["X_train"]
    y_train = dataset_main["y_train"]
    X_calib = dataset_main["X_calib"]
    y_calib = dataset_main["y_calib"]
    r_train = dataset_main["r_train"]
    r_calib = dataset_main["r_calib"]
    fam_train = dataset_main["fam_train"]
    fam_calib = dataset_main["fam_calib"]
    regime_calib = dataset_main["regime_calib"]

    X_calib_no_meta = dataset_no_meta["X_calib"]
    y_calib_no_meta = dataset_no_meta["y_calib"]
    r_calib_no_meta = dataset_no_meta["r_calib"]
    fam_calib_no_meta = dataset_no_meta["fam_calib"]
    regime_calib_no_meta = dataset_no_meta["regime_calib"]

    transition_events_present = bool((events_state_filtered["state_label"] == "TRANSITION").any())
    transition_samples_train = int((events_for_training.loc[X_train.index, "state_label"] == "TRANSITION").sum())
    transition_samples_calib = int((events_for_training.loc[X_calib.index, "state_label"] == "TRANSITION").sum())

    family_summary = pd.DataFrame(
        {
            "family_id": sorted(set(events_for_training["family_id"])),
        }
    )
    family_summary["samples_train"] = family_summary["family_id"].map(fam_train.value_counts()).fillna(0).astype(int)
    family_summary["samples_calib"] = family_summary["family_id"].map(fam_calib.value_counts()).fillna(0).astype(int)
    family_summary["base_rate_train"] = family_summary["family_id"].map(y_train.groupby(fam_train).mean())
    family_summary["base_rate_calib"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).mean())
    logger.debug("Family counts:\n%s", family_summary.to_string(index=False))

    if y_train.nunique() < 2:
        logger.error("Global training labels have a single class; cannot train scorer.")
        args.model_dir.mkdir(parents=True, exist_ok=True)
        detected_path = _with_suffix(
            args.model_dir / f"events_detected_{output_prefix}.csv",
            output_suffix,
        )
        labeled_path = _with_suffix(
            args.model_dir / f"events_labeled_{output_prefix}.csv",
            output_suffix,
        )
        detected_output = _attach_output_metadata(
            _reset_index_for_export(detected_events),
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        labeled_output = _attach_output_metadata(
            _reset_index_for_export(labeled_events),
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        detected_output.to_csv(detected_path, index=False)
        labeled_output.to_csv(labeled_path, index=False)
        logger.info("fallback_events_detected_out=%s", detected_path)
        logger.info("fallback_events_labeled_out=%s", labeled_path)
        if not events_for_training.empty:
            r_values = events_for_training["r_outcome"]
            summary = {
                "base_rate": float(events_for_training["label"].mean()),
                "winrate": float(events_for_training["label"].mean()),
                "r_mean_all": float(r_values.mean()),
                "p10": float(r_values.quantile(0.1)),
                "p90": float(r_values.quantile(0.9)),
            }
            logger.debug("Fallback metrics: %s", summary)
        summary_payload = _base_summary_payload(
            "TELEMETRY_ONLY" if args.phase_e else "SINGLE_CLASS_FALLBACK"
        )
        summary_path = _with_suffix(
            args.model_dir / f"summary_{output_prefix}_event_scorer.json",
            output_suffix,
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)
        logger.info("summary_out=%s", summary_path)
        supply_funnel_payload = {
            "score_total": score_total,
            "after_merge": score_ctx_merged,
            "after_ctx_dropna": score_ctx_dropna,
            "events_detected": len(detected_events),
            "events_labeled": labeled_total,
            "events_post_state_filter": len(events_state_filtered),
            "events_post_meta": len(events_for_training),
        }
        _emit_telemetry(
            supply_funnel_payload=supply_funnel_payload,
            baseline_summary=_baseline_state_summary(events_state_filtered),
            scorer_summary=_scorer_vs_baseline_summary(pd.DataFrame(), phase_e=args.phase_e),
            regime_df=pd.DataFrame(),
            family_summary=pd.DataFrame(),
        )
        return

    warning_summary_rows: list[dict[str, int | str | float]] = []

    def _fit_with_warning_capture(
        scope: str,
        scorer_model: EventScorer,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        calib_x: pd.DataFrame | None,
        calib_y: pd.Series | None,
    ) -> None:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            scorer_model.fit(train_x, train_y, calib_x, calib_y)
        combined = stdout_buffer.getvalue() + stderr_buffer.getvalue()
        warning_hits = combined.count("No further splits with positive gain")
        warning_summary_rows.append(
            {
                "scope": scope,
                "n_samples": len(train_y),
                "n_features": train_x.shape[1],
                "base_rate": float(train_y.mean()) if len(train_y) else float("nan"),
                "split_warning_hits": warning_hits,
            }
        )

    scorer = EventScorerBundle(EventScorerConfig())
    global_scorer = EventScorer(scorer.config)
    _fit_with_warning_capture("global", global_scorer, X_train, y_train, X_calib, y_calib)
    scorer.scorers[scorer.global_key] = global_scorer

    family_summary["train_unique"] = family_summary["family_id"].map(y_train.groupby(fam_train).nunique())
    family_summary["calib_unique"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).nunique())
    family_summary["status"] = "TRAINED"

    for family_id in family_summary["family_id"]:
        train_mask = fam_train == family_id
        calib_mask = fam_calib == family_id
        train_count = int(train_mask.sum())
        calib_count = int(calib_mask.sum())
        train_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "train_unique"].fillna(0).iloc[0])
        calib_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "calib_unique"].fillna(0).iloc[0])

        if train_count < min_samples_train:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_LOW_SAMPLES"
            logger.warning("Skip %s: low samples (%s)", family_id, train_count)
            continue
        if train_unique < 2 or calib_unique < 2:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_SINGLE_CLASS"
            logger.warning(
                "Skip %s: missing class (train_unique=%s calib_unique=%s)",
                family_id,
                train_unique,
                calib_unique,
            )
            continue

        scorer_family = EventScorer(scorer.config)
        calib_X = X_calib.loc[calib_mask] if calib_count else None
        calib_y = y_calib.loc[calib_mask] if calib_count else None
        _fit_with_warning_capture(
            str(family_id),
            scorer_family,
            X_train.loc[train_mask],
            y_train.loc[train_mask],
            calib_X,
            calib_y,
        )
        scorer.scorers[str(family_id)] = scorer_family

    logger.debug("Family training status:\n%s", family_summary.to_string(index=False))

    def _topk_indices(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> pd.Index:
        if scores.empty:
            return pd.Index([])
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return pd.Index([])
        if scores.nunique(dropna=False) <= 1:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(labels_.index.to_numpy(), size=k_eff, replace=False)
            return pd.Index(chosen)
        return scores.nlargest(k_eff).index

    def _bottomk_indices(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> pd.Index:
        if scores.empty:
            return pd.Index([])
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return pd.Index([])
        if scores.nunique(dropna=False) <= 1:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(labels_.index.to_numpy(), size=k_eff, replace=False)
            return pd.Index(chosen)
        return scores.nsmallest(k_eff).index

    def precision_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, labels_, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(labels_.loc[top_idx].mean())

    def lift_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        base_rate = float(labels_.mean())
        if base_rate == 0 or scores.empty:
            return float("nan")
        return precision_at_k(scores, labels_, k, seed=seed) / base_rate

    def mean_r_topk(scores: pd.Series, r_outcome: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, r_outcome, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(r_outcome.loc[top_idx].mean())

    def median_r_topk(scores: pd.Series, r_outcome: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, r_outcome, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(r_outcome.loc[top_idx].median())

    def summarize_metrics(
        scope: str,
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        seed: int = 7,
    ) -> dict[str, float]:
        base_rate = float(labels_.mean()) if not labels_.empty else float("nan")
        r_mean_all = float(r_outcome.mean()) if not r_outcome.empty else float("nan")
        metrics = {
            "scope": scope,
            "samples": len(labels_),
            "base_rate": base_rate,
            "auc": float("nan"),
            "lift@10": lift_at_k(scores, labels_, 10, seed=seed),
            "lift@20": lift_at_k(scores, labels_, 20, seed=seed),
            "lift@50": lift_at_k(scores, labels_, 50, seed=seed),
            "r_mean@10": mean_r_topk(scores, r_outcome, 10, seed=seed),
            "r_mean@20": mean_r_topk(scores, r_outcome, 20, seed=seed),
            "r_mean@50": mean_r_topk(scores, r_outcome, 50, seed=seed),
            "r_median@10": median_r_topk(scores, r_outcome, 10, seed=seed),
            "r_median@20": median_r_topk(scores, r_outcome, 20, seed=seed),
            "r_median@50": median_r_topk(scores, r_outcome, 50, seed=seed),
            "r_mean_all": r_mean_all,
        }
        if len(labels_.unique()) > 1:
            metrics["auc"] = roc_auc_score(labels_, scores)
        metrics["delta_r_mean@10"] = metrics["r_mean@10"] - r_mean_all
        metrics["delta_r_mean@20"] = metrics["r_mean@20"] - r_mean_all
        metrics["delta_r_mean@50"] = metrics["r_mean@50"] - r_mean_all
        metrics["delta_r_mean@20_neg"] = float(metrics["delta_r_mean@20"] < 0)
        return metrics

    def _global_metrics_table(
        label: str,
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        seed: int = 7,
    ) -> dict[str, float | str]:
        metrics = summarize_metrics(label, scores, labels_, r_outcome, seed=seed)
        spearman = scores.corr(r_outcome, method="spearman") if len(scores) else float("nan")
        return {
            "model": label,
            "auc": metrics["auc"],
            "lift@10": metrics["lift@10"],
            "lift@20": metrics["lift@20"],
            "lift@50": metrics["lift@50"],
            "r_mean@10": metrics["r_mean@10"],
            "r_mean@20": metrics["r_mean@20"],
            "r_mean@50": metrics["r_mean@50"],
            "delta_r_mean@20": metrics["delta_r_mean@20"],
            "spearman": spearman,
        }

    def _metrics_block(
        block_name: str,
        X_block: pd.DataFrame,
        y_block: pd.Series,
        r_block: pd.Series,
        fam_block: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series | None]:
        if y_block.empty:
            logger.warning("WARNING | DIAG | Empty calib set for %s metrics", block_name)
            return pd.DataFrame(), None, None, None
        scores_block = scorer.predict_proba(X_block, fam_block)
        rng = np.random.default_rng(seed)
        baseline_scores = pd.Series(rng.random(len(y_block)), index=y_block.index)
        table = pd.DataFrame(
            [
                _global_metrics_table("SCORER", scores_block, y_block, r_block, seed=seed),
                _global_metrics_table("BASELINE", baseline_scores, y_block, r_block, seed=seed),
            ]
        )
        logger.debug("INFO | GLOBAL | METRICS (%s)\n%s", block_name, table.to_string(index=False))
        return table, scores_block, baseline_scores, y_block

    if not is_fallback:
        table_no_meta, preds_no_meta, baseline_no_meta, _ = _metrics_block(
            "NO_META",
            X_calib_no_meta,
            y_calib_no_meta,
            r_calib_no_meta,
            fam_calib_no_meta,
        )
        if meta_policy_on:
            table_meta, preds_meta, baseline_meta, _ = _metrics_block(
                "META",
                X_calib,
                y_calib,
                r_calib,
                fam_calib,
            )
        else:
            table_meta = table_no_meta.copy()
            preds_meta = preds_no_meta
            baseline_meta = baseline_no_meta
    else:
        table_no_meta = pd.DataFrame()
        table_meta = pd.DataFrame()
        preds_no_meta = None
        preds_meta = None
        baseline_no_meta = None
        baseline_meta = None

    def _log_global_metrics(title: str, table: pd.DataFrame) -> None:
        if table.empty:
            logger.debug("%s | AUC=nan lift@10=nan lift@20=nan r_mean@20=nan spearman=nan", title)
            return
        scorer_row = table.loc[table["model"] == "SCORER"]
        if scorer_row.empty:
            scorer_row = table.iloc[[0]]
        row = scorer_row.iloc[0]
        logger.debug(
            "%s | AUC=%s lift@10=%s lift@20=%s r_mean@20=%s spearman=%s",
            title,
            _format_metric(row["auc"]),
            _format_metric(row["lift@10"]),
            _format_metric(row["lift@20"]),
            _format_metric(row["r_mean@20"]),
            _format_metric(row["spearman"]),
        )

    if not is_fallback:
        _log_global_metrics("[GLOBAL METRICS | NO_META | POST FILTER]", table_no_meta)
        _log_global_metrics("[GLOBAL METRICS | META | POST FILTER]", table_meta)
    logger.debug(
        "Transition sanity | present=%s train=%s calib=%s",
        transition_events_present,
        transition_samples_train,
        transition_samples_calib,
    )

    metrics_df = table_meta.copy()
    report_header = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "context_cutoff": allow_cutoff,
        "score_cutoff": score_cutoff,
        "k_bars": k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "feature_count": event_features_all.shape[1],
        "min_samples_train": min_samples_train,
        "seed": seed,
        "meta_policy": args.meta_policy,
        "meta_margin_min": args.meta_margin_min,
        "meta_margin_max": args.meta_margin_max,
        "include_transition": args.include_transition,
    }
    header_table = pd.DataFrame([report_header])
    logger.info("=" * 96)
    logger.info("EVENT SCORER TRAINING REPORT")
    logger.info("%s", header_table.to_string(index=False))

    coverage_global = pd.DataFrame(
        [
            {
                "detected_count": len(detected_events),
                "labeled_count": len(events),
                "dropna_count": dropna_count,
                "train_count": len(y_train),
                "calib_count": len(y_calib),
            }
        ]
    )
    family_detected = detected_events["family_id"].value_counts()
    family_labeled = events["family_id"].value_counts()
    family_share = (
        (family_labeled / family_labeled.sum()).mul(100).round(2)
        if not family_labeled.empty
        else pd.Series(dtype=float)
    )
    coverage_by_family = pd.DataFrame(
        {
            "family_id": family_summary["family_id"],
            "detected_count": family_summary["family_id"].map(family_detected).fillna(0).astype(int),
            "labeled_count": family_summary["family_id"].map(family_labeled).fillna(0).astype(int),
            "dropna_count": family_summary["family_id"]
            .map((dropna_mask).groupby(labeled_events["family_id"]).sum())
            .fillna(0)
            .astype(int),
            "train_count": family_summary["samples_train"],
            "calib_count": family_summary["samples_calib"],
            "share_pct": family_summary["family_id"].map(family_share).fillna(0.0),
        }
    )
    logger.debug("-" * 96)
    logger.debug("A) Coverage (supply)")
    logger.debug("Global coverage:\n%s", coverage_global.to_string(index=False))
    logger.debug("Coverage by family:\n%s", coverage_by_family.to_string(index=False))

    base_rate_global = pd.DataFrame(
        [
            {
                "scope": "global",
                "base_rate_train": float(y_train.mean()) if not y_train.empty else float("nan"),
                "base_rate_calib": float(y_calib.mean()) if not y_calib.empty else float("nan"),
            }
        ]
    )
    family_base_rate = family_summary[
        ["family_id", "base_rate_train", "base_rate_calib"]
    ].copy()
    logger.debug("-" * 96)
    logger.debug("B) Label quality & hardness")
    logger.debug("Base rates:\n%s", pd.concat([base_rate_global, family_base_rate]).to_string(index=False))

    r_stats = (
        r_calib.groupby(fam_calib)
        .agg(
            r_mean="mean",
            r_std="std",
            r_p50=lambda s: s.quantile(0.5),
            r_p75=lambda s: s.quantile(0.75),
            r_p90=lambda s: s.quantile(0.9),
            r_p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    if not r_stats.empty:
        r_stats = r_stats.rename(columns={r_stats.columns[0]: "scope"})
    r_stats_global = pd.DataFrame(
        [
            {
                "scope": "global",
                "r_mean": float(r_calib.mean()) if not r_calib.empty else float("nan"),
                "r_std": float(r_calib.std()) if not r_calib.empty else float("nan"),
                "r_p50": float(r_calib.quantile(0.5)) if not r_calib.empty else float("nan"),
                "r_p75": float(r_calib.quantile(0.75)) if not r_calib.empty else float("nan"),
                "r_p90": float(r_calib.quantile(0.9)) if not r_calib.empty else float("nan"),
                "r_p95": float(r_calib.quantile(0.95)) if not r_calib.empty else float("nan"),
            }
        ]
    )
    logger.debug("r_outcome distribution (calib):\n%s", pd.concat([r_stats_global, r_stats]).to_string(index=False))

    def _regime_breakdown(
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        regimes: pd.Series,
        seed: int = 7,
    ) -> pd.DataFrame:
        if labels_.empty:
            return pd.DataFrame()
        rows: list[dict[str, float | str | int]] = []
        for regime_id, regime_labels in labels_.groupby(regimes, observed=True):
            if isinstance(regime_id, float) and np.isnan(regime_id):
                continue
            scope_scores = scores.loc[regime_labels.index]
            scope_r = r_outcome.loc[regime_labels.index]
            metrics = summarize_metrics(str(regime_id), scope_scores, regime_labels, scope_r, seed=seed)
            top_idx = _topk_indices(scope_scores, regime_labels, 10, seed=seed)
            bottom_idx = _bottomk_indices(scope_scores, regime_labels, 10, seed=seed)
            top_mean = float(scope_r.loc[top_idx].mean()) if len(top_idx) else float("nan")
            bottom_mean = float(scope_r.loc[bottom_idx].mean()) if len(bottom_idx) else float("nan")
            rank_inverted = bool(bottom_mean > top_mean)
            flag = "MIXED"
            if metrics["delta_r_mean@20"] >= 0.10 and metrics["lift@20"] > 1.05:
                flag = "WIN"
            elif metrics["delta_r_mean@20"] <= 0 or metrics["lift@20"] < 1.0:
                flag = "LOSE"
            flag_parts = [flag]
            if rank_inverted:
                flag_parts.append("RANK_INVERTED")
            if metrics["samples"] < 100:
                flag_parts.append("LOW_SAMPLES")
            rows.append(
                {
                    "regime_id": metrics["scope"],
                    "samples_calib": metrics["samples"],
                    "base_rate_calib": metrics["base_rate"],
                    "auc": metrics["auc"],
                    "lift@20": metrics["lift@20"],
                    "r_mean_all": metrics["r_mean_all"],
                    "r_mean@20": metrics["r_mean@20"],
                    "delta_r_mean@20": metrics["delta_r_mean@20"],
                    "spearman": scope_scores.corr(scope_r, method="spearman") if len(scope_scores) else float("nan"),
                    "flag": "|".join(flag_parts),
                }
            )
        return pd.DataFrame(rows)

    regime_df = pd.DataFrame()
    if not is_fallback and preds_meta is not None and not y_calib.empty:
        regime_df = _regime_breakdown(preds_meta, y_calib, r_calib, regime_calib, seed=seed)
        if not regime_df.empty:
            regime_top = regime_df.sort_values("samples_calib", ascending=False).head(8)
            logger.debug("INFO | REGIME | TOP regimes (calib)\n%s", regime_top.to_string(index=False))

    best_regime = "NA"
    worst_regime = "NA"
    recommendation = "NO_EDGE_GENERAL; CONSIDER_SYMBOL_SPECIFIC"
    if not regime_df.empty:
        best_idx = regime_df["delta_r_mean@20"].idxmax()
        worst_idx = regime_df["delta_r_mean@20"].idxmin()
        best_regime = regime_df.loc[best_idx, "regime_id"]
        worst_regime = regime_df.loc[worst_idx, "regime_id"]
        win_mask = regime_df["flag"].str.contains("WIN") & (regime_df["samples_calib"] >= 300)
        if win_mask.any():
            recommendation = "EDGE_FOUND_META"
    if args.phase_e:
        recommendation = "TELEMETRY_ONLY"
    logger.info(
        "INFO | SUMMARY | BEST_REGIME=%s WORST_REGIME=%s",
        best_regime,
        worst_regime,
    )
    logger.info("INFO | SUMMARY | RECOMMENDATION=%s", recommendation)

    supply_funnel_payload = {
        "score_total": score_total,
        "after_merge": score_ctx_merged,
        "after_ctx_dropna": score_ctx_dropna,
        "events_detected": len(detected_events),
        "events_labeled": labeled_total,
        "events_post_state_filter": len(events_state_filtered),
        "events_post_meta": len(events_for_training),
    }
    _emit_telemetry(
        supply_funnel_payload=supply_funnel_payload,
        baseline_summary=_baseline_state_summary(events_state_filtered),
        scorer_summary=_scorer_vs_baseline_summary(table_meta, phase_e=args.phase_e),
        regime_df=regime_df,
        family_summary=family_summary,
        scores=preds_meta,
        r_outcome=r_calib,
        regime_series=regime_calib,
    )

    metrics_summary = {}
    if not metrics_df.empty:
        scorer_row = metrics_df[metrics_df["model"] == "SCORER"]
        if not scorer_row.empty:
            metrics_summary = scorer_row.iloc[0].to_dict()

    metadata = {
        "run_id": run_id,
        "symbol": args.symbol,
        "score_tf": args.score_tf,
        "allow_tf": args.allow_tf,
        "context_tf": args.context_tf,
        "train_ratio": args.train_ratio,
        "k_bars": k_bars,
        "horizon_min": args.horizon_min,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "include_transition": args.include_transition,
        "meta_policy": args.meta_policy,
        "meta_margin_min": args.meta_margin_min,
        "meta_margin_max": args.meta_margin_max,
        "meta_policy_effective": meta_policy_effective,
        "config_hash": config_hash,
        "prompt_version": prompt_version,
        "decision_thresholds": {
            "n_min": args.decision_n_min,
            "winrate_min": args.decision_winrate_min,
            "r_mean_min": args.decision_r_mean_min,
            "p10_min": args.decision_p10_min,
        },
        "config_path": str(config_path) if config_path is not None else None,
        "config_mode": mode_effective,
        "feature_count": event_features_all.shape[1],
        "train_date": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": metrics_summary,
        "research_enabled": research_enabled,
        "phase_e": args.phase_e,
    }
    if research_enabled:
        metadata["research_features"] = research_cfg.get("features", {})
        metadata["research_k_bars_grid"] = research_cfg.get("k_bars_grid")
    scorer_out = _with_suffix(scorer_out_base, output_suffix)
    _save_model_if_ready(is_fallback=is_fallback, scorer=scorer, path=scorer_out, metadata=metadata)

    summary_table = pd.DataFrame(
        [
            {
                "events_total": len(events),
                "labeled": len(events),
                "feature_count": event_features_all.shape[1],
            }
        ]
    )
    logger.info("Summary:\n%s", summary_table.to_string(index=False))
    logger.info("label_distribution=%s", events["label"].value_counts(normalize=True).to_dict())
    logger.info("model_out=%s", scorer_out)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_path = _with_suffix(
            args.model_dir
            / f"metrics_{output_prefix}_event_scorer{mode_suffix}.csv",
            output_suffix,
        )
        metrics_output = _attach_output_metadata(
            metrics_df,
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        metrics_output.to_csv(metrics_path, index=False)
        logger.info("metrics_out=%s", metrics_path)
    family_path = _with_suffix(
        args.model_dir
        / f"family_summary_{output_prefix}_event_scorer{mode_suffix}.csv",
        output_suffix,
    )
    family_output = _attach_output_metadata(
        family_summary,
        run_id=run_id,
        symbol=args.symbol,
        allow_tf=args.allow_tf,
        score_tf=args.score_tf,
        context_tf=args.context_tf,
        config_path=config_path,
        mode=mode_effective,
        config_hash=config_hash,
        prompt_version=prompt_version,
    )
    family_output.to_csv(family_path, index=False)
    logger.info("family_summary_out=%s", family_path)

    if not y_calib.empty and preds_meta is not None:
        sample_cols = ["family_id", "side", "label", "r_outcome"]
        sample_df = events.loc[y_calib.index, sample_cols].copy()
        sample_df["score"] = preds_meta
        sample_df[args.context_margin_col] = df_score_ctx[args.context_margin_col].reindex(sample_df.index)
        sample_df = sample_df.sort_values("score", ascending=False).head(10)
        sample_df = sample_df.reset_index().rename(columns={sample_df.index.name or "index": "time"})
        sample_path = _with_suffix(
            args.model_dir
            / f"calib_top_scored_{output_prefix}_event_scorer{mode_suffix}.csv",
            output_suffix,
        )
        sample_output = _attach_output_metadata(
            sample_df,
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        sample_output.to_csv(sample_path, index=False)
        logger.info("calib_top_scored_out=%s", sample_path)

    if global_scorer._model is not None and hasattr(global_scorer._model, "feature_importances_"):
        importances = pd.Series(global_scorer._model.feature_importances_, index=event_features_all.columns)
        top_features = importances.sort_values(ascending=False).head(20)
        logger.info("Model signature (features=%s):", event_features_all.shape[1])
        logger.info("%s", textwrap.fill(", ".join(event_features_all.columns), width=120))
        logger.info("Top-20 feature importances:\n%s", top_features.to_string())

    warning_summary = pd.DataFrame(warning_summary_rows)
    if not warning_summary.empty and warning_summary["split_warning_hits"].sum() > 0:
        logger.warning(
            "LightGBM split warnings detected. Consider raising min_samples_train or revisiting features."
        )
        logger.info("Split warning summary:\n%s", warning_summary.to_string(index=False))
    coverage_global = diagnostic_report.get("coverage_global", pd.DataFrame())
    score_shape = pd.DataFrame()
    if preds_meta is not None and not y_calib.empty:
        score_shape = _score_shape_diagnostics(preds_meta, fam_calib, y_calib.index, [10, 20, 50])
        if args.phase_e:
            score_shape["verdict"] = "TELEMETRY_ONLY"
        score_shape_path = _with_suffix(
            args.model_dir
            / f"score_shape_{output_prefix}_event_scorer{mode_suffix}.csv",
            output_suffix,
        )
        score_shape_output = _attach_output_metadata(
            score_shape,
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=mode_effective,
            config_hash=config_hash,
            prompt_version=prompt_version,
        )
        score_shape_output.to_csv(score_shape_path, index=False)
        logger.info("score_shape_out=%s", score_shape_path)

    def _scorer_metric(table: pd.DataFrame, metric: str) -> float | None:
        if table.empty:
            return None
        if "model" in table.columns:
            scorer_row = table.loc[table["model"] == "SCORER"]
            if not scorer_row.empty:
                return scorer_row.iloc[0].get(metric)
        return table.iloc[0].get(metric)

    def _coverage_metric(table: pd.DataFrame, metric: str) -> float | int | None:
        if table.empty or metric not in table.columns:
            return None
        return table.iloc[0].get(metric)

    def _split_warning_global(table: pd.DataFrame) -> int | None:
        if table.empty or "scope" not in table.columns or "split_warning_hits" not in table.columns:
            return None
        global_row = table.loc[table["scope"] == "global"]
        if global_row.empty:
            return None
        return int(global_row.iloc[0]["split_warning_hits"])

    summary_payload = {
        "run_id": run_id,
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "score_tf": args.score_tf,
        "allow_tf": args.allow_tf,
        "context_tf": args.context_tf,
        "state_model_path": str(state_model_path),
        "state_model_metadata_timeframe": (
            state_model_metadata.get("timeframe") if isinstance(state_model_metadata, dict) else None
        ),
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash,
        "prompt_version": prompt_version,
        "allow_cutoff": allow_cutoff,
        "context_cutoff": allow_cutoff,
        "m5_cutoff": score_cutoff,
        "score_cutoff": score_cutoff,
        "events_total_post_meta": _coverage_metric(coverage_global, "events_total_post_meta"),
        "events_per_day": _coverage_metric(coverage_global, "events_per_day"),
        "unique_days": _coverage_metric(coverage_global, "unique_days"),
        "auc_no_meta": _scorer_metric(table_no_meta, "auc"),
        "auc_meta": _scorer_metric(table_meta, "auc"),
        "lift20_meta": _scorer_metric(table_meta, "lift@20"),
        "r_mean20_meta": _scorer_metric(table_meta, "r_mean@20"),
        "spearman_meta": _scorer_metric(table_meta, "spearman"),
        "families_trained": int((family_summary["status"] == "TRAINED").sum())
        if "status" in family_summary.columns
        else None,
        "best_regime_id": best_regime,
        "worst_regime_id": worst_regime,
        "recommendation": recommendation,
        "split_warning_hits_global": _split_warning_global(warning_summary),
        "allow_active_pct": allow_active_pct,
        "feature_count": event_features_all.shape[1],
        "min_samples_train": min_samples_train,
        "k_bars": k_bars,
        "horizon_min": args.horizon_min,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "meta_policy": {
            "enabled": args.meta_policy == "on",
            "meta_margin_min": args.meta_margin_min,
            "meta_margin_max": args.meta_margin_max,
        },
        "meta_policy_effective": meta_policy_effective,
        "phase_e": args.phase_e,
        "verdict": "TELEMETRY_ONLY" if args.phase_e else "SCORER_READY",
        "score_shape_verdict": "TELEMETRY_ONLY" if args.phase_e else "DIAGNOSTIC_ONLY",
        "score_shape_diagnostics": score_shape.to_dict(orient="records") if not score_shape.empty else [],
        "event_counts": event_counts,
        "context_columns": {
            "state_col": args.context_state_col,
            "margin_col": args.context_margin_col,
        },
        "context_nan_pct": context_nan_pct,
    }

    research_summary_payload = None
    if research_enabled and preds_meta is not None and not y_calib.empty:
        diagnostics_cfg = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
        train_scores = scorer.predict_proba(X_train, fam_train)
        train_metrics = summarize_metrics("TRAIN", train_scores, y_train, r_train, seed=seed)
        calib_metrics = summarize_metrics("CALIB", preds_meta, y_calib, r_calib, seed=seed)
        guardrails = _research_guardrails(
            score_shape,
            train_metrics=train_metrics,
            calib_metrics=calib_metrics,
            diagnostics_cfg=diagnostics_cfg,
            calib_samples=len(y_calib),
        )
        research_summary_payload = {
            "score_shape": score_shape.to_dict(orient="records"),
            "guardrails": guardrails.to_dict(orient="records"),
        }
        logger.debug("Research diagnostics | score_shape=%s", score_shape.to_dict(orient="records"))
        logger.debug("Research guardrails=%s", guardrails.to_dict(orient="records"))

    if research_summary_payload is not None:
        summary_payload["research"] = research_summary_payload

    summary_path = _with_suffix(
        args.model_dir / f"summary_{output_prefix}_event_scorer.json",
        output_suffix,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, default=str)
    logger.info("summary_out=%s", summary_path)
    logger.info("=" * 96)
    if research_mode:
        _print_research_summary_block(
            diagnostic_report.get("session_conditional_edge", pd.DataFrame()),
            state_col=args.context_state_col,
            phase_e=args.phase_e,
        )
    if research_mode and research_variants:
        if preds_meta is None or y_calib.empty or len(y_calib) == 0:
            variant_report = _empty_research_variant_report(research_variants, "NO_PREDICTIONS")
            family_report = pd.DataFrame(
                columns=[
                    "variant_id",
                    "family",
                    "n",
                    "winrate",
                    "r_mean",
                    "p10",
                    "lift10",
                    "lift20",
                    "qualified",
                    "fail_reason",
                ]
            )
            return variant_report, family_report
        diagnostics_cfg = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
        allow_rate = None
        if allow_cols:
            allow_rate = float((events_for_training[allow_cols].fillna(0).sum(axis=1) > 0).mean())
        report = evaluate_research_variants(
            variants=research_variants,
            scores=preds_meta,
            labels=y_calib,
            r_outcome=r_calib,
            families=fam_calib,
            timestamps=y_calib.index,
            train_count=len(y_train),
            calib_count=len(y_calib),
            allow_rate=allow_rate,
            diagnostics_cfg=diagnostics_cfg,
            baseline_thresholds=baseline_thresholds,
        )
        family_report = build_family_variant_report(
            variants=research_variants,
            scores=preds_meta,
            labels=y_calib,
            r_outcome=r_calib,
            families=fam_calib,
        )
        return report, family_report
    return None


def _effective_mode(requested_mode: str, research_cfg: dict) -> str:
    if requested_mode != "research":
        return requested_mode
    return "research" if bool(research_cfg.get("enabled", False)) else "production"


def _resolve_baseline_thresholds(
    config_payload: dict,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | None], str]:
    production_thresholds = _resolve_decision_thresholds(config_payload, "production")
    prod_thresholds = _resolve_decision_thresholds(config_payload, "prod")
    if production_thresholds:
        return production_thresholds, "production"
    if prod_thresholds:
        return prod_thresholds, "prod"
    baseline_thresholds = {
        "n_min": args.decision_n_min,
        "winrate_min": args.decision_winrate_min,
        "r_mean_min": args.decision_r_mean_min,
        "p10_min": args.decision_p10_min,
    }
    return baseline_thresholds, "cli_fallback"


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    if args.telemetry in {"screen", "triage"} and args.log_level.upper() == "INFO":
        logger.setLevel(logging.WARNING)
    if args.phase_e:
        logger.info("phase_e=ON (telemetry only; no decision / gating)")
    min_samples_train = 200
    seed = 7
    research_cfg: dict[str, object] = {
        "enabled": False,
        "features": {},
        "diagnostics": {},
        "k_bars_grid": None,
        "exploration": {},
    }
    config_hash = None
    prompt_version = None
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    config_path: Path | None = None
    if args.config == "on":
        config_path = PROJECT_ROOT / "configs" / "symbols" / f"{args.symbol}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
    config_payload = _default_symbol_config(args)
    if config_path is not None:
        overrides = load_config(config_path)
        merged = deep_merge(config_payload, overrides)
        config_payload = merged

        event_cfg = merged.get("event_scorer", {})
        if isinstance(event_cfg, dict):
            args.score_tf = event_cfg.get("score_tf", args.score_tf)
            args.allow_tf = event_cfg.get("allow_tf", args.allow_tf)
            args.context_tf = event_cfg.get("context_tf", args.context_tf)
            args.train_ratio = event_cfg.get("train_ratio", args.train_ratio)
            args.k_bars = event_cfg.get("k_bars", args.k_bars)
            args.reward_r = event_cfg.get("reward_r", args.reward_r)
            args.sl_mult = event_cfg.get("sl_mult", args.sl_mult)
            args.r_thr = event_cfg.get("r_thr", args.r_thr)
            args.tie_break = event_cfg.get("tie_break", args.tie_break)

            include_transition = event_cfg.get("include_transition")
            if include_transition is not None:
                args.include_transition = "on" if include_transition else "off"

            meta_cfg = event_cfg.get("meta_policy", {})
            if isinstance(meta_cfg, dict):
                enabled = meta_cfg.get("enabled")
                if enabled is not None:
                    args.meta_policy = "on" if enabled else "off"
                args.meta_margin_min = meta_cfg.get("meta_margin_min", args.meta_margin_min)
                args.meta_margin_max = meta_cfg.get("meta_margin_max", args.meta_margin_max)

            family_cfg = event_cfg.get("family_training", {})
            if isinstance(family_cfg, dict):
                min_samples_train = family_cfg.get("min_samples_train", min_samples_train)

            research_cfg = _resolve_research_config(merged, args.mode)

        logger.info("Loaded config overrides from %s (mode=%s)", config_path, args.mode)
    if config_payload:
        config_hash = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        prompt_version = config_payload.get("prompt_version")

    requested_mode = args.mode
    effective_mode = _effective_mode(requested_mode, research_cfg)
    if requested_mode == "research" and effective_mode != "research":
        logger.warning("Research mode requested but disabled in config; falling back to production behavior.")

    baseline_thresholds, baseline_thresholds_source = _resolve_baseline_thresholds(config_payload, args)

    thresholds = _resolve_decision_thresholds(config_payload, effective_mode)
    if thresholds:
        args.decision_n_min = thresholds.get("n_min", args.decision_n_min)
        args.decision_winrate_min = thresholds.get("winrate_min", args.decision_winrate_min)
        args.decision_r_mean_min = thresholds.get("r_mean_min", args.decision_r_mean_min)
        args.decision_p10_min = thresholds.get("p10_min", args.decision_p10_min)

    args.score_tf = _normalize_timeframe(args.score_tf)
    args.allow_tf = _normalize_timeframe(args.allow_tf)
    event_cfg = config_payload.get("event_scorer", {}) if isinstance(config_payload.get("event_scorer"), dict) else {}
    if event_cfg:
        if "context_tf" in event_cfg:
            args.context_tf = event_cfg.get("context_tf")
        args.k_bars = _resolve_k_bars_by_tf(event_cfg, args.score_tf, args.k_bars)
    if args.horizon_min is not None:
        score_minutes = _timeframe_to_minutes(args.score_tf)
        derived_k_bars = max(1, int(math.ceil(args.horizon_min / score_minutes)))
        logger.info(
            "horizon_min=%.2f score_tf=%s score_tf_minutes=%s derived_k_bars=%s",
            args.horizon_min,
            args.score_tf,
            score_minutes,
            derived_k_bars,
        )
        args.k_bars = derived_k_bars

    logger.info(
        "Run init | symbol=%s config_path=%s allow_tf=%s score_tf=%s mode=%s run_id=%s",
        args.symbol,
        config_path,
        args.allow_tf,
        args.score_tf,
        effective_mode,
        run_id,
    )
    required_allow_map = _required_allow_by_family()

    research_mode = effective_mode == "research"
    research_enabled = research_mode and bool(research_cfg.get("enabled", False))
    mode_suffix = _mode_suffix(effective_mode)
    k_bars_grid = _resolve_k_bars_grid_by_tf(research_cfg, args.score_tf) if research_enabled else None
    if research_enabled:
        research_cfg = {**research_cfg, "k_bars_grid": k_bars_grid}
    exploration_cfg = research_cfg.get("exploration", {}) if research_enabled else {}
    exploration_enabled = bool(exploration_cfg.get("enabled", False)) if isinstance(exploration_cfg, dict) else False
    exploration_kind = exploration_cfg.get("kind", "thresholds_only") if isinstance(exploration_cfg, dict) else "thresholds_only"
    exploration_kind = normalize_exploration_kind(exploration_kind)

    if research_enabled and exploration_enabled and exploration_kind == "kbars_only":
        k_values = k_bars_grid if isinstance(k_bars_grid, list) and k_bars_grid else [args.k_bars]
    else:
        k_values = [args.k_bars]
    multi_k = len(k_values) > 1

    model_path = args.state_model
    if model_path is None:
        model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_state_engine.pkl"

    scorer_out_base = args.model_out
    output_prefix = _output_prefix(args.symbol, args.score_tf, run_id)
    if scorer_out_base is None:
        scorer_out_base = args.model_dir / f"{output_prefix}_event_scorer.pkl"

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv_score = connector.obtener_ohlcv(args.symbol, args.score_tf, fecha_inicio, fecha_fin)
    ohlcv_score["symbol"] = args.symbol
    server_now = connector.server_now(args.symbol).tz_localize(None)

    score_cutoff = server_now.floor(_timeframe_floor_freq(args.score_tf))
    ohlcv_score = ohlcv_score[ohlcv_score.index < score_cutoff]
    score_dupes = int(ohlcv_score.index.duplicated().sum())
    logger.debug("Event scorer training started.")

    if not model_path.exists():
        raise FileNotFoundError(f"State model not found: {model_path}")

    state_model = StateEngineModel()
    state_model.load(model_path)
    context_tf = _resolve_context_tf(
        requested=args.context_tf,
        model_metadata=state_model.metadata,
        logger=logger,
    )
    args.context_tf = context_tf
    state_col = f"state_hat_{context_tf}"
    margin_col = f"margin_{context_tf}"
    args.context_state_col = state_col
    args.context_margin_col = margin_col
    logger.info("context_tf=%s context_state_col=%s context_margin_col=%s", context_tf, state_col, margin_col)

    ohlcv_ctx = connector.obtener_ohlcv(args.symbol, context_tf, fecha_inicio, fecha_fin)
    allow_cutoff = server_now.floor(_timeframe_floor_freq(context_tf))
    ohlcv_ctx = ohlcv_ctx[ohlcv_ctx.index < allow_cutoff]
    h1_dupes = int(ohlcv_ctx.index.duplicated().sum())
    logger.info(
        "Period: %s -> %s | context_cutoff=%s score_cutoff=%s",
        fecha_inicio,
        fecha_fin,
        allow_cutoff,
        score_cutoff,
    )
    logger.info("Rows: %s=%s %s=%s", context_tf, len(ohlcv_ctx), args.score_tf, len(ohlcv_score))

    feature_engineer = FeatureEngineer(FeatureConfig())
    gating = GatingPolicy()
    context_bundle = build_context_bundle(
        symbol=args.symbol,
        context_tf=context_tf,
        score_tf=args.score_tf,
        ohlcv_ctx=ohlcv_ctx,
        ohlcv_score=ohlcv_score,
        state_model=state_model,
        feature_engineer=feature_engineer,
        gating_policy=gating,
        symbol_cfg=config_payload,
        phase_e=args.phase_e,
        logger=logger,
    )
    df_score_ctx = context_bundle.df_score_ctx
    if "atr_14" not in df_score_ctx.columns:
        df_score_ctx["atr_14"] = _ensure_atr_14(df_score_ctx)
    ctx_cols = [col for col in df_score_ctx.columns if col.startswith("ctx_")]
    allow_cols = [col for col in df_score_ctx.columns if col.startswith("LOOK_FOR_")]
    context_columns = [state_col, margin_col, *ctx_cols, *allow_cols]
    logger.info(
        "context_tf=%s context_columns=%s",
        context_tf,
        context_columns,
    )
    logger.info("Rows after merge: %s_ctx=%s", args.score_tf, len(df_score_ctx))
    score_ctx_merged = len(df_score_ctx)
    ctx_nan_cols = [col for col in context_columns if col in df_score_ctx.columns]
    if "atr_short" in df_score_ctx.columns:
        ctx_nan_cols.append("atr_short")
    ctx_nan = df_score_ctx[ctx_nan_cols].isna().mean().mul(100).round(2)
    ctx_nan_table = pd.DataFrame({"column": ctx_nan.index, "nan_pct": ctx_nan.values})
    logger.info("Context NaN rates:\n%s", ctx_nan_table.to_string(index=False))
    context_nan_pct = dict(zip(ctx_nan_table["column"], ctx_nan_table["nan_pct"]))
    df_score_ctx = df_score_ctx.dropna(subset=[state_col, margin_col])
    logger.info("Rows after dropna ctx: %s_ctx=%s", args.score_tf, len(df_score_ctx))
    score_ctx_dropna = len(df_score_ctx)
    feature_builder = FeatureBuilder(context_tf=context_tf)
    features_all = feature_builder.build(df_score_ctx)
    # --- atr_ratio (diagnostic-only, comparable cross-symbol) ---
    atr14 = df_score_ctx["atr_14"].reindex(features_all.index)
    
    if "atr_short" in features_all.columns:
        atr_short = pd.to_numeric(features_all["atr_short"], errors="coerce")
        atr_ratio = (atr_short / atr14.replace(0, np.nan)).astype(float)
    else:
        atr_ratio = pd.Series(np.nan, index=features_all.index)

    if research_enabled:
        research_features = _build_research_context_features(
            df_score_ctx,
            ohlcv_ctx,
            args.symbol,
            research_cfg,
        )
        if not research_features.empty:
            features_all = pd.concat([features_all, research_features], axis=1)
            logger.info("Research features enabled: %s", list(research_features.columns))

    state_labels_before = df_score_ctx[state_col].map(_state_label)
    state_counts_before = state_labels_before.value_counts()
    logger.debug("State mix (%s_ctx): %s", args.score_tf, state_counts_before.to_dict())

    score_total = len(ohlcv_score)

    event_config = EventDetectionConfig(vwap_tz_server=args.vwap_tz_server).for_timeframe(args.score_tf)
    vwap_summary = _vwap_validity_summary(ohlcv_score, event_config)
    vwap_plot_path = _write_vwap_diagnostic_plot(
        ohlcv_score,
        args.symbol,
        event_config,
        logger,
        check_date=args.vwap_check_date,
    )
    if vwap_plot_path is not None:
        logger.info("VWAP diagnostic plot path=%s", vwap_plot_path)
    detected_events = detect_events(df_score_ctx, config=event_config)
    detected_events = _apply_event_allow_gates(
        detected_events,
        required_allow_by_family=required_allow_map,
        logger=logger,
    )
    if args.phase_e:
        logger.info("Phase E contract OK | allow coverage logged above.")
        detected_events = _force_phase_e_allow_identity(
            detected_events,
            required_allow_by_family=required_allow_map,
            logger=logger,
        )
        if not detected_events.empty:
            assert detected_events["allow_id"].astype(str).str.contains(",", na=False).sum() == 0
            assert (detected_events["allow_id"] == "LOOK_FOR_none").sum() == 0
    vwap_valid_pct = detected_events.attrs.get("vwap_valid_pct")
    vwap_invalid_reason = detected_events.attrs.get("vwap_invalid_reason")
    vwap_invalid_bars = detected_events.attrs.get("vwap_invalid_bars")
    logger.info(
        "VWAP validity (events): valid_pct=%s invalid_reason=%s invalid_bars=%s",
        vwap_valid_pct,
        vwap_invalid_reason,
        vwap_invalid_bars,
    )
    if vwap_invalid_bars:
        logger.info("VWAP events excluded on invalid bars=%s", vwap_invalid_bars)
    event_counts = {
        "by_family": detected_events["family_id"].value_counts().to_dict() if not detected_events.empty else {},
        "by_type": detected_events["event_type"].value_counts().to_dict() if not detected_events.empty else {},
    }
    if detected_events.empty:
        logger.warning("No events detected; exiting.")
        events_diag = pd.DataFrame(
            columns=[state_col, "allow_id", "margin", "r", "win", "margin_bin"]
        )
        events_diag.index = pd.DatetimeIndex([])
        is_fallback = _apply_fallback_guardrails(0, args.fallback_min_samples)
        if is_fallback:
            logger.warning(
                "Fallback diagnostics: events_total_post_meta=0 < fallback_min_samples=%s",
                args.fallback_min_samples,
            )
        diagnostic_report = _build_training_diagnostic_report(
            events_diag,
            df_score_ctx,
            thresholds=args,
            state_col=state_col,
            phase_e=args.phase_e,
        )
        supply_funnel = pd.DataFrame(
            [
                {
                    "score_total": score_total,
                    "after_merge": score_ctx_merged,
                    "after_dropna_ctx": score_ctx_dropna,
                    "events_detected": 0,
                    "events_labeled": 0,
                    "events_post_state_filter": 0,
                    "events_post_meta": 0,
                }
            ]
        )
        _persist_diagnostic_tables(
            diagnostic_report=diagnostic_report,
            supply_funnel=supply_funnel,
            model_dir=args.model_dir,
            output_prefix=output_prefix,
            output_suffix="",
            mode_suffix=mode_suffix,
            run_id=run_id,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            config_path=config_path,
            mode=effective_mode,
            config_hash=config_hash,
            prompt_version=prompt_version,
            logger=logger,
        )
        summary_payload = {
            "run_id": run_id,
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "score_tf": args.score_tf,
            "allow_tf": args.allow_tf,
            "context_tf": args.context_tf,
            "state_model_path": str(model_path),
            "state_model_metadata_timeframe": (
                state_model.metadata.get("timeframe") if isinstance(state_model.metadata, dict) else None
            ),
            "config_path": str(config_path) if config_path is not None else None,
            "config_hash": config_hash,
            "prompt_version": prompt_version,
            "allow_cutoff": allow_cutoff,
            "context_cutoff": allow_cutoff,
            "m5_cutoff": score_cutoff,
            "score_cutoff": score_cutoff,
            "k_bars": args.k_bars,
            "reward_r": args.reward_r,
            "sl_mult": args.sl_mult,
            "r_thr": args.r_thr,
            "tie_break": args.tie_break,
            "meta_policy": {
                "enabled": args.meta_policy == "on",
                "meta_margin_min": args.meta_margin_min,
                "meta_margin_max": args.meta_margin_max,
            },
            "phase_e": args.phase_e,
            "verdict": "TELEMETRY_ONLY" if args.phase_e else "NO_EVENTS_DETECTED",
            "event_counts": event_counts,
            "context_columns": {
                "state_col": state_col,
                "margin_col": margin_col,
            },
            "context_nan_pct": context_nan_pct,
        }
        summary_path = args.model_dir / f"summary_{output_prefix}_event_scorer.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)
        logger.info("summary_out=%s", summary_path)
        supply_funnel_payload = {
            "score_total": score_total,
            "after_merge": score_ctx_merged,
            "after_ctx_dropna": score_ctx_dropna,
            "events_detected": 0,
            "events_labeled": 0,
            "events_post_state_filter": 0,
            "events_post_meta": 0,
        }
        if args.telemetry == "screen":
            _emit_screen_telemetry(
                args=args,
                run_id=run_id,
                mode=effective_mode,
                k_bars=args.k_bars,
                allow_cutoff=allow_cutoff,
                score_cutoff=score_cutoff,
                supply_funnel=supply_funnel_payload,
                vwap_summary=vwap_summary,
                baseline_summary=_baseline_state_summary(pd.DataFrame()),
                scorer_summary=_scorer_vs_baseline_summary(pd.DataFrame(), phase_e=args.phase_e),
                best_regime={"regime_id": "NA", "n": 0, "delta_r_mean@20": float("nan"), "flag": "NA"},
                worst_regime={"regime_id": "NA", "n": 0, "delta_r_mean@20": float("nan"), "flag": "NA"},
            )
        elif args.telemetry == "triage":
            _emit_triage_telemetry(
                args=args,
                run_id=run_id,
                mode=effective_mode,
                k_bars=args.k_bars,
                allow_cutoff=allow_cutoff,
                score_cutoff=score_cutoff,
                top_regimes=pd.DataFrame(),
                bottom_regimes=pd.DataFrame(),
                inverted_count=0,
                evaluated_regimes=0,
                stability_summary={},
                family_status_counts={},
            )
        if research_mode:
            _print_research_summary_block(
                diagnostic_report.get("session_conditional_edge", pd.DataFrame()),
                state_col=state_col,
                phase_e=args.phase_e,
            )
        return

    events_dupes = int(detected_events.index.duplicated().sum())
    logger.info("Detected events by type:\n%s", detected_events["event_type"].value_counts().to_string())
    logger.info("Detected events by family:\n%s", detected_events["family_id"].value_counts().to_string())

    event_indexer = ohlcv_score.index.get_indexer(detected_events.index)
    missing_index = int((event_indexer == -1).sum())
    missing_future = int(((event_indexer != -1) & (event_indexer + 1 >= len(ohlcv_score.index))).sum())
    events_index_match_pct = float(detected_events.index.isin(df_score_ctx.index).mean())
    missing_atr_pct = float(detected_events["atr_14"].isna().mean() * 100)
    score_atr14_nan_pct = float(df_score_ctx["atr_14"].isna().mean() * 100)
    events_atr14_nan_pct = float(detected_events["atr_14"].isna().mean() * 100)
    sanity_table = pd.DataFrame(
        [
            {
                "score_dupes_detected": score_dupes,
                "h1_dupes_detected": h1_dupes,
                "event_dupes_detected": events_dupes,
                "events_missing_index": missing_index,
                "events_missing_future_slice": missing_future,
                "events_missing_atr_pct": round(missing_atr_pct, 2),
                "events_index_match_pct": round(events_index_match_pct, 4),
                "score_atr14_nan_pct": round(score_atr14_nan_pct, 2),
                "events_atr14_nan_pct": round(events_atr14_nan_pct, 2),
            }
        ]
    )
    logger.info("Data quality checks:\n%s", sanity_table.to_string(index=False))
    if events_index_match_pct < 0.99:
        event_ts = detected_events["ts"] if "ts" in detected_events.columns else detected_events.index
        event_sample = None if detected_events.empty else detected_events.index[0]
        ctx_sample = None if df_score_ctx.empty else df_score_ctx.index[0]
        logger.warning(
            "Index mismatch details: events_ts_dtype=%s score_ctx_index_dtype=%s events_tz=%s score_ctx_tz=%s "
            "events_sample=%r score_ctx_sample=%r",
            getattr(event_ts, "dtype", None),
            getattr(df_score_ctx.index, "dtype", None),
            getattr(getattr(event_ts, "dt", event_ts), "tz", None),
            getattr(df_score_ctx.index, "tz", None),
            event_sample,
            ctx_sample,
        )

    base_thresholds = {
        "n_min": args.decision_n_min,
        "winrate_min": args.decision_winrate_min,
        "r_mean_min": args.decision_r_mean_min,
        "p10_min": args.decision_p10_min,
    }
    research_variants: list = []
    variants_by_k: dict[int, list] = {}
    if research_enabled and exploration_enabled:
        thresholds_grid = (
            exploration_cfg.get("decision_thresholds_grid", {})
            if isinstance(exploration_cfg, dict)
            else {}
        )
        exploration_seed = int(exploration_cfg.get("seed", seed)) if isinstance(exploration_cfg, dict) else seed
        max_variants = exploration_cfg.get("max_variants") if isinstance(exploration_cfg, dict) else None
        research_variants = generate_research_variants(
            kind=exploration_kind,
            base_k_bars=args.k_bars,
            k_bars_grid=k_bars_grid if isinstance(k_bars_grid, list) else None,
            base_thresholds=base_thresholds,
            thresholds_grid=thresholds_grid,
            seed=exploration_seed,
            max_variants=max_variants,
        )
        for variant in research_variants:
            variants_by_k.setdefault(int(variant.k_bars), []).append(variant)

    research_grid_frames: list[pd.DataFrame] = []
    research_family_frames: list[pd.DataFrame] = []
    for k_bars in k_values:
        output_suffix = f"_k{k_bars}" if multi_k else ""
        variant_results = _run_training_for_k(
            args=args,
            k_bars=k_bars,
            output_suffix=output_suffix,
            output_prefix=output_prefix,
            run_id=run_id,
            config_path=config_path,
            detected_events=detected_events,
            event_config=event_config,
            df_score_ctx=df_score_ctx,
            ohlcv_score=ohlcv_score,
            features_all=features_all,
            atr_ratio=atr_ratio,
            feature_builder=feature_builder,
            score_total=score_total,
            score_ctx_merged=score_ctx_merged,
            score_ctx_dropna=score_ctx_dropna,
            allow_cutoff=allow_cutoff,
            score_cutoff=score_cutoff,
            vwap_summary=vwap_summary,
            min_samples_train=min_samples_train,
            seed=seed,
            scorer_out_base=scorer_out_base,
            mode_suffix=mode_suffix,
            mode_effective=effective_mode,
            research_mode=research_mode,
            research_cfg=research_cfg,
            research_variants=variants_by_k.get(int(k_bars), []),
            baseline_thresholds=baseline_thresholds,
            baseline_thresholds_source=baseline_thresholds_source,
            config_hash=config_hash,
            prompt_version=prompt_version,
            state_model_path=model_path,
            state_model_metadata=state_model.metadata,
            context_nan_pct=context_nan_pct,
            event_counts=event_counts,
        )
        if research_enabled and exploration_enabled and variant_results is not None:
            variant_report, family_report = variant_results
            research_grid_frames.append(variant_report)
            if family_report is not None:
                research_family_frames.append(family_report)
    if research_enabled and exploration_enabled:
        if research_grid_frames:
            combined_grid = pd.concat(research_grid_frames, ignore_index=True)
        else:
            combined_grid = pd.DataFrame()
        combined_families = pd.concat(research_family_frames, ignore_index=True) if research_family_frames else None
        research_summary = _build_research_summary_from_grid(combined_grid, phase_e=args.phase_e)
        _persist_research_outputs(
            model_dir=args.model_dir,
            symbol=args.symbol,
            allow_tf=args.allow_tf,
            score_tf=args.score_tf,
            context_tf=args.context_tf,
            run_id=run_id,
            config_path=config_path,
            grid_results=combined_grid,
            family_results=combined_families,
            summary_payload=research_summary,
            research_cfg=research_cfg,
            config_hash=config_hash,
            prompt_version=prompt_version,
            baseline_thresholds=baseline_thresholds,
            baseline_thresholds_source=baseline_thresholds_source,
            mode_suffix=mode_suffix,
            research_mode=research_mode,
            mode=effective_mode,
            phase_e=args.phase_e,
        )
    return


if __name__ == "__main__":
    main()
