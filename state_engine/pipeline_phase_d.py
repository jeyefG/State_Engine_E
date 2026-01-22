"""Phase D context builder for canonical State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from .context_features import build_context_features
from .features import FeatureEngineer
from .gating import GatingPolicy, LOOK_FOR_COLUMN_CANDIDATES, resolve_look_for_column_map
from .labels import StateLabels
from .model import StateEngineModel
from .quality import QUALITY_LABEL_UNCLASSIFIED


@dataclass(frozen=True)
class ContextBundle:
    ctx_df: pd.DataFrame
    ohlcv_score: pd.DataFrame
    df_score_ctx: pd.DataFrame
    meta: dict[str, Any]


def audit_phase_d_contamination(*, logger: logging.Logger) -> None:
    patterns: dict[str, str] = {
        "train_event_scorer": "Phase D must not depend on Phase E training scripts.",
        "event_scorer": "Phase D must not contain Phase E scorer logic.",
        "phase_e": "Phase D must not branch on Phase E flags.",
        "Phase E": "Phase D must not embed Phase E concerns.",
        "scorer": "Phase D must not embed Event Scorer logic.",
    }
    targets: dict[str, Path] = {
        "gating.py": Path(__file__).resolve().with_name("gating.py"),
        "context_features.py": Path(__file__).resolve().with_name("context_features.py"),
    }
    findings: list[str] = []
    for name, path in targets.items():
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        for pattern, reason in patterns.items():
            if pattern in content:
                findings.append(f"{name}:{pattern} -> {reason}")
    if findings:
        logger.error("AUDIT_FAIL Phase D contamination detected: %s", "; ".join(findings))
    else:
        logger.info("AUDIT_OK Phase D gating/context_features clean.")


def _active_look_for_filters(symbol_cfg: dict | None) -> list[str]:
    if not isinstance(symbol_cfg, dict):
        return []
    phase_d = symbol_cfg.get("phase_d")
    if not isinstance(phase_d, dict):
        return []
    if not phase_d.get("enabled", False):
        return []
    look_for_cfg = phase_d.get("look_fors")
    if not isinstance(look_for_cfg, dict):
        return []
    return sorted(
        [
            look_for_name
            for look_for_name, rule_cfg in look_for_cfg.items()
            if isinstance(rule_cfg, dict) and rule_cfg.get("enabled", False)
        ]
    )


def _look_for_base_state_map(symbol_cfg: dict | None) -> dict[str, str]:
    if not isinstance(symbol_cfg, dict):
        return {}
    phase_d = symbol_cfg.get("phase_d")
    if not isinstance(phase_d, dict):
        return {}
    look_for_cfg = phase_d.get("look_fors")
    if not isinstance(look_for_cfg, dict):
        return {}
    base_state_map: dict[str, str] = {}
    for look_for_name, rule_cfg in look_for_cfg.items():
        if not isinstance(rule_cfg, dict):
            continue
        base_state = rule_cfg.get("base_state") or rule_cfg.get("anchor_state")
        if base_state is None:
            continue
        base_state_map[look_for_name] = str(base_state).strip().lower()
    return base_state_map


def _infer_base_state_from_name(look_for_name: str) -> str | None:
    match = re.search(r"look_for_(balance|transition|trend)(?:_|$)", look_for_name.lower())
    if match:
        return match.group(1)
    return None


def _normalize_state_labels(state_hat: pd.Series) -> pd.Series:
    mapping = {
        StateLabels.BALANCE: "BALANCE",
        StateLabels.TRANSITION: "TRANSITION",
        StateLabels.TREND: "TREND",
        int(StateLabels.BALANCE): "BALANCE",
        int(StateLabels.TRANSITION): "TRANSITION",
        int(StateLabels.TREND): "TREND",
        "balance": "BALANCE",
        "transition": "TRANSITION",
        "trend": "TREND",
        "BALANCE": "BALANCE",
        "TRANSITION": "TRANSITION",
        "TREND": "TREND",
    }
    normalized = state_hat.map(mapping)
    return normalized.fillna("UNKNOWN")


def validate_look_for_context_requirements(
    symbol_cfg: dict | None,
    available_columns: set[str],
    *,
    logger: logging.Logger,
) -> None:
    audit_phase_d_contamination(logger=logger)
    if not isinstance(symbol_cfg, dict):
        return
    phase_d = symbol_cfg.get("phase_d")
    if not isinstance(phase_d, dict):
        logger.info("Phase D disabled: no phase_d config.")
        return
    if not phase_d.get("enabled", False):
        logger.info("Phase D disabled: phase_d.enabled=false.")
        return
    look_for_cfg = phase_d.get("look_fors")
    if not isinstance(look_for_cfg, dict):
        raise ValueError("phase_d.look_fors must be a mapping when phase_d.enabled=true.")

    resolved_column_map = resolve_look_for_column_map(available_columns)

    def _has_any(candidates: list[str]) -> bool:
        return any(col in available_columns for col in candidates)

    missing_by_rule: dict[str, list[str]] = {}
    missing_base_state: dict[str, str] = {}
    allowed_base_states = {"balance", "transition", "trend", "any"}
    for look_for_rule, rule_cfg in look_for_cfg.items():
        if not isinstance(rule_cfg, dict):
            continue
        base_state = rule_cfg.get("base_state") or rule_cfg.get("anchor_state")
        if base_state is None:
            missing_base_state[look_for_rule] = "missing"
        else:
            base_state_norm = str(base_state).strip().lower()
            if base_state_norm not in allowed_base_states:
                missing_base_state[look_for_rule] = f"invalid:{base_state}"
        if not rule_cfg.get("enabled", False):
            continue
        filters_cfg = rule_cfg.get("filters", {})
        if filters_cfg is None:
            filters_cfg = {}
        if not isinstance(filters_cfg, dict):
            raise ValueError(f"phase_d.look_fors.{look_for_rule}.filters must be a mapping.")
        required: list[str] = []
        if filters_cfg.get("sessions_in") is not None and not _has_any(
            list(LOOK_FOR_COLUMN_CANDIDATES["session"])
        ):
            required.append("ctx_session_bucket")
        if (
            filters_cfg.get("state_age_min") is not None
            or filters_cfg.get("state_age_max") is not None
        ) and resolved_column_map.get("state_age") is None:
            required.append(LOOK_FOR_COLUMN_CANDIDATES["state_age"][0])
        if (
            filters_cfg.get("dist_vwap_atr_min") is not None
            or filters_cfg.get("dist_vwap_atr_max") is not None
        ) and resolved_column_map.get("dist_vwap_atr") is None:
            required.append(LOOK_FOR_COLUMN_CANDIDATES["dist_vwap_atr"][0])
        if (
            filters_cfg.get("breakmag_min") is not None
            or filters_cfg.get("breakmag_max") is not None
        ) and resolved_column_map.get("breakmag") is None:
            required.append(LOOK_FOR_COLUMN_CANDIDATES["breakmag"][0])
        if (
            filters_cfg.get("reentry_min") is not None
            or filters_cfg.get("reentry_max") is not None
        ) and resolved_column_map.get("reentry") is None:
            required.append(LOOK_FOR_COLUMN_CANDIDATES["reentry"][0])
        if (
            filters_cfg.get("margin_min") is not None
            or filters_cfg.get("margin_max") is not None
        ) and resolved_column_map.get("margin") is None:
            required.append(LOOK_FOR_COLUMN_CANDIDATES["margin"][0])

        if required:
            missing_by_rule[look_for_rule] = sorted(set(required))

    if missing_by_rule:
        missing_details = "; ".join(
            f"{look_for_rule} missing={cols}"
            for look_for_rule, cols in sorted(missing_by_rule.items())
        )
        raise ValueError(
            "LOOK_FOR filters enabled but missing columns. "
            f"{missing_details}. Available={sorted(available_columns)} "
            f"resolved_column_map={resolved_column_map}"
        )
    if missing_base_state:
        missing_details = "; ".join(
            f"{look_for_rule} base_state={detail}"
            for look_for_rule, detail in sorted(missing_base_state.items())
        )
        raise ValueError(
            "LOOK_FOR filters missing required base_state. "
            f"{missing_details}. Allowed={sorted(allowed_base_states)}"
        )
    logger.info("Phase D look_for requirements OK | look_fors=%s", sorted(look_for_cfg.keys()))


def validate_look_for_columns(
    ctx_df: pd.DataFrame,
    active_look_fors: list[str],
    *,
    logger: logging.Logger,
) -> dict[str, float]:
    if not active_look_fors:
        logger.info("Phase D context contract OK | look_fors=[] look_for_rates={}")
        return {}
    missing = sorted(set(active_look_fors) - set(ctx_df.columns))
    if missing:
        raise ValueError(f"Missing LOOK_FOR columns in Phase D context: {missing}")
    look_for_rates: dict[str, float] = {}
    for look_for_name in active_look_fors:
        series = pd.to_numeric(ctx_df[look_for_name], errors="coerce")
        if series.isna().any():
            raise ValueError(f"LOOK_FOR column {look_for_name} has NaN values in Phase D context.")
        unique_vals = set(series.unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"LOOK_FOR column {look_for_name} must be binary (0/1). Observed={sorted(unique_vals)}"
            )
        look_for_rates[look_for_name] = float(series.mean() * 100.0) if len(series) else 0.0
    logger.info(
        "Phase D context contract OK | look_fors=%s look_for_rates=%s",
        active_look_fors,
        {k: round(v, 2) for k, v in look_for_rates.items()},
    )
    return look_for_rates


def _merge_context_score(
    ctx_df: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    *,
    context_tf: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    ctx = ctx_df.copy().sort_index()
    score = ohlcv_score.copy().sort_index()
    if getattr(ctx.index, "tz", None) is not None:
        ctx.index = ctx.index.tz_localize(None)
    if getattr(score.index, "tz", None) is not None:
        score.index = score.index.tz_localize(None)
    ctx = ctx.reset_index().rename(columns={ctx.index.name or "index": "time"})
    score = score.reset_index().rename(columns={score.index.name or "index": "time"})
    merged = pd.merge_asof(score, ctx, on="time", direction="backward")
    merged = merged.set_index("time")
    context_tf_norm = str(context_tf).upper()
    state_col = f"state_hat_{context_tf_norm}"
    margin_col = f"margin_{context_tf_norm}"
    merged = merged.rename(columns={"state_hat": state_col, "margin": margin_col})
    look_for_cols = [col for col in merged.columns if col.startswith("LOOK_FOR_")]
    if look_for_cols:
        merged[look_for_cols] = merged[look_for_cols].fillna(0).astype(int)
    missing_ctx = merged[[state_col, margin_col]].isna().mean()
    if (missing_ctx > 0.25).any():
        logger.warning("High missing context after merge: %s", missing_ctx.to_dict())
    return merged


def _log_look_for_session_coverage(
    ctx_df: pd.DataFrame,
    look_for_names: Iterable[str],
    *,
    logger: logging.Logger,
) -> None:
    look_for_names = [name for name in look_for_names if name in ctx_df.columns]
    if not look_for_names:
        return
    if "ctx_session_bucket" not in ctx_df.columns:
        return
    coverage = (
        ctx_df.groupby("ctx_session_bucket")[look_for_names]
        .mean()
        .mul(100.0)
        .round(2)
        .reset_index()
    )
    logger.info("LOOK_FOR session coverage:\n%s", coverage.to_string(index=False))


def build_context_bundle(
    *,
    symbol: str,
    context_tf: str,
    score_tf: str,
    ohlcv_ctx: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    state_model: StateEngineModel,
    feature_engineer: FeatureEngineer,
    gating_policy: GatingPolicy,
    symbol_cfg: dict | None,
    phase_e: bool,
    logger: logging.Logger,
    quality_labels: pd.Series | None = None,
    causal_shift: bool | None = None,
) -> ContextBundle:
    """Build canonical Phase D context and merge into score timeframe."""
    full_features = feature_engineer.compute_features(ohlcv_ctx)
    features = feature_engineer.training_features(full_features)
    outputs = state_model.predict_outputs(features)
    ctx_features = build_context_features(
        ohlcv_ctx,
        outputs,
        symbol=symbol,
        timeframe=context_tf,
    )
    features_for_gating = full_features.join(ctx_features, how="left").reindex(outputs.index)
    available_columns = set(features_for_gating.columns) | set(outputs.columns)
    validate_look_for_context_requirements(
        symbol_cfg,
        available_columns,
        logger=logger,
    )
    allows = gating_policy.apply(
        outputs,
        features=features_for_gating,
        logger=logger,
        symbol=symbol,
        config_meta=symbol_cfg,
    )
    ctx_features = ctx_features.loc[:, ~ctx_features.columns.duplicated()]
    ctx_df = pd.concat([outputs[["state_hat", "margin"]], ctx_features, allows], axis=1)
    if quality_labels is not None:
        quality_aligned = quality_labels.reindex(outputs.index)
        ctx_df["quality_label"] = quality_aligned.fillna(QUALITY_LABEL_UNCLASSIFIED)
    elif "quality_label" in outputs.columns:
        ctx_df["quality_label"] = outputs["quality_label"]
    ctx_df = ctx_df.loc[:, ~ctx_df.columns.duplicated()]
    look_for_cols = [col for col in ctx_df.columns if col.startswith("LOOK_FOR_")]
    ctx_cols = ["state_hat"]
    if "margin" in ctx_df.columns:
        ctx_cols.append("margin")
    ctx_cols += [col for col in ctx_df.columns if col.startswith("ctx_")]
    ctx_cols += look_for_cols
    if "quality_label" in ctx_df.columns:
        ctx_cols.append("quality_label")
    do_causal_shift = (score_tf != context_tf) if causal_shift is None else bool(causal_shift)
    logger.info(
        "Phase D causal shift | symbol=%s context_tf=%s score_tf=%s do_shift=%s ctx_cols=%s",
        symbol,
        context_tf,
        score_tf,
        do_causal_shift,
        ctx_cols,
    )
    if do_causal_shift:
        ctx_df[ctx_cols] = ctx_df[ctx_cols].shift(1)
    if not do_causal_shift:
        first_state = ctx_df["state_hat"].iloc[0]
        if pd.isna(first_state):
            raise ValueError(
                "Phase D context misalignment: state_hat has NaN in first row without causal shift."
            )
    state_names = _normalize_state_labels(ctx_df["state_hat"])
    if "quality_label" in ctx_df.columns:
        quality_series = ctx_df["quality_label"]
    else:
        quality_series = pd.Series(pd.NA, index=ctx_df.index, dtype=object)
    quality_label_full = quality_series.where(
        (~quality_series.isna()) & (quality_series != QUALITY_LABEL_UNCLASSIFIED),
        state_names + f"_{QUALITY_LABEL_UNCLASSIFIED}",
    )
    ctx_df["quality_label_full"] = quality_label_full
    if ctx_df["quality_label_full"].isna().any():
        raise ValueError("quality_label_full contains NaN values in Phase D context.")
    if "quality_label" in ctx_df.columns:
        quality_str = quality_series.astype(str)
        known_state = state_names != "UNKNOWN"
        classified = quality_str != QUALITY_LABEL_UNCLASSIFIED
        mismatches = classified & known_state & (~quality_str.str.startswith(state_names))
        mismatch_rate = float(mismatches.mean()) if len(mismatches) else 0.0
        if mismatch_rate > 0.001:
            raise ValueError(
                "Phase D quality/state mismatch above threshold: "
                f"pct_bad={mismatch_rate * 100:.3f}%"
            )
    look_for_base_map = _look_for_base_state_map(symbol_cfg)
    look_for_mismatches: dict[str, int] = {}
    for look_for_name in look_for_cols:
        base_state = look_for_base_map.get(look_for_name) or _infer_base_state_from_name(
            look_for_name
        )
        if base_state is None or base_state == "any":
            continue
        base_state_norm = base_state.strip().upper()
        mask = state_names == base_state_norm
        mask_known = state_names != "UNKNOWN"
        
        mismatches = int(((ctx_df[look_for_name] == 1) & (~mask) & mask_known).sum())
        if mismatches:
            look_for_mismatches[look_for_name] = mismatches
        
        ctx_df.loc[(~mask) | (~mask_known), look_for_name] = 0
    if look_for_cols:
        ctx_df[look_for_cols] = ctx_df[look_for_cols].fillna(0).astype(int)
    if look_for_mismatches:
        raise ValueError(
            "LOOK_FOR base state mismatch after enforcement: "
            + "; ".join(
                f"{look_for_name} mismatches={count}"
                for look_for_name, count in sorted(look_for_mismatches.items())
            )
        )

    active_look_fors = _active_look_for_filters(symbol_cfg)
    look_for_rates = validate_look_for_columns(ctx_df, active_look_fors, logger=logger)
    logger.info(
        "Phase D look_for summary | required_look_fors=%s produced_look_fors=%s",
        active_look_fors,
        sorted(look_for_cols),
    )
    _log_look_for_session_coverage(ctx_df, active_look_fors, logger=logger)
    logger.info("Phase D ctx_df tail:\n%s", ctx_df.tail(3).to_string())
    if phase_e:
        logger.info("Phase D context bundle ready for Phase E consumer.")

    df_score_ctx = _merge_context_score(ctx_df, ohlcv_score, context_tf=context_tf, logger=logger)
    meta = {
        "symbol": symbol,
        "context_tf": context_tf,
        "score_tf": score_tf,
        "active_look_fors": active_look_fors,
        "look_for_rates": look_for_rates,
    }
    return ContextBundle(
        ctx_df=ctx_df,
        ohlcv_score=ohlcv_score,
        df_score_ctx=df_score_ctx,
        meta=meta,
    )


__all__ = [
    "ContextBundle",
    "build_context_bundle",
    "validate_look_for_columns",
    "validate_look_for_context_requirements",
    "audit_phase_d_contamination",
]
