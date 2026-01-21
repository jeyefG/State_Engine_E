"""Phase D LOOK_FOR tagging logic for State Engine outputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from .config_loader import normalize_phase_d_config
from .labels import StateLabels

LOOK_FOR_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "session": ("ctx_session_bucket",),
    "state_age": ("ctx_state_age",),
    "dist_vwap_atr": ("ctx_dist_vwap_atr",),
    "breakmag": ("BreakMag", "ctx_breakmag"),
    "reentry": ("ReentryCount", "ctx_reentry_count", "ctx_reentrycount"),
    "margin": ("margin", "margin_ctx", "margin_context"),
}


def resolve_look_for_column_map(available_columns: set[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for key, candidates in LOOK_FOR_COLUMN_CANDIDATES.items():
        resolved[key] = next((name for name in candidates if name in available_columns), None)
    return resolved


@dataclass(frozen=True)
class PhaseDConfig:
    enabled: bool = False
    look_fors: dict[str, dict[str, Any]] | None = None


def _parse_phase_d_config(symbol_cfg: dict | None) -> PhaseDConfig:
    if not isinstance(symbol_cfg, dict):
        return PhaseDConfig(enabled=False, look_fors=None)
    normalized_cfg = normalize_phase_d_config(deepcopy(symbol_cfg))
    phase_d = normalized_cfg.get("phase_d")
    if not isinstance(phase_d, dict):
        return PhaseDConfig(enabled=False, look_fors=None)
    enabled = bool(phase_d.get("enabled", False))
    look_fors = phase_d.get("look_fors")
    if look_fors is not None and not isinstance(look_fors, dict):
        raise ValueError("phase_d.look_fors must be a mapping when provided.")
    return PhaseDConfig(enabled=enabled, look_fors=look_fors)


def _base_state_mask(state_hat: pd.Series, base_state: str, index: pd.Index) -> pd.Series:
    base_state_norm = str(base_state).strip().lower()
    if base_state_norm == "trend":
        mask = state_hat == StateLabels.TREND
    elif base_state_norm == "balance":
        mask = state_hat == StateLabels.BALANCE
    elif base_state_norm == "transition":
        mask = state_hat == StateLabels.TRANSITION
    elif base_state_norm == "any":
        mask = pd.Series(True, index=index)
    else:
        raise ValueError(
            f"Invalid base_state={base_state}. Expected one of ['balance', 'transition', 'trend', 'any']."
        )
    return mask.reindex(index).fillna(False).astype(bool)


class GatingPolicy:
    """Apply LOOK_FOR_* tags based on StateEngine outputs and context."""

    def __init__(self, config: PhaseDConfig | None = None) -> None:
        self.config = config or PhaseDConfig()

    def apply(
        self,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None = None,
        *,
        logger=None,
        symbol: str | None = None,
        config_meta: dict | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame with LOOK_FOR_* columns."""
        required = {"state_hat", "margin"}
        missing = required - set(outputs.columns)
        if missing:
            raise ValueError(f"Missing required output columns: {sorted(missing)}")

        # --- HARD ALIGN: features must follow outputs index (outputs is post-clean) ---
        if features is not None and not features.index.equals(outputs.index):
            if logger is not None:
                logger.info(
                    "LOOK_FOR_ALIGN symbol=%s outputs_len=%s features_len_before=%s idx_equal_before=%s",
                    symbol,
                    len(outputs),
                    len(features),
                    features.index.equals(outputs.index),
                )
            features = features.reindex(outputs.index)
            if logger is not None:
                logger.info(
                    "LOOK_FOR_ALIGN_DONE symbol=%s outputs_len=%s features_len_after=%s idx_equal_after=%s",
                    symbol,
                    len(outputs.index),
                    len(features.index),
                    features.index.equals(outputs.index),
                )
        # ---------------------------------------------------------------------------

        phase_d = _parse_phase_d_config(config_meta)
        if not phase_d.enabled:
            if logger is not None:
                logger.info("PHASE_D_DISABLED symbol=%s reason=no phase_d config", symbol)
            return pd.DataFrame(index=outputs.index)

        look_fors = phase_d.look_fors or {}
        if not look_fors:
            if logger is not None:
                logger.info("PHASE_D_DISABLED symbol=%s reason=empty look_fors", symbol)
            return pd.DataFrame(index=outputs.index)

        state_hat = outputs["state_hat"]
        look_for_payload: dict[str, pd.Series] = {}
        for look_for_name, rule_cfg in look_fors.items():
            if not isinstance(rule_cfg, dict):
                raise ValueError(f"LOOK_FOR {look_for_name} config must be a mapping.")
            if not rule_cfg.get("enabled", False):
                continue
            base_state = rule_cfg.get("base_state") or rule_cfg.get("anchor_state")
            if base_state is None:
                raise ValueError(f"LOOK_FOR {look_for_name} missing required base_state.")
            base_mask = _base_state_mask(state_hat, base_state, outputs.index)
            filtered_mask, counts, _ = self._apply_look_for_filters(
                look_for_name,
                base_mask,
                outputs,
                features,
                rule_cfg,
                logger,
                symbol,
            )
            look_for_payload[look_for_name] = filtered_mask.astype(int)
            if logger is not None and counts is not None:
                total_rows = max(int(counts.get("total_rows", 0)), 1)
                n_before = int(counts.get("n_base", 0))
                n_after = int(counts.get("n_pass", 0))
                coverage = (n_after / total_rows) * 100.0 if total_rows else 0.0
                logger.info(
                    "LOOK_FOR_AUDIT symbol=%s look_for=%s base_state=%s filters=%s missing_cols=%s "
                    "required_cols=%s available_cols=%s resolved_column_map=%s n_before=%s n_after=%s coverage=%.2f%%",
                    symbol,
                    look_for_name,
                    counts.get("base_state"),
                    counts.get("active_filters"),
                    counts.get("missing_columns"),
                    counts.get("required_cols"),
                    counts.get("available_cols"),
                    counts.get("resolved_column_map"),
                    n_before,
                    n_after,
                    coverage,
                )

        return pd.DataFrame(look_for_payload, index=outputs.index)

    def _apply_look_for_filters(
        self,
        look_for_name: str,
        base_mask: pd.Series,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None,
        rule_cfg: dict[str, Any],
        logger,
        symbol: str | None,
    ) -> tuple[pd.Series, dict[str, Any], pd.DataFrame | None]:
        def _get_col(column: str | None) -> pd.Series | None:
            if column is None:
                return None
            if features is not None and column in features.columns:
                return features[column]
            if column in outputs.columns:
                return outputs[column]
            return None

        idx = outputs.index
        base_mask = base_mask.reindex(idx).fillna(False).astype(bool)
        if features is not None and not features.index.equals(idx):
            features = features.reindex(idx)

        filters_cfg = rule_cfg.get("filters", {}) if isinstance(rule_cfg.get("filters"), dict) else {}
        require_all = bool(rule_cfg.get("require_all", True))

        available_columns = set(outputs.columns)
        if features is not None:
            available_columns |= set(features.columns)
        resolved_column_map = resolve_look_for_column_map(available_columns)

        session_series = _get_col(resolved_column_map.get("session"))
        state_age_series = _get_col(resolved_column_map.get("state_age"))
        dist_series = _get_col(resolved_column_map.get("dist_vwap_atr"))
        breakmag_series = _get_col(resolved_column_map.get("breakmag"))
        reentry_series = _get_col(resolved_column_map.get("reentry"))
        margin_series = _get_col(resolved_column_map.get("margin"))

        masks: list[pd.Series] = []
        mask_map: dict[str, pd.Series] = {}
        applied: set[str] = set()
        missing_columns: list[str] = []
        active_filters: list[str] = []
        required_cols: list[str] = []

        def _mark_required(key: str) -> None:
            resolved = resolved_column_map.get(key)
            if resolved is not None:
                required_cols.append(resolved)
            else:
                required_cols.append(LOOK_FOR_COLUMN_CANDIDATES[key][0])

        sessions_in = filters_cfg.get("sessions_in")
        if sessions_in is not None and session_series is None:
            _mark_required("session")
            missing_columns.append("ctx_session_bucket")
        if sessions_in is not None and session_series is not None:
            _mark_required("session")
            allowed = {str(val) for val in sessions_in}
            mask = session_series.astype(str).isin(allowed)
            masks.append(mask)
            mask_map["session"] = mask
            applied.add("session")
            active_filters.append("sessions_in")

        state_age_min = filters_cfg.get("state_age_min")
        state_age_max = filters_cfg.get("state_age_max")
        if (state_age_min is not None or state_age_max is not None) and state_age_series is None:
            _mark_required("state_age")
            missing_columns.append("ctx_state_age")
        if (state_age_min is not None or state_age_max is not None) and state_age_series is not None:
            _mark_required("state_age")
            series = pd.to_numeric(state_age_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if state_age_min is not None:
                mask &= series >= float(state_age_min)
            if state_age_max is not None:
                mask &= series <= float(state_age_max)
            masks.append(mask)
            mask_map["state_age"] = mask
            applied.add("state_age")
            active_filters.append("state_age")

        dist_min = filters_cfg.get("dist_vwap_atr_min")
        dist_max = filters_cfg.get("dist_vwap_atr_max")
        if (dist_min is not None or dist_max is not None) and dist_series is None:
            _mark_required("dist_vwap_atr")
            missing_columns.append("ctx_dist_vwap_atr")
        if (dist_min is not None or dist_max is not None) and dist_series is not None:
            _mark_required("dist_vwap_atr")
            series = pd.to_numeric(dist_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if dist_min is not None:
                mask &= series >= float(dist_min)
            if dist_max is not None:
                mask &= series <= float(dist_max)
            masks.append(mask)
            mask_map["dist_vwap_atr"] = mask
            applied.add("dist_vwap_atr")
            active_filters.append("dist_vwap_atr")

        breakmag_min = filters_cfg.get("breakmag_min")
        breakmag_max = filters_cfg.get("breakmag_max")
        if (breakmag_min is not None or breakmag_max is not None) and breakmag_series is None:
            _mark_required("breakmag")
            missing_columns.append("BreakMag")
        if (breakmag_min is not None or breakmag_max is not None) and breakmag_series is not None:
            _mark_required("breakmag")
            series = pd.to_numeric(breakmag_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if breakmag_min is not None:
                mask &= series >= float(breakmag_min)
            if breakmag_max is not None:
                mask &= series <= float(breakmag_max)
            masks.append(mask)
            mask_map["breakmag"] = mask
            applied.add("breakmag")
            active_filters.append("breakmag")

        reentry_min = filters_cfg.get("reentry_min")
        reentry_max = filters_cfg.get("reentry_max")
        if (reentry_min is not None or reentry_max is not None) and reentry_series is None:
            _mark_required("reentry")
            missing_columns.append("ReentryCount")
        if (reentry_min is not None or reentry_max is not None) and reentry_series is not None:
            _mark_required("reentry")
            series = pd.to_numeric(reentry_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if reentry_min is not None:
                mask &= series >= float(reentry_min)
            if reentry_max is not None:
                mask &= series <= float(reentry_max)
            masks.append(mask)
            mask_map["reentry"] = mask
            applied.add("reentry")
            active_filters.append("reentry")

        margin_min = filters_cfg.get("margin_min")
        margin_max = filters_cfg.get("margin_max")
        if (margin_min is not None or margin_max is not None) and margin_series is None:
            _mark_required("margin")
            missing_columns.append("margin")
        if (margin_min is not None or margin_max is not None) and margin_series is not None:
            _mark_required("margin")
            series = pd.to_numeric(margin_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if margin_min is not None:
                mask &= series >= float(margin_min)
            if margin_max is not None:
                mask &= series <= float(margin_max)
            masks.append(mask)
            mask_map["margin"] = mask
            applied.add("margin")
            active_filters.append("margin")

        if missing_columns:
            if logger is not None:
                logger.error(
                    "LOOK_FOR_COLUMNS_MISSING symbol=%s look_for=%s required_cols=%s available_cols=%s "
                    "resolved_column_map=%s missing_cols=%s",
                    symbol,
                    look_for_name,
                    sorted(set(required_cols)),
                    sorted(available_columns),
                    resolved_column_map,
                    sorted(set(missing_columns)),
                )
            raise ValueError(
                "LOOK_FOR {name} missing required columns: {missing}. "
                "required_cols={required} available_cols={available} resolved_column_map={resolved}".format(
                    name=look_for_name,
                    missing=sorted(set(missing_columns)),
                    required=sorted(set(required_cols)),
                    available=sorted(available_columns),
                    resolved=resolved_column_map,
                )
            )

        if logger is not None:
            def _typed(value: Any) -> dict[str, Any]:
                return {"value": value, "type": type(value).__name__}

            typed_filters = {
                "sessions_in": _typed(sessions_in),
                "state_age_min": _typed(state_age_min),
                "state_age_max": _typed(state_age_max),
                "dist_vwap_atr_min": _typed(dist_min),
                "dist_vwap_atr_max": _typed(dist_max),
                "breakmag_min": _typed(breakmag_min),
                "breakmag_max": _typed(breakmag_max),
                "reentry_min": _typed(reentry_min),
                "reentry_max": _typed(reentry_max),
                "margin_min": _typed(margin_min),
                "margin_max": _typed(margin_max),
            }
            logger.info(
                "LOOK_FOR_CONFIG_EFFECTIVE symbol=%s look_for=%s base_state=%s enabled=%s "
                "require_all=%s filters=%s",
                symbol,
                look_for_name,
                rule_cfg.get("base_state"),
                True,
                require_all,
                typed_filters,
            )

        if not masks:
            if logger is not None:
                logger.info(
                    "NO_EFFECTIVE_CONDITIONS symbol=%s look_for=%s",
                    symbol,
                    look_for_name,
                )
            counts = {
                "total_rows": int(len(idx)),
                "n_base": int(base_mask.sum()),
                "n_pass": int(base_mask.sum()),
                "n_fail": 0,
                "status": "no_effective_conditions",
                "active_filters": active_filters,
                "missing_columns": sorted(set(missing_columns)),
                "required_cols": sorted(set(required_cols)),
                "available_cols": sorted(available_columns),
                "resolved_column_map": resolved_column_map,
                "base_state": rule_cfg.get("base_state"),
            }
            return base_mask, counts, None

        if require_all:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
        combined_mask = combined_mask.reindex(idx).fillna(False)

        filtered_mask = base_mask & combined_mask
        total_rows = int(len(idx))
        n_base = int(base_mask.sum())
        n_pass = int(filtered_mask.sum())
        n_fail = n_base - n_pass

        def _pass_count(key: str) -> int | None:
            if key not in applied:
                return None
            return int((base_mask & mask_map[key].reindex(idx).fillna(False)).sum())

        counts = {
            "total_rows": total_rows,
            "n_base": n_base,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_session": _pass_count("session"),
            "pass_state_age": _pass_count("state_age"),
            "pass_dist": _pass_count("dist_vwap_atr"),
            "pass_breakmag": _pass_count("breakmag"),
            "pass_reentry": _pass_count("reentry"),
            "pass_margin": _pass_count("margin"),
            "active_filters": active_filters,
            "missing_columns": sorted(set(missing_columns)),
            "required_cols": sorted(set(required_cols)),
            "available_cols": sorted(available_columns),
            "resolved_column_map": resolved_column_map,
            "base_state": rule_cfg.get("base_state"),
        }

        if logger is not None:
            logger.info(
                "LOOK_FOR_FILTER_COUNTS symbol=%s look_for=%s n_base=%s n_pass=%s n_fail=%s "
                "pass_session=%s pass_state_age=%s pass_dist=%s pass_breakmag=%s pass_reentry=%s pass_margin=%s",
                symbol,
                look_for_name,
                counts["n_base"],
                counts["n_pass"],
                counts["n_fail"],
                counts.get("pass_session"),
                counts.get("pass_state_age"),
                counts.get("pass_dist"),
                counts.get("pass_breakmag"),
                counts.get("pass_reentry"),
                counts.get("pass_margin"),
            )
            total_rows = max(total_rows, 1)
            before_pct = 100.0 * n_base / total_rows
            after_pct = 100.0 * n_pass / total_rows
            logger.info(
                "LOOK_FOR_RATE symbol=%s look_for=%s total=%s before=%.2f%% after=%.2f%% delta=%.2f%%",
                symbol,
                look_for_name,
                total_rows,
                before_pct,
                after_pct,
                after_pct - before_pct,
            )

        fail_samples = None
        if n_fail > 0:
            fail_mask = base_mask & ~combined_mask
            failure_rows = pd.DataFrame(index=idx)
            failure_rows["index"] = idx.astype(str)
            failure_rows["ctx_session_bucket"] = (
                session_series.reindex(idx).values if session_series is not None else [None] * len(idx)
            )
            failure_rows["ctx_state_age"] = (
                state_age_series.reindex(idx).values if state_age_series is not None else [None] * len(idx)
            )
            failure_rows["ctx_dist_vwap_atr"] = (
                dist_series.reindex(idx).values if dist_series is not None else [None] * len(idx)
            )
            failure_rows = failure_rows.loc[fail_mask]
            failure_rows = failure_rows.head(5).copy()
            reasons: list[str] = []
            for row_idx in failure_rows.index:
                parts: list[str] = []
                for key, label in [
                    ("session", "session"),
                    ("state_age", "state_age"),
                    ("dist_vwap_atr", "dist_vwap_atr"),
                    ("breakmag", "breakmag"),
                    ("reentry", "reentry"),
                    ("margin", "margin"),
                ]:
                    if key in applied and not bool(mask_map[key].reindex(idx).fillna(False).loc[row_idx]):
                        parts.append(label)
                reasons.append("|".join(parts) if parts else "unknown")
            failure_rows.insert(1, "reason", reasons)
            fail_samples = failure_rows
            if logger is not None:
                logger.info(
                    "LOOK_FOR_FILTER_FAIL_SAMPLES symbol=%s look_for=%s\n%s",
                    symbol,
                    look_for_name,
                    fail_samples.to_string(index=False),
                )

        return filtered_mask, counts, fail_samples


__all__ = [
    "LOOK_FOR_COLUMN_CANDIDATES",
    "resolve_look_for_column_map",
    "PhaseDConfig",
    "GatingPolicy",
]
