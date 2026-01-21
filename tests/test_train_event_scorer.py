from pathlib import Path

import pandas as pd

from scripts.train_event_scorer import (
    _resolve_vwap_report_mode,
    _save_model_if_ready,
    _research_guardrails,
    _build_research_summary_from_grid,
    _persist_research_outputs,
    _effective_mode,
    _resolve_baseline_thresholds,
)
from state_engine.events import EventDetectionConfig, EventExtractor
from state_engine.scoring import EventScorerBundle


def test_fallback_no_model_saved(tmp_path: Path) -> None:
    scorer = EventScorerBundle()
    model_path = tmp_path / "scorer.pkl"

    saved = _save_model_if_ready(is_fallback=True, scorer=scorer, path=model_path, metadata={})

    assert saved is False
    assert not model_path.exists()


def test_vwap_reporting_uses_mt5_daily_metadata() -> None:
    idx = pd.date_range("2024-01-01 21:50", periods=6, freq="5min")
    df_m5 = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [100.0] * len(idx),
            "low": [100.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [10.0] * len(idx),
        },
        index=idx,
    )
    config = EventDetectionConfig()
    extractor = EventExtractor(config=config)

    events = extractor.extract(df_m5, symbol="TEST")

    assert events.attrs["vwap_reset_mode_effective"] == "mt5_daily"
    assert events.attrs["vwap_mode"] == "mt5_daily"

    report_mode = _resolve_vwap_report_mode(events, df_m5, config)
    assert report_mode == "mt5_daily"


def test_guardrails_flags_low_calib_samples() -> None:
    score_shape = pd.DataFrame(
        [
            {
                "k": 20,
                "edge_decay": 0.1,
                "temporal_dispersion": 0.5,
                "family_concentration": 0.2,
                "score_tail_slope": 1.0,
            }
        ]
    )
    diagnostics_cfg = {
        "max_family_concentration": 0.6,
        "min_temporal_dispersion": 0.3,
        "max_score_tail_slope": 2.5,
        "min_calib_samples": 200,
    }

    report = _research_guardrails(
        score_shape=score_shape,
        train_metrics=None,
        calib_metrics=None,
        diagnostics_cfg=diagnostics_cfg,
        calib_samples=50,
    )

    assert report.loc[0, "status"] == "RESEARCH_UNSTABLE"
    assert "LOW_CALIB_SAMPLES<200" in report.loc[0, "reasons"]


def test_research_outputs_persisted_only_in_research(tmp_path: Path) -> None:
    grid_results = pd.DataFrame(
        [
            {
                "variant_id": "thr_n80_rm0.02_p10-0.2_wr0.5_k24",
                "variant_type": "thresholds_only",
                "k_bars": 12,
                "n_min": 80,
                "winrate_min": 0.5,
                "r_mean_min": 0.02,
                "p10_min": -0.2,
                "n_total": 250,
                "n_train": 150,
                "n_test": 100,
                "winrate": 0.6,
                "r_mean": 0.12,
                "p10": -0.05,
                "lift10": 1.2,
                "lift20": 1.3,
                "lift50": 1.4,
                "spearman": 0.1,
                "family_concentration_top10": 0.4,
                "temporal_dispersion": 0.5,
                "score_tail_slope": 1.1,
                "qualified": True,
                "fail_reason": "",
                "research_status": "RESEARCH_OK",
            },
            {
                "variant_id": "kbar_n200_rm0.02_p10-0.2_wr0.52_k36",
                "variant_type": "kbars_only",
                "k_bars": 24,
                "n_min": 200,
                "winrate_min": 0.52,
                "r_mean_min": 0.02,
                "p10_min": -0.2,
                "n_total": 300,
                "n_train": 200,
                "n_test": 100,
                "winrate": 0.58,
                "r_mean": 0.08,
                "p10": -0.02,
                "lift10": 1.15,
                "lift20": 1.1,
                "lift50": 1.05,
                "spearman": 0.12,
                "family_concentration_top10": 0.5,
                "temporal_dispersion": 0.6,
                "score_tail_slope": 1.05,
                "qualified": True,
                "fail_reason": "",
                "research_status": "RESEARCH_OK",
            },
        ]
    )
    summary = _build_research_summary_from_grid(grid_results)
    research_cfg = {
        "enabled": True,
        "d1_anchor_hour": 0,
        "features": {"session_bucket": True},
        "k_bars_grid": [12, 24],
        "exploration": {"enabled": True, "kind": "thresholds_only"},
    }

    _persist_research_outputs(
        model_dir=tmp_path,
        symbol="TEST",
        allow_tf="H1",
        score_tf="M5",
        run_id="RID",
        config_path=None,
        grid_results=grid_results,
        family_results=None,
        summary_payload=summary,
        research_cfg=research_cfg,
        config_hash="abc123",
        prompt_version="v1",
        baseline_thresholds={"n_min": 100, "winrate_min": 0.5, "r_mean_min": 0.02, "p10_min": -0.2},
        baseline_thresholds_source="production",
        mode_suffix="_reas",
        research_mode=True,
        mode="research",
    )

    grid_path = tmp_path / "research_grid_results_TEST_M5_RID_reas.csv"
    assert grid_path.exists()
    assert len(pd.read_csv(grid_path)) > 1

    summary_path = tmp_path / "research_summary_TEST_M5_RID_reas.json"
    assert summary_path.exists()

    prod_dir = tmp_path / "prod"
    _persist_research_outputs(
        model_dir=prod_dir,
        symbol="TEST",
        allow_tf="H1",
        score_tf="M5",
        run_id="RID",
        config_path=None,
        grid_results=grid_results,
        family_results=None,
        summary_payload=summary,
        research_cfg=research_cfg,
        config_hash="abc123",
        prompt_version="v1",
        baseline_thresholds={"n_min": 100, "winrate_min": 0.5, "r_mean_min": 0.02, "p10_min": -0.2},
        baseline_thresholds_source="production",
        mode_suffix="_prod",
        research_mode=False,
        mode="production",
    )

    prod_grid_path = prod_dir / "research_grid_results_TEST_M5_RID_prod.csv"
    assert not prod_grid_path.exists()


def test_effective_mode_falls_back_to_production() -> None:
    research_cfg = {"enabled": False}

    assert _effective_mode("research", research_cfg) == "production"
    assert _effective_mode("production", research_cfg) == "production"


def test_baseline_thresholds_cli_fallback() -> None:
    config_payload = {"event_scorer": {"decision_thresholds": {}}}
    args = type("Args", (), {})()
    args.decision_n_min = 150
    args.decision_winrate_min = 0.55
    args.decision_r_mean_min = 0.01
    args.decision_p10_min = -0.1

    thresholds, source = _resolve_baseline_thresholds(config_payload, args)

    assert source == "cli_fallback"
    assert thresholds["n_min"] == 150
