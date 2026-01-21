import pandas as pd

from scripts.train_event_scorer import _empty_research_variant_report
from state_engine.research import ResearchVariant, evaluate_research_variants, generate_research_variants


def test_variant_report_columns_and_counts() -> None:
    base_thresholds = {
        "n_min": 50,
        "winrate_min": 0.5,
        "r_mean_min": 0.01,
        "p10_min": -0.2,
    }
    thresholds_grid = {
        "n_min": [50, 100],
        "winrate_min": [0.5],
        "r_mean_min": [0.01],
        "p10_min": [-0.2],
    }
    variants = generate_research_variants(
        kind="thresholds_only",
        base_k_bars=24,
        k_bars_grid=None,
        base_thresholds=base_thresholds,
        thresholds_grid=thresholds_grid,
        seed=7,
        max_variants=None,
    )

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    labels = pd.Series([1, 0, 1, 0, 1], index=idx)
    r_outcome = pd.Series([0.2, -0.1, 0.3, -0.2, 0.1], index=idx)
    families = pd.Series(["A", "A", "B", "B", "B"], index=idx)

    report = evaluate_research_variants(
        variants=variants,
        scores=scores,
        labels=labels,
        r_outcome=r_outcome,
        families=families,
        timestamps=idx,
        train_count=3,
        calib_count=len(idx),
        allow_rate=None,
        diagnostics_cfg={"min_temporal_dispersion": 0.3, "max_family_concentration": 0.6, "min_calib_samples": 2},
        baseline_thresholds=base_thresholds,
    )

    required_columns = {
        "variant_id",
        "k_bars",
        "n_min",
        "r_mean_min",
        "p10_min",
        "winrate_min",
    }
    assert required_columns.issubset(report.columns)
    assert len(report) == len(variants)
    assert report["temporal_dispersion"].between(0, 1).all()


def test_empty_research_report_for_no_predictions() -> None:
    variants = [
        ResearchVariant(
            variant_id="thr_n80_rm0.02_p10-0.2_wr0.5_k24",
            variant_type="thresholds_only",
            k_bars=24,
            n_min=80,
            winrate_min=0.5,
            r_mean_min=0.02,
            p10_min=-0.2,
        ),
        ResearchVariant(
            variant_id="kbar_n200_rm0.02_p10-0.2_wr0.52_k36",
            variant_type="kbars_only",
            k_bars=36,
            n_min=200,
            winrate_min=0.52,
            r_mean_min=0.02,
            p10_min=-0.2,
        ),
    ]

    report = _empty_research_variant_report(variants, "NO_PREDICTIONS")

    assert len(report) == len(variants)
    assert (report["fail_reason"] == "NO_PREDICTIONS").all()
    assert (~report["qualified"]).all()
