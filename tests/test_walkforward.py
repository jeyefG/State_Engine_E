import pandas as pd

from state_engine.walkforward import apply_edge_ablation, generate_walkforward_splits


def test_walkforward_splits_do_not_overlap_test_ranges() -> None:
    splits = generate_walkforward_splits(
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-02-15"),
        train_days=10,
        calib_days=5,
        test_days=5,
        step_days=5,
    )
    assert splits
    previous_end = None
    for split in splits:
        assert split.train_end <= split.calib_start
        assert split.calib_end <= split.test_start
        assert split.test_start < split.test_end
        if previous_end is not None:
            assert previous_end <= split.test_start
        previous_end = split.test_end


def test_ablation_constant_produces_single_edge_score() -> None:
    events = pd.DataFrame(
        {
            "family_id": ["A", "A", "B"],
            "state_hat_H1": [0, 1, 2],
            "margin_H1": [0.1, 0.2, 0.3],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="5min"),
    )
    scores = pd.Series([0.1, 0.8, 0.4], index=events.index, name="edge_score")
    ablated = apply_edge_ablation(events, scores, ablation="constant")
    assert ablated.nunique() == 1
    assert float(ablated.iloc[0]) == 0.5
