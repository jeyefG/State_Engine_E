import pandas as pd

from state_engine.features import FeatureEngineer


def test_swing_counts_no_future_dependency() -> None:
    high = pd.Series([1, 2, 3, 2, 1, 2, 3, 4], dtype=float)
    low = pd.Series([0, 1, 2, 1, 0, 1, 2, 3], dtype=float)

    counts_high, counts_low = FeatureEngineer._swing_counts(high, low, window=3)

    high_modified = high.copy()
    low_modified = low.copy()
    high_modified.iloc[-1] = 100.0
    low_modified.iloc[-1] = -100.0

    counts_high_modified, counts_low_modified = FeatureEngineer._swing_counts(
        high_modified,
        low_modified,
        window=3,
    )

    pd.testing.assert_series_equal(counts_high.iloc[:-1], counts_high_modified.iloc[:-1])
    pd.testing.assert_series_equal(counts_low.iloc[:-1], counts_low_modified.iloc[:-1])
