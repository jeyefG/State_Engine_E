import logging

import numpy as np
import pandas as pd

from scripts.train_state_engine import _build_ev_extended_table, _split_time_terciles


def test_ev_extended_allow_columns_and_buckets() -> None:
    index = pd.date_range("2023-01-01", periods=4, freq="H")
    ev_frame = pd.DataFrame(
        {"ret_struct": [0.1, -0.2, 0.3, 0.4], "state_hat": [0, 0, 1, 2]},
        index=index,
    )
    outputs = pd.DataFrame({"quality_label": ["Q1", "Q2", "Q3", "Q4"]}, index=index)
    gating = pd.DataFrame(
        {"LOOK_FOR_ALPHA": [True, True, True, True], "BLOCK": [True, False, True, False]},
        index=index,
    )
    ctx_features = pd.DataFrame(
        {
            "ctx_state_age": [1, 3, 6, 11],
            "ctx_dist_vwap_atr": [0.3, 0.7, 1.5, 2.5],
        },
        index=index,
    )
    logger = logging.getLogger("test")
    warnings_state: dict[str, bool] = {}
    ev_extended, _, _ = _build_ev_extended_table(
        ev_frame=ev_frame,
        outputs=outputs,
        gating=gating,
        ctx_features=ctx_features,
        min_hvc_samples=1,
        logger=logger,
        warnings_state=warnings_state,
    )

    assert set(ev_extended["allow_rule"]) == {"LOOK_FOR_ALPHA"}
    assert set(ev_extended["ctx_state_age_bucket"]) == {"0-2", "3-5", "6-10", "11+"}
    assert set(ev_extended["ctx_dist_vwap_atr_bucket"]) == {"<=0.5", "0.5-1", "1-2", ">2"}


def test_split_time_terciles_three_segments() -> None:
    index = pd.date_range("2023-01-01", periods=9, freq="H")
    masks = _split_time_terciles(index)
    assert len(masks) == 3
    assert sum(int(np.any(mask)) for mask in masks) == 3
