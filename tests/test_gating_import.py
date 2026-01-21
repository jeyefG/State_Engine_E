def test_gating_import() -> None:
    from state_engine import gating

    assert hasattr(gating, "GatingPolicy")
