from pipeline.nodes.tier import SILVER_AGREE_MIN, assign_tier


def test_assign_tier_text_verified_wins_gold():
    assert assign_tier(True, 0.0) == "gold"
    assert assign_tier(True, 1.0) == "gold"


def test_assign_tier_high_agreement_is_silver():
    assert assign_tier(False, SILVER_AGREE_MIN) == "silver"
    assert assign_tier(False, 1.0) == "silver"


def test_assign_tier_low_agreement_is_excluded():
    assert assign_tier(False, SILVER_AGREE_MIN - 0.01) == "excluded"
    assert assign_tier(False, 0.0) == "excluded"


def test_assign_tier_boundary_exact_threshold():
    assert assign_tier(False, 0.65) == "silver"
