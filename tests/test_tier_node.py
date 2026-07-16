from pipeline.nodes.tier import (
    AUTO_GOLD_AGREE_MIN,
    AUTO_GOLD_DNSMOS_MIN,
    BRONZE_AGREE_MIN,
    SILVER_AGREE_MIN,
    assign_tier,
)


def test_assign_tier_text_verified_wins_gold():
    assert assign_tier(True, 0.0) == "gold"
    assert assign_tier(True, 1.0) == "gold"


def test_assign_tier_text_verified_wins_gold_even_over_auto_gold_criteria():
    """text_verified must win even when agreement/dnsmos would also
    qualify for auto_gold -- human verification is always the higher tier."""
    assert assign_tier(True, 1.0, 4.5) == "gold"


def test_assign_tier_high_agreement_without_dnsmos_is_silver():
    """agreement alone (no dnsmos argument) never reaches auto_gold --
    the acoustic-quality gate must be satisfied too."""
    assert assign_tier(False, SILVER_AGREE_MIN) == "silver"
    assert assign_tier(False, 0.90) == "silver"


def test_assign_tier_low_agreement_is_bronze():
    assert assign_tier(False, BRONZE_AGREE_MIN) == "bronze"
    assert assign_tier(False, SILVER_AGREE_MIN - 0.01) == "bronze"


def test_assign_tier_very_low_agreement_is_excluded():
    assert assign_tier(False, BRONZE_AGREE_MIN - 0.01) == "excluded"
    assert assign_tier(False, 0.0) == "excluded"


def test_assign_tier_silver_boundary_exact_threshold():
    assert assign_tier(False, 0.85) == "silver"


def test_assign_tier_bronze_boundary_exact_threshold():
    assert assign_tier(False, 0.70) == "bronze"


def test_assign_tier_auto_gold_both_conditions_met():
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, AUTO_GOLD_DNSMOS_MIN + 0.1) == "auto_gold"
    assert assign_tier(False, 1.0, 5.0) == "auto_gold"


def test_assign_tier_auto_gold_boundary_agreement_exact_threshold():
    """agreement >= 0.92 is inclusive."""
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, 5.0) == "auto_gold"


def test_assign_tier_auto_gold_boundary_dnsmos_exact_threshold_is_inclusive():
    """dnsmos >= 3.5 is INCLUSIVE -- exactly 3.5 must qualify."""
    assert assign_tier(False, 0.99, AUTO_GOLD_DNSMOS_MIN) == "auto_gold"
    assert assign_tier(False, 0.99, AUTO_GOLD_DNSMOS_MIN - 0.01) == "silver"


def test_assign_tier_auto_gold_high_agreement_low_dnsmos_falls_to_silver():
    assert assign_tier(False, 0.99, 2.0) == "silver"


def test_assign_tier_auto_gold_high_dnsmos_low_agreement_falls_to_silver_bronze_or_excluded():
    """dnsmos alone is never enough -- agreement must also clear the auto_gold bar."""
    assert assign_tier(False, 0.90, 5.0) == "silver"
    assert assign_tier(False, 0.70, 5.0) == "bronze"
    assert assign_tier(False, 0.50, 5.0) == "excluded"


def test_assign_tier_auto_gold_none_dnsmos_treated_as_failing():
    """dnsmos=None (e.g. filters row not yet written for this id) must never
    qualify for auto_gold, same as any dnsmos < 3.5."""
    assert assign_tier(False, 0.99, None) == "silver"
