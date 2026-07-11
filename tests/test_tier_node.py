from pipeline.nodes.tier import (
    AUTO_GOLD_AGREE_MIN,
    AUTO_GOLD_CANTO_FT_CONF_MIN,
    BRONZE_AGREE_MIN,
    SILVER_AGREE_MIN,
    assign_tier,
)


def test_assign_tier_text_verified_wins_gold():
    assert assign_tier(True, 0.0) == "gold"
    assert assign_tier(True, 1.0) == "gold"


def test_assign_tier_text_verified_wins_gold_even_over_auto_gold_criteria():
    """text_verified must win even when agreement/canto_ft_confidence would also
    qualify for auto_gold -- human verification is always the higher tier."""
    assert assign_tier(True, 1.0, 0.99) == "gold"


def test_assign_tier_high_agreement_without_confidence_is_silver():
    """agreement alone (no canto_ft_confidence argument) never reaches auto_gold --
    the confidence gate must be satisfied too."""
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
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, AUTO_GOLD_CANTO_FT_CONF_MIN + 0.01) == "auto_gold"
    assert assign_tier(False, 1.0, 0.99) == "auto_gold"


def test_assign_tier_auto_gold_boundary_agreement_exact_threshold():
    """agreement >= 0.95 is inclusive."""
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, 0.99) == "auto_gold"


def test_assign_tier_auto_gold_boundary_confidence_exclusive():
    """canto_ft_confidence > 0.8 is EXCLUSIVE -- exactly 0.8 must not qualify."""
    assert assign_tier(False, 0.99, AUTO_GOLD_CANTO_FT_CONF_MIN) == "silver"
    assert assign_tier(False, 0.99, 0.80) == "silver"


def test_assign_tier_auto_gold_high_agreement_low_confidence_falls_to_silver():
    assert assign_tier(False, 0.99, 0.5) == "silver"


def test_assign_tier_auto_gold_high_confidence_low_agreement_falls_to_silver_bronze_or_excluded():
    """Confidence alone is never enough -- agreement must also clear the auto_gold bar."""
    assert assign_tier(False, 0.90, 0.99) == "silver"
    assert assign_tier(False, 0.70, 0.99) == "bronze"
    assert assign_tier(False, 0.50, 0.99) == "excluded"


def test_assign_tier_auto_gold_none_confidence_treated_as_failing():
    """canto_ft_confidence=None (e.g. canto_ft has no active row for this id) must never
    qualify for auto_gold, same as any confidence <= 0.8."""
    assert assign_tier(False, 0.99, None) == "silver"
