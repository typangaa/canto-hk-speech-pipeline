from pipeline.nodes.g2p import (
    MIN_VALID_FRACTION,
    g2p_one,
    text_to_jyutping,
    validate_jyutping,
)


def test_text_to_jyutping_basic():
    result = text_to_jyutping("你好嘅")
    assert result is not None
    tokens = result.split()
    assert len(tokens) == 3
    assert all(validate_jyutping(t)[0] for t in [result])


def test_text_to_jyutping_excludes_english():
    """English tokens are dropped from the Jyutping output (lang != 'yue')."""
    result = text_to_jyutping("你好 hello")
    assert result is not None
    assert "hello" not in result.split()


def test_text_to_jyutping_empty_string_returns_none():
    assert text_to_jyutping("") is None


def test_text_to_jyutping_pure_english_returns_none():
    assert text_to_jyutping("hello world") is None


def test_validate_jyutping_all_valid():
    accept, frac, bad = validate_jyutping("nei5 hou2 ge3")
    assert accept is True
    assert frac == 1.0
    assert bad == []


def test_validate_jyutping_some_invalid():
    accept, frac, bad = validate_jyutping("nei5 hou2 [BAD] ge3")
    assert frac == 0.75
    assert bad == ["[BAD]"]
    assert accept is (0.75 >= MIN_VALID_FRACTION)


def test_validate_jyutping_empty_string():
    """An empty Jyutping string (e.g. all-English/all-punctuation input) is
    vacuously accepted with valid_fraction 1.0 — matches scripts/07_g2p.py."""
    accept, frac, bad = validate_jyutping("")
    assert accept is True
    assert frac == 1.0
    assert bad == []


def test_validate_jyutping_below_threshold_rejects():
    accept, frac, bad = validate_jyutping("XX YY ZZ")
    assert accept is False
    assert frac == 0.0


# ---------------------------------------------------------------------------
# g2p_one() — pure text -> {jyutping, valid_fraction} dict, always returns a row
# (even on empty/failed input) so discovery never loops forever on a bad segment.
# ---------------------------------------------------------------------------

def test_g2p_one_empty_text_returns_zero_fraction_row():
    row = g2p_one("")
    assert row == {"jyutping": "", "valid_fraction": 0.0}


def test_g2p_one_none_text_returns_zero_fraction_row():
    row = g2p_one(None)
    assert row == {"jyutping": "", "valid_fraction": 0.0}


def test_g2p_one_whitespace_only_returns_zero_fraction_row():
    row = g2p_one("   ")
    assert row == {"jyutping": "", "valid_fraction": 0.0}


def test_g2p_one_pure_english_returns_zero_fraction_row():
    """No Cantonese tokens -> text_to_jyutping() returns None -> zero-fraction row."""
    row = g2p_one("hello world")
    assert row == {"jyutping": "", "valid_fraction": 0.0}


def test_g2p_one_valid_cantonese_text():
    row = g2p_one("你好嘅")
    assert row["jyutping"] == "nei5 hou2 ge3"
    assert row["valid_fraction"] == 1.0


def test_g2p_one_always_returns_both_keys():
    for text in ["", None, "   ", "hello", "你好"]:
        row = g2p_one(text)
        assert set(row.keys()) == {"jyutping", "valid_fraction"}
