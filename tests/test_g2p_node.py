from pipeline.nodes.g2p import (
    MIN_VALID_FRACTION,
    _is_valid_token,
    candidate_preview,
    g2p_one,
    text_to_jyutping,
    text_to_jyutping_codeswitch,
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


def test_candidate_preview_empty_text_returns_empty_list():
    assert candidate_preview("") == []
    assert candidate_preview(None) == []


def test_candidate_preview_unambiguous_text_returns_empty_list():
    # Every character has a single certain reading -- nothing for a reviewer
    # to look at. (Not "心臟病中風": canto-hk-g2p v2.3.0's segmentation-shadow
    # pruning (DECISIONS.md 2026-07-22) removed the purely-compositional
    # multi-char dict entry that used to cover 心臟病, so it now resolves
    # per-character and 心 (sam1/san1) surfaces as a genuine polyphone.)
    assert candidate_preview("早晨") == []


def test_candidate_preview_segmentation_shadow_fix_v2_3_0():
    # canto-hk-g2p v2.3.0 (DECISIONS.md 2026-07-22) pruned purely-compositional
    # dict entries like 我瞓/早瞓/未瞓 that used to greedily shadow the real
    # compound 瞓覺 ("to sleep"), orphaning 覺 into ambiguous ranked fallback
    # (gok3/gaau3/gaau1/gaau4). Regression guard: 瞓覺 must now resolve as a
    # single certain compound, not a polyphone-ambiguous orphan.
    assert text_to_jyutping("我瞓覺先") == "ngo5 fan3 gaau3 sin1"
    assert "覺" not in {entry["token"] for entry in candidate_preview("我瞓覺先")}


def test_candidate_preview_flags_known_polyphone():
    # Bare 重 (out of a disambiguating multi-char context) is a known
    # polyphone: zung6 "heavy" / cung4 "repeat" / cung5 / cung6 readings.
    result = candidate_preview("重, hello")
    assert len(result) == 1
    entry = result[0]
    assert entry["token"] == "重"
    assert entry.keys() == {"token", "candidates", "confidence", "source"}
    assert len(entry["candidates"]) >= 2
    assert entry["confidence"] in ("ranked", "tied")


def test_candidate_preview_excludes_english_and_punctuation():
    result = candidate_preview("重, hello")
    tokens = [entry["token"] for entry in result]
    assert "hello" not in tokens
    assert "," not in tokens and "，" not in tokens


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
# _is_valid_token() -- regex shape is necessary but not sufficient; a token
# must also resolve via canto_hk_g2p.segment() (v1.6.0+) to a real Jyutping
# onset/rime/tone combination. "zzz1" matches the regex but isn't a real
# syllable -- this is the gap the phonological check closes.
# ---------------------------------------------------------------------------

def test_is_valid_token_real_syllable_accepted():
    assert _is_valid_token("nei5") is True


def test_is_valid_token_syllabic_nasal_accepted():
    assert _is_valid_token("m4") is True
    assert _is_valid_token("ng4") is True


def test_is_valid_token_regex_shaped_but_phonologically_invalid_rejected():
    """Passes JYUTPING_TOKEN's `^[a-z]+[1-6]$` regex but isn't a real syllable."""
    assert _is_valid_token("zzz1") is False


def test_is_valid_token_wrong_shape_rejected():
    assert _is_valid_token("[BAD]") is False
    assert _is_valid_token("XX") is False


def test_validate_jyutping_rejects_regex_shaped_garbage():
    """A token like "zzz1" used to pass the old regex-only check -- it must
    not pass now that _is_valid_token() also requires segment() to resolve it."""
    accept, frac, bad = validate_jyutping("nei5 zzz1")
    assert frac == 0.5
    assert bad == ["zzz1"]


# ---------------------------------------------------------------------------
# g2p_one() — pure text -> {jyutping, valid_fraction, jyutping_cs} dict, always
# returns a row (even on empty/failed input) so discovery never loops forever
# on a bad segment.
# ---------------------------------------------------------------------------

def test_g2p_one_empty_text_returns_zero_fraction_row():
    row = g2p_one("")
    assert row == {"jyutping": "", "valid_fraction": 0.0, "jyutping_cs": ""}


def test_g2p_one_none_text_returns_zero_fraction_row():
    row = g2p_one(None)
    assert row == {"jyutping": "", "valid_fraction": 0.0, "jyutping_cs": ""}


def test_g2p_one_whitespace_only_returns_zero_fraction_row():
    row = g2p_one("   ")
    assert row == {"jyutping": "", "valid_fraction": 0.0, "jyutping_cs": ""}


def test_g2p_one_pure_english_returns_zero_fraction_row():
    """No Cantonese tokens -> text_to_jyutping() returns None -> zero-fraction
    jyutping/valid_fraction, but jyutping_cs is computed independently and
    keeps the English text verbatim (it's not part of the accept/reject gate)."""
    row = g2p_one("hello world")
    assert row["jyutping"] == ""
    assert row["valid_fraction"] == 0.0
    assert row["jyutping_cs"] == "hello world"


def test_g2p_one_valid_cantonese_text():
    row = g2p_one("你好嘅")
    assert row["jyutping"] == "nei5 hou2 ge3"
    assert row["valid_fraction"] == 1.0
    assert row["jyutping_cs"] == "nei5 hou2 ge3"


def test_g2p_one_always_returns_all_keys():
    for text in ["", None, "   ", "hello", "你好"]:
        row = g2p_one(text)
        assert set(row.keys()) == {"jyutping", "valid_fraction", "jyutping_cs"}


# ---------------------------------------------------------------------------
# text_to_jyutping_codeswitch() — like text_to_jyutping() but keeps English
# words and punctuation inline instead of dropping them (T30: added so
# downstream code-switch-aware consumers, e.g. canto-tts, no longer need to
# re-run canto-hk-g2p themselves against the raw text).
# ---------------------------------------------------------------------------

def test_text_to_jyutping_codeswitch_keeps_english_and_punctuation_inline():
    result = text_to_jyutping_codeswitch("今日天氣幾好，多謝晒 David。")
    assert "David" in result.split()
    assert "，" in result
    assert "。" in result
    assert result.startswith("gam1 jat6")


def test_text_to_jyutping_codeswitch_empty_text_returns_empty_string():
    assert text_to_jyutping_codeswitch("") == ""
    assert text_to_jyutping_codeswitch(None) == ""


def test_text_to_jyutping_codeswitch_pure_english_returns_verbatim():
    assert text_to_jyutping_codeswitch("hello world") == "hello world"
