import duckdb
import numpy as np
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.filter import (
    MANDARIN_AUDIO_PROB_MIN,
    MAX_DUR,
    MAX_ENG_RATIO,
    MAX_MAN_RATIO,
    MAX_TEXT_CHARS,
    MIN_CJK_CHARS,
    MIN_DUR,
    TARGET_SR,
    cjk_count,
    compute_snr,
    decide_row,
    detect_language,
    discover_decide,
    discover_text,
    english_ratio,
    evaluate_text,
    is_cjk,
    mandarin_ratio,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_is_cjk_basic():
    assert is_cjk("係")
    assert is_cjk("㗎")  # Extension A
    assert not is_cjk("a")
    assert not is_cjk("1")
    assert not is_cjk(" ")


def test_english_ratio_pure_cantonese():
    assert english_ratio("今日天氣好好呀") == 0.0


def test_english_ratio_pure_english():
    assert english_ratio("hello world") == 1.0


def test_english_ratio_empty_text():
    assert english_ratio("") == 0.0


def test_english_ratio_code_switch_within_bounds():
    # 4 CJK chars + 1 English word = 0.2, under MAX_ENG_RATIO (0.30).
    ratio = english_ratio("我今日call佢")
    assert ratio <= MAX_ENG_RATIO


def test_mandarin_ratio_pure_cantonese_zero():
    assert mandarin_ratio("我哋而家去邊度") == 0.0


def test_mandarin_ratio_simplified_chars_flagged():
    # Simplified-only markers with no Cantonese markers present.
    ratio = mandarin_ratio("这是我们的东西")
    assert ratio > MAX_MAN_RATIO


def test_mandarin_ratio_empty_text():
    assert mandarin_ratio("") == 0.0


def test_cjk_count():
    assert cjk_count("abc係嘅123") == 2
    assert cjk_count("") == 0


def test_detect_language_english_dominant():
    lang, conf = detect_language("hello there my friend how are you")
    assert lang == "eng"


def test_detect_language_cantonese():
    lang, conf = detect_language("我哋而家喺邊度呀係咪呀")
    assert lang == "yue"


def test_compute_snr_bursty_speech_beats_flat_noise():
    """compute_snr() is a frame-energy top-10%-vs-bottom-10% ratio, i.e. it scores
    dynamic range, not textbook SNR — a signal with clear loud/quiet contrast (like
    speech over a quiet floor) should score higher than uniform-amplitude noise with
    no contrast at all. (A pure constant-amplitude tone is a degenerate case for
    this heuristic — every frame has near-identical energy either way — so the test
    uses a bursty envelope instead.)"""
    sr = TARGET_SR
    n = sr * 2
    rng = np.random.default_rng(0)
    quiet_floor = rng.normal(0, 0.001, size=n).astype(np.float32)
    t = np.linspace(0, 2.0, n, endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8
    frame_len = int(sr * 0.025)
    mask = (np.arange(n) // (frame_len * 4)) % 2 == 0
    bursty_speech = np.where(mask, tone, 0.0) + quiet_floor

    flat_noise = rng.normal(0, 0.3, size=n).astype(np.float32)

    assert compute_snr(bursty_speech) > compute_snr(flat_noise)


def test_compute_snr_empty_array():
    assert compute_snr(np.array([], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# evaluate_text() — hard gates + text gates, first-failing-gate-wins ordering
# ---------------------------------------------------------------------------

def test_evaluate_text_sample_rate_gate():
    result = evaluate_text(5.0, 16000, "呢個係一個測試句子")
    assert result["pass"] is False
    assert result["fail_reason"] == "sample_rate"


def test_evaluate_text_duration_gate_too_short():
    result = evaluate_text(MIN_DUR - 0.1, TARGET_SR, "呢個係一個測試句子")
    assert result["pass"] is False
    assert result["fail_reason"] == "duration"


def test_evaluate_text_duration_gate_too_long():
    result = evaluate_text(MAX_DUR + 0.1, TARGET_SR, "呢個係一個測試句子")
    assert result["pass"] is False
    assert result["fail_reason"] == "duration"


def test_evaluate_text_too_short_text():
    short_text = "係" * (MIN_CJK_CHARS - 1)
    result = evaluate_text(5.0, TARGET_SR, short_text)
    assert result["pass"] is False
    assert result["fail_reason"] == "text_too_short"


def test_evaluate_text_too_long_text():
    long_text = "係" * (MAX_TEXT_CHARS + 1)
    result = evaluate_text(5.0, TARGET_SR, long_text)
    assert result["pass"] is False
    assert result["fail_reason"] == "text_too_long"


def test_evaluate_text_english_ratio_gate():
    # 6 CJK chars (clears text_too_short) + 5 English words -> ratio 5/11 > MAX_ENG_RATIO.
    result = evaluate_text(5.0, TARGET_SR, "我哋而家傾緊call whatsapp email hello everyone")
    assert result["pass"] is False
    assert result["fail_reason"] == "english_ratio"


def test_evaluate_text_mandarin_ratio_gate():
    result = evaluate_text(5.0, TARGET_SR, "这是我们的东西这是我们的东西")
    assert result["pass"] is False
    assert result["fail_reason"] == "mandarin_ratio"


def test_evaluate_text_passes_all_gates():
    result = evaluate_text(5.0, TARGET_SR, "我哋而家喺呢度傾緊天氣嘅事情")
    assert result["pass"] is True
    assert result["fail_reason"] is None
    assert result["detected_language"] == "yue"


def test_evaluate_text_gate_order_sample_rate_before_duration():
    """Both sample_rate and duration are bad — sample_rate (checked first) wins."""
    result = evaluate_text(1.0, 16000, "呢個係一個測試句子")
    assert result["fail_reason"] == "sample_rate"


# ---------------------------------------------------------------------------
# decide_row() — merges filters_text + filters_acoustic
# ---------------------------------------------------------------------------

def test_decide_row_text_fail_takes_priority():
    row = decide_row(
        "id1", False, "duration", 0.0, 0.0, "yue", 0.9,
        True, None, 30.0, 3.5, 3.0,
    )
    assert row["pass"] is False
    assert row["fail_reason"] == "duration"


def test_decide_row_acoustic_fail_when_text_passes():
    row = decide_row(
        "id2", True, None, 0.1, 0.05, "yue", 0.9,
        False, "dnsmos", 30.0, 2.0, 1.8,
    )
    assert row["pass"] is False
    assert row["fail_reason"] == "dnsmos"


def test_decide_row_both_pass():
    row = decide_row(
        "id3", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
    )
    assert row["pass"] is True
    assert row["fail_reason"] is None
    assert row["snr_db"] == 30.0
    assert row["dnsmos"] == 3.5


def test_decide_row_acoustic_pending_guard():
    """Defensive branch: text passed but acoustic hasn't run — should never occur
    given discover_decide()'s SQL gating, but must fail closed, not raise."""
    row = decide_row(
        "id4", True, None, 0.1, 0.05, "yue", 0.9,
        None, None, None, None, None,
    )
    assert row["pass"] is False
    assert row["fail_reason"] == "acoustic_pending"


def test_decide_row_stores_text_model_count():
    row = decide_row(
        "id5", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2, text_model_count=3,
    )
    assert row["text_model_count"] == 3


# ---------------------------------------------------------------------------
# T5 (2026-07-17): discover_text/discover_decide re-evaluation on a stale
# asr_agreement.model_count, not just bare row-existence.
# ---------------------------------------------------------------------------

def _seed_agreement(conn, seg_id, *, model_count=2, best_text="呢個係測試"):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, sample_rate, program) "
        "VALUES (?, ?, 'podcast', 6.0, 48000, 'test-program')",
        [seg_id, f"/tmp/{seg_id}.flac"],
    )
    conn.execute(
        "INSERT INTO asr_agreement (id, agreement, best_text, text_verified, model_count) "
        "VALUES (?, 0.9, ?, FALSE, ?)",
        [seg_id, best_text, model_count],
    )


def test_discover_text_picks_up_never_evaluated_segment(scratch_conn):
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=2)
    rows = discover_text(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_text_excludes_already_current_segment(scratch_conn):
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=2)
    conn.execute(
        "INSERT INTO filters_text (id, pass, asr_model_count) VALUES ('a', TRUE, 2)"
    )
    assert discover_text(conn) == []


def test_discover_text_reevaluates_when_model_count_advances(scratch_conn):
    """A later ASR model landing bumps asr_agreement.model_count -- filter.text
    must re-pick up the id even though a filters_text row already exists."""
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=2)
    conn.execute(
        "INSERT INTO filters_text (id, pass, asr_model_count) VALUES ('a', TRUE, 1)"
    )
    rows = discover_text(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_text_legacy_null_model_count_reevaluates(scratch_conn):
    """Legacy P0-imported filters_text rows have asr_model_count IS NULL --
    must still be picked up (same legacy-row-collision fix as elsewhere)."""
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=2)
    conn.execute("INSERT INTO filters_text (id, pass) VALUES ('a', TRUE)")
    rows = discover_text(conn)
    assert [r[0] for r in rows] == ["a"]


def _seed_decide_inputs(conn, seg_id, *, text_model_count=2, ft_pass=True, decided=False, decided_model_count=2):
    conn.execute(
        "INSERT INTO filters_text (id, pass, asr_model_count) VALUES (?, ?, ?)",
        [seg_id, ft_pass, text_model_count],
    )
    conn.execute(
        "INSERT INTO filters_acoustic (id, pass, snr_db, dnsmos_sig, dnsmos_ovrl) "
        "VALUES (?, TRUE, 30.0, 3.5, 3.2)",
        [seg_id],
    )
    if decided:
        conn.execute(
            "INSERT INTO filters (id, pass, provenance, text_model_count) "
            "VALUES (?, ?, 'filter_decide', ?)",
            [seg_id, ft_pass, decided_model_count],
        )


def test_discover_decide_picks_up_never_decided_segment(scratch_conn):
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", decided=False)
    rows = discover_decide(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_decide_excludes_already_current_decision(scratch_conn):
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", text_model_count=2, decided=True, decided_model_count=2)
    assert discover_decide(conn) == []


def test_discover_decide_reevaluates_when_filters_text_advances(scratch_conn):
    """filter.text re-evaluated this id under a newer model (asr_model_count now
    3) after filter.decide already decided it at model_count=2 -- must re-decide."""
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", text_model_count=3, decided=True, decided_model_count=2)
    rows = discover_decide(conn)
    assert [r[0] for r in rows] == ["a"]


# ---------------------------------------------------------------------------
# T20 (2026-07-18): audio-based Mandarin gate (labels_lang.lang == 'cmn'), layered
# on top of the text-heuristic mandarin_ratio() gate -- catches genuine spoken
# Mandarin that gets transcribed into fluent standard written Chinese with no
# simplified chars or mainland-specific words for mandarin_ratio() to key off.
# ---------------------------------------------------------------------------

def test_decide_row_rejects_high_confidence_audio_mandarin():
    row = decide_row(
        "id6", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
        audio_lang="cmn", audio_cmn_prob=0.95, lang_label_present=True,
    )
    assert row["pass"] is False
    assert row["fail_reason"] == "mandarin_audio"
    assert row["mandarin_audio_prob"] == 0.95


def test_decide_row_passes_low_confidence_audio_mandarin():
    """Below MANDARIN_AUDIO_PROB_MIN -- e.g. a brief quoted Mandarin speaker inside
    an otherwise-Cantonese segment -- must not trip the gate."""
    row = decide_row(
        "id7", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
        audio_lang="cmn", audio_cmn_prob=MANDARIN_AUDIO_PROB_MIN - 0.01, lang_label_present=True,
    )
    assert row["pass"] is True
    assert row["fail_reason"] is None


def test_decide_row_passes_non_mandarin_audio_lang():
    row = decide_row(
        "id8", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
        audio_lang="yue", audio_cmn_prob=0.01, lang_label_present=True,
    )
    assert row["pass"] is True
    assert row["fail_reason"] is None


def test_decide_row_passes_when_no_label_present_yet():
    """label.suite hasn't reached this segment yet -- must not fail closed."""
    row = decide_row(
        "id9", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
    )
    assert row["pass"] is True
    assert row["lang_label_checked"] is False


def test_decide_row_stores_lang_label_checked_true():
    row = decide_row(
        "id10", True, None, 0.1, 0.05, "yue", 0.9,
        True, None, 30.0, 3.5, 3.2,
        audio_lang="yue", audio_cmn_prob=0.01, lang_label_present=True,
    )
    assert row["lang_label_checked"] is True


def _seed_decide_inputs_with_label(conn, seg_id, *, has_label=False, label_lang="yue", label_prob=0.01):
    _seed_decide_inputs(conn, seg_id, decided=False)
    if has_label:
        conn.execute(
            "INSERT INTO labels_lang (id, lang, cmn_prob) VALUES (?, ?, ?)",
            [seg_id, label_lang, label_prob],
        )


def test_discover_decide_picks_up_segment_with_no_label_yet(scratch_conn):
    conn = scratch_conn
    _seed_decide_inputs_with_label(conn, "a", has_label=False)
    rows = discover_decide(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_decide_excludes_when_decided_and_no_label_landed(scratch_conn):
    """Already decided, still no labels_lang row for this id -- nothing changed,
    must not re-trigger every run just because the label is still missing."""
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", text_model_count=2, decided=True, decided_model_count=2)
    assert discover_decide(conn) == []


def test_discover_decide_reevaluates_once_label_lands_after_decision(scratch_conn):
    """filter.decide already decided this id without a labels_lang row
    (lang_label_checked left NULL/FALSE); label.suite has since landed one --
    must re-decide so the audio-based Mandarin gate gets a chance to run."""
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", text_model_count=2, decided=True, decided_model_count=2)
    conn.execute("INSERT INTO labels_lang (id, lang, cmn_prob) VALUES ('a', 'cmn', 0.95)")
    rows = discover_decide(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_decide_excludes_once_lang_label_checked_true(scratch_conn):
    """Once filter.decide has already accounted for an existing label
    (lang_label_checked=TRUE), it must not re-trigger again on every run."""
    conn = scratch_conn
    _seed_decide_inputs(conn, "a", text_model_count=2)
    conn.execute("INSERT INTO labels_lang (id, lang, cmn_prob) VALUES ('a', 'yue', 0.01)")
    conn.execute(
        "INSERT INTO filters (id, pass, provenance, text_model_count, lang_label_checked) "
        "VALUES ('a', TRUE, 'filter_decide', 2, TRUE)"
    )
    assert discover_decide(conn) == []
