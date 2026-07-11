from pipeline.nodes.asr import (
    ASR_MODELS,
    EXCLUDED_FROM_AGREEMENT,
    WORKER_CLASSES,
    Qwen3ASRWorker,
    SenseVoiceWorker,
    TranscribeWorker,
    is_model_enabled,
    model_field,
    resolve_model_key,
    char_agreement,
    compute_agreement_row,
    discover_agreement,
    run_asr_agreement,
    shard_rows_round_robin,
    _load_and_resample,
)

CANTO_FT = model_field("canto_ft")
WHISPER_V3 = model_field("whisper_v3")
QWEN3_ASR = model_field("qwen3_asr")
SENSE_VOICE = model_field("sense_voice")

import asyncio

import duckdb
import numpy as np
import soundfile as sf
import pytest

from pipeline.catalog.catalog import init_schema, upsert_rows


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_model_field_never_yue_for_faster_whisper():
    """Safety: no faster-whisper-backed model may use lang='yue' (causes Whisper
    large-v3 decoder collapse — CLAUDE.md hard constraint #7 / KNOWN_ISSUES.md §9).

    This is scoped to backend == "faster_whisper" deliberately: non-Whisper
    backends (qwen_asr's "Cantonese", sense_voice's "yue") are architecturally
    unaffected by the Whisper decoder-collapse bug and use their own native
    Cantonese/Yue language codes as documented in each ASR_MODELS entry.
    """
    for model_key, cfg in ASR_MODELS.items():
        if cfg["backend"] != "faster_whisper":
            continue
        assert cfg['lang'] != 'yue', (
            f"Model '{model_key}' uses lang='yue', which triggers a known decoder collapse bug."
        )


def test_non_whisper_backends_may_use_native_dialect_codes():
    """sense_voice's 'yue' is a deliberate, safe choice (funasr, not ctranslate2/
    Whisper) — this guards the scoping in test_model_field_never_yue_for_faster_whisper
    above against being silently widened back to a blanket ban."""
    non_whisper = {k: cfg for k, cfg in ASR_MODELS.items() if cfg["backend"] != "faster_whisper"}
    assert non_whisper, "expected at least one non-faster_whisper backend to be registered"
    assert ASR_MODELS["sense_voice"]["backend"] == "sense_voice"
    assert ASR_MODELS["sense_voice"]["lang"] == "yue"


def test_model_field_format():
    """model_field returns id+'+'+lang for every registered model."""
    for model_key in ASR_MODELS:
        expected = ASR_MODELS[model_key]['id'] + '+' + ASR_MODELS[model_key]['lang']
        assert model_field(model_key) == expected


def test_char_agreement_identical_texts():
    """Two identical strings yield agreement == 1.0."""
    assert char_agreement(['hello world', 'hello world']) == 1.0


def test_char_agreement_different_texts():
    """Two partially-overlapping strings yield agreement strictly between 0.0 and 1.0."""
    result = char_agreement(['hello world', 'hello there'])
    assert 0.0 < result < 1.0


def test_char_agreement_single_text():
    """A single-element list yields 1.0 (no pairs to compare)."""
    assert char_agreement(['only one']) == 1.0


def test_char_agreement_empty_list():
    """An empty list yields 1.0 (fewer than 2 texts)."""
    assert char_agreement([]) == 1.0


def test_resolve_model_key_recognises_active_models():
    assert resolve_model_key(CANTO_FT) == "canto_ft"
    assert resolve_model_key(QWEN3_ASR) == "qwen3_asr"
    assert resolve_model_key(SENSE_VOICE) == "sense_voice"
    assert resolve_model_key(WHISPER_V3) == "whisper_v3"  # resolved, just excluded downstream


def test_resolve_model_key_recognises_legacy_canto_ft_paths():
    assert resolve_model_key(
        "/mnt/Drive3/Development/AI-ML/canto-corpus/data/ct2_models/whisper-large-v2-cantonese+zh"
    ) == "canto_ft"
    assert resolve_model_key(
        "/home/typangaa/Documents/canto-corpus/data/ct2_models/whisper-large-v2-cantonese+zh"
    ) == "canto_ft"


def test_resolve_model_key_unrecognised_returns_none():
    assert resolve_model_key("some/unregistered-model+zh") is None


def test_whisper_v3_disabled_and_excluded_from_agreement():
    assert is_model_enabled("whisper_v3") is False
    assert is_model_enabled("canto_ft") is True
    assert "whisper_v3" in EXCLUDED_FROM_AGREEMENT
    assert "canto_ft" not in EXCLUDED_FROM_AGREEMENT


def test_compute_agreement_row_both_present_agree():
    """Identical texts produce agreement==1.0, best_text is that text, text_verified is False."""
    row = compute_agreement_row('id1', [CANTO_FT, QWEN3_ASR], ['abc', 'abc'], [0.9, 0.5], 2)
    assert row['id'] == 'id1'
    assert row['agreement'] == 1.0
    assert row['best_text'] == 'abc'
    assert row['text_verified'] is False
    assert row['model_count'] == 2
    assert row['canto_ft_confidence'] == 0.9


def test_compute_agreement_row_both_empty():
    """Both empty texts produce agreement==0.0 and best_text==''."""
    row = compute_agreement_row('id2', [CANTO_FT, QWEN3_ASR], ['', ''], [0.0, 0.0], 2)
    assert row['id'] == 'id2'
    assert row['agreement'] == 0.0
    assert row['best_text'] == ''
    assert row['text_verified'] is False


def test_compute_agreement_row_one_empty_one_present():
    """One empty and one non-empty text: agreement==0.0, best_text is the non-empty one."""
    row = compute_agreement_row('id3', [CANTO_FT, QWEN3_ASR], ['', 'hello'], [0.0, 0.7], 2)
    assert row['agreement'] == 0.0
    assert row['best_text'] == 'hello'
    assert row['text_verified'] is False

    # Empty candidate has a higher raw confidence number — must still lose.
    row2 = compute_agreement_row('id4', [CANTO_FT, QWEN3_ASR], ['', 'hello'], [0.99, 0.1], 2)
    assert row2['best_text'] == 'hello', (
        "Empty-text candidate must never be selected even when its raw confidence is higher."
    )
    assert row2['text_verified'] is False


def test_compute_agreement_row_picks_higher_confidence():
    """When both texts are non-empty, the one with higher confidence wins."""
    row = compute_agreement_row('id5', [CANTO_FT, QWEN3_ASR], ['foo', 'bar'], [0.9, 0.3], 2)
    assert row['best_text'] == 'foo'
    assert row['text_verified'] is False


def test_compute_agreement_row_three_way():
    """A 3rd model's candidate is folded into agreement/best_text like the other two."""
    row = compute_agreement_row(
        'id6', [CANTO_FT, QWEN3_ASR, SENSE_VOICE], ['abc', 'abd', 'abc'], [0.5, 0.4, 0.95], 3
    )
    assert row['id'] == 'id6'
    assert 0.0 < row['agreement'] < 1.0
    assert row['best_text'] == 'abc'  # highest confidence (0.95) among non-empty texts
    assert row['model_count'] == 3


def test_compute_agreement_row_three_way_one_empty():
    """With 3 candidates and one empty, agreement is computed over the 2 non-empty texts only."""
    row = compute_agreement_row(
        'id7', [CANTO_FT, QWEN3_ASR, SENSE_VOICE], ['hello', '', 'hello'], [0.6, 0.9, 0.5], 3
    )
    assert row['agreement'] == 1.0  # the two non-empty texts are identical
    assert row['best_text'] == 'hello'
    assert row['model_count'] == 3


def test_compute_agreement_row_excludes_whisper_v3_from_agreement_and_best_text():
    """whisper_v3 is resolved (not silently dropped as unrecognised) but must never
    contribute to the agreement ratio or win best_text, even with the highest confidence."""
    row = compute_agreement_row(
        'id8', [CANTO_FT, WHISPER_V3, QWEN3_ASR],
        ['agree text', 'totally different', 'agree text'],
        [0.5, 0.99, 0.6], 3,
    )
    assert row['best_text'] == 'agree text', "whisper_v3's higher confidence must not win best_text"
    assert row['agreement'] == 1.0, "agreement must be computed over canto_ft/qwen3_asr only"


def test_compute_agreement_row_whisper_v3_only_two_others_empty():
    """If only whisper_v3 has non-empty text, best_text is '' (excluded model never wins by default)."""
    row = compute_agreement_row(
        'id9', [CANTO_FT, WHISPER_V3, QWEN3_ASR], ['', 'only whisper_v3 text', ''], [0.0, 0.9, 0.0], 3
    )
    assert row['best_text'] == ''
    assert row['agreement'] == 0.0


def test_compute_agreement_row_dedupes_canto_ft_legacy_path():
    """A stale-path canto_ft duplicate row must not double-count canto_ft's opinion --
    only the current-path row is used, regardless of list order."""
    legacy_path = "/mnt/Drive3/Development/AI-ML/canto-corpus/data/ct2_models/whisper-large-v2-cantonese+zh"
    row = compute_agreement_row(
        'id10',
        [legacy_path, CANTO_FT, QWEN3_ASR],
        ['stale legacy text', 'current text', 'current text'],
        [0.99, 0.5, 0.5],
        3,
    )
    assert row['agreement'] == 1.0, "stale-path canto_ft text must be dropped, not compared"
    assert row['best_text'] == 'current text'
    assert row['canto_ft_confidence'] == 0.5, "canto_ft_confidence must come from the current-path row"


def test_compute_agreement_row_unresolvable_model_dropped():
    """An unrecognised model string contributes nothing (fail-closed), not an error."""
    row = compute_agreement_row(
        'id11', [CANTO_FT, 'some/unregistered-model+zh', QWEN3_ASR],
        ['abc', 'abc', 'abc'], [0.9, 0.9, 0.9], 3,
    )
    assert row['agreement'] == 1.0
    assert row['best_text'] == 'abc'


def test_compute_agreement_row_canto_ft_confidence_none_when_absent():
    """canto_ft_confidence is None when canto_ft has no active row for this id."""
    row = compute_agreement_row('id12', [QWEN3_ASR, SENSE_VOICE], ['abc', 'abc'], [0.9, 0.9], 2)
    assert row['canto_ft_confidence'] is None


def test_load_and_resample_missing_file_returns_none():
    """A non-existent path returns None without raising any exception."""
    result = _load_and_resample('/nonexistent/path/does_not_exist.wav')
    assert result is None


def test_load_and_resample_produces_16k_from_48k(tmp_path):
    """A 48 kHz mono wav is downsampled to ~16 000 samples per second."""
    duration_s = 1.0
    sr_in = 48000
    t = np.linspace(0, duration_s, int(sr_in * duration_s), endpoint=False)
    wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    wav_path = str(tmp_path / 'sine_48k.wav')
    sf.write(wav_path, wave, sr_in)

    result = _load_and_resample(wav_path)
    assert result is not None

    expected_len = 16000
    assert abs(len(result) - expected_len) <= 5, (
        f"Expected ~{expected_len} samples, got {len(result)}"
    )
    # Sanity-check: result is much closer to len(wave)//3 than to len(wave).
    assert abs(len(result) - len(wave) // 3) < abs(len(result) - len(wave)), (
        "Output length should be ~1/3 of input length (3x downsample)."
    )


def test_load_and_resample_stereo_mixed_to_mono(tmp_path):
    """A stereo 48 kHz wav is mixed to mono before resampling; result is 1-D."""
    sr_in = 48000
    n_samples = 16000
    stereo = np.random.default_rng(0).standard_normal((n_samples, 2)).astype(np.float32)

    wav_path = str(tmp_path / 'stereo_48k.wav')
    sf.write(wav_path, stereo, sr_in)

    result = _load_and_resample(wav_path)
    assert result is not None
    assert result.ndim == 1, (
        f"Expected a 1-D mono array, got shape {result.shape}"
    )


# ---------------------------------------------------------------------------
# qwen3_asr registration + worker-class dispatch
# ---------------------------------------------------------------------------

def test_qwen3_asr_registered_with_qwen_backend():
    """qwen3_asr is present, uses the qwen_asr backend, and never lang='yue'."""
    assert 'qwen3_asr' in ASR_MODELS
    cfg = ASR_MODELS['qwen3_asr']
    assert cfg['backend'] == 'qwen_asr'
    assert cfg['lang'] != 'yue'


def test_model_field_qwen3_asr_does_not_collide_with_whisper_models():
    """The 3 registered models must each produce a distinct asr_results.model string
    (asr_results' primary key is (id, model))."""
    fields = [model_field(k) for k in ASR_MODELS]
    assert len(fields) == len(set(fields)), f"model_field() collision among: {fields}"


def test_worker_classes_dispatch_matches_backend_keys():
    """Every backend value used in ASR_MODELS has a corresponding entry in WORKER_CLASSES,
    and the dispatch picks the right class per model."""
    for model_key, cfg in ASR_MODELS.items():
        assert cfg['backend'] in WORKER_CLASSES, (
            f"No worker class registered for backend '{cfg['backend']}' (model '{model_key}')"
        )
    assert WORKER_CLASSES['faster_whisper'] is TranscribeWorker
    assert WORKER_CLASSES['qwen_asr'] is Qwen3ASRWorker
    assert WORKER_CLASSES['sense_voice'] is SenseVoiceWorker


def test_sense_voice_registered_with_sense_voice_backend():
    """sense_voice is present, uses the sense_voice backend, and its lang='yue'
    is a deliberate, safe choice (funasr, not Whisper — see
    test_non_whisper_backends_may_use_native_dialect_codes)."""
    assert 'sense_voice' in ASR_MODELS
    cfg = ASR_MODELS['sense_voice']
    assert cfg['backend'] == 'sense_voice'
    assert cfg['lang'] == 'yue'


# ---------------------------------------------------------------------------
# Qwen3ASRWorker.forward_batch — confidence defaulting (no logprob signal exposed
# by the qwen-asr package, per the class docstring in pipeline/nodes/asr.py).
# Instantiated via object.__new__ to skip GPUWorkerBase.__init__ (which would try
# to actually import qwen_asr and load real model weights) — we only exercise
# forward_batch()'s own logic here, with a fake .model double.
# ---------------------------------------------------------------------------

class _FakeQwenResult:
    def __init__(self, text):
        self.text = text
        self.language = 'yue'


class _FakeQwenModel:
    def __init__(self, texts):
        self._texts = texts

    def transcribe(self, audio, language=None):
        assert len(audio) == len(self._texts)
        return [_FakeQwenResult(t) for t in self._texts]


def _make_bare_qwen_worker(cfg_lang='Cantonese', cc=None):
    worker = object.__new__(Qwen3ASRWorker)
    worker.model_key = 'qwen3_asr'
    worker.cfg = {**ASR_MODELS['qwen3_asr'], 'lang': cfg_lang}
    worker.device = 'cpu'
    worker._cc = cc
    return worker


def test_qwen3asr_worker_defaults_confidence_to_one_for_non_empty_text():
    worker = _make_bare_qwen_worker()
    worker.model = _FakeQwenModel(['你好嗎'])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{'text': '你好嗎', 'confidence': 1.0}]


def test_qwen3asr_worker_empty_text_gets_zero_confidence():
    worker = _make_bare_qwen_worker()
    worker.model = _FakeQwenModel(['   '])  # whitespace-only -> strips to empty
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{'text': '', 'confidence': 0.0}]


def test_qwen3asr_worker_preserves_item_order():
    worker = _make_bare_qwen_worker()
    worker.model = _FakeQwenModel(['first', 'second', ''])
    rows = worker.forward_batch([np.zeros(1600, dtype=np.float32) for _ in range(3)])
    assert [r['text'] for r in rows] == ['first', 'second', '']
    assert [r['confidence'] for r in rows] == [1.0, 1.0, 0.0]


def test_qwen3asr_worker_applies_opencc_conversion_when_cc_present():
    # Qwen3-ASR intermittently emits Simplified Chinese despite language="Cantonese"
    # (measured 2026-07-10: 15.32% of segments corpus-wide) -- same class of issue
    # as SenseVoiceWorker, fixed the same way with an s2hk pass in forward_batch().
    worker = _make_bare_qwen_worker(cc=_FakeOpenCC())
    worker.model = _FakeQwenModel(['我系讲广东话'])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{'text': '我係讲廣東話', 'confidence': 1.0}]


def test_qwen3asr_worker_skips_conversion_when_cc_absent():
    worker = _make_bare_qwen_worker(cc=None)
    worker.model = _FakeQwenModel(['我系讲广东话'])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{'text': '我系讲广东话', 'confidence': 1.0}]


# ---------------------------------------------------------------------------
# SenseVoiceWorker.forward_batch — tag stripping, emotion/event extraction,
# OpenCC s2hk conversion, confidence defaulting, and the funasr-exception
# placeholder path.  Instantiated via object.__new__ (same rationale as
# _make_bare_qwen_worker above) so load_model()'s real funasr.AutoModel /
# OpenCC imports are never hit — only forward_batch()/_parse_raw()'s own
# logic is exercised, against a fake .model double.
# ---------------------------------------------------------------------------

class _FakeSenseVoiceModel:
    """Mimics funasr.AutoModel.generate()'s list-of-dict return shape."""
    def __init__(self, raw_texts, raise_on_generate=False):
        self._raw_texts = raw_texts
        self._raise = raise_on_generate

    def generate(self, input, language=None, use_itn=None, batch_size_s=None):
        if self._raise:
            raise RuntimeError("simulated funasr inference failure")
        assert len(input) == len(self._raw_texts)
        return [{"text": t} for t in self._raw_texts]


class _FakeOpenCC:
    """Mimics opencc.OpenCC('s2hk').convert() with a tiny fixed mapping —
    enough to prove forward_batch actually calls through _cc.convert()."""
    _MAP = {"广东话": "廣東話", "系": "係"}

    def convert(self, text):
        for simp, trad in self._MAP.items():
            text = text.replace(simp, trad)
        return text


def _make_bare_sense_voice_worker(cc=None):
    worker = object.__new__(SenseVoiceWorker)
    worker.model_key = 'sense_voice'
    worker.cfg = dict(ASR_MODELS['sense_voice'])
    worker.device = 'cpu'
    worker._cc = cc
    return worker


def test_sense_voice_worker_strips_tags_and_extracts_emotion_event():
    worker = _make_bare_sense_voice_worker()
    worker.model = _FakeSenseVoiceModel(["<|yue|><|HAPPY|><|Speech|><|woitn|>你好嗎"])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{
        "text": "你好嗎", "confidence": 1.0,
        "metadata": {"emotion": "HAPPY", "audio_event": "Speech"},
    }]


def test_sense_voice_worker_unknown_emotion_event_when_no_tags_present():
    worker = _make_bare_sense_voice_worker()
    worker.model = _FakeSenseVoiceModel(["淨係得普通文字冇任何 tag"])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows[0]["metadata"] == {"emotion": "UNKNOWN", "audio_event": "UNKNOWN"}
    assert rows[0]["text"] == "淨係得普通文字冇任何 tag"


def test_sense_voice_worker_applies_opencc_conversion_when_cc_present():
    worker = _make_bare_sense_voice_worker(cc=_FakeOpenCC())
    worker.model = _FakeSenseVoiceModel(["<|yue|><|NEUTRAL|><|Speech|>我系讲广东话"])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows[0]["text"] == "我係讲廣東話"  # only the two mapped chars convert


def test_sense_voice_worker_keeps_original_text_when_opencc_unavailable():
    worker = _make_bare_sense_voice_worker(cc=None)
    worker.model = _FakeSenseVoiceModel(["<|yue|><|NEUTRAL|><|Speech|>我系讲广东话"])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows[0]["text"] == "我系讲广东话"  # unchanged — no OpenCC instance to convert with


def test_sense_voice_worker_empty_text_gets_zero_confidence():
    worker = _make_bare_sense_voice_worker()
    worker.model = _FakeSenseVoiceModel(["<|yue|><|NEUTRAL|><|nospeech|>   "])
    rows = worker.forward_batch([np.zeros(16000, dtype=np.float32)])
    assert rows == [{
        "text": "", "confidence": 0.0,
        "metadata": {"emotion": "NEUTRAL", "audio_event": "UNKNOWN"},
    }]


def test_sense_voice_worker_preserves_item_order():
    worker = _make_bare_sense_voice_worker()
    worker.model = _FakeSenseVoiceModel(["<|HAPPY|>first", "<|SAD|>second", "<|nospeech|>"])
    rows = worker.forward_batch([np.zeros(1600, dtype=np.float32) for _ in range(3)])
    assert [r["text"] for r in rows] == ["first", "second", ""]
    assert [r["confidence"] for r in rows] == [1.0, 1.0, 0.0]
    assert [r["metadata"]["emotion"] for r in rows] == ["HAPPY", "SAD", "UNKNOWN"]


def test_sense_voice_worker_generate_exception_returns_placeholders_for_every_item():
    """A funasr inference error must not crash the worker — every item in the
    batch gets an empty placeholder so the caller's id/row-count bookkeeping
    stays intact (mirrors label_music.py's skipped_ids handling upstream)."""
    worker = _make_bare_sense_voice_worker()
    worker.model = _FakeSenseVoiceModel([], raise_on_generate=True)
    items = [np.zeros(1600, dtype=np.float32) for _ in range(3)]
    rows = worker.forward_batch(items)
    assert rows == [{"text": "", "confidence": 0.0, "metadata": {}} for _ in items]


# ---------------------------------------------------------------------------
# N-way discover_agreement / run_asr_agreement against a scratch catalog —
# covers the 2-model regression case, the 3-model case, and the legacy-row
# (model_count IS NULL) preservation guarantee.
# ---------------------------------------------------------------------------

def _insert_asr_result(conn, seg_id, model, text, confidence):
    upsert_rows(conn, 'asr_results', [
        {'id': seg_id, 'model': model, 'text': text, 'confidence': confidence}
    ], ['id', 'model'])


def test_discover_agreement_surfaces_2_model_id(scratch_conn):
    _insert_asr_result(scratch_conn, 'seg1', CANTO_FT, 'hello world', 0.9)
    _insert_asr_result(scratch_conn, 'seg1', QWEN3_ASR, 'hello world', 0.8)
    conn = scratch_conn
    conn.execute("ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS model_count INTEGER")
    rows = discover_agreement(conn)
    assert len(rows) == 1
    seg_id, models, texts, confidences, result_count = rows[0]
    assert seg_id == 'seg1'
    assert result_count == 2
    assert sorted(models) == sorted([CANTO_FT, QWEN3_ASR])
    assert sorted(texts) == ['hello world', 'hello world']


def test_discover_agreement_does_not_surface_single_model_id(scratch_conn):
    _insert_asr_result(scratch_conn, 'seg2', CANTO_FT, 'only one model', 0.9)
    conn = scratch_conn
    conn.execute("ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS model_count INTEGER")
    rows = discover_agreement(conn)
    assert rows == [], "An id with only 1 asr_results row must not surface yet."


def test_run_asr_agreement_end_to_end_2_then_3_models(scratch_conn):
    """A 3rd model arriving later re-triggers agreement recompute for the same id
    (model_count-based re-discovery), while a 2-model-only id is computed once and
    does not spuriously re-surface on the next run."""
    conn = scratch_conn
    _insert_asr_result(conn, 'seg_a', CANTO_FT, 'foo bar', 0.9)
    _insert_asr_result(conn, 'seg_a', QWEN3_ASR, 'foo bar', 0.8)
    _insert_asr_result(conn, 'seg_b', CANTO_FT, 'baz qux', 0.9)
    _insert_asr_result(conn, 'seg_b', QWEN3_ASR, 'baz qux', 0.8)

    result1 = asyncio.run(run_asr_agreement(conn=conn))
    assert result1['processed'] == 2

    # Re-running immediately with no new data must find nothing left to do.
    result_noop = asyncio.run(run_asr_agreement(conn=conn))
    assert result_noop == {'processed': 0, 'errors': 0}

    # A 3rd model arrives for seg_a only.
    _insert_asr_result(conn, 'seg_a', SENSE_VOICE, 'foo bar', 0.95)
    result2 = asyncio.run(run_asr_agreement(conn=conn))
    assert result2['processed'] == 1, "Only seg_a's straggler should re-trigger, not seg_b."

    row = conn.execute(
        "SELECT model_count, agreement, best_text FROM asr_agreement WHERE id = 'seg_a'"
    ).fetchone()
    assert row[0] == 3
    assert row[2] == 'foo bar'


def test_run_asr_agreement_preserves_legacy_2_model_row(scratch_conn):
    """A legacy P0-imported row (model_count IS NULL, predates this column) must never
    be resurfaced/recomputed even if its underlying asr_results rows still total exactly 2."""
    conn = scratch_conn
    _insert_asr_result(conn, 'legacy1', CANTO_FT, 'legacy text', 0.5)
    _insert_asr_result(conn, 'legacy1', QWEN3_ASR, 'legacy text', 0.5)
    # Simulate the P0 legacy import: an asr_agreement row already exists, with no
    # model_count column populated (NULL).
    conn.execute("ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS model_count INTEGER")
    upsert_rows(conn, 'asr_agreement', [
        {'id': 'legacy1', 'agreement': 0.42, 'best_text': 'legacy text', 'text_verified': True}
    ], ['id'])

    rows = discover_agreement(conn)
    assert rows == [], "A legacy row with model_count IS NULL must not be re-surfaced."

    row = conn.execute(
        "SELECT agreement, text_verified FROM asr_agreement WHERE id = 'legacy1'"
    ).fetchone()
    assert row == (0.42, True), "Legacy row must remain byte-for-byte untouched."


# ---------------------------------------------------------------------------
# shard_rows_round_robin — splitting one model's backlog across multiple
# devices (e.g. qwen3_asr on both cuda:0 and cuda:1).
# ---------------------------------------------------------------------------

def test_shard_rows_round_robin_single_device_returns_all_rows_unchanged():
    """The common case (one device per model_key) must behave exactly like before
    this feature existed: all rows go to the single device, same order."""
    rows = [('a',), ('b',), ('c',)]
    shards = shard_rows_round_robin(rows, ['cuda:0'])
    assert shards == {'cuda:0': [('a',), ('b',), ('c',)]}


def test_shard_rows_round_robin_two_devices_interleaves():
    """Row i goes to devices[i % 2] — an alternating split, not a contiguous half,
    so both devices get an even mix of the duration-ascending-sorted queue."""
    rows = [('id0',), ('id1',), ('id2',), ('id3',), ('id4',)]
    shards = shard_rows_round_robin(rows, ['cuda:0', 'cuda:1'])
    assert shards['cuda:0'] == [('id0',), ('id2',), ('id4',)]
    assert shards['cuda:1'] == [('id1',), ('id3',)]


def test_shard_rows_round_robin_covers_every_row_exactly_once():
    rows = [(f'id{i}',) for i in range(23)]
    shards = shard_rows_round_robin(rows, ['cuda:0', 'cuda:1'])
    combined = shards['cuda:0'] + shards['cuda:1']
    assert sorted(combined) == sorted(rows)
    assert len(combined) == len(rows)


def test_shard_rows_round_robin_empty_rows():
    shards = shard_rows_round_robin([], ['cuda:0', 'cuda:1'])
    assert shards == {'cuda:0': [], 'cuda:1': []}
