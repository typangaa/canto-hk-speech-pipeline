from pipeline.nodes.asr import ASR_MODELS, model_field, char_agreement, compute_agreement_row, _load_and_resample

import numpy as np
import soundfile as sf
import pytest


def test_model_field_never_yue():
    """Safety: no model may use lang='yue' (causes Whisper large-v3 decoder collapse)."""
    for model_key in ASR_MODELS:
        assert ASR_MODELS[model_key]['lang'] != 'yue', (
            f"Model '{model_key}' uses lang='yue', which triggers a known decoder collapse bug."
        )


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


def test_compute_agreement_row_both_present_agree():
    """Identical texts produce agreement==1.0, best_text is that text, text_verified is False."""
    row = compute_agreement_row('id1', 'abc', 0.9, 'abc', 0.5)
    assert row['id'] == 'id1'
    assert row['agreement'] == 1.0
    assert row['best_text'] == 'abc'
    assert row['text_verified'] is False


def test_compute_agreement_row_both_empty():
    """Both empty texts produce agreement==0.0 and best_text==''."""
    row = compute_agreement_row('id2', '', 0.0, '', 0.0)
    assert row['id'] == 'id2'
    assert row['agreement'] == 0.0
    assert row['best_text'] == ''
    assert row['text_verified'] is False


def test_compute_agreement_row_one_empty_one_present():
    """One empty and one non-empty text: agreement==0.0, best_text is the non-empty one."""
    row = compute_agreement_row('id3', '', 0.0, 'hello', 0.7)
    assert row['agreement'] == 0.0
    assert row['best_text'] == 'hello'
    assert row['text_verified'] is False

    # Empty candidate has a higher raw confidence number — must still lose.
    row2 = compute_agreement_row('id4', '', 0.99, 'hello', 0.1)
    assert row2['best_text'] == 'hello', (
        "Empty-text candidate must never be selected even when its raw confidence is higher."
    )
    assert row2['text_verified'] is False


def test_compute_agreement_row_picks_higher_confidence():
    """When both texts are non-empty, the one with higher confidence wins."""
    row = compute_agreement_row('id5', 'foo', 0.9, 'bar', 0.3)
    assert row['best_text'] == 'foo'
    assert row['text_verified'] is False


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
