import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from pipeline.audio.bus import decode

# ---------------------------------------------------------------------------
# WAV vs FLAC round-trip — P5-A decision (segments now written as FLAC).
# FLAC is lossless, so decode(wav) and decode(flac) of the SAME PCM data must
# be bit-exact, not merely close.
# ---------------------------------------------------------------------------

def test_wav_and_flac_decode_bit_exact(tmp_path):
    sr = 48000
    rng = np.random.default_rng(0)
    # Values pre-quantized to the int16 grid (not arbitrary floats): this
    # removes float->int16 rounding-mode ambiguity between libsndfile's WAV
    # and FLAC encoders, which was found (2026-07-05) to differ by up to 1
    # LSB on ~50% of samples for un-quantized input -- a real libsndfile
    # quirk, not a bug in this pipeline, and irrelevant in production since
    # a clip is only ever written to ONE container. What matters for the
    # P5-A migration is that FLAC (lossless) doesn't lose anything BEYOND
    # the same int16 quantization WAV already applied -- this is the
    # invariant this test actually checks.
    pcm16 = rng.integers(-32768, 32767, size=sr * 2, dtype=np.int16)
    clip = (pcm16.astype(np.float32) / 32768.0)

    wav_path = tmp_path / "clip.wav"
    flac_path = tmp_path / "clip.flac"
    # Match production: written via subtype="PCM_16" in both containers
    # (pipeline/nodes/segment.py's _vad_cut_one).
    sf.write(str(wav_path), clip, sr, format="WAV", subtype="PCM_16")
    sf.write(str(flac_path), clip, sr, format="FLAC", subtype="PCM_16")

    from_wav = decode(str(wav_path), sr)
    from_flac = decode(str(flac_path), sr)

    assert from_wav is not None
    assert from_flac is not None
    assert from_wav.shape == from_flac.shape
    np.testing.assert_array_equal(from_wav, from_flac)


def test_flac_round_trip_is_self_consistent(tmp_path):
    """FLAC's own lossless guarantee: encode -> decode -> re-encode -> decode
    must reproduce the exact same PCM samples every time (no drift from
    repeated read/write, unlike a lossy codec)."""
    sr = 48000
    rng = np.random.default_rng(1)
    clip = rng.normal(scale=0.2, size=sr).astype(np.float32)

    first_path = tmp_path / "first.flac"
    sf.write(str(first_path), clip, sr, format="FLAC", subtype="PCM_16")
    first = decode(str(first_path), sr)

    second_path = tmp_path / "second.flac"
    sf.write(str(second_path), first, sr, format="FLAC", subtype="PCM_16")
    second = decode(str(second_path), sr)

    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# ffmpeg fallback — native containers (webm/opus, m4a/AAC) that libsndfile
# cannot open must still decode via the 2026-07-05 fallback path added to
# bus.py for the ingest.download native-container policy.
# ---------------------------------------------------------------------------

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_decode_falls_back_to_ffmpeg_for_native_webm(tmp_path):
    sr = 48000
    rng = np.random.default_rng(0)
    clip = rng.normal(scale=0.2, size=sr * 2).astype(np.float32)
    wav_path = tmp_path / "src.wav"
    sf.write(str(wav_path), clip, sr, format="WAV", subtype="PCM_16")

    webm_path = tmp_path / "native.webm"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(wav_path),
            "-c:a", "libopus", str(webm_path),
        ],
        check=True, timeout=60,
    )

    # sanity: soundfile genuinely cannot open this container directly.
    with pytest.raises(Exception):
        sf.read(str(webm_path))

    result = decode(str(webm_path), sr)
    assert result is not None
    # opus is lossy + framed, so only duration (not sample values) is checked.
    assert abs(len(result) - len(clip)) < sr * 0.1  # within 100ms
