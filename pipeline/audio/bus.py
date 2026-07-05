"""
pipeline/audio/bus.py
─────────────────────
Shared decode-once layer for the label-suite node.

Replaces three near-identical hand-rolled audio-read+resample implementations:

  • scripts/11_audio_tag.py    — custom polyphase FIR resample  →  32 000 Hz (PANNs)
  • scripts/12_language_id.py  — librosa.resample               →  16 000 Hz (mms-lid)
  • scripts/13_overlap_detect.py — pass-through to pyannote,    →  16 000 Hz (segmentation-3.0)
  • pipeline/nodes/label_music.py — ad-hoc copy of the 32 k path

Previously every detector opened the same WAV file independently and resampled in
isolation.  This module is the single choke-point: read the file exactly once, hand
the decoded array to ``decode_multi`` to produce all required sample rates in one
pass, then let callers distribute the per-rate arrays to their respective models.

Dependencies (CPU-only, no torch):
    numpy, soundfile, soxr
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
import soxr

log = logging.getLogger(__name__)

# ffmpeg fallback timeout — generous because it must cover a full raw
# recording (up to several hours), not just a short segment clip; audio-only
# decode runs far faster than realtime so this is a safety ceiling, not an
# expected duration.
_FFMPEG_TIMEOUT_SEC = 900


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decode(
    path: str,
    target_sr: int,
    *,
    mono: bool = True,
) -> Optional[np.ndarray]:
    """Read *path* from disk and return a float32 array resampled to *target_sr*.

    Parameters
    ----------
    path:
        Absolute or relative path to any format that ``soundfile`` can open
        (WAV, FLAC, OGG, ...), or a native container ffmpeg can open
        (webm/opus, m4a/AAC, mp3) -- the latter fall back to an ffmpeg pipe
        decode automatically (see ``_read_via_ffmpeg``).
    target_sr:
        Desired output sample rate in Hz (e.g. 16000, 32000, 48000).
    mono:
        If ``True`` (default) and the source is multi-channel, downmix to mono
        by averaging all channels.  Single-channel files are returned as-is.

    Returns
    -------
    numpy.ndarray of shape ``(n_samples,)`` and dtype ``float32``, or ``None``
    if the file cannot be read.  Callers must treat ``None`` as "skip" -- do
    **not** let a single unreadable file abort a batch.
    """
    raw, sr = _read(path)
    if raw is None:
        return None

    y = _to_mono(raw) if mono else raw
    return _resample(y, sr, target_sr)


def decode_multi(
    path: str,
    target_srs: list[int],
    *,
    mono: bool = True,
) -> Optional[dict[int, np.ndarray]]:
    """Read *path* exactly once, then produce a resampled array for every rate in *target_srs*.

    This is the core function that eliminates redundant disk I/O: instead of N
    detector scripts each calling ``sf.read()`` on the same file, the label-suite
    node calls this once and distributes the per-rate arrays.

    Parameters
    ----------
    path:
        Path to the audio file.
    target_srs:
        List of desired output sample rates.  Duplicates are silently de-duped
        (the returned dict is keyed by rate, so each rate appears once).
    mono:
        Downmix to mono before resampling when ``True`` (default).

    Returns
    -------
    ``{sample_rate: ndarray}`` mapping, or ``None`` if the initial read fails.
    A ``None`` return means "skip this file entirely" -- callers must never
    receive a partial dict from a failed read.

    Notes
    -----
    If any entry in *target_srs* equals the file's native sample rate the
    corresponding array is the already-decoded buffer (no resample call made).
    """
    raw, sr = _read(path)
    if raw is None:
        return None

    y = _to_mono(raw) if mono else raw

    result: dict[int, np.ndarray] = {}
    for target_sr in dict.fromkeys(target_srs):  # preserve order, remove dups
        result[target_sr] = _resample(y, sr, target_sr)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read(path: str) -> tuple[Optional[np.ndarray], int]:
    """Attempt ``sf.read`` and return ``(array, native_sr)``.

    Falls back to an ffmpeg pipe decode when libsndfile can't open the
    container (2026-07-05: raw files ingested by ``ingest.download`` since
    the 2026-07-04 native-container policy are opus-in-webm/AAC-in-m4a,
    which libsndfile does not support).  On total failure returns
    ``(None, 0)`` and logs a warning -- never raises.  This fail-soft
    contract mirrors the ``read_audio_*`` helpers in the three reference
    scripts that ``bus.py`` replaces.
    """
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        return y, sr
    except Exception as e:
        log.warning(f"soundfile read fail {path}: {e} -- trying ffmpeg fallback")
        return _read_via_ffmpeg(path)


def _probe_native(path: str) -> tuple[Optional[int], int]:
    """ffprobe the first audio stream's (sample_rate, channels).

    Returns ``(None, 0)`` on any failure -- caller treats that as "give up".
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        info = json.loads(out.stdout)["streams"][0]
        return int(info["sample_rate"]), int(info["channels"])
    except Exception as e:
        log.warning(f"ffprobe fail {path}: {e}")
        return None, 0


def _read_via_ffmpeg(path: str) -> tuple[Optional[np.ndarray], int]:
    """Decode *path* via an ffmpeg subprocess pipe (no temp file).

    Used only when ``sf.read`` fails -- covers native containers (webm/
    opus, m4a/AAC) that libsndfile cannot open.  Decodes at the file's own
    native sample rate (``-ar`` pinned to that same rate, so ffmpeg performs
    no implicit resample) -- the existing ``_resample`` step then behaves
    identically regardless of which path decoded the file.
    """
    sr, channels = _probe_native(path)
    if not sr or not channels:
        return None, 0
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", path,
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ar", str(sr), "-ac", str(channels), "-",
            ],
            capture_output=True, timeout=_FFMPEG_TIMEOUT_SEC, check=True,
        )
    except Exception as e:
        log.warning(f"ffmpeg fallback decode fail {path}: {e}")
        return None, 0
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels > 1:
        y = y.reshape(-1, channels)
    return y.copy(), sr


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Average all channels into one.  No-op if already 1-D."""
    if y.ndim > 1:
        return y.mean(axis=1)
    return y


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample *y* from *orig_sr* to *target_sr* using soxr HQ.

    Returns the input unchanged (same object) when the rates are equal so that
    the native-rate path in ``decode_multi`` incurs zero cost.
    """
    if orig_sr == target_sr:
        return y
    return soxr.resample(y, orig_sr, target_sr, quality="HQ").astype(np.float32)
