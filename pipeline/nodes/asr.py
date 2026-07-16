"""
pipeline/nodes/asr.py
asr.transcribe DAG node — four ASR models across three backends (faster-whisper +
qwen-asr + sense_voice), ported from scripts/04_transcribe.py onto the
orchestrator's GPUWorkerBase + JSONL worker protocol.  asr.agreement is a
separate CPU-only node that computes N-way cross-model char-overlap agreement
once ≥2 models' asr_results rows exist for a segment (item-level dependency —
an id becomes eligible the moment its second model result lands, no stage
barrier; a third/fourth-model straggler re-triggers and improves the row, never
blocks it).

whisper_v3 RETIRED (2026-07-10, owner decision — see DECISIONS.md and
docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md): measurably the least accurate of the
four backends and a disproportionate drag on cross-model agreement (excluding it
raised the 3-way ≥0.90 agreement coverage from 15.7% to 41.1% of the corpus).

canto_ft RETIRED (2026-07-13, owner decision — see DECISIONS.md): faster-whisper's
per-item sequential decode (no batched-tensor path — see TranscribeWorker.forward_batch()
below) hard-ceilings at ~4.45/s/GPU regardless of batching/sharding strategy, making
it the throughput bottleneck for any large backlog (T15's 578,889-segment reingest
was projected at ~18-24h for canto_ft's share alone vs ~4.4h for qwen3_asr on the
same two GPUs). Calibration-review CER also measured canto_ft in the same poor
17-36% band as the already-retired whisper_v3, vs qwen3_asr's ~0.4% (DECISIONS.md
2026-07-10 entry) — same "slow AND inaccurate" profile that justified whisper_v3's
retirement. ASR_MODELS["canto_ft"]["enabled"] = False — asr.transcribe no longer
dispatches it and asr.agreement excludes its historical asr_results text from both
the agreement score and best_text candidacy (EXCLUDED_FROM_AGREEMENT below). Its
historical rows are kept for audit only, never read by any live node going forward.

Known consequence (accepted, not yet re-solved — see DECISIONS.md 2026-07-13):
tier.assign's `auto_gold` gate requires `canto_ft_confidence > 0.8` because canto_ft
was the only active model with a real logprob-derived confidence (qwen3_asr/sense_voice
both report a nominal 1.0 placeholder — see their Worker docstrings below). With
canto_ft retired, `canto_ft_confidence` is always None for new segments, which
assign_tier() already treats as failing the auto_gold gate (see pipeline/nodes/tier.py) —
so new segments cap at silver/bronze until a 2-model-agreement-only auto_gold
threshold is data-driven and adopted (owner wants agreement-distribution stats
checked first, e.g. via a fresh FINDINGS_ASR_AGREEMENT_THRESHOLDS.md-style pass,
before picking a number — do not hardcode a guess).

Two ASR backends (qwen3_asr, sense_voice) are now the active set.

Both/all models run under ONE supervisor process (run_asr_transcribe), never as
separate `pipe run` invocations — DuckDB is single-writer (P2 backlog finding,
2026-07-03): a second concurrent process's RW connect() would just block on the
first, so all devices are dispatched from inside the same asyncio run sharing
one catalog connection.

Hard constraint #7 (CLAUDE.md / KNOWN_ISSUES.md §9): NEVER language="yue" for
Whisper models.  Both faster-whisper models pass language="zh" with a Cantonese
written-form prompt.  Qwen3-ASR is a distinct transformer architecture with
native Cantonese dialect support; it uses language="Cantonese" (full English
name, per the qwen-asr package API).

# Backend key in ASR_MODELS["backend"]:
#   "faster_whisper" — uses faster_whisper.WhisperModel (ctranslate2 backend),
#                      TranscribeWorker subclass.
#   "qwen_asr"       — uses qwen_asr.Qwen3ASRModel (transformers backend),
#                      Qwen3ASRWorker subclass.
#   "sense_voice"    — uses funasr.AutoModel (SenseVoiceSmall, CTC encoder-only,
#                      non-autoregressive), SenseVoiceWorker subclass.
#                      Outputs Traditional HK Chinese via OpenCC s2hk conversion.
#                      Also emits emotion + audio-event tags (stored in metadata
#                      field but not in asr_results.text).
# worker_main() dispatches to the correct class at subprocess start-up via
# WORKER_CLASSES dict; the JSONL stdin/stdout protocol (ready/task/result/error/
# shutdown) is byte-identical regardless of backend.

Resampler note (updated 2026-07-14, owner-approved — see DECISIONS.md): the
16 kHz downsample now uses soxr.resample(quality="HQ") instead of
scipy.signal.resample_poly(wav, 1, 3).  The original scipy choice (2026-07-03)
was golden-set parity with scripts/04_transcribe.py's faster-whisper models —
both of which are now retired (whisper_v3 2026-07-10, canto_ft 2026-07-13), so
that parity constraint no longer binds.  soxr HQ is ~4x faster per resample and
is the same libsoxr engine audio/bus.py and librosa (≥0.10 default) use;
quality is equal-or-better (125 dB SNR at HQ) for the exact 3:1 ratio.  The
swap is part of the decode+resample throughput fix (the CPU preprocessing
stage, not the GPU, was the measured bottleneck once sense_voice's forward
pass was batched — see the worker pipelining note in worker_main()'s docstring).

ASR text is NOT expected to match the legacy snapshot byte-for-byte, even with
identical uv.lock-pinned ctranslate2/faster-whisper versions and identical audio
— confirmed both are unchanged since before the corpus was originally
transcribed, yet output still differs, stably (not randomly: repeat calls, same
or fresh CUDA context, are 100% reproducible).  The likely remaining variable is
GPU-driver-level numerical drift, outside this package's control.  Golden parity
for this node is therefore similarity-tolerance (char_agreement), not exact-match
— see tests/golden/check_asr_parity.py and REARCHITECTURE_IMPLEMENTATION_PLAN.md
§9.1.
"""

import argparse
import asyncio
import difflib
import itertools
import json
import logging
import math
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from pipeline.config import REPO_ROOT
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

TARGET_SR = 48000
ASR_SR = 16000

# Cantonese written-form initial prompt (helps large-v3 produce 粵語白話文) — verbatim
# copy of scripts/04_transcribe.py's CANTO_PROMPT for golden-set parity.
CANTO_PROMPT = (
    "以下係廣東話口語，請用粵語白話文書寫，"
    "例如：係、唔係、冇、喺、佢哋、嘅、嗰、嚟。"
)

_LOCAL_CANTO = str(REPO_ROOT / "data" / "ct2_models" / "whisper-large-v2-cantonese")

# model_key (CLI/internal selector) → cfg.
#
# cfg["id"] must stay byte-identical to scripts/04_transcribe.py's ASR_MODELS so
# the stored `asr_results.model` field (id + "+" + lang) matches the 910,598 rows
# already imported from the legacy manifest — new rows land in the same (id, model)
# primary-key space.
#
# cfg["backend"] selects the worker class:
#   "faster_whisper" — WhisperModel via ctranslate2; used by canto_ft and whisper_v3.
#   "qwen_asr"       — Qwen3ASRModel via transformers; used by qwen3_asr.
#
# The key "qwen3_asr" with id="Qwen/Qwen3-ASR-1.7B" and lang="Cantonese" produces
# model_field() = "Qwen/Qwen3-ASR-1.7B+Cantonese", a distinct string that never
# collides with the two faster-whisper model fields already in asr_results.
ASR_MODELS = {
    # RETIRED 2026-07-13 (owner decision, DECISIONS.md): faster-whisper's per-item
    # sequential decode (TranscribeWorker.forward_batch() below has no batched-tensor
    # path) hard-ceilings at ~4.45/s/GPU no matter how work is sharded -- made it the
    # throughput bottleneck for the T15 578,889-segment backlog (~18-24h projected for
    # canto_ft's share alone). CER also measured in the same poor 17-36% band as the
    # already-retired whisper_v3, vs qwen3_asr's ~0.4% -- same "slow AND inaccurate"
    # profile. "enabled": False means (a) asr.transcribe refuses to dispatch it and
    # (b) asr.agreement excludes its asr_results text from both the agreement score
    # and best_text candidacy (EXCLUDED_FROM_AGREEMENT below). Historical rows are
    # NOT deleted -- kept for audit/reference only, never read by any live node going
    # forward. Known consequence: tier.assign's auto_gold gate needs canto_ft_confidence
    # (the only active model with a real, non-nominal confidence) -- new segments now
    # cap at silver/bronze until a 2-model-agreement-only auto_gold threshold is
    # data-driven and adopted (see module docstring above; do not hardcode a guess).
    "canto_ft": {
        "id": _LOCAL_CANTO,
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "backend": "faster_whisper",
        "description": "Cantonese fine-tuned Whisper large-v2 (local ct2)",
        "enabled": False,
    },
    # RETIRED 2026-07-10 (owner decision, DECISIONS.md): measurably the least accurate of
    # the four backends and a drag on cross-model agreement -- see
    # docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md. "enabled": False means (a) asr.transcribe
    # refuses to dispatch it (pipeline/cli.py guard rail) and (b) asr.agreement excludes its
    # asr_results text from both the agreement score and best_text candidacy (see
    # EXCLUDED_FROM_AGREEMENT below). Its historical asr_results rows are NOT deleted --
    # kept for audit/reference only, never read by any live node going forward.
    "whisper_v3": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "backend": "faster_whisper",
        "description": "Whisper large-v3 with zh + Cantonese prompt",
        "enabled": False,
    },
    # Apache-2.0.  1.7B transformer model with native 52-language support including
    # Cantonese (yue) as a distinct dialect from Mandarin (zh), with HK/Guangdong
    # accent coverage.  Uses qwen-asr transformers backend (NOT faster-whisper /
    # ctranslate2 — a completely separate inference stack).
    # Install:  uv pip install qwen-asr   (NOT uv sync — that would prune CUDA torch)
    "qwen3_asr": {
        "id": "Qwen/Qwen3-ASR-1.7B",
        "lang": "Cantonese",    # full English name required by qwen-asr API; NOT "yue"
        "prompt": None,         # qwen-asr has no separate initial_prompt concept
        "backend": "qwen_asr",
        "description": "Qwen3-ASR-1.7B (native Cantonese/Yue support, Apache-2.0)",
    },
    # ModelScope model license (commercial use permitted, non-OSI).
    # SenseVoice-Small is a CTC encoder-only (non-autoregressive) model with native
    # Cantonese (yue) support trained on 400k+ hours of multilingual data.  It is
    # ~15× faster than Whisper-Large (measured ~105× RTF on this machine 2026-07-08).
    # Extra outputs (emotion, audio-event) are extracted from inline tags and stored
    # in the metadata field of each result row but stripped from the stored text.
    # Output is Simplified Chinese by default — SenseVoiceWorker applies OpenCC
    # s2hk conversion so stored text matches the Traditional HK convention of the
    # other three models.
    # Install:  uv pip install opencc-python-reimplemented funasr modelscope
    #           (NOT uv sync — same CUDA-torch-prune risk as qwen-asr)
    "sense_voice": {
        "id": "iic/SenseVoiceSmall",
        "lang": "yue",          # explicit Cantonese language code (ISO 639-3)
        "prompt": None,
        "backend": "sense_voice",
        "description": "SenseVoice-Small (CTC non-autoregressive, native yue, ~105× RTF)",
    },
}


def model_field(model_key: str) -> str:
    """The exact string stored in asr_results.model — id + '+' + lang, matching
    scripts/04_transcribe.py's transcribe_one().  When lang is falsy (None/''),
    returns just id (no '+' suffix)."""
    cfg = ASR_MODELS[model_key]
    return cfg["id"] + (f"+{cfg['lang']}" if cfg["lang"] else "")


def is_model_enabled(model_key: str) -> bool:
    return ASR_MODELS[model_key].get("enabled", True)


# canto_ft's cfg["id"] is REPO_ROOT-derived (see _LOCAL_CANTO above), and this repo has
# moved directories twice historically -- asr_results therefore has canto_ft rows under
# three distinct absolute-path model strings for the same logical model (measured
# 2026-07-10: 618,695 current-path + 29,341 + 4,730 legacy-path rows). These aliases let
# resolve_model_key() recognise the two stale paths as "canto_ft" too, so agreement
# computation can dedupe them instead of double-counting canto_ft's opinion.
_LEGACY_MODEL_ALIASES = {
    "/mnt/Drive3/Development/AI-ML/canto-corpus/data/ct2_models/whisper-large-v2-cantonese+zh": "canto_ft",
    "/home/typangaa/Documents/canto-corpus/data/ct2_models/whisper-large-v2-cantonese+zh": "canto_ft",
}


def resolve_model_key(model_field_value: str) -> str | None:
    """Map a raw asr_results.model string back to its ASR_MODELS key. Returns None for
    unrecognised strings (fail-closed: a future 5th model is silently excluded from
    agreement/best_text until explicitly added to ASR_MODELS, never silently included)."""
    for key in ASR_MODELS:
        if model_field_value == model_field(key):
            return key
    return _LEGACY_MODEL_ALIASES.get(model_field_value)


# Models whose asr_results text is excluded from both the agreement score AND best_text
# candidacy (owner decision 2026-07-10 for whisper_v3, 2026-07-13 for canto_ft -- see
# their ASR_MODELS comments and docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md). Historical
# rows are kept, just never read by compute_agreement_row() below.
EXCLUDED_FROM_AGREEMENT = {"whisper_v3", "canto_ft"}


# ---------------------------------------------------------------------------
# Audio loading (parity-preserving — see module docstring)
# ---------------------------------------------------------------------------

def _load_and_resample(wav_path: str) -> np.ndarray | None:
    """Load a 48 kHz segment (FLAC or WAV) and downsample to 16 kHz for ASR.
    Thread-safe and GIL-releasing on both stages: libsndfile (sf.read) and
    libsoxr (soxr.resample) each release the GIL, so an 8-16 thread pool scales
    near-linearly — unlike scipy.resample_poly (the pre-2026-07-14 choice, ~4x
    slower per file; see the module-docstring resampler note)."""
    import soxr
    try:
        wav48, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"read fail {wav_path}: {e}")
        return None
    if wav48.ndim > 1:
        wav48 = wav48.mean(axis=1)
    if sr == ASR_SR:
        return wav48.astype(np.float32, copy=False)
    return soxr.resample(wav48, sr, ASR_SR, quality="HQ").astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

TRANSCRIBE_DISCOVER_SQL = """
    SELECT s.id, s.audio_path, s.duration_sec
    FROM segments s
    LEFT JOIN asr_results a ON s.id = a.id AND a.model = ?
    WHERE a.id IS NULL
    ORDER BY s.duration_sec
"""


def discover_transcribe(conn, model_key: str) -> list[tuple]:
    return conn.execute(TRANSCRIBE_DISCOVER_SQL, [model_field(model_key)]).fetchall()


def shard_rows_round_robin(rows: list[tuple], devices: list[str]) -> dict[str, list[tuple]]:
    """Split rows across devices round-robin (row i -> devices[i % len(devices)]).

    Used when the same model_key is assigned to more than one device (e.g. a
    single model split across both GPUs) — discover_transcribe()'s result is
    ordered by duration_sec ascending, so a contiguous split would put all the
    short/fast segments on one device and all the long/slow ones on another;
    round-robin gives every device the same short/long mix so they finish at
    roughly the same time. With a single device, returns {device: rows} unchanged.
    """
    shards: dict[str, list[tuple]] = {device: [] for device in devices}
    for i, row in enumerate(rows):
        shards[devices[i % len(devices)]].append(row)
    return shards


# N-way agreement discovery (replaces the old 2-model self-join).
#
# Design: one row per id in asr_results, grouped to collect all (model, text,
# confidence) tuples.  An id surfaces when:
#   (a) it has ≥2 asr_results rows AND no asr_agreement row yet (first compute), OR
#   (b) it has an asr_agreement row written by this node (model_count IS NOT NULL)
#       AND the current count of asr_results rows exceeds the stored model_count
#       (3rd model straggler arrived — re-compute to include it).
#
# Legacy asr_agreement rows imported by P0 have model_count = NULL (the column
# was added by ALTER TABLE after P0 import), so condition (b) never triggers for
# them.  This preserves the ~910k existing rows untouched, as required.
#
# Each result row has shape:
#   (id, list_of_models, list_of_texts, list_of_confidences, result_count)
# where list_of_models[i]/list_of_texts[i]/list_of_confidences[i] all correspond to the
# same underlying asr_results row (all three list()s share the same ORDER BY key, which
# DuckDB guarantees keeps them aligned).  models is needed so compute_agreement_row() can
# resolve each entry to a canonical model key -- to dedupe canto_ft's legacy-path
# duplicates and exclude EXCLUDED_FROM_AGREEMENT models (see resolve_model_key() above).
AGREEMENT_DISCOVER_SQL = """
    WITH ranked AS (
        SELECT
            r.id,
            list(r.model ORDER BY r.model) AS models,
            list(r.text ORDER BY r.model) AS texts,
            list(r.confidence ORDER BY r.model) AS confidences,
            count(*) AS result_count
        FROM asr_results r
        GROUP BY r.id
        HAVING count(*) >= 2
    )
    SELECT ranked.id, ranked.models, ranked.texts, ranked.confidences, ranked.result_count
    FROM ranked
    LEFT JOIN asr_agreement ag ON ranked.id = ag.id
    WHERE ag.id IS NULL
       OR (ag.model_count IS NOT NULL AND ranked.result_count > ag.model_count)
"""


def discover_agreement(conn) -> list[tuple]:
    return conn.execute(AGREEMENT_DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Agreement computation (pure logic — N-way generalisation of the 2-model
# port from scripts/04_transcribe.py's char_agreement() + write_transcripts())
# ---------------------------------------------------------------------------

_DIGIT_TO_CJK = str.maketrans({
    **{str(d): cjk for d, cjk in zip(range(10), "〇一二三四五六七八九")},
    **{chr(0xFF10 + d): cjk for d, cjk in zip(range(10), "〇一二三四五六七八九")},
})


def _normalize_for_agreement(text: str) -> str:
    """Comparison-only normalization for char_agreement() (Issue #20, see
    docs/PIPELINE_REVIEW_2026-07-13.md §2 row 20 and §5 Q3-1): qwen3_asr (AR)
    infers punctuation from LM context while sense_voice (CTC) never emits it,
    so raw-text SequenceMatcher.ratio() systematically deflates cross-model
    agreement on punctuation alone. Strip all Unicode punctuation (category
    "P*", covers both ASCII and CJK marks) and fold Arabic/full-width digits to
    CJK numerals before comparing. Never mutates stored text/best_text -- those
    keep the original, unnormalized strings."""
    text = text.translate(_DIGIT_TO_CJK)
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def char_agreement(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    normalized = [_normalize_for_agreement(t) for t in texts]
    ratios = [
        difflib.SequenceMatcher(None, a, b).ratio()
        for a, b in itertools.combinations(normalized, 2)
    ]
    return round(sum(ratios) / len(ratios), 3)


def compute_agreement_row(
    seg_id: str,
    models: list[str],
    texts: list[str],
    confidences: list[float],
    result_count: int,
) -> dict:
    """N-way generalisation of the 2-model write_transcripts() logic.

    models/texts/confidences are parallel lists from asr_results, one entry per
    asr_results row (any order). result_count is the raw asr_results row count for
    this id (INCLUDING excluded/legacy-path rows), stored as model_count so future
    re-discover can detect when a later model's result arrives -- unrelated to how
    many entries actually contribute to agreement below.

    Resolution + dedup (owner decision 2026-07-10, see EXCLUDED_FROM_AGREEMENT and
    ASR_MODELS["whisper_v3"]'s comment; docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md
    for the evidence):
      1. Each (model, text, confidence) is resolved to a canonical ASR_MODELS key via
         resolve_model_key(). Unrecognised model strings are dropped.
      2. canto_ft has legacy-path duplicate rows for ~5.5% of segments (see
         _LEGACY_MODEL_ALIASES) -- only the row whose raw model string matches the
         CURRENT model_field("canto_ft") is kept; stale-path rows for the same id are
         dropped (never averaged/maxed with the current one).
      3. Models in EXCLUDED_FROM_AGREEMENT (whisper_v3) are resolved (so their row isn't
         silently mistaken for "unrecognised"/dropped-with-a-warning) but excluded from
         the `active` set used for both the agreement ratio AND best_text candidacy --
         i.e. an excluded model's transcript can never win best_text either.

    Agreement is 0.0 unless ≥2 non-empty ACTIVE candidate texts exist. best_text is the
    non-empty active candidate with the highest confidence; an empty-text candidate
    always loses regardless of its raw confidence value (matching the original
    2-model behaviour). canto_ft_confidence is the active canto_ft candidate's raw
    confidence (or None if canto_ft has no active row for this id), used by
    tier.assign's auto_gold gate -- NOT included in the agreement/best_text logic
    itself, just carried through as its own field.
    """
    current_canto_ft = model_field("canto_ft")
    by_key: dict[str, dict] = {}
    for m, t, c in zip(models, texts, confidences):
        key = resolve_model_key(m)
        if key is None:
            continue
        if key == "canto_ft" and m != current_canto_ft:
            continue  # stale-path duplicate (pre-move REPO_ROOT) -- never authoritative,
                      # regardless of encounter order; only the current-path row is kept
        by_key[key] = {"text": t or "", "confidence": c or 0.0}

    canto_ft_confidence = by_key.get("canto_ft", {}).get("confidence")

    active = {k: v for k, v in by_key.items() if k not in EXCLUDED_FROM_AGREEMENT}
    non_empty_texts = [v["text"] for v in active.values() if v["text"]]
    agreement = char_agreement(non_empty_texts) if len(non_empty_texts) >= 2 else 0.0

    active_candidates = list(active.values())
    if active_candidates:
        best = max(active_candidates, key=lambda c: c["confidence"] if c["text"] else -1)
        best_text = best["text"]
    else:
        best_text = ""

    return {
        "id": seg_id,
        "agreement": agreement,
        "best_text": best_text,
        "text_verified": False,
        "model_count": result_count,
        "canto_ft_confidence": canto_ft_confidence,
    }


# ---------------------------------------------------------------------------
# Supervisor: asr.transcribe — one worker per (model_key, device) assignment,
# dispatched concurrently within a single asyncio run / single DB connection.
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_asr_transcribe(
    assignments: list[tuple[str, str]],
    *,
    conn=None,
    gpu_policy: str = "cap",
    batch_size: int = 8,
    mem_fraction: float | None = None,
    limit: int | None = None,
    prefetch: int = 2,
    io_workers: int = 16,
) -> dict:
    """Supervisor entrypoint for the asr.transcribe node.

    assignments: list of (model_key, device), e.g.
        [("canto_ft", "cuda:0"), ("whisper_v3", "cuda:1"), ("qwen3_asr", "cuda:0")]
    Each assignment gets its own discover() query (different model = different
    anti-join target), own dispatch queue, and own worker subprocess — but all
    run under ONE asyncio.gather() sharing a single DuckDB connection, so this
    is the "N models split across GPUs, running in parallel" node the plan
    calls for, without ever opening two competing RW connections to the catalog.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run asr.transcribe` usage.

    prefetch (2026-07-14): number of tasks kept in flight per worker so the
    worker's preprocess stage (decode+resample) overlaps the GPU forward of
    the previous task — see worker_main()'s docstring. The device pool is now
    acquired around the SEND only (dispatch gating): the foreign-GPU sampler
    still throttles new dispatches by lowering the pool target, and in-flight
    tasks are still never preempted, but preprocessing no longer serialises
    with GPU compute. 1 restores the old strictly-sequential behaviour.

    io_workers: decode+resample thread-pool size inside each worker subprocess
    (passed through as --io-workers; sf.read and soxr both release the GIL).
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = conn or connect()

    # Group devices by model_key so the SAME model can be split across multiple
    # GPUs (e.g. [("qwen3_asr", "cuda:0"), ("qwen3_asr", "cuda:1")]) — discovery
    # is per model_key, not per (model_key, device), so without this grouping
    # every device assigned to the same model would each re-fetch and process
    # the ENTIRE backlog independently (pure duplicated work, not a real split).
    devices_by_model: dict[str, list[str]] = {}
    for model_key, device in assignments:
        devices_by_model.setdefault(model_key, []).append(device)

    per_assignment_rows: dict[tuple[str, str], list[tuple]] = {}
    total = 0
    for model_key, devices in devices_by_model.items():
        rows = discover_transcribe(conn, model_key)
        if limit:
            rows = rows[:limit]
        shards = shard_rows_round_robin(rows, devices)
        for device in devices:
            per_assignment_rows[(model_key, device)] = shards[device]
            total += len(shards[device])
            log.info(f"asr.transcribe[{model_key}]: {len(shards[device])} segments to transcribe on {device}")

    if total == 0:
        return {"processed": 0, "errors": 0}

    registry = PoolRegistry()
    registered: set[str] = set()
    pool_names: dict[tuple[str, str], str] = {}
    for model_key, device in assignments:
        pool_name = f"gpu.{device.split(':')[1]}" if device.startswith("cuda") else "cpu"
        if pool_name not in registered:
            registry.register(pool_name, target=1)
            registered.add(pool_name)
        pool_names[(model_key, device)] = pool_name

    handles = {}
    for model_key, device in assignments:
        cmd = [
            sys.executable, "-m", "pipeline.nodes.asr",
            "--device", device, "--model-key", model_key,
            "--io-workers", str(io_workers),
        ]
        if mem_fraction is not None and device.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=180.0)
        handles[(model_key, device)] = handle
        log.info(f"worker ready: {model_key} -> {device} (pid={handle.pid})")

    gpu_policies = {
        name: GpuPolicy(gpu_policy)
        for name in set(pool_names.values()) if name.startswith("gpu.")
    }
    sampler = Sampler(
        registry, gpu_policies,
        own_pids=lambda: {h.pid for h in handles.values()},
        poll_interval=2.0,
    )
    sampler_task = asyncio.create_task(sampler.run())

    run_id = new_run_id("asr.transcribe")
    processed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(model_key: str, device: str) -> None:
        nonlocal processed, errors
        handle = handles[(model_key, device)]
        pool = registry.get(pool_names[(model_key, device)])
        mf = model_field(model_key)
        queue: asyncio.Queue = asyncio.Queue()
        for batch in _batches(per_assignment_rows[(model_key, device)], batch_size):
            queue.put_nowait(batch)

        # Double-buffered dispatch (2026-07-14, see run_asr_transcribe docstring):
        # keep up to `prefetch` tasks in flight so the worker's preprocess stage
        # decodes task N+1 while its GPU runs task N. The pool is acquired
        # around the SEND only — foreign-GPU yield still gates new dispatches,
        # in-flight tasks are never preempted. Results are matched by task_id
        # (worker emits FIFO, but matching by id keeps a late straggler after a
        # timeout from being attributed to the wrong batch).
        inflight: dict[str, list[tuple]] = {}
        seq = 0

        async def dispatch_next() -> bool:
            nonlocal seq
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return False
            task_id = f"{model_key}@{device}-{seq}"
            seq += 1
            items = [{"id": r[0], "path": r[1]} for r in batch]
            async with pool.acquire():
                await handle.send_task(task_id, items)
            inflight[task_id] = batch
            return True

        while True:
            while len(inflight) < max(1, prefetch) and await dispatch_next():
                pass
            if not inflight:
                return
            try:
                result = await handle.read_message(timeout=600.0)
            except Exception as e:
                # A timeout/death can't be attributed to one specific task —
                # fail everything currently in flight for this worker (same
                # accounting as the old per-batch failure path, scaled to the
                # window). If a straggler result for one of these ids arrives
                # later it is dropped by the unknown-task_id guard below.
                n_failed = sum(len(b) for b in inflight.values())
                log.error(f"{model_key}@{device}: read failed with {len(inflight)} task(s) "
                          f"in flight ({n_failed} segments): {e}")
                errors += n_failed
                inflight.clear()
                continue
            batch = inflight.pop(result.get("task_id"), None)
            if batch is None:
                log.warning(f"{model_key}@{device}: result for unknown/expired task "
                            f"{result.get('task_id')!r} — dropped")
                continue
            if result["type"] == "error":
                log.error(f"{model_key}@{device}: worker error: {result['error']}")
                errors += len(batch)
                continue

            # metadata is only populated by backends that produce backend-specific
            # extras (currently sense_voice's {emotion, audio_event}); r.get(...)
            # defaults to None for every other backend so every row in this batch
            # carries the same key set (upsert_rows derives columns from rows[0]).
            out_rows = [
                {"id": r["id"], "model": mf, "text": r["text"], "confidence": r["confidence"],
                 "metadata": r.get("metadata")}
                for r in result["rows"]
            ]
            # Unreadable audio still gets a placeholder row (empty text, confidence
            # 0.0) so discover()'s anti-join stops resurfacing the same dead id —
            # mirrors label_music.py's skipped_ids handling.
            skipped_rows = [
                {"id": sid, "model": mf, "text": "", "confidence": 0.0, "metadata": None}
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{model_key}@{device}: {len(skipped_rows)} unreadable segment(s): "
                            f"{[r['id'] for r in skipped_rows][:5]}")

            upsert_rows(conn, "asr_results", out_rows + skipped_rows, ["id", "model"])
            record_batch(conn, run_id, "asr.transcribe", [r["id"] for r in out_rows], "ok",
                         metrics=result.get("metrics"))
            if skipped_rows:
                record_batch(conn, run_id, "asr.transcribe", [r["id"] for r in skipped_rows],
                             "error", error="unreadable audio file")

            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{total} processed ({rate:.1f}/s), pools={registry.snapshot()}")

    await asyncio.gather(*(
        worker_loop(model_key, device) for model_key, device in assignments
    ))

    sampler.stop()
    await asyncio.gather(sampler_task, return_exceptions=True)
    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} processed, {errors} errors in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# Supervisor: asr.agreement — CPU-only, pure Python (difflib), no GPU/subprocess
# workers needed. Runs directly in the supervisor process.
# ---------------------------------------------------------------------------

async def run_asr_agreement(*, conn=None, batch_size: int = 2000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run asr.agreement` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()

    # Ensure the model_count/canto_ft_confidence columns exist (added post-P0; safe no-op
    # if already present). Legacy rows imported by P0 will have model_count = NULL,
    # preventing re-trigger.
    conn.execute(
        "ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS model_count INTEGER"
    )
    conn.execute(
        "ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS canto_ft_confidence DOUBLE"
    )

    rows = discover_agreement(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"asr.agreement: {len(rows)} segments to score")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("asr.agreement")
    processed = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_rows = [
            compute_agreement_row(seg_id, models, texts, confidences, result_count)
            for seg_id, models, texts, confidences, result_count in batch
        ]
        upsert_rows(conn, "asr_agreement", out_rows, ["id"])
        record_batch(conn, run_id, "asr.agreement", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} scored ({rate:.1f}/s)")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} scored in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": 0, "run_id": run_id}


# ---------------------------------------------------------------------------
# GPU workers (subprocess side) — asr.transcribe only; asr.agreement has no worker.
# ---------------------------------------------------------------------------

class TranscribeWorker(GPUWorkerBase):
    # NOTE: GPUWorkerBase's mem_fraction cap calls torch.cuda.set_per_process_memory_
    # fraction(), which only constrains PyTorch's own CUDA caching allocator.
    # faster-whisper's ctranslate2 backend allocates CUDA memory through its own
    # independent pool, entirely outside torch's allocator — so --mem-fraction is a
    # silent no-op for this node. Kept for CLI/interface consistency with the other
    # GPU nodes (label.music/label.suite, which ARE torch-backed); real VRAM control
    # for this node is ctranslate2's own compute_type choice (int8_float16 below).
    def __init__(self, device: str, model_key: str, *, mem_fraction: float | None = None,
                 fp16: bool = True) -> None:
        self.model_key = model_key
        self.cfg = ASR_MODELS[model_key]
        super().__init__(device, mem_fraction=mem_fraction, fp16=fp16)

    def load_model(self):
        from faster_whisper import WhisperModel
        is_cuda = str(self.device).startswith("cuda")
        compute_type = "int8_float16" if is_cuda else "int8"
        gpu_index = int(self.device.split(":")[1]) if is_cuda and ":" in str(self.device) else 0
        device_kind = "cuda" if is_cuda else "cpu"
        log.info(f"Loading ASR model: {self.cfg['description']} on {self.device} [{compute_type}]")
        return WhisperModel(
            self.cfg["id"],
            device=device_kind,
            device_index=gpu_index,
            compute_type=compute_type,
            cpu_threads=4,
        )

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        # faster-whisper has no native batched-tensor API here (BatchedInferencePipeline
        # is a separate opt-in path not used by the legacy script) — sequential per-item
        # decode, matching scripts/04_transcribe.py's transcribe_one() exactly.
        return [self._transcribe_one(y16) for y16 in items]

    def _transcribe_one(self, y16: np.ndarray) -> dict:
        kwargs: dict = {
            "beam_size": 1,   # greedy — matches legacy exactly for golden parity
            "vad_filter": False,
            "temperature": 0.0,
        }
        if self.cfg["lang"]:
            kwargs["language"] = self.cfg["lang"]  # NEVER "yue" — hard constraint #7
        if self.cfg["prompt"]:
            kwargs["initial_prompt"] = self.cfg["prompt"]

        segments, info = self.model.transcribe(y16, **kwargs)
        segs_list = list(segments)
        text = "".join(s.text for s in segs_list).strip()
        if segs_list:
            conf = float(np.mean([s.avg_logprob for s in segs_list if hasattr(s, "avg_logprob")]))
            conf = round(max(0.0, min(1.0, math.exp(conf))), 3)
        else:
            conf = 0.0
        return {"text": text, "confidence": conf}


class Qwen3ASRWorker(GPUWorkerBase):
    """Worker for the qwen-asr transformers backend (Qwen3-ASR-1.7B).

    Qwen3ASRModel does not expose a logprob-derived confidence score in its
    result objects (result.text and result.language are available; no numeric
    quality signal is documented or exposed in the transformers backend).
    We therefore default confidence to 1.0 for all non-empty results, which
    means agreement.best_text will prefer the Qwen3-ASR output over a Whisper
    output only when both have the same placeholder confidence — users relying
    on the raw confidence field for downstream filtering should treat Qwen3-ASR
    confidence as nominal, not calibrated.

    Output script: despite being prompted with language="Cantonese", Qwen3-ASR
    intermittently emits Simplified Chinese characters — measured 2026-07-10 at
    ~15.3% of segments corpus-wide (94,783/618,695), often mixed within an
    otherwise-Traditional sentence (e.g. "讲话佢又识法文" — 讲/识 simplified,
    the rest Traditional/Cantonese). Same class of issue as SenseVoiceWorker's
    known Simplified output (see that class's docstring); fixed the same way —
    OpenCC s2hk conversion applied to every non-empty result before it's
    returned, so stored text matches the Traditional HK convention of the
    other three models.
    """

    def __init__(self, device: str, model_key: str, *, mem_fraction: float | None = None,
                 fp16: bool = True) -> None:
        self.model_key = model_key
        self.cfg = ASR_MODELS[model_key]
        super().__init__(device, mem_fraction=mem_fraction, fp16=fp16)
        # OpenCC converter: Simplified → Traditional HK (initialised in subprocess).
        self._cc = None

    def load_model(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        try:
            from opencc import OpenCC
            self._cc = OpenCC("s2hk")   # Simplified → Traditional HK
        except ImportError:
            log.warning(
                "opencc-python-reimplemented not installed — Qwen3-ASR output may "
                "remain in Simplified Chinese. Install: uv pip install opencc-python-reimplemented"
            )
            self._cc = None

        log.info(f"Loading ASR model: {self.cfg['description']} on {self.device} [bfloat16]")
        return Qwen3ASRModel.from_pretrained(
            self.cfg["id"],
            dtype=torch.bfloat16,
            device_map=self.device,
            # Batched inference — matches the supervisor's --batch size so a full
            # queued batch actually runs through the model together instead of
            # being silently serialized internally; OOM halving in GPUWorkerBase
            # still protects against transient spikes if this is too large.
            # Empirically tuned 2026-07-07 on this machine (RTX 4090, 24GB): 1=2.1/s,
            # 16=14.4/s, 32=22.1/s, 64=30.1/s, 128=34.6/s (diminishing returns past 64,
            # measured on the shortest segments in the duration-ascending discovery
            # queue — longer segments later in a full run will use more memory per
            # item and run slower than this early-window number).
            max_inference_batch_size=64,
        )

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        # qwen-asr's transcribe() accepts a list of (np.ndarray, sr) tuples for
        # batch inference, but with max_inference_batch_size=1 the model processes
        # them sequentially anyway.  We pass them as a list so the API handles
        # ordering; one result per input item, same order.
        audio_inputs = [(y16, ASR_SR) for y16 in items]
        lang = self.cfg["lang"] or None  # None → auto-detect; "Cantonese" → forced
        results = self.model.transcribe(audio=audio_inputs, language=lang)
        # Qwen3-ASR result object has .text (str) and .language (str).
        # No logprob/confidence field — default to 1.0 for non-empty text.
        # (See class docstring for the implication on best_text selection.)
        out = []
        for r in results:
            text = (r.text or "").strip()
            if self._cc and text:
                text = self._cc.convert(text)   # Simplified → Traditional HK
            out.append({"text": text, "confidence": 1.0 if text else 0.0})
        return out


class SenseVoiceWorker(GPUWorkerBase):
    """Worker for the funasr SenseVoice-Small backend.

    SenseVoice-Small is a CTC encoder-only (non-autoregressive) model with
    native Cantonese (yue) support.  It is architecturally independent from all
    three autoregressive models (canto_ft, whisper_v3, qwen3_asr), so its
    agreement/disagreement signal is structurally different — genuinely useful
    as a 4th voter in N-way consensus scoring.

    Key properties:
    - Speed: ~105× RTF on RTX 4090 (measured 2026-07-08); entire 618k corpus
      in ~10 minutes on 2 GPUs.
    - Language: native 'yue' (Cantonese) — NOT a zh+prompt workaround.
    - Output script: SenseVoice emits Simplified Chinese; OpenCC s2hk converts
      to Traditional HK to match the other three models' convention.
    - Extra outputs: inline emotion (<|HAPPY|> etc.) and audio-event tags are
      extracted and stored in the result 'metadata' field (not in 'text'), so
      downstream nodes can optionally consume them.
    - Confidence: no calibrated logprob — defaults to 1.0 for non-empty text
      (same nominal treatment as Qwen3-ASR).

    Dependency note: funasr + modelscope + opencc-python-reimplemented must be
    installed in the same venv BEFORE running this worker:
        uv pip install opencc-python-reimplemented funasr modelscope
    (NOT uv sync — that would prune CUDA torch.)
    """

    # Regex patterns compiled once at class level.
    import re as _re
    _TAG_RE   = _re.compile(r"<\|[^|]+\|>")
    _EMOTION_RE = _re.compile(
        r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|DISGUSTED|FEARFUL|SURPRISED|EMO_UNKNOWN)\|>",
        _re.IGNORECASE,
    )
    _EVENT_RE = _re.compile(
        r"<\|(Speech|BGM|Applause|Laughter|Cry|Cough|Sneeze|Breath|Music|UNKNOWN)\|>",
        _re.IGNORECASE,
    )
    _LANG_RE  = _re.compile(r"<\|(zh|en|yue|ja|ko|nospeech)\|>", _re.IGNORECASE)

    def __init__(self, device: str, model_key: str, *, mem_fraction: float | None = None,
                 fp16: bool = True) -> None:
        self.model_key = model_key
        self.cfg = ASR_MODELS[model_key]
        super().__init__(device, mem_fraction=mem_fraction, fp16=fp16)
        # OpenCC converter: Simplified → Traditional HK (initialised in subprocess).
        self._cc = None

    def load_model(self):
        import sys
        import torch
        # Redirect stdout to stderr to prevent funasr/modelscope initialization logs
        # from corrupting the supervisor-worker JSONL stdout protocol.
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            try:
                from funasr import AutoModel
            except ImportError as e:
                raise ImportError(
                    "funasr is not installed. Run: "
                    "uv pip install opencc-python-reimplemented funasr modelscope"
                ) from e
            try:
                from opencc import OpenCC
                self._cc = OpenCC("s2hk")   # Simplified → Traditional HK
            except ImportError:
                log.warning(
                    "opencc-python-reimplemented not installed — SenseVoice output will "
                    "remain in Simplified Chinese. Install: uv pip install opencc-python-reimplemented"
                )
                self._cc = None

            log.info(f"Loading ASR model: {self.cfg['description']} on {self.device}")
            return AutoModel(
                model=self.cfg["id"],
                trust_remote_code=True,
                device=str(self.device),
                disable_update=True,    # skip ModelScope version-check network call
            )
        finally:
            sys.stdout = old_stdout

    def _parse_raw(self, raw_text: str) -> tuple[str, str, str]:
        """Strip inline tags; return (clean_text, emotion, audio_event)."""
        emotion_m = self._EMOTION_RE.search(raw_text)
        event_m   = self._EVENT_RE.search(raw_text)
        clean     = self._TAG_RE.sub("", raw_text).strip()
        if self._cc and clean:
            clean = self._cc.convert(clean)   # Simplified → Traditional HK
        emotion   = emotion_m.group(1).upper() if emotion_m else "UNKNOWN"
        event     = event_m.group(1)           if event_m   else "UNKNOWN"
        return clean, emotion, event

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        """Run SenseVoice inference on a batch of 16 kHz float32 arrays.

        funasr's AutoModel.generate(**cfg) routes to inference() whenever no
        vad_model is configured (our case) -- and inference() only reads a
        literal item-count `batch_size` kwarg (default 1), NOT `batch_size_s`
        (that duration-based param is exclusively consumed by
        inference_with_vad(), a different code path we never take; see
        funasr/auto/auto_model.py generate()/inference()/inference_with_vad()).
        Passing batch_size_s here was a no-op -- every item silently decoded
        one at a time regardless of how many we dispatched (confirmed live,
        2026-07-13: 100% of ~38k logged steps showed 'batch_size': '1').  Pass
        the literal batch_size instead, sized to this dispatch's own item
        count, so funasr's inference() loop does exactly one real batched
        forward pass over the whole chunk instead of len(items) separate ones.
        """
        lang = self.cfg["lang"] or "auto"
        audio_inputs = [(y16, ASR_SR) for y16 in items]
        try:
            results = self.model.generate(
                input=[a for a, _ in audio_inputs],
                language=lang,
                use_itn=True,           # inverse text normalisation
                batch_size=len(items),  # real item-count batch (see docstring -- batch_size_s is a no-op here)
            )
        except Exception as e:
            # Return empty placeholder for every item so caller can handle gracefully.
            log.error(f"SenseVoice batch error: {e}")
            return [{"text": "", "confidence": 0.0, "metadata": {}} for _ in items]

        out = []
        for res in results:
            raw = res.get("text", "") if isinstance(res, dict) else getattr(res, "text", "")
            clean, emotion, event = self._parse_raw(raw)
            out.append({
                "text":       clean,
                "confidence": 1.0 if clean else 0.0,
                "metadata":   {"emotion": emotion, "audio_event": event},
            })
        return out


# Map backend key → worker class; used by worker_main() for dispatch.
WORKER_CLASSES: dict[str, type] = {
    "faster_whisper": TranscribeWorker,
    "qwen_asr":       Qwen3ASRWorker,
    "sense_voice":    SenseVoiceWorker,
}


def worker_main() -> None:
    """Worker-subprocess entrypoint — a 3-stage threaded pipeline (2026-07-14).

    Before this date the loop was strictly sequential per task: decode+resample
    ALL items, then run the GPU forward, then read the next task — so the GPU
    idled during CPU preprocessing and vice versa.  Once sense_voice's forward
    pass was properly batched (2026-07-13 batch_size fix), preprocessing became
    the measured wall-clock bottleneck (cold-cache FLAC decode ~76 ms/file vs a
    ~0.55 s forward for a whole batch of 64).  Standard producer-consumer fix
    (the same pattern as torch DataLoader prefetch / tf.data.prefetch):

        [stdin reader thread] -> raw_q -> [preprocess thread + IO pool] -> ready_q -> [main thread: GPU + emit]

    Bounded queues (maxsize=2) give backpressure; the supervisor keeps up to
    --prefetch tasks in flight (see run_asr_transcribe) so the preprocess stage
    always has a next task to decode while the GPU runs the current one.  Only
    the main thread writes to stdout, so result JSONL lines never interleave.
    Results are emitted in task order (both queues are FIFO), but the
    supervisor matches on task_id anyway.
    """
    import queue as queue_mod
    import threading

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-key", required=True, choices=list(ASR_MODELS.keys()))
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--io-workers", type=int, default=16)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    backend = ASR_MODELS[args.model_key]["backend"]
    worker_cls = WORKER_CLASSES[backend]
    worker = worker_cls(args.device, args.model_key, mem_fraction=args.mem_fraction)

    def emit(msg: dict) -> None:
        # Called from the main thread only (before the pipeline threads start,
        # and then from the GPU loop) — single writer, no interleaving.
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "asr.transcribe", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)
    raw_q: queue_mod.Queue = queue_mod.Queue(maxsize=2)
    ready_q: queue_mod.Queue = queue_mod.Queue(maxsize=2)

    def reader_loop() -> None:
        """stdin → raw_q. A None sentinel (on shutdown message or stdin EOF)
        flows through both queues so each stage drains in-flight work before
        exiting."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg["type"] == "shutdown":
                break
            if msg["type"] != "task":
                continue
            raw_q.put(msg)
        raw_q.put(None)

    def preprocess_loop() -> None:
        """raw_q → decode+resample (GIL-releasing sf/soxr on the IO pool) → ready_q."""
        while True:
            msg = raw_q.get()
            if msg is None:
                ready_q.put(None)
                return
            task_id, items = msg["task_id"], msg["items"]
            t0 = time.time()
            try:
                paths = [it["path"] for it in items]
                wavs = list(ex.map(_load_and_resample, paths))
                keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) >= ASR_SR // 10]
                skipped_ids = [items[i]["id"] for i in range(len(items)) if i not in set(keep_idx)]
                kept_wavs = [wavs[i] for i in keep_idx]
                ready_q.put(("task", task_id, items, kept_wavs, keep_idx, skipped_ids, t0))
            except Exception as e:
                ready_q.put(("error", task_id, str(e)))

    threading.Thread(target=reader_loop, daemon=True, name="stdin-reader").start()
    threading.Thread(target=preprocess_loop, daemon=True, name="preprocess").start()

    # Main thread: GPU inference + stdout emit.
    while True:
        entry = ready_q.get()
        if entry is None:
            break
        if entry[0] == "error":
            _, task_id, err = entry
            emit({"type": "error", "task_id": task_id, "error": err, "retryable": True})
            continue
        _, task_id, items, kept_wavs, keep_idx, skipped_ids, t0 = entry
        try:
            if not keep_idx:
                emit({"type": "result", "task_id": task_id, "rows": [],
                      "skipped_ids": skipped_ids, "metrics": {"items_s": 0.0}})
                continue
            results = worker.infer_with_oom_halving(kept_wavs)
            rows = []
            for i, res in zip(keep_idx, results):
                rows.append({"id": items[i]["id"], **res})
            elapsed = time.time() - t0
            emit({"type": "result", "task_id": task_id, "rows": rows, "skipped_ids": skipped_ids,
                  "metrics": {"items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0}})
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})

    ex.shutdown(wait=False)


if __name__ == "__main__":
    worker_main()
