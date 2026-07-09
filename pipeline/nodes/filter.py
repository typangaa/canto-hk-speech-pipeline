"""
pipeline/nodes/filter.py
filter.text / filter.acoustic / filter.decide DAG nodes — ported from
scripts/06_filter.py onto the orchestrator's item-level pipelining.

Split into three nodes instead of one script, mirroring the asr.transcribe /
asr.agreement split (pipeline/nodes/asr.py):

  filter.text     CPU, in-supervisor (no subprocess) — sample_rate/duration hard
                  gates (catalog columns, no audio read) + CJK-length / English-ratio /
                  Mandarin-ratio text gates (pure Python on ASR text, no audio read).
                  Writes filters_text.
  filter.acoustic CPU worker-subprocess pool (mirrors label_prosody.py's N-CPU-worker
                  pattern) — SNR + DNSMOS, both requiring an actual audio decode.
                  Discovery only picks up ids where filters_text.pass = TRUE, so
                  text-rejected segments never pay for a DNSMOS pass — the same
                  short-circuit scripts/06_filter.py's --use-pregate flag approximated
                  via a separate shard-parallel hack, done here as a plain item-level
                  dependency (no stage barrier: segment A can reach filter.acoustic the
                  moment ITS OWN filter.text finishes; segment B doesn't wait on it).
                  Writes filters_acoustic.
  filter.decide   CPU, in-supervisor — merges filters_text + filters_acoustic into the
                  final `filters` table (pass / fail_reason + the merged numeric
                  fields), the one this node's discover() requires JOIN filters_text ON
                  pass and every downstream node (g2p, tier.assign, manifest.build)
                  reads from.

Three separate raw tables (filters_text / filters_acoustic) feeding one merged
`filters` table, rather than two nodes each upserting a different partial-column
subset of `filters` directly, because upsert_rows() does INSERT OR REPLACE — which
resets any column NOT in a given write's column list. Two nodes partial-writing the
same table would clobber each other's columns on alternating runs. filter.decide is
the only writer of the merged table's post-migration rows (see schema.sql comments —
the 455,299 legacy-imported `filters` rows predate this split and are never touched
by these nodes' discovery, since they have no corresponding filters_text row).

`filters.provenance` note: every one of the 455,299 legacy-imported segments already
has a `filters` row (manifest.jsonl only ever contained already-passing segments), so
a bare row-existence anti-join in filter.decide's discovery would find zero
"undecided" work forever. filter.decide's INSERT OR REPLACE tags its own rows
`provenance = 'filter_decide'`, and its discovery LEFT JOINs on that exact value —
legacy rows (`provenance IS NULL`) correctly read as "not yet decided by this node".

DNSMOS metric note (DECISIONS.md 2026-06-09): the filter gate uses `sig_mos` (speech
clarity), not `ovrl_mos` (overall — penalises RTHK's background music/ambience too
harshly). `dnsmos_ovrl` is still stored for reference. Verbatim continuation of
scripts/06_filter.py's override of the metric named in docs/QUALITY_SPEC.md.

ORT thread-capping note: scripts/06_filter.py capped onnxruntime's default
"physical-core-count" intra-op thread pool by monkeypatching `onnxruntime.
InferenceSession` at import time (so M parallel shard processes don't spawn M×47
threads on 48 cores). This node instead builds its own capped `ort.SessionOptions`
once per worker process and constructs the DNSMOS ONNX sessions directly (see
`_build_capped_dnsmos()`), reusing speechmos.dnsmos.DNSMOS's own (otherwise
unmodified) `audio_melspec` / `get_polyfit_val` / `__call__` methods — no monkeypatch,
same numeric behaviour, and the sessions are built once in AcousticWorker.load_model()
(GPUWorkerBase's established one-time-load convention) rather than re-touched per call.

Golden-parity note: filter.acoustic's audio load goes through pipeline/audio/bus.py's
decode() (matching the P2 decode-once convention every other CPU/GPU node in this
package uses), but only for the 48 kHz read — since segments are already 48 kHz masters
(hard constraint #6), decode(path, 48000) hits bus.py's zero-cost passthrough branch
(`orig_sr == target_sr`) and never invokes its soxr resampler, so this is byte-identical
to a raw sf.read(). The DNSMOS-specific 16 kHz downsample is NOT routed through bus.py
though — it uses torchaudio.transforms.Resample(48000, 16000), verbatim matching
scripts/06_filter.py's compute_dnsmos(), because this numeric output is held to the
|Δ|≤1e-4 golden-parity tolerance (REARCHITECTURE_IMPLEMENTATION_PLAN.md §9.1) and a
different resampler is not a safe substitution for a value-sensitive neural MOS model
(the same reasoning pipeline/nodes/asr.py's module docstring gives for keeping its own
resample_poly instead of bus.py's soxr path).
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pipeline.audio.bus import decode
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

TARGET_SR = 48000
DNSMOS_SR = 16000

# Thresholds — verbatim from scripts/06_filter.py / docs/QUALITY_SPEC.md.
MIN_DUR = 3.0
MAX_DUR = 20.0
MIN_DNSMOS = 3.0  # applied to sig_mos (speech clarity), not ovrl_mos — see DECISIONS.md
MIN_SNR_DB = 25.0
MAX_ENG_RATIO = 0.30
MAX_MAN_RATIO = 0.15
MIN_CJK_CHARS = 5
MAX_TEXT_CHARS = 150

CANTONESE_CHARS = set("係冇佢呢嗰嚟咁嘅囉喎啩咋啦啲喺唔咗嘢搵睇啱攞唞攰𠻹諗㗎喇乜哋俾俾瞓掟喐踎揸揼黐搣𠮶叻咪咩噏嘥嚿搲氹")

# Only characters exclusive to simplified Chinese — never appear in traditional/Cantonese text.
# Conservative list to minimise false positives. Verified against HK corpus samples.
SIMPLIFIED_CHARS = set(
    "这们说时会对为当东乐车书无写听见让应义证认识双农"
    "发动关变质务设机飞门产类总积带济战县观讲谁课谢语译诉读误"
    "岁圆块坏处复万党兴击划创协卖卫归录弥弯弹态忆忧怀"
    "恳恶惊惧惨惩惭惮惯愤愿懒戏户执扩扫扬扭扮扰护报担"
    "拟拢拣拥拦拨择挥挤损换搅携摆摇撑敌数斩显"
)

TRADITIONAL_MANDARIN_INDICATORS = set("是的他她們這那說沒誰吃喝看哪怎")

CANTONESE_WORDS = [
    "而家", "唔係", "點解", "幾時", "即係", "邊度", "乜嘢", "噉樣", "一齊",
    "係咪", "真係", "已經", "好似", "先至", "仲有", "唔使", "或者", "點樣",
    "緊要", "鐘意", "鍾意", "返去", "企喺", "呢個", "呢啲", "嗰個", "嗰啲",
    "佢哋", "我哋", "你哋", "話俾", "話畀", "收聲", "話之", "企定", "睇吓",
    "玩嘢", "諗住", "講嘢", "喇喎", "㗎啦", "㗎喇", "唔好", "咪話", "冇人"
]

MANDARIN_WORDS = [
    "現在", "现在", "為什麼", "为什么", "什麼", "什么", "時候", "时候",
    "這樣", "这样", "我們", "我们", "你們", "你们", "他們", "他们",
    "她們", "她们", "是不是", "真的", "怎麼", "怎么", "怎麼樣", "怎么样",
    "那樣", "那样", "這個", "这个", "這些", "这些", "那個", "那个",
    "那些", "一起", "哪裡", "哪里", "不要", "去哪", "告訴", "告诉", "那是",
    "這是", "这是", "就是", "特別是", "特别是", "什麼的", "什么的", "昨天",
    "明天", "吃過", "吃过", "看過", "看过", "去過", "去过", "走了", "來了",
    "来了", "對不起", "对不起", "謝謝", "谢谢", "剛才", "刚才"
]

_NEUTRAL_WORDS = [
    "的士", "的確", "目的", "了解", "除了", "不得了", "受不了", "甚至乎",
    "著名", "著作", "著急", "著手", "說法", "說明", "小說", "話說",
    "可是", "是否", "總是", "要是", "就是", "國是", "是非", "若是", "看法"
]


# ---------------------------------------------------------------------------
# Pure text-logic functions — verbatim port of scripts/06_filter.py (byte-for-byte
# same classification rules; tested directly in tests/test_filter_node.py).
# ---------------------------------------------------------------------------

def is_cjk(c: str) -> bool:
    code = ord(c)
    return (
        (0x4E00 <= code <= 0x9FFF) or
        (0x3400 <= code <= 0x4DBF) or
        (0x20000 <= code <= 0x2A6DF) or
        (0x2A700 <= code <= 0x2B73F) or
        (0x2B740 <= code <= 0x2B81F) or
        (0x2B820 <= code <= 0x2CEAF) or
        (0x2CEB0 <= code <= 0x2EBF0) or
        (0x30000 <= code <= 0x3134F) or
        (0x31350 <= code <= 0x323AF) or
        (0xF900 <= code <= 0xFAFF)
    )


def get_english_and_cjk_tokens(text: str) -> tuple[list[str], list[str]]:
    english_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    cjk_chars = [c for c in text if is_cjk(c)]
    return english_words, cjk_chars


def english_ratio(text: str) -> float:
    english_words, cjk_chars = get_english_and_cjk_tokens(text)
    total_words = len(english_words) + len(cjk_chars)
    if not total_words:
        return 0.0
    return round(len(english_words) / total_words, 3)


def mandarin_ratio(text: str) -> float:
    cjk = [c for c in text if is_cjk(c)]
    if not cjk:
        return 0.0

    num_simplified = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)

    text_clean = text
    for w in _NEUTRAL_WORDS:
        text_clean = text_clean.replace(w, "")
    cjk_clean = [c for c in text_clean if is_cjk(c)]

    mando_chars = sum(1 for c in cjk_clean if c in TRADITIONAL_MANDARIN_INDICATORS)
    canto_chars = sum(1 for c in cjk if c in CANTONESE_CHARS)

    mando_words_count = sum(text.count(w) for w in MANDARIN_WORDS)
    canto_words_count = sum(text.count(w) for w in CANTONESE_WORDS)

    mando_score = num_simplified * 2.0 + mando_chars * 1.5 + mando_words_count * 2.5
    canto_score = canto_chars * 1.5 + canto_words_count * 2.5

    if canto_score > 0 and mando_score > 0:
        total_mando_features = num_simplified + mando_chars
        if mando_score >= 0.3 * canto_score:
            ratio = max(0.16, total_mando_features / len(cjk))
            return round(min(ratio, 1.0), 3)
        return round(total_mando_features / len(cjk), 3)
    elif canto_score == 0:
        if mando_score > 0:
            total_mando_chars = num_simplified + mando_chars + sum(len(w) for w in MANDARIN_WORDS if w in text)
            ratio = max(0.16, total_mando_chars / len(cjk))
            return round(min(ratio, 1.0), 3)
        return 0.0
    else:
        return 0.0


def cjk_count(text: str) -> int:
    return sum(1 for c in text if is_cjk(c))


def detect_language(text: str) -> tuple[str, float]:
    eng_ratio = english_ratio(text)
    if eng_ratio >= 0.85:
        return "eng", eng_ratio

    cjk = [c for c in text if is_cjk(c)]
    if not cjk:
        return "eng", eng_ratio

    num_simplified = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)

    text_clean = text
    for w in _NEUTRAL_WORDS:
        text_clean = text_clean.replace(w, "")
    cjk_clean = [c for c in text_clean if is_cjk(c)]

    mando_chars = sum(1 for c in cjk_clean if c in TRADITIONAL_MANDARIN_INDICATORS)
    canto_chars = sum(1 for c in cjk if c in CANTONESE_CHARS)

    mando_words_count = sum(text.count(w) for w in MANDARIN_WORDS)
    canto_words_count = sum(text.count(w) for w in CANTONESE_WORDS)

    mando_score = num_simplified * 2.0 + mando_chars * 1.5 + mando_words_count * 2.5
    canto_score = canto_chars * 1.5 + canto_words_count * 2.5

    if eng_ratio >= 0.35:
        return "mixed", round(eng_ratio, 2)

    if canto_score > 0 and mando_score > 0:
        if mando_score >= 0.3 * canto_score and canto_score >= 0.3 * mando_score:
            return "mixed", 0.80
        elif canto_score > mando_score:
            confidence = round(canto_score / (canto_score + mando_score), 2)
            return "yue", confidence
        else:
            confidence = round(mando_score / (canto_score + mando_score), 2)
            return "cmn", confidence

    if canto_score > 0:
        confidence = round(min(1.0, 0.5 + canto_score * 0.1), 2)
        return "yue", confidence

    if mando_score > 0:
        confidence = round(min(1.0, 0.5 + mando_score * 0.1), 2)
        return "cmn", confidence

    return "yue", 0.50


def compute_snr(wav: np.ndarray) -> float:
    """Frame-energy SNR estimate — verbatim port of scripts/06_filter.py."""
    frame_len = int(TARGET_SR * 0.025)
    energies = [np.sum(wav[i:i + frame_len] ** 2) for i in range(0, len(wav) - frame_len, frame_len)]
    if not energies:
        return 0.0
    energies.sort()
    signal_e = np.mean(energies[int(0.9 * len(energies)):]) + 1e-10
    noise_e = np.mean(energies[:int(0.1 * len(energies))]) + 1e-10
    # float() cast: np.log10/np.mean on a Python list of np.float32 sums yields a
    # numpy scalar, which json.dumps() (worker stdio protocol) cannot serialise.
    return round(float(10 * np.log10(signal_e / noise_e)), 1)


# ---------------------------------------------------------------------------
# filter.text — pure logic, no audio, no worker subprocess.
# ---------------------------------------------------------------------------

TEXT_DISCOVER_SQL = """
    SELECT s.id, s.duration_sec, s.sample_rate, a.best_text
    FROM segments s
    JOIN asr_agreement a ON s.id = a.id
    LEFT JOIN filters_text ft ON s.id = ft.id
    WHERE ft.id IS NULL
    ORDER BY s.duration_sec
"""


def discover_text(conn) -> list[tuple]:
    return conn.execute(TEXT_DISCOVER_SQL).fetchall()


def evaluate_text(duration_sec: float | None, sample_rate: int | None, text: str | None) -> dict:
    """Gates 0-4 of scripts/06_filter.py's filter_segment(): sample_rate, duration,
    text length, English ratio, Mandarin ratio — first failing gate wins, matching
    the legacy cascading-return order exactly."""
    text = text or ""
    eng = english_ratio(text)
    man = mandarin_ratio(text)
    lang, lang_conf = detect_language(text) if text.strip() else ("eng", 0.0)

    fail_reason = None
    if sample_rate != TARGET_SR:
        fail_reason = "sample_rate"
    elif not (MIN_DUR <= (duration_sec or 0.0) <= MAX_DUR):
        fail_reason = "duration"
    else:
        n_cjk = cjk_count(re.sub(r"[^\w\s]", "", text))
        if n_cjk < MIN_CJK_CHARS:
            fail_reason = "text_too_short"
        elif len(text) > MAX_TEXT_CHARS:
            fail_reason = "text_too_long"
        elif eng > MAX_ENG_RATIO:
            fail_reason = "english_ratio"
        elif man > MAX_MAN_RATIO:
            fail_reason = "mandarin_ratio"

    return {
        "english_ratio": eng,
        "mandarin_ratio": man,
        "detected_language": lang,
        "language_confidence": lang_conf,
        "pass": fail_reason is None,
        "fail_reason": fail_reason,
    }


async def run_filter_text(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run filter.text` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover_text(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"filter.text: {len(rows)} segments to evaluate")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("filter.text")
    processed = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        out_rows = [
            {"id": seg_id, **evaluate_text(duration_sec, sample_rate, text)}
            for seg_id, duration_sec, sample_rate, text in batch
        ]
        upsert_rows(conn, "filters_text", out_rows, ["id"])
        record_batch(conn, run_id, "filter.text", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} evaluated ({rate:.1f}/s)")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} evaluated in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": 0, "run_id": run_id}


# ---------------------------------------------------------------------------
# filter.acoustic — CPU worker-subprocess pool (mirrors label_prosody.py).
# ---------------------------------------------------------------------------

ACOUSTIC_DISCOVER_SQL = """
    SELECT s.id, s.audio_path, s.duration_sec
    FROM segments s
    JOIN filters_text ft ON s.id = ft.id AND ft.pass = TRUE
    LEFT JOIN filters_acoustic fa ON s.id = fa.id
    WHERE fa.id IS NULL
    ORDER BY s.duration_sec
"""


def discover_acoustic(conn) -> list[tuple]:
    return conn.execute(ACOUSTIC_DISCOVER_SQL).fetchall()


def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


async def run_filter_acoustic(
    *,
    conn=None,
    n_workers: int = 4,
    threads_per_worker: int = 4,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    """Supervisor entrypoint for filter.acoustic. Spawns n_workers CPU worker
    subprocesses (each with its own capped-thread ONNX sessions), dispatches
    length-sorted batches round-robin, commits through journal + upsert_rows —
    same idiom as label_prosody.run_label_prosody.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (DuckDB's write lock
    is per-process, not per-transaction; a cursor on a shared connection is a
    transparent drop-in for upsert_rows/record_batch). Defaults to a fresh
    self-managed connect() for standalone `pipe run filter.acoustic` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.worker import spawn_worker

    conn = conn or connect()
    rows = discover_acoustic(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"filter.acoustic: {len(rows)} segments to process")
    if not rows:
        return {"processed": 0, "errors": 0}

    registry = PoolRegistry()
    pool_names = [f"cpu.{i}" for i in range(n_workers)]
    for name in pool_names:
        registry.register(name, target=1)

    # numpy/torch/torchaudio link against OpenBLAS/MKL, which size their own
    # native thread pools from these env vars at library-load time — read
    # *before* any Python-level torch.set_num_threads() call can take effect,
    # and independent of onnxruntime's own (already-capped) intra_op setting.
    # Left unset they default to os.cpu_count(), so n_workers subprocesses each
    # start their own ~nproc-sized pool: observed 129 OS threads/process and a
    # 48-core box driven to a load average of 131 with just 8 workers, combined
    # throughput no better than one process. Cap them per worker subprocess.
    worker_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    handles = {}
    for pool_name in pool_names:
        cmd = [
            sys.executable, "-m", "pipeline.nodes.filter",
            "--threads", str(threads_per_worker),
        ]
        handle = await spawn_worker(cmd, env=worker_env)
        await handle.wait_ready(timeout=120.0)
        handles[pool_name] = handle
        log.info(f"worker ready: {pool_name} (pid={handle.pid})")

    run_id = new_run_id("filter.acoustic")
    queue: asyncio.Queue = asyncio.Queue()
    for batch in _batches(rows, batch_size):
        queue.put_nowait(batch)

    processed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(pool_name: str, handle) -> None:
        nonlocal processed, errors
        pool = registry.get(pool_name)
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            items = [{"id": r[0], "path": r[1]} for r in batch]
            async with pool.acquire():
                await handle.send_task(f"{pool_name}-{processed}", items)
                try:
                    result = await handle.read_message(timeout=180.0)
                except Exception as e:
                    log.error(f"{pool_name}: batch failed: {e}")
                    errors += len(batch)
                    queue.task_done()
                    continue
            if result["type"] == "error":
                log.error(f"{pool_name}: worker error: {result['error']}")
                errors += len(batch)
                queue.task_done()
                continue

            out_rows = [{"id": r["id"], **{k: r[k] for k in ("snr_db", "dnsmos_sig", "dnsmos_ovrl", "pass", "fail_reason")}}
                        for r in result["rows"]]
            skipped_rows = [
                {"id": sid, "snr_db": None, "dnsmos_sig": None, "dnsmos_ovrl": None,
                 "pass": False, "fail_reason": "unreadable_audio"}
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{pool_name}: {len(skipped_rows)} unreadable segment(s): "
                            f"{[r['id'] for r in skipped_rows][:5]}")

            upsert_rows(conn, "filters_acoustic", out_rows + skipped_rows, ["id"])
            record_batch(conn, run_id, "filter.acoustic", [r["id"] for r in out_rows], "ok",
                         metrics=result.get("metrics"))
            if skipped_rows:
                record_batch(conn, run_id, "filter.acoustic", [r["id"] for r in skipped_rows],
                             "error", error="unreadable audio file")

            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} processed ({rate:.1f}/s)")

    await asyncio.gather(*(
        worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
    ))

    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} processed, {errors} errors in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id}


def _build_capped_dnsmos(threads: int):
    """Build a speechmos.dnsmos.DNSMOS instance whose two ONNX sessions are capped
    to *threads* intra-op threads, without monkeypatching onnxruntime.InferenceSession
    (see module docstring). Reuses DNSMOS's own audio_melspec/get_polyfit_val/__call__
    unmodified — only session construction differs."""
    import onnxruntime as ort
    import speechmos.dnsmos as _dnsmos_mod

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1

    models_dir = Path(_dnsmos_mod.__file__).resolve().parent / "dnsmos_models"
    primary_path = str(models_dir / "sig_bak_ovr.onnx")
    p808_path = str(models_dir / "model_v8.onnx")

    instance = _dnsmos_mod.DNSMOS.__new__(_dnsmos_mod.DNSMOS)
    instance.primary_model_path = primary_path
    instance.onnx_sess = ort.InferenceSession(primary_path, sess_options=so)
    instance.p808_onnx_sess = ort.InferenceSession(p808_path, sess_options=so)
    return instance


def compute_dnsmos(wav48: np.ndarray, dnsmos_instance) -> tuple[float, float]:
    """Returns (sig_mos, ovrl_mos). Verbatim port of scripts/06_filter.py's
    compute_dnsmos(), routed through a pre-built capped DNSMOS instance instead of
    speechmos.dnsmos.run()'s module-global cache."""
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(TARGET_SR, DNSMOS_SR)
    wav16 = resampler(t).squeeze(0).numpy()
    peak = np.abs(wav16).max()
    if peak > 1.0:
        wav16 = wav16 / peak
    result = dnsmos_instance(wav16, DNSMOS_SR, False)
    sig = float(result.get("sig_mos", 0.0))
    ovrl = float(result.get("ovrl_mos", 0.0))
    assert 1.0 <= sig <= 5.0, f"DNSMOS sig_mos out of range: {sig}"
    return round(sig, 2), round(ovrl, 2)


class AcousticWorker(GPUWorkerBase):
    def __init__(self, threads: int = 4) -> None:
        self.threads = threads
        super().__init__("cpu", fp16=False)

    def load_model(self):
        log.info(f"Loading DNSMOS ONNX sessions (intra_op_num_threads={self.threads}) ...")
        return _build_capped_dnsmos(self.threads)

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        return [self._acoustic_one(wav48) for wav48 in items]

    def _acoustic_one(self, wav48: np.ndarray) -> dict:
        snr = compute_snr(wav48)
        if snr < MIN_SNR_DB:
            return {"snr_db": snr, "dnsmos_sig": None, "dnsmos_ovrl": None,
                    "pass": False, "fail_reason": "snr"}
        try:
            sig_mos, ovrl_mos = compute_dnsmos(wav48, self.model)
        except Exception as exc:
            log.warning(f"DNSMOS failed on a clip: {exc}")
            return {"snr_db": snr, "dnsmos_sig": None, "dnsmos_ovrl": None,
                    "pass": False, "fail_reason": "dnsmos_error"}
        if sig_mos < MIN_DNSMOS:
            return {"snr_db": snr, "dnsmos_sig": sig_mos, "dnsmos_ovrl": ovrl_mos,
                    "pass": False, "fail_reason": "dnsmos"}
        return {"snr_db": snr, "dnsmos_sig": sig_mos, "dnsmos_ovrl": ovrl_mos,
                "pass": True, "fail_reason": None}


def worker_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                     help="onnxruntime intra_op_num_threads cap per worker process")
    ap.add_argument("--io-workers", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    # torch defaults to a large intra-op thread pool (e.g. 24 on a 48-core box)
    # for the torchaudio.transforms.Resample call in compute_dnsmos(), on top of
    # the onnxruntime sessions already capped above. Uncapped, N worker processes
    # each spin up ~dozens of torch threads regardless of --threads, so running
    # several workers oversubscribes the machine (observed: 8 workers -> ~130
    # OS threads/process, load average > 2x core count, combined throughput no
    # better than a single process). Cap it the same way onnxruntime is capped.
    torch.set_num_threads(1)

    worker = AcousticWorker(threads=args.threads)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "filter.acoustic", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg["type"] == "shutdown":
            break
        if msg["type"] != "task":
            continue

        task_id = msg["task_id"]
        items = msg["items"]
        t0 = time.time()
        try:
            paths = [it["path"] for it in items]
            wavs = list(ex.map(lambda p: decode(p, TARGET_SR), paths))
            keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) > 0]
            skipped_ids = [items[i]["id"] for i in range(len(items)) if i not in set(keep_idx)]
            if not keep_idx:
                emit({"type": "result", "task_id": task_id, "rows": [],
                      "skipped_ids": skipped_ids, "metrics": {"items_s": 0.0}})
                continue
            kept_wavs = [wavs[i] for i in keep_idx]
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


# ---------------------------------------------------------------------------
# filter.decide — pure aggregation, no audio, no worker subprocess.
# ---------------------------------------------------------------------------

DECIDE_DISCOVER_SQL = """
    SELECT ft.id, ft.pass, ft.fail_reason, ft.english_ratio, ft.mandarin_ratio,
           ft.detected_language, ft.language_confidence,
           fa.pass, fa.fail_reason, fa.snr_db, fa.dnsmos_sig, fa.dnsmos_ovrl
    FROM filters_text ft
    LEFT JOIN filters_acoustic fa ON ft.id = fa.id
    LEFT JOIN filters f ON ft.id = f.id AND f.provenance = 'filter_decide'
    WHERE f.id IS NULL
      AND (ft.pass = FALSE OR fa.id IS NOT NULL)
"""


def discover_decide(conn) -> list[tuple]:
    return conn.execute(DECIDE_DISCOVER_SQL).fetchall()


def decide_row(
    seg_id: str, text_pass: bool, text_reason: str | None,
    eng: float, man: float, lang: str, lang_conf: float,
    ac_pass: bool | None, ac_reason: str | None,
    snr: float | None, dnsmos_sig: float | None, dnsmos_ovrl: float | None,
) -> dict:
    """Merges filters_text + filters_acoustic into the final pass/fail_reason.
    Text gates always take priority (matches scripts/06_filter.py's cascading
    gate order — text gates run before acoustic ones); ac_pass is None only if
    filter.acoustic hasn't run yet, which discover_decide() already excludes for
    text_pass=True rows, so the "acoustic_pending" branch is a defensive guard,
    not an expected path."""
    if not text_pass:
        final_pass, reason = False, text_reason
    elif ac_pass is None:
        final_pass, reason = False, "acoustic_pending"
    elif not ac_pass:
        final_pass, reason = False, ac_reason
    else:
        final_pass, reason = True, None

    return {
        "id": seg_id,
        "snr_db": snr,
        "dnsmos": dnsmos_sig,
        "dnsmos_ovrl": dnsmos_ovrl,
        "english_ratio": eng,
        "mandarin_ratio": man,
        "detected_language": lang,
        "language_confidence": lang_conf,
        "pass": final_pass,
        "fail_reason": reason,
        "provenance": "filter_decide",
    }


async def run_filter_decide(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run filter.decide` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover_decide(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"filter.decide: {len(rows)} segments to decide")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("filter.decide")
    processed = 0
    passed = 0
    t0 = time.time()

    # Wrap all batches in a single transaction so DuckDB only performs one WAL
    # checkpoint flush at the end rather than one per batch.  For a pure
    # in-memory aggregation step like filter.decide (no audio, no workers),
    # this cuts wall-clock time from O(n²) to O(n) on large catalogs.
    conn.begin()
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            out_rows = [decide_row(*r) for r in batch]
            upsert_rows(conn, "filters", out_rows, ["id"])
            record_batch(conn, run_id, "filter.decide", [r["id"] for r in out_rows], "ok")
            processed += len(out_rows)
            passed += sum(1 for r in out_rows if r["pass"])
            rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"{processed}/{len(rows)} decided ({rate:.1f}/s), pass_rate={passed / processed:.3f}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} decided ({passed} passed, {passed / max(processed, 1):.1%}) "
             f"in {elapsed:.0f}s, run_id={run_id}")
    return {"processed": processed, "passed": passed, "errors": 0, "run_id": run_id}


if __name__ == "__main__":
    worker_main()
