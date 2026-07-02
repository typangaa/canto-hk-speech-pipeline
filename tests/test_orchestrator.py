"""P1 gate tests — pool target semantics + kill-9 resume idempotency.
Run: pytest tests/test_orchestrator.py -v
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import pytest

from pipeline.catalog.catalog import connect_ro
from pipeline.config import CATALOG_PATH
from pipeline.orchestrator.pools import PoolRegistry


def test_pool_target_gates_concurrency():
    async def scenario():
        registry = PoolRegistry()
        pool = registry.register("gpu.0", target=1)

        held = []

        async def holder(tag):
            async with pool.acquire():
                held.append(tag)
                await asyncio.sleep(0.2)
            held.remove(tag)

        t1 = asyncio.create_task(holder("a"))
        await asyncio.sleep(0.05)
        assert pool.in_use == 1

        # Second acquirer must wait — target is still 1.
        t2 = asyncio.create_task(holder("b"))
        await asyncio.sleep(0.05)
        assert pool.waiting == 1
        assert held == ["a"]

        await asyncio.gather(t1, t2)
        assert pool.in_use == 0

    asyncio.run(scenario())


def test_pool_set_target_zero_blocks_new_acquires():
    async def scenario():
        registry = PoolRegistry()
        pool = registry.register("gpu.0", target=1)
        pool.set_target(0)
        await asyncio.sleep(0.05)  # set_target schedules via ensure_future

        acquired = asyncio.Event()

        async def blocked():
            async with pool.acquire():
                acquired.set()

        t = asyncio.create_task(blocked())
        await asyncio.sleep(0.1)
        assert not acquired.is_set()
        assert pool.waiting == 1

        pool.set_target(1)
        await asyncio.wait_for(acquired.wait(), timeout=2.0)
        await t

    asyncio.run(scenario())


def _require_catalog():
    if not os.path.exists(CATALOG_PATH):
        pytest.skip("catalog not built — run: python -m pipeline.cli catalog build")


def test_label_music_kill_resume_no_duplicates():
    """Kill -9 mid-run must not double-write or corrupt labels_music: each
    committed batch is a single atomic upsert, and discover()'s anti-join
    re-computes remaining work from scratch on every run — no separate
    resume bookkeeping is needed for correctness.

    Each catalog connection here is opened and closed immediately (not held
    across the subprocess calls below) — DuckDB in this install refuses to
    open a read-write connection while ANY other connection, including a
    read-only one, still holds the file open (confirmed empirically; see
    docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §3.2 single-writer discipline).
    """
    _require_catalog()
    conn = connect_ro()
    before = conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    conn.close()

    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.cli", "run", "label.music",
         "--devices", "cpu", "--limit", "12", "--batch", "3",
         "--gpu-policy", "exempt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(6.0)  # let it load the model and process at least one batch
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    # Resume: same command, same --limit — discover() naturally excludes
    # whatever the killed run already committed.
    subprocess.run(
        [sys.executable, "-m", "pipeline.cli", "run", "label.music",
         "--devices", "cpu", "--limit", "12", "--batch", "3",
         "--gpu-policy", "exempt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=60, check=True,
    )

    conn = connect_ro()
    after = conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    dupes = conn.execute(
        "SELECT id, COUNT(*) FROM labels_music GROUP BY id HAVING COUNT(*) != 1"
    ).fetchall()
    conn.close()

    assert not dupes, f"duplicate ids in labels_music after kill+resume: {dupes[:5]}"
    assert after >= before, "labels_music row count must not decrease"


def test_label_prosody_kill_resume_no_duplicates():
    """Same kill -9 / resume idempotency contract as label.music, exercised
    against label.prosody's N-CPU-worker-process supervisor instead of the
    per-GPU-device one — proves the pattern generalises across the CPU/GPU
    pool axis, not just for label_music's specific pool-naming scheme.
    """
    _require_catalog()
    conn = connect_ro()
    before = conn.execute("SELECT COUNT(*) FROM labels_prosody").fetchone()[0]
    conn.close()

    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.cli", "run", "label.prosody",
         "--workers", "2", "--limit", "12", "--batch", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4.0)  # let workers load Silero VAD and process at least one batch
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    subprocess.run(
        [sys.executable, "-m", "pipeline.cli", "run", "label.prosody",
         "--workers", "2", "--limit", "12", "--batch", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=60, check=True,
    )

    conn = connect_ro()
    after = conn.execute("SELECT COUNT(*) FROM labels_prosody").fetchone()[0]
    dupes = conn.execute(
        "SELECT id, COUNT(*) FROM labels_prosody GROUP BY id HAVING COUNT(*) != 1"
    ).fetchall()
    conn.close()

    assert not dupes, f"duplicate ids in labels_prosody after kill+resume: {dupes[:5]}"
    assert after >= before, "labels_prosody row count must not decrease"


def test_label_suite_kill_resume_no_duplicates():
    """Same kill -9 / resume idempotency contract, exercised against label.suite's
    3-table (labels_lang / labels_overlap / labels_music) decode-once fan-out —
    each batch's per-table upserts commit together before the next batch is
    dispatched, so a kill mid-run leaves at most one batch's worth of tables
    partially ahead of each other, never duplicated or corrupted.
    """
    _require_catalog()
    conn = connect_ro()
    before = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("labels_lang", "labels_overlap", "labels_music")
    }
    conn.close()

    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.cli", "run", "label.suite",
         "--devices", "cpu", "--limit", "12", "--batch", "3",
         "--gpu-policy", "exempt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(8.0)  # 3 models to load (mms-lid + pyannote + PANNs) before first batch
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    subprocess.run(
        [sys.executable, "-m", "pipeline.cli", "run", "label.suite",
         "--devices", "cpu", "--limit", "12", "--batch", "3",
         "--gpu-policy", "exempt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=90, check=True,
    )

    conn = connect_ro()
    for t in ("labels_lang", "labels_overlap", "labels_music"):
        after = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        dupes = conn.execute(
            f"SELECT id, COUNT(*) FROM {t} GROUP BY id HAVING COUNT(*) != 1"
        ).fetchall()
        assert not dupes, f"duplicate ids in {t} after kill+resume: {dupes[:5]}"
        assert after >= before[t], f"{t} row count must not decrease"
    conn.close()
