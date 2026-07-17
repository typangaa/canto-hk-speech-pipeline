"""
pipeline/tools/calibrate_server.py
Local browser UI for human text-verification calibration. NOT a DAG node --
this is a long-running interactive server, not an idempotent-discovery batch
step. It reads the sample queue that `pipe run calibrate.sample` writes into
`calibration_review` and lets the owner listen to each segment, compare ASR
candidates, and record a verified/skipped/rejected/flagged decision. See
pipeline/nodes/calibrate.py's module docstring for why this exists (owner
decision 2026-07-10: text_verified/gold was structurally dead in the DAG).
A dedicated one-click "Mandarin" button (2026-07-15) submits a 'rejected'
decision with a fixed flag_reason ("mandarin", MANDARIN_FLAG_REASON) for
segments that surface for text QA but turn out to be non-HK-Cantonese --
'rejected' now also excludes the segment from the manifest (tiers.tier is
set to 'excluded', see record_decision() in pipeline/nodes/calibrate.py).

Usage:
    .venv/bin/python -m pipeline.tools.calibrate_server [--port 8420] [--batch <sample_batch id>]

Then open http://localhost:8420/ in a browser on the same machine.
`--batch` only pre-selects the batch-jump dropdown's default value -- the
browser UI can switch batch/source/order filters live via query params, it
is not fixed at server startup.

Connection lifetime note (2026-07-10 -- see DECISIONS.md): unlike every other
node, this process does NOT hold a DuckDB connection for its whole lifetime.
DuckDB's write lock is per-process and held for a connection's full life, so
a long-running interactive server that kept one open (the original design)
would block every other `pipe run` node / one-off maintenance script for the
entire review session -- hours at a time. Instead each HTTP request opens
its own short-lived connection (read-only for GETs, read-write for the two
writing endpoints) and closes it before the response is sent, so the write
lock is held for single-digit milliseconds per click, not the whole session.
Concurrent batch jobs can now interleave freely between clicks; the remaining
conflict window is a write landing at the exact moment another process's own
writer connection is open -- fine for a *brief* overlap (retried with a short
backoff below), but not for a long batch node (asr.transcribe, segment.diarize,
etc.) that holds the writer lock for its ENTIRE multi-hour run -- during that
window even connect_ro() is refused, so this per-request design alone doesn't
help; see the offline-mode note below.

Offline-mode note (2026-07-13 -- see DECISIONS.md): while the catalog is
unreachable this server falls back to two on-disk artifacts instead of
blocking or erroring out:
  - Reads (`/api/next`, `/api/item`, `/api/history`, `/api/batches`,
    `/api/sources`, `/api/stats`, `/api/summary`, `/api/audio`) fall back to
    the JSON snapshot written by `pipe calibrate export-snapshot`
    (pipeline/nodes/calibrate.py's SNAPSHOT_PATH) -- a point-in-time dump of
    the pending review queue (candidates, audio paths, everything the UI
    needs) taken while the catalog was free. It's necessarily stale relative
    to the live DB (new calibrate.sample batches queued after the snapshot
    won't appear), which the UI surfaces via each response's `mode` field.
  - Writes ALWAYS go to a local JSONL buffer (append_pending_decision(),
    PENDING_DECISIONS_PATH) rather than calling record_decision() inline --
    not just as an offline fallback, but unconditionally, so a click never
    depends on DB availability at all and the online/offline code paths stay
    the same. `pipe calibrate flush-pending` replays the buffer into the
    catalog once the writer is free (safe to run anytime, safe to re-run).
  - `_local_decisions` (loaded from the buffer at server startup, updated
    live as submissions come in) is overlaid on top of BOTH the live-DB and
    the offline-snapshot read paths: a segment just decided locally still
    reads 'pending' in the DB until the next flush, so every read endpoint
    needs this overlay to avoid re-serving it, and to keep the stats/history
    panels accurate for the reviewer's own session.
"""

import argparse
import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb

from pipeline.catalog.catalog import CATALOG_PATH, connect, connect_ro
from pipeline.nodes.calibrate import (
    PENDING_DECISIONS_PATH,
    SNAPSHOT_PATH,
    _levenshtein,
    append_pending_decision,
    get_item,
    jyutping_preview,
    list_batches,
    list_history,
    list_sources,
    load_pending_decisions,
    next_pending,
    queue_stats,
    run_calibrate_sample,
    summary_stats,
)

# Auto-refill: when /api/next finds nothing AND the UI isn't scoped to one
# specific batch (a batch is a deliberate, fixed sample -- silently minting a
# new one under it would be confusing), top up the queue with a small fresh
# sample instead of just reporting "empty". A manual "Refill" button
# (POST /api/refill) is always available regardless of batch scope.
_AUTO_REFILL_N = 100

log = logging.getLogger(__name__)

_AUDIO_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
}

# Serializes this process's own writers across ThreadingHTTPServer's request
# threads (so two browser tabs clicking at once can't race each other's
# connect()/close() cycle) -- it is NOT a substitute for DuckDB's own
# cross-process file lock, which still applies underneath this.
_write_lock = threading.Lock()

# A per-request connection removes the "server blocks everyone else for its
# whole session" problem, but the reverse still holds: DuckDB is
# single-writer, so while a `pipe run` / `pipe run-many` batch job has ITS
# OWN connection open (which can legitimately be minutes to hours for a big
# backlog node), this server's connect()/connect_ro() calls will hit
# duckdb.IOException just like any other process would. Retry with a short
# backoff instead of failing the click outright -- most overlaps are brief.
# Shortened from 30s to 4s (2026-07-13): a long batch node holds the lock for
# its ENTIRE run, so waiting longer just delays falling back to the offline
# snapshot/buffer for no benefit -- see the module docstring's offline-mode note.
_LOCK_RETRY_DELAY_S = 0.5
_LOCK_RETRY_TIMEOUT_S = 4.0


class CatalogBusyError(RuntimeError):
    """The DuckDB write lock stayed held by another process for longer than
    _LOCK_RETRY_TIMEOUT_S. Read paths catch this and fall back to the offline
    snapshot; only a caller with no offline fallback (e.g. calibrate.sample's
    /api/refill, which needs a live DB query to discover genuinely new
    segments) lets it surface to the browser as a 503."""


def _open_with_retry(open_fn):
    deadline = time.monotonic() + _LOCK_RETRY_TIMEOUT_S
    last_err = None
    while time.monotonic() < deadline:
        try:
            return open_fn()
        except duckdb.IOException as exc:
            last_err = exc
            time.sleep(_LOCK_RETRY_DELAY_S)
    raise CatalogBusyError(
        f"catalog locked by another process (a batch pipeline job?) for over "
        f"{_LOCK_RETRY_TIMEOUT_S:.0f}s -- try again shortly"
    ) from last_err


def _read(fn, *args, **kwargs):
    """Open a read-only connection for exactly one query, then close it.
    Read-only connections are safe alongside other processes' own read-only
    connections, but still block a concurrent writer while open -- scoping
    it to a single request (instead of the server's whole lifetime) is what
    lets batch DAG nodes / one-off scripts run in the gaps between clicks."""
    conn = _open_with_retry(lambda: connect_ro(CATALOG_PATH))
    try:
        return fn(conn, *args, **kwargs)
    finally:
        conn.close()


_VALID_QA_TIERS = ("auto_gold", "silver", "bronze")


def _parse_sample_options(payload: dict) -> tuple:
    """Shared tier/min_agreement/code_switch parsing for /api/refill (JSON body)
    and /api/next's auto-refill (query string, flattened by the caller into the
    same {tier, min_agreement, code_switch} shape first) -- the browser UI's
    equivalent of `pipe run calibrate.sample --tier/--min-agreement/--code-switch`
    (added 2026-07-15, mirrors calibrate.py's own discover()/run_calibrate_sample()
    params). Raises ValueError on a malformed tier/min_agreement so the caller can
    400 instead of silently sampling something the reviewer didn't ask for;
    code_switch is validated downstream by calibrate.discover() itself."""
    tier = (payload.get("tier") or "").strip() or None
    if tier is not None and tier not in _VALID_QA_TIERS:
        raise ValueError(f"tier must be one of {_VALID_QA_TIERS}, got {tier!r}")
    raw_min_agreement = payload.get("min_agreement")
    min_agreement = float(raw_min_agreement) if raw_min_agreement not in (None, "") else None
    code_switch = (payload.get("code_switch") or "").strip() or None
    return tier, min_agreement, code_switch


def _write(fn, *args, **kwargs):
    """Open a read-write connection for exactly one write (plus any reads
    the same handler needs afterward, e.g. refreshed stats), then close it
    immediately. The write lock is held for one request's duration, not the
    whole server session. Only calibrate.sample's /api/refill still uses
    this -- decision writes go through append_pending_decision() instead
    (see module docstring's offline-mode note)."""
    with _write_lock:
        conn = _open_with_retry(lambda: connect(CATALOG_PATH))
        try:
            return fn(conn, *args, **kwargs)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Offline mode: local decision buffer + JSON snapshot fallback (2026-07-13)
# ---------------------------------------------------------------------------

_local_decisions: dict = load_pending_decisions()
_local_decisions_lock = threading.Lock()
_snapshot: dict | None = None


def _load_snapshot() -> None:
    global _snapshot
    if not SNAPSHOT_PATH.exists():
        _snapshot = None
        return
    try:
        _snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"failed to load offline snapshot {SNAPSHOT_PATH}: {exc}")
        _snapshot = None


_load_snapshot()  # best-effort at import time; harmless if the file doesn't exist yet


def _overlay_item(item: dict | None) -> dict | None:
    """Apply a pending-but-not-yet-flushed local decision on top of an item
    fetched from the live DB or the offline snapshot -- both sources still
    read 'pending' for it until the next `pipe calibrate flush-pending`."""
    if item is None:
        return None
    local = _local_decisions.get(item["id"])
    if not local:
        return item
    merged = dict(item)
    merged["decision"] = local["decision"]
    merged["reviewed_text"] = local.get("text")
    merged["flag_reason"] = local.get("flag_reason")
    return merged


def _stats_with_overlay(base_stats: dict, sample_batch: str | None, source: str | None) -> dict:
    stats = dict(base_stats)
    for entry in _local_decisions.values():
        if sample_batch and entry.get("sample_batch") != sample_batch:
            continue
        if source and entry.get("source") != source:
            continue
        # base_stats counted this id as 'pending' (its DB/snapshot state is
        # unchanged until flush) -- move it to its real, locally-decided bucket.
        stats["pending"] = max(0, stats.get("pending", 0) - 1)
        stats[entry["decision"]] = stats.get(entry["decision"], 0) + 1
    stats["total"] = sum(v for k, v in stats.items() if k != "total")
    return stats


def _offline_items(sample_batch: str | None = None, source: str | None = None) -> list[dict]:
    if not _snapshot:
        return []
    items = _snapshot.get("items", [])
    if sample_batch:
        items = [i for i in items if i.get("sample_batch") == sample_batch]
    if source:
        items = [i for i in items if i.get("source") == source]
    return items


def _offline_next_pending(sample_batch, source, order) -> dict | None:
    items = [i for i in _offline_items(sample_batch, source) if i["id"] not in _local_decisions]
    if not items:
        return None
    if order == "agreement_asc":
        items.sort(key=lambda i: (i.get("agreement") is None, i.get("agreement")))
    elif order == "agreement_desc":
        items.sort(key=lambda i: (i.get("agreement") is None, -(i.get("agreement") or 0)))
    else:
        items.sort(key=lambda i: i.get("queued_at") or "")
    return items[0]


def _offline_get_item(seg_id: str) -> dict | None:
    if not _snapshot:
        return None
    for item in _snapshot.get("items", []):
        if item["id"] == seg_id:
            return _overlay_item(item)
    return None


def _offline_list_history(sample_batch: str | None, limit: int) -> list[dict]:
    entries = [
        {
            "id": e["id"], "decision": e["decision"], "reviewed_text": e.get("text"),
            "reviewed_at": e.get("ts"), "source": e.get("source"), "flag_reason": e.get("flag_reason"),
        }
        for e in _local_decisions.values()
        if not sample_batch or e.get("sample_batch") == sample_batch
    ]
    entries.sort(key=lambda e: e.get("reviewed_at") or "", reverse=True)
    return entries[:limit]


def _offline_base_stats(sample_batch: str | None, source: str | None) -> dict:
    n = len(_offline_items(sample_batch, source))
    return {"pending": n, "verified": 0, "skipped": 0, "rejected": 0, "flagged": 0, "total": n}


def _offline_queue_stats(sample_batch: str | None = None, source: str | None = None) -> dict:
    return _stats_with_overlay(_offline_base_stats(sample_batch, source), sample_batch, source)


def _offline_list_batches() -> list[dict]:
    if not _snapshot:
        return []
    batches = sorted({i["sample_batch"] for i in _snapshot.get("items", []) if i.get("sample_batch")})
    return [{"sample_batch": b, **_offline_queue_stats(b, None)} for b in batches]


def _offline_list_sources() -> list[str]:
    if not _snapshot:
        return []
    return sorted({i["source"] for i in _snapshot.get("items", []) if i.get("source")})


def _offline_summary_stats(sample_batch: str | None) -> dict:
    """Necessarily partial vs. the live version: only reflects decisions made
    during THIS offline session (no DB access to past review history)."""
    items_by_id = {i["id"]: i for i in _offline_items(sample_batch, None)}
    by_source: dict[str, dict[str, int]] = {}
    agreements_by_decision: dict[str, list[float]] = {}
    edits = []
    flag_counter: dict[str, int] = {}
    for e in _local_decisions.values():
        if sample_batch and e.get("sample_batch") != sample_batch:
            continue
        item = items_by_id.get(e["id"])
        src = e.get("source") or (item or {}).get("source") or "unknown"
        by_source.setdefault(src, {})[e["decision"]] = by_source.setdefault(src, {}).get(e["decision"], 0) + 1
        if item and item.get("agreement") is not None:
            agreements_by_decision.setdefault(e["decision"], []).append(item["agreement"])
        if e["decision"] == "verified" and item and e.get("text") is not None:
            edits.append(_levenshtein(item.get("best_text") or "", e["text"]))
        if e["decision"] in ("flagged", "rejected") and e.get("flag_reason"):
            flag_counter[e["flag_reason"]] = flag_counter.get(e["flag_reason"], 0) + 1
    return {
        "decision_counts": _offline_queue_stats(sample_batch, None),
        "by_source": by_source,
        "avg_agreement_by_decision": {
            d: round(sum(vs) / len(vs), 3) for d, vs in agreements_by_decision.items()
        },
        "avg_edit_distance_verified": round(sum(edits) / len(edits), 2) if edits else None,
        "verified_edit_sample_size": len(edits),
        "top_flag_reasons": sorted(
            ({"reason": r, "count": n} for r, n in flag_counter.items()), key=lambda x: -x["count"]
        )[:10],
    }


def _read_or_offline(live_fn, offline_fn):
    """Try a live-DB read; on CatalogBusyError fall back to the offline
    snapshot/buffer instead of surfacing a 503. Returns (result, mode)."""
    try:
        return _read(live_fn), "live"
    except CatalogBusyError:
        return offline_fn(), "offline"


def _merge_history(db_history: list[dict], sample_batch: str | None, limit: int) -> list[dict]:
    """DB history only contains decision != 'pending' rows, so it never
    includes an id that's only been decided locally (still 'pending' in the
    DB until flush) -- merge in the local buffer's own entries so a just-
    submitted decision shows up in the panel immediately."""
    local_entries = _offline_list_history(sample_batch, limit)
    seen = {h["id"] for h in db_history}
    merged = list(db_history) + [e for e in local_entries if e["id"] not in seen]
    merged.sort(key=lambda h: h.get("reviewed_at") or "", reverse=True)
    return merged[:limit]


def _live_list_batches(conn) -> list[dict]:
    out = []
    for b in list_batches(conn):
        sb = b["sample_batch"]
        base = {k: v for k, v in b.items() if k != "sample_batch"}
        out.append({"sample_batch": sb, **_stats_with_overlay(base, sb, None)})
    return out


def _live_summary_stats(conn, sample_batch: str | None) -> dict:
    """Only decision_counts (the progress bar) gets the local-buffer overlay
    here -- by_source/avg_agreement/edit-distance for decisions made since
    the last flush are a nice-to-have this doesn't chase; they self-correct
    once `pipe calibrate flush-pending` lands."""
    summary = summary_stats(conn, sample_batch)
    summary["decision_counts"] = _stats_with_overlay(summary["decision_counts"], sample_batch, None)
    return summary


def _build_app(default_batch: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib method name
            try:
                self._do_GET()
            except CatalogBusyError as exc:
                self._send_json({"error": str(exc)}, status=503)

        def _do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/":
                body = _PAGE_HTML.replace(
                    "__DEFAULT_BATCH__", json.dumps(default_batch or "")
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/next":
                batch = qs.get("batch", [None])[0] or None
                source = qs.get("source", [None])[0] or None
                order = qs.get("order", ["queued"])[0]
                item, mode = _read_or_offline(
                    lambda conn: _overlay_item(next_pending(conn, batch, source, order, exclude_ids=set(_local_decisions))),
                    lambda: _offline_next_pending(batch, source, order),
                )
                refilled = None
                if item is None and not batch and mode == "live":
                    # calibrate.sample needs a live DB query to discover genuinely
                    # new segments -- no offline equivalent, so this step is
                    # simply skipped while offline (the reviewer works through
                    # whatever the snapshot already has).
                    # tier/min_agreement/code_switch (added 2026-07-15): the
                    # reviewer's current sample-options selection, so this
                    # auto top-up stays scoped to what they're actually
                    # reviewing instead of diluting it with an unscoped
                    # random sample -- see _parse_sample_options().
                    try:
                        tier, min_agreement, code_switch = _parse_sample_options({
                            "tier": qs.get("tier", [None])[0],
                            "min_agreement": qs.get("min_agreement", [None])[0],
                            "code_switch": qs.get("code_switch", [None])[0],
                        })
                    except ValueError:
                        tier = min_agreement = code_switch = None
                    try:
                        result = _write(
                            lambda conn: asyncio.run(run_calibrate_sample(
                                conn=conn, n=_AUTO_REFILL_N, tier=tier,
                                min_agreement=min_agreement, code_switch=code_switch,
                            ))
                        )
                        if result["queued"]:
                            refilled = result
                            item = _read(lambda conn: _overlay_item(
                                next_pending(conn, batch, source, order, exclude_ids=set(_local_decisions))
                            ))
                    except CatalogBusyError:
                        pass
                self._send_json({"item": item, "refilled": refilled, "mode": mode})
                return

            if parsed.path == "/api/item":
                seg_id = qs.get("id", [None])[0]
                if not seg_id:
                    self.send_error(400, "missing id")
                    return
                item, mode = _read_or_offline(
                    lambda conn: _overlay_item(get_item(conn, seg_id)),
                    lambda: _offline_get_item(seg_id),
                )
                self._send_json({"item": item, "mode": mode})
                return

            if parsed.path == "/api/history":
                batch = qs.get("batch", [None])[0] or None
                limit = int(qs.get("limit", ["20"])[0])
                items, mode = _read_or_offline(
                    lambda conn: _merge_history(list_history(conn, batch, limit), batch, limit),
                    lambda: _offline_list_history(batch, limit),
                )
                self._send_json({"items": items, "mode": mode})
                return

            if parsed.path == "/api/batches":
                batches, mode = _read_or_offline(
                    lambda conn: _live_list_batches(conn),
                    _offline_list_batches,
                )
                self._send_json({"batches": batches, "mode": mode})
                return

            if parsed.path == "/api/sources":
                sources, mode = _read_or_offline(list_sources, _offline_list_sources)
                self._send_json({"sources": sources, "mode": mode})
                return

            if parsed.path == "/api/stats":
                batch = qs.get("batch", [None])[0] or None
                source = qs.get("source", [None])[0] or None
                stats, mode = _read_or_offline(
                    lambda conn: _stats_with_overlay(queue_stats(conn, batch, source), batch, source),
                    lambda: _offline_queue_stats(batch, source),
                )
                stats["mode"] = mode
                self._send_json(stats)
                return

            if parsed.path == "/api/summary":
                batch = qs.get("batch", [None])[0] or None
                summary, mode = _read_or_offline(
                    lambda conn: _live_summary_stats(conn, batch),
                    lambda: _offline_summary_stats(batch),
                )
                summary["mode"] = mode
                self._send_json(summary)
                return

            if parsed.path == "/api/g2p_preview":
                text = qs.get("text", [""])[0]
                preview = jyutping_preview(text)
                self._send_json(preview)
                return

            if parsed.path == "/api/audio":
                seg_id = qs.get("id", [None])[0]
                if not seg_id:
                    self.send_error(400, "missing id")
                    return
                audio_path_str, _mode = _read_or_offline(
                    lambda conn: conn.execute(
                        "SELECT s.audio_path FROM calibration_review c "
                        "JOIN segments s ON c.id = s.id WHERE c.id = ?",
                        [seg_id],
                    ).fetchone(),
                    lambda: ((_offline_get_item(seg_id) or {}).get("audio_path"),),
                )
                if not audio_path_str or not audio_path_str[0]:
                    self.send_error(404, "unknown id")
                    return
                audio_path = Path(audio_path_str[0])
                if not audio_path.exists():
                    self.send_error(404, f"file missing on disk: {audio_path}")
                    return
                content_type = _AUDIO_CONTENT_TYPES.get(
                    audio_path.suffix.lower(), "application/octet-stream"
                )
                data = audio_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(404)

        def do_POST(self):  # noqa: N802 - stdlib method name
            try:
                self._do_POST()
            except CatalogBusyError as exc:
                self._send_json({"error": str(exc)}, status=503)

        def _do_POST(self):
            if self.path == "/api/refill":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length)) if length else {}
                    n = int(payload.get("n", 200))
                    tier, min_agreement, code_switch = _parse_sample_options(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    self._send_json({"error": f"malformed request body: {exc}"}, status=400)
                    return
                try:
                    result = _write(
                        lambda conn: asyncio.run(run_calibrate_sample(
                            conn=conn, n=n, tier=tier, min_agreement=min_agreement,
                            code_switch=code_switch,
                        ))
                    )
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self._send_json(result)
                return

            if self.path != "/api/submit":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length))
                seg_id = payload["id"]
                decision = payload["decision"]
                text = payload.get("text")
                flag_reason = payload.get("flag_reason")
                sample_batch = payload.get("sample_batch")
                source = payload.get("source")
            except (KeyError, json.JSONDecodeError):
                self.send_error(400, "malformed request body")
                return

            # Always buffer to the local JSONL (2026-07-13) -- never call
            # record_decision() inline here. `pipe calibrate flush-pending`
            # replays it into the catalog whenever the writer is free; see
            # the module docstring's offline-mode note for why this is
            # unconditional rather than a busy-catalog-only fallback.
            try:
                with _local_decisions_lock:
                    entry = append_pending_decision(
                        seg_id, decision, text, flag_reason,
                        sample_batch=sample_batch, source=source,
                    )
                    _local_decisions[seg_id] = entry
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            stats, mode = _read_or_offline(
                lambda conn: _stats_with_overlay(queue_stats(conn), None, None),
                lambda: _offline_queue_stats(None, None),
            )
            self._send_json({"ok": True, "stats": stats, "mode": mode})

    return Handler


_PAGE_HTML = r"""<!doctype html>
<html lang="yue">
<head>
<meta charset="utf-8">
<title>Canto Corpus — Text Calibration</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #14161a; --panel: #1c1f25; --panel-2: #20242b; --border: #2b2f37;
    --text: #e8e6e1; --muted: #8b939e; --dim: #5c6470;
    --accent: #6c94ff; --accent-2: #4a7dfc;
    --good: #3d8f5c; --warn: #b98a3a; --bad: #8a3a3a;
    --ins: #2f6b45; --del: #7a3535;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Noto Sans HK", "PingFang HK", sans-serif;
         margin: 0; padding: 1.2rem 1.6rem 3rem; background: var(--bg); color: var(--text); }
  h1 { font-size: 1rem; font-weight: 600; color: var(--muted); letter-spacing: 0.03em;
       margin: 0 0 0.9rem; }
  a, button { font-family: inherit; }

  /* ---- top bar: global stats + filters ---- */
  #topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem;
            background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
            padding: 0.8rem 1.1rem; margin-bottom: 1rem; font-size: 0.85rem; }
  #topbar .stat-group { display: flex; gap: 1rem; font-variant-numeric: tabular-nums; color: var(--muted); }
  #topbar .stat-group b { color: var(--text); }
  #topbar .stat-group .verified b { color: var(--good); }
  #topbar .stat-group .rejected b { color: var(--bad); }
  #progress-bar { flex: 1 1 160px; min-width: 120px; height: 6px; border-radius: 3px;
                  background: var(--border); overflow: hidden; display: flex; }
  #progress-bar span { display: block; height: 100%; }
  .filters { display: flex; gap: 0.5rem; margin-left: auto; flex-wrap: wrap; align-items: center; }
  select { background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
           border-radius: 6px; padding: 0.35rem 0.5rem; font-size: 0.82rem; }
  /* ---- sample-options group: what a Refill click queues, distinct from the
     batch/source/order group above which only scopes browsing of the
     already-queued items (added 2026-07-15) ---- */
  .sample-options { display: flex; gap: 0.4rem; align-items: center; padding-left: 0.6rem;
                     border-left: 1px solid var(--border); }
  .sample-options .label { font-size: 0.72rem; color: var(--dim); }
  #minAgreementInput { width: 5.2rem; background: var(--panel-2); color: var(--text);
                        border: 1px solid var(--border); border-radius: 6px;
                        padding: 0.35rem 0.5rem; font-size: 0.82rem; }

  /* ---- main grid: review card (left) + history (right) ---- */
  #layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 1rem; align-items: start; }
  @media (max-width: 900px) { #layout { grid-template-columns: 1fr; } }

  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 1.3rem; }
  .meta { display: flex; gap: 1.1rem; flex-wrap: wrap; color: var(--muted); font-size: 0.82rem;
          margin-bottom: 0.9rem; font-variant-numeric: tabular-nums; }
  .meta .agreement-pill { padding: 0.1rem 0.5rem; border-radius: 999px; border: 1px solid var(--border); }
  .meta .agreement-pill.low { color: var(--warn); border-color: var(--warn); }
  .meta .decision-pill { padding: 0.1rem 0.5rem; border-radius: 999px; font-weight: 600; }
  .meta .decision-pill.verified { background: var(--good); color: white; }
  .meta .decision-pill.skipped { background: var(--panel-2); color: var(--muted); }
  .meta .decision-pill.rejected { background: var(--bad); color: white; }
  .meta .decision-pill.flagged { background: var(--warn); color: #1c1300; }

  /* ---- audio + playback controls ---- */
  .player-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; flex-wrap: wrap; }
  audio { flex: 1 1 260px; height: 34px; }
  .pbtn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
          border-radius: 6px; padding: 0.3rem 0.55rem; font-size: 0.78rem; cursor: pointer; }
  .pbtn.active { color: var(--accent); border-color: var(--accent); }
  #waveform { width: 100%; height: 56px; display: block; margin-bottom: 1rem; border-radius: 6px;
              background: var(--panel-2); cursor: pointer; }

  /* ---- Cantonese particle quick-insert toolbar ---- */
  #particles { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.7rem; }
  #particles button { background: var(--panel-2); color: var(--accent); border: 1px solid var(--border);
                       border-radius: 5px; padding: 0.2rem 0.5rem; font-size: 0.88rem; cursor: pointer; }
  #particles button:hover { border-color: var(--accent); }
  #particles .plabel { color: var(--dim); font-size: 0.72rem; align-self: center; margin-right: 0.2rem; }

  /* ---- ASR candidates with diff ---- */
  .candidates { display: flex; flex-direction: column; gap: 0.45rem; margin-bottom: 1rem; }
  .cand { display: flex; align-items: flex-start; gap: 0.7rem; padding: 0.55rem 0.75rem;
          background: var(--panel-2); border: 1px solid var(--border); border-radius: 7px; }
  .cand .model { color: var(--accent); font-size: 0.74rem; min-width: 84px; flex-shrink: 0;
                 padding-top: 0.15rem; }
  .cand .diff { flex: 1; font-size: 0.95rem; line-height: 1.5; }
  .cand .diff ins { text-decoration: none; background: var(--ins); color: #d7f5df; border-radius: 2px; }
  .cand .diff del { text-decoration: none; background: var(--del); color: #f7d7d7; border-radius: 2px; }
  .cand button.insert { flex-shrink: 0; background: transparent; color: var(--accent);
                         border: 1px solid var(--accent); border-radius: 6px; padding: 0.25rem 0.55rem;
                         font-size: 0.76rem; cursor: pointer; }
  .cand button.insert:hover { background: var(--accent); color: #0c1220; }
  .cand kbd { font-size: 0.68rem; color: var(--dim); border: 1px solid var(--border); border-radius: 3px;
              padding: 0 0.3rem; margin-left: 0.4rem; }

  textarea { width: 100%; min-height: 4.5rem; font-size: 1.15rem;
             background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
             padding: 0.7rem; margin-bottom: 0.5rem; resize: vertical; }
  .editor-row { display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 0.9rem; font-size: 0.76rem; color: var(--dim); }
  .editor-row #undoBtn { background: none; border: 1px solid var(--border); color: var(--muted);
                          border-radius: 5px; padding: 0.2rem 0.5rem; cursor: pointer; }
  .editor-row #undoBtn:disabled { opacity: 0.35; cursor: default; }

  /* ---- Jyutping validity preview ---- */
  #jyutping-preview { font-size: 0.78rem; color: var(--dim); margin-bottom: 0.9rem;
                       font-family: ui-monospace, "SF Mono", monospace; line-height: 1.5; }
  #jyutping-preview .frac { font-weight: 600; }
  #jyutping-preview .frac.ok { color: var(--good); }
  #jyutping-preview .frac.warn { color: var(--warn); }
  #jyutping-preview .frac.bad { color: var(--bad); }
  #jyutping-preview .bad-token { color: var(--bad); text-decoration: underline wavy; }

  .actions { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; }
  button.action { font-size: 0.95rem; padding: 0.6rem 1.1rem; border-radius: 6px; border: none;
                  cursor: pointer; font-weight: 500; }
  #verify { background: var(--good); color: white; }
  #skip { background: var(--panel-2); color: var(--text); border: 1px solid var(--border) !important; }
  #reject { background: var(--bad); color: white; }
  #mandarin { background: var(--bad); color: white; border: 2px solid #7a0000 !important; }
  #multiSpeaker { background: var(--bad); color: white; border: 2px solid #7a0000 !important; }
  #wrongSpeakerId { background: var(--warn); color: #1c1300; border: 2px solid #7a5c00 !important; }
  #flag { background: var(--warn); color: #1c1300; }
  .hint { color: var(--dim); font-size: 0.74rem; margin-top: 0.8rem; }
  #done { display: none; color: var(--muted); }
  #done button { margin-top: 0.6rem; }

  /* ---- flag-reason inline box ---- */
  #flag-box { display: none; gap: 0.5rem; align-items: center; margin-top: 0.7rem; }
  #flag-box input { flex: 1; background: var(--panel-2); color: var(--text); border: 1px solid var(--warn);
                     border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.85rem; }
  #flag-box button { border-radius: 6px; padding: 0.35rem 0.7rem; font-size: 0.8rem; cursor: pointer; border: none; }
  #flag-confirm { background: var(--warn); color: #1c1300; }
  #flag-cancel { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border) !important; }

  /* ---- refill ---- */
  #refillBtn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
               border-radius: 6px; padding: 0.35rem 0.6rem; font-size: 0.78rem; cursor: pointer; }
  #refill-toast { font-size: 0.76rem; color: var(--accent); margin-top: 0.5rem; display: none; }

  /* ---- history sidebar ---- */
  #history h2 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
                color: var(--muted); margin: 0 0 0.7rem; }
  #history-list { display: flex; flex-direction: column; gap: 0.4rem; max-height: 70vh; overflow-y: auto; }
  .hist-item { padding: 0.5rem 0.6rem; background: var(--panel-2); border: 1px solid var(--border);
               border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
  .hist-item:hover { border-color: var(--accent); }
  .hist-item .hist-top { display: flex; justify-content: space-between; color: var(--dim); font-size: 0.7rem;
                          margin-bottom: 0.2rem; }
  .hist-item .hist-decision.verified { color: var(--good); }
  .hist-item .hist-decision.skipped { color: var(--muted); }
  .hist-item .hist-decision.rejected { color: var(--bad); }
  .hist-item .hist-decision.flagged { color: var(--warn); }
  .hist-item .hist-text { color: var(--text); overflow: hidden; text-overflow: ellipsis;
                           white-space: nowrap; }
  #backBtn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
             border-radius: 6px; padding: 0.3rem 0.6rem; font-size: 0.78rem; cursor: pointer;
             margin-bottom: 0.6rem; }
  #backBtn:disabled { opacity: 0.35; cursor: default; }

  /* ---- summary dashboard ---- */
  #summary { margin-top: 1rem; }
  #summary .summary-head { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem; }
  #summary .summary-head h2 { font-size: 0.85rem; color: var(--muted); margin: 0; text-transform: uppercase;
                               letter-spacing: 0.04em; }
  #summaryRefresh { background: var(--panel-2); color: var(--accent); border: 1px solid var(--border);
                     border-radius: 6px; padding: 0.3rem 0.65rem; font-size: 0.78rem; cursor: pointer; }
  #summary-body { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.9rem; }
  .stat-card { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem; }
  .stat-card h3 { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--dim);
                  margin: 0 0 0.6rem; }
  .stat-card .big-number { font-size: 1.6rem; font-variant-numeric: tabular-nums; color: var(--text); }
  .stat-card .sub { color: var(--dim); font-size: 0.75rem; margin-top: 0.2rem; }
  .bar-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; margin-bottom: 0.4rem; }
  .bar-row .bar-label { width: 82px; flex-shrink: 0; color: var(--muted); overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
  .bar-row .bar-track { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden;
                         display: flex; }
  .bar-row .bar-track span { display: block; height: 100%; }
  .bar-row .bar-value { width: 32px; text-align: right; color: var(--dim); font-variant-numeric: tabular-nums; }
  #summary-empty { color: var(--dim); font-size: 0.82rem; }
  #offline-banner { display: none; background: var(--warn); color: #1c1300; font-size: 0.8rem;
                     font-weight: 600; padding: 0.5rem 0.9rem; border-radius: 8px; margin-bottom: 0.8rem; }
</style>
</head>
<body>
<h1>CANTO CORPUS &mdash; TEXT CALIBRATION</h1>
<div id="offline-banner"></div>

<div id="topbar">
  <div class="stat-group">
    <span class="verified"><b id="s-verified">0</b> verified</span>
    <span><b id="s-skipped">0</b> skipped</span>
    <span class="rejected"><b id="s-rejected">0</b> rejected</span>
    <span><b id="s-flagged">0</b> flagged</span>
    <span><b id="s-pending">0</b> pending</span>
    <span style="color:var(--dim)">/ <span id="s-total">0</span></span>
  </div>
  <div id="progress-bar">
    <span id="pb-verified" style="background:var(--good)"></span>
    <span id="pb-skipped" style="background:var(--dim)"></span>
    <span id="pb-rejected" style="background:var(--bad)"></span>
    <span id="pb-flagged" style="background:var(--warn)"></span>
  </div>
  <div class="filters">
    <select id="batchSelect"><option value="">All batches</option></select>
    <select id="sourceSelect"><option value="">All sources</option></select>
    <select id="orderSelect">
      <option value="queued">Queued order</option>
      <option value="agreement_asc">Lowest agreement first</option>
      <option value="agreement_desc">Highest agreement first</option>
    </select>
    <span class="sample-options" title="Scopes what the Refill button queues (same as pipe run calibrate.sample --tier/--min-agreement/--code-switch) -- does not filter the items already in the queue.">
      <span class="label">Sample:</span>
      <select id="tierSelect">
        <option value="">any tier</option>
        <option value="auto_gold">auto_gold</option>
        <option value="silver">silver</option>
        <option value="bronze">bronze</option>
      </select>
      <input id="minAgreementInput" type="number" step="0.01" min="0" max="1" placeholder="min agr.">
      <select id="codeSwitchSelect">
        <option value="">any code-switch</option>
        <option value="only">code-switch only</option>
        <option value="exclude">no code-switch</option>
      </select>
    </span>
    <button id="refillBtn" title="Queue another random sample using the Sample options above">↻ Refill</button>
  </div>
</div>

<div id="layout">
  <div>
    <div class="card" id="review">
      <div class="meta" id="meta"></div>
      <div class="player-row">
        <audio id="player" controls></audio>
        <button class="pbtn" id="loopBtn" title="Loop playback">Loop</button>
        <button class="pbtn" id="autoplayBtn" title="Autoplay next segment">Autoplay</button>
        <button class="pbtn" data-speed="0.75">0.75×</button>
        <button class="pbtn" data-speed="1" style="color:var(--accent);border-color:var(--accent)">1×</button>
        <button class="pbtn" data-speed="1.25">1.25×</button>
        <button class="pbtn" data-speed="1.5">1.5×</button>
      </div>
      <canvas id="waveform" width="760" height="56" title="Click to seek"></canvas>
      <div class="candidates" id="candidates"></div>
      <div id="particles"><span class="plabel">Particles:</span></div>
      <textarea id="text" autofocus></textarea>
      <div id="jyutping-preview"></div>
      <div class="editor-row">
        <span>Click a candidate's <b>Insert</b> button (or press 1–4) to copy it in — the text box never changes on its own.</span>
        <button id="undoBtn" disabled>↩ Undo insert</button>
      </div>
      <div class="actions">
        <button class="action" id="verify">Verify (Enter)</button>
        <button class="action" id="skip">Skip (S)</button>
        <button class="action" id="reject">Reject (D)</button>
        <button class="action" id="mandarin" title="Not HK Cantonese -- rejects and records the reason">Mandarin (M)</button>
        <button class="action" id="multiSpeaker" title="Audio has more than one speaker -- rejects and records the reason (T9)">Multi-speaker (N)</button>
        <button class="action" id="wrongSpeakerId" title="Audio is single-speaker but filed under the wrong speaker_id -- flags only, does NOT exclude (T9)">Wrong speaker ID (W)</button>
        <button class="action" id="flag">Flag issue (F)</button>
      </div>
      <div id="flag-box">
        <input id="flag-reason" placeholder="What's wrong? (segmentation, wrong language, corrupt audio…)">
        <button id="flag-confirm">Confirm flag</button>
        <button id="flag-cancel">Cancel</button>
      </div>
      <div class="hint">Space replays/pauses audio when the text box isn't focused. Highlighted diff: <ins>green</ins> = candidate adds this, <del>red</del> = candidate removes this (relative to the box's current text). Click the waveform to seek.</div>
    </div>
    <div id="done">
      Queue empty for this filter — no pending segments left.<br>
      <button id="doneRefillBtn">↻ Queue a new sample</button>
    </div>

    <div id="summary">
      <div class="summary-head">
        <h2>Sample summary</h2>
        <button id="summaryRefresh">↻ Refresh</button>
      </div>
      <div id="summary-body"><span id="summary-empty">Click Refresh to load.</span></div>
    </div>
  </div>

  <div id="history">
    <button id="backBtn" disabled>← Back to previous</button>
    <div id="refill-toast"></div>
    <h2>Recently reviewed</h2>
    <div id="history-list"></div>
  </div>
</div>

<script>
const DEFAULT_BATCH = __DEFAULT_BATCH__;
let current = null;
let undoStack = [];
let visitStack = [];   // ids of items shown this session, for Back navigation
let reviewingFromHistory = false;
let audioCtx = null;

// Cantonese particles/fillers ASR commonly drops or mis-renders as a
// Mandarin homophone -- see the chat reference list this toolbar mirrors.
const PARTICLES = [
  '啊', '呀', '喇', '嘅', '㗎', '咩', '呢', '喎', '噃', '咧', '囉', '咯', '嘛', '啦',
  '掛', '添', '喳', '唉', '咦', '嘩', '喂', '吓', '嗯', '噉', '咁', '嗰', '哋', '喺', '畀', '緊', '咗',
];

function currentFilters() {
  return {
    batch: document.getElementById('batchSelect').value,
    source: document.getElementById('sourceSelect').value,
    order: document.getElementById('orderSelect').value,
  };
}

// Scopes what a Refill (manual or auto-on-empty) queues -- distinct from
// currentFilters() above, which only scopes browsing of items already in
// the queue. Mirrors `pipe run calibrate.sample --tier/--min-agreement/
// --code-switch` (2026-07-15).
function currentSampleOptions() {
  return {
    tier: document.getElementById('tierSelect').value,
    min_agreement: document.getElementById('minAgreementInput').value,
    code_switch: document.getElementById('codeSwitchSelect').value,
  };
}

function qs(params) {
  return Object.entries(params).filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

// ---- diff: simple char-level LCS diff (Chinese text has no word boundaries) ----
function charDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  const ops = [];
  let i = n, j = m;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) { ops.push(['eq', a[i - 1]]); i--; j--; }
    else if (dp[i - 1][j] >= dp[i][j - 1]) { ops.push(['del', a[i - 1]]); i--; }
    else { ops.push(['ins', b[j - 1]]); j--; }
  }
  while (i > 0) { ops.push(['del', a[i - 1]]); i--; }
  while (j > 0) { ops.push(['ins', b[j - 1]]); j--; }
  ops.reverse();
  const merged = [];
  for (const [type, ch] of ops) {
    if (merged.length && merged[merged.length - 1].type === type) merged[merged.length - 1].text += ch;
    else merged.push({ type, text: ch });
  }
  return merged;
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderDiffHtml(current, candidate) {
  return charDiff(current, candidate).map(({ type, text }) => {
    const t = escapeHtml(text);
    if (type === 'ins') return `<ins>${t}</ins>`;
    if (type === 'del') return `<del>${t}</del>`;
    return t;
  }).join('');
}

function rerenderCandidateDiffs() {
  if (!current) return;
  const boxText = document.getElementById('text').value;
  document.querySelectorAll('.cand .diff').forEach((el) => {
    el.innerHTML = renderDiffHtml(boxText, el.dataset.candidate);
  });
}

// ---- offline-mode banner: any API response can carry mode:'offline' when
// the catalog is unreachable and we're serving from the last snapshot +
// local decision buffer (see calibrate_server.py's module docstring) ----
function updateOfflineBanner(mode) {
  const el = document.getElementById('offline-banner');
  if (mode === 'offline') {
    el.textContent = '⚠ Catalog busy (a batch pipeline job is running) — showing the last exported snapshot. Your decisions are saved locally and will sync once the catalog frees up (pipe calibrate flush-pending).';
    el.style.display = 'block';
  } else {
    el.style.display = 'none';
  }
}

// ---- stats / progress bar ----
async function refreshStats() {
  const { batch, source } = currentFilters();
  const r = await fetch('/api/stats?' + qs({ batch, source }));
  const s = await r.json();
  updateOfflineBanner(s.mode);
  document.getElementById('s-verified').textContent = s.verified;
  document.getElementById('s-skipped').textContent = s.skipped;
  document.getElementById('s-rejected').textContent = s.rejected;
  document.getElementById('s-flagged').textContent = s.flagged;
  document.getElementById('s-pending').textContent = s.pending;
  document.getElementById('s-total').textContent = s.total;
  const total = Math.max(s.total, 1);
  document.getElementById('pb-verified').style.width = (100 * s.verified / total) + '%';
  document.getElementById('pb-skipped').style.width = (100 * s.skipped / total) + '%';
  document.getElementById('pb-rejected').style.width = (100 * s.rejected / total) + '%';
  document.getElementById('pb-flagged').style.width = (100 * s.flagged / total) + '%';
}

async function refreshHistory() {
  const { batch } = currentFilters();
  const r = await fetch('/api/history?' + qs({ batch, limit: 25 }));
  const { items } = await r.json();
  const list = document.getElementById('history-list');
  list.innerHTML = '';
  for (const it of items) {
    const div = document.createElement('div');
    div.className = 'hist-item';
    const bodyText = (it.decision === 'flagged' || (it.decision === 'rejected' && it.flag_reason))
      ? (it.flag_reason || '(no reason given)')
      : (it.reviewed_text || '(no text)');
    div.innerHTML =
      `<div class="hist-top"><span class="hist-decision ${it.decision}">${it.decision}</span><span>${it.source}</span></div>` +
      `<div class="hist-text">${escapeHtml(bodyText)}</div>`;
    div.onclick = () => openItem(it.id);
    list.appendChild(div);
  }
}

// ---- Cantonese particle quick-insert toolbar ----
function insertAtCursor(str) {
  const box = document.getElementById('text');
  undoStack.push(box.value);
  document.getElementById('undoBtn').disabled = false;
  const start = box.selectionStart ?? box.value.length;
  const end = box.selectionEnd ?? box.value.length;
  box.value = box.value.slice(0, start) + str + box.value.slice(end);
  const pos = start + str.length;
  box.focus();
  box.setSelectionRange(pos, pos);
  rerenderCandidateDiffs();
  refreshJyutpingPreview();
}

function renderParticleToolbar() {
  const bar = document.getElementById('particles');
  for (const p of PARTICLES) {
    const btn = document.createElement('button');
    btn.textContent = p;
    btn.onclick = () => insertAtCursor(p);
    bar.appendChild(btn);
  }
}

// ---- Jyutping validity live preview ----
async function refreshJyutpingPreview() {
  const text = document.getElementById('text').value;
  const el = document.getElementById('jyutping-preview');
  if (!text.trim()) { el.innerHTML = ''; return; }
  const r = await fetch('/api/g2p_preview?' + qs({ text }));
  const p = await r.json();
  const pct = Math.round(p.valid_fraction * 100);
  const cls = p.valid_fraction >= 0.95 ? 'ok' : p.valid_fraction >= 0.80 ? 'warn' : 'bad';
  const badPart = p.bad_tokens.length
    ? ` — invalid: ${p.bad_tokens.map((t) => `<span class="bad-token">${escapeHtml(t)}</span>`).join(' ')}`
    : '';
  el.innerHTML = `jyutping: ${escapeHtml(p.jyutping) || '(none)'} &nbsp; <span class="frac ${cls}">${pct}% valid</span>${badPart}`;
}

async function refreshFilterOptions() {
  const [{ batches }, { sources }] = await Promise.all([
    fetch('/api/batches').then((r) => r.json()),
    fetch('/api/sources').then((r) => r.json()),
  ]);
  const batchSelect = document.getElementById('batchSelect');
  const keepBatch = batchSelect.value;
  batchSelect.querySelectorAll('option[value]:not([value=""])').forEach((o) => o.remove());
  for (const b of batches) {
    const opt = document.createElement('option');
    opt.value = b.sample_batch;
    opt.textContent = `${b.sample_batch} (${b.pending} pending)`;
    batchSelect.appendChild(opt);
  }
  batchSelect.value = keepBatch || DEFAULT_BATCH || '';

  const sourceSelect = document.getElementById('sourceSelect');
  const keepSource = sourceSelect.value;
  sourceSelect.querySelectorAll('option[value]:not([value=""])').forEach((o) => o.remove());
  for (const s of sources) {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    sourceSelect.appendChild(opt);
  }
  sourceSelect.value = keepSource || '';
}

// ---- waveform: decode via Web Audio API, draw peaks, click-to-seek ----
async function drawWaveform(url) {
  const canvas = document.getElementById('waveform');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth || 760;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await (await fetch(url)).arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(buf);
    const data = audioBuffer.getChannelData(0);
    const width = canvas.width, height = canvas.height, amp = height / 2;
    const step = Math.max(1, Math.ceil(data.length / width));
    ctx.fillStyle = 'rgba(108,148,255,0.55)';
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      const base = i * step;
      for (let j = 0; j < step; j++) {
        const datum = data[base + j] || 0;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.fillRect(i, (1 + min) * amp, 1, Math.max(1, (max - min) * amp));
    }
  } catch (err) {
    ctx.fillStyle = 'var(--dim)';
    ctx.font = '12px sans-serif';
    ctx.fillText('(waveform unavailable)', 8, canvas.height / 2);
  }
}

document.getElementById('waveform').addEventListener('click', (e) => {
  const player = document.getElementById('player');
  if (!player.duration) return;
  const rect = e.target.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  player.currentTime = frac * player.duration;
  player.play().catch(() => {});
});

// ---- rendering a review item (shared by loadNext / openItem) ----
function renderItem(item) {
  current = item;
  undoStack = [];
  document.getElementById('undoBtn').disabled = true;
  document.getElementById('flag-box').style.display = 'none';

  const agreementCls = item.agreement < 0.65 ? 'low' : '';
  let metaHtml =
    `<span>${item.source}</span><span>${item.program || ''}</span>` +
    `<span>${item.duration_sec?.toFixed(1)}s</span>` +
    `<span class="agreement-pill ${agreementCls}">agreement ${item.agreement?.toFixed(2)}</span>`;
  if (item.decision && item.decision !== 'pending') {
    metaHtml += `<span class="decision-pill ${item.decision}">${item.decision}</span>`;
    if ((item.decision === 'flagged' || item.decision === 'rejected') && item.flag_reason) {
      metaHtml += `<span style="color:var(--warn)">${escapeHtml(item.flag_reason)}</span>`;
    }
  }
  document.getElementById('meta').innerHTML = metaHtml;

  const audioUrl = '/api/audio?id=' + encodeURIComponent(item.id);
  const player = document.getElementById('player');
  player.src = audioUrl;
  player.loop = document.getElementById('loopBtn').classList.contains('active');
  drawWaveform(audioUrl);

  document.getElementById('text').value = item.reviewed_text || item.best_text || '';

  const cbox = document.getElementById('candidates');
  cbox.innerHTML = '';
  item.candidates.forEach((c, idx) => {
    const div = document.createElement('div');
    div.className = 'cand';
    const diffId = `diff-${idx}`;
    div.innerHTML =
      `<span class="model">${c.model}<br><span style="color:var(--dim)">${c.confidence?.toFixed(2) ?? ''}</span></span>` +
      `<span class="diff" id="${diffId}" data-candidate="${escapeHtml(c.text)}"></span>` +
      `<button class="insert" data-idx="${idx}">Insert<kbd>${idx + 1}</kbd></button>`;
    div.querySelector('button.insert').onclick = () => applyCandidate(c.text);
    cbox.appendChild(div);
  });
  rerenderCandidateDiffs();
  refreshJyutpingPreview();

  document.getElementById('review').style.display = '';
  document.getElementById('done').style.display = 'none';

  if (document.getElementById('autoplayBtn').classList.contains('active')) {
    player.play().catch(() => {});
  }
  document.getElementById('text').focus();
}

// ---- explicit, controlled candidate insertion (never auto-applies on click) ----
function applyCandidate(text) {
  const box = document.getElementById('text');
  undoStack.push(box.value);
  box.value = text;
  document.getElementById('undoBtn').disabled = false;
  rerenderCandidateDiffs();
  refreshJyutpingPreview();
}

function undoInsert() {
  if (!undoStack.length) return;
  document.getElementById('text').value = undoStack.pop();
  document.getElementById('undoBtn').disabled = undoStack.length === 0;
  rerenderCandidateDiffs();
  refreshJyutpingPreview();
}

// ---- navigation ----
async function loadNext() {
  const { batch, source, order } = currentFilters();
  const { tier, min_agreement, code_switch } = currentSampleOptions();
  const r = await fetch('/api/next?' + qs({ batch, source, order, tier, min_agreement, code_switch }));
  const { item, refilled } = await r.json();
  reviewingFromHistory = false;
  if (refilled) {
    showRefillToast(refilled.queued);
    refreshFilterOptions();
  }
  if (!item) {
    document.getElementById('review').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    current = null;
    return;
  }
  visitStack.push(item.id);
  document.getElementById('backBtn').disabled = visitStack.length < 2;
  renderItem(item);
}

async function openItem(id) {
  // Opened from the history panel -- deliberately does NOT touch visitStack,
  // so Back navigation continues to reflect the forward-review sequence.
  const r = await fetch('/api/item?' + qs({ id }));
  const { item } = await r.json();
  if (!item) return;
  reviewingFromHistory = true;
  renderItem(item);
}

async function goBack() {
  if (visitStack.length < 2) return;
  visitStack.pop();               // drop the current item
  const prevId = visitStack[visitStack.length - 1];
  const r = await fetch('/api/item?' + qs({ id: prevId }));
  const { item } = await r.json();
  reviewingFromHistory = false;
  document.getElementById('backBtn').disabled = visitStack.length < 2;
  if (item) renderItem(item);
}

document.getElementById('backBtn').onclick = goBack;

// ---- decisions ----
async function submit(decision, flagReason) {
  if (!current) return;
  const text = document.getElementById('text').value;
  await fetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: current.id, decision, text, flag_reason: flagReason || null,
      sample_batch: current.sample_batch || null, source: current.source || null,
    }),
  });
  await Promise.all([refreshStats(), refreshHistory()]);
  if (reviewingFromHistory) {
    // stay on the history item after re-deciding, just refresh its pill
    openItem(current.id);
  } else {
    loadNext();
  }
}

document.getElementById('verify').onclick = () => submit('verified');
document.getElementById('skip').onclick = () => submit('skipped');
document.getElementById('reject').onclick = () => submit('rejected');
// Mandarin flag (added 2026-07-15): one-click reject with a fixed reason --
// must match MANDARIN_FLAG_REASON in pipeline/nodes/calibrate.py. Submits
// decision='rejected' directly (record_decision excludes the segment from
// the manifest for any 'rejected' decision), so this both flags the issue
// (reason recorded for the top_flag_reasons triage leaderboard) and drops
// the segment, in one click -- no free-text box needed.
document.getElementById('mandarin').onclick = () => submit('rejected', 'mandarin');
// T9 speaker-purity buttons (added 2026-07-17, must match NOT_SINGLE_SPEAKER_FLAG_REASON /
// WRONG_SPEAKER_ID_FLAG_REASON in pipeline/nodes/calibrate.py): two separate buttons for two
// different failure modes -- "multi-speaker" is a real audio defect (excludes, same as
// Mandarin/reject), "wrong speaker ID" is a harmless metadata mislabel (flags only, audio
// stays in the manifest).
document.getElementById('multiSpeaker').onclick = () => submit('rejected', 'not_single_speaker');
document.getElementById('wrongSpeakerId').onclick = () => submit('flagged', 'wrong_speaker_id');
document.getElementById('undoBtn').onclick = undoInsert;

// ---- flag issue (4th decision, does not touch text_verified/tiers) ----
function openFlagBox() {
  document.getElementById('flag-box').style.display = 'flex';
  document.getElementById('flag-reason').value = '';
  document.getElementById('flag-reason').focus();
}
document.getElementById('flag').onclick = openFlagBox;
document.getElementById('flag-cancel').onclick = () => {
  document.getElementById('flag-box').style.display = 'none';
  document.getElementById('text').focus();
};
document.getElementById('flag-confirm').onclick = () => {
  submit('flagged', document.getElementById('flag-reason').value.trim());
};
document.getElementById('flag-reason').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); document.getElementById('flag-confirm').click(); }
  else if (e.key === 'Escape') { document.getElementById('flag-cancel').click(); }
});

// ---- queue refill (manual + reacting to auto-refill from /api/next) ----
async function refillQueue(n) {
  const r = await fetch('/api/refill', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n: n || 200, ...currentSampleOptions() }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    alert('Refill failed: ' + (body.error || r.statusText));
    return;
  }
  await refreshFilterOptions();
  await Promise.all([refreshStats(), refreshHistory()]);
  loadNext();
}
document.getElementById('refillBtn').onclick = () => refillQueue(200);
document.getElementById('doneRefillBtn').onclick = () => refillQueue(200);

function showRefillToast(n) {
  const el = document.getElementById('refill-toast');
  el.textContent = `Auto-refilled: queued ${n} more segments`;
  el.style.display = 'block';
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => { el.style.display = 'none'; }, 4000);
}

// ---- sample summary dashboard (explicit refresh, not auto-polling) ----
function barRow(label, value, total, color) {
  const pct = total > 0 ? (100 * value / total) : 0;
  return `<div class="bar-row"><span class="bar-label">${escapeHtml(label)}</span>` +
         `<span class="bar-track"><span style="width:${pct}%;background:${color}"></span></span>` +
         `<span class="bar-value">${value}</span></div>`;
}

async function refreshSummary() {
  const { batch } = currentFilters();
  const btn = document.getElementById('summaryRefresh');
  btn.disabled = true; btn.textContent = 'Loading…';
  const r = await fetch('/api/summary?' + qs({ batch }));
  const s = await r.json();
  btn.disabled = false; btn.textContent = '↻ Refresh';

  const dc = s.decision_counts;
  const decisionCard = `
    <div class="stat-card"><h3>Decisions (${dc.total} sampled)</h3>
      ${barRow('verified', dc.verified, dc.total, 'var(--good)')}
      ${barRow('skipped', dc.skipped, dc.total, 'var(--dim)')}
      ${barRow('rejected', dc.rejected, dc.total, 'var(--bad)')}
      ${barRow('flagged', dc.flagged, dc.total, 'var(--warn)')}
      ${barRow('pending', dc.pending, dc.total, 'var(--accent)')}
    </div>`;

  const sourceRows = Object.entries(s.by_source).map(([source, counts]) => {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    return `<div style="margin-bottom:0.6rem"><div class="bar-row"><span class="bar-label"><b>${escapeHtml(source)}</b></span><span class="bar-value">${total}</span></div>` +
      barRow('verified', counts.verified || 0, total, 'var(--good)') +
      barRow('flagged', counts.flagged || 0, total, 'var(--warn)') + '</div>';
  }).join('') || '<span style="color:var(--dim)">no data</span>';
  const sourceCard = `<div class="stat-card"><h3>By source</h3>${sourceRows}</div>`;

  const agr = s.avg_agreement_by_decision;
  const agreementCard = `
    <div class="stat-card"><h3>Avg ASR agreement by decision</h3>
      ${Object.entries(agr).map(([d, v]) => `<div class="bar-row"><span class="bar-label">${d}</span><span class="bar-value">${v}</span></div>`).join('') || '<span style="color:var(--dim)">no data</span>'}
    </div>`;

  const editCard = `
    <div class="stat-card"><h3>Human correction size (verified)</h3>
      <div class="big-number">${s.avg_edit_distance_verified ?? '—'}</div>
      <div class="sub">avg characters changed vs ASR text, n=${s.verified_edit_sample_size}</div>
    </div>`;

  const flagRows = s.top_flag_reasons.length
    ? s.top_flag_reasons.map((f) => `<div class="bar-row"><span class="bar-label" style="width:auto;flex:1">${escapeHtml(f.reason)}</span><span class="bar-value">${f.count}</span></div>`).join('')
    : '<span style="color:var(--dim)">none yet</span>';
  const flagCard = `<div class="stat-card"><h3>Top flag reasons</h3>${flagRows}</div>`;

  document.getElementById('summary-body').innerHTML =
    decisionCard + sourceCard + agreementCard + editCard + flagCard;
}
document.getElementById('summaryRefresh').onclick = refreshSummary;

// ---- playback controls ----
document.getElementById('loopBtn').onclick = (e) => {
  e.target.classList.toggle('active');
  document.getElementById('player').loop = e.target.classList.contains('active');
};
document.getElementById('autoplayBtn').onclick = (e) => {
  e.target.classList.toggle('active');
  localStorage.setItem('calib_autoplay', e.target.classList.contains('active') ? '1' : '0');
};
if (localStorage.getItem('calib_autoplay') === '1') {
  document.getElementById('autoplayBtn').classList.add('active');
}
document.querySelectorAll('.pbtn[data-speed]').forEach((btn) => {
  btn.onclick = () => {
    document.getElementById('player').playbackRate = parseFloat(btn.dataset.speed);
    document.querySelectorAll('.pbtn[data-speed]').forEach((b) => {
      b.style.color = ''; b.style.borderColor = '';
    });
    btn.style.color = 'var(--accent)'; btn.style.borderColor = 'var(--accent)';
  };
});

// ---- filters ----
['batchSelect', 'sourceSelect', 'orderSelect'].forEach((id) => {
  document.getElementById(id).onchange = () => { refreshStats(); refreshHistory(); loadNext(); };
});

// ---- live diff + Jyutping preview re-render as you type / paste corrections manually ----
document.getElementById('text').addEventListener('input', () => {
  clearTimeout(window._diffTimer);
  window._diffTimer = setTimeout(() => {
    rerenderCandidateDiffs();
    refreshJyutpingPreview();
  }, 150);
});

// ---- keyboard shortcuts ----
document.addEventListener('keydown', (e) => {
  const inBox = document.activeElement === document.getElementById('text');
  const inFlagReason = document.activeElement === document.getElementById('flag-reason');
  const inMinAgreement = document.activeElement === document.getElementById('minAgreementInput');
  if (inFlagReason || inMinAgreement) return;  // these have their own input handling
  if (e.key === 'Enter' && inBox && !e.shiftKey) { e.preventDefault(); submit('verified'); }
  else if (e.key.toLowerCase() === 's' && !inBox) { submit('skipped'); }
  else if (e.key.toLowerCase() === 'd' && !inBox) { submit('rejected'); }
  else if (e.key.toLowerCase() === 'm' && !inBox) { submit('rejected', 'mandarin'); }
  else if (e.key.toLowerCase() === 'n' && !inBox) { submit('rejected', 'not_single_speaker'); }
  else if (e.key.toLowerCase() === 'w' && !inBox) { submit('flagged', 'wrong_speaker_id'); }
  else if (e.key.toLowerCase() === 'f' && !inBox) { openFlagBox(); }
  else if (e.key === ' ' && !inBox) {
    e.preventDefault();
    const p = document.getElementById('player');
    p.paused ? p.play() : p.pause();
  } else if (e.key === 'b' && !inBox) { document.getElementById('backBtn').click(); }
  else if (/^[1-4]$/.test(e.key) && !inBox && current) {
    const idx = parseInt(e.key, 10) - 1;
    if (current.candidates[idx]) applyCandidate(current.candidates[idx].text);
  }
});

(async function init() {
  renderParticleToolbar();
  await refreshFilterOptions();
  await Promise.all([refreshStats(), refreshHistory()]);
  await loadNext();
})();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--batch", default=None, help="pre-select this calibrate.sample run_id in the batch dropdown")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Fail fast if the catalog doesn't exist yet -- this connection is opened
    # and closed immediately, never held (see module docstring).
    connect_ro(CATALOG_PATH).close()

    handler = _build_app(args.batch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    log.info(f"calibrate_server: listening on http://127.0.0.1:{args.port}/ (default batch={args.batch or 'all'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
