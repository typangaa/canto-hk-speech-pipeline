"""
pipeline/tools/calibrate_server.py
Local browser UI for human text-verification calibration. NOT a DAG node --
this is a long-running interactive server, not an idempotent-discovery batch
step. It reads the sample queue that `pipe run calibrate.sample` writes into
`calibration_review` and lets the owner listen to each segment, compare ASR
candidates, and record a verified/skipped/rejected decision. See
pipeline/nodes/calibrate.py's module docstring for why this exists (owner
decision 2026-07-10: text_verified/gold was structurally dead in the DAG).

Usage:
    .venv/bin/python -m pipeline.tools.calibrate_server [--port 8420] [--batch <sample_batch id>]

Then open http://localhost:8420/ in a browser on the same machine.
`--batch` only pre-selects the batch-jump dropdown's default value -- the
browser UI can switch batch/source/order filters live via query params, it
is not fixed at server startup.

Single-writer note: this process holds ONE read-write DuckDB connection for
its whole lifetime (same rule as every other node -- see catalog.py's
module docstring). Stop the server before running any other `pipe run`
node, and vice versa.
"""

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pipeline.catalog.catalog import connect
from pipeline.nodes.calibrate import (
    get_item,
    list_batches,
    list_history,
    list_sources,
    next_pending,
    queue_stats,
    record_decision,
)

log = logging.getLogger(__name__)

_AUDIO_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
}

_write_lock = threading.Lock()


def _build_app(conn, default_batch: str | None):
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
                with _write_lock:
                    item = next_pending(conn, batch, source, order)
                self._send_json({"item": item})
                return

            if parsed.path == "/api/item":
                seg_id = qs.get("id", [None])[0]
                if not seg_id:
                    self.send_error(400, "missing id")
                    return
                with _write_lock:
                    item = get_item(conn, seg_id)
                self._send_json({"item": item})
                return

            if parsed.path == "/api/history":
                batch = qs.get("batch", [None])[0] or None
                limit = int(qs.get("limit", ["20"])[0])
                with _write_lock:
                    items = list_history(conn, batch, limit)
                self._send_json({"items": items})
                return

            if parsed.path == "/api/batches":
                with _write_lock:
                    batches = list_batches(conn)
                self._send_json({"batches": batches})
                return

            if parsed.path == "/api/sources":
                with _write_lock:
                    sources = list_sources(conn)
                self._send_json({"sources": sources})
                return

            if parsed.path == "/api/stats":
                batch = qs.get("batch", [None])[0] or None
                source = qs.get("source", [None])[0] or None
                with _write_lock:
                    stats = queue_stats(conn, batch, source)
                self._send_json(stats)
                return

            if parsed.path == "/api/audio":
                seg_id = qs.get("id", [None])[0]
                if not seg_id:
                    self.send_error(400, "missing id")
                    return
                with _write_lock:
                    row = conn.execute(
                        "SELECT s.audio_path FROM calibration_review c "
                        "JOIN segments s ON c.id = s.id WHERE c.id = ?",
                        [seg_id],
                    ).fetchone()
                if row is None:
                    self.send_error(404, "unknown id")
                    return
                audio_path = Path(row[0])
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
            if self.path != "/api/submit":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length))
                seg_id = payload["id"]
                decision = payload["decision"]
                text = payload.get("text")
            except (KeyError, json.JSONDecodeError):
                self.send_error(400, "malformed request body")
                return

            try:
                with _write_lock:
                    record_decision(conn, seg_id, decision, text)
                    stats = queue_stats(conn)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "stats": stats})

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
  .filters { display: flex; gap: 0.5rem; margin-left: auto; flex-wrap: wrap; }
  select { background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
           border-radius: 6px; padding: 0.35rem 0.5rem; font-size: 0.82rem; }

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

  /* ---- audio + playback controls ---- */
  .player-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap; }
  audio { flex: 1 1 260px; height: 34px; }
  .pbtn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
          border-radius: 6px; padding: 0.3rem 0.55rem; font-size: 0.78rem; cursor: pointer; }
  .pbtn.active { color: var(--accent); border-color: var(--accent); }

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

  .actions { display: flex; gap: 0.6rem; }
  button.action { font-size: 0.95rem; padding: 0.6rem 1.1rem; border-radius: 6px; border: none;
                  cursor: pointer; font-weight: 500; }
  #verify { background: var(--good); color: white; }
  #skip { background: var(--panel-2); color: var(--text); border: 1px solid var(--border) !important; }
  #reject { background: var(--bad); color: white; }
  .hint { color: var(--dim); font-size: 0.74rem; margin-top: 0.8rem; }
  #done { display: none; color: var(--muted); }

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
  .hist-item .hist-text { color: var(--text); overflow: hidden; text-overflow: ellipsis;
                           white-space: nowrap; }
  #backBtn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
             border-radius: 6px; padding: 0.3rem 0.6rem; font-size: 0.78rem; cursor: pointer;
             margin-bottom: 0.6rem; }
  #backBtn:disabled { opacity: 0.35; cursor: default; }
</style>
</head>
<body>
<h1>CANTO CORPUS &mdash; TEXT CALIBRATION</h1>

<div id="topbar">
  <div class="stat-group">
    <span class="verified"><b id="s-verified">0</b> verified</span>
    <span><b id="s-skipped">0</b> skipped</span>
    <span class="rejected"><b id="s-rejected">0</b> rejected</span>
    <span><b id="s-pending">0</b> pending</span>
    <span style="color:var(--dim)">/ <span id="s-total">0</span></span>
  </div>
  <div id="progress-bar">
    <span id="pb-verified" style="background:var(--good)"></span>
    <span id="pb-skipped" style="background:var(--dim)"></span>
    <span id="pb-rejected" style="background:var(--bad)"></span>
  </div>
  <div class="filters">
    <select id="batchSelect"><option value="">All batches</option></select>
    <select id="sourceSelect"><option value="">All sources</option></select>
    <select id="orderSelect">
      <option value="queued">Queued order</option>
      <option value="agreement_asc">Lowest agreement first</option>
      <option value="agreement_desc">Highest agreement first</option>
    </select>
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
      <div class="candidates" id="candidates"></div>
      <textarea id="text" autofocus></textarea>
      <div class="editor-row">
        <span>Click a candidate's <b>Insert</b> button (or press 1–4) to copy it in — the text box never changes on its own.</span>
        <button id="undoBtn" disabled>↩ Undo insert</button>
      </div>
      <div class="actions">
        <button class="action" id="verify">Verify (Enter)</button>
        <button class="action" id="skip">Skip (S)</button>
        <button class="action" id="reject">Reject (D)</button>
      </div>
      <div class="hint">Space replays/pauses audio when the text box isn't focused. Highlighted diff: <ins>green</ins> = candidate adds this, <del>red</del> = candidate removes this (relative to the box's current text).</div>
    </div>
    <div id="done">Queue empty for this filter — no pending segments left.</div>
  </div>

  <div id="history">
    <button id="backBtn" disabled>← Back to previous</button>
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

function currentFilters() {
  return {
    batch: document.getElementById('batchSelect').value,
    source: document.getElementById('sourceSelect').value,
    order: document.getElementById('orderSelect').value,
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

// ---- stats / progress bar ----
async function refreshStats() {
  const { batch, source } = currentFilters();
  const r = await fetch('/api/stats?' + qs({ batch, source }));
  const s = await r.json();
  document.getElementById('s-verified').textContent = s.verified;
  document.getElementById('s-skipped').textContent = s.skipped;
  document.getElementById('s-rejected').textContent = s.rejected;
  document.getElementById('s-pending').textContent = s.pending;
  document.getElementById('s-total').textContent = s.total;
  const total = Math.max(s.total, 1);
  document.getElementById('pb-verified').style.width = (100 * s.verified / total) + '%';
  document.getElementById('pb-skipped').style.width = (100 * s.skipped / total) + '%';
  document.getElementById('pb-rejected').style.width = (100 * s.rejected / total) + '%';
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
    div.innerHTML =
      `<div class="hist-top"><span class="hist-decision ${it.decision}">${it.decision}</span><span>${it.source}</span></div>` +
      `<div class="hist-text">${escapeHtml(it.reviewed_text || '(no text)')}</div>`;
    div.onclick = () => openItem(it.id);
    list.appendChild(div);
  }
}

async function refreshFilterOptions() {
  const [{ batches }, { sources }] = await Promise.all([
    fetch('/api/batches').then((r) => r.json()),
    fetch('/api/sources').then((r) => r.json()),
  ]);
  const batchSelect = document.getElementById('batchSelect');
  for (const b of batches) {
    const opt = document.createElement('option');
    opt.value = b.sample_batch;
    opt.textContent = `${b.sample_batch} (${b.pending} pending)`;
    batchSelect.appendChild(opt);
  }
  if (DEFAULT_BATCH) batchSelect.value = DEFAULT_BATCH;
  const sourceSelect = document.getElementById('sourceSelect');
  for (const s of sources) {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    sourceSelect.appendChild(opt);
  }
}

// ---- rendering a review item (shared by loadNext / openItem) ----
function renderItem(item) {
  current = item;
  undoStack = [];
  document.getElementById('undoBtn').disabled = true;

  const agreementCls = item.agreement < 0.65 ? 'low' : '';
  let metaHtml =
    `<span>${item.source}</span><span>${item.program || ''}</span>` +
    `<span>${item.duration_sec?.toFixed(1)}s</span>` +
    `<span class="agreement-pill ${agreementCls}">agreement ${item.agreement?.toFixed(2)}</span>`;
  if (item.decision && item.decision !== 'pending') {
    metaHtml += `<span class="decision-pill ${item.decision}">${item.decision}</span>`;
  }
  document.getElementById('meta').innerHTML = metaHtml;

  const player = document.getElementById('player');
  player.src = '/api/audio?id=' + encodeURIComponent(item.id);
  player.loop = document.getElementById('loopBtn').classList.contains('active');

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
}

function undoInsert() {
  if (!undoStack.length) return;
  document.getElementById('text').value = undoStack.pop();
  document.getElementById('undoBtn').disabled = undoStack.length === 0;
  rerenderCandidateDiffs();
}

// ---- navigation ----
async function loadNext() {
  const { batch, source, order } = currentFilters();
  const r = await fetch('/api/next?' + qs({ batch, source, order }));
  const { item } = await r.json();
  reviewingFromHistory = false;
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
async function submit(decision) {
  if (!current) return;
  const text = document.getElementById('text').value;
  await fetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: current.id, decision, text }),
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
document.getElementById('undoBtn').onclick = undoInsert;

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

// ---- live diff re-render as you type / paste corrections manually ----
document.getElementById('text').addEventListener('input', () => {
  clearTimeout(window._diffTimer);
  window._diffTimer = setTimeout(rerenderCandidateDiffs, 120);
});

// ---- keyboard shortcuts ----
document.addEventListener('keydown', (e) => {
  const inBox = document.activeElement === document.getElementById('text');
  if (e.key === 'Enter' && inBox && !e.shiftKey) { e.preventDefault(); submit('verified'); }
  else if (e.key.toLowerCase() === 's' && !inBox) { submit('skipped'); }
  else if (e.key.toLowerCase() === 'd' && !inBox) { submit('rejected'); }
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

    conn = connect()
    handler = _build_app(conn, args.batch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    log.info(f"calibrate_server: listening on http://127.0.0.1:{args.port}/ (default batch={args.batch or 'all'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
