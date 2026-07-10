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
from pipeline.nodes.calibrate import next_pending, queue_stats, record_decision

log = logging.getLogger(__name__)

_AUDIO_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
}

_write_lock = threading.Lock()


def _build_app(conn, sample_batch: str | None):
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
            if parsed.path == "/":
                body = _PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/next":
                with _write_lock:
                    item = next_pending(conn, sample_batch)
                self._send_json({"item": item})
                return

            if parsed.path == "/api/stats":
                with _write_lock:
                    stats = queue_stats(conn, sample_batch)
                self._send_json(stats)
                return

            if parsed.path == "/api/audio":
                qs = parse_qs(parsed.query)
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
                    stats = queue_stats(conn, sample_batch)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "stats": stats})

    return Handler


_PAGE_HTML = """<!doctype html>
<html lang="yue">
<head>
<meta charset="utf-8">
<title>Canto Corpus — Text Calibration</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, "Noto Sans HK", "PingFang HK", sans-serif;
         max-width: 780px; margin: 2rem auto; padding: 0 1.5rem;
         background: #14161a; color: #e8e6e1; }
  h1 { font-size: 1.1rem; font-weight: 600; color: #9aa0a8; letter-spacing: 0.02em; }
  #stats { font-variant-numeric: tabular-nums; color: #8b939e; margin-bottom: 1.5rem; }
  #stats b { color: #e8e6e1; }
  .card { background: #1c1f25; border: 1px solid #2b2f37; border-radius: 10px;
          padding: 1.5rem; }
  .meta { display: flex; gap: 1.2rem; color: #8b939e; font-size: 0.85rem; margin-bottom: 1rem;
          font-variant-numeric: tabular-nums; }
  audio { width: 100%; margin-bottom: 1rem; }
  .candidates { display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1rem; }
  .cand { display: flex; gap: 0.6rem; padding: 0.5rem 0.7rem; background: #14161a;
          border: 1px solid #2b2f37; border-radius: 6px; cursor: pointer; font-size: 0.95rem; }
  .cand:hover { border-color: #4a7dfc; }
  .cand .model { color: #6c94ff; font-size: 0.78rem; min-width: 90px; flex-shrink: 0; }
  textarea { width: 100%; box-sizing: border-box; min-height: 4.5rem; font-size: 1.15rem;
             background: #14161a; color: #e8e6e1; border: 1px solid #2b2f37; border-radius: 6px;
             padding: 0.7rem; margin-bottom: 1rem; resize: vertical; }
  .actions { display: flex; gap: 0.6rem; }
  button { font-size: 0.95rem; padding: 0.6rem 1.1rem; border-radius: 6px; border: none;
           cursor: pointer; font-weight: 500; }
  #verify { background: #3d8f5c; color: white; }
  #skip { background: #2b2f37; color: #e8e6e1; }
  #reject { background: #8a3a3a; color: white; }
  .hint { color: #5c6470; font-size: 0.78rem; margin-top: 0.8rem; }
  #done { display: none; color: #8b939e; }
</style>
</head>
<body>
<h1>CANTO CORPUS &mdash; TEXT CALIBRATION</h1>
<div id="stats">loading…</div>
<div class="card" id="review">
  <div class="meta" id="meta"></div>
  <audio id="player" controls></audio>
  <div class="candidates" id="candidates"></div>
  <textarea id="text" autofocus></textarea>
  <div class="actions">
    <button id="verify">Verify (Enter)</button>
    <button id="skip">Skip (S)</button>
    <button id="reject">Reject (D)</button>
  </div>
  <div class="hint">Click a candidate to copy it into the box. Space replays audio when the box isn't focused.</div>
</div>
<div id="done">Queue empty &mdash; no pending segments left.</div>
<script>
let current = null;

async function refreshStats() {
  const r = await fetch('/api/stats');
  const s = await r.json();
  document.getElementById('stats').innerHTML =
    `<b>${s.verified}</b> verified &middot; <b>${s.skipped}</b> skipped &middot; ` +
    `<b>${s.rejected}</b> rejected &middot; <b>${s.pending}</b> pending &middot; ${s.total} total`;
}

async function loadNext() {
  const r = await fetch('/api/next');
  const { item } = await r.json();
  if (!item) {
    document.getElementById('review').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    current = null;
    return;
  }
  current = item;
  document.getElementById('meta').textContent =
    `${item.source} · ${item.program || ''} · ${item.duration_sec?.toFixed(1)}s · agreement ${item.agreement?.toFixed(2)}`;
  document.getElementById('player').src = '/api/audio?id=' + encodeURIComponent(item.id);
  document.getElementById('text').value = item.best_text || '';
  const cbox = document.getElementById('candidates');
  cbox.innerHTML = '';
  for (const c of item.candidates) {
    const div = document.createElement('div');
    div.className = 'cand';
    div.innerHTML = `<span class="model">${c.model}</span><span>${c.text}</span>`;
    div.onclick = () => { document.getElementById('text').value = c.text; };
    cbox.appendChild(div);
  }
  document.getElementById('text').focus();
}

async function submit(decision) {
  if (!current) return;
  const text = document.getElementById('text').value;
  await fetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: current.id, decision, text }),
  });
  await refreshStats();
  await loadNext();
}

document.getElementById('verify').onclick = () => submit('verified');
document.getElementById('skip').onclick = () => submit('skipped');
document.getElementById('reject').onclick = () => submit('rejected');

document.addEventListener('keydown', (e) => {
  const inBox = document.activeElement === document.getElementById('text');
  if (e.key === 'Enter' && inBox && !e.shiftKey) { e.preventDefault(); submit('verified'); }
  else if (e.key.toLowerCase() === 's' && !inBox) { submit('skipped'); }
  else if (e.key.toLowerCase() === 'd' && !inBox) { submit('rejected'); }
  else if (e.key === ' ' && !inBox) { e.preventDefault(); document.getElementById('player').play(); }
});

refreshStats();
loadNext();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--batch", default=None, help="restrict review to one calibrate.sample run_id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conn = connect()
    handler = _build_app(conn, args.batch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    log.info(f"calibrate_server: listening on http://127.0.0.1:{args.port}/ (batch={args.batch or 'all'})")
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
