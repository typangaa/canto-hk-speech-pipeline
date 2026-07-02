#!/usr/bin/env python3
"""
scripts/10_enrich_manifest.py
Enrich manifest.jsonl with the fields needed for a metadata-only ("reconstruction
recipe") Hugging Face release: per-segment source URL, source identifier, and the
[start, end] offsets into the original source media so users can re-download and
re-slice the audio themselves (no copyrighted audio is redistributed).

Usage:
  python scripts/10_enrich_manifest.py [--dry-run]
  python scripts/10_enrich_manifest.py --checksum --shard n/m   # 2nd pass: PCM sha256

Inputs:
  metadata/manifest.jsonl                       (built by 09_manifest.py)
  metadata/downloaded.jsonl                     (download log: id -> source_url + meta)
  data/segments/{youtube,rthk,podcast}/**/*_segments.jsonl   (start/end offsets)

Outputs (pass 1, metadata):
  metadata/manifest_release.jsonl   enriched + reconstructable rows only (sorted by id)
  metadata/excluded_no_url.jsonl    rows dropped because no source_url could be recovered

URL recovery (98.5% of segments):
  - youtube / rthk-from-youtube : 11-char video id at end of the raw filename stem
                                  (ids may contain '_' / '-', so we slice the last 11
                                  chars, not split on '_'); a stray '.orig' suffix is
                                  stripped first.
  - rthk archive / podcast      : 8-hex id token -> looked up in downloaded.jsonl.
  Anything unresolved is written to excluded_no_url.jsonl and kept OUT of the release
  manifest, so every released row is guaranteed reconstructable.

Pass 2 (--checksum) appends `audio_sha256` (sha256 of the int16 PCM samples of the
current filtered wav) to manifest_release.jsonl, so reconstruct.py can verify a rebuilt
segment matches the original. Sharded because it reads every wav.
"""

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "10_enrich.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

META = ROOT / "metadata"
MANIFEST = META / "manifest.jsonl"
DOWNLOADED = META / "downloaded.jsonl"
RELEASE = META / "manifest_release.jsonl"
EXCLUDED = META / "excluded_no_url.jsonl"
SEG_SOURCES = ["youtube", "rthk", "podcast"]

_SEG_RE = re.compile(r"_seg\d+$")
_HEX8_RE = re.compile(r"[0-9a-f]{8}")
_YTID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def raw_stem(audio_path: str) -> str:
    """filtered/.../<stem>_segNNNNN.wav  ->  <stem>   (handles a stray .orig)."""
    base = audio_path.rsplit("/", 1)[-1]
    base = base[:-4] if base.endswith(".wav") else base
    stem = _SEG_RE.sub("", base)
    if stem.endswith(".orig"):
        stem = stem[:-5]
    return stem


def seg_key(audio_path: str) -> str:
    """Key into the segment index: the seg filename without extension."""
    base = audio_path.rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".wav") else base


def url_kind(url: str) -> str:
    if "youtube" in url or "youtu.be" in url:
        return "youtube"
    if ".mp4" in url or "archive.rthk" in url:
        return "archive_mp4"
    return "podcast_rss"


def load_downloaded() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with open(DOWNLOADED) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("id"):
                    by_id[d["id"]] = d
            except Exception:
                pass
    log.info(f"downloaded.jsonl: {len(by_id)} ids")
    return by_id


def load_segment_index() -> dict[str, dict]:
    """seg-filename(no ext) -> {start_sec, end_sec, source_wav, sample_rate}."""
    idx: dict[str, dict] = {}
    for src in SEG_SOURCES:
        root = ROOT / "data" / "segments" / src
        # NOTE: data/segments/{rthk,podcast} are symlinks; rglob won't descend into a
        # symlinked dir, but iterating the explicit per-source root DOES follow it.
        n = 0
        for jl in root.rglob("*_segments.jsonl"):
            try:
                with open(jl) as f:
                    for line in f:
                        rec = json.loads(line)
                        base = rec["seg_path"].rsplit("/", 1)[-1]
                        key = base[:-4] if base.endswith(".wav") else base
                        idx[key] = {
                            "start_sec": rec.get("start_sec"),
                            "end_sec": rec.get("end_sec"),
                            "source_wav": rec.get("source_wav", "").rsplit("/", 1)[-1],
                            "sample_rate": rec.get("sample_rate", 48000),
                        }
                        n += 1
            except Exception as exc:
                log.warning(f"bad segment jsonl {jl.name}: {exc}")
        log.info(f"  segments/{src}: indexed {n} records")
    log.info(f"segment index: {len(idx)} unique segments")
    return idx


def resolve_url(stem: str, by_id: dict[str, dict]) -> Optional[tuple]:
    """-> (source_url, source_id, source_kind, downloaded_rec|None) or None."""
    last = stem.rsplit("_", 1)[-1]
    if _HEX8_RE.fullmatch(last) and last in by_id:
        d = by_id[last]
        return d["source_url"], last, url_kind(d["source_url"]), d
    # youtube/rthk video id = last 11 chars (may contain '_'/'-'), preceded by '_'
    if len(stem) >= 12 and stem[-12] == "_":
        vid = stem[-11:]
        if _YTID_RE.fullmatch(vid):
            return f"https://www.youtube.com/watch?v={vid}", vid, "youtube", None
    # fallback: any token is a known hash id
    for tok in stem.split("_"):
        if tok in by_id:
            d = by_id[tok]
            return d["source_url"], tok, url_kind(d["source_url"]), d
    return None


def pcm_sha256(wav_path: Path) -> Optional[str]:
    import soundfile as sf
    import numpy as np
    try:
        data, _ = sf.read(str(wav_path), dtype="int16", always_2d=False)
        return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    except Exception as exc:
        log.error(f"checksum failed {wav_path.name}: {exc}")
        return None


# --------------------------------------------------------------------------- #
def pass_metadata(dry_run: bool) -> None:
    by_id = load_downloaded()
    seg_idx = load_segment_index()

    kept, dropped = [], []
    method = Counter()
    miss_seg = 0
    with open(MANIFEST) as f:
        for line in f:
            e = json.loads(line)
            ap = e["audio_path"]
            res = resolve_url(raw_stem(ap), by_id)
            if not res:
                dropped.append(e)
                method["UNRESOLVED"] += 1
                continue
            url, sid, kind, drec = res
            seg = seg_idx.get(seg_key(ap))
            if not seg or seg.get("start_sec") is None:
                miss_seg += 1
                dropped.append(e)
                method["NO_OFFSETS"] += 1
                continue

            e["source_url"] = url
            e["source_id"] = sid
            e["source_kind"] = kind
            e["start_sec"] = round(float(seg["start_sec"]), 3)
            e["end_sec"] = round(float(seg["end_sec"]), 3)
            e["source_sample_rate"] = int(seg.get("sample_rate") or 48000)
            e["reconstructable"] = True
            # backfill empty descriptive fields from the download log
            if drec:
                if not e.get("program"):
                    e["program"] = drec.get("program", "") or ""
                if not e.get("domain") or e["domain"] == "other":
                    e["domain"] = drec.get("domain") or e.get("domain", "other")
                if not e.get("style") or e["style"] == "formal":
                    e["style"] = drec.get("style") or e.get("style", "formal")
                e["title"] = drec.get("title", "") or ""
                e["pub_date"] = str(drec.get("pub_date", "") or "")
            else:
                e.setdefault("title", "")
                e.setdefault("pub_date", "")
            kept.append(e)
            method[kind] += 1

    # report
    tot = len(kept) + len(dropped)
    log.info(f"resolved: {len(kept)}/{tot} ({100*len(kept)/tot:.2f}%)  dropped {len(dropped)}")
    for k, v in method.most_common():
        log.info(f"  {k}: {v}")
    by_src = defaultdict(lambda: [0, 0])
    for e in kept:
        by_src[e["source"]][0] += 1
    for e in dropped:
        by_src[e["source"]][1] += 1
    for src, (c, d) in sorted(by_src.items()):
        log.info(f"  {src}: kept {c}  dropped {d}")

    if dry_run:
        print(f"\n[DRY-RUN] would keep {len(kept)}, drop {len(dropped)} "
              f"(no-offset cases: {miss_seg})")
        return

    with open(RELEASE, "w") as f:
        for e in sorted(kept, key=lambda x: x["id"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(EXCLUDED, "w") as f:
        for e in dropped:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nRelease manifest: {len(kept)} entries -> {RELEASE}")
    print(f"Excluded (no url/offsets): {len(dropped)} -> {EXCLUDED}")
    print(f"Log: {LOG_PATH}")


def pass_checksum(shard: Optional[str]) -> None:
    rows = [json.loads(l) for l in open(RELEASE)]
    if shard:
        n, m = map(int, shard.split("/"))
        mine = [(i, r) for i, r in enumerate(rows) if i % m == n]
    else:
        mine = list(enumerate(rows))
    log.info(f"checksum: {len(mine)} of {len(rows)} rows (shard {shard or 'all'})")
    out = META / (f"checksums_{shard.replace('/', '-')}.jsonl" if shard
                  else "checksums_all.jsonl")
    done = 0
    with open(out, "w") as f:
        for i, r in mine:
            h = pcm_sha256(Path(r["audio_path"]))
            f.write(json.dumps({"id": r["id"], "audio_sha256": h}) + "\n")
            done += 1
            if done % 2000 == 0:
                log.info(f"  {done}/{len(mine)} ...")
    print(f"Checksums -> {out} ({done} rows)")


def merge_checksums() -> None:
    cks = {}
    for p in META.glob("checksums_*.jsonl"):
        for l in open(p):
            d = json.loads(l)
            if d.get("audio_sha256"):
                cks[d["id"]] = d["audio_sha256"]
    rows = [json.loads(l) for l in open(RELEASE)]
    n = 0
    with open(RELEASE, "w") as f:
        for r in rows:
            if r["id"] in cks:
                r["audio_sha256"] = cks[r["id"]]
                n += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged audio_sha256 into {n}/{len(rows)} rows of {RELEASE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--checksum", action="store_true", help="pass 2: compute PCM sha256")
    ap.add_argument("--shard", help="n/m for the checksum pass")
    ap.add_argument("--merge-checksums", action="store_true",
                    help="merge checksums_*.jsonl back into manifest_release.jsonl")
    args = ap.parse_args()
    if args.merge_checksums:
        merge_checksums()
    elif args.checksum:
        pass_checksum(args.shard)
    else:
        pass_metadata(args.dry_run)


if __name__ == "__main__":
    main()
