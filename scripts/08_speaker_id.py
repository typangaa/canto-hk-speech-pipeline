#!/usr/bin/env python3
"""
scripts/08_speaker_id.py
Cross-file speaker clustering using ECAPA-TDNN embeddings (SpeechBrain).
Usage: python scripts/08_speaker_id.py --source [rthk|youtube|podcast|all] [--dry-run]

Pipeline:
  1. Extract d-vector embedding per filtered segment (ECAPA-TDNN)
  2. Cluster across all segments (agglomerative, cosine distance)
  3. Write speaker_id (e.g. rthk_001) and gender (if model available) to *.speaker.json
  4. Write summary to metadata/speaker_report.json

Gender estimation: not automatic — left as "unknown" unless manually corrected in calibration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "08_speaker_id.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

FILTERED_DIR = ROOT / "data" / "filtered"
SPEAKER_REPORT_PATH = ROOT / "metadata" / "speaker_report.json"
TARGET_SR = 48000

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        log.info("Loading ECAPA-TDNN speaker encoder ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=str(ROOT / ".cache" / "speechbrain"),
        )
        log.info(f"Encoder loaded on {device}")
    return _encoder


def extract_embedding(wav_path: Path) -> np.ndarray:
    import torchaudio
    encoder = get_encoder()
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    with torch.no_grad():
        emb = encoder.encode_batch(wav)
    return emb.squeeze().cpu().numpy()


# Agglomerative clustering is O(n²) memory (full pairwise matrix). Above this many
# embeddings it would OOM (e.g. 280k² × 8B ≈ 627 GB), so we cluster a random sample
# then assign every point to its nearest sample-centroid by cosine similarity.
_CLUSTER_SAMPLE_MAX = int(__import__("os").environ.get("SPK_CLUSTER_SAMPLE_MAX", "12000"))


def cluster_embeddings(
    embeddings: np.ndarray,
    source_prefix: str,
    threshold: float = 0.25,
) -> np.ndarray:
    """Cluster by cosine distance. Exact agglomerative for small N; scalable
    sample-then-assign for large N (keeps memory bounded). Returns label per row."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    emb_norm = normalize(embeddings)
    n = len(emb_norm)
    if n < 2:
        return np.zeros(n, dtype=int)

    def _agglom(x: np.ndarray) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average",
        ).fit_predict(x)

    if n <= _CLUSTER_SAMPLE_MAX:
        return _agglom(emb_norm)

    # --- Large source: sample → cluster → assign-to-nearest-centroid ----------
    log.info(f"  {source_prefix}: {n} embeddings > {_CLUSTER_SAMPLE_MAX}; "
             f"using scalable sample-and-assign clustering")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, _CLUSTER_SAMPLE_MAX, replace=False)
    sample = emb_norm[sample_idx]
    sample_labels = _agglom(sample)

    uniq = np.unique(sample_labels)
    # centroid = renormalized mean of each sample cluster
    centroids = np.stack([
        normalize(sample[sample_labels == c].mean(axis=0, keepdims=True))[0]
        for c in uniq
    ])
    # cosine sim = dot (both normalized); assign each point to argmax centroid.
    labels = np.empty(n, dtype=int)
    BATCH = 8192
    for start in range(0, n, BATCH):
        block = emb_norm[start:start + BATCH]
        best = (block @ centroids.T).argmax(axis=1)
        labels[start:start + len(block)] = best  # contiguous 0..len(uniq)-1
    return labels


def find_filtered_wavs(source: str) -> list[Path]:
    if source == "all":
        return sorted(FILTERED_DIR.rglob("*.wav"))
    return sorted((FILTERED_DIR / source).rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Agglomerative clustering distance threshold (default 0.25)")
    args = parser.parse_args()

    wavs = find_filtered_wavs(args.source)
    need_embed = [w for w in wavs if not w.with_suffix(".embed.npy").exists()]
    log.info(f"Found {len(wavs)} WAVs, {len(need_embed)} need embedding extraction")

    # Extract embeddings only for files without cached .embed.npy
    if need_embed and not args.dry_run:
        log.info(f"Extracting {len(need_embed)} new embeddings ...")
        for i, wav_path in enumerate(need_embed):
            if i % 100 == 0:
                log.info(f"  {i}/{len(need_embed)} ...")
            try:
                emb = extract_embedding(wav_path)
                np.save(str(wav_path.with_suffix(".embed.npy")), emb)
            except Exception as exc:
                log.error(f"Embedding failed for {wav_path.name}: {exc}")

    # Load all embeddings from cache (fast path for already-embedded files)
    embeddings = []
    valid_wavs = []
    log.info(f"Loading embeddings for {len(wavs)} files ...")
    for wav_path in wavs:
        cache = wav_path.with_suffix(".embed.npy")
        if not cache.exists():
            continue
        try:
            emb = np.load(str(cache))
            embeddings.append(emb)
            valid_wavs.append(wav_path)
        except Exception as exc:
            log.error(f"Failed to load cached embedding {cache.name}: {exc}")

    if not embeddings:
        if args.dry_run:
            log.info(f"[DRY-RUN] Would extract {len(need_embed)} embeddings and cluster")
            return
        log.error("No embeddings extracted.")
        sys.exit(1)

    emb_array = np.stack(embeddings)
    log.info(f"Loaded {len(emb_array)} embeddings, shape {emb_array.shape}")

    if args.dry_run:
        log.info(f"[DRY-RUN] Would re-cluster {len(embeddings)} embeddings and write speaker IDs")
        return

    # Cluster per source to keep IDs consistent within source
    source_groups: dict[str, list[int]] = {}
    for i, w in enumerate(valid_wavs):
        src = w.relative_to(FILTERED_DIR).parts[0]
        source_groups.setdefault(src, []).append(i)

    all_labels = np.full(len(valid_wavs), -1, dtype=int)
    n_clusters_total = 0

    for src, indices in source_groups.items():
        src_embs = emb_array[indices]
        labels = cluster_embeddings(src_embs, src, args.threshold)
        for local_i, global_i in enumerate(indices):
            all_labels[global_i] = labels[local_i]
        n_src_clusters = len(set(labels))
        n_clusters_total += n_src_clusters
        log.info(f"  {src}: {len(indices)} segs → {n_src_clusters} speakers")

    # Write speaker.json per segment
    for i, wav_path in enumerate(valid_wavs):
        src = wav_path.relative_to(FILTERED_DIR).parts[0]
        cluster_id = int(all_labels[i])
        speaker_id = f"{src}_{cluster_id:03d}"
        record = {
            "wav_path": str(wav_path),
            "speaker_id": speaker_id,
            "cluster_id": cluster_id,
            "gender": "unknown",
        }
        out_path = wav_path.with_suffix(".speaker.json")
        with open(out_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    report = {
        "total_segments": len(valid_wavs),
        "total_speakers_estimated": n_clusters_total,
        "clustering_threshold": args.threshold,
        "source_breakdown": {
            src: {"segments": len(idx), "speakers": len(set(all_labels[i] for i in idx))}
            for src, idx in source_groups.items()
        },
    }
    SPEAKER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEAKER_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone: {len(valid_wavs)} segments → {n_clusters_total} estimated speakers")
    print(f"Speaker report: {SPEAKER_REPORT_PATH}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
