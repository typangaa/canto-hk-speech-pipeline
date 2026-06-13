# canto-hk-speech-pipeline

A reproducible data pipeline for building a **Hong Kong Cantonese speech corpus** suitable for TTS model training. Processes publicly accessible audio sources through VAD, diarization, ASR transcription, human calibration, G2P romanisation, and speaker clustering to produce a structured JSONL manifest.

> **The pipeline code is open source (Apache 2.0). The source audio and any derived dataset are NOT included in this repository.** See [Copyright & Data Licensing](#copyright--data-licensing) below.

---

## Pipeline Overview

| Stage | Script | Input | Output | Tool |
|-------|--------|-------|--------|------|
| 00 | `00_reingest.py` | Legacy data | Re-indexed segments | internal |
| 01 | `01_discover.py` | `sources/*.yaml` | `metadata/discovery.json` | yt-dlp (simulate) |
| 02 | `02_download.py` | `sources/*.yaml` | `data/raw/` WAVs | yt-dlp / feedparser |
| 03 | `03_segment.py` | `data/raw/` | `data/segments/` WAVs | pyannote, VAD |
| 03b | `03b_acoustic_pregate.py` | `data/segments/` | pregate JSON | speechmos, SNR |
| 04 | `04_transcribe.py` | `data/segments/` | `.transcript.json` | faster-whisper (2-pass) |
| 05 | `05_calibrate.py` | `.transcript.json` | verified `text` field | human review |
| 06 | `06_filter.py` | `.transcript.json` | `.filter.json` | DNSMOS, SNR, length |
| 07 | `07_g2p.py` | `.filter.json` | `.jyutping.json` | canto-hk-g2p |
| 08 | `08_speaker_id.py` | `data/segments/` | global speaker IDs | SpeechBrain |
| 09 | `09_manifest.py` | `.jyutping.json` | `metadata/train.jsonl` | — |
| 10 | `10_report.py` | `metadata/train.jsonl` | `DATASET_REPORT.md` | — |

Each stage is **idempotent** — already-processed files are skipped. Stages can be run per-source with `--source rthk|youtube|podcast|hktv|all`.

---

## Audio Strategy

- Every segment is stored as a **48 kHz mono WAV** master.
- 16 kHz copies are generated transiently in memory for VAD / ASR / DNSMOS — never written to disk.
- This preserves compatibility with all modern TTS codecs (NeuCodec 24k, F5-TTS 24k, MOSS-TTS-Nano 48k).

---

## Requirements

**System tools** (install separately):

```bash
# ffmpeg — audio conversion
sudo apt install ffmpeg

# yt-dlp — video/audio download
pip install yt-dlp
```

**Python 3.10+**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Cantonese G2P** (build from source until PyPI release):

```bash
git clone https://github.com/typangaa/canto-hk-g2p
cd canto-hk-g2p
pip install maturin
maturin develop --release
```

**pyannote.audio** requires accepting model terms on Hugging Face before first use:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Set `HF_TOKEN` environment variable to your token.

---

## Usage

```bash
# 1. Configure sources
#    Edit sources/rthk_sources.yaml, sources/youtube_channels.yaml,
#    sources/podcast_sources.yaml with the programs you want to collect.

# 2. Discover available content (no download)
python scripts/01_discover.py --source rthk

# 3. Download audio
python scripts/02_download.py --source rthk

# 4. Segment + diarize
python scripts/03_segment.py --source rthk

# 5. Transcribe (two-pass Whisper)
python scripts/04_transcribe.py --source rthk

# 6. Human calibration — review low-agreement segments
python scripts/05_calibrate.py --source rthk

# 7. Filter by quality (DNSMOS, SNR, duration)
python scripts/06_filter.py --source rthk

# 8. G2P romanisation (Jyutping)
python scripts/07_g2p.py --source rthk

# 9. Speaker clustering
python scripts/08_speaker_id.py --source rthk

# 10. Build manifest
python scripts/09_manifest.py

# 11. Quality report
python scripts/10_report.py
```

All scripts support `--dry-run` (log actions without writing files) and `--limit N` (process only N files for testing).

---

## Output Format

`metadata/train.jsonl` — one JSON object per line:

```json
{
  "wav_path": "data/filtered/rthk/segment_0001.wav",
  "text": "香港係一個國際城市。",
  "jyutping": "hoeng1 gong2 hai6 jat1 go3 gwok3 zai3 sing4 si5",
  "duration": 4.2,
  "speaker_id": "SPK_042",
  "source": "rthk",
  "domain": "documentary",
  "dnsmos_sig": 3.7,
  "snr_db": 32.1,
  "text_verified": true
}
```

See `docs/MANIFEST_SCHEMA.md` for full field definitions.

---

## Data Sources & Copyright

> **Read this section carefully before using this pipeline.**

This pipeline downloads audio from publicly accessible sources. **The pipeline code is licensed under Apache 2.0. The audio content is not.**

### Sources used

| Source | Rights holder | Terms |
|--------|--------------|-------|
| RTHK (Radio Television Hong Kong) | © RTHK / Hong Kong SAR Government | [RTHK Terms of Use](https://www.rthk.hk/about/terms.htm) — for personal, non-commercial and educational use |
| YouTube channels | © respective creators | [YouTube Terms of Service](https://www.youtube.com/t/terms) — downloading requires compliance with creator licence and platform ToS |
| Podcast RSS feeds | © respective publishers | Varies per podcast — check individual RSS licence |
| HKTV | © HK Television Entertainment Co. Ltd | Commercial copyright — research use only |

### What this means for you

- **Do not redistribute source audio** downloaded by this pipeline.
- **Do not publish derived audio** (segments, re-encoded clips) unless the source licence explicitly permits it.
- **Metadata you generate** (transcripts, Jyutping, speaker IDs, JSONL manifests) may be releasable under a permissive licence if you authored them — but consult a lawyer for your specific jurisdiction.
- **RTHK content** is the most permissive: the public broadcaster publishes for public benefit, and some programmes carry Creative Commons notices. If you release a dataset, prefer RTHK-sourced segments and document the programme licence individually.

### Releasing a dataset derived from this pipeline

If you build a dataset and want to publish it (e.g. on Hugging Face):

1. **Audio**: Only include audio from sources whose licence permits redistribution. Treat RTHK CC-licensed programmes as a separate subset with explicit attribution.
2. **Metadata-only release**: Release `text`, `jyutping`, `duration`, `speaker_id`, `source_url`, and a download script. Users download the audio themselves.
3. **Dataset card**: Declare the licence for each field. Reference this pipeline repo and any paper you write.
4. **Do not scrape at scale** from sources that prohibit it in their ToS (YouTube ToS §5B prohibits automated downloading without explicit permission).

---

## Project Structure

```
canto-hk-speech-pipeline/
├── scripts/                    # Pipeline stages 00–10
│   ├── 01_discover.py
│   ├── 02_download.py
│   ├── 03_segment.py
│   ├── 03b_acoustic_pregate.py
│   ├── 04_transcribe.py
│   ├── 05_calibrate.py
│   ├── 06_filter.py
│   ├── 07_g2p.py
│   ├── 08_speaker_id.py
│   ├── 09_manifest.py
│   └── 10_report.py
├── sources/                    # Source configuration (YAML)
│   ├── rthk_sources.yaml
│   ├── youtube_channels.yaml
│   └── podcast_sources.yaml
├── docs/                       # Design documents
│   ├── PIPELINE_SPEC.md        # Stage-by-stage implementation details
│   ├── QUALITY_SPEC.md         # Filter thresholds and rationale
│   ├── MANIFEST_SCHEMA.md      # Output field definitions
│   ├── KNOWN_ISSUES.md         # Failure modes and workarounds
│   └── SOURCE_GUIDE.md         # How to add new audio sources
├── requirements.txt
├── LICENSE                     # Apache 2.0 (pipeline code only)
└── README.md
```

Not committed to this repo (see `.gitignore`):
- `data/` — downloaded audio, segments, filtered WAVs
- `metadata/logs/` and `metadata/*.json` — machine-generated reports
- `.cache/` — model weight caches
- `PROGRESS.md` — personal session log

---

## Related Projects

- [canto-hk-g2p](https://github.com/typangaa/canto-hk-g2p) — Rust-core Cantonese G2P library used in Stage 7

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{canto-hk-speech-pipeline-2026,
  title   = {canto-hk-speech-pipeline: A Hong Kong Cantonese Speech Corpus Pipeline},
  author  = {Tak Yin Pang},
  year    = {2026},
  url     = {https://github.com/typangaa/canto-hk-speech-pipeline}
}
```

---

## License

Pipeline code: **Apache License 2.0** — see [LICENSE](LICENSE).

Source audio downloaded by this pipeline is subject to the rights holders' own terms (see [Data Sources & Copyright](#copyright--data-licensing)). The Apache 2.0 licence applies only to the code in this repository, not to any audio or derived data.
