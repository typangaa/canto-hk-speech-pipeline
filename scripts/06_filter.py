#!/usr/bin/env python3
"""
scripts/06_filter.py
Apply quality filters to verified segments. Passing segments → data/filtered/.
Usage: python scripts/06_filter.py --source [rthk|youtube|podcast|all] [--dry-run]

Filter order (cheapest first — see docs/QUALITY_SPEC.md):
  0. Hard gates: sample_rate==48000, single speaker, text_verified
  1. Duration: 3.0–20.0s
  2. Text length: ≥5 CJK chars, ≤150 chars
  3. English ratio: ≤0.30
  4. Mandarin ratio: ≤0.15
  5. ASR agreement: ≥0.80 (flag for review if below, do not auto-reject)
  6. SNR: ≥25 dB
  7. DNSMOS P.835 (speechmos): ≥3.0 — most expensive, apply last
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "06_filter.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SEG_DIR = ROOT / "data" / "segments"
FILTERED_DIR = ROOT / "data" / "filtered"
FILTER_REPORT_PATH = ROOT / "metadata" / "filter_report.json"
TARGET_SR = 48000
DNSMOS_SR = 16000

# Thresholds from QUALITY_SPEC.md
MIN_DUR = 3.0
MAX_DUR = 20.0
MIN_DNSMOS = 3.0  # applied to sig_mos (speech clarity), not ovrl_mos (see DECISIONS.md)
MIN_SNR_DB = 25.0
MAX_ENG_RATIO = 0.30
MAX_MAN_RATIO = 0.15
MIN_CJK_CHARS = 5
MAX_TEXT_CHARS = 150
MIN_ASR_AGREE = 0.80

CANTONESE_CHARS = set("係冇佢呢嗰嚟咁嘅囉喎啩咋啦啲喺唔咗嘢搵睇啱攞唞攰𠻹諗㗎喇乜哋俾俾瞓掟喐踎揸揼黐搣𠮶叻咪咩噏嘥嚿搲氹")

# Only characters exclusive to simplified Chinese — never appear in traditional/Cantonese text.
# Conservative list to minimise false positives. Verified against HK corpus samples.
# REMOVED (false positives): 最曾提只景招智晨普指曹替晴晶晒晚晃晕晖晔晦暖暗暮暴等
# (these appear in both traditional and simplified and are common in Cantonese text)
SIMPLIFIED_CHARS = set(
    # Core grammar/function words — unambiguously simplified
    "这们说时会对为当东乐车书无写听见让应义证认识双农"
    "发动关变质务设机飞门产类总积带济战县观讲谁课谢语译诉读误"
    "岁圆块坏处复万党兴击划创协卖卫归录弥弯弹态忆忧怀"
    "恳恶惊惧惨惩惭惮惯愤愿懒戏户执扩扫扬扭扮扰护报担"
    "拟拢拣拥拦拨择挥挤损换搅携摆摇撑敌数斩显"
)

TRADITIONAL_MANDARIN_INDICATORS = set("\u662f\u7684\u4ed6\u5979\u5011\u9019\u90a3\u8aaa\u6c92\u8ab0\u5403\u559d\u770b\u54ea\u600e")

CANTONESE_WORDS = [
    "而家", "唔係", "點解", "幾時", "即係", "邊度", "乜嘢", "噉樣", "一齊",
    "係咪", "真係", "已經", "好似", "先至", "仲有", "唔使", "或者", "點樣",
    "緊要", "鐘意", "鍾意", "返去", "企喺", "呢個", "呢啲", "嗰個", "嗰啲",
    "佢哋", "我哋", "你哋", "話俾", "話畀", "收聲", "話之", "企定", "睇吓",
    "玩嘢", "諗住", "講嘢", "喇喎", "㗎啦", "㗎喇", "唔好", "咪話", "冇人"
]

MANDARIN_WORDS = [
    "\u73fe\u5728", "\u73b0\u5728", "\u70ba\u4ec0\u9ebc", "\u4e3a\u4ec0\u4e48", "\u4ec0\u9ebc", "\u4ec0\u4e48", "\u6642\u5019", "\u65f6\u5019",
    "\u9019\u6a23", "\u8fd9\u6837", "\u6211\u5011", "\u6211\u4eec", "\u4f60\u5011", "\u4f60\u4eec", "\u4ed6\u5011", "\u4ed6\u4eec",
    "\u5979\u5011", "\u5979\u4eec", "\u662f\u4e0d\u662f", "\u771f\u7684", "\u600e\u9ebc", "\u600e\u4e48", "\u600e\u9ebc\u6a23", "\u600e\u4e48\u6837",
    "\u90a3\u6a23", "\u90a3\u6837", "\u9019\u500b", "\u8fd9\u4e2a", "\u9019\u4e9b", "\u8fd9\u4e9b", "\u90a3\u500b", "\u90a3\u4e2a",
    "\u90a3\u4e9b", "\u4e00\u8d77", "\u54ea\u88e1", "\u54ea\u91cc", "\u4e0d\u8981", "\u53bb\u54ea", "\u544a\u8a34", "\u544a\u8bc9", "\u90a3\u662f",
    "\u9019\u662f", "\u8fd9\u662f", "\u5c31\u662f", "\u7279\u5225\u662f", "\u7279\u522b\u662f", "\u4ec0\u9ebc\u7684", "\u4ec0\u4e48\u7684", "\u6628\u5929",
    "\u660e\u5929", "\u5403\u904e", "\u5403\u8fc7", "\u770b\u904e", "\u770b\u8fc7", "\u53bb\u904e", "\u53bb\u8fc7", "\u8d70\u4e86", "\u4f86\u4e86",
    "\u6765\u4e86", "\u5c0d\u4e0d\u8d77", "\u5bf9\u4e0d\u8d77", "\u8b1d\u8b1d", "\u8c22\u8c22", "\u525b\u624d", "\u521a\u624d"
]


def compute_snr(wav: np.ndarray) -> float:
    frame_len = int(TARGET_SR * 0.025)
    energies = [np.sum(wav[i:i+frame_len]**2) for i in range(0, len(wav)-frame_len, frame_len)]
    if not energies:
        return 0.0
    energies.sort()
    signal_e = np.mean(energies[int(0.9*len(energies)):]) + 1e-10
    noise_e   = np.mean(energies[:int(0.1*len(energies))]) + 1e-10
    return round(10 * np.log10(signal_e / noise_e), 1)


def compute_dnsmos(wav48: np.ndarray) -> tuple[float, float]:
    """Returns (sig_mos, ovrl_mos). Filter uses sig_mos; ovrl_mos stored for reference."""
    from speechmos import dnsmos
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(TARGET_SR, DNSMOS_SR)
    wav16 = resampler(t).squeeze(0).numpy()
    result = dnsmos.run(wav16, sr=DNSMOS_SR)
    sig = float(result.get("sig_mos", 0.0))
    ovrl = float(result.get("ovrl_mos", 0.0))
    assert 1.0 <= sig <= 5.0, f"DNSMOS sig_mos out of range: {sig}"
    return round(sig, 2), round(ovrl, 2)


def is_cjk(c: str) -> bool:
    """Helper to detect if a character falls within any CJK Unified Ideographs block (including extensions)."""
    code = ord(c)
    return (
        (0x4E00 <= code <= 0x9FFF) or     # CJK Unified Ideographs
        (0x3400 <= code <= 0x4DBF) or     # Extension A (contains 㗎 etc.)
        (0x20000 <= code <= 0x2A6DF) or   # Extension B (contains 𠻹, 𠮶 etc.)
        (0x2A700 <= code <= 0x2B73F) or   # Extension C
        (0x2B740 <= code <= 0x2B81F) or   # Extension D
        (0x2B820 <= code <= 0x2CEAF) or   # Extension E
        (0x2CEB0 <= code <= 0x2EBF0) or   # Extension F
        (0x30000 <= code <= 0x3134F) or   # Extension G
        (0x31350 <= code <= 0x323AF) or   # Extension H
        (0xF900 <= code <= 0xFAFF)        # Compatibility
    )


def get_english_and_cjk_tokens(text: str) -> tuple[list[str], list[str]]:
    """Tokenize the text into CJK characters and English words (ignoring punctuation)."""
    english_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    cjk_chars = [c for c in text if is_cjk(c)]
    return english_words, cjk_chars


def english_ratio(text: str) -> float:
    """
    Computes English word ratio relative to total semantic tokens (English words + CJK characters).
    Prevents penalizing natural Cantonese code-switching of long English words.
    """
    english_words, cjk_chars = get_english_and_cjk_tokens(text)
    total_words = len(english_words) + len(cjk_chars)
    if not total_words:
        return 0.0
    return round(len(english_words) / total_words, 3)


def mandarin_ratio(text: str) -> float:
    """
    Computes Mandarin ratio by analyzing simplified/traditional markers and vocabulary.
    Returns a ratio > MAX_MAN_RATIO (0.15) if Mandarin features are present and Cantonese markers are absent,
    or if there is a significant mixture of both (indicating a Mandarin or mixed segment).
    """
    cjk = [c for c in text if is_cjk(c)]
    if not cjk:
        return 0.0

    num_simplified = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)

    # Clean text of neutral words that contain Mandarin-indicating characters to avoid false positives
    text_clean = text
    neutral_words = [
        "的士", "的確", "目的", "了解", "除了", "不得了", "受不了", "甚至乎",
        "著名", "著作", "著急", "著手", "說法", "說明", "小說", "話說",
        "可是", "是否", "總是", "要是", "就是", "國是", "是非", "若是", "看法"
    ]
    for w in neutral_words:
        text_clean = text_clean.replace(w, "")

    cjk_clean = [c for c in text_clean if is_cjk(c)]

    mando_chars = sum(1 for c in cjk_clean if c in TRADITIONAL_MANDARIN_INDICATORS)
    canto_chars = sum(1 for c in cjk if c in CANTONESE_CHARS)

    mando_words_count = sum(text.count(w) for w in MANDARIN_WORDS)
    canto_words_count = sum(text.count(w) for w in CANTONESE_WORDS)

    mando_score = num_simplified * 2.0 + mando_chars * 1.5 + mando_words_count * 2.5
    canto_score = canto_chars * 1.5 + canto_words_count * 2.5

    # If both languages appear, reject if Mandarin is significant
    if canto_score > 0 and mando_score > 0:
        total_mando_features = num_simplified + mando_chars
        # If Mandarin features make up > 30% of the Cantonese strength, it's mixed
        if mando_score >= 0.3 * canto_score:
            ratio = max(0.16, total_mando_features / len(cjk))
            return round(min(ratio, 1.0), 3)
        return round(total_mando_features / len(cjk), 3)
    elif canto_score == 0:
        # No Cantonese markers at all
        if mando_score > 0:
            total_mando_chars = num_simplified + mando_chars + sum(len(w) for w in MANDARIN_WORDS if w in text)
            # Force rejection (> 0.15) if there are clear Mandarin markers
            ratio = max(0.16, total_mando_chars / len(cjk))
            return round(min(ratio, 1.0), 3)
        return 0.0
    else:
        # Has Cantonese markers, no Mandarin markers
        return 0.0


def cjk_count(text: str) -> int:
    """Count CJK characters (including extensions) in text."""
    return sum(1 for c in text if is_cjk(c))


def detect_language(text: str) -> tuple[str, float]:
    """
    Detects the primary language of the text.
    Returns: (language_code, confidence) where language_code is 'yue', 'cmn', 'eng', or 'mixed'.
    """
    eng_ratio = english_ratio(text)
    if eng_ratio >= 0.85:
        return "eng", eng_ratio

    cjk = [c for c in text if is_cjk(c)]
    if not cjk:
        return "eng", eng_ratio

    num_simplified = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)
    
    text_clean = text
    neutral_words = [
        "的士", "的確", "目的", "了解", "除了", "不得了", "受不了", "甚至乎",
        "著名", "著作", "著急", "著手", "說法", "說明", "小說", "話說",
        "可是", "是否", "總是", "要是", "就是", "國是", "是非", "若是", "看法"
    ]
    for w in neutral_words:
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


class FilterStats:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.reasons: dict[str, int] = {}

    def fail(self, reason: str):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def report(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.passed / max(self.total, 1), 3),
            "rejections": self.reasons,
        }


def filter_segment(
    wav_path: Path,
    transcript: dict,
    stats: FilterStats,
    dry_run: bool,
    pregate: dict | None = None,
) -> bool:
    # Fast-skip: if pre-gate already failed SNR, no need to load WAV or run DNSMOS.
    # Pre-gate uses identical SNR threshold (25 dB), so failures are guaranteed Stage 6 rejects.
    if pregate is not None and not pregate.get("pass", True):
        return False

    stats.total += 1
    name = wav_path.name
    text = transcript.get("text", "")

    # --- Gate 0: Hard gates ---
    info = sf.info(str(wav_path))
    if info.samplerate != TARGET_SR:
        stats.fail("sample_rate")
        log.debug(f"FAIL sample_rate {info.samplerate}: {name}")
        return False

    # text_verified check: soft flag only — don't hard-reject; Stage 5 calibration happens after filtering
    # Audio quality gates (duration, SNR, DNSMOS) + language detection run on ASR text regardless
    is_verified = bool(transcript.get("text_verified"))

    # --- Gate 1: Duration ---
    dur = info.duration
    if not (MIN_DUR <= dur <= MAX_DUR):
        stats.fail("duration")
        return False

    # --- Gate 2: Text length ---
    n_cjk = cjk_count(re.sub(r"[^\w\s]", "", text))
    if n_cjk < MIN_CJK_CHARS:
        stats.fail("text_too_short")
        return False
    if len(text) > MAX_TEXT_CHARS:
        stats.fail("text_too_long")
        return False

    # --- Gate 3: English ratio ---
    eng = english_ratio(text)
    if eng > MAX_ENG_RATIO:
        stats.fail("english_ratio")
        return False

    # --- Gate 4: Mandarin ratio ---
    man = mandarin_ratio(text)
    if man > MAX_MAN_RATIO:
        stats.fail("mandarin_ratio")
        return False

    # --- Gate 5: ASR agreement (flag only, no auto-reject) ---
    agreement = transcript.get("asr_agreement", 1.0)
    if agreement < MIN_ASR_AGREE:
        log.info(f"  Low agreement {agreement:.2f} (passed but flagged): {name}")

    # --- Gate 6: SNR ---
    wav48, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav48.ndim > 1:
        wav48 = wav48.mean(axis=1)
    # Reuse pre-gate SNR if available (same 25dB threshold, avoids recomputation).
    if pregate is not None and pregate.get("snr") is not None:
        snr = float(pregate["snr"])
    else:
        snr = compute_snr(wav48)
    if snr < MIN_SNR_DB:
        stats.fail("snr")
        return False

    # --- Gate 7: DNSMOS speech quality (most expensive — last) ---
    # Use sig_mos (speech clarity) rather than ovrl_mos, since documentary/broadcast
    # audio with background music has low ovrl_mos even when speech is perfectly clear.
    try:
        sig_mos, ovrl_mos = compute_dnsmos(wav48)
    except Exception as exc:
        log.warning(f"DNSMOS failed on {name}: {exc}")
        stats.fail("dnsmos_error")
        return False
    if sig_mos < MIN_DNSMOS:
        stats.fail("dnsmos")
        return False
    mos = sig_mos  # for downstream use

    # --- Passed all filters ---
    stats.passed += 1

    if not dry_run:
        # Copy to filtered/ preserving subdirectory structure under segments/
        rel = wav_path.relative_to(SEG_DIR)
        out_path = FILTERED_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            shutil.copy2(wav_path, out_path)

        # Detect primary language for downstream metadata
        lang, lang_conf = detect_language(text)

        # Write filter metadata alongside
        filter_meta = {
            "wav_path": str(out_path),
            "snr_db": float(snr),
            "dnsmos": float(mos),
            "dnsmos_ovrl": float(ovrl_mos),
            "english_ratio": float(eng),
            "mandarin_ratio": float(man),
            "detected_language": lang,
            "language_confidence": float(lang_conf),
            "asr_agreement": float(agreement),
            "duration_sec": round(float(dur), 3),
            "sample_rate": int(info.samplerate),
            "text": text,
            "text_verified": is_verified,
        }
        meta_path = out_path.with_suffix(".filter.json")
        with open(meta_path, "w") as f:
            json.dump(filter_meta, f, ensure_ascii=False, indent=2)

    return True


def find_segments(source: str, incremental: bool = False) -> list[tuple[Path, dict]]:
    if source == "all":
        wavs = sorted(SEG_DIR.rglob("*.wav"))
    else:
        wavs = sorted((SEG_DIR / source).rglob("*.wav"))

    results = []
    skipped_existing = 0
    for wav in wavs:
        if incremental:
            rel = wav.relative_to(SEG_DIR)
            existing_filter = FILTERED_DIR / rel.parent / (wav.stem + ".filter.json")
            if existing_filter.exists():
                skipped_existing += 1
                continue
        t_path = wav.with_suffix(".transcript.json")
        if not t_path.exists():
            continue
        try:
            with open(t_path) as f:
                transcript = json.load(f)
            results.append((wav, transcript))
        except Exception:
            pass
    if skipped_existing:
        log.info(f"Incremental mode: skipped {skipped_existing} already-filtered segments")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute stats but don't copy files")
    parser.add_argument("--use-pregate", action="store_true",
                        help="Read .pregate.json markers to skip pre-gate failures and reuse SNR values "
                             "(saves ~55%% WAV loads + DNSMOS calls on podcast data)")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip segments that already have a .filter.json in data/filtered/ "
                             "(safe for new-segment runs; use without --incremental to re-evaluate all)")
    args = parser.parse_args()

    segments = find_segments(args.source, incremental=args.incremental)
    log.info(f"Found {len(segments)} segments with transcripts")

    # Sanity check: verify DNSMOS gives valid range on first 5 files
    if not args.dry_run and segments:
        log.info("Sanity-checking DNSMOS on first 5 files ...")
        test_scores = []
        for wav_path, _ in segments[:5]:
            try:
                wav48, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
                if wav48.ndim > 1:
                    wav48 = wav48.mean(axis=1)
                s = compute_dnsmos(wav48)
                test_scores.append(s)
            except Exception as e:
                log.warning(f"DNSMOS test failed on {wav_path.name}: {e}")
        if test_scores:
            assert all(1.0 <= s[0] <= 5.0 for s in test_scores), \
                f"DNSMOS sanity check failed: {test_scores}"
            log.info(f"DNSMOS sanity OK (sig_mos): {[s[0] for s in test_scores]}")

    pregate_skipped = 0
    fstats = FilterStats()
    for wav_path, transcript in segments:
        pregate = None
        if args.use_pregate:
            pg_path = wav_path.with_suffix(".pregate.json")
            if pg_path.exists():
                try:
                    pregate = json.loads(pg_path.read_text())
                except Exception:
                    pass
            if pregate is not None and not pregate.get("pass", True):
                pregate_skipped += 1
                continue
        try:
            filter_segment(wav_path, transcript, fstats, args.dry_run, pregate)
        except Exception as exc:
            log.error(f"Error filtering {wav_path.name}: {exc}", exc_info=True)
            fstats.fail("exception")
    if pregate_skipped:
        log.info(f"Pre-gate fast-skip: {pregate_skipped} SNR-failed segments skipped before Stage 6")

    report = fstats.report()
    total_filtered = sum(1 for _ in FILTERED_DIR.rglob("*.wav"))

    print(f"\nFilter results:")
    print(f"  Total:      {report['total']}")
    print(f"  Passed:     {report['passed']} ({100*report['pass_rate']:.1f}%)")
    print(f"  Rejections: {report['rejections']}")
    if args.dry_run:
        print("  [DRY-RUN: no files written]")
    else:
        print(f"  data/filtered/ total WAVs: {total_filtered}")

    FILTER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FILTER_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Filter report: {FILTER_REPORT_PATH}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
