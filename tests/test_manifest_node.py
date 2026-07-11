from pipeline.nodes.manifest import (
    INCLUDED_TIERS,
    _stratified_split_new_entries,
    build_entry,
    train_val_split,
)


def _row(**overrides):
    base = (
        "seg001", "/mnt/Drive4/canto/filtered/youtube/x.wav", "youtube", "", "", "other",
        6.97, 48000, "youtube_001", "male", "formal", "2026-06-09",
        "hello world", False, 0.91,
        "nei5 hou2",
        35.2, 3.8, 0.02,
        "silver",
    )
    if not overrides:
        return base
    fields = [
        "seg_id", "audio_path", "source", "source_url", "program", "domain",
        "duration_sec", "sample_rate", "speaker_id", "gender", "style", "created_at",
        "best_text", "text_verified", "agreement",
        "jyutping",
        "snr_db", "dnsmos", "english_ratio",
        "tier",
    ]
    d = dict(zip(fields, base))
    d.update(overrides)
    return tuple(d[f] for f in fields)


def test_build_entry_basic_fields():
    entry = build_entry(_row(), asr_candidates=[{"model": "m", "text": "hello world", "confidence": 0.9}])
    assert entry["id"] == "seg001"
    assert entry["source"] == "youtube"
    assert entry["text"] == "hello world"
    assert entry["text_verified"] is False
    assert entry["tier"] == "silver"
    assert entry["jyutping"] == "nei5 hou2"


def test_build_entry_rounding_matches_manifest_schema():
    entry = build_entry(_row(duration_sec=6.9666, snr_db=35.249, dnsmos=3.801, agreement=0.9051), [])
    assert entry["duration_sec"] == 6.967
    assert entry["snr_db"] == 35.2
    assert entry["dnsmos"] == 3.8
    assert entry["asr_agreement"] == 0.905


def test_build_entry_missing_speaker_id_falls_back_to_source_unk():
    entry = build_entry(_row(speaker_id=None), [])
    assert entry["speaker_id"] == "youtube_unk"


def test_build_entry_none_gender_style_default_unknown_formal():
    entry = build_entry(_row(gender=None, style=None), [])
    assert entry["gender"] == "unknown"
    assert entry["style"] == "formal"


def test_build_entry_missing_created_at_falls_back_to_today():
    entry = build_entry(_row(created_at=None), [])
    assert entry["created_at"]  # non-empty ISO-ish string


def test_included_tiers_excludes_the_excluded_sentinel():
    assert "excluded" not in INCLUDED_TIERS
    assert set(INCLUDED_TIERS) == {"gold", "auto_gold", "silver"}


# ---------------------------------------------------------------------------
# train/val split -- preserve existing membership, extend for new ids only
# ---------------------------------------------------------------------------

def _entry(id_, source, speaker_id):
    return {"id": id_, "source": source, "speaker_id": speaker_id}


def test_stratified_split_new_entries_no_speaker_leakage():
    entries = [_entry(f"id{i}", "youtube", f"spk{i % 10}") for i in range(100)]
    train, val = _stratified_split_new_entries(entries, val_frac=0.2)
    train_speakers = {e["speaker_id"] for e in train}
    val_speakers = {e["speaker_id"] for e in val}
    assert not (train_speakers & val_speakers)
    assert len(train) + len(val) == len(entries)


def test_train_val_split_preserves_existing_membership(tmp_path):
    existing_train = tmp_path / "train.jsonl"
    existing_val = tmp_path / "val.jsonl"
    existing_train.write_text('{"id": "old1"}\n{"id": "old2"}\n')
    existing_val.write_text('{"id": "old3"}\n')

    entries = [
        _entry("old1", "youtube", "spkA"),
        _entry("old2", "youtube", "spkA"),
        _entry("old3", "youtube", "spkB"),
        _entry("new1", "youtube", "spkC"),
    ]
    train, val = train_val_split(entries, train_path=existing_train, val_path=existing_val, val_frac=0.05)

    train_ids = {e["id"] for e in train}
    val_ids = {e["id"] for e in val}
    assert {"old1", "old2"} <= train_ids
    assert "old3" in val_ids
    # the new id must land in exactly one split, not both or neither
    assert ("new1" in train_ids) != ("new1" in val_ids)


def test_train_val_split_no_existing_files_falls_back_to_full_stratified_split(tmp_path):
    entries = [_entry(f"id{i}", "podcast", f"spk{i % 5}") for i in range(50)]
    train, val = train_val_split(
        entries, train_path=tmp_path / "missing_train.jsonl", val_path=tmp_path / "missing_val.jsonl",
        val_frac=0.2,
    )
    assert len(train) + len(val) == 50
