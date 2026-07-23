import pytest

from pipeline.nodes.pause_plan import compute_pause_plan


def _chars(text_no_punct, start=0.0, step=0.3, dur=0.2):
    """Build an alignments.chars-shaped list: one [char, start, end] per
    character in *text_no_punct*, evenly spaced by *step* with a fixed
    per-char duration *dur* (matches how align.chars is documented to look
    for CJK input -- pipeline/catalog/schema.sql's `alignments` comment)."""
    out = []
    t = start
    for ch in text_no_punct:
        out.append([ch, round(t, 4), round(t + dur, 4)])
        t += step
    return out


# ---------------------------------------------------------------------------
# include_timestamps=False (default) -- must reproduce the pre-P4 shape
# exactly, since pause_plan_one()/run_pause_plan() never pass the flag and
# the corpus-wide `pause_plan` table's stored shape must not drift.
# ---------------------------------------------------------------------------

def test_default_shape_has_no_timestamp_keys():
    chars = _chars("你好世界")
    plan, unalignable = compute_pause_plan("你好，世界。", chars, duration_sec=3.0)

    assert unalignable is False
    assert len(plan) == 2
    for entry in plan:
        assert "t_start" not in entry
        assert "t_end" not in entry


def test_normal_kind_delta_t_and_verdict_unaffected_by_flag():
    chars = _chars("你好世界")
    plan_a, _ = compute_pause_plan("你好，世界。", chars, duration_sec=3.0)
    plan_b, _ = compute_pause_plan("你好，世界。", chars, duration_sec=3.0, include_timestamps=True)

    for a, b in zip(plan_a, plan_b):
        assert a["offset"] == b["offset"]
        assert a["mark"] == b["mark"]
        assert a["kind"] == b["kind"]
        assert a.get("delta_t") == b.get("delta_t")
        assert a.get("verdict") == b.get("verdict")


# ---------------------------------------------------------------------------
# include_timestamps=True -- additive t_start/t_end
# ---------------------------------------------------------------------------

def test_normal_kind_gets_flanking_char_timestamps():
    chars = _chars("你好世界")  # 你@0.0-0.2  好@0.3-0.5  世@0.6-0.8  界@0.9-1.1
    plan, unalignable = compute_pause_plan("你好，世界。", chars, duration_sec=3.0, include_timestamps=True)

    assert unalignable is False
    normal = [e for e in plan if e["kind"] == "normal"]
    assert len(normal) == 1  # trailing "。" is trailing_tail, not normal
    ev = normal[0]
    assert ev["mark"] == "，"
    assert ev["t_start"] == chars[1][2]  # end of 好
    assert ev["t_end"] == chars[2][1]    # start of 世
    assert ev["t_end"] - ev["t_start"] == pytest.approx(ev["delta_t"])


def test_trailing_tail_gets_last_char_end_and_duration():
    chars = _chars("你好世界")
    plan, _ = compute_pause_plan("你好世界。", chars, duration_sec=3.0, include_timestamps=True)

    tail = [e for e in plan if e["kind"] == "trailing_tail"]
    assert len(tail) == 1
    assert tail[0]["t_start"] == chars[-1][2]
    assert tail[0]["t_end"] == 3.0


def test_leading_tail_gets_zero_and_first_char_start():
    chars = _chars("你好世界")
    plan, _ = compute_pause_plan("，你好世界", chars, duration_sec=3.0, include_timestamps=True)

    leading = [e for e in plan if e["kind"] == "leading_tail"]
    assert len(leading) == 1
    assert leading[0]["t_start"] == 0.0
    assert leading[0]["t_end"] == chars[0][1]


def test_unalignable_returns_empty_plan_regardless_of_flag():
    # chars has 5 entries but best_text's walk only ever matches 4 of them
    # (the 5th, trailing "你", never appears again in the text) -> ptr never
    # reaches n_chars by the end of the walk -> desync.
    chars = _chars("你好世界你")
    plan, unalignable = compute_pause_plan("你好世界，", chars, duration_sec=3.0, include_timestamps=True)

    assert unalignable is True
    assert plan == []
