# Pause Calibration Report — P1 (標點-聲學統計 + calibration)

> **Status**: informational analysis only — feeds the P1 human gate in
> `docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md`. Nothing here is frozen. The owner (+ canto-tts)
> makes the final bucket-threshold call; this report exists to inform that call with real
> measured numbers, not to make it.
> **Date**: 2026-07-21 · **Repo rev**: `9b73455` · **Draft constants**:
> `metadata/labels/pause_calibration_draft.json`
> **Data**: `alignments.chars` (100% of gold+auto_gold, 279,348/279,348 rows,
> `provenance='qwen3_aligner'`) joined with `asr_agreement.best_text`, `segments.duration_sec`,
> and `labels_prosody.gaps`, scoped to `tiers.tier IN ('gold','auto_gold')`.
> **Sample size**: **full scope, no sampling** — the join + Python walk over all 279,348 rows
> completed in under 10 seconds end-to-end (DuckDB fetch ~1.3s, char-walk + aggregation ~7s on
> a single core), so there was no need to subsample.

---

## Executive summary

- **~29% of mid-sentence punctuation marks have no measurable acoustic pause** (Δt < 80ms,
  many exactly 0.000s) — this is qwen3_asr's LM inventing punctuation with no acoustic basis,
  exactly the failure mode the plan set out to quantify. It is worst for `！`(56%) and
  `？`(46%), moderate for `：`(42%), `、`(31%), `，`(30%), and lowest for `。`(17%) and
  `；`(12%) — the sentence-structural marks are the most acoustically grounded, the
  emphasis/question marks the least.
- **Marks differ meaningfully by pause length, in the expected prosodic order**: `。`
  (full stop, mean 0.36s) > `；` (0.32s) > `，` (0.22s) > `？`/`、` (~0.17s) > `：` (0.15s) >
  `！` (0.12s). This is a clean, orderly signal — not noise.
- **Trailing (segment-final) punctuation confirms the plan's §0 hypothesis**: the same mark
  (`。`) measures p50=0.40s / p90=0.64s when mid-segment but only p50=0.16s / p90=0.34s when
  it is the last character of the segment — because `segment.vad_cut` has already trimmed
  most of the true post-utterance silence away. **Trailing-punctuation Δt should not be used
  for bucket calibration**, and should probably not get a `<pause-*>` token at all (see §5).
- **The distribution does NOT split cleanly into two well-separated humps** — it is closer to
  one broad, right-skewed continuum from ~0.08s to ~0.5s with a long thin tail beyond that
  (see §4 histogram). A short/long split is still defensible (see recommendation), but don't
  expect two textbook Gaussians.
- **The VAD-gap cross-check came back weak (5.8% overall match rate)** — independent Silero
  VAD gaps rarely land within 150ms of an aligner-measured punctuation pause, even restricting
  to Δt ≥ 0.35s (12.7% match) or Δt ≥ 1.0s (24.4% match). This does not necessarily mean the
  aligner is wrong (see §3 for the two competing explanations) but it means the aligner signal
  and the VAD signal should **not** be treated as interchangeable, and the aligner-based
  numbers in this report should be the primary basis for calibration, not the older VAD-gap
  numbers.
- This punctuation-anchored measurement also **refines, not just confirms**, the plan's §0
  prior: the earlier *all-gaps* VAD measurement found within-segment p90=0.26s; this
  punctuation-*specific* aligner measurement finds a noticeably longer tail (pooled p90=0.48s).
  The earlier number pooled in a lot of non-punctuation micro-hesitation gaps and undercounted
  because VAD only reports gaps ≥0.2s at a fairly conservative threshold — see §4.

---

## 0. Method recap (for anyone jumping straight to the numbers)

For every segment in scope, `best_text` is walked character-by-character against
`alignments.chars` (one entry per non-punctuation character, in original order) using a
subsequence-matching pointer: a text character that matches `chars[ptr]` is a "kept" char and
advances the pointer; a text character that does not match is treated as dropped by the
aligner (punctuation, quotes, spaces, etc.). For each of the 7 mid-sentence marks
(，。？！、；：, half-width `,?!;:` normalized to full-width; half-width `.` deliberately
excluded — see caveat in §6), Δt is:

```
Δt = chars[ptr].start_time − chars[ptr-1].end_time
```

i.e. the gap between the end of the last "kept" character before the mark and the start of the
next "kept" character after it, in the *original* text's flanking order. Three outcomes per
punctuation instance:

| kind | condition | n | meaning |
|---|---|---|---|
| `normal` | has both a preceding and following kept char | 646,001 | genuine flanked acoustic gap — **primary analysis population** |
| `trailing_tail` | last char of segment reached with no following kept char (i.e. this mark sits at/after the very end of the recognized text) | 269,001 | segment-boundary punctuation — analyzed **separately**, see §5 |
| `leading_tail` | mark before the *first* kept char | 1 | negligible, ignored |

**Segment-level exclusion**: 8,869 of 279,348 segments (3.2%) had a character-walk that did not
fully consume `alignments.chars` (i.e. `best_text` and `chars` fell out of sync) — these are
excluded from all stats below, not silently coerced. See §6 caveats.

---

## 1. Per-mark Δt distribution (kind=`normal`, n=646,001)

Negative deltas (aligner timestamp overlap) were 0 in this cohort — clipped-at-zero and raw
percentiles are identical. All times in seconds.

| mark | n | p25 | p50 | p75 | p90 | mean | no-pause rate (Δt<0.08s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 。 (period) | 75,002 | 0.24 | 0.40 | 0.48 | 0.64 | 0.359 | 17.1% |
| ； (semicolon) | 1,025 | 0.24 | 0.32 | 0.40 | 0.56 | 0.320 | 11.9% |
| ， (comma) | 496,584 | 0.00 | 0.24 | 0.32 | 0.48 | 0.223 | 29.8% |
| ？ (question) | 9,307 | 0.00 | 0.08 | 0.32 | 0.40 | 0.168 | 45.8% |
| 、 (enum. comma) | 34,816 | 0.00 | 0.16 | 0.32 | 0.40 | 0.173 | 30.9% |
| ： (colon) | 24,224 | 0.00 | 0.08 | 0.24 | 0.40 | 0.152 | 42.5% |
| ！ (exclaim) | 5,043 | 0.00 | 0.00 | 0.24 | 0.40 | 0.124 | 56.1% |
| **pooled (all marks)** | **646,001** | **0.00** | **0.24** | **0.40** | **0.48** | **0.232** | **29.3%** |

**Reading this**: marks do differ, and in a linguistically sensible order. `。` and `；` —
both hard clause/sentence boundaries — carry the longest, most reliably-real pauses and the
lowest hallucination rate. `！` and `？` carry the shortest pauses and by far the highest
hallucination rate (>45%) — consistent with qwen3_asr's LM inserting these for *emphasis/tone*
inference from wording rather than from any acoustic cue (a rising pitch contour or exclamatory
phrasing doesn't require a pause the way a clause boundary does). `，` (the overwhelming
majority of the sample, 77% of all `normal` instances) sits in the middle at both mean Δt and
hallucination rate — the single mark that will dominate any pooled statistic by sheer volume.

---

## 2. No-pause ("LM hallucinated punctuation") rate

Using the plan's own industry-prior starting threshold of **80ms**:

- **Pooled: 29.3%** of mid-sentence punctuation marks (kind=`normal`) have Δt below 80ms —
  i.e. nearly 3 in 10 punctuation marks qwen3_asr inserted have essentially no acoustic pause
  behind them.
- **26.9% land at *exactly* 0.000s** (not just "below 80ms but nonzero") — the aligner puts the
  preceding and following character's end/start at the identical timestamp. This is the
  cleanest signature of a punctuation mark with zero acoustic correlate at all.
- Per-mark hallucination rate ranges from **11.9%** (`；`, rare but almost always real) to
  **56.1%** (`！`, more often fake than real).
- This directly operationalizes the plan's §3.2 observation: punctuation is 100%
  LM-inferred by qwen3_asr (sense_voice's CTC output carries no punctuation at all, and
  `asr.agreement` strips punctuation before comparing models), so **no punctuation mark in
  this corpus has ever had cross-model acoustic verification** until this Δt measurement. This
  report is the first such verification pass.

---

## 3. Cross-check against `labels_prosody.gaps` (independent VAD signal)

`labels_prosody.gaps` is Silero VAD run independently at label-generation time
(`min_silence_duration_ms=200`, i.e. it **only ever records silences ≥0.2s**). Coverage: 96.0%
of the "real pause" (Δt≥0.08s) cohort has `labels_prosody` data at all.

Matching rule: a VAD gap `[start, start+dur]` is considered a match for an aligner-measured
pause window `[prev_char_end, next_char_start]` if the two intervals overlap once each is given
±150ms of slack.

| Δt bucket | n | match rate |
|---|---:|---:|
| 0.08–0.15s | 48,314 | 0.2% |
| 0.15–0.25s | 125,292 | 0.3% |
| 0.25–0.35s | 96,476 | 3.6% |
| 0.35–1.0s | 167,615 | 12.7% |
| ≥1.0s | 1,009 | 24.4% |
| **overall (≥0.08s)** | **438,706** | **5.8%** |

**This match rate is low even where it should be structurally comparable** (VAD's 0.2s floor
means buckets below that are trivially near-zero-match by construction, but the 0.35s+ and 1.0s+
buckets — well above VAD's floor — still only match 12.7% / 24.4% of the time). Two
non-exclusive explanations, neither of which this report can adjudicate:

1. **VAD under-detects**: Silero's `threshold=0.5` speech-probability gate plus the requirement
   of a full 200ms dip is a fairly conservative bar; trailing breath noise, low-level room tone,
   or a speaker who doesn't fully stop voicing during a comma pause can all keep VAD's speech
   probability above threshold even while the aligner (operating at ~40ms frame resolution,
   see §6) registers a real gap between two specific characters.
2. **Aligner timing bias**: forced aligners trained on data where punctuation commonly
   co-occurs with pauses can learn a slight systematic "buffer" around punctuation-adjacent
   character boundaries, inflating Δt beyond the true silence. The clean, prosodically-ordered
   per-mark ranking in §1 argues against this being a large effect, but it cannot be ruled out
   from this data alone.

**Practical takeaway**: treat the VAD-gap list and the aligner-Δt measurement as two
*independent, partially-overlapping* signals, not as a validation/ground-truth pair. Don't
discard either — but do not expect them to agree closely, and prefer the aligner Δt (finer
resolution, directly anchored to the punctuation position) as the primary signal for bucket
calibration.

---

## 4. Distribution shape and bucket-split recommendation

### 4.1 What the plan's §0 prior said

The plan's earlier measurement (all within-segment VAD gaps, not punctuation-anchored) found
p50=0.23s / p90=**0.26s** corpus-wide, and flagged that `segment.vad_cut`'s
`min_silence_duration_ms=300` cutting rule means pauses ≥0.3s mostly become **segment
boundaries** rather than surviving inside a segment — i.e. the corpus may structurally lack
enough "long pause" samples to support a genuine short/long split.

### 4.2 What this measurement found

The punctuation-*anchored* measurement (pooled `normal`, n=646,001) finds a **longer** tail
than that prior: p50=0.24s, **p90=0.48s** — nearly double the earlier p90. This is a genuine
**refinement**, not a contradiction: the earlier measurement pooled *all* VAD gaps regardless of
position (diluting punctuation-specific pauses with shorter non-punctuation micro-hesitations),
and VAD's 0.2s floor + conservative threshold structurally truncates the tail relative to what
the aligner's finer 40ms-resolution timestamps can resolve (see §3). The `segment.vad_cut`
300ms-boundary effect the plan flagged is real and still visible — see the fraction dropping
off past ~0.6s below — but it does not compress the distribution as much as the earlier
all-gaps number suggested.

### 4.3 Histogram (pooled `normal`, 40ms bins, 0–1.2s)

```
[0.00,0.04)  174,817  27.1%  ███████████████████████████
[0.04,0.08)   14,149   2.2%  ██
[0.08,0.12)   51,529   8.0%  ████████
[0.12,0.16)   16,168   2.5%  ██
[0.16,0.20)   37,605   5.8%  ██████
[0.20,0.24)   19,034   2.9%  ███
[0.24,0.28)   59,300   9.2%  █████████
[0.28,0.32)   37,585   5.8%  ██████
[0.32,0.36)   62,133   9.6%  ██████████
[0.36,0.40)   37,378   5.8%  ██████
[0.40,0.44)   40,467   6.3%  ██████
[0.44,0.48)   12,058   1.9%  ██
[0.48,0.52)   35,162   5.4%  █████
[0.52,0.56)    9,034   1.4%  █
[0.56,0.60)   14,913   2.3%  ██
[0.60,0.64)    5,941   0.9%  █
[0.64,0.68)    6,811   1.1%  █
[0.68,0.72)    3,020   0.5%
[0.72,0.76)    2,949   0.5%
 ... (long thin tail beyond 0.76s, 2.3% of mass total)
```

**Shape**: a dominant spike at/near zero (the no-pause population from §2), then — critically —
**not a second, separate hump**. It's one broad, comb-shaped (aliasing from the aligner's 40ms
quantization, see §6) continuum spanning roughly 0.08s–0.6s, decaying gradually rather than
falling off a cliff, with only a thin tail (<3% of mass) past 0.6s. There is no clean valley
that would mark an obvious "this is where short ends and long begins" boundary — any cut point
in the 0.25–0.45s range is defensible but somewhat arbitrary, because the underlying
distribution doesn't hand you a natural seam there.

### 4.4 Mass fractions at candidate cut points (pooled `normal`)

| cut point | fraction ≥ cut |
|---|---:|
| 0.08s | 70.7% |
| 0.15s | 62.7% |
| 0.25s | 42.3% |
| 0.35s | 26.9% |
| 0.50s | 7.5% |

### 4.5 Recommendation (non-binding — owner + canto-tts decide)

- **This corpus supports a two-bucket split, but not a strongly bimodal one.** Using the plan's
  own industry-prior cut points (no-pause<80ms / short 80–350ms / long≥350ms) against the
  *measured* distribution: no-pause 29.3%, short 43.8% (0.08–0.35s), long 26.9% (≥0.35s). That
  is a workable three-way split with real mass in all three bins — **not** a short-only
  degenerate case, contrary to what the plan's §0 all-gaps prior might have suggested.
- If a cleaner short/long separation is wanted, **0.32–0.36s** sits at a local peak-to-trough
  transition in the histogram (bins `[0.32,0.36)` at 9.6% is the highest single bin past the
  no-pause spike, with `[0.36,0.40)` and `[0.44,0.48)` dipping around it) and could serve as an
  alternative to 350ms with very similar mass split — this is a minor variant, not a materially
  different recommendation.
- **Do not calibrate long-bucket thresholds off `。` alone** even though it's the "cleanest"
  mark (lowest hallucination rate) — `。`'s own p50/p90 (0.40/0.64) is pulled up by being
  disproportionately a genuine clause-final mark; pooling is dominated by `，` (77% of the
  sample) and the pooled numbers above are the more representative default for a general
  `<pause-short>`/`<pause-long>` vocabulary that must also handle `，`.
- **Whatever cut point is chosen, expect it to classify a comma as "long" a meaningful
  fraction of the time** (`，` alone: p90=0.48s, i.e. 10% of real commas already exceed a
  0.35–0.48s cut) — this is realistic Cantonese prosody (commas before a strong turn or
  contrastive clause can carry a substantial pause), not a data artifact.

---

## 5. Segment-boundary (trailing) punctuation — different distribution, treat separately

269,001 punctuation instances (in 270,441 analyzed segments — i.e. essentially every segment)
are `trailing_tail`: the mark is the last character reached with no following kept character in
the segment, overwhelmingly `。` (259,643 / 96.5%), then `？` (6,431), `！` (1,970), `，` (954).
For these, Δt is measured against `segments.duration_sec` (the recorded audio's own end) rather
than a following character, since there is no following character.

| cohort | n | p25 | p50 | p75 | p90 | mean | no-pause rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `。`, mid-segment (`normal`) | 75,002 | 0.24 | 0.40 | 0.48 | 0.64 | 0.359 | 17.1% |
| `。`, **trailing** | 259,643 | 0.108 | 0.156 | 0.236 | 0.342 | 0.192 | 16.4% |

**The same mark measures a systematically shorter, more compressed pause when it's
segment-final** — median drops from 0.40s to 0.16s, p90 from 0.64s to 0.34s. This is exactly
the plan's §4 risk prediction confirmed: `segment.vad_cut` trims the true post-utterance
silence down to a small residual buffer (a few hundred ms at most) before the segment master
ever gets written, so trailing-punctuation Δt reflects *leftover trim slack*, not the real
discourse-final pause. It is not literally meaningless (median 0.156s is nonzero and the
no-pause rate is actually slightly lower than mid-segment `。`, since a hard segment cut rarely
lands with zero residual silence at all) — but it is **not comparable** to mid-segment Δt and
should not be pooled into the same calibration.

Also checked, a smaller/weaker version of the same effect exists *within* the `normal`
cohort itself: comparing the first vs. last *fully-flanked* mark per segment (i.e. excluding
true trailing_tail) shows `last_computable_mark` has a slightly lower no-pause rate (24.0% vs.
31.6% for `first_computable_mark`, 30.3% for interior marks) and a slightly higher mean
(0.250 vs 0.221/0.232) — consistent with pauses generally growing a bit as a segment approaches
its own natural end, but this effect is much smaller than the trailing-boundary distortion
above.

**Recommendation**: exclude `trailing_tail` instances from calibration entirely (as this
report already does for §1/§4), and in P2/P3 do not insert a `<pause-*>` token after
segment-final punctuation — there is no reliable acoustic pause left to anchor it to once
`vad_cut` has already trimmed the tail.

---

## 6. Data-quality caveats

- **3.2% of segments (8,869 / 279,348) excluded** for a character-walk mismatch — `best_text`
  and `alignments.chars` didn't line up as a clean subsequence (pointer didn't fully consume
  `chars` by end of text). Not investigated further here (out of scope for this report); if
  material, worth a follow-up before P2 freezes discovery logic, but at 3.2% it does not change
  any conclusion above.
- **Aligner frame resolution ≈ 40ms**: 99.5% of observed Δt values land within 5ms of a 0.04s
  multiple (Qwen3-ForcedAligner-0.6B-hf's frame stride). This produces a visible "comb" pattern
  in the histogram (§4.3) — an artifact of quantized-timestamp subtraction, not real
  periodicity in speech. An 80ms no-pause threshold is 2 frames wide and well clear of this
  noise floor; thresholds finer than ~40ms would not be trustworthy with this aligner.
- **Zero-width character spans**: 8.81% of all 10,680,675 aligned characters (941,439) have
  `start_sec == end_sec` (documented as valid-not-error in `align.py`'s docstring). These are
  not filtered out — a zero-width character immediately before/after a punctuation mark just
  means that character's own duration collapsed to zero in the flanking Δt calculation, it does
  not itself invalidate the Δt measurement. Not separately broken out here; flagged as a
  possible follow-up if a future pass wants to test sensitivity to this population specifically.
- **Consecutive marks** (e.g. a comma immediately followed by a closing quote then another
  mark) share the same flanking pointer window and will report identical Δt for both — a rare
  edge case in Cantonese punctuation, not separately quantified, does not materially affect
  pooled percentiles at this sample size.
- **Half-width `.`** was deliberately excluded from mark normalization (ambiguous with decimal
  points in numbers); the plan's own §0 audit found only ~15 examples corpus-wide, immaterial.
- **`；` sample is thin** (n=1,025 for `normal`, n=3 for `trailing_tail`) — directionally
  consistent with the rest of the marks but treat its specific percentiles as lower-confidence
  than the higher-volume marks.

---

## 7. What this report does NOT do

Per the plan's ownership boundaries: this report does not pick a bucket threshold, does not
write to `pause_calibration.json` (only a clearly-marked `_draft` file), does not touch
`asr_agreement.best_text` or any canonical text field, and does not implement `pause.plan` (P2).
It is read-only analysis over the existing catalog, for the owner (and canto-tts) to use in the
P1 human gate.
