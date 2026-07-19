# G2P Migration Note: ToJyutping → canto-g2p

**Date**: 2026-06-13
**Author**: canto-g2p project analysis
**Relevant stage**: Stage 7 (`scripts/07_g2p.py`) — this flat-script stage no longer
exists; the same responsibility now lives in the `g2p` DAG node
(`pipeline/nodes/g2p.py`). Migration described below is complete; canto-hk-g2p is the
current and only G2P tool (see `CLAUDE.md` "Issue 1 — G2P tool history"). Kept here as
the historical rationale for why ToJyutping was rejected.

---

## Current situation

Stage 7 uses **ToJyutping** (`ToJyutping.get_jyutping_list(text)`) to generate the
`jyutping` field in the corpus manifest. English characters are handled letter-by-letter:

```
"send"  → [S] [E] [N] [D]
"email" → [E] [M] [A] [I] [L]
"OK"    → [O] [K]
```

This jyutping is then passed to `convert_corpus_to_moss.py` as the `instruction` field:
```
instruction: "Jyutping: keoi5 [S] [E] [N] [D] zo2 [E] [M] [A] [I] [L] bei2 ngo5"
```

---

## The alignment problem

MOSS-TTS-Nano learns to align `text` with `instruction` during training.
When the text contains "send" but the instruction contains "[S] [E] [N] [D]" (4 tokens),
the model has to figure out a 1-to-4 alignment — which is ambiguous and confusing.

```
text:        "佢 send 咗"
instruction: "keoi5 [S] [E] [N] [D] zo2"
              ↑      ↑   ↑   ↑   ↑  ↑
              1:1    1:4 alignment mismatch for "send"
```

---

## Recommended replacement: canto-g2p with Latin passthrough

**canto-g2p** (`pip install canto-g2p`) handles English with full Latin passthrough —
English tokens appear unchanged in the jyutping output:

```
"send"  → "send"    (1:1 alignment, perfect)
"email" → "email"   (1:1 alignment, perfect)
```

Resulting instruction:
```
instruction: "Jyutping: keoi5 send zo2 email bei2 ngo5"
```

The model sees "send" in text AND "send" in instruction — unambiguous 1:1 alignment.

### Additional improvements over ToJyutping

| Feature | ToJyutping | canto-g2p |
|---|---|---|
| English handling | `[S][E][N][D]` (letter-by-letter) | `send` (passthrough, 1:1 align) |
| Number expansion | ❌ None | ✅ 2026年 → ji6 ling4 ji6 luk6 nin4 |
| Date expansion | ❌ None | ✅ 6月13日 → luk6 jyut6 sap6 saam1 jat6 |
| Percent | ❌ None | ✅ 50% → baak3 fan6 zi1 ng5 sap6 |
| Speed | Python single-thread | Rust + Rayon parallel |
| License | BSD-2 | Apache-2.0 |

---

## Migration plan for Stage 7

Replace in `scripts/07_g2p.py`:

```python
# BEFORE (ToJyutping):
import ToJyutping

def text_to_jyutping(text: str) -> Optional[str]:
    pairs = ToJyutping.get_jyutping_list(text)
    tokens = []
    for char, jp in pairs:
        if not jp:
            if char.isascii() and char.isalpha():
                tokens.append(f"[{char.upper()}]")
        else:
            tokens.extend(jp.strip().split())
    return " ".join(tokens) if tokens else None

# AFTER (canto-g2p):
from canto_g2p import Pipeline
_g2p = Pipeline()

def text_to_jyutping(text: str) -> Optional[str]:
    result = _g2p.convert(text)
    return result if result else None
```

### Important: re-run affected stages after migration

If Stage 7 is re-run with canto-g2p:
1. Delete existing `.jyutping.json` files (or add `--force` flag)
2. Re-run Stage 7: `python scripts/07_g2p.py --source all`
3. Re-run Stage 9: `python scripts/09_manifest.py` (regenerates train.jsonl)
4. Re-run `convert_corpus_to_moss.py --hint-rate 1.0` (see below)

---

## Recommended: increase hint-rate to 1.0

The current `convert_corpus_to_moss.py --hint-rate 0.20` only adds the jyutping
`instruction` field to 20% of training records. With canto-g2p available at inference
time (always generating hints for new text), training at 100% hint rate is recommended:

```bash
# Current (20% hints):
python scripts/convert_corpus_to_moss.py --hint-rate 0.20 ...

# Recommended (100% hints, consistent with inference):
python scripts/convert_corpus_to_moss.py --hint-rate 1.00 ...
```

**Note**: A model trained at `hint-rate 1.0` expects hints at inference. Since
canto-g2p generates hints for any text, this is not a limitation.

---

## Format compatibility

The output format is identical to MOSS training expectations:

```
# canto-g2p output for "今日係2026年6月13日":
"gam1 jat6 hai6 ji6 ling4 ji6 luk6 nin4 luk6 jyut6 sap6 saam1 jat6"

# Wrapped as MOSS instruction:
"Jyutping: gam1 jat6 hai6 ji6 ling4 ji6 luk6 nin4 luk6 jyut6 sap6 saam1 jat6"
```

---

## canto-g2p project

- **Repo**: `~/Documents/canto-g2p/`
- **Install**: `pip install canto-g2p` (once published) or build from source
- **License**: Apache-2.0 (compatible with this project)
- **Status**: Phases 0-5 complete, 92 tests passing
