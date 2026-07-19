# Label Framework — Design Spec

> **狀態**:Draft for review · 2026-06-30
> **先讀**:`docs/archive/PIPELINE_SPEC.md`、`MANIFEST_SCHEMA.md`、`QUALITY_SPEC.md`
> **緣起**:`canto-tts/docs/L2_CONTROL_DATA_PREP_PLAN.md`(消費端 L2 control 設計)
> **範圍**:把 quality + L2-control 標註**統一**成一個 upstream「label production」框架,令任何
> downstream model(Nano 0.1B CPU / Local-1.7B GPU production / 未來)自己揀用邊啲 label、點 render。

---

## 0. TL;DR

- **邊界重畫**:corpus pipeline = **label production**(model-agnostic);downstream = **label
  consumption / rendering**(codec/tokenizer-specific)。
- **Contract = 「Raw + reference bucket」(Option A)**:每個 label 同時存連續 `raw` 值同一個由
  corpus calibration 算出嘅 reference `bucket` enum。Downstream 可直接用 enum,或由 raw 用自己嘅
  閾值 re-bucket → 改閾值唔使重跑 audio,兩個 model 共用一份數據。
- **統一**:現有 quality sidecar(`lang_id` / `overlap` / `audio_tags`)+ 新 control label
  (`rate` / `pitch` / `pause` / `emotion`)歸入同一框架、同一 SSOT schema、同一 merge step。
- **三層**:`detector layer`(逐 label raw sidecar)→ `calibrate`(算 corpus 百分位 / per-speaker μσ
  + emotion floor,寫 versioned 常數)→ `build label store`(join + 套 reference bucket →
  `metadata/labels.jsonl`,downstream 一次 join)。
- **唯一硬約束**:`tokens`(=len(audio_codes))同 `<pause-*>` token 注入**必須留 downstream**(要 codec/
  tokenizer);upstream 只出 pause **raw gap list**。
- **Gate**:`emotion` 落框架前**必先過粵語 emotion2vec spot-check**;GPU label(emotion)同 PANNs 一樣,
  **等 GPU free(training 完)先跑**。

---

## 1. 原則(已拍板決定)

| 決定 | 結論 |
|---|---|
| Label contract | **A — Raw + reference bucket**(raw 保留 → future-proof;enum 即用 → 方便) |
| 今期 derive | `rate`、`pitch`、`pause`、`emotion`(`energy` 留 schema 位,暫不算;`duration/tokens` 留 downstream) |
| Refactor scope | **統一** quality + control 入一個 label framework + 一個 SSOT schema |
| Production audio | **先研究** 源頭 stereo 可行性(見 §11),未拍板 |
| 非破壞 | 全部 id-keyed sidecar,additive,可 resume,絕不改 audio,manifest 維持現 schema |

---

## 2. 邊界:Production(upstream)vs Consumption(downstream)

```
┌─────────────────── corpus pipeline (本 repo) ───────────────────┐
│  LABEL PRODUCTION — model-agnostic                              │
│   • 逐 label raw sidecar (detector layer)                       │
│   • corpus calibration → reference buckets                      │
│   • build label store → metadata/labels.jsonl (id-keyed)        │
└──────────────┬─────────────────────────────────────────────────┘
               │  join by `id`  (manifest.jsonl + labels.jsonl)
┌──────────────▼─────────────── canto-tts (另一 repo,read-only 上游) ─┐
│  LABEL CONSUMPTION / RENDERING — codec/tokenizer-specific       │
│   • 揀用邊啲 label;需要時由 raw re-bucket                        │
│   • tokens = len(audio_codes)                                   │
│   • 砌 MOSS `instruction` 字串                                   │
│   • <pause-*> token 注入 text(由 raw gap list + alignment)      │
└────────────────────────────────────────────────────────────────┘
```

**點解 `pause` 跨邊界**:silence gap 嘅**偵測**(時間/時長)係 audio-derived、model-agnostic → upstream。
但 gap → phoneme token 位置嘅 **mapping + `<pause-*>` 注入** 要 tokenizer → downstream。所以 upstream 出
**raw gap list**,downstream 揀閾值同注入。

---

## 3. Label 目錄(統一 catalog)

| label | family | raw 欄位 | bucket enum | bucket 方法 | 來源 / provenance | status |
|---|---|---|---|---|---|---|
| `lang` | quality | `yue_prob`,`cmn_prob` | `yue` / `cmn` / `other` | `cmn_prob≥0.90`→cmn | mms-lid-126 | ✅ done(`lang_id.jsonl`) |
| `overlap` | quality | `overlap_ratio`,`overlap_sec` | (連續;Tier gate `≥0.20`/`≥0.05`) | 閾值 | pyannote seg-3.0 | ✅ done(`overlap.jsonl`) |
| `music` | quality | `music_prob`,`music_tags` | (連續;Tier-B gate) | 閾值(conservative) | PANNs CNN14 | ⏸ 23%(`audio_tags.s*.jsonl`) |
| `rate` | control | `rate_raw`(syll/voiced_s) | `slow`/`normal`/`fast` | corpus 百分位 P25/P75 | jyutping + silero-VAD | ▢ new |
| `pitch` | control | `f0_median_hz`,`f0_z` | `low`/`normal`/`high` | **per-speaker** z(±0.5σ) | parselmouth/pyin | ▢ new |
| `pause` | control | `gaps:[[t_sec,dur_sec]]` | (raw;`<pause-*>` 下游) | — | silero-VAD | ▢ new |
| `emotion` | control | `probs{}`,`top`,`conf` | `neutral`/`serious`/`gentle`/`lively`/(`sad`/`angry` reserved) | argmax + conf floor | emotion2vec | ▢ new ⚠ gated |
| `energy` | control | `energy_dbfs` | `low`/`normal`/`high` | per-speaker z | RMS dBFS | ▢ slot only(暫不算) |
| `tier` | derived | — | `A`(pretrain)/`B`(clean) | 規則(§10) | label store 消費者 | ▢ new |

> manifest 已有嘅 `dnsmos` / `snr_db` / `asr_agreement` / `speaker_id` / `gender` / `style` /
> `english_ratio` **唔重複存**入 label store;tier 規則直接 join manifest 攞。

---

## 4. Contract:Raw + reference bucket(Option A)

每個 control label 喺 label store 都係一個 object:

```json
"rate":  {"raw": 4.5,   "bucket": "normal"},
"pitch": {"raw_hz": 180.0, "z": -0.2, "bucket": "normal"},
"emotion": {"top": "neutral", "conf": 0.72, "bucket": "neutral", "probs": {"neutral":0.72, ...}}
```

- `raw` = 連續值,**permanent、model-agnostic**(改 bucketing 唔使重算 audio)。
- `bucket` = 由 **§9 calibration 常數** 算出嘅 reference enum;downstream 想要唔同 granularity 就由 raw
  自己 re-bucket(用 §6 嘅 `bucket()`,傳自己嘅閾值)。
- `bucket: "unknown"` / 整個 label omit = 該段呢個屬性不可靠(e.g. `rate` 喺 `english_ratio>0.5`、
  `emotion` 喺 `conf<floor`)。

---

## 5. 架構 / Stages

> **實現筆記(2026-07-19)**:下面呢個 diagram 係設計時(2026-06-30)嘅原始構想 ——
> `scripts/labels/` 子套件 + flat `metadata/labels/<name>.jsonl` sidecar。**實際落地嘅實現
> 唔係咁**:detector layer 變咗 `pipeline/nodes/label_*.py` DAG node,raw 輸出寫入 DuckDB
> `labels_*` table(唔係 flat per-id jsonl sidecar)。Node 對應:
> `11_audio_tag.py` → `label.music`(`pipeline/nodes/label_music.py`,寫 `labels_music`);
> `12_language_id.py`/`13_overlap_detect.py` → `label.suite`
> (`pipeline/nodes/label_suite.py`,decode-once fan-out 寫 `labels_lang`/`labels_overlap`
> 連埋 `labels_music`);`14_prosody.py` → `label.prosody`
> (`pipeline/nodes/label_prosody.py`,寫 `labels_prosody`);`15_emotion.py` 呢個 gated GPU
> label**未實現**(仍然停留喺呢份文件描述嘅設計階段)。`calibrate_labels.py` →
> `label.calibrate`(`pipeline/nodes/label_calibrate.py`);`build_label_store.py` →
> `label.store`(`pipeline/nodes/label_store.py`,輸出 `metadata/labels.jsonl`);
> `assign_tier.py`(§10)→ `quality_tier.assign`(`pipeline/nodes/quality_tier.py`,寫獨立
> `quality_tiers` table,唔係寫返 label store/manifest 欄)。下面嘅概念設計(raw+bucket
> contract、schema 概念)保持有效,淨係實現路徑同物理儲存格式已經改變。

```
[detector layer] 逐 label → metadata/labels/<name>.jsonl   (id-keyed raw, resumable, sharded)
  11_audio_tag.py    → labels/music.jsonl      (現有,改 --out)
  12_language_id.py  → labels/lang.jsonl       (現有,改 --out)
  13_overlap_detect.py → labels/overlap.jsonl  (現有,改 --out)
  14_prosody.py  [NEW, CPU]  → labels/prosody.jsonl
        VAD(silero)一次 → voiced_sec + silence gaps
        F0(parselmouth/pyin)→ f0_median_hz
        rate_raw = jyutping 音節數(去英文 placeholder)÷ voiced_sec
        (energy_dbfs slot,暫 null)
  15_emotion.py  [NEW, GPU, gated]  → labels/emotion.jsonl
        emotion2vec → probs + top + conf      (等 GPU free;先過 spot-check)
        │
        ▼
[calibrate]  scripts/labels/calibrate_labels.py
        掃全部 raw → corpus 百分位(rate)+ per-speaker μσ(pitch/energy)+ emotion floor
        → 寫 metadata/labels/calibration.json (versioned: date + git rev + N)
        │
        ▼
[build store]  scripts/labels/build_label_store.py
        join manifest + 全部 labels/*.jsonl by id
        套 label_schema.bucket() → reference buckets
        → metadata/labels.jsonl   ← downstream 一次 join 嘅單一目標
        │
        ▼
[consumer] assign_tier.py (§10) → tier;canto-tts → render
```

`pause` 嘅 gap list 由 `label.prosody` 嘅同一個 VAD pass 一齊出(免重跑 VAD)。

---

## 6. SSOT Schema (concept — 原構想一個獨立 `label_schema.py`,實現後 schema 邏輯分佈落
`pipeline/nodes/label_*.py` 各 node 入面,唔係獨立一個 module)

單一真相源,detector / calibrate / build / **canto-tts infer 共用**(train/infer 一致鐵律)。

```python
# 概念骨架(非最終 API)
@dataclass
class LabelDef:
    name: str
    family: str                 # "quality" | "control" | "derived"
    raw_fields: list[str]
    enum: list[str] | None
    bucket_method: str          # "corpus_pct" | "speaker_z" | "threshold" | "argmax_floor"
    params: dict                # calibration 常數(由 calibrate 填,versioned)
    status: str                 # "active" | "candidate" | "reserved"
    provenance: str

REGISTRY: dict[str, LabelDef] = { ... }
CALIBRATION_VERSION = "..."     # 由 calibration.json load

def bucket(name, raw, *, speaker_stats=None, params=None) -> str:
    """raw → enum。params 可由 caller 覆寫(downstream 自訂閾值 = re-bucket)。"""
```

- calibration 常數**唔 hardcode 入 .py**,而係 load `metadata/labels/calibration.json`(versioned)→
  re-calibrate 唔使改 code,而且 version 進 label store 每行,可追溯。
- canto-tts pin 一個 `CALIBRATION_VERSION`;升版要 explicit。

---

## 7. Label store 格式 — `metadata/labels.jsonl`

```json
{
  "id": "youtube_3f2a...",
  "quality": {"lang":"yue","cmn_prob":0.01,"overlap_ratio":0.01,"music_prob":0.07},
  "control": {
    "rate":  {"raw":4.5,"bucket":"normal"},
    "pitch": {"raw_hz":180.0,"z":-0.2,"bucket":"normal"},
    "pause": {"gaps":[[2.1,0.4]],"total_sec":0.4},
    "emotion": {"top":"neutral","conf":0.72,"bucket":"neutral","probs":{"neutral":0.72,"serious":0.11}}
  },
  "calibration_version": "2026-07-01",
  "provenance": {"rate":"jyutping+silero","pitch":"parselmouth","emotion":"emotion2vec@<rev>"}
}
```

- 唔可靠嘅屬性整個 key omit(唔寫 `null`)→ downstream 見唔到 = unconditional。
- **唔重複** manifest 已有欄位;downstream join `manifest`(or `train/val`)+ `labels` by `id`。

---

## 8. Control label derivation 規格

### 8.1 `rate`(語速)
- 分子 = `jyutping` 音節數(`split()`,去 `[WORD]` placeholder)。
- 分母 = `voiced_sec`(`duration_sec − Σ silence_gap`,silero-VAD)。
- `rate_raw = syll / voiced_sec` → corpus 百分位 bucket(<P25 slow,>P75 fast)。
- **例外**:`english_ratio>0.5` → `rate` omit(音節定義不可靠)。

### 8.2 `pitch`(音高)
- `parselmouth`(Praat,robust)抽 voiced F0 → median(robust 過 mean)。
- **per-speaker 正規化**:group by `speaker_id` 算 μ/σ → z-score → bucket(±0.5σ)。
  「high pitch」= 相對該 speaker 嘅高,先有控制語義。
- speaker 樣本 `<N`(e.g. 5)→ fallback corpus 分佈,bucket 標低信心(可 omit)。

### 8.3 `pause`(停頓)
- silero-VAD silence gap → `gaps:[[start_sec, dur_sec]]`(只留 `dur≥0.2s`)。
- **upstream 只出 raw**。downstream:`<pause-short>`(0.3–0.6s)/`<pause-long>`(>0.6s)嘅閾值 +
  gap→token 位置 mapping(v2.0 保守:只喺標點位)+ token 注入。

### 8.4 `emotion`(唯一弱標,⚠ gated)
- `emotion2vec` → class softmax + conf。
- `conf < floor`(初 0.6,spot-check 校準)→ omit。
- 粗類映射:原生(angry/happy/sad/neutral/…)→ 我哋 enum;初期只保 corpus 有量 + spot-check 過嘅類。
- **GPU label**:同 PANNs 一樣,**等 training 完、GPU free 先跑**(co-run 重 GPU pass 會搶 training)。
- ⚠ **落框架前必過 §12 粵語 spot-check gate**。

---

## 9. Calibration pass

`label.calibrate` (`pipeline/nodes/label_calibrate.py`):
1. 掃全部 raw sidecar。
2. `rate`:corpus-wide P25 / P75。
3. `pitch` / `energy`:per-`speaker_id` μ/σ(+ corpus fallback μ/σ);記低每 speaker 樣本數。
4. `emotion`:由 spot-check 定 conf floor + 保留類。
5. `music` / `overlap` / `lang`:現有閾值一齊收入(統一)。
6. → `metadata/labels/calibration.json`(含 `version`、`date`、`git_rev`、`n_samples`、每屬性常數)。

兩-pass:detector 出 raw → calibrate → build 套 bucket。Calibration 改 → 只重跑 build(快),唔重跑 audio。

---

## 10. Tier framework(label store 嘅 consumer)

`quality_tier.assign` (`pipeline/nodes/quality_tier.py`) join manifest + labels.jsonl,套規則(SSOT 入 schema):

- **Tier A(pretrain,permissive)** = 全部 **減 stage-1-fatal**:壞 transcript、`overlap≥0.20`、純 music/
  silence、`lang=cmn`。
- **Tier B(clean fine-tune,strict)** = `dnsmos≥X` + `music_prob<Y` + `overlap<0.05` + pure speaker +
  好 duration + gold/high-agreement。
- 可額外 bin by DNSMOS(WenetSpeech4TTS 式 curriculum)。
- 輸出:`tier` 寫返 label store(或 manifest 加 `tier` 欄)。

---

## 11. Production audio — stereo 可行性研究(獨立 task)

`canto-tts` doc 提 1.7B 想要 **48k stereo**,但 corpus 現為 **48k mono master**。落任何 re-process 決定前
先**量度**(唔改嘢):
1. 抽 source 原始檔(youtube/rthk/podcast)統計**真 stereo vs mono / dual-mono** 比例(`ffprobe` channels +
   L/R 相關係數;dual-mono = 兩聲道相同 → stereo 無價值)。
2. 估 re-segment/re-filter 成本(stereo 重跑 Stage 3/6 嘅 audio 部分)。
3. 出 short report → 你決定(`mono both` / `保留 stereo for production` / `分開 master`)。
> 同 label framework **解耦**;唔阻 §1–§10 落地。

---

## 12. 驗證 / QC(落 pipeline 前必做)

1. **emotion2vec 粵語 spot-check**:抽 ~100 段人手聽 + 對 label → confusion;粗類準確 <~70% → 只保
   `neutral` vs `non-neutral`,或改 audio-LLM labeler(Qwen2-Audio)。**呢個係 emotion 嘅 gate。**
2. **Bucket 分佈**:每屬性 enum 直方圖;rate/pitch 應接近設計百分位,emotion 預期失衡(→ downstream dropout)。
3. **Pause 抽查**:隨機聽 N 段對 gap 位置。
4. **Schema round-trip**:`raw → bucket → 解析` 一致;canto-tts 用同一 schema 砌 prompt 對拍。
5. **Join 完整性**:`labels.jsonl` 每 id 喺 manifest 有對應;coverage 報告(每 label 幾多 % 有值)。

---

## 13. 時序 / GPU 約束

- **CPU label**(`rate`/`pitch`/`pause`)**而家可跑**(training 進行中),但要記取 PANNs 教訓:適度
  thread cap + prefetch,**唔好食晒 CPU 搶 training dataloader**。
- **GPU label**(`emotion`)**+ PANNs music 收尾**:**等 training 完、GPU free 先跑**(重 GPU pass co-run
  active DDP training 會搶 kernel,throughput 大幅波動)。
- 全部 detector resumable、sharded(雙卡 `0/2`+`1/2`)、id-keyed → 隨時停得、續得。

---

## 14. 分階段落地

| 階段 | 內容 | 產出 | gate |
|---|---|---|---|
| **0. Schema 骨架** | `labels/label_schema.py` REGISTRY + `bucket()` + calibration.json loader | SSOT | round-trip 過 |
| **1. 統一現有** | `11/12/13` 改寫 `--out metadata/labels/<name>.jsonl`;`build_label_store.py` v0(只 quality)| `labels.jsonl`(quality)| join 完整 |
| **2. CPU control** | `14_prosody.py`(rate/pitch/pause raw)跑全量(CPU,co-run-safe)| `labels/prosody.jsonl` | 分佈合理 |
| **3. Calibrate + build** | `calibrate_labels.py` → buckets;`build_label_store.py` 加 control | full `labels.jsonl` | bucket 分佈 / coverage |
| **4. Tier** | `assign_tier.py` A/B | tier | 規則 review |
| **5. emotion(GPU-free 後)** | spot-check gate → `15_emotion.py` 全量 → 併入 store | emotion label | §12.1 過 |
| **(平行)stereo 研究** | §11 report | 決定 | — |

---

## 15. Open questions

1. Label store 係**獨立 `labels.jsonl`**(本 spec 採用,解耦 + 唔郁 manifest 嚴格 schema)定**fold 入
   manifest**?— 採獨立;如你想單一檔可改。
2. `calibration.json` 嘅 per-speaker μσ 表可能大(數千 speaker)→ 放 calibration.json 定另一 sidecar?
3. canto-tts 點 import upstream `label_schema`?(copy-pin 版本 / 做細 package / 純照 spec 重寫)——
   建議 copy-pin `CALIBRATION_VERSION` + schema,升版 explicit。
4. `<pause-*>` token 注入 v2.0 只靠標點位 → 句中停頓會 miss(可接受,留 forced-alignment v2.2)。
5. emotion 粵語若 spot-check 唔過 → 二分類 fallback,定索性 v1 唔上 emotion?
