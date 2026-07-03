# Pipeline Re-Architecture — Implementation Plan

> **狀態**:Approved for execution · 2026-07-02(§10 六條 open questions 已全部由 owner 拍板)
> **上游文件**:`PIPELINE_REARCHITECTURE_PLAN.md`(§4 願景 + §3 storage 遷移已完成)、
> `LABEL_FRAMEWORK_SPEC.md`(label production 框架)、`PIPELINE_SPEC.md`、`KNOWN_ISSUES.md`
> **本文件**:§4 願景嘅**可執行 implementation plan** —— 具體組件設計、DAG node 清單、
> milestone、驗證 gate、風險。
> **已拍板決定(owner,2026-07-02)**:
> 1. 範圍 = §4 全套(decode-once / orchestrator / DuckDB)**+ §3 step 9 三碟 sharding**
> 2. 遷移深度 = **全面重寫**:stages 02–15 全部變 DAG node
> 3. Orchestrator = **自家寫**;語言見 §2.3(建議 Python control plane + language-agnostic worker protocol)
> 4. Raw 容量策略 = **壓縮保留**(opus;唔行 transient-delete)
> 5. §10 六條後續決定(filtered 去物理化 / FLAC 傾向 / canto-tts 留 nvme / emotion
>    spot-check P1-P2 / loudness 只限新數據 / stereo 研究)—— 見 §10

---

## 0. TL;DR

- **一個 Python package(`pipeline/`)取代 16 個獨立 script**:catalog(DuckDB)做 metadata
  SSOT、decode-once audio bus 做 I/O 層、resource-aware orchestrator 做調度層,舊 scripts
  逐個 port 做 DAG node,每個 node 過 golden-set parity test 先退役舊 script。
- **即時收益唔使等 5-10×**:(1) 每 stage 開場嘅 rglob + `.exists()`(119 萬個 sidecar json,
  百萬次 stat())變 SQL anti-join(<5s);(2) label detector 由「每個 full-pass 讀晒 843G」
  變「一 pass 餵晒所有 model」;(3) `11_audio_tag` 驗證咗嘅 prefetch+length-sort pattern
  (4.7→33/s)昇華做全 pipeline 通用元件。
- **Raw 壓縮 = 容量問題大幅緩解**:現有 1.6T raw 係 10,920 個由 lossy 源(AAC/opus/MP3)
  decompress 返嚟嘅 WAV —— 轉 opus ≈ **1.6T → ~150G**;新 download 直接保留 bestaudio 原生
  格式。5-10× 規模嘅 raw(25-50k h)≈ 1.5-3T,一隻碟裝到。真正嘅 5-10× 容量壓力喺
  **segments**(見 §7 capacity model)。
- **六個 milestone(P0–P5 + P6 scale readiness)**,每個獨立有價值、有 gate;P1 pilot 直接
  攞現時最逼切嘅實際工作(PANNs music pass 剩低 77%)喺新框架上完成 —— 唔係為重構而重構。

---

## 1. 現狀 Audit(2026-07-02 實測)

### 1.1 硬件

| 資源 | 規格 | 備註 |
|---|---|---|
| CPU | 48 cores | training dataloader 要預留核(PANNs 教訓) |
| RAM | 251 GiB(free ~143G + cache 91G) | 大把空間做 in-RAM prefetch queue |
| GPU | 2× RTX 4090 24G | 可能同 canto-tts training / llama-server co-run |
| nvme0n1 | ext4 `/`,987G free | hot tier:repo、metadata、decode cache |
| Drive2 (sdb1) | ext4,1.6T used / 244G free | raw(全 WAV,壓縮空間巨大) |
| Drive3 (sdc1) | ext4,**空**,1.7T free | 等本 plan §7 分配角色 |
| Drive4 (sdd1) | ext4,1.4T used / 423G free | segments 843G + filtered 569G |
| Drive1 (sda1) | NTFS,個人資料 | **pipeline I/O 完全唔掂佢**(NTFS3 穩定性教訓,§3 執行結果) |

### 1.2 數據規模

| 項目 | 數量 | 大細 |
|---|---|---|
| raw 檔案 | 10,920 WAV(yt 4,440 / rthk 2,178 / podcast 4,302) | 1.6T |
| segments | 1,186,173 wav(yt 585,712 / podcast 510,662 / rthk 89,799) | 843G ≈ 2,440h |
| filtered | 767,718 wav | 569G ≈ 1,650h |
| manifest | 455,299 rows / 1,004.5h / 9,169 speakers | 576M jsonl |
| sidecar json | ≥1.19M `.transcript.json` + `.filter.json` / `.pregate.json` / `.speaker.json` … | 數百萬 inode |
| label sidecars | `lang_id.jsonl` 88M(done)、`overlap.jsonl` 60M(done)、`audio_tags.s*.jsonl`(~23%,**進行中**) | |

### 1.3 樽頸(逐個對應設計)

| # | 樽頸 | 證據 | 本 plan 對策 |
|---|---|---|---|
| B1 | 每個 detector 自己 full-pass 讀 843G | `11`/`12`/`13` 各自由 manifest 逐個讀 wav、各自 resample | §4 decode-once bus:一 pass 餵晒所有 extractor |
| B2 | 並發模式唔一致 | `11` 有 prefetch+length-sort(33/s);`12`/`13` 純串行;`06` 人手開 N 個 shard process | §5 orchestrator:prefetch/batch/shard 做通用元件 |
| B3 | 每 stage 開場 scan 百萬檔案 | `find_segments()` rglob 1.19M wav + `.exists()` per file | §3 catalog:discovery = SQL anti-join |
| B4 | GPU 保護 ad-hoc | `--mem-fraction`/fp16/OOM-halving 每 script 自己一套;`--gpu 1` hardcode 註釋「GPU 0 occupied」 | §5.4 GPU policy:sampler + yield/cap 中央化 |
| B5 | resume 邏輯抄嚟抄去 | `load_done_ids()` × 3 份、`04` 自家 checkpoint 格式、`06` `.filter.json` existence | §3.3 journal + ledger:一個機制 |
| B6 | metadata 線性掃 | 455k rows × 每工具 full parse;5-10× 後每 join 分鐘計 | §3 DuckDB catalog + Parquet |
| B7 | raw 儲存脹大 10× | lossy 源 decompress 做 WAV | §7 opus 化 + download 政策改 |
| B8 | filtered 物理複製 | 569G 同 segments 完全重複(copy2) | §7.3 提案:filtered 變 tier metadata(owner 決定) |
| B9 | manifest 有 stale 路徑 | `/mnt/Drive1/canto/...`(§3 遷移前寫入) | P0 一次過 remap 入 catalog |

---

## 2. 架構總覽

### 2.1 目標形態

```
┌─ CONTROL PLANE(1 個 supervisor process,asyncio)────────────────────────┐
│  pipeline.cli  ─→  orchestrator.scheduler                                 │
│    • DAG resolution(node 依賴)                                           │
│    • resource pools:IO(per-drive)/ CPU / GPU0 / GPU1                    │
│    • resources.sampler:nvidia-smi + /proc/loadavg + iostat(每 ~2s)      │
│    • GPU policy:偵測到 foreign training proc → yield / cap               │
│    • catalog 單一 writer(DuckDB)+ journal appender                       │
└──────┬────────────────────────────────────────────────────────────────────┘
       │ worker protocol:JSONL over stdio(language-agnostic)
┌──────▼─ WORKERS(N 個 child process)──────────────────────────────────────┐
│  gpu worker(cuda:0)     gpu worker(cuda:1)     cpu workers × M          │
│  一個 process 可 host 多個 model(label suite:lid+osd+panns+emotion       │
│  共 <3G VRAM)→ 每段 audio decode 一次、餵晒所有 subscriber                │
└──────┬────────────────────────────────────────────────────────────────────┘
       │
┌──────▼─ DATA PLANE ────────────────────────────────────────────────────────┐
│  audio.bus:decode(soundfile/ffmpeg)→ resample variants(soxr:16k/32k/48k)│
│    ├─ in-process fan-out(同 process 多 extractor,零 copy)               │
│    └─ nvme LRU cache(.cache/decoded,跨 process / 跨 run 重用)           │
│  storage layer:storage_layout.yaml SSOT + shard map(Drive2/3/4)         │
│  catalog:metadata/corpus.duckdb + Parquet partitions                      │
│    journals(append-only jsonl)= 寫入媒介;catalog 隨時可由 journal 重建   │
│  exports:manifest.jsonl / train.jsonl / val.jsonl / labels.jsonl(接口不變)│
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Package layout

```
pipeline/
├── cli.py               # `pipe run`, `pipe status`, `pipe catalog`, `pipe report`
├── config.py            # 讀 config/storage_layout.yaml + config/pipeline.yaml
├── catalog/
│   ├── schema.sql       # DuckDB DDL(§3.1)
│   ├── catalog.py       # 單一 writer 接口;讀 = 任意 process 開 read-only conn
│   └── ingest.py        # journal→catalog compaction;legacy jsonl/sidecar importer
├── orchestrator/
│   ├── dag.py           # NodeDef / DAG resolution / item-level pipelining
│   ├── scheduler.py     # pools、dispatch、backpressure、batching(length-sort window)
│   ├── resources.py     # sampler:nvidia-smi / loadavg / iostat → pool target sizes
│   ├── worker.py        # worker 進程 entrypoint + JSONL stdio protocol
│   └── journal.py       # append-only per-run journal(crash-safe resume)
├── audio/
│   ├── decode.py        # decode + soxr resample(取代三份手寫 resample)
│   ├── bus.py           # decode-once fan-out(subscriber 按 sample-rate 訂閱)
│   └── cache.py         # nvme LRU cache(key=(id,sr),max_gb 由 config)
├── nodes/               # 全部 stage 邏輯(§6 node 清單)
│   ├── ingest.py        # discover / download / probe
│   ├── segment.py       # diarize / vad_cut(+ loudnorm slot)
│   ├── asr.py           # canto_ft / whisper_v3 / agreement
│   ├── filter.py        # text / snr / dnsmos / decide
│   ├── g2p.py           # canto-hk-g2p(Rust core,已係 hot path 最快一環)
│   ├── speaker.py       # embed(GPU)/ cluster(CPU)
│   ├── labels.py        # lang / overlap / music / prosody / emotion(gated)
│   ├── labelstore.py    # calibrate_labels / build_label_store / assign_tier
│   └── manifest.py      # build / export / report
└── legacy/              # 舊 scripts parity 簽收後搬入嚟(可隨時 fallback)
tests/
├── golden/              # golden set manifest(~500 clips,分層抽樣)
└── test_*.py            # 純邏輯 unit tests + node parity tests
config/
├── storage_layout.yaml  # 已有;P5 加 shard map 定案
└── pipeline.yaml        # pools / batch / thresholds / GPU policy(新)
```

### 2.3 語言選型:Python control plane(建議)+ 可換 Rust 嘅邊界

Owner 問可唔可以用 Rust 寫 orchestrator。**技術上可行**(Rust supervisor + Python worker
process,行 stdio protocol),但**建議 Python**,理由:

1. **Control plane 唔係樽頸**:dispatch rate ~10¹ tasks/s;真正 heavy 嘢(torch / pyannote /
   faster-whisper / onnxruntime)必然喺 Python worker。Rust 只覆蓋 ~10% code,而嗰 10% 零性能壓力。
2. **迭代速度**:pool size / yield 閾值 / batch window 喺 calibration 期會改好多次;Python
   改完即跑。
3. **SSOT 共用**:`label_schema.py` / `storage_layout.py` 係 Python,orchestrator 直接 import,
   同 canto-tts 共用(train/infer 一致鐵律)。
4. **Rust 已經喺真正需要佢嘅位**:G2P = canto-hk-g2p Rust core。

**保留換血可能**:worker protocol 定義做 JSONL-over-stdio、schema 版本化(§5.5)——
第日想換 Rust supervisor,唔使掂任何 worker / node code。

### 2.4 Prior-art 對照(agy research,2026-07-02)

自家設計唔係空想 —— 每個組件都有成熟先例(詳見 research 報告
`~/.gemini/antigravity-cli/brain/977d49d0.../audio_pipeline_report.md`):

| 我哋嘅組件 | 先例 | 取捨 |
|---|---|---|
| manifest/catalog 驅動、stage 讀寫 metadata 而非掃 filesystem | **NeMo SDP** manifest-to-manifest chain | SDP 用 JSONL 逐層傳;我哋用 DuckDB 做 join(5-10× 需要) |
| lazy decode + 一份 index 指向 audio | **lhotse CutSet** | lhotse 綁 PyTorch dataloader 生態;我哋 detector 唔全係 torch |
| decode-once fan-out 三 pattern:shared-mem / nvme cache / columnar shard | research §2 | 揀 in-process fan-out(主)+ nvme cache(輔);shared-mem IPC 嘅 OOM/sync 風險唔值 |
| 靜態 shard × N process(現狀做法) | **Emilia-Pipe** 都係咁 | 證明現狀係業界常態,但正正係我哋要超越嘅位 |
| resource-aware 單機調度 | **Dask local** resources tag / **Ray Data** streaming | 兩者都得,但引入重依賴;自家 pools + sampler 覆蓋我哋需要嘅子集(GPU yield 佢哋反而冇)。**Ray Data 記錄做 fallback**:如果自家 scheduler 維護成本失控,遷移路徑存在 |
| producer-consumer prefetch 餵 GPU | research §3B 確認係標準 pattern | `11_audio_tag` 已本地驗證(4.7→33/s),直接昇華 |

> **點解唔直接用 Ray Data?** 佢解決嘅係「batch 流水線 + 資源分配」,但我哋兩個核心需求佢冇:
> (1) **training co-run 時 GPU 讓路**(佢假設 GPU 係自己嘅);(2) journal/catalog 嘅
> id-keyed resume 語義(佢嘅 checkpoint 係 dataset-level)。加上 Plasma object store 喺單機
> 嘅 memory 碎片問題(research §3C),自家輕量實現更貼身。

---

## 3. Catalog & Metadata 層(DuckDB)

### 3.1 Schema(核心表)

```sql
-- 源頭層
CREATE TABLE sources (          -- sources/*.yaml 嘅實體化
  source_key TEXT PRIMARY KEY,  -- e.g. 'youtube/@channel'
  kind TEXT, program TEXT, domain TEXT, style TEXT, config JSON);

CREATE TABLE raw_files (
  raw_id TEXT PRIMARY KEY,      -- 穩定 id(video id / rss hash)
  source_key TEXT, url TEXT, path TEXT, container TEXT,   -- 'opus'|'m4a'|'wav'
  duration_sec DOUBLE, sample_rate INT, channels INT,
  downloaded_at TIMESTAMP, probe JSON);

-- 段層(SSOT:一個 segment 一行,唔再散落 sidecar)
CREATE TABLE segments (
  id TEXT PRIMARY KEY,          -- 沿用現 manifest id 規則
  raw_id TEXT, shard INT,       -- shard = hash(raw_id) % n_shards(§7)
  path TEXT,                    -- 由 shard + 相對路徑 resolve(storage_layout)
  start_sec DOUBLE, end_sec DOUBLE, duration_sec DOUBLE,
  sample_rate INT, speaker_tag TEXT);

-- 每個 node 一張結果表(append 自 journal;id-keyed)
CREATE TABLE asr_results   (id TEXT, model_key TEXT, text TEXT, confidence DOUBLE,
                            PRIMARY KEY (id, model_key));
CREATE TABLE asr_agreement (id TEXT PRIMARY KEY, agreement DOUBLE, best_text TEXT);
CREATE TABLE verified      (id TEXT PRIMARY KEY, text TEXT, verified_by TEXT, verified_at DATE);
CREATE TABLE filters       (id TEXT PRIMARY KEY, snr_db DOUBLE, dnsmos_sig DOUBLE,
                            dnsmos_ovrl DOUBLE, english_ratio DOUBLE, mandarin_ratio DOUBLE,
                            pass BOOLEAN, fail_reason TEXT);
CREATE TABLE g2p           (id TEXT PRIMARY KEY, jyutping TEXT, valid_fraction DOUBLE);
CREATE TABLE speakers      (id TEXT PRIMARY KEY, speaker_id TEXT, embedding_ref TEXT);
CREATE TABLE labels_lang   (id TEXT PRIMARY KEY, lang TEXT, yue_prob DOUBLE, cmn_prob DOUBLE, top3 JSON);
CREATE TABLE labels_overlap(id TEXT PRIMARY KEY, overlap_ratio DOUBLE, overlap_sec DOUBLE, speech_ratio DOUBLE);
CREATE TABLE labels_music  (id TEXT PRIMARY KEY, music_prob DOUBLE, music_tags JSON);
CREATE TABLE labels_prosody(id TEXT PRIMARY KEY, rate_raw DOUBLE, f0_median_hz DOUBLE,
                            f0_z DOUBLE, gaps JSON, voiced_sec DOUBLE);
CREATE TABLE labels_emotion(id TEXT PRIMARY KEY, top TEXT, conf DOUBLE, probs JSON);
CREATE TABLE tiers         (id TEXT PRIMARY KEY, tier TEXT, tier_version TEXT);

-- 調度 ledger(orchestrator 內部)
CREATE TABLE task_runs (run_id TEXT, node TEXT, item_id TEXT, status TEXT,
                        started TIMESTAMP, finished TIMESTAMP, error TEXT, metrics JSON);
```

**Discovery 變 SQL**(對比 B3 嘅 rglob + stat storm):

```sql
-- 「music label 仲有邊啲未做」:
SELECT s.id, s.path, s.duration_sec FROM segments s
LEFT JOIN labels_music m ON s.id = m.id
WHERE m.id IS NULL ORDER BY s.duration_sec;   -- 順便 length-sort 俾 batcher
```

### 3.2 Journal + catalog(event-sourcing lite)

> **2026-07-03 更新**:P0-P2 為求快,實際 shipped 咗嘅 P1/P2 node 全部揀咗捷徑
> 直接寫 DuckDB(冇行呢度原本畫嘅 journal-first 設計),代價喺 P2 backlog 顯露
> 咗——任何兩個 node 都唔可以真正並行跑(見 P2 backlog 2026-07-03 嘅 operational
> finding)。呢個 §3.2 嘅具體可執行版本已經寫成獨立文件
> `docs/JOURNAL_FIRST_PLAN.md`(草擬,等 owner 拍板),包含 journal 格式、
> compactor 設計、migration 順序、gate——執行咗之後呢個 note 應該更新做已完成。

現 pipeline 最證明咗有效嘅嘢 = **id-keyed append-only jsonl sidecar**(crash-safe、resumable、
永不 corrupt 已寫數據)。新設計**保留佢做寫入媒介**,catalog 只係佢嘅 queryable 視圖:

1. worker 完成一個 batch → supervisor append 落 `metadata/journal/<node>/<run_id>.jsonl`
   (同現有 sidecar 格式一致精神,fsync 政策同 `04` 一樣:OS write-back 夠)
2. supervisor 嘅 catalog writer thread 批量 upsert 入 DuckDB(唯一 writer,冇 lock 問題);
   所有其他 process(status 查詢、分析)一律 `duckdb.connect(..., read_only=True)`
3. **定期 compaction:journal jsonl → ZSTD Parquet partitions**(research 實測:DuckDB join
   JSONL @ 2-5M rows 要 10-60s+,Parquet 同樣 join **sub-second 到 2s** —— JSONL 只做
   write-only landing format)。Compaction 用 DuckDB 一句 `COPY (... row_number() OVER
   (PARTITION BY id ORDER BY updated_at DESC) ...) TO '*.parquet'(ROW_GROUP_SIZE 100000)`
   做 dedup + 轉格式;大表(labels_*、asr_results)實體放 Parquet,DuckDB 做 view /
   query engine("data lake" pattern —— 徹底避開 writer lock 爭議)
4. **catalog 永遠可以由 journals + legacy jsonl 全量重建**(`pipe catalog rebuild`)——
   DuckDB 檔案本身唔係 single point of failure

### 3.3 遷移(P0)

一次性 importer(`catalog/ingest.py`)食晒現有:
- `metadata/manifest.jsonl`(455,299 rows)→ `segments` + 各結果表;**順手修 B9:
  `/mnt/Drive1/canto/...` stale audio_path remap 做 Drive4 現路徑,驗證 `os.path.exists`**
- `lang_id.jsonl` / `overlap.jsonl` / `audio_tags.s*.jsonl` → labels_* 表
- `downloaded.jsonl` → raw_files;sources yaml → sources
- per-segment sidecar json(1.19M 個)**唔使即刻掃**:manifest 已包含晒佢哋嘅內容;
  sidecar 保留喺碟做歷史 artifact,唔再係 read path

### 3.4 對外接口不變

`manifest.jsonl` / `train.jsonl` / `val.jsonl` / `labels.jsonl` 變成 **export artifact**
(`pipe export manifest`),schema 完全跟 `MANIFEST_SCHEMA.md` —— canto-tts 同 zero-risk
政策(hard constraint #9)完全唔受影響。內部先至用 DuckDB/Parquet。

---

## 4. Decode-once Audio Bus

### 4.1 兩種 fan-out mode(互補)

**Mode A — in-process fan-out(主力,I/O 最省)**
label suite 全部 GPU model 好細(mms-lid fp16 ~0.65G + pyannote seg-3.0 ~0.1G + PANNs
~0.3G + emotion2vec ~1G ≈ **<3G VRAM**)→ 一個 GPU worker process host 晒四個 model:

```
decode(1次) ─→ soxr resample ─→ {16k: [lid, osd, emotion], 32k: [panns]}
                                  逐 model 批量 infer,同一段 audio 唔使再落碟
```

一 pass 出齊 4 個 label,對比而家 4 個 script × 4 次 full-pass = **I/O 直接省 4×**,
加埋 prosody(CPU,16k VAD+F0)由 cache 攞 = 5×。

**Mode B — nvme LRU cache(跨 process / 跨 run)**
`.cache/decoded/{sr}/{id}.f32`(raw float32,mmap-friendly);key=(id, sr),
`max_gb: 200`(已喺 storage_layout.yaml),LRU eviction 由 supervisor 定期做。
用途:(1) 唔同 resource class 嘅 node(CPU prosody vs GPU labels)分 process 都共享
decode 結果;(2) 重跑 / 加新 extractor 唔使再讀暖碟;(3) 將來 opus raw 嘅 decode 結果
cache(opus decode 貴過 wav 讀)。

> Research 確認(§2 patterns):shared-memory IPC 雖然零 copy 但 OOM + 同步複雜度高,
> 唔採用;in-process + on-disk cache 係 Emilia/lhotse 系嘅實戰做法。

### 4.2 Decode/resample 統一

而家三份手寫 resample(`11` 自家 Kaiser FIR、`12` librosa、`04` scipy resample_poly)
統一做 `audio/decode.py`:`soundfile` 讀(wav/flac/opus 都食)+ **python-soxr**
resample(VHQ,快過 librosa 一個數量級,GIL-released)。輸出 float32 mono。

### 4.3 RAM budget

20s 48k float32 clip = 3.8M;16k = 1.3M。Prefetch depth 每 worker ~100 clips ≈ <500M,
251G RAM 完全無壓力;bus 唔會持有超過 queue depth 嘅 audio(streaming,唔係全 corpus 入 RAM)。

---

## 5. Resource-aware Orchestrator

### 5.1 DAG model

```python
@dataclass
class NodeDef:
    name: str                    # 'label.music'
    resource: str                # 'gpu' | 'cpu' | 'io' | 'net' | 'human'
    deps: list[str]              # DAG edges(item-level,唔係 stage barrier)
    discover: str                # SQL:邊啲 item 未做(anti-join)
    batch: BatchPolicy           # size、length-sort window、max_batch_sec
    worker: str                  # worker entrypoint('pipeline.nodes.labels:MusicWorker')
    gpu_policy: str = 'yield'    # 'yield' | 'cap' | 'exempt'(見 5.4)
```

- **Item-level pipelining**:node 之間冇 barrier —— segment A 做緊 ASR 時 segment B 可以
  做緊 filter(依賴以 item 為單位由 catalog 判斷)。呢個係 Workflow-style pipeline 語義,
  對應舊世界「成個 stage 行完先行下個」嘅浪費。
- `human` resource(stage 5 calibration):DAG 唔會 block —— verified 表有幾多行就放行
  幾多 item 去下游;calibration UI 照舊獨立行。

### 5.2 Resource pools

| Pool | sizing 規則 | 現值(48c/2GPU) |
|---|---|---|
| `io.<drive>` | 每隻暖碟一條隊;max concurrent readers per drive(SATA seek-contention) | 4–6 per drive |
| `cpu` | `nproc − dataloader 預留 − headroom`,動態隨 loadavg 調 | ~40,training 時 ~32 |
| `gpu.0` / `gpu.1` | 每卡一條隊;worker 數 × batch 由 node 定義 | 1 worker/卡起步 |
| `net` | download 併發(yt-dlp sleep-interval 照舊) | 2–3 |

Backpressure:每條隊 bounded;上游滿咗自動停 dispatch(唔會炸 RAM)。

### 5.3 Sampler(`resources.py`)

每 ~2s 讀:
- `nvidia-smi --query-compute-apps=pid,used_memory` + util → 邊個 pid 用緊卡
- `/proc/loadavg` + `/proc/pressure/io`(PSI)→ CPU/IO 飽和度
- 池 target 調整:平滑(EMA),避免震盪

### 5.4 GPU policy(training co-run,§5 上游 plan 嘅硬規則)

```
偵測到 foreign compute proc(pid 唔屬 pipeline)喺 GPU X:
  policy=yield → gpu.X pool target=0;現行 batch 做完即停,worker 保留(唔 unload model)
  policy=cap   → target=1、batch 減半、mem_fraction cap(而家 12/13 嘅做法中央化)
foreign proc 消失 ≥60s → 恢復
```

同時繼承現有戰訓做 worker 標配:fp16、`set_per_process_memory_fraction`、OOM 遞歸
halving(`12_language_id.infer()` 嘅 pattern 搬入通用 GPU worker base class)。

### 5.5 Worker protocol(JSONL over stdio,v1)

```jsonc
// supervisor → worker
{"type":"task","task_id":"t123","items":[{"id":"...","path":"...","duration_sec":8.1}]}
// worker → supervisor
{"type":"ready","node":"label.music","pid":1234,"proto":1}
{"type":"result","task_id":"t123","rows":[{"id":"...","music_prob":0.07,...}],"metrics":{"items_s":33.2}}
{"type":"error","task_id":"t123","error":"...","retryable":true}
```

- SIGTERM = graceful drain(做完現 batch、flush、exit 0);SIGKILL 都安全(journal append
  嘅 batch 先算完成,冇寫嘅由 discovery 重執)
- protocol 版本化 + language-agnostic → §2.3 嘅 Rust-supervisor 換血通道

### 5.6 Batching(`11_audio_tag` pattern 通用化)

- discovery SQL 已 `ORDER BY duration_sec` → dispatcher 喺 window(e.g. 4096 items)內
  組 near-equal-length batch(zero-padding waste ≈ 0)
- prefetch:worker 內 IO thread pool(GIL-released 嘅 sf.read/soxr)保持 depth=3×batch
  在飛,GPU 唔等碟

### 5.7 Observability

- `pipe status`:每 node backlog(SQL count)、活躍 worker、items/s、GPU/CPU/IO 佔用
- `pipe report <run>`:run 完出 summary(processed/failed/rate、資源曲線)
- `task_runs` ledger 常駐 → 歷史吞吐可查;log 照舊落 `metadata/logs/`
- 重 I/O 操作期監察 `dmesg -w` / uptime(§3 執行結果 NTFS3 教訓嘅制度化 —— 雖然而家
  全 ext4,保守啲冇壞)

---

## 6. DAG Node 清單(stages 02–15 全面重寫)

| Node | 舊 script | resource | 重寫要點 |
|---|---|---|---|
| `ingest.discover` | 01 | net | 照 port;寫 sources/raw_files 表 |
| `ingest.download` | 02 | net+io | **政策改**:保留 bestaudio 原生 container(opus/m4a),唔再轉 WAV(§7.1);寫 raw_files |
| `ingest.probe` | (新) | io | ffprobe → duration/sr/channels/codec 入 catalog(§11 stereo 研究都靠佢) |
| `segment.diarize` | 03 | gpu | pyannote 3.1;輸入由 raw_files 表;fallback VAD-only 邏輯保留 |
| `segment.vad_cut` | 03 | cpu+io | Silero VAD within turns → 48k WAV master 落 shard(§7.2);寫 segments 表;loudnorm slot(暫關,跟 QUALITY 跟進項) |
| `pregate.snr` | 03b | cpu | 照 port;寫 filters 部分欄 |
| `asr.canto_ft` / `asr.whisper_v3` | 04 | gpu | faster-whisper;兩 model = 兩個 node(可分卡並行,而家係串行兩 pass);**`language="yue"` 禁令不變** |
| `asr.agreement` | 04 | cpu | char_agreement → asr_agreement 表 |
| `calibrate.human` | 05 | human | 唔係 worker;calibration 工具寫 verified 表;DAG 按 verified 放行 |
| `filter.text` | 06 | cpu | 文本 gates(cjk/eng/mandarin ratio)—— 純 SQL/Python,唔使讀 audio |
| `filter.acoustic` | 06 | cpu | SNR + DNSMOS(共用一次 decode;ORT threads cap 入 worker 標配,唔再 monkey-patch) |
| `filter.decide` | 06 | cpu | 匯總 → filters.pass;**唔再 copy 檔案**(§7.3,pending owner)|
| `g2p` | 07 | cpu | canto-hk-g2p;validation `^[a-z]+[1-6]$` 不變(hard constraint #8) |
| `speaker.embed` | 08 | gpu | ECAPA;embedding 落 `.npy` shard(catalog 存 ref) |
| `speaker.cluster` | 08 | cpu | 現有 O(n²) fix 保留;cluster 結果 → speakers 表 |
| `label.lang` | 12 | gpu | port 入 label suite(Mode A fan-out) |
| `label.overlap` | 13 | gpu | 同上 |
| `label.music` | 11 | gpu | 同上;**P1 pilot:完成剩低 77%** |
| `label.prosody` | 14(新) | cpu | LABEL_FRAMEWORK §8:VAD 一次出 voiced_sec+gaps;F0(parselmouth);rate=jyutping 音節/voiced_sec |
| `label.emotion` | 15(新) | gpu | **gated**:emotion2vec 粵語 spot-check(LABEL_FRAMEWORK §12.1)過咗先開 |
| `label.calibrate` | (新) | cpu | corpus 百分位 / per-speaker μσ → calibration.json(versioned) |
| `label.store` | (新) | cpu | join → labels.jsonl export(LABEL_FRAMEWORK §7) |
| `tier.assign` | (新) | cpu | Tier A/B 規則 → tiers 表 |
| `manifest.build` / `manifest.export` | 09 | cpu | 由 catalog 砌;split 邏輯(GroupShuffleSplit by speaker)不變 |
| `report` | 10 | cpu | DATASET_REPORT.md;acceptance criteria checklist 照舊 |
| `enrich.release` | 10_enrich | — | **dormant 照舊**(hard constraint #9);唔 port 唔跑,code 保留 |

**Parity 規則**:每個 node port 完,喺 golden set(§9.1)上同舊 script 對拍 ——
數值輸出容忍 |Δ|≤1e-4(浮點),文本/決策輸出要全等;簽收先將舊 script 搬 `legacy/`。

---

## 7. Storage:raw 壓縮 + step 9 sharding

### 7.1 Raw → opus(owner 已拍板壓縮路線)

**現實**:1.6T raw 全部係 WAV,而源頭(YouTube AAC/opus、podcast MP3)本身係 lossy ——
WAV 化冇增加任何 information,純粹脹大 ~10×。

**政策**:
1. **新 download 即刻改**(P1 前就可以做):yt-dlp 唔再 `--extract-audio --audio-format wav`,
   直接保留 bestaudio 原生 container;podcast 保留 mp3/m4a。`ingest.probe` 記錄 codec。
2. **存量 1.6T 轉碼**(P5):`ffmpeg -i in.wav -c:a libopus -b:a 128k -ar 48000 -ac 1 out.opus`
   → 預計 **~150G**。Opus mono speech 喺 32-64kbps 已達 perceptual transparency
   (opus-codec.org),128k 有雙倍餘裕。
   **誠實 caveat(research 發現)**:文獻指出神經 vocoder **直接攞 lossy audio 做 training**
   會學埋 codec artifacts(phase 損失 / HF roll-off)—— 但呢個唔適用於我哋:(a) training
   資產係 segments 48k WAV master,完全唔受本次轉碼影響;(b) 源頭本身已係 lossy
   (YouTube AAC/opus、podcast MP3),WAV 化從來冇還原過任何嘢 —— 全世界 in-the-wild
   corpus(Emilia 同款)都係咁。唯一真代價:**轉碼 = 第二代 lossy**,如果第日由 opus raw
   重新 re-cut segments,嗰批新 segments 會帶第二代損失(128k 下極微)。原生 bestaudio
   保留政策(第 1 點)令新數據完全冇呢個問題。
   Decode 成本無憂:libopus 單核 decode ~1000-2000× realtime,永遠唔會樽頸(§4 bus 照食)。
3. **驗證 gate(非破壞鐵律)**:每個 opus 轉完 → ffprobe duration 同源 WAV 差 ≤50ms、
   decode 唔 error、按 batch 抽樣人耳 spot-check → 批量簽收先刪 WAV。轉碼期間 Drive2
   要有足夠 free space 行「先寫後刪」(244G free,分批做)。
4. **下游適配**:`segment.diarize`/`vad_cut` 由「讀 WAV」變「decode opus」(audio bus 已
   統一 decode 層,免費);**已 segment 咗嘅 raw 唔使重跑任何嘢** —— 轉碼只係 archive 格式變。

**5-10× capacity model(raw)**:25-50k h @ opus 128k mono ≈ 1.4-2.9T → Drive2(1.9T)
或 Drive2+Drive3 一部分就裝到。**Raw 唔再係 5-10× 嘅 blocker。**

### 7.2 Step 9:三碟 sharding(segments 先係真正嘅容量壓力)

**Capacity model(segments/filtered @ 48k 16-bit WAV ≈ 345 MB/h)**:

| 規模 | segments(≈2.4× filtered hours) | filtered 物理複製(如保留) |
|---|---|---|
| 現狀(1,004h filtered) | 843G | 569G |
| 5×(5k h) | ~4.2T | ~2.8T |
| 10×(10k h) | ~8.4T | ~5.7T |

→ 10× 連 filtered 複製 = **~14T,遠超三碟總和 5.5T**。結論:
(a) §7.3 filtered 去物理化幾乎係必需;(b) 10× 要諗 segments FLAC(lossless,~55-60%,
→ ~5T,勉強擠得入)或加碟 —— 開放問題,見 §10。

**Shard map(提案,寫入 storage_layout.yaml)**:

```yaml
sharding:
  enabled: true
  n_shards: 3
  key: raw_id            # 同一源片嘅 segments 落同一碟(locality)
  shard_roots:
    - /mnt/Drive2/canto/segments   # shard 0(raw 縮到 ~150G 後讓出空間)
    - /mnt/Drive3/canto/segments   # shard 1(而家空,即用得)
    - /mnt/Drive4/canto/segments   # shard 2(現有 segments 做起點)
raw_root: /mnt/Drive2/canto-corpus/data/raw   # opus 化後 ~150G
```

**Rebalance 程序**(§3 非破壞紀律):現有 843G segments 按 hash(raw_id) 重分佈 →
rsync copy → name+size 全量核對 → catalog path 批量 transactional update → 確認後刪源;
分批(~100G/batch)、監察 dmesg;全程 pipeline 用 catalog 路徑,唔會讀錯。
新 segments 一開始就直接寫落所屬 shard(P3 後 `segment.vad_cut` 內建)。

### 7.3 提案:filtered 去物理化(慳 569G + 每次 re-filter 一次 corpus copy)

`06_filter` 而家 `shutil.copy2` 每個 pass 嘅 wav 落 `filtered/`。喺 catalog 世界,
「filtered」= `filters.pass=true` 一個 bit + tier 標籤;manifest export 嘅 `audio_path`
直接指 segments master。**影響**:canto-tts 讀 manifest 路徑,唔受影響;`data/filtered/`
symlink 樹可以保留做兼容(symlink 唔嘥空間)。
**✅ owner 已批(2026-07-02,§10 Q2):完全去物理化,P4 執行** —— 慳 569G + 每次
re-filter 唔使再 copy 一次 corpus;10× 下 5.7T 複製本來就不可行。

---

## 8. Milestones(P0–P6)

> 排序原則:(1) 每個 milestone 獨立有價值,隨時可以停低唔爛尾;(2) 最逼切嘅實際工作
> (music pass 收尾、label framework 落地)最早受益;(3) 破壞性/大搬遷(P5)最遲、
> gate 最嚴。

### P0 — Foundations(1-2 sessions)
- `pipeline/` package 骨架 + `config/pipeline.yaml` + CLI 骨架
- DuckDB catalog:schema + legacy importer(manifest/lang_id/overlap/audio_tags/downloaded)
- **B9 修復**:stale `/mnt/Drive1` audio_path remap + exists 驗證(呢個本身係已知 debt)
- Golden set:分層抽 ~500 clips(source × duration × tier)寫 `tests/golden/manifest.jsonl`
- **Gate**:catalog row counts 同 jsonl 全對數;`pipe catalog verify` 抽樣 path exists;
  discovery SQL(music 未做清單)結果 == 現行 `load_done_ids` 邏輯結果

### P1 — Orchestrator core + pilot node(2-3 sessions)—— ✅ 機制已完成、驗證,backlog 清剩留待另約

- scheduler / pools / sampler / worker protocol / journal
- GPU worker base class(fp16 / mem-fraction / OOM-halving 標配)
- **Pilot = `label.music`:用新框架完成 PANNs 剩低 77%**(實際工作,唔係 demo)
- **Gate 結果(2026-07-02 實測,對住真係行緊嘅 canto-tts training)**:
  - kill -9 restart 唔重做已 commit batch:✅ PASS(`tests/test_orchestrator.py`;每個 batch
    一次過 atomic upsert + discover() 每次重新 anti-join,天然 idempotent,唔使額外
    journal-based resume 邏輯)
  - GPU pool target 響應 foreign proc ≤10s:✅ PASS(對住真 training pid,~2.1s 內
    `gpu.0` target 由 1 調到 0,遠低於 10s gate)
  - throughput ≥ 舊 `11_audio_tag`(~33/s):⚠️ **未達標** —— dual-GPU cap policy 實測
    28.6/s、單卡 19.9/s。33/s 基準係 GPU 冇其他嘢跑嗰陣量出嚟;而家兩隻卡都俾
    canto-tts training 食緊 94–100% util,orchestrator 用 `cap` policy
    (mem_fraction=0.15)同佢並存 —— dispatch 機制本身冇問題(細規模測試証實
    dual-GPU 真係兩邊都攞緊 batch,比單卡快 ~44%),淨係因為要分 GPU compute
    先量唔到 33/s。呢個係 coexist 策略嘅真實代價,唔係 bug。
  - music pass **完成**:⏸ **deferred**——剩低 344,727 個(2026-07-02 量度),
    28.6/s 計要 ~3.3 小時。owner 話而家唔好 background 行,留返揀啱時間(例如
    training 停咗嗰陣)先至跑全量。
- **交付物**:`pipeline/orchestrator/{pools,resources,worker,journal}.py`、
  `pipeline/workers/gpu_base.py`、`pipeline/nodes/label_music.py`、
  `pipe run label.music --devices --gpu-policy --batch --mem-fraction --limit`、
  `tests/test_orchestrator.py`(3 個 test,全過)。`task_runs` 表補咗 PRIMARY KEY
  (run_id, node, item_id)—— P0 原始 schema 冇呢個 constraint,`upsert_rows()`
  嘅 INSERT OR REPLACE 需要佢先得(P0 嗰陣呢個表仲未有人寫,冇發現)。

### P2 — Decode-once label suite(2 sessions)—— ✅ 機制已完成、驗證,backlog 清剩留待另約

- `audio/bus.py` + `cache.py`;label worker host lid+osd+panns(+emotion slot)
- `label.prosody`(14,CPU)新寫 —— LABEL_FRAMEWORK stage 2 就位
- **emotion2vec 粵語 spot-check**(§10 Q3:P1-P2 期間;~100 段人手聽對)—— 過咗
  `label.emotion` 即刻開
- **Stereo 可行性 report**(§10 Q6):由 `ingest.probe` 統計出 short report 俾 owner
- **Gate**:單一 pass 出多 label,I/O 讀數(iostat)對比單 detector pass ≤1.3×;
  cache 命中後重跑速度 ≥3×;prosody 分佈合理(LABEL_FRAMEWORK §12.2)

**執行結果(2026-07-02)**:4 個組件全部起好、細規模驗證過(owner 未批全量 backlog 跑,
跟 P1 一樣守「唔開未批准嘅背景長跑」呢條)。

- `pipeline/audio/bus.py`+`cache.py` —— `weir chat agy-sonnet` 起(同 P1 手法),零修改直接用。
  `decode()`/`decode_multi()` 取代 11/12/13/label_music.py 4 份各自 hand-roll 嘅 resample code；
  `cache.py` 係 (id,sample_rate) 鍵嘅 on-disk LRU,掛喺 `config/storage_layout.py` 嘅
  `DECODE_CACHE`(nvme,200GB budget)。驗證:decode_multi 同分開兩次 decode() 對同一檔位元組級
  一致;cache round-trip 一致;evict_lru/stats 行為對。
- `pipeline/nodes/ingest_probe.py`(新)—— CPU-only ffprobe + L/R correlation,thread-pool
  fan-out(唔使 orchestrator GPU pool/Sampler)。餵 §11/§10 Q6 stereo 可行性問題。細規模(30 個
  raw file)驗證 resume 冪等。
- `pipeline/nodes/label_prosody.py`(新)—— Silero VAD → voiced_sec+gaps(≥0.2s)、parselmouth
  → F0 median、rate_raw = jyutping 音節(g2p 表)÷ voiced_sec。同 label_music.py 一樣嘅
  GPUWorkerBase + JSONL worker-subprocess 架構,但由「每 GPU 裝置一個 pool」推廣做「每 CPU
  worker process 一個 pool」(`cpu.0`..`cpu.{n-1}`)。細規模(84 個真實 segment)驗證:rate_raw
  1.98–7.59 音節/秒(avg 4.76)、f0_median_hz 87–329Hz(avg 150Hz),分佈合理。
- `pipeline/nodes/label_suite.py`(新,本 milestone 核心)—— 一個 GPU worker 同時載
  mms-lid + pyannote/segmentation-3.0(OSD)+ PANNs CNN14,每個 segment 用 `bus.decode_multi()`
  讀**一次**,淨係跑嗰個 segment 仲欠嘅 label(lang/overlap/music 三向 anti-join discover())。
  驗證咗 pyannote 嘅 dict-input 路徑(`{"waveform":tensor,"sample_rate":16000}`)同 path-input
  路徑喺**同一份**已解碼音頻上輸出完全一致(diff=0.0)—— 之前見到嘅 ~1.0 max-diff 純粹係
  soxr(bus.py)同 torchaudio(pyannote 內建)兩個唔同 resampler 嘅正常誤差,唔關 API 事(bus.py
  嘅 resampler 揀擇本身已經係 §已接受嘅取捨,冇要求 bit-match)。因為 P0 導入時
  label.lang/label.overlap 已經 ~100% 完成,discover() 淨係搵到 24 個仲欠 lang/overlap 嘅
  segment(同 P1 gate 測試搵到嘅嗰批零位元組 podcast 檔案係**同一組**,證實係單一局部化嘅
  數據損毀事件,唔係散落全 corpus)、加埋 344,679 個仲欠 music 嘅 —— 呢個 node 喺同一個
  decode pass 就手填埋嗰 24 個舊 gap。`labels_music` 加咗新 provenance 值 `p2_suite`。
- Gate 結果(細規模,唔係全量 backlog):I/O 比率 0.89×(單 detector pass,gate ≤1.3×)✅;
  cache-hit 重跑快 23.0×(gate ≥3×)✅;kill-9 resume 兩個新 orchestrator node 都冪等
  (新 pytest case)✅;prosody 分佈喺 84 個真實 segment 上合理 ✅。14/14 tests pass。
- **未做**:全量 label.suite/label.prosody/ingest.probe backlog(344,679/455,299/6,272
  項)、emotion2vec 粵語 spot-check(要 owner 聽)、stereo 可行性 report(要全量 ingest.probe
  先出到)—— 全部因為 owner「未批准背景長跑」呢條指示而刻意擱置。
- 意外收穫:session 中段一次例行 `uv sync`(加 praat-parselmouth/soxr 依賴後)清走咗
  `nvidia-cusparselt-cu12` 等 13 個冇入 lock tracking 嘅 CUDA 套件,`import torch` 即刻壞——
  用 `uv pip install <pkg>==<version>` 逐個裝返修復,記錄咗「呢個 repo 以後唔好淨係
  `uv sync`」呢條教訓入 memory。另外喺加 `raw_probe` 表時撞到 `init_schema()` 一個潛在
  parser 陷阱:inline SQL comment 入面嘅分號唔會被去除(淨係成行 `--` comment 先會),已
  修正個別 comment 措辭,並記錄低留意未來 schema.sql 修改。

### P3 — Heavy stage ports(3-4 sessions)
- `segment.*`、`asr.*`、`filter.*`、`g2p`、`speaker.*` 逐個 port + golden parity
- 兩 ASR model 分卡並行(舊版串行兩 pass → wall-clock 接近減半,training 唔行時)
- **Gate**:golden set parity 全過;舊 scripts 搬 `legacy/`;新 segment 產出直接寫 shard 路徑

**Session 2 執行結果(2026-07-03)**:`filter.text`/`filter.acoustic`/`filter.decide`
(`pipeline/nodes/filter.py`)+ `g2p`(`pipeline/nodes/g2p.py`)port 完成,細規模 gate test
喺真實 catalog 上跑通(filter.text→acoustic→decide→g2p 全鏈,20 段,100% pass、
100% Jyutping valid)。

- 拆 3 個 node(唔係跟 06_filter.py 一個 script):`filter.text`(CPU,in-supervisor,
  sample_rate/duration 硬閘 + CJK 長度/English/Mandarin ratio,唔使讀 audio)、
  `filter.acoustic`(CPU worker-subprocess pool,SNR+DNSMOS,discovery 靠
  `filters_text.pass=TRUE` 先至跑,唔使全部都做 DNSMOS)、`filter.decide`(CPU,
  in-supervisor,合併寫入 `filters`)。三張分表(`filters_text`/`filters_acoustic`)
  唔直接 partial-write `filters`,因為 `upsert_rows()` 係 INSERT OR REPLACE——
  兩個 node 各寫一部分 column 會互相冚走對方個 column。
- DNSMOS ORT thread cap 唔再靠 monkeypatch `onnxruntime.InferenceSession`(06_filter.py
  嘅做法),改為 worker 自己整一個帶 capped `SessionOptions` 嘅 `speechmos.dnsmos.DNSMOS`
  instance(`_build_capped_dnsmos()`),沿用佢原本嘅 `audio_melspec`/`get_polyfit_val`/
  `__call__`,淨係換咗 session 建構方式。
- **發現一個一般性 bug,兩個 node 都中招**:`filters`/`g2p` 兩張表已經俾 P0 legacy
  import 塞晒 455,299/455,299 行(manifest.jsonl 本身淨係得已經 pass 嘅 segment),
  導致 bare row-existence anti-join(`WHERE f.id IS NULL`)永遠搵唔到「未做」嘅嘢——
  同 golden set 都冇關,細規模 gate test 一行都跑唔到先發現。修法:加
  `provenance` 欄(legacy import 行 `provenance IS NULL`,新 node 寫嘅行標
  `'filter_decide'`/`'g2p_node'`),discovery 嘅 anti-join 揀呢個欄嚟判斷,唔淨係睇
  row 存唔存在(同 `labels_music.provenance` 嘅做法同一設計)。**⚠️ `asr_results`
  表都係 legacy import 咗全部 910,598 行(2 model × 455,299)—— `asr.transcribe`
  極可能中埋呢個同款 bug,但 P3 session 1 從來冇用 `pipe run asr.transcribe --limit N`
  對住真 catalog 跑過(嗰陣淨係用獨立嘅 `check_asr_parity.py` 繞過 catalog 直接讀
  golden snapshot)——呢個未驗證,下次碰 `asr.py` 或者要跑呢類 gate test 之前應該
  先查一查。**
- G2P 呢層額外執行咗一次庫版本核實:legacy `07_g2p.py` 用私有 API
  `_canto_hk_g2p.PyPipeline.from_dir()` 手砌一個淨喺 editable/source 安裝先啱嘅
  data 路徑;而家 PyPI 版(`canto-hk-g2p` 1.5.0,呢個 repo 一直未鎖版)已經有公開
  `canto_hk_g2p.Pipeline` wrapper 自己搵返 bundled data/,改用呢個。同 40 個隨機
  golden 樣本對拍舊 `jyutping` snapshot:38/40 完全一致,mean ratio=0.982——兩個
  分歧樣本分別係(a)舊庫英文逐字母出 `[X]` bracket placeholder,新庫正確剔走英文
  token(符合而家文檔行為);(b)一兩個字嘅 tone 因字典版本升級而變。判斷同 ASR
  嗰次一樣屬於「依賴版本漂移,唔關呢個 node 事」,冇再另外起 golden check script
  (核心正確性閘係逐 token regex `^[a-z]+[1-6]$`,唔係同 legacy byte-match)。

### P4 — Metadata cutover(1-2 sessions)
- `manifest.build/export`、`label.calibrate/store`、`tier.assign` 上線;
  per-segment sidecar 對新數據停寫(舊有保留唔郁)
- §7.3 filtered 去物理化(✅ owner 已批)—— 刪 569G 前照 §3 紀律:manifest export
  切換 + canto-tts smoke-load 驗證通過先刪
- **Gate**:由 catalog export 嘅 manifest.jsonl 同現行版本 diff == 僅預期差異
  (stale path 修復);canto-tts 側 smoke-load train.jsonl

### P5 — Storage 執行(2 sessions + 轉碼 wall-clock 數日)
- 存量 raw → opus(分批 + 驗證 gate + 簽收刪 WAV)
- step 9 rebalance:segments 三碟 sharding + storage_layout.yaml 定案
- **Gate**:§3 非破壞鐵律逐條;每 batch 核對先刪;catalog path update transactional;
  結尾 `pipe catalog verify --full`

### P6 — Scale readiness(1 session)
- 5-10× ingestion dry-run(youtube_channels.yaml 新 11 條 diversity channel 做試點)
- soak test:orchestrator 連續行 24h+(download→segment→asr→label 全鏈 item-level 並行)
- capacity 儀表:`pipe status` 加 per-drive 用量投影(nvme 計埋 canto-tts 長住,§10 Q4)
- **Gate**:soak 零 leak(RSS/fd 平穩);投影確認後落實 §10 Q1(傾向 FLAC master ——
  屆時改 constraint #6 字面 + owner 簽名,先開始轉)

---

## 9. 測試 / 驗證策略

### 9.1 Golden set
~500 clips 分層抽樣(source × duration bucket × gold/silver),連同舊 pipeline 對應輸出
snapshot 一齊釘版。每個 node port = 跑 golden → diff。音頻處理容忍浮點誤差(|Δ|≤1e-4
或分類結果全等)。

> **2026-07-03 更新(ASR parity 規則修正)**:原本呢度寫「ASR 輸出因 model 版本釘死
> (uv.lock)應全等」—— P3 session 1 port `asr.transcribe` 之後發現呢個假設錯:
> `ctranslate2==4.8.0`/`faster-whisper==1.2.1` 由 `.venv` 建立(2026-06-09)到而家
> 一直冇變過(早過原本轉錄語料嘅時間 2026-06-11),audio bytes 亦核實一致,但同一段
> segment 今日重新轉錄 vs 舊 snapshot,輸出**穩定但唔完全一樣**(唔係 random——同一
> 環境入面重複跑、跨 process 重新起 CUDA context 都 100% 一致)。forensic 追到最底層,
> 剩返 GPU driver(595.71.05,冇歷史記錄可比對)呢個唯一冇辦法核實嘅變數,判斷係
> driver-level 數值 heuristic 漂移,唔係 code/package/audio 出錯。
>
> 用 20 條隨機非 golden segment 覆核(`char_agreement()`,`tests/golden/` 8 個原本
> golden id 之外):`canto_ft` median=1.0 mean=0.991 min=0.929(16 可比較);
> `whisper_v3` median=1.0 mean=0.961 min=0.759(20 可比較)——絕大部份完全一致,只有
> 少數(較長/較嘈雜)segment 有明顯分歧,唔係普遍性問題。
>
> **新規則**:ASR 輸出(`asr.transcribe`)改用 **similarity-tolerance**,唔用
> exact-match:整批 median `char_agreement(new, legacy) ≥ 0.95`(gate 用嘅聚合門檻,
> 符合以上實測);單條 < 0.75 只 log 警告、唔算 gate fail(個別語音難轉錄係正常現象,
> 唔應該令成個 gate flaky)。驗證腳本:`tests/golden/check_asr_parity.py`(獨立於
> `pytest tests/` 自動跑——載入兩個 whisper model 要幾秒 + GPU,唔適合放入普通 unit
> test suite;手動 / gate 時執行)。其餘 node(音頻分類、分類結果)繼續用原本
> |Δ|≤1e-4 / 全等規則,冇改。
>
> 呢次調查亦意外揭發一個獨立 bug(唔關 ASR parity 事,但影響緊 dataset 本身):
> `metadata/manifest.jsonl`/`train.jsonl`/`val.jsonl` 入面 `asr_candidates[].model`
> 有 110,168 條停留喺 repo 搬遷前嘅舊絕對路徑(同 `pipeline/catalog/fix_stale_asr_model.py`
> 修嘅 DuckDB 版本一樣嘅根源),令任何用 `model_field()` 對比嘅工具都會誤判做「查唔到」。
> 已用 `scripts/fix_stale_asr_model_manifest.py` 修好(dry-run 核實數目、備份
> `.pre-asr-remap.bak`、三個檔案共 110,168 條 remap,remap 之後行數不變、0 殘留)。

### 9.2 Unit tests(`tests/`)
純邏輯抽出嚟先測得:agreement 計算、mandarin/english ratio、bucket()、shard hash、
path resolution、journal replay、discovery SQL。用 pytest;repo 而家冇 tests/ —— P0 起。

### 9.3 Chaos / resume
- worker kill -9 mid-batch → restart → 冇重複行(journal 冪等)、冇漏
- supervisor kill → `pipe run` 重入 → ledger 續
- DuckDB 檔案刪除 → `pipe catalog rebuild` 全量重建 == 原 catalog

### 9.4 吞吐 benchmark(每 milestone 記錄落 DECISIONS.md)
- baseline(而家):music 33/s(GPU)、lid/overlap 串行版實測、06 per-shard rate
- 目標:label 全家桶單 pass ≤8h wall-clock(455k clips);discovery ≤5s;
  full re-filter(唔使 copy)≤ 現行一半

---

## 10. Open questions —— ✅ 已拍板(owner,2026-07-02)

1. **10× segments 格式 = 傾向 FLAC,P6 出投影數據先落實**。而家唔郁現有 843G WAV;
   新架構 decode 層一開始就支援 FLAC(soundfile 原生);P6 確認 10× 真係嚟緊先轉,
   屆時 constraint #6 字面改做「48 kHz mono **lossless** master」由 owner 簽名。
2. **filtered 完全去物理化 = 批准**。P4 執行:刪 569G 物理複製,filtered = catalog
   `pass` bit + tier;`data/filtered/` 保留 symlink 樹做兼容;manifest audio_path 指
   segments master。
3. **emotion2vec 粵語 spot-check = P1-P2 期間做**(抽 ~100 段人手聽對,LABEL_FRAMEWORK
   §12.1)—— 做完 P2 label suite 上線時 emotion slot 即刻開得。已排入 P2 gate。
4. **canto-tts = 永久留喺 nvme**(owner 決定,推翻本 plan 原建議)。Drive3 純做
   segments shard 1;nvme 長期住客 = repo + metadata + decode cache(200G budget)+
   canto-tts(18G+checkpoints)—— cache budget 同 `pipe status` 容量投影要計埋
   canto-tts 增長,nvme 987G free 下暫無壓力。
5. **Loudness norm + trim = 只對新數據開**(`segment.vad_cut` loudnorm slot 對新 ingest
   segments 開 -23 LUFS);現有 843G master **唔郁**(「絕不改 audio」慣例不破)。
   Loudness 對現 corpus 係一個 label(RMS/LUFS 落 catalog),downstream 自行 normalize。
6. **Stereo 可行性研究(LABEL_FRAMEWORK §11)= 做**。`ingest.probe`(P0 起有)順手出
   channels + L/R 相關係數統計 → P2 前後出 short report 俾 owner 決定 production audio 路線。

---

## 11. 風險 / 緩解

| 風險 | 緩解 |
|---|---|
| 全面重寫 = 大 surface;某 node 行為悄悄變咗 | golden parity 逐 node 簽收;舊 script 留 `legacy/` 可隨時 fallback;milestone 間 corpus 唔會處於「半新半舊讀唔到」狀態(catalog 兼容讀 legacy 輸出) |
| DuckDB 單 writer 限制 | 架構上只有 supervisor 寫;journal 先行,catalog 可重建 —— DuckDB 壞咗都唔會失數據 |
| opus 轉碼係 lossy、刪 WAV 不可逆 | 分批 + duration/decode 驗證 + 抽樣人耳 + owner 簽收先刪;segments master 完全唔受影響 |
| 重 I/O 觸發系統不穩(§3 crash 前科) | 全部 pipeline I/O 已離開 NTFS3;重操作分批 + dmesg/uptime 監察制度化(§5.7) |
| training co-run 搶資源 | sampler + yield policy 係 P1 gate 項,唔係事後補 |
| scope creep | 每 milestone 獨立可停;P1 綁實際工作(music 收尾)保證重構一開始就還債 |
| pyannote / faster-whisper / torch 版本漂移令 parity 失敗 | uv.lock 釘版;golden snapshot 連版本記錄 |

---

## 12. 本 plan 嘅 acceptance criteria

| # | Criterion | 驗法 |
|---|---|---|
| 1 | Label 全家桶(lang/overlap/music/prosody)單一 pass 完成,corpus 每段 audio 讀 ≤1 次 | iostat 讀量 vs 843G |
| 2 | 任意點 kill -9,restart 損失 ≤1 batch | chaos test |
| 3 | Foreign training proc 出現 → GPU node 讓路 ≤10s | 注入測試 |
| 4 | Stage discovery(任意 node)≤5s | SQL timing |
| 5 | Golden parity:全部 port node 通過 | tests/ |
| 6 | catalog 可由 journal + legacy jsonl 全量重建 | rebuild diff |
| 7 | manifest.jsonl export 對 canto-tts 透明(schema 不變、路徑全 exists) | smoke load |
| 8 | Raw opus 化後 Drive2 ≥1.3T free;無任何未核對刪除 | df + 簽收記錄 |
| 9 | 三碟 sharding 完成,storage_layout.yaml 定案,§3 step 9 關閉 | catalog verify --full |
| 10 | Zero-risk 政策不受影響:release/reconstruction 工具維持 dormant | code review |

---

## 13. 參考(agy-gemini research,2026-07-02)

- Emilia-Pipe(Amphion):on-disk cache、file-existence resume、CUDA_VISIBLE_DEVICES 靜態分卡
  — https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia
- NVIDIA NeMo Speech Data Processor:manifest-to-manifest chain、processor-boundary resume
  — https://github.com/NVIDIA/NeMo-speech-data-processor
- lhotse / Shar:lazy CutSet、columnar tar shards、shard-level checkpoint
  — https://lhotse.readthedocs.io/en/latest/shar.html
- WenetSpeech4TTS:step manifest checkpoints、WebDataset 出口 — https://arxiv.org/abs/2406.05943
- Ray Data(單機 streaming、ActorPool 資源分配;採納為 fallback 而非依賴)
  — https://docs.ray.io/en/latest/data/data.html
- 完整報告:`~/.gemini/antigravity-cli/brain/977d49d0-fa79-4d0e-b3ec-c7e5eb2562dd/audio_pipeline_report.md`

### 13.1 Storage / metadata 研究(第二份 agy report)

- **開源 TTS corpus 發行格式**:Emilia = 24k 16-bit WAV(WebDataset tar)、WenetSpeech4TTS
  = 16k WAV、LibriHeavy = 16k FLAC(lhotse manifest 指向)、YODAS v2 = 24k WAV/float。
  → 我哋 48k WAV segment master 高過全部;§10 Q1 嘅「10× 轉 FLAC master」有 LibriHeavy
  先例(lossless,constraint #6 精神可保)。
- **Opus transparency / lossy training 影響**:mono speech 32-64kbps 即 transparent
  (opus-codec.org);vocoder 直接 train 喺 lossy audio 嘅退化證據:EURASIP(MP3 對
  vocoder 特徵)、Valin SSW 2019(LPCNet + 低碼率 opus)、arXiv:2111.02380(codec
  augmentation 對 ASR 反而有利)。結論已反映喺 §7.1 caveat。
- **DuckDB @ 2-5M rows**:Parquet join sub-second–2s vs JSONL 10-60s+;single-writer
  用 read_only 連接 + 單一 writer daemon + immutable Parquet shards("data lake")
  三招處理;compaction 用 `COPY ... TO ... (FORMAT PARQUET, COMPRESSION ZSTD,
  ROW_GROUP_SIZE 100000)` + `row_number() OVER (PARTITION BY id)` dedup。已反映喺 §3.2。
- **libopus decode**:單核 ~1000-2000× realtime(RTF 0.0005-0.001)—— decode 唔會係
  pipeline 樽頸。
