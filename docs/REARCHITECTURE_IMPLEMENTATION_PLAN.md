# Pipeline Re-Architecture — Implementation Plan

> **狀態**:Approved for execution · 2026-07-02(§10 六條 open questions 已全部由 owner 拍板)。
> **2026-07-19**:P0–P5 已全部完成(見底下 §8),文件瘦身 —— 詳細設計章節(現狀 audit、
> 組件設計、catalog schema、DAG node 清單、storage 方案、測試策略、open questions、風險、
> 參考資料)已搬去 `docs/archive/REARCHITECTURE_IMPLEMENTATION_PLAN_DESIGN_DETAIL.md`,
> 本檔淨低 TL;DR + §8 Milestones(CLAUDE.md 引用嘅權威 milestone 狀態來源)。
> **上游文件**:`PIPELINE_REARCHITECTURE_PLAN.md`(§4 願景 + §3 storage 遷移已完成)、
> `LABEL_FRAMEWORK_SPEC.md`(label production 框架)、`PIPELINE_SPEC.md`、`KNOWN_ISSUES.md`
> **本文件**:§4 願景嘅**可執行 implementation plan** —— 具體組件設計、DAG node 清單、
> milestone、驗證 gate、風險。
> **已拍板決定(owner,2026-07-02)**:
> 1. 範圍 = §4 全套(decode-once / orchestrator / DuckDB)**+ §3 step 9 三碟 sharding**
> 2. 遷移深度 = **全面重寫**:stages 02–15 全部變 DAG node
> 3. Orchestrator = **自家寫**;語言見 archive doc §2.3(建議 Python control plane + language-agnostic worker protocol)
> 4. Raw 容量策略 = **壓縮保留**(opus;唔行 transient-delete)
> 5. §10 六條後續決定(filtered 去物理化 / FLAC 傾向 / canto-tts 留 nvme / emotion
>    spot-check P1-P2 / loudness 只限新數據 / stereo 研究)—— 見 archive doc §10

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
  格式。5-10× 規模嘅 raw(25-50k h)≈ 1.5-3T,一隻碟裝到。真正嘅 5-10× 容量壓力喺 segments
  (詳細 capacity model 見 archive doc §7)。
- **六個 milestone(P0–P5 + P6 scale readiness)**,每個獨立有價值、有 gate;P1 pilot 直接
  攞現時最逼切嘅實際工作(PANNs music pass 剩低 77%)喺新框架上完成 —— 唔係為重構而重構。

詳細設計(現狀 audit、架構總覽、catalog schema、decode-once bus、orchestrator、DAG node
清單、storage 方案、測試策略、open questions、風險、acceptance criteria、參考資料)見
`docs/archive/REARCHITECTURE_IMPLEMENTATION_PLAN_DESIGN_DETAIL.md` —— P0–P5 已完成落地,
呢啲章節而家係歷史設計記錄,唔再係日常參考。

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

**Session 3 執行結果(2026-07-03)**:`speaker.embed`/`speaker.cluster`(`pipeline/nodes/speaker.py`,
commit `1d629c7`)完成——同 filter/g2p 一樣嘅 hybrid reuse-first 設計,呢次係喺 speaker.embed
先發現:對現有 catalog 隨機抽 2,000 個樣本,100% 已經有舊 `scripts/08_speaker_id.py` 留低嘅
`.embed.npy` sidecar,於是全量 backlog(455,269 段)淨係使 CPU thread-pool 做 sidecar
existence check 就搞掂(455,245 reused + 24 corrupt file,0 GPU compute),~65 分鐘走完。
`speaker.cluster` verbatim port 咗 `cluster_embeddings()`(agglomerative + 大源
sample-and-assign fallback),用 co-clustering Adjusted Rand Index(唔係 speaker_id 字串對
拍,因為 cluster 整數 id 本身冇意義)驗證:rthk 全源(36,726 段)ARI=0.8248,同舊版高度
吻合。9/9 新 test 過。呢個係第一個由 `agy-sonnet` 起草、我 review+接線+補測試+commit 嘅 node
(建立咗呢個 session 起沿用嘅分工模式)。

**Session 4 執行結果(2026-07-03)**:`segment.diarize`/`segment.vad_cut`/`pregate.snr`
(`pipeline/nodes/segment.py`,port 自 `scripts/03_segment.py` + `scripts/03b_acoustic_pregate.py`)
起好,`agy-sonnet` 起草、我 review+接線+補測試+commit(同 S3 一樣嘅分工)。

- Schema 加咗 4 樣嘢:`segments.raw_id`(連返 raw_files,legacy 455,299 行 NULL)、
  `diarization_turns`(segment.diarize 自己嘅輸出——每個 speaker turn 一行,俾
  segment.vad_cut 讀)、`raw_segments`(per-raw-file 完成 marker,`provenance` 分
  `legacy_reused`/`segment_vad_cut`/`diarize_failed` 三種,兩個 node 都憑呢張表
  anti-join 判斷「呢個 raw file 洗唔洗再做」)、`pregate`(pregate.snr 自己嘅輸出,
  獨立於 `filters_acoustic`,因為呢個係 ASR 之前嘅平早 reject,同最終權威嘅
  filter.acoustic 用緊唔同嘅 SNR 公式,故意唔要求數值一致)。
- **關鍵發現(抽查證實)**:現有 6,272 個 `raw_files` 幾乎全部(隨機 50 個抽樣 48/50 命中)
  已經有舊 `scripts/03_segment.py` 留低嘅 `{stem}_segments.jsonl` sidecar——即係話呢批
  raw file 已經俾舊 pipeline 全面 diarize+VAD-cut 完,再行一次全量 GPU diarization
  會係純浪費(重複起新 id 嘅 segment WAV)。故此 segment.diarize 抄返 speaker.embed
  嗰個 hybrid reuse-first 設計:sidecar 命中就淨係寫一行 `raw_segments`
  marker(`legacy_reused`),唔起 GPU;剩低嗰 4-5% 真係冇 sidecar 嘅先至真係走 GPU
  fallback(pyannote 3.1,或者冇 HF token 就跌落 VAD-only 模式——同舊 script 一樣)。
- **真實(非合成)gate test**:`pipe run segment.diarize --limit 20`——20 個真 raw_files 入面
  16 個 sidecar 命中(`legacy_reused`),4 個真係冇 sidecar,自動起咗 2 隻 GPU worker
  (呢個環境冇設 `HUGGING_FACE_HUB_TOKEN`,故正確跌落 VAD-only fallback,每個 file 寫低
  一行 `SPEAKER_UNKNOWN` 全長 turn),`diarization_turns` 表寫入 4 行、`raw_segments`
  寫入 16 行,全部行為符合預期。
- **`segment.vad_cut`/`pregate.snr` 嘅真實剪片邏輯**:淨係喺 unit test 用 monkeypatch
  嘅 VAD window(唔靠 Silero VAD 對合成噪音嘅真實判斷,但 audio I/O / WAV 寫檔 /
  segment id 生成 / duration 邊界 gate 全部行緊真代碼)驗證,冇喺 production 走全套
  真實剪片——因為呢一步會喺 `/mnt/Drive4/canto/segments/` 底下寫低**新嘅實體 segment
  WAV 檔**同喺 catalog 新增真 `segments` 行,同之前 P3 S1-S3 淨係寫 metadata 行嘅性質
  唔同,故意留返俾 owner 話事先至喺 production 對住嗰 4 個真 raw_id 跑
  `segment.vad_cut`。14/14 新 test 過,90/90 全 suite 過。
- **下一步(owner 話事)**:(a) 對住嗰 4 個已經有 `diarization_turns` 嘅真 raw_id 跑
  `segment.vad_cut` + `pregate.snr`,驗證真實剪片+SNR gate 喺 production 數據上work;
  (b) P3 全部 4 個 session 完成後,標誌住 P3 milestone 全部 node port 完(asr→
  filter/g2p→speaker→segment),可以開始諗 P4(metadata cutover)。

### P4 — Metadata cutover(1-2 sessions)
- `manifest.build/export`、`label.calibrate/store`、`tier.assign` 上線;
  per-segment sidecar 對新數據停寫(舊有保留唔郁)
- §7.3 filtered 去物理化(✅ owner 已批)—— 刪 569G 前照 §3 紀律:manifest export
  切換 + canto-tts smoke-load 驗證通過先刪
- **Gate**:由 catalog export 嘅 manifest.jsonl 同現行版本 diff == 僅預期差異
  (stale path 修復);canto-tts 側 smoke-load train.jsonl

**執行結果(2026-07-03)**:`tiers.provenance` 欄(同 filters/g2p 一樣嘅 legacy-row-collision
修法)+ `pipeline/nodes/tier.py`(`tier.assign`)+ `pipeline/nodes/manifest.py`
(`manifest.build`/`manifest.export`)+ `pipeline/nodes/label_calibrate.py`/`label_store.py`
(`label.calibrate`/`label.store`)全部起好、對住真 catalog gate 測試過。分工:三組 node 嘅初稿
交 `weir chat agy-sonnet`(精確 spec 先寫、agy 出碼、我 review+wire CLI+test+commit),
manifest.py 因為涉及 legacy-row-collision 正確性(hard-constraint 敏感)自己直接寫。

- **asr_results carryover(P3 S2 遺留、標為「非常可能有同一個 bug」)先查清**:對住真 catalog 直接
  驗證,證實**冇呢個 bug**——`asr_results` 用 `(id, model)` 做 composite PK,discovery 用
  `a.model = model_field(key)` 精確字串比對(唔係淨睇 row 存唔存在),新 segment 嘅 id 本身就唔喺
  表入面,天然唔會撞到 filters/g2p 嗰種「成表俾 legacy 佔晒」嘅陷阱——刻意設計成咁,因為 GPU
  Whisper 全 corpus 重跑成本極高,唔應該好似 filters/g2p 咁畀所有 legacy row 都排隊等重新驗證。
  冇改任何 code。
- **tier.assign**:port `09_manifest.py` 內嵌嘅 gold/silver 邏輯做獨立 node,`excluded`(agreement
  <0.65 且未人手驗證)呢個 sentinel 係新加嘅(舊 script 淨係 `return None` 唔寫嘢,呢度改成一定
  寫一行,同 g2p_node 嘅「always write a row」原則一致,唔使 discovery 每次都重新掃到)。真實
  20,000 行 backfill 中途停低(邏輯太簡單,唔係 gate-critical,manifest.build 讀 `tier IN
  ('gold','silver')` 唔理 provenance)——驗證 0 重複、分布合理、`tiers` 行數仍然啱好
  455,299。全量 legacy backlog 重新標記留返 owner 決定要唔要跑,唔影響 manifest 正確性。
- **manifest.build/export**:單一 catalog join 取代 `09_manifest.py` 嘅 file-glob+sidecar 讀法。
  關鍵設計:「新 node 旗標 OR legacy 冇 provenance 嘅舊 row」呢個 OR 條件用嚟喺**讀端**處理
  filters.pass/g2p.valid_fraction 嘅 legacy-row-collision(filter.py/g2p.py 之前淨係喺**寫端**
  fix 咗,讀端要自己諗清楚);刻意**冇** join `speakers` 表(淨係 `rthk` 一個 source 做過真正
  clustering,join 咗會靜雞雞淨改一個 source 嘅 speaker_id,同現行 manifest 唔一致);train/val
  split 讀番現有 train.jsonl/val.jsonl 嘅 id 做「保留現有 membership」,淨係將**真係新**嘅 id
  行返 legacy 嘅 stratified-split 邏輯——唔係次次 export 都由頭切一次(嗰個「揀邊個 speaker 做
  val」次序本身就係 filesystem glob 順序嘅意外產物,唔係有意義嘅 key,由頭切會靜雞雞 leak 返
  之前揀開做 val 嘅 speaker 入 train)。
- **label.calibrate/label.store**:對應 `docs/LABEL_FRAMEWORK_SPEC.md` §§7-9,統一現有
  lang/overlap/music/prosody 做 `metadata/labels.jsonl`,rate 用 corpus P25/P75、pitch 用
  per-speaker z-score(<5 樣本 fallback corpus 統計,σ=0 clamp 落 1.0Hz epsilon)做 bucket。
  Emotion/energy 刻意剔除(emotion 未過 owner 粵語 spot-check;energy 淨係 schema 位,detector
  未跑過)。
- **真實全量 gate test**(先備份 3 個現有檔案先再覆寫):`manifest.build()` 完全重現現有已知
  基線(455,299 entries / 1004.5h / 9,169 speakers / gold=16,585 / silver=438,714 —— 同持久化
  memory 完全一致)。真正重新 export 之後同備份逐行 diff:**455,299 行入面淨係 2 行有差**
  (一個 `jyutping`、一個 `snr_db`),兩個都解釋到——係 P3 S2 自己嗰次細規模 gate test 已經真係
  行過 20 個真實 segment 過新嘅 filter.decide/g2p_node,啱啱好改咗嗰兩個 catalog row(係真係
  live 嘅更新,唔係壞咗)。val.jsonl 100% 一模一樣(唔理順序)。呢個結果好過原本 plan 預期嘅
  gate criterion(「淨係預期嘅 stale-path 差異」——因為 stale path 早喺 P0 已經修好咗)。
- 23 條新 unit test + 2 條新 catalog gate test(tiers 行數/重複/合法值 invariant、manifest.build
  總數同已知基線一致),全套 116/116 過。
- §7.3 filtered 去物理化(刪 569G)**刻意未執行**——plan 本身要求 canto-tts 側 smoke-load
  train.jsonl 先,呢個係跨 repo 動作,唔喺呢個 session 範圍入面,留返 owner 明確授權先做。

**Follow-up 執行結果(同日,2026-07-03)**:owner 拍板「commit 呢批改動 + 跑埋 tier.assign 全量
backlog」。Commit `1b993df`(14 個檔案:6 改動 + 8 新增,pipeline/nodes/{tier,manifest,
label_calibrate,label_store}.py + 對應 4 個 test 檔)。跟住跑 `pipe run tier.assign`(冇
`--limit`)去到全量:剩低 430,296 行 legacy backlog 喺 3521 秒(~122 rows/s)內全部重新標記
`provenance='tier_assign'`,0 error、0 excluded。驗證:`tiers` 表仍然啱好 455,299 行、100%
帶有 `tier_assign` provenance、gold/silver 分布(16,585 / 438,714)同遷移前一模一樣(證明呢個
port 係 behavior-preserving,冇改變任何邏輯結果)、0 重複 id、backlog 清零。全套 116/116 test
再跑一次仍然過。tier.assign 呢一項而家**完全 close**(唔再係「code-complete 等跑」,而係
「成個 corpus 真係行咗一次」)。

**§7.3 執行結果(2026-07-04)**:canto-tts 側 smoke-load 用真實嘅
`canto-tts/scripts/convert_corpus_to_moss.py`(train.jsonl/val.jsonl 嘅實際下游消費者)對住
3,000 行 train 隨機樣本 + 全部 2,024 行 val 跑,100% `ok`、零 OOV。過程中發現 manifest 嘅
`audio_path` 其實仲指緊 `filtered/`(569G 複製),未指 `segments/` master(843G)——§7.3
講嘅「manifest export 切換去 segments master」呢步之前從未做過。跟足 §3 紀律,喺改 catalog 之前
做咗全量 name+size 核對:455,299 行入面 0 missing、24 個 size mismatch——查清楚全部 24 個都係
`filtered/` 果邊 0-byte 壞檔、`segments/` 反而係完整檔案(單一 podcast 來源
`馬修香港靈異故事集sp09夜更的士一`),即轉去 segments/ 會順便修好呢 24 行現存壞資料,唔係新增
風險。攞到明確 owner 授權(auto-mode classifier 第一次啱啱擋咗個冇 scope 嘅嘗試,要求先問清楚)
之後,執行 `UPDATE segments SET audio_path = replace(...)`:455,299 行更新,0 行殘留
`/filtered/`,`segments` 表總數 455,330(455,299 legacy + 31 P3 S4 新 segment)。再全量驗證:
0 missing、0 zero-size。重新 `manifest.export`:總數完全一致(455,299/1004.5h/9,169
speakers/gold=16,585/silver=438,714,train/val split membership 保留),同備份逐行 diff:
**455,299 行全部差異淨係 `audio_path`**,冇任何其他欄位變動。再做多次 canto-tts smoke-load
(新路徑):100% ok,零 OOV。全套 116/116 test 再過。

**§7.3 最終執行(同日)**:owner 對實際 `rm -rf` 指令明確拍板。刪除前最後再核一次(manifest 三
個檔案 + catalog `segments.audio_path` 都係 0 reference `/filtered/`),然後執行
`rm -rf /mnt/Drive4/canto/filtered/`(exit code 0)。`/mnt/Drive4/canto/` 而家淨返
`segments/`;`df -h` 可用空間由 422G 升到 991G,即刻回收 569G(同 `du -sh` 估算完全脗合)。
全套 116/116 test 再過一次。**P4(全部,包括 §7.3)而家真係完全 close** —— 兩個 gate criteria
(manifest 重現基線總數、canto-tts smoke-load 過)喺 remap 前後各驗證咗一次,569G 物理複製亦已
不存在。下一個 milestone 係 P5(storage 執行:raw→opus 轉碼 + 三碟 sharding),未開始,一樣要
owner 明確拍板先做。

## 儲存容量調查(2026-07-04,P5 未開始前嘅前置分析)

Owner 問:全部 dataset 都用壓縮格式儲存,Drive2/3/4 加埋總共可以裝幾多小時訓練用 segment?
量咗真實檔案密度(48kHz/16-bit mono WAV = 精確 96,000 bytes/秒,同 catalog 完全脗合,無異議)。

**發現一個重要現象,已核實唔係 bug**:`/mnt/Drive4/canto/segments/` 目錄實際有
**1,186,204** 個 `.wav`(834 GiB),但 catalog `segments` 表淨係 **455,330** 行(1004.5h)——
比例 2.6×,三個 source(youtube/podcast/rthk)獨立核對都係 2.3-3.0× 咁上下,唔係個別現象。
抽樣核實(`20251016_tvb_無綫新聞_yi9MD8XPMck`:磁碟 176 個 wav vs catalog 得 73 個)確認
根本原因對返 Stage 3/Stage 6 嘅設計:**Stage 3(`segment`)將 diarization+VAD 切出嚟嘅全部
clip 寫入 `data/segments/`(唔理質素);Stage 6(`filter`)先揀返合格嗰批入 manifest/catalog**
——即係話磁碟上 61%(~511GiB)嘅音頻其實係已經被 filter 淘汰、但從來冇刪過嘅候選 clip,
Stage 3 output 本身就係 Stage 6 pass 嘅 superset,非 migration 遺留或者 orphan 資料。

**Owner 決定(2026-07-04)**:呢 511GiB **暫時保留**,唔刪。理由:保留呢批已淘汰候選 clip
可以喺將來 quality threshold 改動(好似 `lang_screen.auto` 果日改咗 3 次咁)時直接重新跑
`filter.decide` 揀多啲/少啲,唔使由頭再做一次 diarization+VAD-cut(先係真正貴嘅嗰步)。
**只有喺磁碟空間真係唔夠嘅時候**,先考慮淨係保留 `filters.pass=true` 嗰批(即係重新做返類似
舊時 `filtered/` 咁嘅去蕪存菁),到時先刪呢批候選 clip。呢個係一個**容量壓力觸發嘅刪除
選項**,唔係即時行動——決策記喺呢度,連同 [[canto-corpus-rearchitecture]] memory,等下次
真係捉襟見肘嘅時候唔使重新調查一次。

**容量規劃嘅結論**(基於「保留候選 clip」呢個前提,唔計呢 511GiB 做可回收空間):
- Raw→opus 轉碼(P5)預計喺 Drive2 釋放 ~1.3-1.5TiB(現時 1.6T raw → 估計壓縮到
  ~150-300GiB)
- 3 碟現時 free space 加埋 ~2.9TiB
- 總可用(唔計候選 clip 回收)≈ 4.3TiB,對比 WAV(345.6MB/h)/FLAC(~190MB/h,~55% WAV)/
  Opus 128kbps(57.6MB/h)分別可裝 ~13,700h / ~24,900h / ~82,100h 新 segment(呢個係
  **未套用 reject-clip overhead 嘅理論值**——見下面修正)。

**修正(2026-07-04,同日跟進):上面嘅 13,700h/24,900h/82,100h 冇計「保留候選 clip」政策
對將來新 segments 嘅持續成本**。因為 owner 已經拍板保留 Stage-6-rejected clip(唔止舊資料,
新 raw 未來一樣咁做),所以將來每加一個 catalog-hour,實際磁碟寫入都係 ~2.46×(實測
ratio:834GiB 實際 / 339GiB catalog 對應量)。套用呢個 overhead 落 4.3TiB 可用空間:

| 格式 | 理論新增(冇 overhead) | 套 2.46× overhead 嘅**實際**新增 | + 現有 1,004.5h | vs 10× target(10,000h) |
|---|---|---|---|---|
| WAV | ~13,000h | ~5,300h | ~6,300h | ❌ 唔夠(只夠 5×) |
| FLAC | ~23,700h | ~9,650h | ~10,650h | ✅ 剛好夠(~6.5% margin) |
| Opus 128k | ~78,300h | ~31,800h | ~32,800h | ✅ 遠超 |

**結論:WAV 實際上守唔住 10× scale target,FLAC 剛好可以** —— 呢個係 §10 Q1 嘅
「P6 投影數據」,已經足夠 owner 拍板落實 FLAC(見 §10 Q1 同 `DECISIONS.md` 2026-07-04
條目)。Sensitivity:如果將來因為容量壓力觸發咗「淨保留 pass=true」嘅 fallback,
overhead 會跌返落 ~1×,WAV 都會夠——但依家嘅政策係暫時保留候選 clip,所以呢個修正
先反映現行政策下嘅真實數字。

### P5 — Storage 執行(2 sessions + 轉碼 wall-clock 數日)

**P5-A + P5-B 已於 2026-07-05 執行(P5-C 三碟 sharding 留待下個 session,owner 選擇)** —— 下面係
3 個子步驟嘅詳細 implementation plan(2026-07-04 補充,配合 FLAC master 決定,見 §10 Q1 /
`DECISIONS.md`)。三步有嚴格先後次序(見「排序理由」),唔可以打亂。執行結果見本節末
「✅ P5-A + P5-B 執行結果(2026-07-05)」。

#### P5-A. Segments 轉 FLAC 輸出(最先做,風險最低,純 additive)

- **改嘅位**:`pipeline/nodes/segment.py` `_vad_cut_one()` 第 894-897 行:
  ```python
  seg_name = f"{stem}_seg{n_seg:05d}.wav"          # 改做 .flac
  sf.write(str(seg_path), clip, TARGET_SR, subtype="PCM_16")  # 改做 format="FLAC"
  ```
  `soundfile`(libsndfile 1.2.0,已裝)原生支援 FLAC 寫入,唔使加任何新 dependency。
- **Catalog 唔使改 schema** —— `segments.audio_path` 本身已經存全路徑(包括副檔名),
  `pipeline/audio/bus.py` 嘅 `decode()` 已經文檔明寫「any format soundfile can open
  (WAV, FLAC, OGG, ...)」,filter/g2p/speaker/manifest 全部經呢層讀,對格式透明。
- **Golden-set 驗證要調整**:`_vad_cut_one` 輸出而家係 FLAC,同舊 WAV snapshot 唔可以
  逐 byte diff(壓縮格式唔同),要改做「decode 後 PCM 逐 sample allclose(WAV vs FLAC
  重新解壓)」——寫一個 5-10 個 clip 嘅 smoke test:同一段 clip 分別用
  `subtype="PCM_16"`(WAV)同 `format="FLAC"` 寫,兩者經 `decode()` 讀返嚟,斷言完全
  一致(FLAC 係 lossless,理論上 100% bit-exact,唔應該有 |Δ|>0 嘅情況)。
- **Rollout gate**:先淨係對 51.6h 嘅「246 個從未 segment 過」raw file(今日調查發現,
  memory `[[canto-corpus-rearchitecture]]`)行 `segment.vad_cut`,人手核對幾個新
  `.flac` clip 播放正常、catalog row 寫得啱,先擴大到日常新 raw ingest。
- **完成標準**:新 raw 由呢日起 100% 產出 `.flac` segment;現有 843G legacy `.wav`
  完全唔郁。

#### P5-B. 存量 raw → FLAC 轉碼(第二步,前提係 P5-A 已經喺跑緊)—— **改用 FLAC,owner 已確認(2026-07-04)**

> **✅ 已確認**:owner 睇完下面嘅 opus vs FLAC 取捨表之後,揀咗 FLAC(接受多用 ~420GB
> 換零額外 lossy 損耗,同 segments 決定一致)。跟住 owner 問「咁將來新 download 係咪都
> 應該轉 FLAC 嚟壓縮?」——**答案係唔應該**,已經用 `ffmpeg` 實測:將原生 lossy source
> (YouTube opus / RTHK AAC)decode 完再 encode 做 FLAC,檔案會**大 2.1-3.5×**(YouTube
> 樣本 33.4MB opus → 70.7MB FLAC;RTHK 樣本 5.2MB AAC → 18.1MB FLAC)——lossy codec 本身
> 已經捨棄咗人耳聽唔到嘅資訊嚟換取遠高於任何 lossless 格式嘅壓縮率,將已經 lossy 嘅內容
> 再包一層 FLAC 淨係加大檔案、冇任何 fidelity 著數。**將來新 download 應該保持而家
> §7.1 政策第 1 點嘅做法:直接保留原生 bestaudio container(opus/AAC/mp3),唔轉 WAV
> 都唔轉 FLAC。**
>
> 但查落去發現呢個政策由 2026-07-02 拍板到而家**從未真正落實**:`sources/youtube_channels.yaml`
> 第 2771 行依然寫死 `audio_format: "wav"`,`pipeline/nodes/` 入面亦冇 `ingest.download`
> node(淨係得 `ingest_probe.py` 睇 metadata,冇實際 download-and-store 邏輯)——即係話
> 新 download 一直沿用返舊嘅 `scripts/02_download.py`,每日仍然浪費咁轉緊 WAV。
>
> **✅ 已落實(2026-07-04,同日跟進)**:
> - 起咗 `pipeline/nodes/ingest_download.py`(`ingest.download` node,`pipe run
>   ingest.download --source [rthk|youtube|podcast|all] [--dry-run] [--limit N]`),
>   RTHK/podcast RSS enclosure 同 YouTube yt-dlp bestaudio 都**唔再做任何 ffmpeg
>   轉碼**——冇 `--extract-audio`/`--audio-format`/postprocessor,原生 container
>   (opus/webm、AAC/m4a、mp3)照收。`duration_sec`/`sample_rate` 有意留空,交返俾
>   已經存在嘅 `ingest.probe`(ffprobe)事後補,唔喺呢個 node 重複一次解碼邏輯。
> - 呢個 node 跟其他 P3+ node 嘅慣例,**直接寫 `raw_files` catalog**(用現有嘅
>   `raw_id` 命名法:RSS = md5(audio_url)[:8],YouTube = 11 字 video id),
>   唔再借 `metadata/downloaded.jsonl` 中轉。
> - `sources/youtube_channels.yaml`(第 2771 行)、`sources/podcast_sources.yaml` 嘅
>   `download_config` 已經拆走 `audio_format`/`postprocessor_args`/`convert_to_wav`
>   呢啲誤導字段(其實舊 script 都從來冇讀過呢啲字段——`load_yaml_entries()` 攞完
>   config dict 之後即刻掉咗唔用,轉碼行為全部係 Python code 入面寫死,唔係 yaml 決定
>   嘅),加返清楚註明而家政策嘅 comment。
> - `pipeline/cli.py` 加咗 `cmd_run_ingest_download` + `ingest.download` subparser。
> - **順手執到一個獨立 bug**:測試時發現 `sources/podcast_sources.yaml` 第 1382-1481
>   行(「Health / Medical Podcasts」起之後 9 個 entries)縮排跌返做 0-space,跳出咗
>   `sources:` 呢個 list 嘅巢狀結構,令成個檔案 YAML parse 直接 raise
>   `yaml.YAMLError`——而 `load_yaml_entries()`/`_load_entries()` 兩邊都係
>   `except yaml.YAMLError: return []` 靜默吞錯,結果係**成個 podcast_sources.yaml
>   102 個 entries,由呢個 bug 引入嗰日起就已經對 downloader 完全隱形**,同呢次
>   storage format 政策冇關,但阻住咗 `ingest.download` 測試先發現。已經改返 2-space
>   縮排,`yaml.safe_load` 確認 102 個 entries 全部讀到。
> - 舊 `scripts/02_download.py` 保留低做歷史參考(同 03/04/06/07/08/09 一樣嘅做法),
>   唔刪除、亦冇再改佢嘅轉碼行為。
>
> 詳見 `DECISIONS.md` 2026-07-04「Raw backlog format: FLAC confirmed by owner;
> new-download policy clarified」同「Storage format policy FINALIZED」兩條目。

> ~~⚠️ 呢個唔係新決定,係重新質疑一個已經拍咗板嘅決定~~(已由 owner 確認,見上)——
> §7.1「Raw → opus(owner 已拍板
> 壓縮路線)」本身已經係 2026-07-02 owner 明確簽收嘅決定,連「第二代 lossy」呢個風險都已經
> 喺 §7.1 嘅「誠實 caveat」段度講明咗、並且已經有相應緩解(政策第 1 點:**新 download 由
> 2026-07-02 起已經改做保留原生 bestaudio container,唔再轉 WAV**,即係話呢個 opus-vs-FLAC
> 嘅取捨其實只影響「現存 1.6T WAV backlog」呢一批歷史資料,對將來新落嘅資料完全冇影響,
> 兩種方案都一樣)。2026-07-04 因為 owner 提出「點解仲用緊 opus」呢條問題,先重新攞出嚟計
> 過——**下面係雙方論點,最終選邊由 owner 決定,唔係我單方面判 FLAC 贏**:
>
> | | Opus(§7.1 原方案,已拍板) | FLAC(重新提出) |
> |---|---|---|
> | 現存 1.6T backlog 轉碼後大小 | ~150G(估計) | ~570G(2026-07-04 實測 5 個真實檔案,ratio≈35%) |
> | 對「未來重新 segment 現存 raw」嘅影響 | 引入第二代 lossy(§7.1 已承認,128kbps 下判斷「極微」) | 零額外損耗(lossless) |
> | 對「未來新 download」嘅影響 | 冇影響(新 download 已經唔轉 WAV,直接保留原生格式) | 同左,冇影響 |
> | 同 segments 嘅 FLAC 決定一唔一致 | 唔一致(raw lossy,segments lossless) | 一致(兩層都 lossless) |
> | Drive2 free space(5-10× raw capacity model) | 25-50kh raw ≈ 1.4-2.9T,一隻碟就夠 | 需要重新計(FLAC 大隻好多,可能要兩隻碟) |
>
> `ffprobe` 核實咗一個 §7.1 已經知道但呢度值得重申嘅事實:YouTube 源頭本身已經係 **Opus**
> (48kHz 立體聲),RTHK 源頭係 **AAC 32kHz/64kbps**——依家嘅 raw WAV 本來就唔係「第一代
> lossless master」,呢點 §7.1 原文都已經寫明(「WAV 化冇增加任何 information」)。
>
> 下面嘅 schema/node 設計已經係 owner 確認咗嘅 FLAC 路線,§7.1 原本嘅 opus 章節(第
> 2-4 點、`raw_opus` table)保留做歷史記錄,唔再係實作依據。

- **(FLAC 分支)Schema**:`pipeline/catalog/schema.sql` 第 346 行已經有伏筆註解等呢個 table(註解入面
  提到嘅「opus transcoding」字眼需要一併更新做「FLAC transcoding」)——新增:
  ```sql
  CREATE TABLE IF NOT EXISTS raw_flac (
      raw_id        TEXT PRIMARY KEY,
      flac_path     TEXT,
      duration_sec  DOUBLE,     -- 轉碼後量出嚟,同 raw_files.duration_sec 對比做驗證
      verified      BOOLEAN,    -- decode 後 PCM bit-exact 核對過先 true(lossless,唔使人耳聽)
      wav_deleted_at TIMESTAMP, -- NULL = 原 wav 仲喺度;non-null = 已刪
      transcoded_at TIMESTAMP
  );
  ```
- **新 node**:`pipeline/nodes/raw_flac.py`,跟現有 node 慣例(`discover_*` + per-item
  worker function):
  - `discover_flac_transcode(conn)`:SELECT raw_id FROM raw_files WHERE raw_id IN
    (SELECT raw_id FROM raw_segments) AND raw_id NOT IN (SELECT raw_id FROM raw_flac)
    —— **關鍵**:`raw_id IN raw_segments` 呢個 join 就係防止今日發現嗰 246 個未
    segment 過嘅 raw file 被搶先轉碼(先切好晒先轉,避免中途出錯要重新 decode 一次
    源頭)。呢個 exclusion 直接借用現有 `raw_segments` table,唔使新加 tracking 邏輯。
  - `_flac_transcode_one(raw_id, wav_path)`:用 `soundfile`(libsndfile 原生支援,
    同 P5-A 一樣,唔使 shell 出 ffmpeg)`sf.write(flac_path, data, sr, format="FLAC")`;
    轉碼完**decode 兩邊(原 wav vs 新 flac)做 PCM 逐 sample 核對**(FLAC 係
    lossless,理論上應該 100% bit-exact,同 P5-A 個 smoke test 同一手法)——呢個比
    opus 嗰套「duration 核對 + 抽樣人耳」簡單同可靠好多,因為根本冇 perceptual loss
    需要人耳判斷;寫 `raw_flac` 行。
  - 分批(~100GB/batch,跟 §7.1 rebalance 慣例),每 batch 完:
    1. PCM bit-exact 核對 100% pass 先可以將 `verified` 揈做 true(唔需要人耳抽聽,
       lossless 冇 perceptual artifact 呢回事)
    2. **owner 簽收呢個 batch**(依然要——刪原 WAV 始終係不可逆操作,即使轉碼本身
       lossless,都要人手confirm 先執行實際刪除)
    3. 淨係 `verified=true` 嘅 raw_id 先可以刪原 `.wav`(寫 `wav_deleted_at`)
- **完成標準**:全部已 segment 嘅 raw 都有 verified FLAC 版本,原 WAV 已刪,Drive2 釋放
  ~1.03TiB(2026-07-04 用 5 個真實 raw file 實測 FLAC ratio ≈35% 算出嚟,唔係估計值——
  見 §9「儲存容量調查」段修正)。

#### P5-C. 三碟 sharding rebalance(最後做,前提係 P5-A/B 都已經完成)

- **`config/storage_layout.yaml`** 第 62-67 行 flip:
  ```yaml
  sharding:
    enabled: true          # 而家 false
    n_shards: 3            # 而家 2(Drive2 raw + Drive4 seg)
    shard_roots:
      - /mnt/Drive2/canto/segments   # shard 0 —— raw FLAC 化後讓出嘅空間
      - /mnt/Drive3/canto/segments   # shard 1 —— 而家完全空,即刻可用
      - /mnt/Drive4/canto/segments   # shard 2 —— 現有 843G segments 做起點
  raw_root: /mnt/Drive2/canto-corpus/data/raw   # FLAC 化後(實測 ratio ~35%)~570GiB
  ```
  (呢個同 §7.2 原本草擬嘅 shard map 一致,依家先實際落地。)
- **Rebalance script**(新增,例如 `scripts/rebalance_shards.py`,或者做返一個
  `pipe rebalance` CLI 子命令):對每個 segment,`hash(raw_id) % 3` 決定目標 shard;
  同現存路徑所在 shard 唔同就:
  1. rsync copy 去新 shard
  2. name+size(或 checksum)全量核對
  3. `segments.audio_path` catalog 路徑 transactional UPDATE(單一 SQL transaction,
     唔會有「刪咗源頭但 catalog 仲指去舊路徑」嘅中間態)
  4. 核對完先刪源頭
  分批(~100G/batch)、監察 `dmesg`(§3 非破壞鐵律 + 過往 crash 前科,見 §11 風險表)。
  由於 hash key 係 `raw_id`(唔係檔案路徑或副檔名),FLAC(新)同 WAV(舊)混合都會
  一致咁分佈,唔會因為格式唔同而傾側落某一碟。
- **完成標準**:`storage_layout.yaml` 定案,`pipe catalog verify --full` 全部 path
  exists,§3 step 9 正式關閉。

**排序理由(A → B → C,唔可以打亂)**:
1. A 要盡快做,因為依家每一日新增嘅 segment 都仲係用緊舊 WAV 格式寫入,拖得就拖唔到
   「新 segments = FLAC」呢個已經拍板嘅決定盡快生效。
2. B 要喺 A 之後,理由淨係邏輯上獨立(A 改 segments,B 改 raw),但 B 必須排喺 C
   之前,因為 B 釋放嘅 Drive2 空間正正係 C 嘅 shard 0 要用嘅位——冚唪唥掉轉次序會
   令 C 冇位執行。
3. C 排最尾,因為佢係全部三個當中最大型、最唔可逆嘅資料搬遷(§11 風險表「重 I/O
   觸發系統不穩」);等 A/B 兩個編碼決定都塵埃落定先做一次搬,好過分兩次搬(FLAC
   轉碼完再搬一次 vs 一次過搬)。

**Gate(全部三步共通)**:§3 非破壞鐵律逐條(先核對後刪、絕不 blind overwrite);每
batch 核對先可以進行下一步;catalog path/verified 狀態更新一律 transactional;三步
完成後結尾跑一次 `pipe catalog verify --full`。

#### ✅ P5-A + P5-B 執行結果(2026-07-05)

Owner 揀咗今個 phase 淨係做 P5-A + P5-B(C 留待下個 phase,full-rebalance 原設計)。

**P5-A(segment.vad_cut → FLAC)—— 完全完成、關閉**:
- `_vad_cut_one()` 改咗輸出 `.flac`(`format="FLAC"`,`subtype="PCM_16"`,同計劃一致)。
- 執行前發現同修正三個 pre-existing / 新暴露嘅 bug(唔關 FLAC 輸出本身事,但阻住咗
  backlog 順利跑完):
  1. `DiarizeWorker.load_model()` 之前淨係讀 `HUGGING_FACE_HUB_TOKEN` 環境變數,冇設就
     靜默行 VAD-only——但呢部機器實際上有 working 嘅 `huggingface-cli login` cache。
     改成默認交俾 huggingface_hub 自己解析 cache token(同 `label_suite.py` 一致做法),
     env var 有設先覆蓋。
  2. `pipeline/orchestrator/worker.py` 嘅 JSONL worker protocol 冇設 `limit=`,
     asyncio 預設 64KB 行緩衝上限——一個 1,106-turn(~100KB)嘅真實 diarization 結果
     觸發 `LimitOverrunError`,拖冧成個 worker 嘅 stdout stream(連鎖累到同一 batch
     嘅 244 個檔案全部報錯)。修正:`spawn_worker()` 加 `limit=32MiB`。
  3. `pregate.snr` 嘅 DNSMOS 分支用 `torchaudio.transforms.Resample` 48k→16k 之後,
     source 貼近滿刻度嘅音頻會令 sinc resampler overshoot 出 [-1,1] 範圍,觸發
     `speechmos.dnsmos.run()` 嘅嚴格檢查報錯(fail-open 靜默漏咗 409 個 segment 嘅
     DNSMOS 評分)。修正:resample 完 `np.clip(wav16, -1.0, 1.0)`。
- 全 backlog 執行結果:`segment.diarize`(10,177 個未 diarize 嘅 raw file,含 249 個
  backfill 之後先出現嘅真正 miss)→ 9,933 legacy-reused(sidecar hit,零 GPU)+ 244
  真 GPU pyannote diarization,0 error;`segment.vad_cut`(249 個 raw_id)→ 11,380 個
  新 `.flac` segment,4 個 raw_id 合理咁產出 0 clip(VAD window 全部超出 3-20s 範圍);
  `pregate.snr`(11,266 個新 segment,含 409 個 DNSMOS-clip bug 重跑)→ 2,258 pass。
- P5-A rollout gate 驗證:cross-correlation 對比新 FLAC segment 同源頭 raw WAV,喺
  offset ≈203.9s 搵到近乎完全一致(誤差 ~1e-4,浮點精度級別)——確認 lossless 抽取
  正確、路徑/catalog metadata 正確(`sample_rate=48000`、`raw_id` 有填、`.flac` 副檔名)。
- 129/129 tests pass(新增 `tests/test_audio_bus.py` WAV/FLAC bit-exact round-trip +
  ffmpeg fallback 測試)。

**P5-B(raw WAV → FLAC 轉碼)—— 完全完成、關閉(2026-07-05 續)**:
- 新 `pipeline/nodes/raw_flac.py` + `raw_flac` schema table + `pipe run raw.flac`
  CLI。Discovery SQL 喺原計劃基礎上加咗兩個修正(執行前查證發現):(1) 明確排除
  native container(`wav_path LIKE '%.wav'`)——ingest.download 落嘅 webm/m4a 永遠
  唔轉 FLAC,轉咗只會脹大 2.1-3.5×;(2) 加入 `lang_screen` reject 嘅 195 個 raw_id
  ——呢啲永遠唔會經 `segment.diarize` 入到 `raw_segments`,原計劃嘅 join 會令佢哋永遠
  轉唔到碼、WAV 空間永遠釋放唔到。
- 執行時發現一個獨立 schema.sql bug:`init_schema()` 嘅 naive split-on-`;` 淨係識剝走
  「成行都係 `--` 開頭」嘅註解,一個同 code 同行嘅 inline 註解入面帶 `;` 會拆亂 DDL——
  改成同碼庫其他地方一致嘅寫法(續行註解獨立成行)。
- Gate test(`--limit 20`)全部 verified,人手核對 FLAC 檔可以正常 decode、壓縮比
  ≈33.7%(貼近 §9 實測嘅 ~35% 估計)。
- **Batch 1(679 個 raw file,~100GB)+ Batch 2(658 個,~100GB)已完成轉碼+verify**:
  合共 1,757 個 raw_id 全部 `verified=true`,0 failed。Drive2 free space 244G→149G
  (FLAC 同原 WAV 並存,未刪除任何嘢)。
- **`--delete-verified` 簽收記錄**:batch 1+2(1,757 個)嘅刪除問咗 owner 兩次
  (AskUserQuestion)都冇回應(離開咗鍵盤)——跟返 owner 本身定嘅政策(「頭 1-2
  batch 人手簽」),當時停低未刪。Owner 返嚟後喺同一個 session 內明確確認執行
  (「1. 係咪確認執行 `--delete-verified`...2. 繼續轉碼餘下...」),於是:
  1. 即刻執行 batch 1+2 嘅 `--delete-verified`:1,757 個 WAV 全部刪除,0 error。
  2. Owner 嘅確認同時滿足咗「頭 1-2 batch 人手簽,之後自動」政策嘅門檻,於是寫咗
     一個 loop script(`run_raw_flac_remaining.sh`):`--batch-gb 100` 轉碼
     (內建逐 block bit-exact verify)→ `--delete-verified` → check 剩餘 → 重複,
     直至 discovery 返 0 為止,全程用 `nohup ... & disown` 背景執行。
  3. **剩餘 9,153 個 raw_id(batch 3–16,14 個 auto batch)全部轉碼+verify+delete
     完成,0 failed。連同 batch 1+2,合共 10,910/10,910(100%)raw file 已由 WAV
     轉為 FLAC,原 WAV master 全部安全刪除。**
- 磁碟成果:Drive2 free space 由 backlog 開始前嘅 149G,逐 batch 遞增釋放,完成後
  達 **1.3T free**(used 由 1.7T 跌到 589G)——實測壓縮比 ≈33-35%,同 gate test
  估算一致。
- 最終 catalog 核實(`raw_flac` 表):10,910 verified=true、10,910 wav_deleted_at
  非空、0 failed;`raw_files.wav_path` 100% 指向 `.flac`,0 殘留 `.wav` reference。
  Disk 上發現 10 個 catalog 外殘留 `.wav`(P5 開始前已存在嘅雜物,不受呢次轉碼
  影響,唔屬於呢個 node 嘅職責範圍)。
- 執行過程中兩次背景長跑 job 被 harness 意外 kill(非 crash,catalog 冇損壞,
  idempotent 可安全 resume)——改用 `nohup ... & disown` 寫落 log file 嘅方式代替
  auto-background/`run_in_background`,穩定行完全程,之後全部 16 個 batch(含
  auto loop 嘅 14 個)都用呢個模式,冇再被 kill 過。
- 129/129 tests pass(新增 `tests/test_raw_flac_node.py`,9 個測試,包括獨立 scratch
  DuckDB 嘅 discovery-SQL eligibility 驗證,冚 native-container 排除、reject-raw 納入、
  已轉碼排除三種情況)。
- **P5-B 正式關閉**——冇再有 eligible raw WAV 待轉碼。下一步係 P5-C(三碟 sharding),
  留返俾 owner 決定幾時開始。

**`pipe catalog verify --full`**:呢個 flag 實際上未存在(CLI 淨係得無參數嘅
`pipe catalog verify`)——plan doc 呢句原本寫嘅係前瞻性描述,唔係已有工具。跑咗現有
嘅 `pipe catalog verify`:17 項有 5 項顯示 FAIL,但全部都係 P0 milestone(2026-07-02)
凍結嘅 exact-match baseline(例如 `segments` expected=455299,而家已經合法增長到
466710)——呢個 script 本身冚喺舊快照冇跟住更新,唔係新問題;日常真正把關嘅
`tests/test_catalog.py`(floor-based monotonic-growth 斷言)129/129 全過。
`path_exists[segments]`/`path_exists[raw_files]` 兩項都 PASS(2000 個 sample 全部
存在)。

**殘留雜物(已報告,未自動刪)**:raw 目錄有 3 個 `.part`(中斷 download 殘留,
`20251117_tvb_無綫新聞_PX51uDj-vdE.webm.part` 等)+ 13 個 webm/1 個 m4a/1 個 mp3
(catalog 外、早於 native-container 政策落實嘅殘留,對 pipeline 冇影響)——留俾
owner 決定去留。

**Owner 返嚟後續(2026-07-05 同日完成)**:owner 明確確認咗 batch 1+2 嘅
`--delete-verified` 同繼續轉碼餘下 9,153 個 raw_id 兩個步驟,兩者已喺同一 session
內全部執行完畢(見上面「完全完成、關閉」段落)。P5-A + P5-B **正式全部關閉**。

#### ✅ P5-C 執行結果 + legacy-orphan recovery detour(2026-07-06)

**P5-C(三碟 sharding rebalance)—— 完全完成、關閉**:
- `config/storage_layout.yaml` 已 flip:`sharding.enabled: true`、`n_shards: 3`
  (`/mnt/Drive2|3|4/canto/segments`)。`config/storage_layout.py` 新增
  `shard_index()`/`shard_root()`(md5-based,跨 process 穩定),hash key = `coalesce(raw_id, id)`
  ——同原計劃一致,唔受 FLAC/WAV 混合格式影響分佈。
- `segment.vad_cut` 已改用 `shard_root()` 揀輸出目錄,每個新 segment 一出世就落喺啱嘅
  shard,之後永遠唔使再 rebalance 一次。
- 新 node `pipeline/nodes/rebalance.py`(`pipe run rebalance.segments`,同 P5-B 一樣
  兩段式:copy+byte-verify → 獨立 `--delete-verified`)。新 schema table
  `segment_shard_migrations`。
- Gate test 期間搵到兩個 bug 並修正:(1) `_copy_one()` 嘅 `mkdir` 之前喺 try/except
  之外,一個 filesystem error 會累到成 batch 一齊 crash,而唔係記做單一 item 失敗;
  (2) `/mnt/Drive3` 掛載後屬 `root:root`(唔可寫),owner 執行
  `sudo chown typangaa:typangaa /mnt/Drive3` 修好。
- **執行結果**:156,162 個 segment 本身已喺啱嘅 shard(純記錄,零 I/O)+ 310,548 個
  複製並 byte-verify(0 failed)+ `--delete-verified` 刪晒舊 Drive4 原檔(0 error)。
  全部 466,710 個 catalog segment 現已正確三碟分佈。磁碟:Drive2 589G→697G、
  Drive3 0→110G、Drive4 846G→628G。
- `pipe catalog verify --full` 呢個 flag 從未存在(CLI 淨係得無參數嘅
  `pipe catalog verify`);跑咗現有版本,17 項有 5 項 FAIL,但全部都係 P0
  milestone(2026-07-02)凍結嘅 exact-match baseline 過時(例如 `segments`
  expected=455299,而家合法增長到 466710),唔係新問題——日常真正把關嘅
  `tests/test_catalog.py`(floor-based monotonic-growth 斷言)全過。

**Detour:legacy-orphan recovery(rebalance 過程中意外發現,同日處理完)**:
- 調查「點解 Drive2/3 得返 ~110G 咁少」期間,發現 Drive4 嘅 `segments/` 目錄實際存
  ~730,885 個 WAV 檔完全唔喺 catalog 度——舊 pipeline(`scripts/03_segment.py`,
  2026-06-09~20 行嗰陣)將每個 VAD candidate 都切落碟,但淨係捱過舊 filter stage 嘅
  先入到 `manifest.jsonl`(P0 catalog import 嘅唯一來源),reject 咗嘅從未清理過。
  跨三個 source 抽樣(13%-55% pass rate)+ 每個 orphan 自己殘留嘅 sidecar
  (`.pregate.json`、`.transcript.json`)確認呢個結論,亦令逐檔精細判斷成為可能
  (而唔係盲目 keep/delete 全部)。
- 新 node `pipeline/nodes/recover_orphans.py`(`pipe run recover.orphans`)+ 新
  `orphan_segments` table:靠 sidecar 分類每個 orphan——`pregate_pass` 或
  `transcript_high_agreement`(≥0.80 agreement)→ RECOVER(back填 `segments` +
  `asr_results` + `asr_agreement`,`text_verified=false`,同一個新切嘅 segment
  一樣走真正嘅 `filter.text`/`filter.acoustic`/`filter.decide`/`tier.assign` 去做
  實際 accept/reject/tier 判斷,呢個 node 本身唔判斷);其餘 → `orphan_segments.status
  = 'pending_delete'`,**唔郁任何檔案**,純粹排隊等日後獨立簽收嘅清理 pass。
  (呢個 node 用 0.80 門檻,比 `tier.py` 實際嘅 `SILVER_AGREE_MIN = 0.65` 嚴——
  日後如果想撈多啲,一次 0.65 嘅 looser rescan 隨時得。)
- **執行結果**(`nohup` 背景全量跑,無 `--limit`):730,824 scanned、**151,981
  recovered**、578,843 queued pending_delete、0 errors,耗時 5,114s
  (`run_id=recover_orphans_c366e9db3a86`)。
- 157/157 tests pass(新增 `tests/test_storage_layout.py`、`tests/test_rebalance_node.py`、
  `tests/test_recover_orphans_node.py`;`tests/test_segment_node.py` 3 個測試因
  sharding 而家預設開咗,要 monkeypatch `_SHARDING` 關返先過)。
- **殘留雜物**(已報告,未自動刪):raw 目錄 3 個 `.part`(中斷 download 殘留)+
  13 個 webm/1 個 m4a/1 個 mp3(catalog 外、早於 native-container 政策嘅殘留)——
  留俾 owner 決定去留,對 pipeline 冇影響。

**Follow-up backlog(2026-07-07,進行中)**:`recover.orphans` backfill 咗
audio+ASR text,但唔會自己跑 filter/tier——跟返 PROGRESS.md 原定「Next」清單,
`filter.text` 已對 151,981 個 recovered segment 行完,現正用
`pipe run-many asr.transcribe --models canto_ft,whisper_v3 --devices cuda:0,cuda:1
-- filter.acoustic --workers 12 --threads 2`(09:47 起跑,~13.8/s,ETA 同日
下午)一次過補齊缺嘅 ASR candidate(canto_ft 45,482 / whisper_v3 11,411)同
`filters_acoustic`(352,813 條)。跑完之後仲要 `filter.decide` → `tier.assign` →
`speaker.embed`/`speaker.cluster` → `manifest.build`/`manifest.export`,先算真正
將呢批 151,981 個 recovered segment 併入可用嘅 manifest。

**下一步(P5 全套已完全關閉,P5-C 唔再係「留返下個 phase」)**:
1. 等 2026-07-07 嘅 asr.transcribe+filter.acoustic backlog job 行完。
2. `filter.decide` → `tier.assign` → `speaker.embed/cluster` → `manifest.build/export`。
3. `orphan_segments.pending_delete` 隊列(578,843 檔,~440GB+)——owner sign-off 未批,
   空間唔緊張,故意擺喺度做 inventory,唔會自動執行刪除。
4. P6(下面)。

### ✅ ASR 擴展:qwen3_asr(2026-07-07)+ sense_voice(2026-07-08)+ orchestrator conn 注入全部完成

**qwen3_asr(2026-07-07)**:第 3 個 ASR model 加入(`pipeline/nodes/asr.py`),
transformers backend(`qwen_asr` package,非 ctranslate2),native `language="Cantonese"`,
唔受 Whisper `yue` decoder-collapse 影響。雙 GPU 切分跑全量 backlog(618,695 segments),
中途搵到並修正一個真 bug:discovery 淨係 keyed 落 model_key、唔理 device,如果同一個
model 分兩張卡跑會令兩張卡各自攞成個 backlog 做重複運算——加咗
`shard_rows_round_robin()` 解決(round-robin,唔係 contiguous split,因為 discovery SQL
`ORDER BY duration_sec` 遞增,contiguous split 會令一張卡淨係攞短 clip 一張淨係攞長
clip)。`asr.agreement` 由 2-model self-join 改寫做 N-way(`GROUP BY id` + `list()` 聚合),
用 `model_count` 欄做 provenance-style re-trigger(legacy P0 rows 嘅 `model_count IS NULL`
永遠唔會重觸發,新 rows 有第 3/4 個 model 到會自動重算)。全量跑完:618,695/618,695,
0 errors,~4h(56.8/s → 36.3/s,duration-ascending queue 令後段變慢)。

**sense_voice(2026-07-08)**:第 4 個 ASR model,funasr backend(`iic/SenseVoiceSmall`,
ModelScope license,non-OSI 但商用容許),CTC non-autoregressive,native `language="yue"`
(呢個 model 唔係 Whisper 架構,`yue` 用喺度安全,唔犯 hard constraint #7 —— 已喺
`tests/test_asr_node.py` 明確 scope 好,見下面「本 session 修復」)。~105× RTF(RTX
4090 實測)。輸出簡體轉繁體(OpenCC s2hk),emotion/audio-event inline tag 抽取出嚟存落
`asr_results.metadata`(JSON 欄,`text` 淨係存乾淨文字)。全量跑完(分兩輪,中途一次
worker-ready JSON decode 錯誤已自動 recover):618,695/618,695,0 errors。跟住
`asr.agreement`(4-way,618,695/618,695,97.9/s,6,322s)。

**Orchestrator `conn=` 注入 —— 23/23 全部完成**(見 `docs/ORCHESTRATOR_PLAN.md` 詳細
inventory):所有會自己 `connect()` 嘅 node function 而家都收 `conn=` kwarg,可以喺
`pipe run-many` 底下同其他 node 共用一個 DuckDB connection 並行跑。

**本 session 修復(2026-07-09,review 後即場做)**:
1. `test_model_field_never_yue` 改名做 `test_model_field_never_yue_for_faster_whisper`,
   scope 落 `backend=="faster_whisper"` 先 check —— 呢個 test 寫喺 qwen3_asr 之前,
   對成個 `ASR_MODELS` 做 blanket check,sense_voice 加入之後即刻紅咗(佢合法用
   `lang="yue"`)。加咗一個反向 guard test 防止日後改返做 blanket ban。
2. `pyproject.toml` 補齊 `funasr>=1.3.14`、`modelscope>=1.38.1`、
   `opencc-python-reimplemented>=0.1.7`(之前淨係喺 code comment 度提,冇正式入
   dependencies——同 `qwen-asr` 嗰種做法唔一致)。版本數字對返 `.venv` 實際裝緊嘅。
   冇碰 `uv.lock`(同項目慣例一致,GPU-adjacent 依賴淨用 `uv pip install`)。
3. `asr_results` 加咗 `metadata JSON` 欄(`schema.sql` CREATE TABLE + `ALTER TABLE ...
   ADD COLUMN IF NOT EXISTS`,跟 `raw_id` 嗰條已有嘅 pattern,`init_schema()` 每次
   `connect()` 都會補落現有 production catalog)。之前 `SenseVoiceWorker` 產生嘅
   emotion/audio_event tag 一直被 `run_asr_transcribe()` 靜靜雞丟棄(out_rows 淨揀
   `id/model/text/confidence` 四個 key)——而家 `r.get("metadata")` 會帶埋過去。
   (過程中順手搵到並修正一個新 bug:schema.sql 個 inline comment 本身有分號,撞正
   `init_schema()` 天真嘅 `split(";")` parser,搞到成個 DDL parse 壞晒。)
4. 補齊 `SenseVoiceWorker` 單元測試(tag 解析、emotion/event 抽取、OpenCC 轉換、
   confidence 預設、batch 順序、funasr exception → placeholder rows)——同
   `Qwen3ASRWorker` 嗰種覆蓋深度睇齊。`tests/test_asr_node.py` 而家 38 個 test 全過。

**已知限制(記錄低,未修)**:`filter.text`/`filter.decide`/`tier.assign` 嘅 discovery
係 bare row-existence anti-join(`filter.text`)或 `provenance='tier_assign'` scoped
anti-join(`tier.assign`,只擋自己之前 tier 過嘅 row,唔擋 legacy P0 rows)——都冇好似
`asr_agreement.model_count` 咁做「內容變咗就重觸發」。即係話已經 `filters_text`/已經
俾 `tier.assign` tier 過嘅 segment,即使佢哋嘅 `asr_agreement.best_text` 因為 sense_voice
加入而變咗(可能質素更好),都唔會自動重新評估——淨係新發現(未 filter/未 tier)嘅
segment 先食到 4-model agreement 嘅提升。如果想全部 refresh,要幫 `filters_text`/
`tiers` 都加一個類似 `model_count` 嘅 provenance/count 重觸發欄。

**Phase A(2026-07-09 下午,進行中)**:`speaker.embed`(dual-GPU)→ `speaker.cluster`
→ `tier.assign` → `manifest.build/export`,補返 163,376 個新 segment(recovered orphans
+ 新 ingest)缺嘅 speaker embedding/tier/manifest 資料(呢批之前得 filter+g2p,冇
speaker/tier/manifest)。

### P6 — Scale readiness(1 session)
- 5-10× ingestion dry-run(youtube_channels.yaml 新 11 條 diversity channel 做試點)
- soak test:orchestrator 連續行 24h+(download→segment→asr→label 全鏈 item-level 並行)
- capacity 儀表:`pipe status` 加 per-drive 用量投影(nvme 計埋 canto-tts 長住,§10 Q4)
- **Gate**:soak 零 leak(RSS/fd 平穩);投影確認後落實 §10 Q1(傾向 FLAC master ——
  屆時改 constraint #6 字面 + owner 簽名,先開始轉)

---

