# Pipeline Comprehensive Review + Cleanup Plan — 2026-07-13

> **Status**: **PLAN ONLY**(cleanup 部份未執行)。文檔同步部份(§6)已於本 session 完成。
>
> Owner 確認咗四個前提(AskUserQuestion,2026-07-13):
> ① 新開呢份 2026-07-13 doc,唔延長 07-11 嗰份(每份 review 一個完整 snapshot);
> ② cleanup **plan-only** — owner review 完逐 phase 指示先執行;
> ③ 「unused files」範圍**包埋退役 ASR model weights**(repo 內 `data/ct2_models/` +
>   machine-level `~/.cache/huggingface`);
> ④ 外部調研用 **targeted 補充**(重用 07-11 doc §5 嘅 comprehensive 對照,今次淨係
>   針對 2-model agreement 缺口做窄範圍調研 — 見 §5)。
>
> **執行任何 cleanup 之前**:T15 `asr.transcribe` 而家跑緊(揸住 DuckDB writer)——
> 任何要碰 catalog 嘅動作(§3 Phase C 全部)必須等佢完;純磁碟動作(Phase A)同
> git 動作(Phase B)唔受影響。
>
> 上一份:`docs/archive/PIPELINE_REVIEW_2026-07-11.md`(EXECUTED,16 issues 中 11 RESOLVED)。
> 本 doc 嘅 issue 編號由 **#17** 起,同上份唔重疊,方便交叉引用。

---

## §1 現況快照(2026-07-13 ~17:00)

| 項目 | 數值 |
|---|---|
| Git tip | `e299dd8`(master,**領先 origin 3 commits 未 push**:`f511918`/`78d66ef`/`e299dd8`,07-11 round-2 嘅三個)|
| Working tree | **12 modified + 2 untracked 未 commit**(見 Issue #18)|
| Milestone | P0–P5 done;P6 未開始(不變)|
| Catalog | `corpus.duckdb` **3.7GB**(07-11 時 2.3GB — T15 admission + embedding column 令佢增長);`segments` 1,241,610(662,721 原有 + 578,889 T15 re-admitted)|
| ASR 活躍 model | **2 個**:`qwen3_asr` + `sense_voice`。`canto_ft` 今日(2026-07-13)退役,`whisper_v3` 2026-07-10 退役 — 兩個都係「慢/唔準」同一 profile,DECISIONS.md 有完整 evidence |
| T15 進度 | `asr.transcribe`(qwen3_asr,dual-GPU,`--batch 64`)**125,440/568,977 @ 42.1/s**,ETA ~2.9h;之後 sense_voice pass(~105× RTF,預計 <1h)|
| T15 之後未做 | `asr.agreement` → `filter.*` → `g2p` → `tier.assign` → `speaker.cluster` whole-source recompute → `manifest.export` → `report.build` 全部排隊緊 |
| Tier 分佈 | 07-11 嘅數(gold=58 / auto_gold≈72k / silver≈236k / bronze≈151k)仍然係現行 catalog 狀態,但 T15 落地後會大幅變動 — 呢刻唔好攞嚟做任何決策 |
| 預設 manifest | **STALE**(2026-07-09,已隔兩次 backfill + T15)— 維持 07-11 Issue #2 嘅「等一次過 export」策略 |
| Tests | 304 passed + **2 個已記錄、預期中嘅 live-catalog failures**(canto_ft-only rows 未被 2 active model 覆蓋 → 會自愈;corpus totals 超出 baseline tolerance → 等 T15 落地後 deliberate re-baseline)|
| QA 狀態 | 3 個 300-segment pilot batch(auto_gold/silver/bronze)**仍然全部未 review**(T1,owner 人手動作)|
| Disk | `/` 917G free;Drive2 881G;Drive3 1.6T;Drive4 1.2T — 冇壓力 |
| 進程 | `calibrate serve`(PID 971786)live;T15 script(PID 1617642)live |

**架構健康度**:同 07-11 結論一致 — catalog-driven DAG 核心設計冇結構性問題。今次
review 嘅新 issue 集中喺三類:(a) canto_ft 退役嘅**未收尾後果**(信任 gate、dead
weights、dead dependency);(b) 呢兩日高強度 debugging 產生嘅**文檔/git 滯後**;
(c) 兩個由 throughput 調查副產品發現嘅**結構性小缺陷**(`--batch` default、
punctuation-blind agreement)。

---

## §2 Issues Register(#17 起,接續 07-11 doc 嘅 #1–16)

嚴重度定義同上份:**High** = 違反 hard constraint 或會令下游用錯數據;**Medium** =
功能缺口/會誤導;**Low** = hygiene/磁碟;**Info** = 記錄在案。

| # | 嚴重度 | Issue | 一句話補救 |
|---|---|---|---|
| 17 | **High** | `auto_gold` gate 失去 confidence signal:gate 要求 `canto_ft_confidence > 0.8`,canto_ft 退役後新 segment 呢欄永遠 `NULL` → gate fails closed,新數據封頂 silver/bronze。唔係 data corruption(fails closed 係安全方向),但 T15 嘅 578,889 segments tier 完會冇一個 auto_gold — 直接影響可訓練 pool 嘅規模 | T16(§3 Phase C3):backfill agreement(exclude canto_ft)→ 分佈分析 → owner 定新 threshold。§5 調研畀咗三個候選補強 signal |
| 18 | **High** | **12 modified + 2 untracked 未 commit**,當中 `pipeline/catalog/schema.sql`(embedding column migration)、`pipeline/nodes/speaker.py`(columnar read)、`recover_orphans.py`(reingest node)已經**live 行過 production catalog** — 即係 code 同 DB 狀態已綁定,呢啲 diff 一旦遺失,catalog 入面嘅 `re_admitted` rows / `embedding` column 會變成無 code 對應嘅孤兒狀態 | §3 Phase B:三個 logical commits,owner review 後連 3-commit backlog 一齊 push |
| 19 | Medium | CLI `--batch` default(8)嚴重 under-feed `qwen3_asr` 嘅 `max_inference_batch_size=64` — 實測 17.4/s vs 42.6/s(**2.4×**)。今次靠 shell script 手寫 `--batch 64` 補救,但任何人日後直接 `pipe run asr.transcribe` 唔記得帶 flag 就會再中 | §3 Phase B2:per-model default(model config 度加 `dispatch_batch`)或直接升 CLI default;一個細 code change + test |
| 20 | Medium | ✅ **FIXED 2026-07-13**:`char_agreement()`(`pipeline/nodes/asr.py`)原本用 raw `difflib.SequenceMatcher`,零 text normalization — AR model(qwen3_asr)靠 context 補標點、CTC model(sense_voice)天生唔識標點,兩者比較會被 punctuation mismatch 系統性壓低 agreement。已加 `_normalize_for_agreement()`(strip 全部 Unicode punctuation + 阿拉伯/全形數字歸一做 CJK 數字),對 normalized string 計 overlap,`best_text`/原文照存唔變。5 個新/擴充 test 全過,見 DECISIONS.md 同一日 entry | T16 step 1 done;step 2(backfill,exclude canto_ft)待 T15 drain 完先跑,唔可以喺 T15 held 住 writer lock 期間做 |
| 21 | Medium | T15 killed run 留低 ~107,668 個 canto_ft-only `asr_results` rows — 政策上屬「historical rows 留底」(同 whisper_v3 一致,**唔刪**),但令 `test_asr_results_at_least_two_architectures_per_segment` 持續紅,直至 qwen3_asr/sense_voice 覆蓋晒嗰批 id(T15 跑完自愈) | 唔使行動;記錄喺度等佢自愈後 verify 返 test 轉綠 |
| 22 | Medium | 預設 manifest/train/val stale(2026-07-09)— 07-11 Issue #2 續期,而家仲隔多咗 T15 | 維持原策略:T15 全 drain 完先一次過 `manifest.export` + `report.build`(§3 Phase C2)|
| 23 | Low | **退役 ASR model weights ~13GB dead**:`data/ct2_models/` 2.9GB(canto_ft ct2 本體,repo 內)+ HF cache `simonl0909/whisper-large-v2-cantonese` 5.8GB(canto_ft 原始 weights)+ `Systran/faster-whisper-large-v3` 2.9GB(whisper_v3)+ `facebook/mms-lid-256` 1.4GB(全 repo 零 reference,active 嘅係 mms-lid-126)。全部可重新 download,唔屬不可逆 | §3 Phase A(純磁碟,零 git/DB)|
| 24 | Low | `faster-whisper` + `ctranslate2` dependency 而家 runtime 完全冇用(兩個 Whisper backend 都退役);`TranscribeWorker` code path 保留(historical reference + import 係 lazy 嘅,唔會 crash) | §3 Phase D:從 `pyproject.toml` 移除(**記住 `uv pip`,永遠唔好 `uv sync`**);唔急,冇害處淨係佔位 |
| 25 | Low | 3-shard 上嘅 legacy per-segment `.npy` embedding sidecars 已被 in-table `embedding` column 取代(IO plan Phase 3,`embedding_ref` 留作 fallback)— 潛在幾百 GB 級數嘅 file-count/空間回收,但**未量化、未 verify 100% column coverage** | §3 Phase D:先 `SELECT count(*) WHERE embedding IS NULL` verify,再量 size,先至入 plan;唔好靠估 |
| 26 | Info | `run_t15_asr_sequential.sh`(untracked 一次性 script):T15 完成前保留;完成後刪(postmortem 已完整寫入 DECISIONS.md + pending_task.md,script 本身冇 archive 價值) | T15 完成後 `rm` |
| 27 | Info | `~/.cache/modelscope` 6.5GB 入面 5.6GB 係 `ASLP-lab/Cosyvoice2-Yue` — 屬 [[canto-tts]] 實驗,唔係本 repo 範圍;`iic/SenseVoiceSmall`(active)好細 | 唔屬本 repo cleanup;記錄俾 owner 知 machine-level 有呢舊嘢 |
| 28 | Info | 07-11 review 嘅 5 個 OPEN item 續期:T1(pilot QA,owner 人手,**仍然係最高優先**)、T5(re-eval 機制,下個新 ASR model 前必須)、T9/T10(P6)、T11/T12(optional) | 見 `pending_task.md` |
| 29 | Info | PROGRESS.md 缺咗 2026-07-12/13 兩日 session entry(session-start protocol 違規)+ CLAUDE.md 有 6 處仲當 canto_ft active + DECISIONS.md/pending_task.md 未記 batch=64 發現 | ✅ **本 session 已全部修復**(§6)|

---

## §3 Cleanup / 收尾 Plan(plan-only;分四個 Phase,低風險行先)

> 每個 Phase 完結跑 `pytest tests/ -q`(基線:304 passed + 2 個已記錄 live-catalog
> failures)確認冇新增破壞。Phase C 需要 DuckDB writer — **必須等 T15 script 完**。

### Phase A — 退役 model weights(純磁碟,零 git/零 DB,即批即做)

| 目標 | 動作 | 回收 | 前置檢查 |
|---|---|---|---|
| `data/ct2_models/`(canto_ft ct2) | `rm -rf data/ct2_models` | 2.9GB | 無 — `asr.py` 嘅 `_LOCAL_CANTO` 常數保留(resolve 歷史 rows 用),指住唔存在嘅路徑冇問題,model 永不 dispatch |
| HF `models--simonl0909--whisper-large-v2-cantonese` | `rm -rf` 該 hub 目錄 | 5.8GB | `grep -r simonl0909 ~/Documents/canto-tts/` 確認冇其他 project 用 |
| HF `models--Systran--faster-whisper-large-v3` | 同上 | 2.9GB | 同上 pattern |
| HF `models--facebook--mms-lid-256` | 同上 | 1.4GB | 本 repo 零 reference 已確認(active 係 mms-lid-126);照樣 grep 其他 project 一次 |

**合計 ~13GB**。全部可由 HF/ModelScope 重新 download — 唔係不可逆操作,但退役決定
本身已有完整 DECISIONS.md 記錄,冇預期會回頭。

### Phase B — Git 收尾(三個 logical commits,owner review 後 push)

**B1 — canto_ft 退役 + throughput 修正**(今日嘅工作):
`pipeline/nodes/asr.py`、`pipeline/cli.py`、`tests/test_asr_node.py`、`DECISIONS.md`、
`pending_task.md`、`CLAUDE.md`、本 doc + `docs/archive/PIPELINE_REVIEW_2026-07-11.md` 嘅
pointer 更新、`PROGRESS.md`。建議 message:
`Retire canto_ft ASR backend; fix qwen3_asr batch starvation (2.4x); 2026-07-13 review docs`

**B2 —(可選,同 B1 一齊)`--batch` default 修正**(Issue #19):
最細改法 — `ASR_MODELS["qwen3_asr"]` 加 `"dispatch_batch": 64`,supervisor dispatch 時
`batch_size = max(batch_size, model_cfg.get("dispatch_batch", 0))`,或者直接
`p_run_asr` 嘅 `--batch` default 8→64(sense_voice 都食得起)。加一個 regression test。

**B3 — T15/IO-optimization 基建**(前兩日嘅工作,已 live 行過 production):
`pipeline/catalog/schema.sql`(embedding column)、`pipeline/nodes/speaker.py`(columnar
read + run-many 註記)、`pipeline/nodes/recover_orphans.py`(`recover.reingest_pending`)、
`docs/IO_OPTIMIZATION_PLAN.md`(untracked → add)。建議 message:
`T15 reingest admission node + columnar embedding storage (IO plan phase 3)`

**B4 — Sources + ingest 韌性**:
`sources/podcast_sources.yaml` / `youtube_channels.yaml`(新 channel)、
`pipeline/nodes/ingest_download.py`(`--socket-timeout 30`)。建議 message:
`Expand podcast/youtube sources; add yt-dlp socket timeout`

Push 順序:B1→B4 全部 commit 完,連同現有 3-commit backlog(`f511918`/`78d66ef`/
`e299dd8`)一齊 push。**Push 前照舊過一次 HC#9 檢查**:`git diff origin/master..HEAD
--stat` 確認冇 source_url 數據/reconstruction recipe 內容誤入。

### Phase C — Post-T15 catalog 動作(等 writer free,順序做)

1. **C1 — downstream drain**:`asr.agreement` → `filter.text` → `filter.acoustic` →
   `filter.decide` → `g2p` → `tier.assign`(T15 剩餘 chain);`speaker.cluster`
   whole-source recompute(1.24M segments — **solo 跑**,唔好 run-many pair,見
   pending_task T15 point 5;考慮先落 point 14 提出嘅 `upsert_rows` chunking fix)。
2. **C2 — 一次過收割**:`manifest.export` + `report.build`(清 Issue #22 / T6);之後
   deliberate re-baseline `test_manifest_build_matches_expected_corpus_totals`(照佢
   docstring 嘅規矩,verify 完先改數);verify Issue #21 嘅 test 自愈轉綠。
3. **C3 — T16:auto_gold 新 gate**(Issue #17+#20,**必須跟呢個次序**):
   a. 先落 Issue #20 嘅 normalization fix(punctuation-strip 版 agreement);
   b. 全 corpus backfill `asr_agreement`(exclude canto_ft,mirror 2026-07-10
      whisper_v3 backfill 嘅做法);
   c. FINDINGS-doc-style 分佈分析(agreement histogram × 現有 gold/QA ground truth);
   d. Owner 決定新 bar — §5 調研建議:2-model overlap 閾值放寬到 0.90–0.93(normalized
      text 上),**再加第三個非 ASR signal** 補返失去嘅 confidence gate(現成候選:
      `filters_acoustic` 嘅 DNSMOS ≥ 3.5 — 零新 compute;進階候選見 §5)。
4. **C4 — T15 成效覆盤**(pending_task T15 follow-up):對比 re-admitted 578,889 嘅
   pass-rate/tier 分佈 vs 原本 `pending_delete` 分類,量化 legacy ASR 嘅 false-negative
   率,寫入 DECISIONS.md。

### Phase D — 延後/條件性(唔急,逐項獨立)

| 目標 | 條件/動作 | 回收 |
|---|---|---|
| `.npy` embedding sidecars(3 shards) | 先 verify `embedding IS NULL` count = 0(或只餘 read_failed),再量 size,先出執行 plan — **未 verify 前唔准刪** | 未量化(可能好大)|
| `faster-whisper`+`ctranslate2` 出 `pyproject.toml` | `uv pip uninstall` + pyproject 改;跑 full test suite 確認 lazy import 冇被踩中 | ~500MB venv |
| `run_t15_asr_sequential.sh` | T15 完成後 `rm`(Issue #26)| — |
| `metadata/` stale exports(manifest/train/val ~1.18GB) | **唔刪** — Phase C2 export 直接覆蓋 | 0 |
| Log retention 自動化(T12)| logrotate 或 startup truncate;39M 而家唔急 | — |
| `metadata/release_dormant/` 數據檔搬位(T11)| 兩個 `mv`,純 cosmetic | 0 |

### 預計總回收

| 類別 | 大約 |
|---|---|
| Phase A model weights | **~13GB** |
| Phase D venv 依賴 | ~0.5GB |
| Phase D sidecars | 未量化(執行前先量)|
| **即批即得** | **~13GB** |

---

## §4 執行次序 + 驗證 checklist

```
而家(T15 跑緊都做得):  Phase A(磁碟)、Phase B(git commits;push 留 owner 批)
T15 完成後:             Phase C1 → C2 → C3(T16)→ C4
任何時候,逐項:         Phase D
持續(owner 人手):      T1 pilot QA review(仍然係全 project 最高優先嘅人手動作)
```

每步驗證:`pytest tests/ -q`(基線 304+2);Phase C 每個 node 跑完 check
`metadata/logs/{node}.log` 尾部 + `pipe catalog verify`;C2 之後 `report.build` 嘅
11 項 acceptance criteria 對照 07-11 嘅 10/11 PASS 基線。

---

## §5 Targeted 外部調研(agy-gemini via weir,2026-07-13)

> 範圍:淨係針對 Issue #17(2-model trust gate)— 07-11 doc §5 嘅 comprehensive
> best-practice 對照(LibriTTS-R/Emilia/WenetSpeech4TTS/GigaSpeech 2/YODAS)不重做,
> 結論照舊有效。原始調研輸出存於 session scratchpad,要點如下。

**Q1 — 兩個 ASR system 嘅 cross-agreement 夠唔夠做 auto-trust?**
業界共識(2025–26):**唔夠**。AR 同 CTC 嘅 failure mode 唔同(hallucination vs
deletion),兩者完全一致往往只代表音頻「淺」,唔一致又唔一定代表差。現代 pipeline
(GigaSpeech 2 / Emilia-Pipe)唔會純靠 ASR text overlap — 佢哋疊加 **LID confidence**
同 **DNSMOS** 做 gate;YODAS distilled 子集要 3 個獨立 source 至少兩兩 match。
**對本 project 嘅落地**:2-model overlap 閾值可以放寬(0.90–0.93,normalized text 上),
但 auto_gold 必須補一個**非 ASR 嘅第三 signal** — 最平嘅現成選項係
`filters_acoustic.dnsmos`(≥3.5 檔位),零新 compute,catalog 已有欄。

**Q2 — 兩個現役架構有咩平嘅 per-utterance confidence 可以攞?**
- `qwen3_asr`(AR):HF `generate()` 加 `output_scores=True` 攞 token logprobs →
  normalized sequence log-prob / utterance entropy。⚠️ AR logprob 有「hallucination
  overconfidence」問題 — 流暢但錯嘅句子都會高分,唔好單獨用。
- `sense_voice`(CTC):**CTC posterior frame entropy** — 調研指同 acoustic CER 相關性
  最好嘅 signal(peaky 分佈=確定,flat=唔確定)。另有零成本 proxy:SenseVoice 嘅
  emotion/event tag 落 `<|Unknown|>` 類 → 低 confidence 訊號(呢啲 tag 我哋本身已存喺
  `asr_results.metadata`,**依家就可以做離線相關性分析,唔使重跑 ASR**)。
- External LM perplexity:**唔建議** — 分唔開「流暢嘅 hallucination」同真 transcript。
- 進階(如果 DNSMOS gate 唔夠):**forced-alignment score 做 tie-breaker**(MMS-align /
  wav2vec2 CTC aligner)— 將音頻分別對兩個 transcript 做 forced alignment,邊個
  alignment score 高邊個做 best_text,score 本身即係一個真・acoustic confidence,
  完整取代失去咗嘅 canto_ft logprob。

**Q3 — AR vs CTC 做 char-overlap 嘅系統性 pitfall?**
1. **Punctuation/ITN mismatch(系統性壓低)**:CTC 冇 acoustic cue 預測標點,AR 靠
   LM 大量補標點 — raw text 直接比較必然 deflate。**業界做法:比較前 strip 晒標點 +
   數字歸一(阿拉伯→中文),對 stripped string 計 overlap**。⇒ 正正係 Issue #20:
   我哋 `char_agreement()` 而家係 raw `SequenceMatcher`,冇任何 normalization。
2. **錯誤方向唔對稱**:CTC fail 係 deletion(快速粵語尤甚),AR fail 係 insertion/
   looping — 對稱嘅 Levenshtein 一視同仁會誤判;有需要可考慮非對稱計分(進階,唔急)。
3. **Tie-breaker**:兩者中度一致(~0.85)時邊個啱?AR 流暢但唔忠於音頻、CTC 忠於音頻
   但可能甩字 — forced alignment 係公認嘅裁判(見 Q2)。

**對 T16 嘅具體建議次序**:先修 Issue #20(normalization)→ backfill → 分佈分析 →
新 gate 起點提案「`normalized_agreement ≥ 0.90–0.93` **AND** `dnsmos ≥ 3.5`」→ 用
T1 pilot QA 嘅人手 ground truth 驗證呢個 gate 嘅 precision,先至定案。SenseVoice tag
proxy 同 CTC entropy 留作第二階段(如 precision 唔達標先加)。

---

## §6 本 session 已完成嘅文檔同步(唔屬 plan,已做)

| 檔案 | 更新內容 |
|---|---|
| `CLAUDE.md` | 6 處 staleness 修正:`ct2_models` 標記 DEAD、`asr.transcribe`/`asr.agreement` node 表改 2 active、ASR strategy 大段重寫(canto_ft 退役 + throughput conventions)、`conn=` 10/22 → 23/23(連 run-many 兩個實測 caveat)、tier 注釋加 auto_gold fails-closed 警告、faster-whisper dependency 行改 removal candidate |
| `DECISIONS.md` | 2026-07-13 entry 補 addendum:batch-size mismatch 發現 + 修正(17.4→42.6/s)|
| `pending_task.md` | T15 point 13 補實測數;新 point 14(batch64);新 **T16** task(auto_gold gate 重建,引用本 doc §5)|
| `PROGRESS.md` | 補返缺咗嘅 Session 2026-07-12 entry(T15 admission/embed/IO plan)+ 新 Session 2026-07-13 entry(canto_ft 退役 + throughput 調查 + 本 review)|
| `docs/archive/PIPELINE_REVIEW_2026-07-11.md` | 頂部加 pointer 指向本 doc(status line 唔改 — 嗰份係已完成嘅歷史記錄)|
