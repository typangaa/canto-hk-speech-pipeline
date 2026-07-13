# Pipeline Comprehensive Review + Cleanup Plan — 2026-07-11

> **➡️ 後繼版本:`docs/PIPELINE_REVIEW_2026-07-13.md`**(canto_ft 退役後嘅新一輪 review;
> issue 編號 #17 起接續本 doc 嘅 #1–16;本 doc 唔再更新,只作歷史記錄)。
>
> **Status**: **EXECUTED + PUSHED 2026-07-11**。全部四個 phase(A/B/C/D)已完成並逐一驗證
> (`pytest tests/` + `pipeline.cli catalog verify`)。Phase A/B/D 純磁碟操作,已生效。
> Phase C 產生兩個 commit(`420981f`+`c68c5f2`),連同 DECISIONS.md 措辭修正(`16bdd96`)
> 已於同日 push 上 `origin/master` — local 而家 **0 commits ahead**,Issue #14 嘅 push
> backlog 全清。詳細執行記錄見 `PROGRESS.md`「Session 2026-07-11 (cleanup)」。
> **Round-2 post-execution review 見 §6**(同日晚間覆核:逐項 issue disposition + 新發現)。
>
> Owner 確認咗四個前提(AskUserQuestion,2026-07-11):
> ① legacy scripts 分級處理(唔係一刀切);② `reconstruct.py` 公開風險用 `git rm` + 保留
> history 處理(唔做 history purge);③ 範圍包晒 repo tracked 檔案、disk 雜物、大型 disk
> artefacts、`metadata/` 舊 backup/export 四類;④ 呢份 doc 淨係出 plan,執行要 owner 另行指示。
>
> **執行任何一項之前**:先閂咗 `pipe calibrate serve` 同任何跑緊嘅 node(DuckDB
> single-writer),再逐 phase 做,每 phase 完跑一次 `pytest tests/` + `pipe catalog verify`。

---

## §1 現況快照(2026-07-11)

| 項目 | 數值 |
|---|---|
| Git tip | `01dff10`(master,working tree clean,**領先 origin 8 commits 未 push**)|
| Milestone | P0–P5 全部完成;**P6(scale readiness)未開始** |
| Catalog | `metadata/corpus.duckdb` 2.3GB,618,695 segments tiered |
| Tier 分佈(manifest-eligible,`filters.pass=TRUE`)| gold=43 / auto_gold=72,014 / silver=235,646 / bronze=151,140 / excluded=25,989 |
| Manifest-eligible pool | **458,843 segments / 1,018.9h / 8,817 speakers** |
| 最新 export | `manifest_tier_auto_gold.jsonl`(72,057 entries / 150.6h / 2,370 speakers,2026-07-11)|
| 預設 manifest | **STALE**(2026-07-09,早過兩次 backfill)|
| Tests | 289 passed |
| QA 狀態 | 3 個 300-segment pilot batch(auto_gold/silver/bronze)queued,**全部未 review** |
| Acceptance criteria | **10/11 PASS**(`report.build` 已 port,見 §6)— 只有 `text_verified` FAIL(0.0%,58/458,844),符預期,見 §6.4 |

**Pipeline 架構本身係健康嘅**:catalog-driven DAG、idempotent provenance-tagged discovery、
decode-once audio bus、run-many 並行、48kHz lossless master 政策、multi-ASR agreement +
人手 calibration — 呢啲核心設計冇發現結構性問題。以下 issues 主要係 hygiene、
staleness 同未完成嘅收尾。

---

## §2 Issues Register

嚴重度定義:**High** = 違反 hard constraint 或會令下游用錯數據;**Medium** = 功能缺口/
會誤導;**Low** = hygiene/磁碟空間;**Info** = 記錄在案,唔使行動。

| # | 嚴重度 | Issue | 一句話補救 |
|---|---|---|---|
| 1 | **High** | `reconstruct.py` + `reconstruct_dead_sources.txt` 已喺公開 GitHub(違反 Hard Constraint #9) | `git rm` + 本地收埋去 gitignored 位置,push 後 tip 唔再有 |
| 2 | **High** | 預設 `metadata/manifest.jsonl`/`train.jsonl`/`val.jsonl` stale(2026-07-09,早過 whisper_v3 退役 backfill 同 tier 收緊 backfill) | Owner 決定時機後重跑 `pipe run manifest.export` |
| 3 | Medium | `report.build` node 未 port;legacy `scripts/10_report.py` 讀緊 `data/filtered/`(symlink 已死)= 完全壞 | Port `report.build`(讀 catalog),之後先可以刪 `10_report.py` |
| 4 | Medium | `filter.text`/`filter.decide`/`tier.assign` discovery 係 bare row-existence anti-join — 後來 model(sense_voice)改善咗 `best_text` 嘅 segment 唔會自動重評 | 加 agreement-version/provenance re-eval 機制(P6 候選工作) |
| 5 | Medium | `run-many` `conn=` injection 得 10/22 node 完成 | 機械式 follow-up(`docs/ORCHESTRATOR_PLAN.md` 有清單) |
| 6 | Medium | `requirements.txt` 嚴重過時(冇 duckdb/qwen-asr/funasr/opencc;話 canto-hk-g2p 未上 PyPI — 其實已上)且同 `pyproject.toml`+`uv.lock` 雙軌;照住裝會整壞環境 | `git rm requirements.txt`,README 指去 `uv pip install`(**永遠唔好 `uv sync`** — venv 有 lock 外 GPU torch) |
| 7 | Medium | `data/` symlink tree 過時:`data/filtered` → 已死目標;`data/segments/*` 淨係指 Drive4(P5-C 之後 segments 3-way sharded,經 symlink 睇只見 ⅓ corpus,rglob 唔跟 symlink 嘅舊教訓再疊加) | 刪死鏈 + 刪誤導性 segments symlinks;SSOT 係 `config/storage_layout.yaml` |
| 8 | Medium | `scripts/10_enrich_manifest.py`(release-recipe 生成器)都喺公開 repo — 同 Issue #1 同類,屬 HC#9 reconstruction-recipe 工具鏈 | 同 `reconstruct.py` 一齊 `git rm` + 本地收埋 |
| 9 | Low | `.venv_ina/` 6.8GB — inaSpeechSegmenter 實驗 venv,全 codebase 零 reference | 刪(owner 已同意納入評估範圍;執行前最後 grep 一次) |
| 10 | Low | `metadata/` 已核實遷移嘅舊 backup ~2.26GB(6 個 manifest/train/val `.bak` + 2 個 downloaded `.bak` + `asr_agreement_2model_backup.parquet`) | 刪 — 對應遷移全部 verify 完好耐 |
| 11 | Low | `metadata/logs/` 1.7GB 冇 retention policy;最大單檔 400MB(`06_filter.log`,legacy stage) | 刪 legacy-stage logs,現役 logs 壓縮/輪替 |
| 12 | Low | Repo root 雜物:`守下留情_results.yml`、`.search_results.png`、`.playwright-mcp/`、`.pytest_cache/`、各處 `__pycache__` | 全部可刪(gitignored/可再生) |
| 13 | Low | 文檔漂移:CLAUDE.md 提及 `sources/hktv_sources.yaml`(唔存在);CLAUDE.md scripts 清單同磁碟實況唔符;`metadata/DATASET_REPORT.md` 停留喺 2026-06-11 | Cleanup 執行後同步更新 CLAUDE.md;DATASET_REPORT 等 `report.build` 重生 |
| 14 | Info | Local 領先 origin 8 commits 未 push | Owner 決定 push 時機(建議連 Issue #1/#8 嘅 removal 一齊 push) |
| 15 | Info | 3 個新 pilot QA batch 未 review;舊 batch `fd9269e121be` 只剩 109/300 仲係 auto_gold | Owner 下一步天然動作,唔屬 cleanup |
| 16 | Info | `metadata/manifest_release.jsonl`(672MB)+ `excluded_no_url.jsonl`(8.4MB)— dormant release 數據,喺 gitignored `metadata/` 內,政策係「dormant 唔刪」 | 保留;可搬入 `metadata/release_dormant/` 令意圖更清晰 |

### Issue #1/#8 詳情(公開風險 — 最高優先)

- **證據**:`git ls-tree origin/master --name-only | grep reconstruct` 命中兩個檔案;repo
  `https://github.com/typangaa/canto-hk-speech-pipeline` 係 public。
- **實際洩露程度低**:兩個 script 本身唔含 source URL 或音頻數據(數據喺 gitignored
  `metadata/manifest_release.jsonl` 等,從未 commit);洩露嘅係「重建方法論」。
- **Owner 決定**:`git rm` + 保留 history(唔做 `git filter-repo` purge)。History 入面
  舊版仍然搵得返 — 呢一點 owner 已知悉並接受。如日後想徹底清除,filter-repo + force
  push 嘅選項記錄在此,但屬 destructive,必須另行明確授權。
- **注意**:push 之前 tip 上嘅移除唔會生效於 GitHub — Issue #14 嘅 push 應該喺
  removal commit 之後先做。

### Issue #4 詳情(re-evaluation 缺口)

`asr.agreement` 用 `model_count` 做 straggler re-trigger(遲到 model 會令 agreement 重計),
但下游 `filter.text` → `filter.decide` → `tier.assign` 係「有 row 就當做咗」。即係:
sense_voice(2026-07-08)上線前已 filter/tier 嘅 segment,agreement 分數已更新,但
filter/tier verdict 冇跟住重評。兩次全量 backfill(2026-07-10、07-11)已經抹平咗**現有**
數據嘅唔一致,所以呢個係「日後新 model 加入時會重現」嘅結構性缺口,唔係現行數據錯。
建議修法:`filters_text`/`tiers` 加 `agreement_version`(或直接記 `asr_agreement` 嘅
last-modified marker),discovery 改成「無 row **或** version 落後」— 屬 P6 前嘅小型工程。

---

## §3 Cleanup Plan(分四個 Phase,低風險行先)

> 每個 Phase 完結:`pytest tests/` 全綠 + `python -m pipeline.cli catalog verify` 通過先入
> 下一 Phase。所有 `git rm` 集中做一個 commit,方便 revert。

### Phase A — Disk 雜物(零風險,唔郁 git)

| 目標 | 動作 | 回收 |
|---|---|---|
| `守下留情_results.yml`(root,gitignored ad-hoc result) | `rm` | 42KB |
| `.search_results.png`(research screenshot) | `rm` | 391KB |
| `.playwright-mcp/`(舊 MCP 工具目錄,現已不用) | `rm -rf` | 8.4MB |
| `.pytest_cache/` | `rm -rf`(自動再生) | 細 |
| 全部 `__pycache__/`(scripts/config/tests/pipeline/*) | `find . -name __pycache__ -not -path "./.venv*" -exec rm -rf {} +`(自動再生) | 細 |
| `data/filtered` **死 symlink** | `rm data/filtered` | — |
| `data/segments/{podcast,rthk,youtube}` symlinks(只指 Drive4,誤導) | `rm` 三條 symlink;`rmdir data/segments` | — |
| `data/final/`(空目錄,無 reference) | `rmdir` | — |

**唔郁**:`data/raw` symlink(仲有效,指 Drive2)、`data/ct2_models/`(**2.9GB 但係 active**
— `canto_ft` ctranslate2 model 本體,`asr.transcribe` 用緊)、`.claude/`、`.cache/`(28KB,
speechbrain)。

### Phase B — `metadata/` 舊 backup / 已消化 sidecar(disk only,永不入 git)

**B1. 直接刪 — 已核實遷移嘅 backup(~2.26GB)**:

| 檔案 | 大細 | 點解安全 |
|---|---|---|
| `manifest.jsonl.pre-remap.bak` / `train...` / `val...`(×3) | ~1.10GB | 2026-06-26 path-remap,verify 完、後續已重寫多代 manifest |
| `manifest.jsonl.pre-asr-remap.bak` / `train...` / `val...`(×3) | ~1.10GB | 2026-07-02 ASR-model-path remap,同上 |
| `downloaded.jsonl.pre-remap.bak`、`downloaded.jsonl.bak-20260704T123411` | 7.7MB | backfill_downloaded_jsonl 修復完成並 verify(2026-07-04) |
| `asr_agreement_2model_backup.parquet` | 59MB | 2026-07-08 sense_voice 加入前嘅 2-model agreement snapshot;之後已經歷兩次全量重計+backfill,無回滾價值 |

**B2. 舊 log 清理(~1.4–1.5GB)** — `metadata/logs/` 入面 legacy stage(`scripts/NN_*`)嘅
log 已無診斷價值(stage 本身退役):

```bash
# 例:legacy-stage + 一次性 fix 嘅 log
rm metadata/logs/0[1-9]_*.log metadata/logs/fix_stale_paths.log \
   metadata/logs/run_6_to_9*.log metadata/logs/phaseA_*.log \
   metadata/logs/04_transcribe_*.log metadata/logs/02_download_*.log
# 現役 node log({node_name}.log)保留;建議日後加 logrotate 或 startup truncate
```

**B3. P0-import sidecar(已入 catalog,catalog 係 SSOT)— 壓縮封存,唔即刻刪**:

`lang_id.jsonl`(85MB)、`overlap.jsonl`(57MB)、`audio_tags.s0/s1.jsonl`(19MB)、
`tag_calib.jsonl`/`lang_calib.jsonl`/`overlap_calib.jsonl`/`tag_calib_review.tsv`(<300KB)
— 全部係 `pipeline/catalog/ingest.py` 一次性 import 嘅 input,對應 rows 已喺
`labels_lang`/`labels_overlap`/`labels_music`。建議:

```bash
mkdir -p metadata/archive
tar --zstd -cf metadata/archive/p0_import_sidecars_20260711.tar.zst \
    metadata/lang_id.jsonl metadata/overlap.jsonl metadata/audio_tags.s*.jsonl \
    metadata/*_calib.jsonl metadata/tag_calib_review.tsv && \
  rm metadata/lang_id.jsonl metadata/overlap.jsonl metadata/audio_tags.s*.jsonl \
     metadata/*_calib.jsonl metadata/tag_calib_review.tsv
```
(~160MB → 大約 40–60MB 封存;想再進取可以連封存都唔要,但 catalog 一旦要重建就冇底稿,
唔建議。)

**B4. 保留唔郁**:`corpus.duckdb`(SSOT)、`downloaded.jsonl`、`yt_archive.txt`、
`raw_files_known_ids.json`、`ingest_download_staging.committed-*.jsonl`(ingest 現役)、
`labels.jsonl` + `labels/calibration.json`(label.store/calibrate 現役輸出)、
`legacy_filenames/`(tier2_legco 翻案底稿)、`manifest_tier_auto_gold` 三件套(現役 export)、
`manifest.jsonl`/`train.jsonl`/`val.jsonl`(stale 但唔刪 — 等 re-export 覆蓋,見 Issue #2)、
`manifest_release.jsonl`/`excluded_no_url.jsonl`(dormant 政策,見 Issue #16;可選搬入
`metadata/release_dormant/`)。

### Phase C — Git tracked 檔案(一個 commit,owner review 後 push)

**C1. HC#9 公開風險(Issue #1/#8)**:

```bash
mkdir -p metadata/release_dormant
git mv reconstruct.py reconstruct_dead_sources.txt scripts/10_enrich_manifest.py \
    metadata/release_dormant/ 2>/dev/null || true
# git mv 入 gitignored 目錄唔會自動 untrack,實際做法:
cp reconstruct.py reconstruct_dead_sources.txt scripts/10_enrich_manifest.py metadata/release_dormant/
git rm reconstruct.py reconstruct_dead_sources.txt scripts/10_enrich_manifest.py
```
(先 `cp` 落 `metadata/release_dormant/`(gitignored,永不 commit),再 `git rm` — dormant
政策維持:本地留底,公開 repo tip 移除。)

**C2. Legacy scripts 分級處置(owner 揀咗「分級處理」)**:

| 檔案 | 處置 | 理由 / 替代品 |
|---|---|---|
| `00_reingest.py` | `git rm` | 一次性 legacy 重下載,2026-06-09 完成 |
| `01_discover.py` | `git rm` | source 調研一次性;`sources/*.yaml` + `docs/SOURCE_GUIDE.md` 承接 |
| `02_download.py` | `git rm` | → `ingest.download`/`ingest.commit` |
| `03_segment.py` | `git rm` | → `segment.diarize`/`segment.vad_cut` |
| `03b_acoustic_pregate.py` | `git rm` | → `pregate.snr` |
| `04_transcribe.py` | `git rm` | → `asr.transcribe`(3 active backends) |
| `05_calibrate.py` | `git rm` | → `pipe calibrate serve` browser UI |
| `06_filter.py` | `git rm` | → `filter.text`/`filter.acoustic`/`filter.decide` |
| `07_g2p.py` | `git rm` | → `g2p` node |
| `08_speaker_id.py` | `git rm` | → `speaker.embed`/`speaker.cluster` |
| `09_manifest.py` | `git rm` | → `manifest.build`/`manifest.export` |
| `10_report.py` | **保留** | 唯一未有 node 替代(`report.build` 未 port);本身已壞(讀死咗嘅 `data/filtered`),只作 port 參考。**去留條件:`report.build` port 完即刪** |
| `10_enrich_manifest.py` | `git rm`(見 C1) | HC#9 dormant release 工具,本地留底 |
| `11_audio_tag.py` | `git rm` | → `label.music`/`label.suite` |
| `12_language_id.py` | `git rm` | → `lang_screen.auto` + label suite |
| `13_overlap_detect.py` | `git rm` | → label suite(`labels_overlap`) |
| `backfill_downloaded_jsonl.py` | `git rm` | 一次性修復,2026-07-04 完成並 verify |
| `fix_stale_asr_model_manifest.py` | `git rm` | 一次性 hotfix,完成;canto_ft path dedupe 已入 `asr.agreement` 邏輯 |
| `fix_stale_paths.py` | `git rm` | 一次性 path remap,完成 |
| `test_sensevoice.py` | `git rm` | 整合驗證 scratch;sense_voice 已正式入 `asr.py` + tests |

全部 `git rm` 嘅檔案 git history 永遠攞得返(`git show 01dff10:scripts/04_transcribe.py`),
呢個就係 archive 機制 — 唔另設 `legacy/` 目錄。

**C3. 依賴聲明去重**:`git rm requirements.txt`(過時且危險 — 見 Issue #6),README
加一句:依賴以 `pyproject.toml` + `uv.lock` 為準,安裝用 `uv pip install`,
**切勿 `uv sync`**(venv 有 lock 外 GPU torch,sync 會 prune CUDA)。

**C4. 文檔同步(同一 commit)**:
- `CLAUDE.md`:更新 directory layout(scripts/ 剩 `10_report.py`;root 冇咗
  `reconstruct.py`/`requirements.txt`);刪 `sources/hktv_sources.yaml` 嗰行(檔案不存在)
  或改為「未建立」;`data/` 一段改講「symlink tree 已清理,只餘 `data/raw` + `data/ct2_models`,
  一切路徑以 `config/storage_layout.yaml` 為 SSOT」。
- `DECISIONS.md`:加 2026-07-11 cleanup entry(引用本 doc)。
- 本 doc status line 改返「EXECUTED YYYY-MM-DD」。

### Phase D — 大型 artefacts(執行前逐項最後確認)

| 目標 | 動作 | 回收 | 風險 |
|---|---|---|---|
| `.venv_ina/`(inaSpeechSegmenter 實驗) | `rm -rf .venv_ina` | **6.8GB** | 零 reference 已確認;刪前再 `grep -r venv_ina` 一次 |
| `metadata/manifest_release.jsonl` 等 dormant release 數據 | **唔刪**(政策);可選搬 `metadata/release_dormant/` | 0(搬位) | — |
| `.venv/`(9.2GB)、`data/ct2_models/`(2.9GB) | **唔郁** — 現役 | — | — |

### 預計總回收

| 類別 | 大約 |
|---|---|
| `.venv_ina` | 6.8GB |
| `metadata/` backups(B1) | 2.26GB |
| 舊 logs(B2) | ~1.4GB |
| sidecar 壓縮淨回收(B3) | ~0.1GB |
| Phase A 雜物 | ~10MB |
| **合計** | **~10.5GB** |

---

## §4 執行次序 + 驗證 checklist

1. 停 `pipe calibrate serve` / 任何跑緊嘅 node。
2. **Phase A** → `pytest tests/` + `pipe catalog verify`。
3. **Phase B**(B1→B2→B3)→ 同上驗證;另跑
   `python -m pipeline.cli run manifest.export --min-tier auto_gold --dry-run`
   確認 export 路徑無恙。
4. **Phase C** 一個 commit(建議 message:`Repo hygiene: retire ported legacy scripts,
   remove dormant release recipe from public tip, drop stale requirements.txt`)→
   owner review → push(順手清埋 Issue #14 嘅 8-commit backlog)。
5. **Phase D**(`.venv_ina`)最後做。
6. 全部完成後:更新 `PROGRESS.md`,並將本 doc status line 改「EXECUTED」。

**明確唔喺此 plan 範圍**(另行決策):Issue #2 嘅 default manifest re-export、Issue #3
`report.build` port、Issue #4 re-eval 機制、Issue #5 conn= 收尾、pilot QA review、
history purge(owner 已揀唔做)。

---

## §5 外部 best-practice 對照(agy-gemini via weir,2026-07-11 網上調研)

調研範圍:2025–2026 大規模 TTS data pipeline 架構、QC 同 repo hygiene 慣例
(參照 LibriTTS-R / Emilia / WenetSpeech4TTS / GigaSpeech 2 / YODAS 等公開 corpus 做法)。
逐項對照本 project:

| Best practice(業界) | 本 project 現況 | 評估 |
|---|---|---|
| Centralized DB 做 metadata SSOT,production 期間唔靠 sidecar | `corpus.duckdb` 單一 catalog,節點間零 sidecar 依賴 | ✅ 一致(甚至比常見 SQLite/Postgres 方案多咗 per-process writer 紀律) |
| Idempotent stage:執行前 check 已完成狀態,防重複計算 | Provenance-tagged SQL anti-join discovery | ✅ 一致;Issue #4(改良 best_text 唔會觸發 re-eval)係呢個模式嘅已知缺口,業界對應做法係 content-hash / version marker |
| Hash-based linkage 防 audio 搬位後 metadata 變孤兒 | 用絕對路徑 + `shard_index()` 決定位;P5-C rebalance 靠一次性 node 修路徑 | ⚠️ 部分 — 依賴「所有寫路徑必經 `shard_root()`」嘅紀律;如日後再搬 drive,可考慮補 content-hash 欄(P6 候選,唔急) |
| Durable orchestration(Temporal/Prefect)+ 分佈式 compute(Ray) | 自研 `run-many` + resource pools + run journal | ✅ 規模匹配 — 單機 3-GPU 場景用 Temporal/Ray 係 over-engineering;journal 已提供 resume 語義 |
| Data versioning(DVC/lakeFS) | 冇;靠 catalog provenance + backup 紀律 | ⚠️ 可接受 — 單機 ext4、私有 corpus,lakeFS 唔適用;manifest export 用 tagged filename(`manifest_tier_auto_gold.jsonl`)已係輕量版本化。如日後多人協作先再考慮 |
| Multi-ASR consensus filtering(WER/CER 閾值 5–10%) | 3-model char-overlap agreement,tier 化(0.95/0.85/0.70) | ✅ 一致並更精細(tier 分級 vs 二元 cut) |
| DNSMOS OVRL ≥ 3.0 gate(Emilia-Pipe 同款) | `filter.acoustic` DNSMOS ≥ 3.0 + SNR ≥ 25dB | ✅ 完全一致 |
| 質量分層俾使用者揀 subset(WenetSpeech4TTS 嘅 Premium/Standard/Basic/Rest) | `gold/auto_gold/silver/bronze/excluded` + `--min-tier` export | ✅ 同構;另有未建嘅 A/B TTS-quality 軸(LABEL_FRAMEWORK_SPEC §10) |
| Speaker purity:embedding + centroid cosine 剪枝(<0.75 剪走) | ECAPA-TDNN + agglomerative clustering;**冇 per-segment centroid-distance 剪枝** | ⚠️ 缺口 — 建議做 P6 候選:cluster 完加一步「同 speaker centroid cosine 過低嘅 segment 降級/剔除」,直接對應 acceptance criteria 嘅 speaker-purity 要求 |
| Human QA:全自動 + 高風險位抽樣(GigaSpeech 2 淨係人手做 10h 驗證集) | `calibrate.sample` risk-scaled 抽樣(1.5%/4%/10% by tier)+ browser review UI | ✅ 一致甚至更系統化 |
| One-off backfill script:verify 完就刪出 git,靠 history 做 archive | 今次 Phase C 正正係咁做(fix_*/backfill_* git rm) | ✅ 本 plan 同業界慣例一致 |
| 大 log 唔留 repo,30–90 日 lifecycle 自動清 | logs 喺 gitignored `metadata/logs/` 但無 retention(1.7GB) | ⚠️ Phase B2 處理;後續可加簡單 logrotate |
| 公開 code / 私有數據 artefact 嚴格分離(私有嘢入 private repo/secrets) | `metadata/`+`data/` gitignored;**但 reconstruct.py 呢類 recipe 工具流出咗公開 tip** | ⚠️ Issue #1/#8 — Phase C1 修;業界做法係呢類工具一開始就放 private repo,`metadata/release_dormant/` 承擔呢個角色 |

**總結**:核心架構(SSOT catalog、idempotency、multi-ASR consensus、DNSMOS gate、tiered
subsetting、sampled human QA)全部同 2025–26 業界做法對齊,個別位仲行前咗。真正值得跟進嘅
外部啟發係兩樣:① **speaker-centroid cosine 剪枝**(業界標配,本 pipeline 未有,直接影響
speaker-purity acceptance criterion);② **content-hash linkage** 作為日後再搬 drive 嘅保險。
兩樣都建議排入 P6。

---

## §6 Round-2 Post-Execution Review(2026-07-11 晚)

四個 phase 執行 + push 完成後嘅同日覆核。覆核時環境:`pipe calibrate serve`(PID 971786)
同兩個 `ingest.download`(podcast + youtube,13:55 起)**live 跑緊** — 即係 corpus 數字
持續漂移中,本節所有 count 都係覆核時刻嘅 snapshot。

### §6.1 覆核時驗證結果

| 驗證 | 結果 |
|---|---|
| `git status` | clean(淨係本 doc untracked — 工作文檔,唔入 git)|
| `git rev-list --count origin/master..HEAD` | **0** — 完全同步 |
| `git ls-files scripts/` | 淨係 `10_report.py` ✓(19 個 legacy scripts 已 git rm)|
| `metadata/release_dormant/` | 3 個 HC#9 dormant 檔案齊(`reconstruct.py`/`reconstruct_dead_sources.txt`/`10_enrich_manifest.py`)✓ |
| Repo 總大細 | 24G → **17G**(四 phase 合共回收 ~10.9GB)|
| `metadata/logs/` | 1.7GB → **17M** ✓ |
| `pytest tests/` | **288 passed, 1 failed** — failure 係 snapshot-drift flake,見新發現 N2 |
| `pipe catalog verify` | 7/7 實質 check PASS;10 個 `row_count` FAIL 全屬 stale baseline,見新發現 N1 |

### §6.2 §2 Issues Register 逐項 disposition

| # | Issue | Disposition |
|---|---|---|
| 1 | `reconstruct.py` 公開(HC#9) | ✅ **RESOLVED** — `420981f` git rm + push;dormant 本地留底 |
| 2 | 預設 manifest stale(2026-07-09) | ⏳ **OPEN** — 仍未 re-export;而家 ingest 又跑緊,建議等今輪 ingest→downstream 消化完先一次過 export |
| 3 | `report.build` 未 port | ✅ **RESOLVED**(同日晚間)— 新 node 讀 live catalog,12 項 Acceptance Criteria 全查;`scripts/10_report.py` git rm,`scripts/` 歸零;見 `pending_task.md` T4 |
| 4 | filter/tier re-evaluation 缺口 | ⏳ **OPEN**(P6 候選;現有數據已由兩次 backfill 抹平)|
| 5 | `conn=` injection 10/22 | ⏳ **OPEN**(機械式 follow-up)|
| 6 | `requirements.txt` 過時危險 | ✅ **RESOLVED** — git rm + README 改 `uv pip install -e .` |
| 7 | `data/` 死 symlink / 誤導 symlink | ✅ **RESOLVED**(Phase A)|
| 8 | `10_enrich_manifest.py` 公開(HC#9) | ✅ **RESOLVED**(同 #1 一齊處理)|
| 9 | `.venv_ina/` 6.8GB | ✅ **RESOLVED**(Phase D,刪前零-reference 再確認)|
| 10 | `metadata/` 舊 backup 2.26GB | ✅ **RESOLVED**(Phase B1)|
| 11 | `metadata/logs/` 1.7GB | ✅ **RESOLVED**(Phase B2,現 17M)— retention 自動化(logrotate)未做,降級做 nice-to-have |
| 12 | Repo root 雜物 | ✅ **RESOLVED**(Phase A)|
| 13 | 文檔漂移 | ✅ **RESOLVED**(Phase C4 + 同日晚間再同步一次 CLAUDE.md/README);`metadata/DATASET_REPORT.md` 已隨 #3 一齊解決,而家係 live-generated |
| 14 | 8-commit push backlog | ✅ **RESOLVED** — `c5fb416..16bdd96` 已 push,0 ahead |
| 15 | 3 個 pilot QA batch 未 review | ⏳ **OPEN** — owner 人手動作,`pipe calibrate serve` 已 live |
| 16 | dormant release 數據搬位 | ⏳ **OPEN(可選)** — `manifest_release.jsonl`(672MB)+`excluded_no_url.jsonl` 仍喺 `metadata/` root;scripts 已入 `release_dormant/`,數據檔未搬。純意圖清晰化,無風險差異 |

**計數(同日晚間再更新)**:16 項入面 **11 項 RESOLVED**(包括兩項 High 全清 + Issue #3/#13),
**5 項 OPEN**(全部 Medium 以下或 owner-action/可選)。

### §6.3 新發現(round-2)

**N1 — `catalog verify` 嘅 golden baseline 全面過期(Medium)— ✅ RESOLVED(同日晚間)**
10 個 `row_count[*]` check 嘅 expected 值停留喺 P0 import 時代(455,299),而 corpus
已增長到 618,695 segments — 結果每次 verify 都「10/17 FAILED — P0 gate not met」。
呢啲 FAIL 而家係純噪音,但**長期會造成 alarm fatigue**:如果將來有真嘅 row 流失,
會匿埋喺十個「照舊 FAIL」入面冇人發現。**修法**:`pipeline/catalog/verify.py` 改成
floor(+ceiling)語義,跟 `tests/test_catalog.py` 已有嘅 `*_monotonic_growth` 慣例 ——
`row_count` 而家 17/17 PASS。見 `pending_task.md` T2。

**N2 — Snapshot-hardcode 測試對 live session 天然 flaky(Low)— ✅ RESOLVED(同日晚間)**
`tests/test_catalog.py::test_manifest_build_matches_expected_corpus_totals` 將
count/n_speakers/gold 寫死做常數(458843/8817/43)。今日兩次驗證分別見到 gold 43→49、
count 458843→458844 — 全部係 live `calibrate serve` review 嘅預期副作用(test docstring
自己都寫明)。docstring 有解釋唔等於 test 唔紅:CI/驗證輪一紅就要人手判斷,同 N1 係
同一種 alarm-fatigue 問題。**修法**:改做 tolerant assertion(count/n_speakers 用
floor+合理上限;gold 淨係 floor;auto_gold/silver/bronze 用 ±1000 容錯窗)——full
suite 而家 304 passed。見 `pending_task.md` T3。

**N3 — Ingest 重新活躍,§1 snapshot 開始過期(Info)**
兩個 `ingest.download`(podcast/youtube)由 13:55 跑到而家,`ingest_download_staging.jsonl`
持續增長中 — 新 raw files 落地後會排隊入 lang_screen→segment→ASR downstream。§1 嘅
618,695/458,843 等數字會繼續變;Issue #2 嘅 manifest re-export 建議等呢輪 ingest 消化完
先做,一次過收割。

### §6.4 建議後續優先次序(供 owner 決定,唔屬本 cleanup 範圍)

> 更新(同日晚間):N1/N2(Tier-1 假警報)同 Issue #3(`report.build`)已經全部解決 ——
> 詳細任務清單搬咗去 `pending_task.md`(git-tracked,完成一項就要更新嗰個檔案)。
> 以下係 `pending_task.md` 度嘅建議次序,呢度淨係摘要:

1. **Pilot QA batch review**(Issue #15 / `pending_task.md` T1)— 人手動作,決定
   auto_gold/silver/bronze 閾值係咪企得住,直接影響之後所有 export 嘅可信度。**而家
   優先度最高嘅一項**——`report.build` 已經證實 458,844 個 manifest-eligible entries
   入面得 58 個(0.0%)係人手驗證,10/11 acceptance criteria PASS 但 `text_verified`
   FAIL,呢個係下一步要解決嘅缺口。
2. **今輪 ingest 消化完後重跑 `manifest.export`**(Issue #2 / T6)— 順手清埋 N3,
   再跑一次 `report.build` 攞新數。
3. P6 排程時帶埋:Issue #4(re-eval 機制,T5)、Issue #5(conn= 收尾,T8)、§5 嘅
   speaker-centroid 剪枝(T9)+ content-hash linkage(T10)。
