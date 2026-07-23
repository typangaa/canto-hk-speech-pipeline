# Pause Token + 標點符號 Handling — Implementation Plan

> **狀態**:P0-P4 全部 done(2026-07-22,見 DECISIONS.md)——29 段人手聽 QC → 發現
> v1 閾值卡喺感知死區 → **calibration v2 已凍結**(no_pause 0.08→0.16s,
> long 0.35→0.48s)→ 全量 corpus-wide `pause_plan` reprocess 已完成(279,348 段
> 重算,verdict 分佈按新閾值重新分桶)→ `manifest.export`/`label.store` 已重新
> export → `catalog verify` 17/17 PASS。P4 全流程完結。**P5(交返 canto-tts)已完成
> (2026-07-23,自助式,非正式交接動作)**——canto-tts 嗰邊直接讀呢個 repo 匯出嘅
> `metadata/train.jsonl`/`val.jsonl`(`text_pause` 欄位)+ `pause_calibration.json`,
> `core/control_schema.py` 抄低嘅 `calibration_version`/`git_rev` 同呢邊實際版本完全
> 對得上(`9b73455`,v2),`convert_corpus_to_moss.py --insert-pause-tokens` 已跑完
> (628.6h train / 5.4h val,pause marker 對齊率 99.96%),GPU encode 進行緊。**§0 嘅
> vad_cut 結構性發現已核實係過時/唔適用**(2026-07-23 更正,見下方 §0 備註同
> DECISIONS.md)——`<pause-long>` 喺實際訓練數據入面樣本充足(189,432 個 token,
> 佔 short+long 總數 23.8%),唔存在原先擔心嘅結構性稀缺,冇嘢要再交低。
> **緣起**:`canto-tts/docs/PAUSE_TOKEN_CALIBRATION_HANDOFF.md`(engine 側已備好
> `<pause-short>`/`<pause-long>` vocab,等 pipeline 出數據)
> **先讀**:`docs/LABEL_FRAMEWORK_SPEC.md` §8.3(pause raw 層已落地)、`docs/MANIFEST_SCHEMA.md`
> **Owner 拍板(2026-07-21,四項)**:①標點做 audit + 新欄位(canonical text 不動);
> ②bucket 閾值實測先、再同 canto-tts 協調(唔硬套 handoff 草擬嘅 0.3/0.6s);
> ③輸出「pause plan(SSOT)+ 已插 token 文本(convenience)」兩樣;
> ④aligner scope = gold+auto_gold 先。

---

## 0. 實測發現(2026-07-21,本 plan 嘅根據)

以下全部由 catalog 實數量度,唔係估:

| 發現 | 數字 | 後果 |
|---|---|---|
| gold+auto_gold 有句中標點 | 90.2%(樣本 n=4,589) | 標點位充足,handoff v2.0「只喺標點位插」策略可行 |
| 標點 inventory | ，。、：？!為主,half-width `.` 僅 15 例 | normalization 負擔輕(§4) |
| VAD gap 數 == 句中標點數 | **只有 12.2%**(n=11,611),偏差以 gap 少過標點為主 | **順序對應法不可行 → 必須 forced alignment**(§2) |
| within-segment gap 分佈 | p50=0.23s、**p90=0.26s**;平均每段僅 ~0.15 個 gap | `segment.vad_cut` 用 `min_silence_duration_ms=300` 切段,≥0.3s 停頓已成 segment 邊界——**handoff 草擬嘅 short=0.3–0.6/long>0.6s 喺 segment 內幾乎零樣本** |
| labels_prosody 覆蓋(gold+auto_gold) | 95.2%(265,930/279,348) | handoff §2.1(VAD 抽 gap)已完成,免重做 |

**點解要 forced alignment**:冇 char-level timestamp,只知 gap 嘅次序、唔知邊個 gap
屬於邊個標點;上表 12.2% 就係順序對應法嘅命中率上限。Alignment 令每個字有時間戳,
標點處嘅實際停頓 Δt = `start(下一字) − end(上一字)`,直接可量,仲順便審計到
qwen3_asr 憑 LM 亂加、聲學上冇停頓嘅標點(industry 已知 zero-length comma 問題)。

> **2026-07-23 更正**:上表「within-segment gap 分佈」一行(VAD 自己嘅 gap 偵測,
> p90=0.26s)當初俾人誤讀成「最終 pause_plan 嘅 long 分桶會近乎冧樣本」,但呢個
> 只係「點解要轉用 forced alignment」嘅論證(VAD gap 本身量錯嘢,同標點對唔上),
> **唔係最終數據嘅真實分佈**——forced-alignment 量出嚟嘅 Δt 同 VAD 自己嘅 silence
> gap 係兩件唔同嘢(VAD 靠 speech-probability threshold 判斷,aligner 靠逐字時間戳),
> 兩者唔對應。核實實際 `pause_plan` 表:long 有 95,835 個(佔 646,001 個已分類標點
> 嘅 14.8%),77,220/279,348(27.6%)個 segment 至少有一個 long。canto-tts 實際
> encode 出嚟嘅 `data/v7_pause_gold_full/v7_pause_train.jsonl`:`<pause-short>`
> 607,162 個、`<pause-long>` 189,432 個(long 佔 short+long 總數 23.8%)——樣本量
> 健康,唔存在結構性稀缺。呢個更正推翻咗之前(同一日)喺 DECISIONS.md/PROGRESS.md
> 寫低「要交低呢個發現畀 canto-tts」嘅講法,冇嘢要交,詳見 DECISIONS.md 2026-07-23
> 嘅更正 entry。

**Industry 參照**(詳見 agy-gemini 報告 + arXiv 2302.13652 / 2604.21164):
explicit break token 流派(NaturalSpeech 2/3、CosyVoice 2)慣用
short≈80–350ms / long>350ms(prosodic hierarchy B2/B3),同我哋實測分佈範圍吻合;
reconciliation 三段式:Δt<80ms → 標點無聲學根據;80–350ms → short;≥350ms → long。
呢啲係 P2 calibration 嘅 prior,唔係最終數——最終數由 P2 實測 + owner 拍板凍結。

---

## 1. 職責邊界(對齊 LABEL_FRAMEWORK_SPEC + handoff 折衷)

- 本 repo 出:**pause plan**(char offset + Δt + bucket,raw 永久保留、model-agnostic,
  SSOT)+ **已插 `<pause-*>` token 嘅文本欄位**(convenience view,canto-tts 直接食)。
- `<pause-short>`/`<pause-long>` 係 literal vocab token,插入原文後 canto-tts renderer
  原樣保留——所以插 token 呢步可以喺 upstream 做,唔違反「tokenizer 留 downstream」
  嘅原意(我哋插嘅係字面 marker,唔係 phoneme token)。
- canto-tts 想改閾值 → 由 plan 嘅 raw Δt 自行 re-bucket,唔使重跑 audio。

---

## 2. Phases

### P0 — `align.chars` node(新 DAG node,GPU)

- **模型**:`Qwen/Qwen3-ForcedAligner-0.6B-hf`(Apache-2.0,原生 Cantonese,
  word-level timestamp,≤5min 音頻)。⚠ 未入 transformers 正式版,要從 source 裝
  (**`uv pip install`,絕不 `uv sync`** — CUDA torch prune 風險,CLAUDE.md 鐵律)。
- **Node 形態**:跟 `asr.py` 嘅 GPUWorkerBase + JSONL worker-subprocess pattern;
  16kHz transient 重採樣(soxr HQ,同 asr.py 一致)。
- **Discovery**:gold+auto_gold ∩ `asr_agreement.best_text` 非空,anti-join
  `alignments.provenance = 'qwen3_aligner'`(唔係 bare row-existence)。
- **輸入文本**:`asr_agreement.best_text`(gold 段經 calibrate serve 驗證後亦係呢欄)。
  標點先 strip 再餵 aligner(aligner 只對齊可發音 token),但**保留 char↔原文 offset
  映射**,對齊結果寫返原文座標。
- **寫**:新表 `alignments`:`id`, `chars`(JSON `[[char, start_sec, end_sec], ...]`,
  原文 offset 對齊),`model`, `provenance`。
- **GPU 紀律**:同其他 GPU model sequentially-exclusive(CLAUDE.md caveat 2);
  兩卡 shard 跟 `shard_rows_round_robin` 現成 pattern。
- **Pilot 先行**:`--limit 200` 實跑,人手抽 10 段核對 char timestamp 合理性
  (特別係 code-switching 段 + word-level 輸出對中文係咪逐字),先至開全量。
- Node conventions checklist 全套:`conn=None` 注入、`RUN_MANY_ADAPTERS` 註冊、
  `tests/test_run_many.py` regression、`metadata/logs/align_chars.log`。

### P1 — 標點-聲學統計 + calibration(一次性,CPU)

- 新 node `pause.calibrate`(或 `label.calibrate` 加 section——實現時揀,傾向獨立):
  join `alignments` + `best_text` 標點位置 + `labels_prosody.gaps`,對每個句中標點
  (，。？!、;:)計 Δt;句尾標點另計(段尾 silence 已被 vad_cut trim,預期無意義,
  用數據證實)。
- 輸出報告:per-mark Δt 分佈(p25/50/75/90)、無聲學根據標點比例、VAD gap 同
  aligner gap 嘅互相印證率(兩個獨立信號 sanity check)。
- 寫 `metadata/labels/pause_calibration.json`(versioned:date + git rev + n;
  跟 LABEL_FRAMEWORK_SPEC §9 慣例)。
- **Gate(人手)**:owner 攞住實測分佈同 canto-tts 協調 bucket 定義
  (預期方向:no-pause <80ms / short 80–~250ms / long >~250ms,或 short-only;
  由數據話事)。**閾值一凍結就永不再郁**(handoff §2.3 鐵律)——凍結值同時抄一份
  俾 canto-tts 更新 `core/control_schema.py` comment。

### P2 — `pause.plan` node(reconciliation + plan 產出,CPU)

- 對每段:每個句中標點 → `{offset, mark, delta_t, verdict}`,
  verdict ∈ `no_pause`(<下限,即 LM 加嘅、聲學無根據)/ `short` / `long`。
  非標點位嘅 gap 記入 `unpunctuated_gaps`(v2.0 唔 tokenize,留 v2.2,同 handoff §5.1 一致)。
- 寫新表 `pause_plan`:`id`, `plan`(JSON), `n_punct`, `n_no_pause`, `n_short`,
  `n_long`, `calibration_version`, `provenance = 'pause_plan'`。
- 依賴凍結咗嘅 `pause_calibration.json`——P1 gate 未過唔准跑全量。

### P3 — 輸出欄位(label.store + manifest.export 擴展)

- `labels.jsonl` 嘅 `control.pause` 由而家淨 `gaps` 擴展為:
  `{"gaps": [...], "plan": [...], "calibration_version": ...}`(additive,舊 key 不動)。
- manifest export 加兩個 additive 欄位(canonical `text` 絕不改——owner 拍板①):
  - `text_pause`:原文 + 喺 verdict=short/long 嘅標點後插 `<pause-short>`/`<pause-long>`
    literal token;verdict=no_pause 嘅句中標點喺**呢個欄位入面** strip 走
    (reconciled 變體,原 `text` 保留方便 A/B——正正係 handoff §2.4「唔好覆蓋」要求)。
  - `punct_audit`:`{n_punct, n_no_pause, n_short, n_long}` 摘要,俾人一眼睇到
    邊啲段標點聲學根據弱。
- `--min-tier` 等現有 flag 照用;pause 欄位只喺 gold+auto_gold 有值,其他 tier omit
  (LABEL_FRAMEWORK_SPEC §7「唔可靠就 omit,唔寫 null」慣例)。

### P4 — QC(handoff §4,落貨前必做)

- 擴充 `pipe calibrate serve`:加 pause-preview 模式——顯示 gap/標點 marker 時間軸 +
  `text_pause` 渲染 + 播放;沿用現有 review 工作流(owner 拍板嘅 QC 方式)。
- 人手聽 ≥30 段:核對 ①token 位置對唔對得上實際停頓;②short/long 聽感分唔分得開;
  ③被 strip 嘅 no_pause 標點係咪真係冇停頓。
- QC 唔過 → 返 P1 重議閾值(凍結前唯一容許回頭嘅位)。

### P5 — Handoff 返 canto-tts — 完成(2026-07-23,自助式)

- 交:`labels.jsonl`(pause plan)+ 含 `text_pause` 嘅 manifest export +
  `pause_calibration.json` 抄本(佢哋更新 `core/control_schema.py` comment)。**已達成**
  ——但唔係經一個正式「交貨」步驟,而係 canto-tts 嗰邊(2026-07-23)直接指向呢個
  repo 匯出嘅 `metadata/train.jsonl`/`val.jsonl` 檔案讀取(sibling repo,同一
  filesystem),`control_schema.py` 已抄低啱嘅 `calibration_version: 2026-07-22-
  9b73455-v2` / `git_rev: 9b73455`,同 `metadata/labels/pause_calibration.json`
  實際值一致。`convert_corpus_to_moss.py --insert-pause-tokens` 已跑出
  `data/v7_pause_gold_full/`(628.6h train / 5.4h val,pause marker 對齊率
  99.96%,0% OOV),GPU encode 2026-07-23 進行緊。
- 一併交:§0 嘅 pause-long 發現。**核實後(2026-07-23)冇嘢要交**——§0 表入面
  「within-segment gap p90=0.26s → long 幾乎零樣本」呢句,量嘅係 VAD 自己嘅
  silence-gap 偵測,唔係最終 `pause_plan` 用 forced alignment 量出嚟嘅 Δt(兩者
  唔係同一件事,§0 本身都解釋咗點解要棄用 VAD gap 改用 forced alignment)。核實
  實際 `pause_plan`:long 佔已分類標點 14.8%(95,835/646,001),27.6% segment
  至少有一個;canto-tts 實際 encode 出嚟嘅數據 `<pause-long>` token 189,432 個
  (佔 short+long 總數 23.8%)——樣本量健康,冇結構性稀缺,見 §0 更正備註。

---

## 3. 標點符號 handling 規格(audit + 新欄位,canonical 不動)

1. **Normalization(comparison/plan 用,唔改 stored text)**:half-width `,.?!;:` →
   full-width ，。?!;:;`…`/`⋯` 統一;「」《》引號書名號**唔插 pause、唔 strip**
   (非停頓性標點);此規則寫做共用 helper,pause.plan 同 calibrate UI 共用。
2. **驗證現況記錄在案**:標點 100% 來自 qwen3_asr LM 推斷(sense_voice CTC 冇標點;
   `asr.agreement` strip 標點後先比較)——即係標點從無 cross-model 驗證。本 plan 嘅
   reconciliation(P2 verdict)就係首次為標點提供聲學驗證信號。
3. **Gold 段**:calibrate serve 人手改過嘅 text 標點視為可信,但 Δt audit 照做
  (人都會留低 ASR 原標點唔改)。
4. **唔做**(明確 out of scope):標點 restoration model(ct-punc/BERT——我哋標點
   覆蓋率 90%+,問題係「多咗假標點」唔係「冇標點」);sense_voice 標點補全;
   canonical text 改寫(owner 拍板否決)。

---

## 4. 風險 / 已知未知

| 風險 | 處理 |
|---|---|
| Qwen3-ForcedAligner 未入 transformers 正式版,source 安裝或同現有 pin 衝突 | P0 pilot 前先喺 `.venv` 試裝 + `python -c` smoke test;裝法記入 DECISIONS.md |
| aligner 係 word-level,中文「word」granularity 未實證係逐字 | P0 pilot 10 段人手核對;唔係逐字就用 word 邊界近似(標點只需要前後邊界,夠用) |
| 段尾標點(99.4% 段有)Δt 無意義(段尾 silence 被 trim) | P1 分開統計證實;`text_pause` 段尾標點保留但唔插 token(除非 P1 數據話仲有殘餘 silence) |
| bucket 閾值同 engine 兩 token 設計唔夾(可能得 short 有量) | P1 gate 帶數據協調,必要時 engine 接受 short-only(佢哋 §5.2 已預咗類似情況) |
| gold 只有 189 段,auto_gold 279k 係統計信心而非人手驗證 | calibration 用 gold+auto_gold(owner 拍板③);QC 抽樣覆蓋兩者 |

---

## 5. 執行順序 TL;DR

```
P0 align.chars pilot(--limit 200 + 人手核對)
  → P0 全量(gold+auto_gold,~279k 段,GPU 數粒鐘)
  → P1 統計 + calibration report → 【人手 gate:owner + canto-tts 定閾值,凍結】
  → P2 pause.plan 全量
  → P3 label.store / manifest 擴展
  → P4 calibrate serve pause-preview + 人手聽 ≥30 段 → 【QC gate】
  → P5 交貨 canto-tts —— 完成(2026-07-23,自助式;§0 vad_cut 發現核實過時,見上)
```
