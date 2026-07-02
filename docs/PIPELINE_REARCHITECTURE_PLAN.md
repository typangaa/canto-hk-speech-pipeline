# Pipeline Re-Architecture & Storage Migration Plan

> **狀態**:Draft for review · 2026-06-30 · **§3 Phase A + Phase B storage migration 已完成
> (2026-07-02)** — Drive1/2/4 ext4 化(Drive2/4)+ 遷移(Drive1)+ Drive3 reformat 全部做完,
> 見底部「§3 執行結果」。§4 pipeline re-architecture(decode-once/orchestrator/DuckDB)**未開始**。
> **先讀**:`PIPELINE_SPEC.md`、`LABEL_FRAMEWORK_SPEC.md`、`KNOWN_ISSUES.md`
> **目標**:重構 data-prep pipeline,**充分用盡 CPU / GPU / storage 頻寬**,解決現有 I/O-bound +
> CPU/GPU 爭用樽頸,並撐到 **5-10× 數據增長**(5-10k h filtered)同新 label 需求。
> **野心**:**完整 re-architecture** —— resource-aware scheduler + work queue + caching tier +
> cross-disk sharding。

---

## 0. TL;DR

- **樽頸根因(實測)**:(1) audio 喺 **NTFS3** SATA-SSD,Linux 每讀都食 CPU + 並發差(唔係 disk 慢);
  (2) **每個 detector 各自 full-pass 讀晒 corpus** → 5+ 次完整讀 455k wav;(3) 各 stage **單線程串行**
  read→compute→GPU,零 overlap;(4) GPU 同 training **硬爭** kernel。
- **三大改造**:
  1. **Storage**:Drive2/3/4 → **ext4**(除 Drive1 保 NTFS 做 Windows share),分**冷/暖/熱**三層,
     **nvme0n1 做熱 cache 層**;cross-disk sharding 令並行 I/O 唔互搶頻寬。
  2. **Decode-once fan-out**:單一 reader 解碼+resample **一次**,經記憶體 bus 餵**所有** CPU+GPU
     feature extractor → I/O 省 N 倍。
  3. **Resource-aware orchestrator**:DAG + 分資源池(IO / CPU / GPU),按 `nvidia-smi` + load 動態
     調節並發,**training 行緊時 GPU 工作自動讓路**。
- **遷移分兩階段**:**Phase A 即刻**(Drive2→ext4、Drive4→ext4,唔掂 training);**Phase B 等 v1_run_4_spk
  training 完**(~23h)+ 你授權搬 canto-tts/個人資料後,Drive3→ext4。
- **非破壞鐵律**:reformat = 毀滅性 → 每步**先確認有第二份 copy 至 format**;絕不喺得一份 copy 時 format。

---

## 1. 現狀硬件 / Storage 盤點

> 原始盤點(2026-06-30)保留喺 git history;下表係 **§3 遷移執行完之後嘅實測(2026-07-02)**。

| 裝置 | 媒體 | fs | 掛載 | 用量 | 內容 |
|---|---|---|---|---|---|
| `nvme0n1p1` | **NVMe** | **ext4** | `/` | 1.9T,795G used / 987G free | Ubuntu OS + `/home`(repo + `metadata/` 4.7G + `ct2_models` 2.9G + **`canto-tts` 18G,暫托**)|
| `nvme1n1` | NVMe | — | — | — | **Windows OS 碟 —— 禁止使用** |
| `sda1` Drive1 | SATA SSD | **ntfs3**(保留,Windows share) | `/mnt/Drive1` | 1.9T,578G used / 1.3T free | `Development/{AI-ML(非 corpus),_Archive,Games,Mobile,Robotics,Systems,Web}` 199G、`Personal`(Photo 主體)374G |
| `sdb1` Drive2 | SATA SSD | **ext4** | `/mnt/Drive2` | 1.9T,1.6T used / 244G free | `canto-corpus/data`(corpus **raw**,1.6T)|
| `sdc1` Drive3 | SATA SSD | **ext4**(2026-07-02 reformat) | `/mnt/Drive3` | 1.8T,28K used / 1.7T free | 空,得 `lost+found`,**等 §3 step 9 sharding 分配** |
| `sdd1` Drive4 | SATA SSD | **ext4** | `/mnt/Drive4` | 1.8T,1.4T used / 423G free | `canto/filtered`(569G)+ `canto/segments`(843G) |

**Footprint(du 實測 2026-07-02,遷移完成後)**

| 目錄 | size | 備註 |
|---|---|---|
| Drive2 `canto-corpus/data`(raw)| **1.6T** | 全部 corpus raw(唯一 copy),已 ext4 化 |
| Drive4 `canto/segments` | **843G** | 已 ext4 化 |
| Drive4 `canto/filtered` | **569G** | 已 ext4 化;`data/filtered`、`data/segments/{podcast,rthk,youtube}` symlink 指去呢度 |
| Drive1 `Development`(非 corpus:AI-ML 舊/deprecated 項目、Games/Mobile/Robotics/Systems/Web/_Archive) | 199G | NTFS,個人開發資料,同 pipeline 無關 |
| Drive1 `Personal`(Photo 為主) | 374G | NTFS,個人資料 |
| Drive3 | **0**(1.7T free) | ✅ 已 reformat ext4,等分配 sharding |
| nvme `canto-tts` | 18G | Active training,owner 暫托喺度(見 §3 執行結果),日後應搬返 Drive3 |
| nvme `metadata/` + `ct2_models` | 4.7G + 2.9G | jsonl + sidecar + ASR 模型 |

**關鍵約束 / 發現(2026-07-02 更新)**
- **Drive1 保 NTFS**(Windows 共享碟);free = **1.3T**,遠比原估寬鬆 —— 因為 corpus
  filtered/segments 最終冇留喺 Drive1(去咗 Drive4),加上 §3 執行時刪咗 ~630G 冗餘資料
  (`cantonese-tts-old`/`canto-corpus-bak`/`ComfyUI`/`gemma-hermes--tts`),原本擔心嘅「~98%
  爆滿」冇發生。
- **Drive3 已 reformat ext4**,而家淨係一個空碟(1.7T free),等 §3 step 9 sharding 分配。
- **canto-tts 暫托咗喺 nvme**(冇跟原 plan 去 Drive2/4)—— 同 §2「nvme 係容量緊絀 hot cache
  層」原則有張力,建議之後搬返 Drive3。
- nvme1n1(Windows)**完全唔掂**;nvme0n1 OS 碟(987G free)做 cache/hot tier,而家多咗
  `canto-tts` 18G 暫住,**仍未填爆但要留意唔好變相長住**。
- **⚠️ 5-10× 規模 = raw 容量爆煲**:25-50k h raw ≈ **16-17TB** > 4 碟總和(nvme 987G + Drive1
  1.3T + Drive2 244G + Drive3 1.7T + Drive4 423G ≈ 4.6T free,總容量 ~9.2T)→ raw 必須
  **transient 或壓縮**(見 §4.6)。呢個開放問題**未解決**。

---

## 2. 目標 Storage 佈局(三層 + sharding)

```
┌ HOT (nvme0n1 ext4, /)  —— 最快,容量細,OS 碟唔填爆 ──────────────┐
│  • metadata/ (jsonl, label store, calibration)                  │
│  • cache/ : decoded+resampled audio cache(LRU,可棄)            │
│            feature cache(mel / embeddings,可重算)              │
│  • 現處理 shard 嘅 working set                                   │
└─────────────────────────────────────────────────────────────────┘
┌ WARM (Drive2 + Drive3* + Drive4 → ext4) —— 快、Linux-native、主力 ┐
│  • corpus raw  (跨碟 sharded,並行讀唔互搶)                       │
│  • segments    (跨碟 sharded)                                    │
│  • filtered    (跨碟 sharded;active 訓練/標註讀呢度)            │
│   * Drive3 Phase B 先加入                                         │
└─────────────────────────────────────────────────────────────────┘
┌ COLD / SHARE (Drive1 保 NTFS) ──────────────────────────────────┐
│  • Windows 共享                                                  │
│  • 個人資料(Photo/Uni/_Archive/Personal 集中喺度)              │
│  • 冷備份 / release artifact(可選)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Sharding 原則**:同一類大資料(raw / segments / filtered)**按 hash(id) 分散落多個 ext4 暖碟**,令
N 個 worker 並行讀時打唔同物理碟 → 聚合頻寬 = N× 單碟,唔互相 seek-contend。佈局表(drive→shard 範圍)
寫入 `config/storage_layout.yaml`,pipeline 由佢解析路徑(唔再 hardcode 單一 symlink)。

---

## 3. Storage 遷移計劃(兩階段,非破壞)✅ **已完成(2026-07-02)**

> Phase A + Phase B 步驟 1–8 全部做完(見底部「§3 執行結果」)。step 9(三碟最終 sharding +
> `storage_layout.yaml`)順延做獨立跟進項,唔再擋住呢個 section 標記完成。

> **每步守則**:format 前該碟資料**必已有第二份 copy**(rsync `--checksum` 核對)至 format;
> 全程 `df` 監空間;先 `--dry-run`。

### Phase A —— 即刻(唔掂 training)
1. **個人資料集中**:`Photo`/`Uni`/`_Archive`(Drive3)+ 維持 `Personal`(Drive1)→ 全部歸 **Drive1**
   (NTFS share)。rsync + 核對 + 確認後刪源。（Drive3 個人資料搬走,為 Phase B 鋪路）
2. **Drive2 → ext4**:Drive2 空 → 直接 `mkfs.ext4`、掛載、設 fstab(UUID)。即得一個 **1.9T ext4 暖碟**。
3. **Drive4 raw → Drive2**:`canto-corpus`(~1.6T raw)rsync 去 Drive2(ext4)→ 核對。
4. **Drive4 → ext4**:確認 raw 已喺 Drive2 後 `mkfs.ext4` Drive4、掛載、fstab。
5. **rebalance**:raw 跨 Drive2+Drive4 sharding(按 §2);更新 `storage_layout.yaml` + 重指 symlink。
6. **segments/filtered**:由 Drive1(NTFS)逐步遷去 Drive2/4(ext4)暖層,Drive1 保留 NTFS 共享角色。
   (filtered 仲俾緊標註讀 → 遷移時用「複製→改 config→驗證→刪源」避免中斷)

### Phase B —— training 完 + 你授權後
7. **搬 canto-tts**:由你 / canto-tts agent 將 `Development/AI-ML/canto-tts` 搬走(暫去 Drive2/4 ext4 或
   nvme)。**我唔自行郁另一 agent 嘅 domain。**
8. **Drive3 → ext4**:確認 Drive3 已清(canto-tts + 個人資料 + raw 都有第二份)→ `mkfs.ext4`、掛載、fstab。
9. **三暖碟最終 sharding**:raw/segments/filtered 跨 Drive2+Drive3+Drive4 平均分佈;`storage_layout.yaml` 定案。

### 風險 / 回滾
- 每步保留源直到核對通過;`mkfs` 係**唯一不可逆**步,前面有 gate。
- fstab 用 **UUID**(SATA 重新枚舉 sd? 會變)→ 防開機掛錯碟。
- 全程 `nvidia-smi`/`ps` 確認唔影響 training(Phase A 完全唔掂 Drive3)。

---

### §3 執行結果(2026-07-02)

Phase A + Phase B 步驟 1–8 **已完成**,同原 plan 有幾點出入(記錄低俾之後 sharding 參考):

- **Drive1 crash 事故**:遷移期間(2026-07-01)Linux 連續 crash/reboot 5 次(疑同 NTFS3 driver
  高並發壓力有關,同 §0 TL;DR 根因診斷脗合),觸發 chkdsk 將 ~114k 個孤兒 corpus 檔案(31G)
  搬入 `Drive1/found.000`。核對(name+size 全量 + md5 抽樣)確認全部喺 Drive4 有齊份,已安全
  刪除。**教訓**:大量小檔案(corpus segments)喺 NTFS3 底下做 `rsync --remove-source-files`
  呢類重 I/O 操作,穩定性有風險,§4 decode-once/orchestrator 落地前呢類操作要更保守
  (分批、監察 dmesg/uptime)。
- **清出遠超預期嘅冗餘空間**(~630G,原 plan 冇估到):`cantonese-tts-old`(227G,legacy 已被
  `00_reingest.py` 完全消化)、`canto-corpus-bak`(180G,舊 pipeline 快照)、`ComfyUI`(138G,
  同 corpus 無關個人工具)、`gemma-hermes--tts`(2.4G)—— 呢啲**刪咗而唔係搬**,因為已核實
  同現行完整 corpus/資料完全冗餘。
- **Drive1 容量結果比原估寬鬆**:原 plan 擔心 Drive1 會去到 ~98% 滿;因為 §3-6(corpus
  filtered/segments 遷去 Drive2/4 而唔留 Drive1)加上上述刪除,Drive1 而家 ~30% used,遠有
  餘裕。
- **canto-tts 去咗 nvme 而唔係 Drive2/4**:owner 選擇搬去 `/home/typangaa/Documents`
  (nvme0n1,ext4)—— 比 Drive2/4 快,但同 §2 「nvme 係容量緊絀 hot cache 層」原則有張力;
  建議之後(canto-tts training 完 / 有空檔)搬返落而家已經 ext4 化嘅 Drive3。
- **Drive3 reformat 已完成**:`mkfs.ext4 /dev/sdc1`,UUID `06299a41-1cec-4a44-9e32-d37984aa1a44`,
  fstab 已更新(`ext4 defaults,nofail 0 2`),掛載核實(1.8T,得 `lost+found`)。

**未做**:§3 step 9(三碟 sharding + `config/storage_layout.yaml`)、整個 §4(decode-once fan-out
/ resource-aware orchestrator / DuckDB metadata)。

---

## 4. Pipeline Re-Architecture

### 4.1 Decode-once fan-out(I/O 省 N 倍)
現狀:`lang_id` / `overlap` / `music` / 未來 `prosody` / `emotion` **各自全量讀 wav** → 5+ 次完整讀。
目標:**一次讀+解碼+resample**,經記憶體 bus 餵所有 extractor。

```
reader pool ──(decoded 48k + 16k + 32k 變體,放 cache)──► feature bus ──► [lang][overlap][music][prosody][emotion]
   (IO-bound,           (resample 一次,           (各 extractor 訂閱所需取樣率;
    跨碟並行)             cache 落 nvme/RAM)          CPU 的 CPU 池、GPU 的 GPU 池)
```
- 一個 segment 嘅多取樣率變體(16k 俾 VAD/pyannote、32k 俾 PANNs、48k 原始)**解一次、共享**。
- decoded/resampled cache 落 **nvme0n1 `cache/`**(LRU)→ 重跑/加新 extractor 唔使再讀暖碟。

### 4.2 Resource-aware orchestrator(DAG + 資源池)
- **DAG**:stage 之間嘅依賴(download→segment→transcribe→filter→…→label→tier)用 DAG 描述;node = 對一
  shard 嘅一個 operation。支援**逐 item pipeline**(item A 喺 stage 3、item B 仲喺 stage 1,無 barrier)。
- **分資源池**(各自 backpressure 隊列):
  - **IO pool**:size = f(暖碟數);跨碟 sharding,避免同碟並發 seek-contend。
  - **CPU pool**:size 動態 = `nproc − 訓練dataloader預留 − headroom`;feature/resample/VAD。
  - **GPU pool**:size 動態 = 由 `nvidia-smi` 探可用;**training 行緊 → 縮細甚至暫停**(PANNs 教訓:
    重 GPU pass co-run active DDP 會搶 kernel)。
- **動態調節**:periodic sampler 讀 `nvidia-smi`(util/mem)+ `/proc/loadavg` + IO stat → 調各池並發;
  hard rule:**training 存在時 GPU 工作讓路**(可配置 yield / cap)。
- **prefetch / overlap**:每 worker 用 producer-consumer(已喺 `11_audio_tag.py` 驗證:length-sort +
  prefetch 令 4.7→33/s),IO 同 compute overlap。

### 4.3 Caching & 可重算
- **decoded cache**(nvme):同一 audio 多 extractor / 多次重跑唔重讀暖碟。
- **feature cache**:mel / embedding 等貴 feature cache 落 ext4;id-keyed,可棄可重算。
- **idempotent + resumable**:全部 op id-keyed sidecar,skip done(沿用現慣例)。

### 4.4 Metadata at scale(5-10×)
- 5-10k h → 數百萬 segment;JSONL 線性掃會慢。
- **label store / manifest 轉 Parquet + DuckDB**(或 LMDB)做隨機 join / 過濾;JSONL 保留做 interchange。
- id-keyed sidecar 寫入仍 append-friendly,定期 compact 入 columnar。

### 4.5 與 LABEL_FRAMEWORK 整合
- §4.1 feature bus 嘅輸出 = LABEL_FRAMEWORK 嘅 detector raw sidecar。
- decode-once 直接服務 `rate/pitch/pause`(共享 16k VAD pass)+ `music`(32k)+ `emotion`(GPU)。
- orchestrator 嘅 GPU yield policy = `emotion` / PANNs 「等 GPU free」規則嘅實作。

---

## 5. 與 training 協調(GPU policy)
- **硬規則**:`nvidia-smi` 偵測到 training compute proc → GPU pool **讓路**(可配 `yield`=暫停 /
  `cap`=限 1 卡細並發)。
- CPU 工作可 co-run 但**留 dataloader 核**(thread cap + 池上限,PANNs 教訓)。
- Storage Phase A 完全唔掂 Drive3(training 所在);Phase B 等 training 完。

---

## 6. 分階段落地

| 階段 | 內容 | gate / 風險 |
|---|---|---|
| **S0. 個人資料 → Drive1** | rsync Photo/Uni/_Archive → Drive1 + 核對 | 核對通過先刪源 |
| **S1. Drive2 → ext4** | 空碟 mkfs + fstab(UUID) | 非破壞(空碟) |
| **S2. Drive4 raw → Drive2 → Drive4 ext4** | 搬 raw、核對、format Drive4 | format 前確認第二 copy |
| **S3. storage_layout.yaml + 路徑層** | pipeline 由 config 解析路徑、sharding | 不再 hardcode symlink |
| **S4. decode-once feature bus** | 單 reader → 多 extractor;cache 落 nvme | 對住現有 detector 驗證一致 |
| **S5. orchestrator(DAG+資源池)** | scheduler + GPU yield + prefetch | 先 dry-run / 小 shard |
| **S6. metadata → Parquet/DuckDB** | columnar 隨機 join | 5-10× 規模準備 |
| **S7. (training 完)Drive3 → ext4** | 搬 canto-tts(你/agent)+ format + 最終 sharding | Phase B gate |

---

## 7. Open questions
1. Orchestrator 用**純 Python**(自家輕量 DAG/池,沿用現 stack)定引入**workflow engine**(Prefect/
   Dask/Ray)?— 建議自家輕量 Python(零外部依賴、貼合現 script),除非你想要 Ray 嘅分散式。
2. Metadata columnar 揀 **DuckDB**(SQL、零 server)定 **LMDB**(KV)?— 建議 DuckDB(join/過濾方便)。
3. decoded/feature cache size budget on nvme0n1(OS 碟 594G free)上限定幾多?LRU 淘汰策略?
4. filtered/segments 由 NTFS Drive1 遷去 ext4 暖碟 —— active 標註讀緊,遷移窗口點安排(複製→切 config→刪源)?
5. canto-tts 搬遷(Phase B)由你定 canto-tts agent 做?我只可讀,需你協調。
6. 5-10k h 嘅**新數據點嚟**(多 source download?)→ 影響 raw 容量規劃同 download stage 設計。
