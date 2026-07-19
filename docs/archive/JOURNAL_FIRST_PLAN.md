# Journal-First Catalog Write Path — Implementation Plan

> ⚠ **2026-07-06 update**:呢份文件 §「觸發原因」嘅前提(「任何兩個 node 都冇得真正並行,
> 淨係得順序 chain」)已經證實錯——DuckDB 嘅 single-writer 限制係 per-**process**,唔係
> per-transaction。已經起咗一個更輕量嘅方案(`pipe run-many`,見
> `docs/ORCHESTRATOR_PLAN.md`):幾個 node coroutine 喺**同一個** process 入面共用一條
> connection(每個 node 攞自己嘅 `conn.cursor()`),asyncio 協作式排程令寫入零衝突,已經
> 喺 live catalog 實測 `label.music`(GPU)+ `filter.acoustic`(CPU)成功並行。呢份
> journal-first 設計(獨立 journal 檔 + 定期 replay 落 catalog)仍然係一個更重型、更適合
> 「跨機器/跨 process 都要並行」場景嘅方案,但今日嘅並行需求已經用 run-many 解決咗,
> 唔再係迫切——留低呢份文件供將來需要更強隔離性(例如真係要跨獨立 process,而唔止喺
> 一個 supervisor 入面)先再評估。
>
> **狀態**:草擬,等 owner 拍板先執行(2026-07-03)
> **上游文件**:`docs/archive/REARCHITECTURE_IMPLEMENTATION_PLAN_DESIGN_DETAIL.md` §3.2(「Journal + catalog」已經畫咗個
> 願景,呢份文件將佢寫成可執行嘅具體設計 + milestone + gate)
> **觸發原因**:P2 backlog(2026-07-03)證實咗 DuckDB single-writer 唔止擋 GPU-vs-GPU,
> 連 GPU-vs-CPU、任何兩個 `pipe run <node>` process 都唔可以同時攞 RW connection —— 依家
> 每個 node 嘅 supervisor 都自己攬住一條 RW connection 成個 run 咁耐(可以係幾個鐘),
> 令任何兩個 node 都冇得真正並行,淨係得順序 chain(P2 backlog 三個 node 行咗 ~90 分鐘,
> 如果可以並行,實際 wall-clock 應該少過一半)。

---

## 0. TL;DR

- **問題唔喺 DuckDB 本身,喺我哋點用佢**:現時每個 node(`label.music`/`label.prosody`/
  `label.suite`/`ingest.probe`/新嘅 `asr.transcribe`)嘅 supervisor 一開波就 `connect()`
  攞一條 RW connection,然後成個 run(discover → dispatch → 逐 batch upsert)都攬住佢唔放。
  DuckDB 嘅 lock 規則好簡單但好硬:**RW connection = exclusive lock,擋晒所有其他
  connection(唔理 RW 定 RO);RO connection 之間可以互相並存,但唔可以同一條 RW 並存**。
  攬住成個 run 咁耐,就等於將一條理論上得幾條 SQL(discover + N 次 batch upsert)嘅 lock
  window,人為擴大到成個 run 嘅 wall-clock。
- **修法 = 將「寫結果」同「寫入 catalog」分開兩件事**:node 唔再直接寫 DuckDB,改為
  append 落一個 per-run 嘅 append-only journal 檔案(冇 lock 問題,幾多個 process 都得,
  因為每個 run 有自己獨立嘅檔案);一個獨立、短命嘅 **compactor**(可以係 daemon 定
  on-demand)先係唯一會開 RW connection 嘅嘢,而且每次只開好短時間(一個 batch 嘅
  upsert,毫秒到秒級)。
- **代價 = 輕微、有上限嘅「重做」風險,冇損壞風險**:kill -9 或者 compactor 未追上,
  最多令上次 compaction 之後嗰批(通常幾十到幾百項)喺 restart 之後重新被 discover 一次
  ——因為所有寫入都係 upsert-by-primary-key,重複執行絕對唔會令數據錯,淨係浪費少少
  GPU/CPU 時間。呢個同現有 pipeline「crash-safe、resumable、永不 corrupt」嘅慣例完全一致
  (同 `04_transcribe.py` 個 checkpoint、`labels_music` 嘅 idempotent upsert 係同一種諗法)。

---

## 1. 現狀問題(2026-07-03 P2 backlog 實測確認)

### 1.1 邊個攬住 lock 幾耐

睇 `pipeline/nodes/label_prosody.py::run_label_prosody()`(其他 node 一樣嘅 pattern):

```python
async def run_label_prosody(...):
    conn = connect()                      # <- RW connection 開喺呢度
    rows = discover(conn)                 # 用佢
    ...
    async def worker_loop(...):
        ...
        upsert_rows(conn, "labels_prosody", out_rows, ["id"])   # <- 又用佢
        record_batch(conn, run_id, "label.prosody", ...)        # <- 又用佢
    await asyncio.gather(*(worker_loop(...) for ...))
    ...
    # conn 由呢個 function 開波用到尾,冇 explicit close,靠 process exit 先釋放
```

一個 455,179 項嘅 backlog 跑 ~76 分鐘,即係呢條 RW connection 攬咗 lock 攬咗 76 分鐘 ——
中間任何一個其他 node(`ingest.probe`)、任何一個 `pipe catalog verify`、甚至 `pytest`
入面掂到 catalog 嘅 test,通通連唔到。

### 1.2 點解唔可以淨係「郁少少」就修好

有冇可能淨係將 `conn = connect()` 挪去 loop 入面,每次 upsert 先開,寫完即刻關?
技術上得,但**代價太大**:DuckDB 每次 `connect()` 都要重新 `init_schema()`(等於重跑
成份 schema.sql 嘅 DDL,192 行 CREATE TABLE IF NOT EXISTS/INDEX),仲要每次都要
attempt 攞 exclusive lock —— 如果兩個 node 啱啱好交替寫緊,會出現持續嘅
lock-acquire-retry,可能仲衰過而家(一開波就攬到尾)。呢個唔係真修法,只不過將
lock contention 嘅頻率同粒度掉轉,冇解決「兩個 node 想真正並行跑」呢個核心問題。

**真正嘅答案係:node 唔應該直接接觸 DuckDB 嘅 RW 通道** —— 呢個先係 §3.2 原本諗嘅
journal-first 設計要解決嘅嘢。

---

## 2. 目標架構

```
┌─ node 1(GPU,label.suite)──┐   ┌─ node 2(CPU,label.prosody)──┐   ┌─ node 3(asr)──┐
│ discover(): connect_ro()   │   │ discover(): connect_ro()     │   │ ...            │
│   短開短close,讀完即放     │   │   短開短close,讀完即放       │   │                │
│ dispatch worker批次...      │   │ dispatch worker批次...        │   │                │
│ 每個 batch 完:              │   │ 每個 batch 完:                │   │                │
│   journal_append(run_id,    │   │   journal_append(run_id,      │   │                │
│     "labels_music", rows)   │   │     "labels_prosody", rows)   │   │                │
└──────────┬───────────────────┘   └──────────┬─────────────────────┘   └───────┬────────┘
           │ metadata/journal/label.suite/<run_id>.jsonl                        │
           │                metadata/journal/label.prosody/<run_id>.jsonl       │
           └───────────────────────┬──────────────────────────────────────────┘
                                    ▼
                    ┌─ compactor(唯一 RW writer)────────────┐
                    │ 掃描所有 metadata/journal/*/*.jsonl     │
                    │ 逐個未 compact 嘅 offset 開始 replay    │
                    │ upsert_rows(conn, table, rows, pk)     │
                    │ 更新 journal_offsets watermark          │
                    │ 每次開 connect() 淨係開幾百毫秒         │
                    └──────────────────────────────────────┘
```

**核心原則**:node 嘅 hot loop(discover 之後、成個 dispatch 過程)完全唔接觸 RW
connection;唯一會開 RW 嘅係 compactor,而且佢每次開嘅時間應該同「一個 batch 嘅
upsert」成正比,唔係同「成個 run」成正比。

---

## 3. Journal 格式

**位置**:`metadata/journal/<node>/<run_id>.jsonl`(同 §3.2 原本諗嘅一致;`run_id` 由
`orchestrator/journal.py::new_run_id(node)` 產生,已經存在,唔使改)。

**每行一個 batch 嘅寫入指令**(自描述,compactor 唔使識個別 node 嘅邏輯,淨係識點
replay 一個 upsert):

```jsonc
{"table": "labels_prosody", "pk": ["id"], "rows": [{"id": "...", "rate_raw": 4.2, ...}, ...]}
{"table": "task_runs", "pk": ["run_id", "node", "item_id"], "rows": [{"run_id": "...", ...}]}
```

- 冇 fsync(同現有 checkpoint 慣例一致 —— OS write-back 夠,呢層本身就係「容忍少量
  重做」嘅設計,唔追求 zero-loss)。
- 一個 run 期間,`asr_results` 嘅結果同 `task_runs` 嘅記錄可以寫入**同一個** journal
  檔(用 `table` 欄位分辨),唔使開兩個檔——減少檔案數量。
- **Torn-write 處理**:kill -9 可能令最後一行寫到一半。Compactor replay 時,逐行
  `json.loads()`,遇到最後一行解唔到(`JSONDecodeError`)就當佢未寫完,跳過並且**唔
  推進 watermark 過嗰行**——下次 compact 再由嗰行嘅 byte offset 開始試(如果嗰陣已經
  寫完整咗就會成功;呢個係 append-only journal 標準做法,同 write-ahead-log 嘅
  torn-tail handling 一致)。

---

## 4. 新增/修改嘅組件

### 4.1 `pipeline/orchestrator/journal.py` 加 `journal_append()`

```python
def journal_append(run_id: str, node: str, table: str, pk: list[str], rows: list[dict]) -> None:
    """Append one journal line. No DB connection — pure file I/O, safe from any
    number of concurrent processes as long as each has its own run_id (guaranteed
    by new_run_id()'s uuid suffix)."""
    path = JOURNAL_DIR / node / f"{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"table": table, "pk": pk, "rows": rows}, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
```

`record_batch()` 嘅**簽名不變**,但內部實現改做 call `journal_append(..., "task_runs", ...)`
——呢個係關鍵嘅「一處改晒晒」位:所有 node 已經係透過 `record_batch()` 寫
`task_runs`,唔使逐個 node 改呢部分。

### 4.2 `pipeline/catalog/compact.py`(新)

```python
def compact_once() -> dict:
    """Scan all journal files, replay unapplied lines into DuckDB, advance watermarks.
    Opens exactly one short-lived RW connection for the whole pass."""
    conn = connect()
    applied = 0
    for journal_path in JOURNAL_DIR.glob("*/*.jsonl"):
        offset = get_watermark(conn, journal_path)      # from journal_offsets table
        with open(journal_path, "rb") as f:
            f.seek(offset)
            for raw_line in f:
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    break   # torn tail — stop here, retry next pass
                upsert_rows(conn, entry["table"], entry["rows"], entry["pk"])
                applied += len(entry["rows"])
            new_offset = f.tell()
        set_watermark(conn, journal_path, new_offset)
    return {"applied": applied}

async def watch(interval: float = 15.0) -> None:
    """Background daemon mode: compact_once() every `interval` seconds, forever."""
    while True:
        result = compact_once()
        if result["applied"]:
            log.info(f"compacted {result['applied']} rows")
        await asyncio.sleep(interval)
```

**Watermark 表**(加入 `schema.sql`):

```sql
CREATE TABLE IF NOT EXISTS journal_offsets (
    journal_path TEXT PRIMARY KEY,
    byte_offset  BIGINT,
    updated_at   TIMESTAMP
);
```

放喺 DuckDB 入面(唔係獨立檔案)嘅原因:同實際數據寫入係**同一個 transaction/connection
生命週期**,watermark 推進同數據落地天然一致,唔使諗額外嘅 crash-consistency 問題。

### 4.3 CLI

```
pipe catalog compact              # one-shot:應用晒所有 pending journal,然後退出
pipe catalog compact --watch [--interval 15]   # daemon 模式,長駐
```

Node 嘅 CLI 介面(`pipe run asr.transcribe` 等)**唔變**——改嘅只係內部寫入路徑,對外
行為一致(呢個係刻意嘅:唔想再逼你記多樣嘢,operator 淨係要記得「跑 node 之前/期間
起埋個 compactor」)。

### 4.4 Node 側改動(逐個 mechanical port)

每個 node 嘅 `run_*()` 改動 pattern 一致:

```diff
-    conn = connect()
-    rows = discover(conn)
+    conn = connect_ro()
+    rows = discover(conn)
+    conn.close()
     ...
-    upsert_rows(conn, "labels_prosody", out_rows, ["id"])
-    record_batch(conn, run_id, "label.prosody", [...], "ok", metrics=...)
+    journal_append(run_id, "label.prosody", "labels_prosody", ["id"], out_rows)
+    record_batch(run_id, "label.prosody", [...], "ok", metrics=...)   # 簽名少咗 conn
```

要改嘅檔案:`label_music.py`、`label_prosody.py`、`label_suite.py`、`ingest_probe.py`、
`asr.py`(5 個 node,每個改動幅度細但要逐個驗證)。

---

## 5. `discover()` 嘅 race 同埋佢冇解決嘅嘢(要講清楚)

### 5.1 解決咗嘅

- **唔同 node、唔同表**(例如 `label.prosody` vs `ingest.probe`)可以真係同時跑—— 佢哋
  各自嘅 journal 檔完全獨立,冇任何寫入時互相阻塞。
- **discover() 唔再攬住 lock 成個 run**——攞完就放,同 compactor 之間最多爭一嗰下
  (compactor 亦都刻意設計成短開短關,爭用窗口係毫秒到秒級,唔係分鐘級)。

### 5.2 冇解決咗嘅(要明確接受)

- **同一個 node 起兩次**(例如手快手慢喺兩個 terminal 都 `pipe run asr.transcribe`)
  仍然會令兩邊 discover() 睇到同一批未完成 item,兩邊都去處理——呢個唔會整壞數據
  (upsert-by-PK,最後寫嗰個贏,結果應該一樣),但會浪費一份運算。呢個唔係本計劃嘅
  範圍——如果想完全防呢個情況,要加一個好輕嘅 per-node file lock(例如
  `metadata/locks/<node>.lock`,`flock()` 拎唔到就即刻拒絕起第二個 instance),留返
  之後(可能 P4)先做,唔喺呢個 plan 嘅 MVP scope 之內。
- **Compaction 之後嘅可見度有 lag**(等於 `--interval` 嗰個數,建議 15-30s)—— 依賴
  「兩個 model 都寫低咗」先可以起步嘅 node(例如 `asr.agreement` 等 `asr.transcribe`
  兩個 model 嘅結果)理論上會慢 compaction 嗰個 interval 先睇到最新結果。對一個跑幾
  分鐘到幾個鐘嘅 backlog 嚟講,呢個 lag 完全唔重要;如果將來要做真正嘅
  「item-level、零延遲」pipelining(§5.1 原文講嘅嗰種),要再諗一層(例如 compactor
  每次 apply 完即刻觸發一次 downstream discover,或者縮短 interval)。

### 5.3 Resume / kill-9 語義嘅改變

**之前**(直接寫 DuckDB):discover() 睇到嘅永遠係「已經 commit 咗嘅最新狀態」,kill -9
之後 restart,最多重做緊做緊嗰個 batch。

**之後**(journal-first):discover() 睇到嘅係「上次 compaction 嗰刻嘅狀態」。如果 kill -9
發生喺「worker 完成咗 3 個 batch,但 compactor 仲未追到」嗰個窗口,restart 之後呢 3 個
batch 會被重新 discover、重新處理一次。**呢個係刻意接受嘅代價**:
- 冇損壞風險(idempotent upsert)。
- 浪費上限 = 一個 compaction interval 嘅工作量,對成個 backlog 嚟講可以忽略。
- 換嚟嘅係「node 之間唔再互相 lock 死」,呢個 trade-off 抵做。

要喺 `test_orchestrator.py` 嘅 kill-9 resume test 度更新呢個新語義(見 §7 gate)。

---

## 6. Migration 順序(唔係一步到位)

1. **核心 library**(1 session):`journal_append()`(加入 journal.py)、
   `pipeline/catalog/compact.py`(one-shot + watch 模式)、`journal_offsets` 表、
   CLI(`pipe catalog compact`)。全部自己寫(呢層係 crash-safety/正確性核心,唔delegate)。
2. **Pilot port:`ingest.probe`**(0.5 session)—— 揀佢係因為佢最細(CPU-only、
   `raw_probe` 表細、6,272 rows)、風險最低。Port 完做 §7 嘅完整 gate。
3. **Pilot gate 過咗先port 餘低 4 個**(`label.music`/`label.prosody`/`label.suite`/
   `asr.py`)—— pattern 同 pilot 一致,適合delegate 俾 `weir chat agy-sonnet`(mechanical
   find-replace 為主),但每個 apply 之後自己驗證(唔係 hard-constraint 但係
   crash-safety-critical,跟 P1/P2 一貫「delegate mechanical、自己 review」嘅做法)。
4. **更新 `test_orchestrator.py`** 反映新嘅 resume 語義(§5.3)。
5. **Retire 直寫 code path**、更新 memory/`docs/archive/REARCHITECTURE_IMPLEMENTATION_PLAN_DESIGN_DETAIL.md` §3.2
   標記做已執行,呢個 pattern 成為之後所有 P3/P4 新 node(`filter.*`/`g2p`/`speaker.*`/
   `segment.*`)嘅預設寫法。

**粗略時間估算**:核心 library 1 session,pilot + gate 0.5-1 session,餘低 4 個 port +
重新全套 gate ~1 session —— 總共 **2.5-3 session**,唔係一次過做完嘅嘢。

---

## 7. Gate(每個 milestone 要過)

- **Unit**:`journal_append()` 寫出嚟嘅每行係合法 JSON、欄位齊;`compact_once()` 喺一個
  假 journal 檔上 replay 出嚟嘅 DuckDB 內容同直接 `upsert_rows()` 一致;torn-tail(手動
  截斷最後一行)唔會令 compactor 中咗、watermark 唔會越過壞行。
  Duplicate-replay(watermark 冇推進、再 compact 多次)結果不變(idempotent)。
- **並行 smoke test**(直接驗證呢份 plan 想解決嘅問題):真係用 2 個 OS process 同時
  跑 2 個唔同 node(例如 pilot port 咗嘅 `ingest.probe` + 一個未 port 嘅舊 node 做對照
  組,或者等 2 個都 port 完之後跑 `label.prosody` + `ingest.probe`),確認**冇任何一方
  出現 `duckdb.IOException`**,compactor 開住 `--watch` 情況下兩邊結果都最終落到
  catalog。
- **Chaos**:kill -9 一個 node(喺佢 journal 寫咗幾個 batch,compactor 未追到嗰個窗口),
  restart,確認:(a) 冇 crash,(b) 冇數據錯(upsert 結果同無中斷跑一次一致),
  (c) 重做嘅項目數 ≈ 一個 compaction interval 嘅工作量,唔會攤大。
- **Golden parity**:pilot port 完嘅 `ingest.probe`(同之後全部 4 個)要同 port 之前嘅
  輸出完全一致——呢次改動淨係換咗寫入路徑,唔應該影響任何計算出嚟嘅數值。

---

## 8. 未來擴展(呢個 plan 嘅範圍以外,唔喺 MVP 做)

- **journal → Parquet compaction**(§3.2 原文第 3 點):大表(`labels_*`、`asr_results`)
  最終實體放 ZSTD Parquet partitions,DuckDB 淨係做 query engine——connect_ro() 都可以
  唔使開,直接 `read_parquet()`,徹底冇 lock。呢個係 P5/P6 規模級先需要嘅優化
  (455k rows 依家 DuckDB 已經夠快,唔急)。
- **同 node 重複起嘅 file lock 防護**(§5.2)。
- **compactor 觸發 downstream discover 嘅零延遲通知機制**(而唔係靠 poll interval)。

---

## Related
[[canto-corpus-rearchitecture]] · `docs/archive/REARCHITECTURE_IMPLEMENTATION_PLAN_DESIGN_DETAIL.md` §3.2 · `docs/PIPELINE_REARCHITECTURE_PLAN.md`
