# Source Guide — canto-hk-speech-pipeline

> How to research, evaluate, and add new HK Cantonese audio sources.
> All sources must be publicly accessible web audio/video that can be downloaded with yt-dlp or HTTP.
> No paid, restricted, or third-party dataset sources (see Hard Constraints in CLAUDE.md).

---

## Source Evaluation Criteria

Before adding a source, score it on these criteria:

| Criterion | Good | Marginal | Reject |
|-----------|------|----------|--------|
| Language | Clearly HK Cantonese (係/冇/喺) | Mixed HK/Guangzhou Cantonese | Mandarin or overseas Cantonese |
| Audio quality | Studio/professional mic | Decent room acoustic | Outdoor noise, heavy compression |
| Speaker diversity | Multiple different speakers | Same 1-3 speakers | Single speaker only |
| Content volume | > 50 episodes or > 20h | 5-50 episodes | < 5 episodes |
| Accessibility | No login, no geo-block | Minor geo-block (solvable with proxy) | Paywall or login required |
| Content style | Diverse (topics/registers) | Single domain | Very narrow |

A source scoring "Reject" on any criterion must not be added.

---

## Existing Configured Sources

### RTHK

RTHK (Radio Television Hong Kong) is the primary source. It publishes content on YouTube through multiple official channels.

**Why RTHK first**: Professional studio audio, diverse programs, Cantonese-native presenters, large back-catalogue, public broadcaster with clear public access intent.

**Main channels**:
- `@rthkhongkong` — general channel
- `@rthkradio1` — Radio 1 (news, current affairs)
- `@rthkradio2` — Radio 2 (popular culture)
- `@rthkradio3` — Radio 3 (English — do not use)
- `@rthkradio4` — Radio 4 (classical music — minimal speech)
- `@rthkradio5` — Radio 5 (Cantonese interest)

**Programs to prioritise** (diverse domain coverage):
| Program | Domain | Est. hours | Notes |
|---------|--------|-----------|-------|
| 鏗鏘集 | documentary | 200h+ | Long-form interviews, diverse speakers |
| 頭條新聞 | talk_show | 100h+ | Political satire, casual Cantonese |
| 財經快訊 | news | 50h+ | Financial news, formal |
| 創科新里程 | documentary | 30h (done) | Already collected in old dataset |
| 城市論壇 | talk_show | 80h+ | Public forum, many speakers |
| 一路走來 | documentary | 40h+ | Historical interviews |
| 香港故事 | documentary | 60h+ | Local interest stories |

See `sources/rthk_sources.yaml` for full program list with playlist URLs.

---

## How to Add a New Source

### Step 1: Evaluate

Use yt-dlp to quickly survey a channel without downloading:
```bash
# List first 10 videos from a channel
yt-dlp --flat-playlist --dump-json "https://www.youtube.com/@channelname" 2>/dev/null \
  | python3 -c "
import sys,json
for line in sys.stdin:
    v=json.loads(line)
    print(v.get('title','')[:60], '|', round(v.get('duration',0)/3600,1), 'h')
" | head -20
```

Listen to 2–3 episodes manually and check, if doing manual review:
1. Is it clearly HK Cantonese? (not Guangzhou, not Mandarin)
2. Is the audio clean? (no heavy background music, no severe reverb)
3. Are there multiple different speakers?

**Manual listening is optional, not required** (clarified 2026-07-20, see DECISIONS.md
same date). `status: evaluate` does not gate `ingest.download` — it is provenance
metadata only, not a hard pre-approval step. The three checks above are already covered
automatically, per raw file, source-blind, before any expensive GPU stage runs:
language purity by `lang_screen.auto` (before `segment.diarize`), audio cleanliness by
`pregate.snr` (before `asr.transcribe`), speaker count by `segment.diarize` itself. It's
fine to add a candidate straight at `status: "evaluate"` (or `"active"`, functionally
equivalent) and let the pipeline's own gates sort it out — manual listening is only worth
doing if you want an early sanity check before spending download bandwidth on a source
you suspect is bad.

### Step 2: Check yt-dlp compatibility

```bash
yt-dlp --simulate --dump-json "https://www.youtube.com/watch?v=SAMPLE_VIDEO_ID" | python3 -c "
import sys,json
v=json.loads(sys.stdin.read())
print('Title:', v['title'])
print('Duration:', v.get('duration'))
print('Language:', v.get('language'))
print('Subtitles:', list(v.get('subtitles',{}).keys()))
"
```

### Step 3: Create a YAML config entry

Add to the appropriate `sources/*.yaml` file. Use the exact schema below:

```yaml
# For YouTube channels:
- name: "Program Name in Chinese"
  url: "https://www.youtube.com/@channelname"
  type: "channel"           # or "playlist" if a specific playlist
  source: "youtube"         # rthk | youtube | podcast | hktv
  language: "yue"
  domain: "documentary"     # use DOMAIN enum from MANIFEST_SCHEMA.md
  style: "formal"           # use STYLE enum
  priority: "high"          # high | medium | low
  estimated_hours: 100
  notes: "Why this source was added"

# For specific playlists:
- name: "Program Name"
  url: "https://www.youtube.com/playlist?list=PLXXXXXXXX"
  type: "playlist"
  source: "rthk"
  domain: "talk_show"
  style: "casual"
  priority: "high"
  estimated_hours: 80
  notes: "Multiple presenters, diverse guests"
```

### Step 4: Test download

Download 2–3 sample videos before committing the source to the full pipeline:
```bash
yt-dlp --format "bestaudio/best" --audio-format wav --audio-quality 0 \
  --max-downloads 3 \
  --output "data/raw/test/%(title)s.%(ext)s" \
  "https://www.youtube.com/@channelname"
```

Run a mini-pipeline on the samples (segment → transcribe → filter) and check the pass rate. If pass rate < 40%, investigate before committing.

### Step 5: Add to DECISIONS.md

Record the source addition:
```markdown
## YYYY-MM-DD — New source: [Program Name]
**Added**: [URL]
**Domain**: [domain]
**Reason**: [why this source adds value — what diversity it provides]
**Test pass rate**: X% (N=10 sample segments)
```

---

## YouTube HK Cantonese Channels to Evaluate

These are candidate channels that have not yet been added. Evaluate before committing.

### High Priority (Evaluate First)

| Channel | Type | Why |
|---------|------|-----|
| 毛記葵涌 (`@maukeikwaichung`) | Entertainment | Large following, HK Cantonese, diverse cast |
| 立場新聞 (archived) | News | If accessible, high-quality journalism |
| 果籽 (`@appledaily_appledaily_appledaily`) | Feature | Feature stories, on-location, diverse speakers |
| 人山人海 | Music/Talk | If speech-heavy episodes exist |
| 周凱怡 (`@kei_chow_`) | Educational | Clear Cantonese, educational content |

### Medium Priority

| Channel | Type | Concern |
|---------|------|---------|
| Various cooking/lifestyle HK YouTubers | Vlog | Audio quality varies significantly |
| HKMAO channels | Politics | May be Mandarin-heavy |
| Podcast-style YouTube channels | Podcast | Often 1-2 speakers only |

### Reject Immediately

- Mandarin-dubbed content (check if original is Cantonese or dubbed)
- Music channels (no clear speech)
- Content behind login/paywall
- Overseas Cantonese (Canada, Australia, UK HK diaspora) — different phonological patterns

---

## Podcast Sources

Podcasts typically have:
- More consistent audio quality (home studio)
- Smaller speaker diversity (2–3 hosts per show)
- Strong Cantonese identity (podcasters explicitly choose to use Cantonese)
- Lower volume per show, but many shows exist

### RSS Feed Approach

```python
import feedparser, requests

def enumerate_podcast_episodes(rss_url: str) -> list[dict]:
    feed = feedparser.parse(rss_url)
    episodes = []
    for entry in feed.entries:
        audio = next((l.href for l in entry.links
                      if l.type.startswith("audio")), None)
        if audio:
            episodes.append({
                "title":   entry.get("title",""),
                "url":     audio,
                "pub_date": entry.get("published",""),
                "duration": entry.get("itunes_duration","")
            })
    return episodes
```

### Candidate Podcasts

These are example categories to search — verify availability and content before adding:

| Category | Example search terms | Notes |
|----------|---------------------|-------|
| HK current affairs | 香港時事 podcast | High speech ratio |
| HK society | 香港社會 廣東話 podcast | Diverse speakers if panel format |
| HK culture | 本土文化 廣東話 | Good for casual style |
| Cantonese learning | 廣東話 podcast | Clear speech, good for baseline |

Search platforms: Apple Podcasts, Spotify, Google Podcasts, RTHK podcasts page (`podcast.rthk.hk`).

---

## New Source Categories (2026-07-20, T31)

Beyond RTHK/generic-YouTube/podcast, six new source categories were researched (via
online search) to close domain-diversity gaps — 87% of the generic `youtube` raw rows
had no `domain` tag, and the corpus skewed heavily toward RTHK+podcast for diversity.
Each new category is a distinct distribution channel (own `sources/*_sources.yaml`,
own `data/raw/{source}/` folder) but reuses the existing RSS/yt-dlp channel download
mechanism unchanged — see `pipeline/nodes/ingest_download.py`'s `SOURCE_FILES` dict.

All entries in all six new files are seeded at `status: "evaluate"` (unverified
provenance — sourced from online research, not manually spot-checked). **This does not
block `ingest.download` from picking them up** — see the "Manual listening is optional"
note under Step 1 above and DECISIONS.md 2026-07-20 for why no source-level pre-approval
gate is needed on top of the pipeline's existing per-file automated screens.

| Category | File | Top candidate(s) | Notes |
|----------|------|-------------------|-------|
| `hktv` | `sources/hktv_sources.yaml` | HOY 78×Cable News (`@hoy78cablenews`), TVB NEWS (`@tvbnewsofficial`) | Avoid TVB Anywhere overseas channels (`@TVBVariety`, `@TVBBestDramaChannel`) — regionally geo-blocked. `viu.tv` full episodes are HK-IP geo-blocked; only `@viu1hk`/`@viutv` YouTube clips are in scope. |
| `radio` | `sources/radio_sources.yaml` | D100 Radio (`@d100hk`) — 10,000+ VODs, 1–3h each | Commercial Radio (`@cr881903`) and Metro Radio (`@metro_broadcast`) full show archives are paywalled apps (881903.com / MetroPod) — only free YouTube clips are in scope, do not pursue the paid archives. |
| `audiobook` | `sources/audiobook_sources.yaml` | 粵語有聲戲 (`@CantoneseStoryteller`, 100+h) | Single-narrator, clean studio audio — high value for TTS. Children's-storytelling channel (`@mr.sharkstories`) is a distinct register (slow, highly enunciated) — useful but skews style distribution if over-sampled. |
| `gov` | `sources/gov_sources.yaml` | **LegCo (`@legcogovhken`) — 1,000+h, 100+ unique speakers, public record, no geo-block/login.** | Highest-confidence candidate in this round — directly helps the ≥100 unique speakers acceptance criterion. Avoid the simultaneous-interpretation audio track where present, use the primary Cantonese track. |
| `drama` | `sources/drama_sources.yaml` | Listen Watch Learn (`@ListenWatchLearn_amtb`, 200+h) | ⚠️ Radio drama commonly mixes BGM/SFX under dialogue — higher DNSMOS-filter-yield risk than other categories (see "Quality Red Flags" #1 below). Test a small `--limit` batch through segment→transcribe→filter before bulk download. |
| `edu` | `sources/edu_sources.yaml` | HKMU《都大講堂》(`@HKMUChannel`, 100+ full 45–90min lectures) | Long-form single-speaker lecture audio, directly fills the `educational` domain gap. University channels often mix English-medium content — verify Cantonese-medium per playlist, not just per channel (HKU's `@abouthku` flagged as highest risk here). |

Research method: online search via `weir chat agy-gemini`, cross-checked against this
guide's evaluation criteria — not yet verified by direct yt-dlp/listening. Treat every
row above as a lead, not a confirmed source.

---

## Quality Red Flags — When to Reject a Source

Even if yt-dlp can download it, reject the source if:

1. **Audio has background music throughout**: The music will not separate cleanly and will lower DNSMOS < 3.0 for most segments.

2. **Video has pre-mixed audio**: Drama content where dialogue audio is mixed with sound effects at roughly equal volume. Check if dialogue-only audio track exists.

3. **Most content uses heavily Cantonese-accented Mandarin**: Some HK political figures speak Cantonese-accented Mandarin. WhisperX may partially transcribe as Cantonese, producing mixed G2P output.

4. **Geo-blocked content**: If content is geo-blocked to HK, it will fail at the download stage on the current machine. Do not add unless a working proxy is configured.

5. **Dynamic/live content**: Streams and live videos produce unpredictable audio quality. Use VOD (Video on Demand) replays instead.

6. **Single speaker only**: A channel with one person speaking alone has low speaker diversity. Only add if the content is otherwise rare (specific domain, very large volume, unique speaker characteristics).

---

## Source Inventory

After adding a source, update the inventory table in DECISIONS.md with:
- Source name
- Date added
- Estimated hours available
- Estimated hours downloaded
- Last checked date
- Pass rate from most recent batch

This prevents duplicate research and tracks staleness.
