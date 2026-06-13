# HK Cantonese TTS Corpus — New Source Research Report

**Date:** 2026-06-10
**Purpose:** Identify NEW source types beyond existing YouTube channels, RSS podcasts, and RTHK YouTube
**Scope:** Twitch/YouTube Live, Spotify/Apple exclusives, HK radio, TEDx, podcast aggregators, Clubhouse, Bilibili, niche platforms

---

## EXECUTIVE SUMMARY

| Priority | Source | Est. Hours | Access | TTS Suitability |
|----------|--------|-----------|--------|-----------------|
| P0 | RTHK Podcast One (RSS) | 10,000+ | Open RSS feeds | EXCELLENT |
| P0 | Bilibili (廣東話) | 900-2,500 | yt-dlp, login | HIGH |
| P0 | Twitch VODs (Cantonese) | 500-700 | yt-dlp, immediate | HIGH |
| P1 | HK01 / 01TV YouTube | 1,500+ | yt-dlp | EXCELLENT |
| P1 | SCMP YouTube | 2,000+ | yt-dlp | EXCELLENT |
| P1 | ListenNotes Cantonese | 5,000-10,000 | RSS | GOOD |
| P2 | Apple Podcasts HK Cantonese | 5,000-10,000 | RSS | GOOD |
| P2 | YouTube cooking/commentary channels | 200-500 | yt-dlp | EXCELLENT |
| P2 | WeChat podcast (via Apple/Spotify) | 100-500 | RSS | GOOD |
| P3 | TEDx Cantonese talks | 50-100 | YouTube | GOOD (limited vol) |
| P3 | Commercial Radio 881903 (paid) | 35 months | Paid subscription | LIMITED |
| P3 | Metro Radio 997 | 7 days replay | Free (limited) | LOW |
| X | Spotify-exclusive Cantonese | 0 | N/A | N/A |
| X | Apple Podcasts+ Cantonese | 0 | N/A | N/A |
| X | Clubhouse archives | 0 | Not available | N/A |
| X | LinkedIn Learning | 0 | N/A | N/A |
| X | Udemy | 50-80 (DRM) | Not downloadable | POOR |
| X | Bilibili (Mandarin-only) | 0 | N/A | N/A |
| X | Dedao / Kaiching / Zhihu | ~0-10 | N/A | N/A |
| X | Ximalaya (喜马拉雅) | 50-200 | Possible | LOW |

---

## 1. TWITCH / STREAMING PLATFORMS — HIGH VALUE

### 1.1 Cantonese Twitch Streamers (5 PRIORITY)

**Critical constraint: Twitch VOD retention is 14-60 days.** Must archive immediately.

| Streamer | Followers | Content | Est. VOD Archive | TTS Suitability |
|----------|-----------|---------|------------------|-----------------|
| 艾怡 (irissiri129) | 383K | Just Chatting | 500+ hrs | HIGH — clean single-speaker |
| 達哥 (underground_dv) | ~150K | Gaming + commentary | 200-300 hrs | MEDIUM — gaming BG noise |
| kachingchuk | 19K | Just Chatting + gaming | 200+ hrs | MEDIUM |
| 清兒 (chingyii12) | Partner | Gaming + chat | 200+ hrs | MEDIUM-HIGH |
| 聲大哥 (cksjerry311) | - | Gaming + commentary | 150+ hrs | MEDIUM |
| 雅麗 (lilybbb1) | - | Just Chatting | 400+ hrs | HIGH |

**Audio quality:** Twitch streams use Opus 96-128kbps, 48kHz. Just Chatting streams = clean single-speaker audio. Gaming streams have BGM interference.

**Extraction:** `yt-dlp --audio-format wav "twitch.tv/username/vod/{VOD_ID}"`

### 1.2 YouTube Long-Form (Non-Traditional)

| Channel | Subscribers | Content | Est. Hours | TTS Suitability |
|---------|-------------|---------|-----------|-----------------|
| 大J JASON | 1M+ | Commentary, interviews | 1,000+ hrs | EXCELLENT |
| 馬田 (Dim Cook Guide) | 1.1M | Cooking tutorials | 500+ hrs | EXCELLENT — clean monologue |
| 搞神馬 | 1.3M | Comedy duo | 300+ hrs | GOOD — natural conversation |
| JapHK LIVE | - | Travel, gaming, lifestyle | 200+ hrs | MEDIUM-HIGH |
| 有啖好食 | - | Food/culture commentary | 200+ hrs | EXCELLENT |

### 1.3 VTubers (HK-based, Cantonese)
~10+ HK VTubers (e.g., 月島クロス, 酒吞クロナ). ~400-800 hrs total.
**Caveat:** Many use pitch modification — verify before use for TTS.

---

## 2. SPOTIFY / APPLE PODCASTS EXCLUSIVES — NEGATIVE RESULT

**No Cantonese exclusives found on either platform.**

- Spotify abandoned podcast exclusivity strategy. No Cantonese-exclusive shows exist.
- Apple Podcasts+ exclusives are all English-language (US/UK).
- All Cantonese shows found have public RSS feeds and are distributed on multiple platforms.

**Premium subscription content:** A few shows (e.g., Speak Cantonese on Day 1, 好青年荼毒室) have premium episodes locked behind Spotify paywalls. Limited value (~50-100 hrs).

**API/Scraping:** Spotify Web API only provides metadata, not audio downloads. No viable extraction path.

---

## 3. HK RADIO STATIONS — RTHK IS THE GOLD MINE

### 3.1 RTHK Podcast One (podcast.rthk.hk) — BEST SOURCE

- **100+ Cantonese programs** with RSS feeds
- **RSS URLs:** `podcast.rthk.hk/podcast/rss.xml`, per-program: `podcast.rthk.hk/podcast/item.php?pid=XXXX`
- **Categories:** News, Current Affairs, Art & Culture, Education, Lifestyle, Family, Children
- **Key programs:** 晨早新聞天地, 千禧年代, 自由風自由PHONE, 是日快樂, 一桶金, 講東講西, 三五成群, 晨光第一線, 有你同行, 晚間新聞天地
- **Archive depth:** 12 months via web player (rthk.hk/archive), ongoing via podcast RSS
- **Downloadable:** YES — direct MP3 download or RSS subscribe
- **TTS suitability:** EXCELLENT — professional speakers, clear speech, diverse topics

### 3.2 Other Radio Stations

| Station | Free Archive | RSS | Downloadable | TTS Value |
|---------|-------------|-----|--------------|-----------|
| Commercial Radio 881903 | 35 months (paid: HK$30/10 days) | No | App only | LIMITED |
| Metro Radio 997 | 7 days | No | No | LOW |
| RTHK Web Archive | 12 months | No (web player) | No (streaming) | GOOD |
| RFA Cantonese | Ongoing | YES | YES | MEDIUM — overseas news |

**Verdict:** Focus on RTHK Podcast One. It is the single best free source of Cantonese speech with open RSS feeds.

---

## 4. TEDx / CONFERENCE PRESENTATIONS

### 4.1 TEDx Hong Kong (YouTube)

Found Cantonese talks on the main TEDx Talks channel and specific event channels:

| Talk | Channel | Duration | Views | Language |
|------|---------|----------|-------|----------|
| 遺憾中的無悔 | TEDxHSUHK | 18:43 | - | Cantonese |
| 王維基 | TEDxKowloon | 23:37 | - | Cantonese |
| 終結與開始 | TEDxHSUHK | 19:00 | 129K | Cantonese |
| How Cantonese Slang Connects Generations | TEDxTinHau Women | 9:13 | 2.8K | Cantonese + English |
| TEDx talks in Mandarin and Cantonese | TEDx Talks (playlist) | var. | - | Mixed |

**TEDxKowloon:** @TEDxKowloon — the first Cantonese TEDx event in HK. Channel exists but video listing hard to scrape.

**Estimated Cantonese TEDx hours: 50-100 total** — clean, single-speaker, professional delivery. Good quality but limited volume.

### 4.2 Other Cantonese Conference Content

- HK university public lectures (HKU, CUHK, HKUST) — limited online availability
- No significant Cantonese conference content found on other platforms

---

## 5. PODCAST AGGREGATORS

### 5.1 ListenNotes (listennotes.com)

- Search "Cantonese": 15+ podcasts on first page (ABP Cantonese Bible, Cantonese Coffee Break, 講台, HKPUG Podcast)
- Search "廣東話": 10+ more (生命恩泉, 吹水奇懸, 今日一齊講廣東話啦, 職女聲, SBS Cantonese)
- **Total Cantonese-labeled podcasts: 500+**
- **Estimated total Cantonese podcast hours: 5,000-10,000+**
- CSV export and RSS feeds available

### 5.2 Podchaser (podchaser.com)

- ~10-20 dedicated Cantonese podcasts found
- **Estimated total: 500-1,000 hours**
- Narrower coverage than ListenNotes

### 5.3 小宇宙 (xiaoyuzhou.com)

- Chinese podcast app (iOS/Android only)
- Cantonese content is a small minority on this Mandarin-dominated platform
- Web search not working — limited utility

### 5.4 Apple Podcasts HK Cantonese

- Many Cantonese podcasts available via RSS feeds (already covered in project's existing podcast_sources.yaml)
- ~15+ dedicated Cantonese podcasts with open RSS

### 5.5 香港01 (HK01) Podcast

- 01檔案粵語 (true crime, Apple Podcasts), 01國際Podcast, news podcasts
- **Estimated: 500-1,000 hours**, regularly updated
- Also has YouTube channel (香港01) with daily news videos — HIGH VALUE

---

## 6. CLUBHOUSE — NOT FEASIBLE

- Cantonese rooms were very active in HK during 2021 (peak of Clubhouse boom)
- **BUT:** Replays only accessible inside the app (host must enable recording)
- **No public archive** — cannot extract audio
- **Verdict: 0 usable hours.** Not recommended as a corpus source.

---

## 7. NICHE PLATFORMS

### 7.1 Bilibili (bilibili.com) — HIGH VALUE

Bilibili is China's largest video platform for young people. Cantonese content is a minority but substantial.

| Content Type | Est. Videos | Est. Hours | TTS Suitability |
|-------------|-------------|-----------|-----------------|
| 粵語教學 (Cantonese teaching) | 10,000+ | 100-200 | EXCELLENT — clear, deliberate |
| RTHK content (user-uploaded) | 30+ programs | 50-100 | EXCELLENT — professional |
| TVB dramas (Cantonese) | 5,000+ | 500-1000 | MEDIUM — BGM, multiple speakers |
| Vlogs/commentary | 1,000+ | 200-500 | MEDIUM-HIGH |
| Cantonese animation | 1,000+ | 50-100 | MEDIUM |
| **TOTAL** | | **900-2,500+** | |

**Download feasibility:** HIGH — yt-dlp fully supports Bilibili.
`yt-dlp -x --audio-format wav "VIDEO_URL"` (use 720p max due to 1080p extraction bug)

**Key Cantonese teaching channels on Bilibili:**
- 粤芝士 (bilingual teacher)
- 殿下在香港 (100-sentence series)
- 博布知识学习网 (1000-episode collection, 81.6万 views)

### 7.2 SCMP (South China Morning Post) Video

- Professional Cantonese news videos on YouTube + website
- **Est. hours: 2,000+**
- **Audio quality: EXCELLENT** — professional broadcast quality
- **Downloadable:** YES — yt-dlp for YouTube

### 7.3 HK01 / 01TV

- Daily Cantonese news videos (今日新聞 8分鐘, 今日娛樂 8分鐘)
- **Est. hours: 1,500+** (daily since ~2014)
- **Audio quality: GOOD** — professional studio quality
- **Downloadable:** YES — YouTube channel (香港01) + embedded website videos

### 7.4 Other Niche Platforms

| Platform | Cantonese Hours | Downloadable | Verdict |
|----------|-----------------|--------------|---------|
| LinkedIn Learning | 0 | N/A | DEAD END |
| Udemy | 50-80 | NO (DRM) | POOR |
| WeChat Public Accounts | 100-500 | Very difficult | LOW (dup of RSS) |
| Kaiching (開眼) | 0-10 | Difficult | NOT VIABLE |
| Zhihu audio | 0-5 | N/A | NOT VIABLE |
| Dedao (得到) | 0 | N/A | NOT VIABLE |
| Ximalaya (喜马拉雅) | 50-200 | Possible | LOW |
| Apple Daily archives | 1,000-2,000 | LIMITED | LOW (superseded) |

---

## 8. RECOMMENDED ACTION PLAN

### Phase 1 — Immediate, High-Value (Start Now)

1. **RTHK Podcast One RSS** — Already partially covered. Expand to ALL Cantonese programs via `podcast.rthk.hk/podcast/list_all.php`. This alone gives 10,000+ hours of clean Cantonese speech.

2. **Twitch VOD archiving** — CRITICAL TIMING. Start archiving Cantonese Just Chatting VODs NOW (before 14-60 day window expires). Priority streamers: 艾怡, 雅麗, kachingchuk, 清兒.

3. **HK01 / 01TV YouTube** — Already on YouTube but may not be in your existing config. Download daily news and interview videos. ~1,500+ hours.

4. **Bilibili 粵語教學** — Targeted scraping of Cantonese teaching content. Clear pronunciation, natural speech patterns. ~100-200 hours of high-quality material.

### Phase 2 — Medium Priority

5. **ListenNotes Cantonese RSS** — Use CSV export to discover all Cantonese podcasts. Cross-reference with existing sources. ~5,000-10,000 hours available.

6. **SCMP YouTube** — Professional Cantonese broadcast quality. ~2,000+ hours.

7. **YouTube cooking/commentary channels** — 大J, 馬田, 搞神馬, 有啖好食. Already partially covered but worth expanding.

### Phase 3 — Lower Priority

8. **Commercial Radio 881903** — Consider paid subscription (HK$365/year) for 35-month archive. Worth it if 10,000+ hours.

9. **TEDx Cantonese** — Only 50-100 hours. Worth grabbing if time permits.

### DO NOT INVEST TIME IN

- Spotify/Apple exclusives (none exist for Cantonese)
- Clubhouse (no public archive)
- LinkedIn Learning (no Cantonese courses)
- Dedao, Kaiching, Zhihu (Mandarin-only)
- WeChat (dup of podcast RSS)

---

## 9. TOTAL ESTIMATED NEW HOURS (BEYOND EXISTING)

| Source | Hours | Confidence |
|--------|-------|------------|
| RTHK Podcast One (expanded RSS) | +5,000-10,000 | High |
| Twitch VODs (archived) | +500-700 | Medium (time-sensitive) |
| HK01 / 01TV | +1,500 | High |
| SCMP YouTube | +2,000 | High |
| Bilibili 粵語教學 | +100-200 | High |
| Bilibili general Cantonese | +800-2,300 | Medium |
| ListenNotes (new RSS finds) | +500-1,000 | Medium |
| YouTube cooking/commentary expansion | +200-500 | Medium |
| TEDx | +50-100 | High |
| **TOTAL NEW HOURS** | **~10,650-25,800** | |

Note: These figures represent ADDITIONAL hours beyond existing YouTube/RSS/RTHK collections. Actual usable hours after quality filtering will be lower (~30-60% pass rate based on prior pipeline experience).

---

## APPENDIX A — Key URLs

### RTHK
- Podcast One main: `https://podcast.rthk.hk/podcast/rss.xml`
- All categories: `https://podcast.rthk.hk/podcast/list_all.php`
- Audio programs: `https://podcasts.rthk.hk/podcast/list_audio.php`
- Web archive: `https://www.rthk.hk/archive`

### Twitch
- 艾怡: `https://www.twitch.tv/irissiri129`
- 達哥: `https://www.twitch.tv/underground_dv`
- kachingchuk: `https://www.twitch.tv/kachingchuk`
- 清兒: `https://www.twitch.tv/chingyii12`
- 聲大哥: `https://www.twitch.tv/cksjerry311`

### YouTube (New Sources)
- HK01: `https://www.youtube.com/@hk01`
- SCMP: `https://www.youtube.com/@SCMPNews`
- 大J: `https://www.youtube.com/@HUNGGY`
- 馬田: `https://www.youtube.com/@DimCookGuide`
- 搞神馬: `https://www.youtube.com/@Goshenma`
- 有啖好食: `https://www.youtube.com/@youganghaosis`

### Bilibili
- Search 粵語: `https://search.bilibili.com/video?keyword=粵語`
- yt-dlp Bilibili support: `yt-dlp --help-formats "VIDEO_URL"`

### ListenNotes
- Cantonese: `https://www.listennotes.com/c/cantonese/`
- 廣東話: `https://www.listennotes.com/c/廣東話/`

### Aggregators
- Podchaser: `https://www.podchaser.com/search/podcasts?keywords=cantonese`
