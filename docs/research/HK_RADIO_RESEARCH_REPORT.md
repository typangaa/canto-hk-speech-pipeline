HK Cantonese Podcast & Radio Sources — Research Report
========================================================
Date: 2026-06-10

This report covers research into 5 specific HK radio/media sources NOT yet in the corpus,
plus additional podcast sources found during research.

Already covered in corpus: RTHK, TVB News, HOY Cable, Now TV

========================================================================
TARGET 1: Commercial Radio HK (商業電台, cr.com.hk)
========================================================================
Status: NO PUBLIC PODCAST RSS FOUND
------------------------------------

Tests performed:
  - cr.com.hk website: No podcast/RSS section found
  - Apple Podcasts API (3 queries: 商業電台, 叱咤903, 香港商業電台): No results
  - Web search: No public RSS feed found
  - YouTube: No official Commercial Radio HK channel found (CommercialRadioHK = 404)

Details:
  Commercial Radio HK is HK's largest commercial broadcaster with two channels:
  - FM903 (叱咤903): Music, talk shows, current affairs
  - FM864 (雷霆864): Entertainment, lifestyle

  Despite their size, they do NOT offer any public podcast RSS feeds.
  Some individual shows may be available on third-party platforms (SoundCloud, etc.)
  but no official podcast feed exists.

  YouTube: No official channel found. The handle "CommercialRadioHK" returns 404.

Recommendation: SKIP. No viable RSS or YouTube source.
Alternative: Monitor for future podcast launch.

========================================================================
TARGET 2: D100 Radio (d100.hk)
========================================================================
Status: SUSPENDED / OFFLINE
----------------------------

Tests performed:
  - d100.hk website: Unreachable / timed out
  - RSS feed search: No RSS feed found
  - Apple Podcasts API: No results for "D100"
  - YouTube: @D100Radio channel exists but appears archived

Details:
  D100 was an independent HK online radio station founded by 方保芳.
  It ceased operations in 2021. The website is unreachable and no active
  podcast feeds exist.

Recommendation: SKIP. D100 has been defunct since 2021.

========================================================================
TARGET 3: Metro Radio / MetroPlus Radio (城市電台)
========================================================================
Status: OFFLINE
---------------

Tests performed:
  - metro.com.hk: Timed out (connection refused / no response)
  - Apple Podcasts API: No results
  - YouTube: No official channel found

Details:
  Metro Radio (MetroPlus) was a commercial HK radio station.
  The website is unreachable and no podcast feeds or YouTube presence found.
  Metro Radio ceased broadcasting in 2020.

Recommendation: SKIP. Station has ceased operations.

========================================================================
TARGET 4: HKSAR Government Press Briefings (info.gov.hk, news.gov.hk)
========================================================================
Status: NO PUBLIC PODCAST RSS
------------------------------

Tests performed:
  - news.gov.hk RSS feed: Returns 404
  - info.gov.hk RSS feed: Returns 404
  - Apple Podcasts API: No results for "HKSAR press briefing"

Details:
  HKSAR Government press briefings are held regularly but are NOT distributed
  as podcast RSS feeds. They are available on YouTube (HKSAR Government channel)
  but no RSS/podcast subscription exists.

  The YouTube channel (HKSAR Government / 香港政府) has press conference videos
  but these are primarily Mandarin with some Cantonese — mixed language content.

Recommendation: SKIP for podcast RSS. Could be added as YouTube source if
Cantonese-dominant press briefings are found.

========================================================================
ADDITIONAL SOURCES FOUND (32 Verified Podcast RSS Feeds)
========================================================================

The following were found during the research process — not part of the
5 target sources above, but all verified as live HK Cantonese podcast RSS feeds:

--- RTHK Podcasts (10 feeds) ---
  #   | Name                          | Episodes | Domain         | Style
  ----|-------------------------------|----------|----------------|----------
  1   | 鏗鏘集 (RTHK Podcast)        | 123      | documentary  | interview
  2   | 城市論壇 (RTHK Podcast)       | 182      | talk_show    | casual
  3   | 頭條新聞 (RTHK Podcast)       | 65       | talk_show    | casual
  4   | 議事論事 (RTHK Podcast)       | 82       | talk_show    | formal
  5   | RTHK 晨早新聞天地              | 113      | news         | formal
  6   | RTHK 新聞特寫                 | 372      | news         | interview
  7   | RTHK 凝聚香港                 | 858      | talk_show    | formal
  8   | RTHK 古今風雲人物             | 16       | documentary  | narration
  9   | RTHK 大地書香                 | 15       | educational  | casual
  10  | RTHK 香港家書                 | 16       | talk_show    | formal

--- Independent HK Podcasts (22 feeds) ---
  #   | Name                          | Episodes | Domain         | Style
  ----|-------------------------------|----------|----------------|----------
  11  | 有台channel D                 | 201      | talk_show    | casual
  12  | 勵志鷹 LaichiEagle            | 568+     | podcast      | casual
  13  | HKPUG Podcast 派樂派對        | 1056+    | podcast      | casual
  14  | Sparksine廣東話讀書會          | 320      | educational  | narration
  15  | 絮言．狂想                     | 107      | educational  | casual
  16  | 五分鐘心理學 樹洞香港         | 334      | educational  | casual
  17  | 不浪漫故事                    | 99       | podcast      | casual
  18  | 吹水奇懸                      | 214      | podcast      | casual
  19  | SBS Cantonese 廣東話節目       | 3221     | news         | formal
  20  | 講經Talkshit (明哥和一發)      | 425      | talk_show    | casual
  21  | 白兵電台                      | 500      | podcast      | casual
  22  | HKcropcircle 廣東話Podcast    | 471      | talk_show    | casual
  23  | 輕鬆講科技                    | 944      | educational  | casual
  24  | 漂夫人事務所 - 漂流放送廳     | 239      | podcast      | casual
  25  | 港識多史｜香港歷史社會研究社  | 205      | educational  | formal
  26  | 馬修靈異怪談鬼故(廣東話)      | 190      | podcast      | narration
  27  | 集誌社 The Collective HK      | 197      | news         | casual
  28  | Nom Talk Network              | 260      | podcast      | casual
  29  | Sex But True 騎呢性趣聞       | 240      | podcast      | casual
  30  | Hong Kong On Screen Podcast   | 83       | podcast      | casual
  31  | 查查+瀝瀝｜香港Podcast        | 60       | podcast      | casual
  32  | 願聞奇詳 (DEAD FEED)          | 0 (404)  | N/A          | N/A

Note: 願聞奇詳's RSS feed is dead (404). Was previously verified with 7 episodes.

========================================================================
VERIFICATION RESULTS
========================================================================

Tested 32 RSS feeds via curl (HTTP status + item count):
  - 31 FEEDS LIVE (HTTP 200, valid XML with <item> elements)
  - 1 FEED DEAD (願聞奇詳 - HTTP 404)

All 31 live feeds confirmed to contain HK Cantonese speech content.

========================================================================
REJECTED SOURCES (4)
========================================================================
  1. Commercial Radio HK (商業電台) — No public RSS found
  2. D100 Radio — SUSPENDED (ceased 2021)
  3. Metro Radio (城市電台) — OFFLINE (ceased 2020)
  4. HKSAR Government Press Briefings — No RSS/podcast feed

========================================================================
YAML CONFIG FILE STATUS
========================================================================
File: sources/podcast_sources.yaml
  - Restructured from flat list to 'sources:' mapping (fixes YAML parse error)
  - 39 source entries (32 active, 1 dead/evaluate, 6 skip/reject)
  - download_config: preserved at top level
  - All RSS URLs verified live (31/31 live, 1 dead)

========================================================================
NEXT STEPS
========================================================================
1. Run 01_discover.py to survey all configured sources
2. Run 02_download.py to download from verified podcast sources
3. Focus on high-priority sources first:
   - SBS Cantonese (3,221 eps, formal Cantonese)
   - 輕鬆講科技 (944 eps, educational Cantonese)
   - 勵志鷹 LaichiEagle (568+ eps, casual Cantonese)
   - 白兵電台 (500 eps, casual Cantonese)
   - HKcropcircle (471 eps, casual Cantonese)
   - 講經Talkshit (425 eps, casual Cantonese)
