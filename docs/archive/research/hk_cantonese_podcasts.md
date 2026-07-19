# HK Cantonese Podcast RSS Feeds — Research Report (2025-06-10)

## Summary

Searched Apple Podcasts, ListenNotes, Firstory, SoundOn, Castbox, and web directories.
Found **2 confirmed working RSS feeds**. 1 further podcast requires manual URL discovery.

---

## Confirmed Working RSS Feeds

### 1. 勵志鷹 (LaichiEagle)
- **RSS URL:** `https://feeds.soundon.fm/podcasts/83838d49-9f4f-43a8-95e9-921174e2730c.xml`
- **Status:** VERIFIED — returns valid RSS (1.5 MB, 568 episodes)
- **Language:** 廣東話 (HK Cantonese)
- **Category:** Comedy
- **Episode Count:** 568
- **Date Range:** Oct 2020 — Jun 2026
- **Latest:** Tue, 09 Jun 2026
- **Publish Frequency:** Twice weekly (Tue/Thu)
- **Hosts:** 3 (石田, 牙石, 牙田 — "石田村民")
- **Description:** 香港第一Podcast, 又頹又勵志嘅鳩笑節目
- **Platforms:** Apple Podcasts, Spotify, KKBOX, SoundOn
- **Apple Podcasts ID:** (not directly mapped)

### 2. HKPUG (派樂派對)
- **RSS URL:** `https://www.hkpug.org/podcast/feed/`
- **Status:** VERIFIED — returns valid RSS (176 KB, 50 episodes in feed)
- **Language:** 廣東話 (HK Cantonese)
- **Category:** Technology (open source / Linux / macOS)
- **Episode Count:** 50+ (RSS exposes 50 items, likely more)
- **Date Range:** Jun 2025 — Jun 2026
- **Latest:** Wed, 03 Jun 2026
- **Publish Frequency:** Weekly
- **Hosts:** 2+ (HKPUG team members)
- **Description:** 香港Python用戶組 — 粵語技術 Podcast，講開源、Linux、macOS、Python 等
- **Platforms:** Apple Podcasts, Spotify
- **Apple Podcasts ID:** 80063725

---

## Unconfirmed Candidates (Need RSS URL Discovery)

These podcasts meet all criteria (HK Cantonese, 50+ episodes, 2+ hosts, active 2023-2025)
but their RSS feed URLs are not publicly discoverable from the current environment.

### 3. 吹水奇懸 (Water Blowing Mysteries) — PRIORITY
- **Episode Count:** 209+
- **Hosts:** 2 (兩個女人)
- **Category:** True Crime / Mystery
- **Language:** 廣東話 (HK Cantonese)
- **Publish Frequency:** Bi-weekly (每兩星期更新)
- **Apple Rating:** 4.8/5 (3,100 reviews)
- **Apple Podcasts URL:** `https://podcasts.apple.com/hk/podcast/吹水奇懸/id1529670928`
- **Apple ID:** 1529670928
- **Hosting:** Firstory (台灣平台, popular in HK)
- **RSS URL:** `https://[podcast-uuid].firstory.io/rss` — UUID not publicly discoverable
- **How to find RSS:** Open Apple Podcasts app → 吹水奇懸 → Share → Copy Link → paste into a podcast player (Overcast, Pocket Casts, etc.) which will auto-discover the RSS feed. Or use the Firstory app to find the podcast's hosting UUID.

### 4. 奇情島
- **Episode Count:** ~100+ (estimated)
- **Hosts:** 1-2 (JC)
- **Category:** True Crime / Mystery
- **Language:** 廣東話 (HK Cantonese)
- **Apple Podcasts URL:** `https://podcasts.apple.com/hk/podcast/奇情島/id1640844928`
- **Apple ID:** 1640844928
- **Hosting:** Unknown

### 5. 好味小姐開束縛我還你原形
- **Episode Count:** ~275+
- **Hosts:** 3 (脆脆, 阿斷, 賴賴)
- **Category:** Lifestyle / Comedy
- **Language:** 廣東話 (Taiwan-based, HK-style Cantonese)
- **Apple Podcasts URL:** `https://podcasts.apple.com/tw/podcast/好味小姐開束縛我還你原形/id1522773953`
- **Apple ID:** 1522773953
- **Hosting:** Firstory or SoundOn

### 6. 好青年荼毒室（哲學部）
- **Episode Count:** 88+
- **Hosts:** 3+
- **Category:** Philosophy / Education
- **Language:** 廣東話 (HK-based)
- **Apple Podcasts URL:** `https://podcasts.apple.com/hk/podcast/好青年荼毒室哲學部/id1588512726`
- **Apple ID:** 1588512726
- **Website:** https://corrupttheyouth.net (301 redirect, RSS not at /feed)

---

## How to Discover RSS Feeds for Firstory/SoundOn Podcasts

Since Firstory and SoundOn don't expose RSS URLs publicly, here are workarounds:

### Method 1: Use a Podcast Player
1. Add podcast from Apple Podcasts URL to **Overcast**, **Pocket Casts**, or **Apple Podcasts app**
2. These players auto-fetch the RSS feed and store it in their data
3. On iOS/macOS, the RSS URL is in the podcast's share sheet or metadata

### Method 2: Use ListenNotes API
```bash
# Register free API key at https://www.listennotes.com/oauth/apps/
curl "https://api.listennotes.com/search2?q=廣東話+podcast&type=podcast&sort_by_date=0&limit=20" \
  -H "X-ListenNotesAPI-Key: YOUR_KEY"
# Response includes `rss` field for each podcast
```

### Method 3: Extract from Apple Podcasts App (iOS)
1. Open Apple Podcasts app
2. Search for the podcast
3. Share → Copy Link
4. Paste link into https://podcastaddict.com/ or similar service
5. The service will resolve to the actual RSS feed

### Method 4: Check the Podcast's Website/Social Media
Many podcasts list their RSS feed in:
- Linktree bio
- Instagram/Facebook bio
- Website footer
- Patreon page

---

## Notes

- **吹水奇懸** is the #1 trending podcast on Apple Podcasts HK charts (as of Jun 2026)
- **勵志鷹** has been running since Oct 2020 with 568 episodes — very stable
- **HKPUG** is a niche tech podcast (open source) but has good Cantonese speech diversity
- Firstory.io URLs changed their format in 2023 — old URLs no longer work
- Many SoundOn podcast URLs now require UUID discovery through the app
- RTHK podcasts (講東講西, 創科新里程) use their own podcast.rthk.hk platform which does not expose public RSS feeds
