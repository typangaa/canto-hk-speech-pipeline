# Drama & Scripted Dialogue Sources — Research Findings

**Date**: 2026-06-10
**Researcher**: Research Agent
**Purpose**: Fill the critical gap in TTS corpus — scripted dialogue with natural sentence-level prosody

---

## Summary

Found **15+ high-value sources** of Cantonese drama/scripted content across YouTube, museum channels, and radio drama archives. Most significant find: **TVB Anywhere's official YouTube channels** (4 channels, 2M+ combined subscribers) with full drama episodes in Cantonese. Also found massive amounts of **Commercial Radio HK (商台) radio dramas**, **classic Cantonese films (粵語長片)**, and **HK museum narration**.

**Key insight**: YouTube is the primary free distribution platform for Cantonese scripted content. No RSS feeds for drama content found — all access via YouTube video download.

---

## Tier 1 — High Priority (Scripted, Professional, Large Volume)

### 1. TVB Anywhere YouTube Channels (Official — BEST SOURCE)
| Channel | Handle | Subscribers | Content | Access |
|---------|--------|-------------|---------|--------|
| TVB Best Drama 熱播劇場 | @TVBBestDramaChannel | 653K | Full TVB dramas, Cantonese, Chinese subtitles | yt-dlp |
| TVB Drama - Crime & Mystery | @TVB_Mystery | 389K | Crime/mystery dramas | yt-dlp |
| TVB Drama – Action & WuXia | @TVB_WuXia_KungFu | 452K | Martial arts dramas | yt-dlp |
| TVB Anywhere (SG MY) | @tvbanywhere_sgmy | 612K | General TVB dramas | yt-dlp |
| TVB Sitcom 處境喜劇 | (found via search) | — | Sitcoms (結．分@謊情式, etc.) | yt-dlp |
| TVB Food & Travel 飲食旅遊 | @TVBFoodTravel | — | Food/travel shows, scripted | yt-dlp |

**Total combined**: ~2.1M subscribers, thousands of episodes
**Content type**: Scripted drama (professional actors, written dialogue, studio audio)
**Audio quality**: Excellent (official broadcast quality, usually 1080p video → extract audio)
**Estimated hours**: 2000+ hours of drama content
**Domain**: `drama` (multiple subtypes: crime, mystery, martial arts, sitcom, historical, contemporary)
**Is_scripted**: YES — this is exactly what we need
**Notes**: 
- Each episode 40-45 min, full Cantonese dialogue
- NOT available in HK, China, Malaysia (geo-blocked for those regions)
- **Perfect for Cantonese TTS training** — professional actors, clear articulation, scripted dialogue
- Drama examples found: 使徒行者, 黑色月光, 幕後玩家, 迷, 火玫瑰, 寫意人生, 本草藥王, 再版人
- Also has TVB classic dramas from 1980s-2000s (梁朝偉, 劉德華 era)

### 2. Commercial Radio HK (香港商業電臺) — YouTube Radio Dramas
| Source | Description | Access |
|--------|-------------|--------|
| @cantonesefilm_classic (粵語長片台) | 205K subs, 1.9K videos — uploaded radio dramas | yt-dlp |
| "Listen Watch Learn . 聆聽 觀看 學習" | Full radio dramas: 書劍恩仇錄 (3h42m), 奇俠司馬洛 (2h15m), 倫文敘天降福星 (1h37m), 聊齋誌異 (2h7m), 神探福爾摩 (2h49m), 香港素描 | yt-dlp |
| "Suet Taylor" playlist | RTHK 廣播劇 playlists, 恐怖星期二, 鬼故 | yt-dlp |
| 碌斌深夜粵語廣播 @lubinlost | 143K subs — Guangzhou professional voice actor, regular updates | yt-dlp |

**Content type**: Audio-first radio dramas, professional voice actors, multiple speakers
**Audio quality**: Good to excellent (audio-first production)
**Estimated hours**: 500+ hours
**Is_scripted**: YES — radio dramas are fully scripted
**Domain**: `drama` (ghost stories, wuxia, detective, historical, comedy, slice-of-life)
**Notes**:
- 粵語長片台 has 108 episodes of 怪談 from 香港商業電臺
- Commercial Radio HK is THE radio drama powerhouse in HK — produced thousands of dramas over decades
- Mix of old classic dramas and newer content

### 3. Classic Cantonese Films (粵語長片)
| Source | Description | Access |
|--------|-------------|--------|
| @cantonesefilm_classic (粵語長片台) | 1940s-1970s Cantonese films, full movies 1-2 hours each | yt-dlp |

**Content type**: Classic Cantonese films from golden age (1940s-70s)
**Audio quality**: Variable (older recordings, but clear enough)
**Estimated hours**: 1000+ hours
**Is_scripted**: YES — all films are scripted
**Domain**: `drama` (period pieces, comedies, martial arts)
**Notes**:
- Classic Cantonese pronunciation (slightly different from modern HK Cantonese)
- Multiple professional actors per film
- Important for historical Cantonese but may need age-filtering

---

## Tier 2 — Medium Priority (Good Volume, Scripted)

### 4. ViuTV Official
| Source | Description | Access |
|--------|-------------|--------|
| ViuTV @ViuTV | 541K subs, official channel | yt-dlp |

**Content type**: Original ViuTV dramas, trailers, program promos
**Estimated hours**: 50-100 hours (mostly trailers/clips, fewer full episodes)
**Is_scripted**: YES
**Domain**: `drama`, `talk_show`
**Notes**:
- More modern, contemporary Cantonese
- Younger actors, different demographic from TVB
- Fewer full episodes on YouTube compared to TVB

### 5. HK Museum YouTube Channels (Narration — Scripted)
| Channel | Handle | Subscribers | Content |
|---------|--------|-------------|---------|
| 香港歷史博物館 | @hongkongmuseumofhistory7109 | 12.2K | Museum exhibitions, "香港故事" series |
| 香港文化博物館 | @hongkongheritagemuseum8188 | 7.36K | Cultural heritage content |
| 優遊香港博物館 | @VisitHKMuseums | 10.2K | LCSD museum network content |
| Easy Languages | — | — | "Hong Kong Museum of History | Easy Cantonese" series |

**Content type**: Museum narration, exhibition descriptions, "香港故事" (HK Story) series
**Audio quality**: Excellent (professional narration, studio recording)
**Estimated hours**: 200-300 hours
**Is_scripted**: YES — museum narration is fully written and rehearsed
**Domain**: `educational`, `narration`
**Notes**:
- Clear, measured, formal Cantonese
- Good for training formal/narration prosody
- Single-speaker segments (easier for VAD/diarization)
- Each video 3-12 minutes

### 6. RTHK Content (YouTube)
| Source | Description | Access |
|--------|-------------|--------|
| RTHK 香港電台 @rthkhongkong | 300K+ subs | yt-dlp |
| Suet Taylor playlist | RTHK radio dramas | yt-dlp |

**Content type**: RTHK programs including radio dramas, educational content
**Estimated hours**: 100-200 hours
**Is_scripted**: PARTIALLY (some radio dramas are scripted, some talk shows are semi-improv)
**Domain**: `documentary`, `drama`, `educational`
**Notes**:
- Need to filter for scripted content only
- RTHK official YouTube has catch-up TV/radio content

---

## Tier 3 — Supplementary Sources

### 7. Children's Content YouTube
| Source | Description | Access |
|--------|-------------|--------|
| 好聽故事書Books for the Little Soul | 粵語兒童故事, picture book narration | yt-dlp |
| 采姐姐的故事王國 Lillian's Story Kingdom | Animated stories with Cantonese narration | yt-dlp |
| CreativeZoo 童想樂園 | Animated Cantonese stories (獅子和老鼠, etc.) | yt-dlp |
| 擔櫈仔 STORYTIME in Cantonese | Children's stories in Cantonese | yt-dlp |

**Content type**: Children's stories, picture book narration, animated stories
**Audio quality**: Good to excellent (designed for children, very clear)
**Estimated hours**: 100-200 hours
**Is_scripted**: YES — fully written stories
**Domain**: `educational`
**Notes**:
- Very clear, slow, measured Cantonese — excellent for TTS prosody
- Single-speaker, clean audio
- May sound "childish" for adult TTS but good for prosody training

### 8. ATV (Asia Television) Classics
| Source | Description | Access |
|--------|-------------|--------|
| 亞視精選 Drama Asia | Full ATV dramas: 新包青天 (160 eps!), Cantonese audio | yt-dlp |

**Content type**: Classic ATV dramas from 1990s
**Estimated hours**: 200+ hours
**Is_scripted**: YES
**Domain**: `drama` (historical, crime, wuxia)

### 9. HK Film Archive (Limited)
| Source | URL | Status |
|--------|-----|--------|
| HK Film Archive | https://www.filmarchive.gov.hk | Returns 200 but limited downloadable content |

**Status**: Website is accessible but no clear download mechanism found
**Estimated hours**: <10 hours (uncertain)
**Notes**: Would need deeper investigation; may require physical visit or special request

### 10. TVB Sitcoms (via TVB channels)
| Source | Description | Access |
|--------|-------------|--------|
| TVB Sitcom 處境喜劇 (found via search) | Full sitcom episodes: 結．分@謊情式 (139 eps) | yt-dlp |

**Content type**: Multi-cam sitcoms (daily life Cantonese, casual speech)
**Estimated hours**: 50-100 hours
**Is_scripted**: YES — sitcoms are fully scripted with laugh tracks
**Domain**: `drama` (comedy, daily life)
**Notes**:
- Great for casual, conversational Cantonese
- Multiple speakers per episode
- Natural sentence-level prosody

---

## Sources That Are NOT Viable

| Source | Reason |
|--------|--------|
| HKTV (hktv.com.hk) | Redirected to hktvmall.com (shopping site), no drama content |
| ViuTV OTT (viu.com) | Subscription-only, geo-blocked, no free episodes |
| RTHK archive (archive.rthk.hk) | Returns 403 — blocked/geo-restricted |
| RTHK old site (rthk9.rthk.hk) | Returns empty page |
| RTHK podcast RSS (rthk.hmt.rthk.hk/api) | Returns 000 — connection refused |
| HK Film Archive | No accessible download mechanism for drama audio |
| Museum audio guide apps | No public API or download mechanism |
| ETV (education TV) | Not found as accessible online source |
| BiliBili (bilibili.com) | Returns 404 for Cantonese drama searches |

---

## Recommended Priority Order for Collection

### Phase 1 (Immediate — highest ROI):
1. **TVB Best Drama + TVB Anywhere channels** — 2000+ hours, perfect quality, fully scripted
2. **粵語長片台** — 1.9K videos, mix of films and radio dramas
3. **Listen Watch Learn + Suet Taylor** — Commercial Radio HK drama archive

### Phase 2 (Add variety):
4. **ViuTV official channel** — Modern, young Cantonese
5. **HK Museum channels** — Formal narration, single-speaker
6. **ATV classics** — Different actor pool, 1990s era

### Phase 3 (Fine-tuning):
7. **Children's story channels** — Clear pronunciation, good for prosody
8. **碌斌深夜粵語廣播** — Professional voice actor, regular updates
9. **TVB Sitcom** — Casual daily-life Cantonese

---

## Technical Notes for yt-dlp Collection

### Best Practices:
```bash
# Download full episodes, extract audio only, 48kHz
yt-dlp -x --audio-format wav --audio-quality 0 -o "%(title)s.%(ext)s" "VIDEO_URL" --postprocessor-args "ffmpeg:-ar 48000 -ac 1"

# Batch download from channel
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "%(channel)s/%(playlist_index)s-%(title)s.%(ext)s" \
  "https://www.youtube.com/@TVBBestDramaChannel/videos" \
  --postprocessor-args "ffmpeg:-ar 48000 -ac 1"

# Download specific playlist
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "%(playlist_index)s-%(title)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PLAYLIST_ID" \
  --postprocessor-args "ffmpeg:-ar 48000 -ac 1"
```

### Important Considerations:
- TVB content is geo-blocked in HK/China/Malaysia — use non-geo IP if needed
- All YouTube content has CC (closed captions) available — useful for transcript verification
- Quality varies by upload date — older uploads may be 480p, newer are 720p/1080p
- Audio is embedded in video — must extract audio via ffmpeg
- Some channels upload same content with slight title differences

### Content-Type Classification for Filtering:
When processing, use these keywords to identify drama/scripted content:
- `粵語`, `粵語中字`, `粵語原聲` — Cantonese with subtitles
- `劇`, `電視劇`, `完整劇`, `全劇` — drama/TV series
- `廣播劇`, `廣播` — radio drama
- `電影`, `長片` — film
- `旁白`, `導賞` — narration
- `故事`, `童話` — stories

---

## Estimated Total Hours by Source Category

| Category | Source Count | Est. Hours | Scripted? | Quality |
|----------|-------------|------------|-----------|---------|
| TVB Drama (official) | 6 channels | 2000+ | YES | Excellent |
| Radio Dramas (商台) | 4 sources | 500+ | YES | Good-Excellent |
| Classic Cantonese Films | 1 channel | 1000+ | YES | Variable |
| ViuTV | 1 channel | 50-100 | YES | Good |
| Museum Narration | 4 channels | 200-300 | YES | Excellent |
| Children's Stories | 4 channels | 100-200 | YES | Excellent |
| ATV Classics | 1 source | 200+ | YES | Good |
| TVB Sitcom | 1 playlist | 50-100 | YES | Good |
| **TOTAL** | **~20 sources** | **~4000-4500h** | **YES** | **Good-Excellent** |

This is MORE than enough for a 100-500 hour TTS corpus. The key will be filtering for quality and speaker diversity.

---

## Next Steps

1. **Start with TVB channels** — most content, best quality, fully scripted
2. **Verify yt-dlp works** with a test download from TVB Best Drama channel
3. **Set up audio extraction pipeline** — video → 48kHz mono WAV
4. **Build domain classifier** to separate drama from other content types
5. **Track speaker diversity** across all sources
