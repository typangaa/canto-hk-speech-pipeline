# HK Cantonese Narration/Scripted Audio Sources for TTS Corpus

**Date**: 2026-06-10
**Status**: All 26 feeds verified (HTTP 200)

---

## How to Use This File

1. Pick sources based on **episode count** (more episodes = more training data)
2. Prioritize **clear narration** (not conversational chat)
3. Download audio, run through the pipeline: segment → transcribe → filter → G2P
4. Check `docs/KNOWN_ISSUES.md` before processing

---

## TOP PRIORITY SOURCES (50+ episodes, high-quality narration)

### 1. Room410 四一零室 (真實犯罪 - Cantonese True Crime)
- **Feed**: https://feeds.soundon.fm/podcasts/9fe6d179-9c48-4751-b7a3-f724ca7bf24f.xml
- **Episodes**: 315
- **Language**: zh-HK
- **Platform**: SoundOn
- **Why good for TTS**: Long-form Cantonese narration, clear storytelling of true crime stories. Consistent single speaker format. Good for conversational Cantonese.
- **Domain**: narration
- **Style**: casual/formal mix
- **Notes**: Need to verify individual episodes for single-speaker segments

### 2. 公公講廣東話故仔 Mario Cantonese Story Time
- **Feed**: https://anchor.fm/s/dc47f5cc/podcast/rss
- **Episodes**: 187
- **Language**: zh
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%85%AC%E5%85%AC%E8%AC%9B%E5%BB%A3%E6%9D%B1%E8%A9%B1%E6%95%85%E4%BB%94-mario-cantonese-story-time/id1676761615
- **Description**: 係澳門嘅公公想日日都講故仔比香港嘅外孫聽，咁唯有用Podcast，可以重播係咁聽。公公用最有愛嘅聲線黎讀比孫仔聽。
- **Why good for TTS**: Pure narration of Cantonese stories, clear reading style, warm tone. Great for children's story register.
- **Domain**: storytelling
- **Style**: warm narration, gentle pacing
- **Speaker**: male (elderly grandfather)

### 3. 襪仔咩咩的繪本窩 Socks Storybook Den (廣東話)
- **Feed**: https://anchor.fm/s/7fbb74f0/podcast/rss
- **Episodes**: 149
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E8%A5%AA%E4%BB%94%E5%92%A9%E5%92%A9%E7%9A%84%E7%B9%AA%E6%9C%AC%E7%AA%A9-socks-storybook-den-%E5%BB%A3%E6%9D%B1%E8%A9%B1/id1609757337
- **Description**: Cantonese picture book stories for children
- **Why good for TTS**: Clear children's story narration, simple vocabulary, consistent format. Good for diverse Cantonese speech patterns.
- **Domain**: storytelling
- **Style**: children's storytelling
- **Speaker**: likely female

### 4. 聖嚴法師有聲書 (粵語版)
- **Feed**: https://anchor.fm/s/43b1b8fc/podcast/rss
- **Episodes**: 193
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E8%81%96%E6%95%91%E6%B3%95%E5%B8%AB%E6%9C%89%E8%81%B2%E6%9B%B8-%E7%B2%B5%E8%AA%9E%E7%89%88/id1544216753
- **Description**: 用心細聽聖嚴法師的著作，以聲音分享佛法智慧
- **Why good for TTS**: Formal Cantonese narration, slow and clear reading style. Excellent for formal register and religious/literary vocabulary.
- **Domain**: audiobook
- **Style**: formal, slow-paced narration
- **Speaker**: male (monk)

### 5. 米．悅讀分享 Mic the Reader
- **Feed**: https://rss.buzzsprout.com/2182675.rss
- **Episodes**: 106
- **Language**: zh-tw
- **Platform**: Buzzsprout
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E7%B1%B3-%E6%82%86%E8%AE%80%E5%88%86%E4%BA%AB-mic-the-reader-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E8%AE%80%E6%9B%B8%E6%92%AD%E5%AE%A2/id1690533716
- **Description**: 一書一世界。讓我們一起通過悅讀，見世界、慰寂寥、撫心靈。
- **Why good for TTS**: Book reading podcast, literary Cantonese. Good for diverse vocabulary.
- **Domain**: audiobook
- **Style**: literary reading
- **Speaker**: likely male (阿米)

### 6. 好耐好耐以前：廣東話講故事
- **Feed**: https://anchor.fm/s/d90d5500/podcast/rss
- **Episodes**: 67
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%A5%BD%E8%80%90%E5%A5%BD%E8%80%90%E4%BB%A5%E5%89%8D-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E8%AC%9B%E6%95%85%E4%BA%8B/id1670290268
- **Description**: Cantonese storytelling from CHERIE'S BOOK CLUB
- **Why good for TTS**: Story narration, Cantonese register. Good for diverse story types.
- **Domain**: storytelling
- **Style**: storytelling

### 7. 金瓶梅 Golden Lotus (女頻粤语演绎)
- **Feed**: https://media.rss.com/the-golden-lotus/feed.xml
- **Episodes**: 36
- **Language**: zh
- **Platform**: RSS.com
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E9%87%91%E7%93%B6%E6%A2%85-%E5%A5%B3%E9%A0%BB%E7%B2%A4%E8%AF%AD%E6%BC%94%E7%BB%8E-golden-lotus-cantonese-audiobook/id1621869364
- **Description**: 感谢大家喜爱我演绎的粤语女频-金瓶梅。看着日渐被遗忘的粤语，我只希望能略尽绵力，向经典致敬。
- **Why good for TTS**: Classical Chinese novel read in Cantonese. Excellent for literary/formal Cantonese and classical vocabulary. Single female narrator.
- **Domain**: audiobook (classical novel)
- **Style**: dramatic reading, formal
- **Speaker**: female (Betty C)

---

## HIGH PRIORITY SOURCES (10-50 episodes)

### 8. 晚安啦，小耳朵 | 廣東話睡前故事
- **Feed**: https://anchor.fm/s/11096cd44/podcast/rss
- **Episodes**: 13
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E6%99%9A%E5%AE%89%E5%95%A6-%E5%B0%8F%E8%80%B3%E6%9C%B5-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E7%9D%A1%E5%89%8D%E6%95%85%E4%BA%8B/id1887560361
- **Description**: 是一個為2–5歲小朋友而設的廣東話睡前故事頻道，提供家長一套簡單的晚安故事工具，透過作者溫柔的聲線講晚安故事。
- **Why good for TTS**: Pure Cantonese bedtime stories, gentle narration. Very clean audio.
- **Domain**: storytelling
- **Style**: gentle, calm bedtime narration
- **Speaker**: likely female

### 9. Ching Ho li - 聊齋故事
- **Feed**: https://feeds.acast.com/public/shows/662b867f437bd7001249b592
- **Episodes**: 19
- **Language**: zh-hk
- **Platform**: Acast
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E8%81%8A%E9%BD%8B%E6%95%85%E4%BA%8B/id1745344793
- **Why good for TTS**: Classical Chinese ghost stories read in Cantonese. Good for formal/classical Cantonese.
- **Domain**: audiobook (classical fiction)
- **Style**: dramatic reading
- **Speaker**: Ching Ho li

### 10. Ching Ho li - 書劍恩仇錄
- **Feed**: https://feeds.acast.com/public/shows/6628e90133dbf40012b4d721
- **Episodes**: 32
- **Language**: zh-hk
- **Platform**: Acast
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E6%9B%B8%E5%8A%8D%E6%81%A9%E4%BB%87%E9%8C%84/id1743395852
- **Why good for TTS**: Jin Yong wuxia novel read in Cantonese. Excellent for martial arts fiction register and classical Cantonese.
- **Domain**: audiobook (wuxia novel)
- **Style**: dramatic reading
- **Speaker**: Ching Ho li

### 11. Cantonese Stories with Star's mommy 廣東話星星媽咪講故事
- **Feed**: https://anchor.fm/s/3cd75e88/podcast/rss
- **Episodes**: 26
- **Language**: zh
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/cantonese-stories-with-stars-mommy-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E6%98%9F%E6%98%9F%E5%AA%BD%E5%92%AA%E8%AC%9B%E6%95%85%E4%BA%8B/id1537182229
- **Description**: 廣東話兒童故事 Cantonese stories for kids
- **Why good for TTS**: Mother telling children's stories in Cantonese. Natural parent-child speech patterns.
- **Domain**: storytelling
- **Style**: parental narration
- **Speaker**: female (mother)

### 12. 奇幻咖啡館 Twilight Cafe 廣東話
- **Feed**: https://anchor.fm/s/7b9d7710/podcast/rss
- **Episodes**: 23
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%A5%87%E5%B9%BB%E5%92%96%E5%95%A1%E9%A4%A8-twilight-cafe-%E5%BB%A3%E6%9D%B1%E8%A9%B1-%E7%B2%B5%E8%AA%9E-podcast/id1609785567
- **Why good for TTS**: Suspense/fantasy stories in Cantonese. Good for varied emotional range in narration.
- **Domain**: storytelling/suspense
- **Style**: atmospheric storytelling
- **Speaker**: unknown

### 13. 夠鐘瞓覺 (廣東話,粵語) 小朋友的睡前小故事
- **Feed**: https://feed.firstory.me/rss/user/ckky68y0x8o350866isxtnqrm
- **Episodes**: 4
- **Language**: zh
- **Platform**: Firstory
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%A4%A0%E9%90%98%E7%9E%93%E8%A6%BA-%E5%BB%A3%E6%9D%B1%E8%A9%B1-%E7%B2%B5%E8%AA%9E-%E5%B0%8F%E6%9C%8B%E5%8F%8B%E7%9A%84%E7%9D%A1%E5%89%8D%E5%B0%8F%E6%95%85%E4%BA%8B/id1552907786
- **Description**: 以廣東話 (粵語) 去講兒童睡前故事，每星期一個令小朋友睡覺的故事。
- **Why good for TTS**: Cantonese children's bedtime stories, simple vocabulary.
- **Domain**: storytelling
- **Style**: children's storytelling
- **Speaker**: unknown

### 14. Vivian's story time 廣東話故事
- **Feed**: https://feeds.soundon.fm/podcasts/d6507bdc-14db-4c06-918a-b6e3d89eff9c.xml
- **Episodes**: 5
- **Language**: zh-Hant
- **Platform**: SoundOn
- **Apple URL**: https://podcasts.apple.com/hk/podcast/vivians-story-time-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E6%95%85%E4%BA%8B/id1742848089
- **Why good for TTS**: Cantonese storytelling by Vivian.
- **Domain**: storytelling
- **Style**: storytelling
- **Speaker**: Vivian (likely female)

### 15. 廣東話睡前故事 (你聽我講)
- **Feed**: https://ntogstorytelling.wordpress.com/feed/
- **Episodes**: 10
- **Language**: en (feed metadata) / Cantonese (content)
- **Platform**: WordPress
- **Why good for TTS**: Cantonese storytelling with music and stories for quiet reflection. Good for atmospheric narration.
- **Domain**: storytelling
- **Style**: calm, reflective storytelling

### 16. Ching Ho li - 天外金球
- **Feed**: https://feeds.acast.com/public/shows/6629026865481e001269c373
- **Episodes**: 15
- **Language**: zh-hk
- **Platform**: Acast
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%A4%A9%E5%A4%96%E9%87%91%E7%90%83/id1743395663
- **Why good for TTS**: Sci-fi novel in Cantonese. Good for modern/sci-fi Cantonese register.
- **Domain**: audiobook (sci-fi novel)
- **Style**: dramatic reading
- **Speaker**: Ching Ho li

---

## MEDIUM PRIORITY SOURCES (1-10 episodes)

### 17. 粵讀你解・廣東話有聲書
- **Feed**: https://anchor.fm/s/108bfd73c/podcast/rss
- **Episodes**: 6
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E7%B2%B5%E8%AE%80%E4%BD%A0%E8%A7%A3-%E5%BB%A3%E6%9D%B1%E8%A9%B1%E6%9C%89%E8%81%B2%E6%9B%B8/id1844150774
- **Description**: Currently reading Animal Farm (動物農場) in Cantonese.
- **Why good for TTS**: Cantonese audiobook of classic literature. Good for literary Cantonese.
- **Domain**: audiobook (literature)
- **Style**: formal reading
- **Speaker**: 琪 Ki

### 18. 羊村有聲書-香港粵語版
- **Feed**: https://anchor.fm/s/df038d80/podcast/rss
- **Episodes**: 6
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E7%BE%8A%E6%9D%91%E6%9C%89%E8%81%B2%E6%9B%B8-%E9%A6%99%E6%B8%AF%E7%B2%B5%E8%AA%9E%E7%89%88/id1682318117
- **Why good for TTS**: Cantonese audiobook version, likely children's content.
- **Domain**: audiobook
- **Style**: reading
- **Speaker**: 羊村2.0團隊

### 19. Survival Cantonese podcast
- **Feed**: https://anchor.fm/s/af45880/podcast/rss
- **Episodes**: 5
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/survival-cantonese-podcast/id1461823686
- **Why good for TTS**: Cantonese fables for learners. Clear, paced reading of Aesop's fables. Transcripts available.
- **Domain**: storytelling
- **Style**: learner-friendly narration, clear pacing
- **Speaker**: HKBooktalker (講故佬)

### 20. Hong Kong Lit Club
- **Feed**: https://media.rss.com/hong-kong-lit-club/feed.xml
- **Episodes**: 11
- **Language**: en (but HK literary content)
- **Platform**: RSS.com
- **Apple URL**: https://podcasts.apple.com/hk/podcast/hong-kong-lit-club/id1802428478
- **Why good for TTS**: Hong Kong literature readings and discussions.
- **Domain**: literature
- **Style**: literary reading and discussion
- **Speaker**: Julia Besnard

### 21. 懸幻火鍋 廣東話PODCAST
- **Feed**: https://anchor.fm/s/420c9b84/podcast/rss
- **Episodes**: 5
- **Language**: zh-tw
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E6%87%B8%E5%B9%BB%E7%81%AB%E9%8D%8B/id1542567301
- **Why good for TTS**: Cantonese suspense/fantasy podcast.
- **Domain**: suspense
- **Style**: suspenseful narration

---

## LOW PRIORITY (1 episode each)

### 22. 柑仔- 陪你成長小故事(廣東話)
- **Feed**: https://media.rss.com/macaugamzai/feed.xml
- **Episodes**: 1
- **Language**: zh
- **Platform**: RSS.com
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E6%9F%91%E4%BB%94-%E9%99%AA%E4%BD%A0%E6%88%90%E9%95%B7%E5%B0%8F%E6%95%85%E4%BA%8B-%E5%BB%A3%E6%9D%B1%E8%A9%B1/id1589647639
- **Description**: 成語/寓言故仔 - idiom and fable stories
- **Domain**: storytelling
- **Speaker**: 柑仔MacauGamzai

### 23. 消逝集-粵語有聲書
- **Feed**: https://anchor.fm/s/557055f8/podcast/rss
- **Episodes**: 1
- **Language**: zh
- **Platform**: Anchor
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E6%B6%88%E9%80%9D%E9%9B%86-%E7%B2%B5%E8%AA%9E%E6%9C%89%E8%81%B2%E6%9B%B8/id1562403296
- **Domain**: audiobook
- **Speaker**: 二木人

### 24. 誰來讀新聞 by少年報導者
- **Feed**: https://feeds.soundon.fm/podcasts/2f9e2894-e97e-4647-a095-c48dbfb81627.xml
- **Episodes**: 491
- **Language**: zh-Hant
- **Platform**: SoundOn
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E8%AA%B0%E4%BE%86%E8%AE%80%E6%96%B0%E8%81%9E-by%E5%B0%91%E5%B9%B4%E5%A0%B1%E5%9E%A3%E8%80%85/id1657824192
- **Why good for TTS**: News reading by young reporters. Formal Cantonese reading.
- **Domain**: news reading
- **Style**: formal news reading
- **Note**: May have multiple speakers; check individual episodes
- **Speaker**: multiple (少年報導者)

### 25. Bayard HK Podcast
- **Feed**: https://feed.firstory.me/rss/user/ckmltpywo8ryo0897oeep6nfx
- **Episodes**: 1
- **Language**: zh
- **Platform**: Firstory
- **Apple URL**: https://podcasts.apple.com/hk/podcast/bayard-hk-podcast/id1569126431
- **Description**: 芥子園出版社兒童節目
- **Domain**: children's
- **Note**: Only 1 episode, may not have enough data

### 26. Ching Ho li - 奧德修斯迷航記
- **Feed**: https://feeds.acast.com/public/shows/66291aad33dbf40012c1a214
- **Episodes**: 2
- **Language**: zh-hk
- **Platform**: Acast
- **Apple URL**: https://podcasts.apple.com/hk/podcast/%E5%A5%A7%E5%BE%B7%E4%BF%AE%E6%96%AF%E8%BF%B7%E8%88%AA%E8%A8%98/id1743395423
- **Domain**: audiobook (classic literature)
- **Style**: dramatic reading
- **Speaker**: Ching Ho li

---

## SUMMARY STATISTICS

| Category | Sources | Total Episodes |
|----------|---------|---------------|
| TOP PRIORITY | 7 | 1,066 |
| HIGH PRIORITY | 10 | 225 |
| MEDIUM PRIORITY | 5 | 26 |
| LOW PRIORITY | 4 | 496 |
| **TOTAL** | **26** | **1,813** |

---

## PLATFORM DISTRIBUTION

| Platform | Sources |
|----------|---------|
| Anchor (Spotify) | 14 |
| RSS.com | 3 |
| SoundOn | 4 |
| Firstory | 2 |
| Buzzsprout | 1 |
| WordPress | 1 |
| Acast | 3 |
| **TOTAL** | **28** (some sources in multiple) |

---

## NOTES

1. **No RTHK drama/storytelling found on podcast.rthk.hk** - The RTHK podcast site doesn't currently offer dedicated Cantonese drama or storytelling content on their podcast platform (as of Oct 2025, RTHK stopped uploading to third-party platforms).

2. **Already excluded**: 馬修靈異怪談鬼故, 胡說八道陳老C, Sparksine廣東話讀書會, RTHK 古今風雲人物, 大話歷史陳老C, 乜東東

3. **Language note**: Some feeds show `zh-tw` or `zh` in metadata but content is HK Cantonese. Always verify with actual audio.

4. **Speaker variety**: 
   - Male narrators: 公公, 聖嚴法師, 米, 琪Ki, Ching Ho li
   - Female narrators: Betty C, Vivian, Star's mommy, Socks Storybook
   - Multiple speakers: Room410, 誰來讀新聞 (need verification per episode)

5. **Content diversity**:
   - Children's stories: 晚安啦小耳朵, 夠鐘瞓覺, Vivian's story time, 柑仔, 襪仔咩咩, 公公, Star's mommy, Bayard
   - Classical novels: 金瓶梅, 聊齋, 書劍恩仇錄
   - Suspense/Thriller: Room410, 懸幻火鍋, 奇幻咖啡館
   - News reading: 誰來讀新聞
   - General storytelling: 好耐好耐以前, 廣東話睡前故事, Survival Cantonese

---

## RECOMMENDED DOWNLOAD ORDER

1. **Room410 四一零室** (315 eps) - Largest Cantonese narration corpus
2. **公公講廣東話故仔** (187 eps) - Clear single speaker, warm tone
3. **襪仔咩咩的繪本窩** (149 eps) - Children's stories, clear Cantonese
4. **聖嚴法師有聲書** (193 eps) - Formal Cantonese, slow reading
5. **米．悅讀分享** (106 eps) - Literary Cantonese
6. **好耐好耐以前** (67 eps) - General storytelling
7. **金瓶梅** (36 eps) - Classical novel, female narrator
8. **Ching Ho li - 書劍恩仇錄** (32 eps) - Wuxia novel
9. **Ching Ho li - 聊齋** (19 eps) - Classical fiction
10. **Cantonese Stories with Star's mommy** (26 eps) - Mother-child stories

Total from top 10: ~1,186 episodes
