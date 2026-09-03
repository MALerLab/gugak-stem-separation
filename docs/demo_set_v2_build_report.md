# demo_set_v2 — Stage 1 build report (acquire · ingest · QC · candidates)

**Built 2026-08-30.** Stage 1 of 2 only: the manifest is **UNFROZEN** (no picks), Notion
untouched, no model inference anywhere. Freeze happens after listening.

- Items: **16** (14 primary + 2 backup) → `manifests/demo_sets/v2.{parquet,csv}`
- Candidates: **108** rows (102 new proposed windows + 6 carried v1 picks) →
  `manifests/demo_sets/v2_candidates.{parquet,csv}`
- Audio: `~/storage/gugak-demo-set/v2/{raw,ingest,candidates,provenance}` — v1 media
  reused via symlinks under video-id names (nothing re-downloaded), new sources fetched
  with yt-dlp best-audio (all WebM/Opus 48 kHz), ingested via the standard chain
  (44.1 kHz / PCM_24, DC removal, peak clamp, channel rule). No loudness normalisation
  baked in.
- Tooling: `configs/demo_set_v2.yaml` + a v2 path in
  `scripts/build_demo_set_candidates.py` (sources × items model; slot system retired).
  **v1 reproducibility verified**: re-running the v1 config regenerates the committed v1
  tables byte-identically.

## Headline verdicts

### gTAK-O-ENAc (시나위, 360º VR) — plain stereo, INGESTED
Every audio format on the upload is 2-channel (`yt-dlp -F`: opus/aac 2ch only, no 4ch
stream exists). ffprobe on the downloaded stream: opus, `channels: 2`,
`channel_layout: stereo`, **no side data**, and zero ambisonic/spatial markers in the
stream tags or the YouTube metadata. Only the video track is 360 ("mesh" projection).
No downmix happened — the stream is stereo at source. Matches the camera-spin test.
Post-ingest L/R correlation **0.298** — the widest image in the set (spaced VR mics).

### _opLG86rrtM (정악풍류 별곡) — TRUE STEREO ✅
2 channels kept by the channel rule, L/R correlation **0.564** (well clear of the
0.9999 dual-mono gate), −22.4 LUFS, 83:33 total. This is the true-stereo, 8-of-9-class,
one-per-part item you were hoping for. Windows were searched **only in 0:05–32:00**
per the hard constraint; the description corroborates the cut — programme item 02
(after your 32:00 mark) is 평조회상 by the full sectional 정악단 (피리×7, 대금×7,
해금×8, 가야금×9, 거문고×6+3, …). Note the description has **no chapter timestamps**,
so your watched 32:00 boundary is the only source for the bound (recorded in the config
with a comment).

## Credit list vs real descriptions (parsed from verbatim `provenance/*.txt`)

Mismatches against your list:

1. **dzfOA-2x4_M (대풍류): the description credits 아쟁/배런**, which your list omitted.
   Real lineup: 피리×2 (한세현·이호진) · 대금 · 해금 · **아쟁** · 장구 · 좌고. All map to
   modelled classes, so ood=false still holds. (Consistent with the commenter complaining
   the 아쟁 is inaudible — it is credited.)
2. **IDjzL4AXIJc (목요풍류 영산회상): per-part credits DO exist** — you flagged a
   possible hold, but the description credits a **trio**: 대금/김영헌 · 거문고/조경선 ·
   해금/고수영. Item proceeds as backup. Two consequences you should know before
   listening: (a) it is a 3-source item, not an ensemble; (b) the video is the **full
   57:49 영산회상 suite (상령산 through 군악)**, not ~14 min of 상령산 — the description
   itself says 상령산 runs "14분 넘게", i.e. the first ~quarter of the file. Windows are
   spread over the whole file; c01–c03 (2:55, 7:25, 13:55) land inside 상령산.
3. Everything else matches your list exactly, including 사물×4 in 장구춤, the
   one-per-part lineups of 몽금포타령/장한가/시나위/별곡, and the ×4 string sections in
   정상지곡. Non-performer credits (무용, 작곡) were parsed and dropped with a note.

## Per-item acquisition · ingest · QC

Shared context: the ensemble-dataset norm is 2ch, L/R corr 0.80–0.995, −25…−13 LUFS.
`CHANNEL_ACTION=pick_channel` = dual-mono source collapsed to 1ch by the standard rule
(logged, not fixed, as specified). A visible pattern: **NGC uploads from 2015
(73E8ZL0CdG0, dzfOA-2x4_M, IDjzL4AXIJc) are all dual-mono; the 2018+ uploads are all
true stereo.**

### Carried v1 items (retagged, audio symlinked, v1 picks kept as candidate c01)
| item | tags | QC notes |
|---|---|---|
| sanjo_master_0253 | studio · in-domain · primary | dataset master, 2ch corr 0.987, −13.6 LUFS, no flags |
| pungnyu_master_0113 | studio · in-domain · primary | dataset master, 2ch corr 0.909, −14.1 LUFS, no flags |
| salpuri | hall · in-domain · primary | dual-mono → 1ch, −29.4 LUFS (known-quiet v1 source) |
| daechwita | outdoor · ood: out_of_vocabulary · primary | corr 0.783 (marginally atypical, outdoor spaced image) |
| haegeum_concerto | hall · ood: orchestral_doubling · primary | corr 0.570 true stereo; −12.4 LUFS (hot broadcast master) |
| pansori | hall · ood: missing_class · primary | **ENVIRONMENT_UNVERIFIED** flag logged as requested — `hall` is inferred (corr 0.93, −19 LUFS), confirm by eye |

### New items
- **monggeumpo_taryeong · janghanga · jeongsangjigok** (all cut from 73E8ZL0CdG0, hall):
  inherit the source's dual-mono→1ch collapse and −29.4 LUFS (flagged on each row, not
  fixed). Piece bounds taken from my parse of the description's chapter list
  (34:52 / 47:21 / 56:13 / 65:44), not from your message — they agree.
  정상지곡 additionally carries **SECTIONAL_DOUBLING** (거문고×4 · 가야금×4) and is the
  quietest material in the set (window LUFS −42…−40): a chamber 줄풍류 texture.
  Its last ~3.5 min (44:00–47:21) yielded no viable windows (2 bins unfilled —
  sustained quiet/dead-air stretch), so picks sit in 35:47–43:57.
- **yeongsanhoesang_hall** (_opLG86rrtM, hall, primary): see headline verdict. 12
  windows spread across 0:55–29:50, dead-air rejection minimal (12/378) — continuous
  ensemble playing nearly throughout.
- **geomungo_sanjo** (WANUsKDUt84, studio, primary): true stereo, corr 0.988, and
  **−13.4 LUFS integrated** — the loudest source in the set, corroborating your note
  that the mastering differs sharply from NGC material (window LUFS −14.3…−11.2 vs the
  NGC halls' −38…−16). No windows were gated on 술대 noise or vocalisations, per spec.
  Zero dead-air rejections (album-style continuous take).
- **gayageum_sanjo** (zTAWuGlw0z0, hall, primary): true stereo (corr 0.536).
  Item-level **VOCALISATIONS** flag (추임새 throughout — automatic per-window 추임새
  detection isn't possible with content features, so the flag lives on the item and
  every window; none were rejected for it). Windows ranked by **audible-activity
  fraction** (surfaced per window as `act`): c01 is the sparse tail case (act 0.75),
  everything else ≥0.92. Two early bins unfilled — the opening ~3 min is the sparsest
  passage and mostly failed the dead-air gate (23/133 windows rejected).
- **janggochum** (Uwuw3hsPC4A, hall, primary): true stereo corr 0.986, −19.6 LUFS, no
  QC flags. Full percussion-band pass (file level): band energy 20–150 Hz **13.6 %**,
  150–500 Hz **51.3 %** (장구 fundamental territory dominates), 500 Hz–2 kHz 26.6 %,
  above 2 kHz ~8.5 %. Crest factor 22.9 dB overall, per-second crest p50 **16.6 dB** /
  p95 **21.1 dB** — that p95 is very high and is the "brutally raw" transient content
  in numbers (heavily processed drums typically sit well under 12 dB short-term).
  Per-window low-band (20–150 Hz) fraction is in the candidate table (`low_band`).
- **sinawi** (gTAK-O-ENAc, hall, primary): see headline verdict. 5:54 source → the 8
  windows overlap by construction (63 analysed, 16 dead-air-rejected).
- **daepungnyu** (dzfOA-2x4_M, hall, BACKUP): dual-mono → 1ch, −19.6 LUFS. **Tone
  check found the squeal**: a persistent narrowband peak at **10,002 Hz**, median level
  ≈ −68 dBFS, prominence 12.4 dB above the local floor, present in 66 % of frames →
  flagged `TONAL_INTERFERENCE(10002Hz)`. Audible as a high squeal in quiet passages;
  quiet enough that loud tuttis mask it. A second flagged peak at **221 Hz**
  (prominence 13.8 dB, presence 51 %) is more likely *musical* — a sustained unison
  pitch (≈A3) rather than interference — I'd disregard that flag; judge by ear.
  Mix-balance complaint recorded in `notes_from_listening`, not verified (needs stems).
- **yeongsanhoesang_room** (IDjzL4AXIJc, room, BACKUP): dual-mono → 1ch, **−32.0
  LUFS** — the quietest source (consistent with an un-amplified 풍류사랑방 recording);
  the inference-time −19 LUFS normalisation will bring it up ~13 dB, so expect audible
  room noise. Trio + full-suite caveats above. Sparse sustains still passed the
  −55 dBFS dead-air gate easily (18/686 rejected) — the slow 상령산 is quiet but not
  discontinuous. Windows ranked by activity and spread 2:55→56:40 across all movements.

## Tagging arguments (where I'd push back or flag a judgement call)

1. **정상지곡 ood=false is defensible but not airtight.** Credits include 생황 and 단소
   (both mapped to the 기타 class — fine) and **대쟁, which has no row in the taxonomy
   at all** — by the letter of `ood_reason: out_of_vocabulary` ("no class at all"), one
   credited instrument qualifies. I kept your `ood=false | SECTIONAL_DOUBLING` because
   the texture is dominated by in-vocabulary strings and 대쟁 is a continuo role, but
   if you want the manifest strictly consistent, this is the row to revisit.
2. **The 대취타 out_of_vocabulary rationale has the same wrinkle in reverse**: 태평소
   *is* in the taxonomy (기타 class, via 71470 solo clips). 나발·나각·자바라·용고 are
   genuinely unmapped, so the tag stands, but "no class at all" is only true for part
   of the lineup. No change made — just noting the asymmetry with 정상지곡.
3. **yeongsanhoesang_room**: if the point of the item is specifically the extreme-slow
   상령산, consider constraining the freeze pick to c01–c03 (inside 상령산) — later
   movements are faster and less of a heterophony stress case. Tags unchanged.
4. Everything else I'd tag exactly as you specified.

## Candidate windows (102 proposed + 6 carried) — ready to listen

Files: `~/storage/gugak-demo-set/v2/candidates/<item>_cNN_<mmss>-<mmss>.wav` (as-is
level, un-normalised — the freeze step makes the −19 LUFS twins). Carried v1 picks are
c01 of their items (symlinks to the v1 audio, human-confirmed 2026-08-17, omitted from
the table below). Columns: `dBFS` = window mean RMS; `LUFS` = integrated window
loudness; `dead` = fraction of 0.5 s blocks below −55 dBFS; `act` = fraction above
−45 dBFS (audible-activity proxy, the ranking key for the two sparse items); `flat` =
spectral flatness (applause/ambience reads high); `low_band` = 20–150 Hz power share
(janggochum only). Selection was content-criteria only — no model output, no scores,
no inference.
| item | c | start–end | dBFS | LUFS | dead | act | flat | low-band |
|---|---|---|---|---|---|---|---|---|
| monggeumpo_taryeong | c01 | 47:56–48:26 | -38.9 | -38.6 | 0.00 | 1.00 | 0.004 |  |
| monggeumpo_taryeong | c02 | 48:26–48:56 | -35.7 | -33.5 | 0.00 | 1.00 | 0.004 |  |
| monggeumpo_taryeong | c03 | 49:41–50:11 | -39.4 | -37.1 | 0.00 | 0.92 | 0.004 |  |
| monggeumpo_taryeong | c04 | 50:16–50:46 | -33.0 | -31.1 | 0.00 | 0.97 | 0.008 |  |
| monggeumpo_taryeong | c05 | 51:31–52:01 | -34.3 | -29.7 | 0.00 | 0.98 | 0.007 |  |
| monggeumpo_taryeong | c06 | 51:46–52:16 | -31.2 | -28.5 | 0.00 | 1.00 | 0.008 |  |
| monggeumpo_taryeong | c07 | 52:36–53:06 | -31.7 | -30.6 | 0.00 | 1.00 | 0.009 |  |
| monggeumpo_taryeong | c08 | 53:16–53:46 | -31.5 | -30.2 | 0.00 | 1.00 | 0.008 |  |
| monggeumpo_taryeong | c09 | 53:21–53:51 | -31.3 | -30.1 | 0.00 | 1.00 | 0.008 |  |
| monggeumpo_taryeong | c10 | 54:21–54:51 | -32.5 | -29.3 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c01 | 56:43–57:13 | -37.2 | -35.8 | 0.00 | 0.98 | 0.002 |  |
| janghanga | c02 | 57:38–58:08 | -35.0 | -33.6 | 0.00 | 1.00 | 0.005 |  |
| janghanga | c03 | 58:28–58:58 | -33.3 | -31.8 | 0.00 | 1.00 | 0.005 |  |
| janghanga | c04 | 59:43–1:00:13 | -31.3 | -30.4 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c05 | 59:48–1:00:18 | -32.0 | -30.8 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c06 | 1:01:18–1:01:48 | -31.2 | -30.1 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c07 | 1:02:08–1:02:38 | -29.7 | -28.5 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c08 | 1:03:13–1:03:43 | -28.6 | -27.5 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c09 | 1:03:58–1:04:28 | -27.5 | -26.0 | 0.00 | 1.00 | 0.006 |  |
| janghanga | c10 | 1:04:53–1:05:23 | -31.7 | -29.5 | 0.00 | 1.00 | 0.005 |  |
| jeongsangjigok | c01 | 35:47–36:17 | -43.0 | -42.1 | 0.00 | 0.80 | 0.008 |  |
| jeongsangjigok | c02 | 36:27–36:57 | -41.0 | -40.1 | 0.00 | 0.97 | 0.006 |  |
| jeongsangjigok | c03 | 36:37–37:07 | -40.6 | -39.8 | 0.00 | 0.97 | 0.006 |  |
| jeongsangjigok | c04 | 36:47–37:17 | -41.0 | -40.0 | 0.00 | 0.93 | 0.006 |  |
| jeongsangjigok | c05 | 37:27–37:57 | -40.9 | -40.0 | 0.00 | 0.98 | 0.006 |  |
| jeongsangjigok | c06 | 38:27–38:57 | -41.1 | -40.1 | 0.00 | 0.97 | 0.006 |  |
| jeongsangjigok | c07 | 40:17–40:47 | -42.0 | -40.8 | 0.00 | 0.82 | 0.007 |  |
| jeongsangjigok | c08 | 41:07–41:37 | -42.2 | -40.8 | 0.00 | 0.80 | 0.006 |  |
| jeongsangjigok | c09 | 43:17–43:47 | -41.7 | -40.5 | 0.00 | 0.87 | 0.007 |  |
| jeongsangjigok | c10 | 43:27–43:57 | -41.1 | -40.2 | 0.00 | 0.93 | 0.007 |  |
| yeongsanhoesang_hall | c01 | 0:55–1:25 | -31.9 | -28.2 | 0.00 | 1.00 | 0.013 |  |
| yeongsanhoesang_hall | c02 | 2:30–3:00 | -32.1 | -27.9 | 0.00 | 1.00 | 0.014 |  |
| yeongsanhoesang_hall | c03 | 5:40–6:10 | -32.0 | -28.0 | 0.00 | 1.00 | 0.013 |  |
| yeongsanhoesang_hall | c04 | 8:30–9:00 | -32.6 | -28.4 | 0.00 | 1.00 | 0.016 |  |
| yeongsanhoesang_hall | c05 | 11:15–11:45 | -32.2 | -27.8 | 0.00 | 1.00 | 0.014 |  |
| yeongsanhoesang_hall | c06 | 15:40–16:10 | -32.9 | -27.7 | 0.02 | 0.98 | 0.017 |  |
| yeongsanhoesang_hall | c07 | 17:15–17:45 | -31.7 | -28.1 | 0.00 | 1.00 | 0.012 |  |
| yeongsanhoesang_hall | c08 | 19:50–20:20 | -31.1 | -27.1 | 0.00 | 1.00 | 0.014 |  |
| yeongsanhoesang_hall | c09 | 21:55–22:25 | -32.1 | -27.8 | 0.00 | 1.00 | 0.011 |  |
| yeongsanhoesang_hall | c10 | 25:50–26:20 | -31.9 | -27.8 | 0.00 | 1.00 | 0.013 |  |
| yeongsanhoesang_hall | c11 | 26:40–27:10 | -31.1 | -26.9 | 0.00 | 1.00 | 0.015 |  |
| yeongsanhoesang_hall | c12 | 29:20–29:50 | -31.9 | -27.7 | 0.00 | 1.00 | 0.015 |  |
| geomungo_sanjo | c01 | 1:25–1:55 | -20.9 | -14.2 | 0.00 | 1.00 | 0.004 |  |
| geomungo_sanjo | c02 | 1:30–2:00 | -21.9 | -14.3 | 0.00 | 1.00 | 0.005 |  |
| geomungo_sanjo | c03 | 4:00–4:30 | -21.1 | -14.1 | 0.02 | 0.98 | 0.008 |  |
| geomungo_sanjo | c04 | 5:25–5:55 | -17.5 | -13.0 | 0.00 | 1.00 | 0.006 |  |
| geomungo_sanjo | c05 | 7:15–7:45 | -18.1 | -12.7 | 0.00 | 1.00 | 0.007 |  |
| geomungo_sanjo | c06 | 9:00–9:30 | -17.0 | -13.0 | 0.00 | 1.00 | 0.005 |  |
| geomungo_sanjo | c07 | 10:55–11:25 | -15.5 | -12.1 | 0.00 | 1.00 | 0.006 |  |
| geomungo_sanjo | c08 | 11:10–11:40 | -16.6 | -12.7 | 0.00 | 1.00 | 0.008 |  |
| geomungo_sanjo | c09 | 13:25–13:55 | -16.0 | -12.3 | 0.00 | 1.00 | 0.007 |  |
| geomungo_sanjo | c10 | 15:35–16:05 | -14.3 | -11.2 | 0.00 | 1.00 | 0.007 |  |
| gayageum_sanjo | c01 | 3:05–3:35 | -36.6 | -24.5 | 0.02 | 0.75 | 0.020 |  |
| gayageum_sanjo | c02 | 3:50–4:20 | -33.4 | -25.1 | 0.00 | 0.98 | 0.017 |  |
| gayageum_sanjo | c03 | 5:35–6:05 | -30.2 | -22.2 | 0.00 | 0.98 | 0.023 |  |
| gayageum_sanjo | c04 | 6:05–6:35 | -30.0 | -23.2 | 0.00 | 1.00 | 0.021 |  |
| gayageum_sanjo | c05 | 6:20–6:50 | -29.5 | -22.9 | 0.00 | 1.00 | 0.022 |  |
| gayageum_sanjo | c06 | 7:35–8:05 | -30.5 | -22.9 | 0.00 | 1.00 | 0.020 |  |
| gayageum_sanjo | c07 | 8:25–8:55 | -30.2 | -23.2 | 0.00 | 1.00 | 0.021 |  |
| gayageum_sanjo | c08 | 10:00–10:30 | -28.8 | -22.9 | 0.00 | 1.00 | 0.010 |  |
| gayageum_sanjo | c09 | 10:15–10:45 | -29.2 | -22.7 | 0.00 | 1.00 | 0.009 |  |
| gayageum_sanjo | c10 | 10:30–11:00 | -29.5 | -23.1 | 0.00 | 1.00 | 0.011 |  |
| janggochum | c01 | 0:30–1:00 | -29.9 | -21.7 | 0.02 | 0.90 | 0.048 | 0.038 |
| janggochum | c02 | 0:55–1:25 | -23.4 | -19.2 | 0.00 | 1.00 | 0.054 | 0.025 |
| janggochum | c03 | 2:00–2:30 | -25.8 | -21.6 | 0.00 | 1.00 | 0.042 | 0.044 |
| janggochum | c04 | 2:05–2:35 | -25.4 | -21.2 | 0.00 | 1.00 | 0.045 | 0.039 |
| janggochum | c05 | 3:25–3:55 | -22.1 | -18.5 | 0.00 | 1.00 | 0.046 | 0.079 |
| janggochum | c06 | 3:30–4:00 | -22.8 | -19.0 | 0.00 | 1.00 | 0.046 | 0.092 |
| janggochum | c07 | 4:50–5:20 | -20.7 | -17.2 | 0.00 | 1.00 | 0.052 | 0.188 |
| janggochum | c08 | 5:00–5:30 | -19.8 | -17.0 | 0.00 | 1.00 | 0.049 | 0.180 |
| janggochum | c09 | 5:40–6:10 | -22.9 | -18.2 | 0.00 | 0.97 | 0.034 | 0.272 |
| janggochum | c10 | 6:40–7:10 | -18.9 | -16.2 | 0.00 | 1.00 | 0.052 | 0.221 |
| sinawi | c01 | 0:30–1:00 | -32.0 | -25.8 | 0.00 | 1.00 | 0.034 |  |
| sinawi | c02 | 0:40–1:10 | -31.2 | -25.3 | 0.00 | 1.00 | 0.038 |  |
| sinawi | c03 | 1:35–2:05 | -34.1 | -24.6 | 0.00 | 0.93 | 0.015 |  |
| sinawi | c04 | 2:25–2:55 | -29.0 | -22.6 | 0.00 | 1.00 | 0.048 |  |
| sinawi | c05 | 3:05–3:35 | -27.4 | -21.2 | 0.00 | 1.00 | 0.050 |  |
| sinawi | c06 | 3:30–4:00 | -33.6 | -24.3 | 0.00 | 0.98 | 0.039 |  |
| sinawi | c07 | 4:45–5:15 | -29.8 | -21.7 | 0.00 | 1.00 | 0.030 |  |
| sinawi | c08 | 4:55–5:25 | -32.5 | -23.8 | 0.00 | 1.00 | 0.021 |  |
| daepungnyu | c01 | 0:20–0:50 | -25.8 | -23.1 | 0.00 | 1.00 | 0.023 |  |
| daepungnyu | c02 | 1:15–1:45 | -25.2 | -21.9 | 0.00 | 1.00 | 0.020 |  |
| daepungnyu | c03 | 2:30–3:00 | -20.2 | -18.3 | 0.00 | 1.00 | 0.015 |  |
| daepungnyu | c04 | 3:35–4:05 | -20.4 | -18.7 | 0.00 | 1.00 | 0.016 |  |
| daepungnyu | c05 | 4:40–5:10 | -19.7 | -18.7 | 0.00 | 1.00 | 0.021 |  |
| daepungnyu | c06 | 5:30–6:00 | -19.7 | -18.4 | 0.00 | 1.00 | 0.020 |  |
| daepungnyu | c07 | 5:40–6:10 | -19.9 | -18.7 | 0.00 | 1.00 | 0.020 |  |
| daepungnyu | c08 | 6:15–6:45 | -19.7 | -18.7 | 0.00 | 1.00 | 0.020 |  |
| daepungnyu | c09 | 7:20–7:50 | -19.9 | -18.7 | 0.00 | 1.00 | 0.019 |  |
| daepungnyu | c10 | 10:00–10:30 | -21.9 | -20.1 | 0.00 | 1.00 | 0.022 |  |
| yeongsanhoesang_room | c01 | 2:55–3:25 | -38.3 | -37.0 | 0.00 | 0.98 | 0.005 |  |
| yeongsanhoesang_room | c02 | 7:25–7:55 | -35.9 | -32.7 | 0.02 | 0.98 | 0.003 |  |
| yeongsanhoesang_room | c03 | 13:55–14:25 | -36.4 | -33.7 | 0.00 | 1.00 | 0.004 |  |
| yeongsanhoesang_room | c04 | 15:40–16:10 | -35.2 | -33.3 | 0.00 | 1.00 | 0.004 |  |
| yeongsanhoesang_room | c05 | 21:40–22:10 | -33.0 | -30.9 | 0.00 | 1.00 | 0.002 |  |
| yeongsanhoesang_room | c06 | 24:20–24:50 | -31.4 | -29.1 | 0.00 | 1.00 | 0.002 |  |
| yeongsanhoesang_room | c07 | 30:40–31:10 | -33.4 | -30.9 | 0.00 | 1.00 | 0.003 |  |
| yeongsanhoesang_room | c08 | 36:50–37:20 | -30.1 | -28.4 | 0.00 | 1.00 | 0.003 |  |
| yeongsanhoesang_room | c09 | 40:15–40:45 | -30.5 | -28.4 | 0.00 | 1.00 | 0.003 |  |
| yeongsanhoesang_room | c10 | 43:55–44:25 | -30.2 | -28.1 | 0.00 | 1.00 | 0.003 |  |
| yeongsanhoesang_room | c11 | 51:05–51:35 | -28.6 | -27.9 | 0.00 | 1.00 | 0.003 |  |
| yeongsanhoesang_room | c12 | 56:10–56:40 | -27.6 | -25.9 | 0.00 | 1.00 | 0.001 |  |

## Next step

Listen through `~/storage/gugak-demo-set/v2/candidates/`, pick per item, then stage 2:
add a `freeze:` section to `configs/demo_set_v2.yaml` and run with `--freeze`
(renders `final/` as-is + `_normalised` twins and freezes `manifests/demo_sets/v2`).

## Stage 2 — FROZEN 2026-08-31

Picks listened and chosen 2026-08-31 (one per item; carried items keep their v1 picks).
`manifests/demo_sets/v2.{parquet,csv}` is now the frozen item manifest (16 rows + pick
columns); audio in `~/storage/gugak-demo-set/v2/final/` — 32 files split as
`final/original/` (as-is byte copies of the picked candidates) + `final/normalised/`
(the twins; same layout applied retroactively to v1's final/). Manifest filename columns
stay bare names — the original/normalised split is directory-level only. Twins follow
the inference-time rule
(one gain to −19 LUFS, then a 0.99 peak guard — identical to `mix_dataset._normalize`,
so the `_normalised` file IS the model's input).

Seven items hit the peak guard and land short of −19 LUFS (realised): salpuri −21.2,
janghanga −22.6, jeongsangjigok −24.3, gayageum_sanjo −22.4, sinawi −23.4, daepungnyu
−21.5, yeongsanhoesang_room −20.1. That is the rule working as designed on high-crest
material, not an error — the model sees exactly the same guard at inference.

Still open (tag/QC level, does not affect the freeze): pansori ENVIRONMENT_UNVERIFIED
(confirm by eye), and the 정상지곡/대취타 ood grey zone (대쟁 unmapped / 태평소
in-vocabulary).
