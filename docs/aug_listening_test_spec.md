# Augmentation listening test — hand-off spec

Contract for the automated sessions that build this test. Agreed with Jae 2026-10-20.
Anything not covered here is a question for Jae, not a judgement call.

## Goal

Decide go/no-go on further recording-condition augmentation R&D (prof mtg 2026-10-19).
SDR cannot score real-world recordings, and an augmented eval set is circular, so the
decision rests on (a) a master-input SDR eval, which is real processing with ground truth,
and (b) a blind A/B listening test by Jae (later possibly prof).

This is a pilot for an internal decision, not a paper-grade listening study. Keep it simple.

## The pair

| role | run | checkpoint | val SI-SDR |
|---|---|---|---|
| augmented | exp007 (EQ + capture chain + room, p=0.5) | `experiments/exp007_260925_bsroformer_room_distance/checkpoints/model_bs_roformer_ep_34_si_sdr_10.7842.ckpt` | 10.78 |
| baseline | exp006 (same recipe, no augmentation) | `experiments/exp006_260925_bsroformer_partial_coherent/checkpoints/model_bs_roformer_ep_34_si_sdr_11.2352.ckpt` | 11.24 |

exp007's config is derived from exp006's; only the augmentation differs.

## Stage 1 — master-input eval of exp007 ep34

- Mirror `configs/eval/exp006_ep34_test_master_aligned.yaml` and
  `configs/eval/exp006_ep34_test_sumstem.yaml` for exp007 ep34; outputs under
  `experiments/exp007_260925_bsroformer_room_distance/eval/`.
- Master input must be lag-corrected (`manifests/parquet/master_lag.parquet`).
- Report per-stem AND per-genre, exp007 vs exp006, both variants.
- GPU job: show the configs to Jae, get approval, `nvidia-smi` for an idle card, run in
  `tmux`, time cap inside the launch script. Never hard-kill.

## Stage 2 — pick the master songs and windows

Input: per-song, per-stem master-aligned SDR for both models (stage 1 + existing exp006).
Let d = exp007 − exp006.

1. One song per genre (7 test genres): the song whose song-level d is closest to its
   genre's median d.
2. Top-up: while any of the 9 model stems (가야금 거문고 기타 대금 아쟁 양금 타악기 피리 해금)
   has fewer than 2 picked songs, add the song (any genre) whose per-stem d for that stem
   is closest to that stem's median d. Coverage counts only if the stem is active in the
   chosen window (rule 3). Expect ~8–9 songs total; 양금 (13 test songs) and 기타 (33)
   are the likely top-ups.
3. Window: 30 s, chosen from the activity scan (`activity_segments`) to cover the stems
   the song was picked for, as active as possible.
4. Write the picks + windows to a manifest table (parquet + csv twin via `table_paths()`),
   with the pick reason per song (genre-typical / top-up for <stem>).

## Stage 3 — renders

- Items: the picked master windows (lag-corrected master input) + the 11 in-vocabulary
  YouTube items from demo set v2 (`manifests/demo_sets/v2.parquet`, normalised files).
  Exclude 대취타, 해금 협주곡, 판소리 (out of vocabulary / orchestral / voice).
- For each item: both models' 9 stems; for master items also the true stems for the
  same window.
- Reuse the existing render path (`configs/render/`) rather than new inference code.
- Instrument coverage counts master items only; YouTube items carry no stem labels.

## Stage 4 — A/B listening page

- One row per (item, stem); skip stems silent in both outputs and, for master items, in the
  true stem.
- **Blind by default.** Per row, exp006/exp007 are assigned to A/B at random (seeded,
  stored in the page data).
- **Reveal button** per row shows which model is which; the row records
  `revealed_before_answer` if clicked before answering.
- Master items only: a third player with the **true stem**.
- Per row: A better / B better / no real difference · optional tags (bleed, artifacts,
  missing content, room/reverb) · optional free-text note.
- **Save is an explicit button, not autosave.** Nothing persists until Jae presses Save.
  Save writes the answers file; reopening the page restores from it.
- Answers are kept per listener (signed-in identity via the artifact `user` capability),
  so prof's answers stay separate from Jae's.
- Hosting: a private claude.ai Artifact, shared by link with prof —
  https://claude.ai/artifact/BTAZaqZ3tvtoFJhhVSmVww (published 2026-10-20).
  - Share with prof at Editor or Contributor level: Viewers can listen but not save.
  - Publish from a scratchpad copy of page.out_dir: data/ is a symlink to ~/storage, which
    the Artifact tool refuses as a publish root. Batches of ≤ 58 MB (4 publishes for ~178 MB).
  - Answers: Save writes the listener's answers to the artifact's `db`, one doc per
    listener; stage 5 reads them back with `ArtifactData`.
  - Audio: MP3 160 kbps (normal music-file quality, ~0.6 MB per 30 s clip; plays in every
    browser). Lossless FLAC/WAV stay on sym8 for spot checks.
  - Checked 2026-10-20: artifacts serve .mp3 .wav .ogg .mp4 .webm as published files
    (not .m4a). Limits: ≤ 255 files / 64 MB per publish, ≤ 511 files / 256 MB per version.

## Stage 5 — tally

- Per stem, per item type (master / YouTube): counts of A-better / B-better / no
  difference, unblinded; blind answers (not revealed before answering) counted separately.
- Tag counts per model win; flag stems where exp007 wins mainly on room/reverb.
- Side by side with stage 1's per-stem d.

## Conventions

- `soundfile` for audio I/O; no `librosa`.
- No silence equality checks; tolerance from config (`SILENCE_EPS`).
- Configs in `configs/`, no hardcoded paths.
- No commits or pushes without Jae's explicit go-ahead; propose the commit set at the end.
