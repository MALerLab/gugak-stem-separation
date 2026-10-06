# CLAUDE.md — Gugak Stem Separation
<!-- STABLE project truth only. Living plan (roadmap, stage, decisions, strategy) → Notion,
     see below. Working style, plan contract, code defaults, sym8 etiquette, guardrails →
     ~/.claude/CLAUDE.md. Commit format → git-commit skill. -->

## Plan & tracking — Notion is the single source of truth
- **Fetch at session start, and before any planning/status discussion:** Notion page
  **Gugak Stem Separation** — id `3a4cb377-8743-8136-8f9a-c2bc24ef8269`
  (https://app.notion.com/p/3a4cb377874381368f9ac2bc24ef8269).
  Child pages: **Experiments** `3a4cb377-8743-8012-921b-e6174eb0effb` ·
  **Survey** `3a4cb377-8743-8075-932e-ca518395c3e1`.
- **On any conflict, Notion wins** — it's the newer, authoritative plan. Trust Notion for
  anything plan-shaped; trust this file for stable facts/conventions.
  (Edit protocol — propose, don't write until confirmed → global CLAUDE.md, "Living plan".)

## Environment
- `uv` project, Python 3.11. `ffmpeg` = system dependency (server-provided), not a Python dep.
- PyTorch cu128 via the PyTorch index (uv), pinned in `pyproject.toml`.
  **GPU (confirmed 2026-07-19):** sym8 driver 590.48 (CUDA 13.1) → cu128 build (sm_120).
- **Audio I/O: `soundfile`** (handles float wavs; stdlib `wave` chokes on them).
  **Avoid `librosa`** — lab convention.

## Datasets  (full status → `docs/dataset_status.md`; taxonomy → `docs/stem_taxonomy.md` +
canonical mapping `configs/stem_taxonomy.yaml`; counts/tables → Notion)
- Refer to sets as **ensemble dataset** / **solo dataset**, not their idx.
- **71955 (ensemble, ours-for-everything):** 903 songs with audio (published 1,004 minus
  101 withheld-test songs); 1 master + N stems each; 6,670 WAVs. 48 kHz/24-bit/stereo,
  except 130 files at 96 kHz (창작국악 → resampled).
- **71470 (solo phrases, train-only pool):** ~9,945 usable single-instrument clips +
  per-clip MIDI (악보) and annotations. Format-heterogeneous — 15 (sr, bit, ch) combos incl.
  float wavs, clips ~1–78 s → resampled + unified on ingest. What a 10 s training segment
  costs out of this pool → `docs/solo_pool_duration_analysis.md`.
- **On disk:** `data/` holds one symlink per set:
  - `data/gugak_ensemble_71955/` → `~/storage/nia-gugak` — `source/<song>/` =
    `<song>_master.wav` + `<song>_<instrument>.wav`; tagging JSONs in `labels/`.
  - `data/gugak_solo_71470/` → `~/storage/ngc-gugak` — flat `audio/` · `midi/` · `labels/`.
- **Manifests = source of truth** (never walk directories). Layout: every table =
  `manifests/parquet/<name>.parquet` (canonical, committed) + `manifests/csv/<name>.csv`
  (eyeball twin, mostly gitignored). Writers derive both from one basename via
  `table_paths()`. Tables:
  - `eval_manifest` — frozen 71955 song-level split (`src/data/data_splitter.py`).
  - `ingest_manifest` — one row per ingested file: provenance + ops (`src/data/ingest.py`).
  - `audio_qc*` — QC scans: `audio_qc` = raw sources, `audio_qc_ingest_<set>` = processed
    store (`scripts/audio_qc.py`). `audio_qc.parquet` is the channel-decision table ingest
    consumes.
  - **`source_manifest_v2` = the one dataloaders read** — ingest ⋈ QC ⋈ taxonomy, one row
    per ingested source file, keyed by stable `file_id`
    (`src/data/build_source_manifest.py`; what v2 changed → `docs/manifest_v2_provenance.md`).
    v2 supersedes v1 for every experiment from exp002 onward. ⚠️ the code default in
    `src/data/mix_dataset.py` is still v1 — every experiment config names v2 explicitly, and
    a new config that forgets to will silently train on v1.
    The pitch-shift pool will be a SEPARATE table at (source × semitone) grain,
    foreign-keying here — must not re-copy split/instrument/content columns.
  - `master_lag` — per-song master↔stem sample offset for the master-input eval
    (`scripts/master_lag_scan.py`); see the master-lag gotcha below.
  - `activity_index` · `activity_segments` · `activity_summary` · `chunk_activities` —
    stem activity scan outputs (`src/data/activity_scan.py` stage 1 →
    `src/data/build_activity_manifest.py` stage 2); envelope blob gitignored at
    `data/activity/`.
- **Split principle** (publisher's split CSVs/folders inconsistent & incomplete — ignored):
  own song-level split, stratified by genreSub, seed-pinned, no song crossing splits.
  Test + val are frozen 71955 song lists; the training side is generated augmented mixes,
  not a song-count split. Ratios/counts + augmentation design → Notion.

### Key facts & gotchas
- **Vocals:** 71955 has no vocal/소리 stem anywhere (incl. 판소리 — its masters also appear
  voiceless). 71470 does supply clean solo vocal clips → whether the scheme gains a voice
  class is a live decision → Notion. Do not assume "no voice stem" as settled.
- **master ≠ Σ(stems) (verified, Phase 2).** Publisher master is mastered (per-stem faders
  + reverb/FX) and genre-dependent (민요 ≈ clean sum, 창작국악 heavily processed).
  → never invert master → stems; stem targets don't sum to it. Training mixtures are
  generated augmented mixes → Notion (Training Data Strategy).
- **Two variants per eval song:** publisher master + Σstem mix. **Early stopping and
  monitoring on the Σstem variant only** — master-val carries an irreducible error floor
  (the mastering residual) that muddies curves/early-stop. Master-val = real-world
  reference, never model selection. Rationale → Notion. (Σstem test tree build →
  `docs/sumstem_test_build_report.md`.)
- **Master↔stem lag (verified 2026-09-05, publisher defect).** In 19 of 135 test songs the
  publisher master is offset from its own stems by a fixed whole number of samples, with no
  duration difference — 59 samples (1.34 ms) across 대풍류 and 9 판소리, 338 samples (7.7 ms)
  in 4 판소리, −2138 samples (−48 ms) in 2 창작국악. SI-SDR and uSDR compare sample by
  sample, so even the 1.34 ms case collapses the score to about −16 dB however good the
  separation actually is. **Always lag-correct before any master-input eval** — offsets live
  in `manifests/parquet/master_lag.parquet` (`scripts/master_lag_scan.py`). Σstem is
  unaffected by construction. Numbers + per-genre effect → `docs/master_eval_lag_report.md`.
- **Audio QC — raw scanned 2026-07-25, processed store verified 2026-07-27**
  (`scripts/audio_qc.py` → `manifests/parquet/audio_qc*.parquet`; findings → Notion). Ingest
  verification: 16,615/16,615 files at 44.1k/PCM_24, 0 peaks >1.0, 0 dead, 0 anti-phase or
  dual-mono survivors, duration conserved 465.38 h. Standing rules:
  - **Sparse ≠ dead** — do NOT auto-drop low-activity stems (박 etc. play rarely by design).
  - Normalize the mixture, never per-stem (loudness ~−19 LUFS in 71955).
  - 71470 ingest: clamp/normalize peaks >1.0, DC-remove, and never naive-average L/R to
    mono (anti-phase clips cancel) — one channel or phase-aware downmix.
- **판소리 length-align (verified, Phase 2):** stems start-aligned; in 28 판소리 songs one
  stem (almost always 가야금) runs longer than the master (usually <0.5 s, one 11 s tail)
  with real content the master lacks. Rule: **trim every stem to the shortest per song
  before summing** (never pad — padding injects content the master never had). 판소리-only.
- **Multi-instrument stems** (피리1/피리2/피리3…) → strip trailing digit to base, sum
  same-base into one source.
- **Korean filenames need NFC** normalization for any name join. Master naming translation:
  on disk `<song>_master.wav` vs metadata records `<song>.wav`.

### Silence is not zero
Post-ingest silence is **not** bit-exact zero. DC removal leaves a ±1-LSB residue of
`1.192e-07` (2⁻²³, one bit of 24-bit depth) at roughly −138 dBFS.
- Never test for silence with equality (`x == 0`, `np.all(arr == 0)`, hash-against-zeros).
  Use a tolerance: `np.abs(arr).max() < SILENCE_EPS` or a dBFS floor.
- Set `SILENCE_EPS` from config, not inline. Anything at or above `1e-6` is safely above
  the residue and far below every threshold in use (activity scan, −50 dB pool gate).
- Consequence to expect: identical silent excerpts may hash to *different* values, since
  the residue differs per file. Group silence by level, never by hash.

## Repo conventions
- **Experiment configs:** one per experiment (`configs/*.yaml`) = one wandb run; change
  configs, not code, for hyperparameter variation. Shared/cross-experiment configs
  (e.g. `configs/stem_taxonomy.yaml`) also live in `configs/`; split
  `configs/experiments/` vs `configs/shared/` if it grows.
- **Run naming:** `expNNN_<model>_<stemscheme>_<key-hparam>`
  (e.g. `exp003_bsroformer_4stem_lr1e-5`).
- **Experiment folders** (`experiments/`): numbered = `exp<NNN>_<YYMMDD>_<rest>`
  (e.g. `exp001_260722_htdemucs_4stem`); named one-offs = `<YYMMDD>_<name>`
  (e.g. `260719_zeroshot_baseline`). Track metrics (parquet/md); figures gitignored
  (regenerable from tracked metrics + script).
- **wandb:** cloud target is entity `maler-gye`, project `gugak_stem_separation`, set in
  `.env` (untracked) — never hardcoded, never by editing the MSST submodule.
  - **Every launch script must source `.env` before training:** `set -a; source .env; set +a`
    — nothing loads it automatically.
  - ⚠️ `WANDB_PROJECT` does NOT override MSST's hardcoded `project='msst'` (an explicit
    `wandb.init()` kwarg beats the env var; `WANDB_ENTITY` does apply) → fix the project at
    upload: `wandb sync -p gugak_stem_separation <run-dir>` (verified 2026-08-05, exp002).
  - `--wandb_offline` controls *streaming only* — wandb always writes the local transaction
    log either way, so an offline run is never a lost run (`wandb sync <dir>` uploads later).
  - Never log audio or checkpoints as artifacts: free tier = 5 GB of file storage, metrics
    don't count against it but artifacts do. `WANDB_IGNORE_GLOBS` in `.env` is the backstop.
- Log git commit hash + full config to wandb for every run.
- Split frozen in the manifest; never let chunks of one song cross splits (audio leakage).
- Notebooks (`notebooks/`) = EDA only.

## Modeling tooling
- **MSST-first:** wrap ZFTurbo/Music-Source-Separation-Training. Config-driven HTDemucs,
  BS-RoFormer, Mel-Band RoFormer, SCNet; **fine-tune from pretrained** (~370 h ensemble
  stems + ~40 h solo clips = fine-tuning territory). Don't reimplement what MSST provides.
- **Custom later:** `src/models/` for gugak-specific ideas + ablations.
- Experiment order, stem-class scheme, model roadmap = live decisions → Notion.

## Evaluation & logging
Full protocol + wandb design → Notion (Evaluation & Logging); pipeline build notes →
`docs/eval_pipeline_build_report.md`. Always-on rules:
- Training metric: SI-SDR (fast). Reported: museval-style chunked SDR (literature-comparable).
- Always break results down **per-stem AND per-genre**, never global mean only.
- Val monitoring/early stopping on the **Σstem variant** (see gotchas).
- Expect lower SDR than Western benchmarks (heterophony → high source overlap).
  Quantify, don't panic.