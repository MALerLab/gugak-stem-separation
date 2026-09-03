# Σstem test tree — build report (2026-08-30)

Prebuilt the **test-split Σstem evaluation tree** (135 songs) with the existing
`src/data/build_sumstem_eval.py`, unmodified, exactly as the val tree was built.
Build + verification only — **no evaluation, inference, or scoring was run; the test
set remains unscored.** Nothing existing was touched: the val tree, the frozen
manifests, and all checkpoints are exactly as before.

## Step-0 findings (what existed before the build)

- `sumstem_9stem/` (on storage NVMe, reached through the repo symlink
  `data/gugak_ensemble_71955`) held **only `val/`** — 91 songs, ~18 GB. **No test
  sibling existed anywhere**: swept `~/storage/` (nia-gugak, ngc-gugak,
  gugak-demo-set, gugak-stemsep-experiments) and the repo `data/` dirs through
  symlinks; no directory named or shaped like a test Σstem tree.
- All 13 experiment configs agree on the same
  `sumstem_eval.out_root: data/gugak_ensemble_71955/sumstem_9stem` — one shared tree,
  split subdirs underneath. The eval runner (`src/eval/references.py:sumstem_songs`)
  resolves `out_root / <split> / <song_id>`, so `test/` is exactly the sibling it
  expects.
- Conclusion: genuinely not built before → proceeded to build.

## Storage

- **Free space before:** 3.0 TB available on `/home/jae.gye/storage` (NVMe, 59% used).
  Repo disk (`~/userdata`, 96% full) receives **zero audio** — the write path goes
  through the existing symlink onto storage.
- **Written to:** `~/storage/nia-gugak/gugak_ensemble_71955/sumstem_9stem/test/`
  (sibling of `val/`, same naming pattern as the existing storage layout).
- **Repo-side path:** the pre-existing symlink
  `data/gugak_ensemble_71955 → ~/storage/nia-gugak/gugak_ensemble_71955` already
  covers the new tree — no new symlink needed. Verified `readlink -f` resolves and
  `git check-ignore` confirms `/data/gugak_ensemble_71955` is ignored
  (`.gitignore:11`) — no audio can be committed.
- **Tree size:** 25 GB (FLAC PCM_24). **Free space after:** 2.9 TB available.

## What the script did

```
uv run python src/data/build_sumstem_eval.py \
    --config configs/exp001_htdemucs_9stem.yaml --split test --workers 8 --verify 5
```

Run in tmux session `sumstem_test`; idempotent (skips existing song dirs unless
`--overwrite`). Frozen recipe, identical to val by construction (same code path, same
config): classes from `gugak_mix.classes` (9-class canon), quarantined
pitched_percussion (편종/편경/방향) never enters, trim-to-shortest before summing,
mixture peak-normalized to 0.99 with the same linear gain on every target, mono
center-duplicated to stereo, FLAC PCM_24 @ 44.1 kHz, MSST-native layout
(`<song>/mixture.flac` + `<song>/<class>.flac`, absent classes get no file).

- Script banner: `split=test: 135 songs · 854 stems -> 135 to build (0 already
  exist)` — 854 = the manifest's 866 test stems minus the 12 quarantined
  pitched_percussion stems.
- Built **135/135 songs · 7.96 h** of mixture audio · classes per song median 7
  (min 2, max 9). **Wall-clock: 99 s** (8 workers, NVMe).
- The 4 pitched-percussion test songs (`0155/0156/0170/0173_정악_궁중음악`) built from
  modeled classes only, so mixture ≡ Σ(targets) holds for them too (0155 spot-checked:
  6 class files + mixture, no 편종/편경/방향 file).

## Verification gates (all passed; full-tree scan, every file read back)

| gate | result |
|---|---|
| **G1** song set == frozen manifest test list | **PASS** — 135 tree dirs, set-equal to the 135 manifest `song_id`s (asserted against `source_manifest.parquet`, not file counts) |
| **G2** mixture ≡ Σ(class targets), all 135 songs | **PASS** — max error **4.77e-07**, median 3.58e-07; **0 songs > 1e-6** (and 0 > the script's 1e-4 PCM_24 gate) |
| **G3** format + peaks, all 901 files | **PASS** — every file 44.1 kHz / PCM_24 / stereo (2ch only); max peak 0.998260, **0 peaks > 1.0** |
| **G4** per-class file counts vs manifest | **PASS** — all 9 classes match exactly (타악기 129 · 대금 105 · 해금 101 · 거문고 101 · 피리 100 · 가야금 93 · 아쟁 91 · 기타 33 · 양금 13); total 766 targets + 135 mixtures = 901 files, as expected |
| **G5** structural identity vs val tree | **PASS** — spot-checked 3 test songs (incl. quarantined-class 0155) against val songs: same per-song dir layout, same `mixture.flac + <class>.flac` naming (NFC Korean), same 44.1 kHz / PCM_24 / 2ch, absent-class files omitted in both |
| **G6** eval pipeline finds the tree | **PASS** — `src.eval.references.sumstem_songs(out_root, "test", song_ids, "flac")` (the runner's own code path) resolved all 135 songs with no missing-song error |

Gate script (G1–G4, parallel full read-back): session scratchpad
`verify_sumstem_test.py`; build log `sumstem_test_build.log`.

## Deviations / notes

- **Duration:** the task brief estimated ~14 h; the actual trim-to-shortest mixture
  total is **7.96 h** (manifest-derived pre-build estimate matched exactly). Total
  audio written including class targets is far larger (~60 h-equivalent → 25 GB FLAC).
  No action needed — the 135-song count and per-class counts are the binding checks,
  and they reconcile.
- The 1e-6 identity tolerance is satisfied outright despite PCM_24 quantization
  (up-to-10-file sums of ±2⁻²⁴ round-off stay below it); the script's own 1e-4 gate
  remains the formal bound.
- No script modification was needed; nothing else in the working tree was changed by
  this build (audio is on storage, behind the gitignored symlink).
