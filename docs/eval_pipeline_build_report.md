# Evaluation pipeline — build report (2026-08-30)

Config-driven evaluation pipeline: SI-SDR + uSDR on the frozen splits, streaming
inference (no separated audio written to disk), raw per-(song, class) rows persisted,
aggregation fully decoupled from scoring. Built and gate-verified on the val split;
**the 135-song test set remains untouched.**

## Files added (nothing existing was modified)

| file | role |
|---|---|
| `configs/silence.yaml` | **the single config home of the silence tolerance** (see below) |
| `configs/eval/exp003.0_val_sumstem.yaml` | eval-job config for the dry run (the template for future jobs) |
| `src/eval/metrics.py` | metric registry: `si_sdr` (MSST-verbatim), `usdr` (MDX'21), `csdr` stub |
| `src/eval/references.py` | per-variant mixture + lazy per-class reference access (`sumstem` / `master`) |
| `src/eval/runner.py` | streaming loop: infer one song → score in memory → keep rows → discard audio; CLI entrypoint (`python -m src.eval.runner`) |
| `src/eval/aggregate.py` | ALL aggregation (both headline means, per-stem, per-genre, absent levels) |
| `src/eval/verify_gates.py` | gates G1–G6 |
| `docs/eval_pipeline_build_report.md` | this report |

Dry-run outputs land in `experiments/exp003.0_260809_bsroformer_pilot/eval/`
(rows parquet + csv twin, aggregate tables, run log).

## The silence-tolerance key (the resolved open item)

**`configs/silence.yaml` → `silence.absent_rms_dbfs: -80.0`** — the one config home
for "is this silent?", resolving the Notion legal-pad item (*`SILENCE_EPS` has no
config home, found 2026-08-11*). Semantics: a reference stem whose full-song RMS
(channels pooled) is below −80 dBFS is classified **ABSENT** for evaluation.

- −80 dBFS RMS sits ~58 dB above the post-ingest DC residue (2⁻²³ ≈ −138 dBFS) and
  far below real content — the quietest *present* reference in the val dry run is
  listed in the dry-run section below.
- Absent-class references are constructed as **zeros in code**; a silent file in the
  ingest store is never read as a target.
- `scripts/verify_coherent_sampler.py`'s `--silence-eps` (a peak-amplitude tolerance,
  different quantity) still carries its own CLI default `1e-6`. Migrating it onto this
  config file is a follow-up, deliberately not done in this build (out of scope; it
  would change an existing verified script). Same for `configs/leakage_analysis.yaml`'s
  `silence_energy_eps`.

## Eval-job config schema (`configs/eval/*.yaml`)

One eval config = one scoring run = one long-format rows parquet.

```yaml
run_id: eval_exp003.0_ep59_val_sumstem   # names the rows parquet
checkpoint: experiments/.../model_bs_roformer_ep_59_si_sdr_10.7013.ckpt  # any arm, never hardcoded
model_config: configs/exp003.0_bsroformer_pilot.yaml  # MSST config; also supplies classes,
model_type: bs_roformer                               #   Σstem tree root, source manifest
targets:                                 # any combination of split × variant
  - {split: val, variant: sumstem}       # val (91) | test (135) × sumstem | master
metrics: [si_sdr, usdr]                  # registry names; csdr = registered stub
usdr_eps: 1.0e-7                         # MDX'21 eps (si_sdr keeps MSST's internal 1e-8)
metric_cap_db: 100.0                     # saturation cap — perfect reconstruction reports
                                         #   +100.0 dB, never inf
silence_config: configs/silence.yaml     # ← THE silence-tolerance home
eval_manifest: manifests/parquet/eval_manifest.parquet   # frozen split, read-only
out_dir: experiments/exp003.0_260809_bsroformer_pilot/eval
seed: 42
device: cuda:0
render:                                  # optional listening render of NAMED songs only
  enabled: false                         # default off — full-run rendering would be TB-scale
  songs: []
  out_dir: null
```

Variant semantics:

- **sumstem** — mixture + class targets read from the prebuilt Σstem tree
  (`sumstem_eval.out_root` of the model config): the exact files training-time
  validation scored.
- **master** — mixture = publisher master from the ingest store; references summed
  on the fly from the ingested per-instrument stems (source manifest rows, modeled
  classes only, trim-to-shortest, mono→stereo center) at **native scale, no peak
  gain** — the master number deliberately measures against the actual recorded stems,
  mastering residual included. uSDR is scale-dependent, so the Σstem peak gain
  (0.99/peak of the Σ mixture) must not be applied to master-variant references.

Row schema (one row per run × split × variant × song × class):
`run_id, checkpoint, split, variant, song_id, genre_sub, stem_class, status
(present|absent), si_sdr, usdr, pred_energy_dbfs, ref_rms_dbfs, duration_sec,
config_hash, git_commit, seed, timestamp`.

- `pred_energy_dbfs` and `ref_rms_dbfs` are populated for *all* rows (`ref_rms_dbfs`
  is NaN when no stem file exists at all); SDR columns are NaN on absent rows.
- Naming follows the repo's one-concept-one-word rule where it conflicts with the
  task wording: the Σstem variant is `sumstem` (repo word, not `sigma_stem`) and the
  genre column is `genre_sub` (as in every existing table, not `genre`).

## Metric definitions as implemented

- **si_sdr** — copied **verbatim** from MSST (`external/msst/utils/metrics.py`),
  including its non-standard per-element eps placement and `eps = 1e-8`, because gate
  G5 requires bit-reproducing training-time numbers. Channels and time pooled into
  one scalar per (song, class); no chunking.
- **usdr** — `10·log10((Σs² + eps)/(Σ(s−ŝ)² + eps))`, eps `1e-7` from config
  (MDX'21), channels + whole song summed, not scale-invariant, no gain matching, no
  filtering, no alignment. Computed in float64.
- **cap** — both metrics clamped at `metric_cap_db = 100.0` dB. The eps terms already
  make every value finite (a perfect reconstruction's raw values are ~+110–120 dB,
  set by signal energy vs eps); the cap replaces that energy-dependent saturation
  with one documented number. No real separation result approaches it.
- **csdr** — registered in the metric registry, raises `NotImplementedError` naming
  museval (BSSEval v4) as the intended backend. museval was NOT added as a dependency.
- Present-pair scoring first trims reference and estimate to their common length
  (MSST's rule; covers 판소리 stem tails and master-length mismatches).

## Gate results

Run: `uv run python -m src.eval.verify_gates --config configs/eval/exp003.0_val_sumstem.yaml --rows <rows.parquet>`
(G1–G4 exercise the real scoring path — `classify_and_score` + `build_metrics` — on
seeded synthetic signals, not a re-implementation.)

| gate | result | evidence |
|---|---|---|
| G1 perfect reconstruction | **PASS** | `si_sdr = usdr = 100.00` (the documented cap; raw eps-limited values ~+110–120 dB sit above it, never inf) |
| G2 estimate = 0.5×ref | **PASS** | `usdr = 6.0206` (target 6.02 ± 0.01); `si_sdr = 100.00` (scale-invariant → saturates) |
| G3 estimate = zeros | **PASS** | `usdr = 0.0000` exactly; (`si_sdr = −3.3` on the same pair — unconstrained by the gate, see G6 notes) |
| G4 absent routing | **PASS** | ref at DC-residue level (RMS −138.5 dBFS < −80) → `status=absent`, no SDR, `pred_energy_dbfs = −40.0` populated; missing-file path routes identically |
| G5 regression | **see below** | pipeline ≡ MSST's own scorer bit-exactly; checkpoint history differs ≤0.035 dB (training-context artifact) |
| G6 invariant | **see dry run** | checked on every present row of the real val run |

### G5 in full (the one gate that needed investigation)

Target: reproduce the training-time per-song SI-SDR frozen in the ep59 checkpoint's
own eval history (`all_metrics`, song identity recovered via
`src/analysis/read_msst_checkpoint.py`). Song: `0029_정악_풍류음악` (first in MSST's
traversal order, 6 present classes). Side by side:

| stem_class | checkpoint (epoch_59) | new pipeline | Δ |
|---|---|---|---|
| 기타 | 10.3221 | 10.3489 | +0.0268 |
| 대금 | 15.0824 | 15.0860 | +0.0036 |
| 아쟁 | 14.1163 | 14.1264 | +0.0100 |
| 타악기 | 24.5168 | 24.5316 | +0.0148 |
| 피리 | 9.7546 | 9.7538 | −0.0008 |
| 해금 | 9.6336 | 9.6680 | +0.0343 |

The deltas exceed the 0.01 dB gate tolerance, so per the gate rule I stopped and
investigated **without adjusting either side**:

1. The new pipeline is **self-reproducible to all four decimals** across repeated
   runs (two identical G5 executions).
2. **MSST's own `valid.py`, run standalone today** on the same checkpoint and song,
   produces the new pipeline's numbers **exactly** (10.3489 / 15.0860 / 14.1264 /
   24.5316 / 9.7538 / 9.6680) — not the checkpoint's.

Conclusion: the new pipeline is bit-faithful to the current scoring tool; the
≤0.035 dB residual is a property of the *live training process's* GPU kernel context
(MSST sets `torch.backends.cudnn.deterministic = False`; the in-training eval ran
inside a process with active training state), not of either implementation. Every
standalone re-scoring — including MSST's own — lands on the new numbers. Checkpoint
selection is unaffected (differences are two orders of magnitude below the ±0.5–1.4 dB
epoch-to-epoch val noise). Full-set corroboration on all 91 songs is in the dry-run
section below.

### G5 corroborated on the full val set

After the dry run, all 521 present (song, class) pairs were compared against the
checkpoint's epoch_59 history (count assertion inside `build_per_song` passed —
traversal mapping is sound):

- matched pairs 521/521 · |Δ| mean 0.057 dB, median 0.029 dB, max 2.66 dB
- mean-of-class-means: checkpoint **10.7013** vs pipeline **10.7301** (+0.03 dB)
- the max-Δ row (0897_창작국악 아쟁, checkpoint −12.06 vs pipeline −9.40) was re-run
  through **MSST's own valid.py standalone: −9.4006** — bit-equal to the pipeline.
  Even the worst outlier is a training-context artifact, shared by every standalone
  re-scoring, concentrated on already-low-scoring pairs where small kernel
  differences move the tiny error signal the most.

## Val-split dry run (exp003.0 ep59, Σstem variant)

91 songs × 9 classes = **819 rows (521 present, 298 absent)** in ~24 min on one GPU
(~12× realtime, BS-RoFormer 8 s chunks, 50 % overlap, fp32). No separated audio
written — rows only. Output:
`experiments/exp003.0_260809_bsroformer_pilot/eval/eval_exp003.0_ep59_val_sumstem.parquet`
(+ csv twin, aggregate tables, `dryrun_val_sumstem.log`).

Headline aggregations (both computed by `src/eval/aggregate.py` from the rows
parquet, no re-inference):

| metric | mean of per-class means | pooled over pairs |
|---|---|---|
| si_sdr | **10.73** | 11.84 |
| usdr | **11.38** | 12.38 |

(si_sdr 10.73 / 11.84 vs the known +10.70 headline and the paper's pooled 11.81 —
consistent to within the G5 context delta.)

Per-stem, si_sdr vs usdr side by side (means over present songs):

| stem_class | n songs | si_sdr | usdr |
|---|---|---|---|
| 가야금 | 68 | 9.68 | 10.24 |
| 거문고 | 64 | 7.75 | 8.64 |
| 기타 | 23 | 8.96 | 9.60 |
| 대금 | 72 | 13.30 | 13.66 |
| 아쟁 | 59 | 8.20 | 9.09 |
| 양금 | 9 | 6.30 | 7.65 |
| 타악기 | 88 | 20.42 | 20.50 |
| 피리 | 71 | 10.21 | 10.87 |
| 해금 | 67 | 11.76 | 12.19 |

Absent-class false-activation levels (`pred_energy_dbfs`, the model's output RMS
where the class is not in the song — lower = better silence):

| stem_class | n absent | median | max (worst) |
|---|---|---|---|
| 가야금 | 23 | −96.9 | −65.3 |
| 거문고 | 27 | −103.0 | −75.6 |
| 기타 | 68 | −88.8 | **−39.9** |
| 대금 | 19 | −109.6 | −67.5 |
| 아쟁 | 32 | −101.7 | −63.0 |
| 양금 | 82 | −90.2 | −73.6 |
| 타악기 | 3 | −110.1 | −61.6 |
| 피리 | 20 | −107.4 | −80.0 |
| 해금 | 24 | −108.1 | −95.5 |

The model is strikingly quiet on absent classes (medians −90…−110 dBFS, i.e. at or
below the ingest silence floor). The one notable false activation is 기타 at
−39.9 dBFS on one song — audible-ish energy on a class the song doesn't contain;
worth a listen when convenient. The quietest *present* reference in the run sits at
−52.7 dBFS RMS, ~27 dB above the −80 dBFS absent threshold — the tolerance cleanly
separates the populations with a wide dead zone.

## G6: the invariant `si_sdr >= usdr` is FALSE in practice — reported, not "fixed"

**499 of 521 present rows violate it** (median violation −0.38 dB, worst −9.18 dB;
only 22 rows satisfy it). This is not an implementation bug — the assumed invariant
does not hold mathematically for this metric pair:

- Decompose the estimate against the reference: ŝ = βs + e⊥ with
  β = ⟨ŝ,s⟩/‖s‖². Then SI-SDR = β²‖s‖²/‖e⊥‖² while
  uSDR = ‖s‖²/((1−β)²‖s‖² + ‖e⊥‖²). Algebra gives
  SI-SDR ≥ uSDR ⟺ β²(1−β)²‖s‖² + β²‖e⊥‖² ≥ ‖e⊥‖² — guaranteed only for β ≥ 1.
  For an under-scaled estimate (β < 1), SI-SDR's numerator shrinks by β² and the
  inequality generically flips.
- Measured on a real song, every stem has **β between 0.91 and 0.997** — the model
  systematically under-shoots the reference scale (typical for L1-trained
  separators), so si_sdr < usdr is the *expected* outcome here.
- The familiar guarantee points the other way and belongs to a different metric:
  museval's BSSEval SDR optimizes a 512-tap distortion filter (a generalization of
  gain matching), which makes *that* SDR ≥ SI-SDR. Plain uSDR does no gain matching
  by definition (per this build's spec), so no ordering holds.
- Synthetic pre-echo of the same effect: gate G3's zero estimate gives usdr = 0 but
  si_sdr = −3.3.

Per the gate instructions the check is implemented and asserts as specified:
`src/eval/verify_gates.py --rows …` reports **G6 FAIL** with the offending rows,
and `src/eval/aggregate.py` prints the violations on every run (`--strict` makes
them fatal). Nothing was altered to make it pass. **Recommendation:** drop the
invariant (or replace it with the one that is actually guaranteed,
`si_sdr <= csdr_museval`, once csdr lands) — your call, the gate stays as specified
until then.

**Update 2026-08-30:** decided — the gate is dropped. Removed from
`src/eval/verify_gates.py` (gates are now G1–G5) and the matching violation
check removed from `src/eval/aggregate.py`; both scripts point back at this
section for the history. The rows parquet still carries both metrics per row, so the
ordering can always be inspected after the fact.

## Deviations & follow-ups

1. **G5 tolerance not met vs the checkpoint history** (≤0.035 dB single-song,
   median 0.03 dB full-set) — investigated, root-caused to the live-training kernel
   context, pipeline proven bit-equal to MSST's own scorer run today. Neither side
   adjusted.
2. **G6 invariant is false** for si_sdr vs uSDR (see above) — reported with proof;
   gate faithfully FAILs.
3. **Naming**: `sumstem` (not `sigma_stem`) and `genre_sub` (not `genre`), per the
   repo's one-concept-one-word rule.
4. **usdr eps** is 1e-7 (MDX'21, from config) while MSST's internal `sdr()` uses
   1e-8 — irrelevant numerically at song energies, but our uSDR is the spec's
   formula, implemented in `src/eval/metrics.py`, not a call into MSST.
5. **Master variant**: code path built and smoke-tested on real val songs (reference
   construction, trim, absent routing, scoring — incl. the 0905 solo-해금 outlier,
   which hit the +100 dB cap exactly when fed itself as a dummy estimate). No full
   GPU pass was run (only the val Σstem dry run was in scope).
6. **Test split**: untouched, per instructions. Note for the future go-signal: the
   test Σstem tree does **not** exist yet
   (`data/gugak_ensemble_71955/sumstem_9stem/test/` — build with
   `src/data/build_sumstem_eval.py --split test` first, ~14 h of audio); the master
   variant needs no prebuild.
7. **Not migrated** (scope): `verify_coherent_sampler.py --silence-eps` and
   `leakage_analysis.yaml silence_energy_eps` still carry their own values; folding
   them onto `configs/silence.yaml` is a clean follow-up.
8. `table_paths()`/`find_root()` remain duplicated across 11/5 files repo-wide; the
   new code deliberately avoids adding more copies (paths handled locally) but the
   dedup itself stays a follow-up.

## Proposed commit (not executed — working tree left for review)

All new files, no modifications:
`configs/silence.yaml`, `configs/eval/exp003.0_val_sumstem.yaml`, `src/eval/*`,
`src/eval/runner.py`, `src/eval/aggregate.py`, `src/eval/verify_gates.py`,
`docs/eval_pipeline_build_report.md`, plus the dry-run rows parquet + aggregate
parquets under `experiments/exp003.0_260809_bsroformer_pilot/eval/` (csv twins per
existing convention). Suggested message:
`[eval] config-driven eval pipeline: si_sdr+usdr, streaming, raw rows + separate aggregation`
