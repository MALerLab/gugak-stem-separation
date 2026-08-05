# source_manifest v2 — provenance

Built 2026-08-05. Supersedes v1 for every experiment from exp002 onward.

| | path |
|---|---|
| v1 (historical, what exp001 / exp001.2 trained against) | `manifests/parquet/source_manifest.parquet` |
| **v2 (current)** | `manifests/parquet/source_manifest_v2.parquet` |
| exclusion rules | `configs/manifest_exclusions.yaml` |
| builder | `src/data/build_source_manifest.py --exclusions ...` |
| verifier | `scripts/verify_manifest_v2.py` |

v1 is **not** overwritten or renamed. It stays exactly where it was so exp001's training
record remains reproducible; an experiment selects its version through
`gugak_mix.source_manifest` in its YAML, never by editing a path in code.

## What changed

v2 is v1 minus **16 rows**, all of them files defective at the publisher (AI Hub 71955),
found by the 2026-07-29 integrity sweep. Nothing else differs — verified, see below.

### 1. `0714_민속악_민요` — 대금, 아쟁, 피리 stems (3 rows)

These three stems are **sample-identical to one another** (maximum absolute sample
difference exactly 0.0) and carry an ensemble mix rather than their instrument parts.
The shared signal is the song's master scaled by ≈1.33× — correlation 0.9965, residual
8% of RMS after best-fit scaling. Their raw md5 hashes differ only in header bytes.

The real 대금/아쟁/피리 parts were overwritten at the publisher and are unrecoverable. The
song's other four stems — 가야금, 거문고, 장구, 해금 — are genuine and **stay in the pool**;
so does its master. The song is in the train split, so this was never an eval-leakage
issue, only label noise in the draw pool.

> **Correction to the Notion write-up.** Notion (Dataset 1 → preprocessing notes) describes
> these as "byte-exact copies of that song's master mix". Two details are off: they are
> byte-exact copies *of each other*, not of the master, and the relationship to the master
> is a ~1.33× gain rather than an exact copy. The conclusion — three stems carrying a mix
> instead of instrument parts — is unaffected. Notion is not edited from this session.

### 2. `0885_창작국악_창작국악` — the entire song (13 rows: 1 master + 12 stems)

`0885` is `0886_창작국악_창작국악` looped seven times: it claims 336 s and delivers the same
48 s of unique music seven times over. Trimming would not help, because the trimmed
remainder is that identical 48 s. Excluding 0885 while keeping 0886 removes the 2×
sampling weight that music would otherwise carry in the draw pool. Both songs are in the
train split.

### Deliberately NOT excluded: `0905_창작국악_창작국악`

Its single 해금 stem is bit-identical to its master. This is **benign and confirmed by
listening** — a genuine solo 해금 piece, so the mix really is the instrument. It is also a
val song and the val split is frozen. Left entirely alone; both its rows survive in v2.
Recorded here and in the exclusions config so nobody "fixes" it later.

## Row counts

| | v1 | v2 | Δ |
|---|---|---|---|
| total rows | 16,615 | 16,599 | −16 |
| 71955 songs | 903 | 902 | −1 |
| 71955 masters | 903 | 902 | −1 |
| 71955 stems | 5,767 | 5,752 | −15 |
| 71955 stem hours | 368.78 | 367.39 | −1.39 |
| 71470 rows (untouched) | 9,945 | 9,945 | 0 |
| **train draw pool** (71955 · train · 9 modeled classes) | 4,265 sources / 277.90 h | 4,250 sources / 276.51 h | −15 / −1.39 h |

The train draw pool loses **0.35% of its sources and 0.50% of its hours** — the point of
this change is correctness, not volume.

Per-class train-pool source counts in v2: 가야금 508 · 거문고 470 · 기타 185 · 대금 566 ·
아쟁 523 · 양금 72 · 타악기 819 · 피리 568 · 해금 539.

## Verification

`uv run python scripts/verify_manifest_v2.py` — all four assertions pass:

1. **Exact removals.** The set of `file_id`s missing from v2 equals the set the exclusion
   rules declare, with nothing added. 16 rows, listed individually in the script output.
2. **Survivors untouched.** Every one of the 16,599 surviving rows is identical to its v1
   counterpart across all 41 columns.
3. **Keep-list intact.** `0886_창작국악_창작국악` 13 → 13 rows, `0905_창작국악_창작국악` 2 → 2
   rows, same `file_id`s in both cases.
4. **Pool sanity.** All nine modeled classes still have drawable train-split sources.

A fifth check ran once by hand before the build and is worth recording: rebuilding the
manifest from `ingest_manifest ⋈ audio_qc_ingest_* ⋈ stem_taxonomy.yaml` **with no
exclusions** reproduces v1 exactly (16,615 rows, all 41 columns equal). So nothing
upstream — ingest, QC scan, or taxonomy — has drifted since v1 was written, and every
difference between v1 and v2 is attributable to the exclusion rules alone.

## Known scope limit

The exclusions apply to `source_manifest` only. `chunk_activities.parquet` — used solely
to derive the *measured* mix-density histogram — still contains 0885's windows (33 of
15,295 train windows at 10 s, ≈0.2%, and duplicated 7× within themselves) and 0714's 32
windows, whose audible-class count is inflated by the three mix-carrying stems. This does not
affect exp002, which draws density **uniformly** and never reads that table. If the
measured density mode is ever revived, regenerate `chunk_activities` against v2 first.
