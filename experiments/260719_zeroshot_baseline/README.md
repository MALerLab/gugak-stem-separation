# zeroshot_baseline_260719

> ⚠️ **SUPERSEDED (2026-07-24) — historical footnote.** In the 2026-07-24 dataset restructure the
> seed-42 manifest, the `data/_val/` prediction audio, and the mixtures this run depended on were
> deleted; the dataset dir was renamed to `gugak_ensemble_71955/`. These metrics are **not
> reproducible as-is and not comparable** to future runs (a fresh zero-shot under the settled
> stem-class scheme is planned — see Notion). Kept only as a record that a zero-shot was done and
> what it showed. Relevancy minimal.

Zero-shot Western separators on gugak — **pipeline validation + OOD floor**, no fine-tuning.

## Setup
- **Val set:** 91 songs (our seed-42 stratified split).
- **Mixtures:** Σ(stems) → trim-to-shortest → mixture peak-norm 0.99 (native SR → model SR at inference).
- **Models:**
  - `htdemucs` — native `demucs`, 4-stem (drums/bass/other/vocals).
  - `bsroformer` — MSST 4-stem MUSDB checkpoint `model_bs_roformer_ep_17_sdr_9.6568`.
- **Framing:** Western heads ≠ gugak stems. Only **타악 ↔ drums** loosely maps → scored with SI-SDR.
  Everything else read as an energy-distribution signal (where does gugak energy land?).

## Pipeline
`scripts/infer_val.py` (preds → `data/_val/`) → `scripts/eval_val.py` (→ `metrics.parquet` + `.csv`
here) → `scripts/fig_zeroshot.py` (→ `figures/fig_zeroshot_baseline.png`) →
`scripts/wandb_log_zeroshot.py` (offline run, **not yet synced**).

## Headline findings
- **Melodic gugak has no Western home** — energy dumps into "other" (htdemucs 69% / bsroformer 52% avg).
- **타악↔drums SI-SDR negative overall** (htdemucs median **−2.1 dB**, bsroformer **−9.8 dB**) — worse
  than useless on average.
- **창작국악 is the bright spot** — SI-SDR **positive** (htdemucs **+9.8**, bsroformer **+12.1 dB** median):
  where gugak is Western-adjacent, the percussion head genuinely transfers.
- **htdemucs > bsroformer zero-shot** on the one honest metric — but bsroformer remains the stronger
  architecture + exp002 warm-start base; it just lands rougher OOD.
- **Hardest genres:** 궁중음악 / 풍류음악 / 민요 (most out-of-domain).
- ⇒ Fine-tuning is justified and necessary; domain gap now quantified.

## Reproduce the figure
```bash
uv run python scripts/fig_zeroshot.py   # metrics.parquet -> figures/fig_zeroshot_baseline.png
```

## Caveats
- SI-SDR only computed where a 타악 stem exists (e.g. 창작국악 = 15 of 19 songs).
- Ranges are wide (산조 spans −30 to +33 dB) — read medians, not single songs.
- wandb run is **offline**; `wandb sync wandb/offline-run-*` once a team/entity is chosen.
