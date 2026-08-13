# exp003.0 — launch report

**Run:** `exp003.0_260809_bsroformer_pilot` · BS-RoFormer, 9 gugak classes, manifest v2,
incoherent uniform-density mixing, 8 s segments, bf16 train / fp32 eval.
**Launched:** 2026-08-09 17:08:22 KST · GPU 0 · tmux session `exp003-0` · PID 95210
(wrapper 95206) · wandb offline run `offline-run-20260809_170826-jvspw9oy`.
**Status at handover:** running, ~1.64 loader steps/s, 0 non-finite steps.

⚠️ **This is a pilot, not the canonical BS-RoFormer experiment.** It is `exp003.0` and not
`exp003` on purpose. Its job is to prove the pipeline stands up end to end and to produce
a first sanity curve. The learning rate is an untuned guess. **Numbers from this run are
not reportable results.**

Its second job is diagnostic. exp002 and its seed twin both show 가야금 · 기타 · 양금
collapsing toward silence while 대금 gains. HTDemucs has a shared decoder; BS-RoFormer
gives every source its own mask estimator. If head death vanishes here, that points at
the shared decoder. If it appears anyway, the cause is in the loss or the data.

---

## Hard gates

| # | Gate | Result | Measured |
|---|------|--------|----------|
| 1 | Head reshape seeded + reproducible | **PASS** | two independent builds at seed 42 bit-identical across all 2,595 tensors; 363 tensors / 33,152,772 params transferred (13.0%), 2,232 tensors / 221,741,640 params fresh (87.0%) |
| 2 | bf16 active, GradScaler disabled | **PASS** | run log: `AMP: use_amp=True amp_dtype=bfloat16 grad_scaler_enabled=False` |
| 3 | STFT / iSTFT pinned to fp32 | **PASS** (needed a new patch) | direct dtype probe over every `torch.stft`/`torch.istft` call: `stft:torch.float32`, `istft:torch.complex64` — no half type anywhere |
| 4 | Eval precision fp32 | **PASS** | `inference_amp_dtype: float32`; forced a second fork patch, see below |
| 5 | Manifest v2 resolves, exclusions gone, all 9 classes drawable | **PASS** | 16,599 rows; 16 v1 file_ids absent; **0** of them drawable; pool 4,250 sources, smallest class 양금 = 72 |
| 6 | `mixture ≡ Σ(targets)` | **PASS** | max abs error **1.79e-07** over 96 items (exp002 measured 2.4e-07) |
| 7 | Forward + backward finite | **PASS** | 128/128 steps finite, 0 non-finite params after the update, grad-norm finite |
| 8 | `training.instruments == gugak_mix.classes`, same order | **PASS** | identical lists, order preserved |
| 9 | Three seeds pinned, recorded, logged | **PASS** | all 42 → `seeds.txt` |
| 10 | Non-finite step counter on | **PASS** | `nonfinite_running = 0` present in the live wandb history, logged every step |

Density sanity draw (full 200,000-draw verification waived — sampler unchanged from
exp002): 20,000 draws, support exactly {1..9}, mean **5.0182** against a uniform
expectation of 5.0.

---

## Two fork patches this run required

Both are in `external/msst`, both are new, and **neither was anticipated by the brief.**

### 1. fp32 spectral path for BS-RoFormer (`models/bs_roformer/bs_roformer.py`)

The exp001.1 numerics patches (`amp_dtype`, bf16-safe GradScaler, per-step telemetry)
live in `train.py` and `utils/model_utils.py`, so they are model-agnostic and fired for
this model path unchanged. **The fp32 STFT pinning did not** — that patch was written
into `models/demucs4ht.py` and is HTDemucs-only. BS-RoFormer had an entirely unpinned
spectral path.

Three sites now pinned with `autocast(enabled=False)`: the input STFT, the complex mask
multiply + iSTFT, and the internal multi-resolution STFT loss. The `.float()` on the mask
is load-bearing rather than defensive — `torch.view_as_complex` has no bfloat16 overload
at all, so this would have been a hard crash, not a silent precision loss.

### 2. fp32 attention fallback (`models/bs_roformer/attend.py`)

**There is no fp32 flash-attention kernel.** On compute capability ≥ 8.0 (our Blackwell
cards) `attend.py` selects flash **and explicitly disables the math and mem-efficient
backends**. Training is fine because bf16 autocast makes q/k/v bf16 — but fp32 eval
raised `RuntimeError: No available kernel` and every validation would have died.

The patch falls back to the math / mem-efficient backends when the inputs are not a half
type. Training behaviour is unchanged by construction: under bf16 autocast the original
flash config is still selected.

**This is the single most valuable thing this session found.** It was caught only because
the eval path was exercised against the start checkpoint before launch rather than being
discovered ~7 hours in, at the first in-run validation.

---

## Head reshape — what actually changed

BS-RoFormer splits the spectrum into 62 frequency bands, runs alternating time/frequency
transformers over them, and then hands the shared feature stack to **one separate mask
estimator per source**. Each estimator is 62 small MLPs — one per band — that decide how
much of each band belongs to that source.

The surgery: **keep the trunk, rebuild all nine estimators.**

| Part | Fate | Tensors | Params |
|---|---|---|---|
| `band_split` + `layers.0-7` + `final_norm` (shared trunk) | transferred from MUSDB | 363 | 33,152,772 (13.0%) |
| `mask_estimators.0-8` (nine per-source heads) | **freshly initialised, seed 42** | 2,232 | 221,741,640 (87.0%) |

The naive load would have "worked" and been wrong. Unlike HTDemucs, where the 4→9 change
is a shape mismatch that a name+shape transfer resolves by itself, BS-RoFormer's
estimators are all *the same shape* — so a stock load silently fills slots 0–3 with the
pretrained drums/bass/other/vocals heads, handing 가야금·거문고·기타·대금 a head start and
leaving 아쟁·양금·타악기·피리·해금 from scratch. That asymmetry sits on exactly the axis
this run is trying to read. All nine were reinitialised instead (your call this session).

**Consequence to be honest about: only 13% of the parameters are actually warm-started.**
BS-RoFormer keeps roughly three quarters of its capacity in the per-source heads, so
"fine-tune from a MUSDB checkpoint" here really means "reuse a pretrained feature extractor
and train nine new heads". Measured starting point on three val songs: **−33.6 dB average
SI-SDR.** Expect a long warm-up.

### Where the parameters actually sit (counted, not quoted)

| | 4-stem pretrained | 9-stem (ours) |
|---|---|---|
| shared trunk (`band_split` + `layers.0-7` + `final_norm`) | 33,152,772 (25.2%) | 33,152,772 (13.0%) |
| per mask estimator | 24,637,960 | 24,637,960 |
| all mask estimators | 4 × = 98,551,840 (74.8%) | 9 × = 221,741,640 (87.0%) |
| **total** | **131,704,612** | **254,894,412** |

Heads scale exactly linearly — the per-head cost is identical in both, so 9 heads cost
98.55M × 9/4 = 221.7M. The *total* actually grows more slowly than the head count (2.25×
more heads, 1.94× more model) because the trunk does not scale.

⚠️ **Corrects a standing error in our notes.** Both the exp001.1 bf16 report and Notion put
BS-RoFormer at "~93 M parameters". That figure was never measured and matches nothing in
this repo — the nearest MSST BS-RoFormer config (dim 192 / depth 6, 1 stem) counts
24,591,340. The real pretrained 4-stem checkpoint is **131.7M**. Read against the correct
baseline, our 254.9M is not a surprise or a blow-up; it is exactly what adding five more
per-source heads costs.

**Why one head is 24.6M** (the useful thing to be able to say out loud): a mask estimator is
not one network but **62 small MLPs, one per frequency band**, each taking the shared
384-dim feature vector up to a 768-dim hidden layer and back down to its band's bins.

- 62 × (384 × 768) hidden layers = **18,284,544** ← dominant, and paid once per band
  *regardless of band width*, so a 2-bin band in the low end costs the same hidden layer as
  the 129-bin band at the top
- the 62 output layers together = **6,297,600**
- biases etc = 55,816

So **one single head is 0.74× the size of the entire shared trunk**, and with nine of them
87% of this model is per-source machinery. That is the architectural claim in one number:
where HTDemucs gives its sources a shared decoder and a handful of distinguishing output
channels, BS-RoFormer gives each source three-quarters of a trunk's worth of its own
parameters.

---

## Batch geometry

Effective batch was held at **32**, matching exp002, so the physical batch was the only
free variable.

| Physical batch | Peak GPU memory | Throughput | Verdict |
|---|---|---|---|
| 2 | **41.5 GiB** | 4.07 items/s | chosen |
| 4 | **78.6 GiB** | 4.62 items/s (+13%) | fits, but 83% of the card for 13% |
| 6 | OOM | — | — |
| 8 | OOM | — | — |

*(Re-measured 2026-08-10 on an idle card over 480 items per setting, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Corrects the initial reading**: batch
4's peak was first described as "still creeping upward" — it is not unbounded drift, it
plateaus at ~78.6 GiB and stays there. Batch 4 is therefore viable, just thin on headroom;
and since it buys only 13% throughput for 1.9× the memory, batch 2 remains the right
choice. Single-GPU batch tuning is effectively exhausted — the card is compute-saturated
at batch 2.)*

**Chosen: physical 2 × grad-accum 16 = effective 32.** Live peak in the running job:
44.2 GiB of 95.6 GiB. Batch 3 would have fit but cannot reach 32 with an integer
accumulation count. `num_steps` was set to 40,000 loader iterations so an "epoch" is
still 2,500 optimizer steps and still 80,000 training excerpts — the same amount of data
exp002 saw per epoch.

Measured throughput **1.64 loader steps/s** (`step_seconds ≈ 0.61`) → **~6.8 h per epoch**,
so the first in-run eval lands roughly **6.8 h after launch**.

---

## Deviations from the brief

- **Inference overlap.** Spec says 25%; MSST's `num_overlap` is an integer divisor, so 25%
  is not expressible. Used `num_overlap: 2` → 50% overlap, the same call exp002 made.
  Quality ≥ spec. Cross-fade is MSST's linear ramp over chunk/10, as specified.
- **`--use_standard_loss` added.** Without it MSST routes `bs_roformer` through the model's
  *internal* loss (waveform L1 **plus** a multi-resolution STFT term), which would have
  silently changed the loss relative to exp002. The spec says L1 on waveforms, unchanged,
  so the flag enforces it. This is not optional and is commented as such in `launch.sh`.
- **Checkpoint had to be re-downloaded.** `model_bs_roformer_ep_17_sdr_9.6568.ckpt` was not
  on disk — the zero-shot baseline's copy went in the 2026-07-24 restructure. Re-fetched
  from MSST release v1.0.12 (503 MiB, md5 `23e2d3dbe75dc73c5f8a712ffb542fc2`) to
  `~/storage/gugak-stemsep-experiments/pretrained/`, along with its architecture config.
- **Clean tree waived** per brief. HEAD + dirty list recorded in `git_commit.txt`, now also
  including the MSST submodule's dirty list, since this run depends on two uncommitted
  patches there.

---

## Waived, as instructed

Full 200,000-draw density verification · field-by-field wandb config provenance diff ·
clean git tree · any tuning of lr/batch/schedule/augmentation · `chunk_activities`
regeneration at 8 s (uniform density reads no table, so it is window-length independent) ·
chunk loudness reference at 8 s (loudness matching is off — no solo pool) · a stopping
rule (MSST has no early stopping; stop this by hand with SIGINT).

---

## Not verified — state plainly

- **The first in-run eval has not happened yet** (~6.8 h out). The eval *path* was proven
  separately against the start checkpoint on a 3-song probe set: it ran to completion and
  printed per-class SI-SDR for all nine classes individually, which is the per-head
  visibility this run depends on. What is unverified is the full 91-song eval — its
  wall-clock (extrapolates to ~15–25 min) and its peak memory under fp32.
- **Loss curve health beyond the first ~150 steps.** At handover the loss is flat around
  1.07–1.08 (MSST prints 100×L1), which is what a randomly initialised set of heads should
  look like. Whether it descends is unobserved.
- **wandb project name.** As expected, MSST hardcodes `project='msst'`; `WANDB_ENTITY`
  applies but `WANDB_PROJECT` does not. Not patched, per brief. Fix at sync time:
  `wandb sync -p gugak_stem_separation wandb/offline-run-20260809_170826-jvspw9oy`.
- **Whether lr 1e-5 is anywhere near right.** It is a declared guess. With 87% of the model
  freshly initialised there is a real chance it is too *low*, not too high.

## Improvised / guessed

- **All-fresh head policy** — a genuine design decision, not something the brief settled.
- **`num_steps: 40000`** — derived, not specified: chosen so the eval cadence stays at
  2,500 optimizer steps once the physical batch dropped to 2.
- **The attention fallback patch** — invented this session in response to a crash. It is
  narrow and preserves the training path exactly, but it is new code on the eval path with
  one 3-song test behind it.
