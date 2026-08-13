# exp002.1 — seed twin of exp002 · launch report

Launched **2026-08-06 08:00:54 +09:00**, GPU 1, tmux `exp002_1`.
wandb offline run `evh0v0d3` (`wandb/offline-run-20260806_080057-evh0v0d3`).

## What this run is for

Two questions, one run.

**1. The init lottery (the primary reason).** HTDemucs has no per-source subnetworks —
every layer is shared, and the sources diverge only at the final conv of each decoder,
which emits `n_sources × audio_channels` channels that get reshaped into per-source
slices. Our nine "heads" are therefore nine thin projections reading one shared feature
space, and they were randomly initialized when the pretrained 4-source output layer was
reshaped to 9. Which slice claims which acoustic content may be partly an accident of
where each slice started, amplified by rich-get-richer dynamics.

- 가야금 poor in **both** arms by a similar margin → intrinsic to instrument + data; stop
  investigating initialization, look at register overlap and unison instead.
- 가야금 poor in **one** arm only → assignment stochasticity is real, and the warm-start
  ablation (exp002.2) becomes the interesting follow-up rather than a curiosity.

**2. The noise floor (by-product).** exp001.1 measured epoch-to-epoch val swing at roughly
±0.5–1.4 dB, which makes any experiment-to-experiment claim below ~1 dB unreadable today.
A matched pair gives a run-to-run range for the first time.

**Unplanned third value, as of launch day.** exp002 threw a large excursion at ep12 —
가야금 −12.3 dB, 기타 −12.8, 양금 −8.2, while 대금 gained +7.2, all in one eval, with zero
non-finite steps and training loss still falling (→ exp002's `LAUNCH_REPORT.md`). This arm
is now also a direct test of whether that excursion is competence-triggered (reproduces at
the same quality level) or seed-specific (absent here).

## Seeds — three surfaces, one knob

| surface | exp002 | exp002.1 | what it controls |
|---|---|---|---|
| `gugak_mix.seed` (config) | 42 | **43** | the incoherent-mix draw stream |
| `--seed` (trainer, launch.sh) | 42 | **43** | torch/numpy global, data order |
| `--seed` (init_start_checkpoint.py) | 42 | **43** | fresh init of the 9 output heads |

Rule: each new seed increments the base — exp002 = 42, twin = 43, exp002.3 = 44,
exp002.4 = 45. (exp002.2 deliberately held 42 to isolate the coherence variable.)

Surface 3 is load-bearing for question 1. Reusing exp002's `start_checkpoint.ckpt` would
give both arms bit-identical heads and leave the init lottery untested.

⚠️ **Deliberate caveat, do not "fix".** Moving the seed moves both data order and weight
initialization, so this strictly measures run-to-run stochasticity, not initialization
specifically. That is the intended scope for this pass. If 가야금 does move between arms,
isolating which of the two caused it is a cheap follow-up.

## Config diff — verified mechanically, not by eye

exp002's config was read from the config **actually logged in its wandb transaction log**
(`offline-run-20260805_095738-476pciph`), not reconstructed from a file. 114 leaf fields
compared; **2 differ**:

| field | exp002 | exp002.1 | |
|---|---|---|---|
| `gugak_mix.seed` | `42` | `43` | the experiment |
| `training.run_name` | `exp002_260805_htdemucs_v2_uniform_n` | `exp002.1_260806_twin` | run-scoped |

Everything else — manifest v2, `density_mode: uniform`, lr 1e-4, bf16, the 9-class list,
the whole architecture block — is identical.

> **Gotcha worth keeping.** The first diff run reported a *third* difference,
> `htdemucs.dconv_init` `0.001` vs `'1e-3'`. Both files carry the identical line
> `dconv_init: 1e-3`; the difference was in the *parser*. `yaml.safe_load` follows YAML 1.1
> and reads unpunctuated scientific notation as a **string**, while MSST loads htdemucs
> configs through `OmegaConf`, which reads it as a float. Verify a config with the loader
> the trainer actually uses, or you will manufacture differences that do not exist.

**Launcher:** with names, paths, seed and GPU mask normalized, every executable line
matches exp002's launcher. Only `--seed 43`, `CUDA_VISIBLE_DEVICES=1`, the paths, and one
echo string differ.

**Start checkpoint:** 533 tensors, **529 bit-identical** to exp002's, 4 differ — exactly
the reshaped output projections `decoder.3.conv_tr.{weight,bias}` (spectral branch,
36 = 9 sources × 4 under complex-as-channels) and `tdecoder.3.conv_tr.{weight,bias}`
(waveform branch, 18 = 9 sources × 2). The shared trunk is byte-for-byte identical between
arms, so the init lottery is isolated as cleanly as this architecture permits.

## Pre-launch checks

| check | result |
|---|---|
| GPU 1 free, not exp002's card | ✅ exp002 on GPU 0; GPUs 1–2 idle at launch |
| exp002 untouched | ✅ PID 61603 alive, config mtime still 08-05 09:40, 13 checkpoint files unchanged |
| same manifest as exp002 | ✅ both configs → `source_manifest_v2.parquet`, md5 `7be4a5c0…`, mtime 08-05 09:36 (predates exp002's launch) |
| GradScaler disabled under bf16 | ✅ `AMP: use_amp=True amp_dtype=bfloat16 grad_scaler_enabled=False` |
| STFT/iSTFT fp32 | ✅ `demucs4ht.py:446,455` under `autocast(enabled=False)` |
| `inference_amp_dtype: float32` | ✅ |
| non-finite step counter on | ✅ `nonfinite_running` / `nonfinite_steps` logged per step |
| one forward/backward, counter 0 | ✅ smoke: 0/4 non-finite, 0 non-finite params after update, grad_norm 0.0139, peak 36.99 GiB |
| checkpoint loads into the model | ✅ strict load, 41,996,006 params, 9 sources |
| wandb separate run, config + commit | ✅ `evh0v0d3`, config logged, `git_commit.txt` written by the launcher |

**Stagger.** exp002 launched ~22 h earlier, so the ≥20 min separation is satisfied many
times over. ⚠️ Throughput figures below are **contended** — two runs share the machine and
its dataloader I/O — and must not be read as clean benchmarks.

## Early telemetry

| | exp002 (GPU 0) | exp002.1 (GPU 1) |
|---|---|---|
| median step time | 0.455 s → **2.20 it/s** | 0.478 s → **2.09 it/s** |
| non-finite steps | **0** | **0** |
| GPU memory | 40,743 MiB | 40,741 MiB |

Twin runs ~5% slower than exp002 does alongside it; exp002 solo measured 2.21 it/s, so
sharing the machine costs little here. Memory matches to within 2 MiB, as a true twin
should. Early loss is high and the numbers look bad — the output heads are randomly
initialized and the model has to learn where to route nine classes first. That is warm-up,
not a bug.

## Stop policy

Unchanged from exp002: 60 epochs / 150k optimizer-step ceiling, ReduceLROnPlateau
(patience 5), manual plateau stop. **SIGINT only** — never `kill -9`.

## Comparison harness

`src/analysis/compare_runs.py` + `configs/analysis/compare_runs.yaml`. Aligns arms at matched
optimizer steps (never wall-clock, never epoch index), and reports the between-arm gap
**divided by each arm's own eval-to-eval swing** over the same window, so a gap inside the
noise is reported as such instead of as a number that looks like a finding. Highlights
가야금 and 양금; shows all nine. Reads MSST's `all_metrics` out of each arm's rolling
checkpoint, so it is read-only and safe against a live run.

**No conclusions yet** — both arms must be well past warm-up before the comparison means
anything. Default warm-up cutoff is 25,000 optimizer steps (10 eval cycles).
