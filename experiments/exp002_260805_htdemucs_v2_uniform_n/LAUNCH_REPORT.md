# exp002 — launch report

**Run:** `exp002_260805_htdemucs_v2_uniform_n` · launched 2026-08-05 09:57:36 KST · GPU 0 ·
tmux session `exp002` · wandb offline `wandb/offline-run-20260805_095738-476pciph`

exp002 is the **canonical HTDemucs baseline**. exp001 was a shakedown: it trained on a
manifest containing four publisher-defective files, switched precision mid-run after the
fp16 cliff incident, and continued from a resume whose optimizer carried a halved learning
rate. Nothing here is inherited from a checkpoint — every value is declared in the config
and the run starts fresh from a warm-start checkpoint.

## Config diff against exp001.2

Derived from `configs/exp001.2_htdemucs_9stem_resumed.yaml`, which was first verified
field-by-field against the config actually logged in its wandb transaction log
(`offline-run-20260802_141155-nqw026nj`): **113 keys, zero substantive differences** — the
only mismatch was `dconv_init` reading as the string `'1e-3'` on disk and the float `0.001`
in the log, a YAML scalar-parsing artifact inherited verbatim from the MSST musdb18
template. Verifying against the log rather than the launch config is what caught exp001.1's
trap, and the same trap is visible here: exp001.2's config declares `lr: 5e-5` while its
live optimizer carried **2.5e-5**, because `--load_optimizer` makes the restored state win.

Exactly four keys differ:

| Key | exp001.2 | exp002 |
|---|---|---|
| `training.run_name` | `exp001.2_260802_htdemucs_9stem_resumed` | `exp002_260805_htdemucs_v2_uniform_n` |
| `training.lr` | `5e-05` (live: 2.5e-5) | **`1e-04`** |
| `gugak_mix.source_manifest` | *(absent → v1 default)* | **`manifests/parquet/source_manifest_v2.parquet`** |
| `gugak_mix.density_mode` | *(absent → `measured`)* | **`uniform`** |

Everything else is byte-identical: 9 classes with pitched percussion quarantined, 71955
ensemble sources only, 10 s segments, activity-aware excerpt starts, −19 LUFS one-gain
mixture normalisation, L1 waveform loss with silent targets included, Adam + grad clip
L2 5.0, batch 8 × accum 4 = effective 32, eval every 2,500 optimizer steps, 60-epoch /
150k-step ceiling, EQAugment off, pitch-shift pool unused, `loudness_match` absent.

The launch differs too: **fresh**, with `--load_only_compatible_weights` and *no*
`--load_optimizer / --load_scheduler / --load_epoch`. That is what makes lr 1e-4 the rate
the run actually starts at.

## Deviations from the brief

1. **"Early stopping — patience 10 evals" is not implementable as specified.** MSST has no
   early-stopping mechanism at all; `training.patience` feeds only ReduceLROnPlateau. exp001
   was the same — it ran to its ceiling and was stopped by hand. Decision (user, 2026-08-05):
   run to the 150k ceiling with manual monitoring rather than patch a new stopping rule into
   the trainer immediately before the anchor run. The launch script documents the manual
   SIGINT stop policy.
2. **"exp001's current config (the fp32 arm)" was stale.** The current arm is exp001.2,
   which adopted bf16 on 2026-08-02 and ran until a SIGINT stop on 2026-08-03. Since exp002
   wants bf16 from launch anyway, exp001.2 is the correct base and is what was used.
3. **Launched from a dirty working tree** (user's call, 2026-08-05). `git_commit.txt` records
   HEAD `0ea3ace` plus the uncommitted-file list; wandb independently captured the same
   commit hash. Commits follow after launch.
4. **wandb project reads `msst`, not `gugak_stem_separation`** — see Findings below.

## Pre-launch verification

| Check | Result |
|---|---|
| GPUs | 3× RTX PRO 6000 all at 0 MiB / 0% — no stale process, no leaked CUDA context |
| tmux | no sessions before launch |
| Git | tree clean at `0ea3ace` before this session's edits; MSST submodule at pinned `8dbc33e`, no drift |
| Config | loads and resolves; `training.instruments == gugak_mix.classes`, same order |
| Classes | the expected 9 (가야금 거문고 기타 대금 아쟁 양금 타악기 피리 해금) |
| Manifest v2 | path resolves; 4,250 drawable train sources; all 16 excluded files absent from the pool; 0886's 12 stems present |
| Density uniform | 200,000 draws through the real per-item RNG path: max deviation from uniform **1.05%**, mean n **5.0051** (expected 5.0), support exactly 1–9, no n=0 |
| Density measured | parked path still builds and returns the old bimodal histogram; exp001's configs still default to it |
| GradScaler | `AMP: use_amp=True amp_dtype=bfloat16 grad_scaler_enabled=False` |
| STFT/iSTFT | fp32-pinned autocast blocks confirmed firing during the step |
| Fwd/bwd | 4 loader steps + 1 optimizer step: loss 0.0149→0.0135, grad norm 0.0118, **0 non-finite steps, 0 non-finite params** |
| Peak memory | 36.99 GiB torch-allocated of 96 GB |
| Warm start | 529/533 tensors transferred (41,975,216 / 41,996,006 params); 4 head tensors re-initialised; **reproducible under seed 42**, verified by building it twice |
| Dataset smoke | mixture ≡ Σ(targets) to 2.4e-07; LUFS median −19.0; peak ≤ 0.99; 47.6 items/s at 4 workers |
| Σstem val | 91 songs, reused unchanged — every excluded file is train-split |
| wandb | one run, name `exp002_260805_htdemucs_v2_uniform_n`, full config logged, git commit `0ea3ace…` captured |

## Seeds — all three pinned and recorded

| Source of randomness | Seed | Where |
|---|---|---|
| Output-head initialisation | 42 | `scripts/init_start_checkpoint.py --seed 42`, recorded in `checkpoint_init_log.txt` |
| Mix draw stream (density, class identity, excerpt, gain, swap) | 42 | `gugak_mix.seed` in the config |
| Trainer / torch global | 42 | `--seed 42` in the launch script |

The head-init seed is new. The script previously drew the re-initialised head layers from
PyTorch's global RNG with no seed at all, so two runs of the "same" experiment would start
from different output heads with nothing recording the difference. Since exp002 is the
anchor for a seed twin, that gap had to close.

## Findings worth carrying forward

**wandb project name.** `CLAUDE.md` states that MSST's hardcoded `project='msst'` is
overridden by `WANDB_PROJECT`. **It is not.** `utils/settings.py:591` passes
`project='msst'` as an explicit keyword to `wandb.init()`, and an explicit kwarg beats the
environment variable. `WANDB_ENTITY` *does* apply (the run is correctly scoped to
`maler-gye`) precisely because entity is not passed explicitly. The run is offline, so
nothing is misfiled in the cloud; the fix at upload time is
`wandb sync -p gugak_stem_separation wandb/offline-run-20260805_095738-476pciph`. The
CLAUDE.md line should be corrected.

**Launch scripts were not sourcing `.env`.** Neither exp001's nor exp001.2's launcher ran
`set -a; source .env; set +a`, contrary to the repo convention. exp002's does.

## Speed and memory

| Arm | Precision | Loader steps/s (median) |
|---|---|---|
| exp001 (fp32 arm, log includes earlier fp16 attempts) | fp32 | 1.56 |
| exp001.2 | bf16 | 2.20 |
| **exp002** | bf16 | **2.21** |

exp002 matches exp001.2's bf16 rate. At 2.21 it/s an epoch of 10,000 loader steps
(= 2,500 optimizer steps) takes ~75 min plus validation, so the 60-epoch ceiling is
roughly 3.5 days of wall clock.

## First eval cycles

*(filled in below as they land)*
