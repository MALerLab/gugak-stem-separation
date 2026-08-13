# exp002.4_260813_seed45 — launch report

**Launched** 2026-08-13T10:37:13+09:00 · **GPU 2** · tmux `exp002_4` · PID 122869
(uv wrapper 122865) · wandb `offline-run-20260813_103716-21200npo`
· repo HEAD `43621e7`, MSST `8dbc33e`

## Why this run exists

All three prior seeds hit the same head collapse at ep11–12 (가야금 · 기타 · 양금 down,
대금 up; seed 44 additionally dropped 아쟁). Outcomes were **bimodal**: exp002 (seed 42)
never recovered and was stopped at ep32; exp002.1 (seed 43) recovered within four evals
and finished at **+3.99** (ep59); exp002.3 (seed 44) recovered and finished at **+4.66**
(ep58, its best — it was still climbing when the 60-epoch ceiling hit on 2026-08-13).

A fourth seed grows both of the numbers that matter: the recovery tally (currently
2 recover : 1 terminal → becomes 3:1 or 2:2, informative either way) and the
completed-run final-score spread (currently 0.67 dB on n=2). Launched while GPUs sat
idle, per the standing many-seeds-while-free rationale (prof-endorsed 2026-08-11).

## The diff from exp002 — exactly two keys

Asserted mechanically against the config exp002 *actually logged* at launch
(`wandb/offline-run-20260805_095738-476pciph`), not against exp002's file on disk:

```
uv run python scripts/diff_run_config.py \
  --reference-run wandb/offline-run-20260805_095738-476pciph \
  --candidate configs/exp002.4_htdemucs_seed45.yaml \
  --expect gugak_mix.seed --expect training.run_name
```

**114 leaf fields compared · 2 differ · PASS (exit 0).**

| key | exp002 | exp002.4 |
|---|---|---|
| `gugak_mix.seed` | 42 | **45** |
| `training.run_name` | `exp002_260805_htdemucs_v2_uniform_n` | `exp002.4_260813_seed45` |

Everything else is exp002's, unchanged: manifest v2, `density_mode: uniform` over n ∈ 1..9,
lr 1e-4, bf16 from launch, 10 s segments, L1 with silent targets, Adam with clip 5.0,
effective batch 32, eval every 2,500 optimizer steps, Σstem-val selection on 91 songs, no
solo pool. Seeds and their three surfaces → `SEEDS.txt`.

## Pre-launch checks

| # | check | result |
|---|---|---|
| 1 | resolved config differs from exp002 in exactly the 2 declared keys | ✅ 114 leaves, 2 differ |
| 2 | `training.instruments` == `gugak_mix.classes`, same order | ✅ trainer startup prints the 9-class list in canonical order |
| 3 | start checkpoint built at seed 45; counts match siblings | ✅ 533 tensors, **529 transferred / 4 fresh**, 41,975,216 / 41,996,006 params (100.0%); init log reads `seed (head init): 45` |
| 3b | head init genuinely independent of seed 44 | ✅ 529 trunk tensors bit-identical to seed 44's; **4/4 head tensors differ**. (Seeds 42/43 start checkpoints were pruned 2026-08-11, so this is a two-way check, not exp002.3's three-way; the builder is seeded, so they are rebuildable if ever needed.) |
| 4 | `grad_scaler_enabled=False` under bf16 | ✅ trainer's own startup line prints it |
| 5 | fresh launch mode on the *running* process's command line | ✅ `--start_check_point .../start_checkpoint.ckpt --load_only_compatible_weights`; no `--load_optimizer/--load_scheduler/--load_epoch/--load_best_metric/--load_all_*` |
| 6 | declared lr is the lr that trains | ✅ first trainer line: `Train epoch: 0 Learning rate: 0.0001` |
| 7 | GPU 2 idle immediately pre-launch; GPUs 0 untouched after | ✅ GPU 2 at 0 MiB / 0% pre-launch; post-launch GPU 2 at 40.7 GiB / 67% under this run, exp003.0 (GPU 0, PID 95210) unperturbed |
| 8 | HEAD commit + uncommitted files + MSST pin recorded | ✅ `git_commit.txt` |
| 9 | disk headroom for ~20 GB of checkpoints | ✅ storage NVMe 3.3 TB free; `checkpoints/` symlinks there (userdata at 95%) |

**Waived deliberately** (unchanged since exp002.3 verified them 4 days ago, zero
intervening manifest/data/code changes): the manifest-v2 resolution check (16,599-row
draw pool, exclusions) · the 200,000-draw density-sampler verification (verified at
exp002) · the standalone one-step smoke test (superseded by the live process passing
gates 4–6 on its own startup) · any tuning of any value.

## Launch mode — fresh, nothing inherited

Same as every sibling: `--load_only_compatible_weights` from this run's own seed-45
`start_checkpoint.ckpt`, no optimizer/scheduler/epoch state loaded. The project has been
bitten twice by `--load_optimizer` silently restoring a halved learning rate; gate 6
above is the confirmation that matters.

`.env` sourced by the launcher (`set -a; source .env; set +a`), so `WANDB_PROJECT`
overrides MSST's hardcoded `project='msst'` and `WANDB_IGNORE_GLOBS` backstops artifacts.

## Intervention policy — LET IT RIDE

- **Collapse around ep11–12: do NOT stop, do NOT resume from an earlier checkpoint, do
  NOT adjust the learning rate, do NOT touch the scheduler.** Seeds 43 and 44 recovered
  unaided; whether this one does too is the question. The collapse is the measurement.
- No intervention on plateaus either — ReduceLROnPlateau behaves as configured.
- The **only** reasons to stop early are genuine faults: non-finite steps, a crash, a
  disk or CUDA error. **Quality going down is not a fault.**
- Stop, if ever, with SIGINT (`kill -INT 122869` or Ctrl-C in the pane), never `kill -9`.

## Pace expectations

Siblings ran ~89 min/epoch with 2–3 concurrent runs on the box; the box now carries two
runs (this + exp003.0) at load ~3.3/24 cores pre-launch, so expect the same or slightly
better. Mapping exp002.3's timings onto this launch:

> **the ep11–12 collapse window lands overnight tonight, roughly 04:00–07:00 on
> 2026-08-14; the 60-epoch ceiling lands ~2026-08-17 (Monday).**

## Context at launch

exp002.3 (GPU 2's previous tenant) finished clean at its 60-epoch ceiling
2026-08-13T07:38, tmux session gone, GPU 2 read 0 MiB / 0% pre-launch. exp003.0
(BS-RoFormer pilot) live on GPU 0, mid-epoch-12, best +8.93 @ ep11.
