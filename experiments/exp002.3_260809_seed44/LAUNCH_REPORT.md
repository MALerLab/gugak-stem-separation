# exp002.3_260809_seed44 — launch report

**Launched** 2026-08-09T16:41:50+09:00 · **GPU 2** · tmux `exp002_3` · PID 92465
(uv wrapper 92461) · wandb `offline-run-20260809_164152-qdlrba2q`
· repo HEAD `43621e7`, MSST `8dbc33e`

## Why this run exists

exp002 (seed 42) and exp002.1 (seed 43) hit the **same** head collapse at epoch 12 — the
same three heads down (가야금 · 기타 · 양금), the same head up (대금) — and then diverged
completely. exp002 never recovered and was stopped; exp002.1 recovered within four evals
and has improved ever since, reaching +3.81 dB.

That makes the failure look **bimodal — recover or don't** — rather than smooth run-to-run
variance. With n=2 there is no way to say which outcome is the outlier. A third seed does
not estimate a recovery rate (n=3 is nowhere near enough for that); it breaks the tie.

⚠️ **The collapse is the measurement.** A collapse at ~ep12 is the datum, not a problem.
See the intervention policy below.

## The diff from exp002 — exactly two keys

Asserted mechanically against the config exp002 *actually logged* at launch
(`wandb/offline-run-20260805_095738-476pciph`), not against exp002's file on disk:

```
uv run python scripts/diff_run_config.py \
  --reference-run wandb/offline-run-20260805_095738-476pciph \
  --candidate configs/exp002.3_htdemucs_seed44.yaml \
  --expect gugak_mix.seed --expect training.run_name
```

**114 leaf fields compared · 2 differ · PASS (exit 0).**

| key | exp002 | exp002.3 |
|---|---|---|
| `gugak_mix.seed` | 42 | **44** |
| `training.run_name` | `exp002_260805_htdemucs_v2_uniform_n` | `exp002.3_260809_seed44` |

Everything else is exp002's, unchanged: manifest v2, `density_mode: uniform` over n ∈ 1..9,
lr 1e-4, bf16 from launch, 10 s segments, L1 with silent targets, Adam with clip 5.0,
effective batch 32, eval every 2,500 optimizer steps, Σstem-val selection on 91 songs, no
solo pool.

Seeds and their three surfaces → `SEEDS.txt`.

## Pre-launch checks

| # | check | result |
|---|---|---|
| 1 | resolved config differs from exp002 in exactly the 2 declared keys | ✅ 114 leaves, 2 differ |
| 2 | `training.instruments` == `gugak_mix.classes`, same order | ✅ both are the 9-class list, identical order |
| 3 | manifest v2 resolves; excluded `file_id`s absent from the draw pool | ✅ 16,615 → 16,599 rows, removals match the declared set exactly; survivors identical on all 41 columns; 0886 and 0905 intact; all 9 classes still drawable (양금 lowest at 72 train sources) |
| 4 | start checkpoint built at seed 44; counts match exp002's | ✅ 533 tensors, **529 transferred / 4 fresh**, 41,975,216 / 41,996,006 params (100.0%); init log reads `seed (head init): 44` |
| 4b | head init genuinely independent of seeds 42 and 43 | ✅ 529 trunk tensors bit-identical across all three seeds; **4/4 head tensors differ in all three pairwise comparisons** |
| 5 | `grad_scaler_enabled=False` under bf16 | ✅ smoke test and the trainer's own startup line both print it |
| 6 | STFT/iSTFT fp32 pinning firing | ✅ two `autocast(enabled=False)` regions entered during the forward |
| 7 | one forward + backward | ✅ **0 non-finite steps / 4**, **0 non-finite parameters**, grad_norm 0.1297 (pre-clip, clip 5.0), peak 36.99 GiB |
| 8 | GPU 2 idle before allocation; GPU 1 untouched | ✅ GPU 2 at 0 MiB / 0% immediately pre-launch; exp002.1 (PID 65008) alive and unperturbed throughout |
| 9 | HEAD commit + uncommitted files recorded | ✅ `git_commit.txt` |

**Waived deliberately:** the 200,000-draw density-sampler verification (unchanged from
exp002, verified there) and any tuning of any value.

## Launch mode — fresh, nothing inherited

Audited on the **running process's own command line**, not on the script:

```
--start_check_point .../start_checkpoint.ckpt --load_only_compatible_weights
```

No `--load_optimizer`, `--load_scheduler`, `--load_epoch`, `--load_best_metric`,
`--load_all_metrics`, `--load_all_losses`. This project has been bitten twice by
`--load_optimizer` silently restoring a halved learning rate, so the confirmation that
matters is the trainer's own first line: **`Train epoch: 0 Learning rate: 0.0001`** — the
declared lr is the lr that trains.

`.env` is sourced by the launcher (`set -a; source .env; set +a`) per repo convention.
Note: `WANDB_PROJECT` does **not** override MSST's hardcoded `project='msst'` — the
explicit `wandb.init()` kwarg beats the env var (`WANDB_ENTITY` does apply). Fix at
upload: `wandb sync -p gugak_stem_separation <run-dir>` (→ repo CLAUDE.md).

## Deviation from exp002's launcher — checkpoint location

`checkpoints/` is a **symlink to `~/storage/gugak-stemsep-experiments/exp002.3_260809_seed44/checkpoints`**
rather than a real directory on `userdata`. At launch `/home/jae.gye/userdata` was at
**97% (130 GB free)** while storage NVMe had 3.3 TB. MSST keeps every improving epoch at
~500 MB each — exp002.1 alone holds 19 GB across 41 checkpoints — so a local write would
have competed with the live exp002.1 for the last of the disk. Same pattern as the
2026-08-09 15:48 housekeeping that moved the finished runs. Affects storage only; no
training behaviour changes.

## Intervention policy — LET IT RIDE

- **Collapse around ep12: do NOT stop, do NOT resume from an earlier checkpoint, do NOT
  adjust the learning rate, do NOT touch the scheduler.** exp002.1 recovered unaided four
  evals later, and whether this one does too is the entire question.
- Do not intervene on a plateau either — ReduceLROnPlateau behaves as configured.
- The **only** reasons to stop early are genuine faults: non-finite steps, a crash, a disk
  or CUDA error. **Quality going down is not a fault.**
- Stop, if ever, with SIGINT (`kill -INT 92465` or Ctrl-C in the pane), never `kill -9`.

## Pace and the epoch-12 decision point

Observed at launch: **2.09 it/s** over the 10,000-loader-step epoch → ~80 min train +
~9 min validation ≈ **89 min/epoch**. exp002.1's checkpoint timestamps corroborate:
87–88 min/epoch early, drifting to ~94 as a second run joined the box.

exp002.1 reached its ep-12 eval **20 h 03 m** after launch. Mapping that onto this launch:

> **epoch 12 lands Monday 2026-08-10, roughly 12:00–13:00 (centre ~12:45).**

⚠️ Estimate caveat: exp003.0 launching on GPU 0 adds 8 more dataloader workers to a
24-core box (load average was already 18.7 with two runs). A third concurrent run could
push epoch 12 an hour or more later. Read the projection as a window, not a clock.

## Context — exp002.2 was already stopped

GPU 2 was freed before this session began. `exp002.2_260806_coherent` was stopped by
SIGINT at **2026-08-09T15:26:06+09:00**; its launcher recognised exit 130 and correctly
declined to auto-resume. Its best checkpoint `model_htdemucs_ep_20_si_sdr_-9.1103.ckpt` is
intact on storage, nothing was pruned, and GPU 2 read 0 MiB at **0%** utilisation — the
0% being the evidence of no leaked CUDA context (a leak shows 100% util at 0 MiB).
