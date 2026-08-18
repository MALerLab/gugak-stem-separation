# exp004.1_260817_incoherent_n2 — launch report

**Launched** 2026-08-17T10:13:16+09:00 · **GPU 2** · tmux `exp004_1` · python PID 156730
(uv wrapper 156726) · wandb `offline-run-20260817_101319-1twf15wx`
· repo HEAD `1b27d13` (dirty, see below), MSST `dd1dae4`

## What this run is

The **incoherent n ≥ 2 control for exp004** — the missing corner of a 2×2 in
(coherence × density floor), all three arms at seed 45 with the same output-head init:

|  | incoherent (p = 0) | coherent (p = 1.0 uniform) |
|---|---|---|
| n ~ U{1..9} | exp002.4 (finished, +4.80 @ ep59, never broke) | — |
| n ~ U{2..9} | **exp004.1 ← this run** | exp004 (running, GPU 1, +5.85 @ ep23) |

exp004 moved two variables against exp002.4 — `coherent_mix_prob` 0.0 → 1.0 **and**
`density_uniform_min` 1 → 2 — so its lead is unattributable. With exp004.1:

- **exp004 vs exp004.1 isolates coherence** (both n ≥ 2, both seed 45, same head init)
- **exp004.1 vs exp002.4 isolates the density floor** (both incoherent, both seed 45,
  same head init)

At p = 0 the anchor-cluster path never fires (`_draw_coherent_count` returns 0,
`_fill_clusters` is never entered — verified below: 0 clusters in 200,000 draws on this
exact config), so `anchor_selection` / `cluster_min_melodic` / `cluster_shared_*` ride
along **inert**. That is why the config is derived from exp004's, not exp002.4's: the
diff against the arm we are controlling for is exactly two keys.

Caveat carried over from Notion's exp004 entry: seed 45 is the one seed of four that
never collapsed under the incoherent recipe, so this control cleans the **score**
comparison, not the "does coherence prevent the collapse" question.

## Config diff 1 — vs exp004 (the gate): exactly two leaves

Asserted against the config exp004 *actually logged* at launch
(`wandb/offline-run-20260815_201100-4932rh1e`), not its file on disk
(`config_diff_vs_exp004.txt`):

```
uv run python scripts/diff_run_config.py \
  --reference-run wandb/offline-run-20260815_201100-4932rh1e \
  --candidate configs/exp004.1_htdemucs_incoherent_n2.yaml \
  --expect gugak_mix.coherent_mix_prob --expect training.run_name
```

**119 leaf fields compared · 2 differ · PASS (exit 0).**

| key | exp004 (logged) | exp004.1 |
|---|---|---|
| `gugak_mix.coherent_mix_prob` | 1.0 | **0.0** |
| `training.run_name` | `exp004_260815_coherent_p1_uniform` | `exp004.1_260817_incoherent_n2` |

## Config diff 2 — vs exp002.4 (reported, not gated): six leaves

Against exp002.4's logged config (`wandb/offline-run-20260813_103716-21200npo`,
`config_diff_vs_exp002.4.txt`): **119 leaf fields compared · 6 differ.**

| key | exp002.4 (logged) | exp004.1 | role |
|---|---|---|---|
| `gugak_mix.density_uniform_min` | absent (= 1) | **2** | **the variable under test** in this pair |
| `gugak_mix.anchor_selection` | absent | `uniform` | inert at p = 0 |
| `gugak_mix.cluster_min_melodic` | absent | 2 | inert at p = 0 |
| `gugak_mix.cluster_shared_channel_swap` | absent | true | inert at p = 0 |
| `gugak_mix.cluster_shared_gain` | absent | false | inert at p = 0 |
| `training.run_name` | `exp002.4_260813_seed45` | `exp004.1_260817_incoherent_n2` | run-scoped |

`coherent_mix_prob` (0.0) and `gugak_mix.seed` (45) are **identical** on both sides — as
intended: exp004.1 is exp002.4 with the density floor raised (plus four keys the p = 0
path never reads).

## Seeds — three surfaces, all 45 (`SEEDS.txt`)

| surface | value | where |
|---|---|---|
| mix draw stream (n, S, per-stem file/offset/gain/swap) | 45 | `gugak_mix.seed` |
| trainer global (torch/np, data order) | 45 | `launch.sh --seed 45` |
| output-head init | 45 | **reused** exp002.4's `start_checkpoint.ckpt` (the same file exp004 reused) — `cp -p` into `checkpoints/`, sha256 `d2f4f940…8d5a65` identical on exp002.4's, exp004's and this copy (`start_checkpoint_sha256.txt`) |

Head-init verification (`verify_start_checkpoint.py` → `start_checkpoint_check.txt`):
**533 tensors; 529 trunk bit-identical to the official pretrained htdemucs; 4 heads
(9-class shapes) bit-identical to a `torch.manual_seed(45)` fresh init through MSST's
model builder; and — the literal assertion asked for — 533/533 tensors bit-identical to a
FRESH seed-45 REBUILD** produced by `scripts/init_start_checkpoint.py --seed 45` on this
config into the scratchpad (rebuild deleted afterwards; the file-level sha differs only
because `torch.save` serialisation is not byte-stable — the tensors are). Also 533/533
tensor-identical to exp002.4's and exp004's on-disk copies. No new checkpoint was built
for the run; all three 2×2 arms train from the same head init.

⚠️ Same seed ≠ same data. exp004.1 and exp004 draw the **same n and class set S per item
index** (n and S are drawn before p is consulted; the launch gates below reproduce
exp004's n-histogram and per-class exposure to four digits) — from there exp004 spends
RNG on anchors/clusters and exp004.1 does not. Holding the seed removes seed as an
explanatory variable; it does not make streams comparable item-for-item.

## Pre-launch gates — ENFORCED, none waived, all pass

| # | check | result |
|---|---|---|
| 1 | config-diff gate vs exp004's logged config | ✅ 119 leaves, **exactly 2 differ** (`coherent_mix_prob`, `run_name`) — `config_diff_vs_exp004.txt` |
| 2 | config-diff vs exp002.4's logged config (report only) | ✅ 119 leaves, 6 differ (`density_uniform_min` + 4 inert cluster keys + `run_name`) — `config_diff_vs_exp002.4.txt` |
| 3 | start checkpoint = seed-45 head init, bit-identical to a fresh rebuild | ✅ 529 trunk == pretrained · 4/4 heads == `manual_seed(45)` fresh init · **533/533 tensors == fresh `init_start_checkpoint.py --seed 45` rebuild** · 533/533 == exp002.4's copy · 533/533 == exp004's copy · sha256 identical three-way (`start_checkpoint_check.txt`, `start_checkpoint_sha256.txt`) |
| 4 | **200,000-draw density check**, real per-item RNG path (`plan_item` = `default_rng([seed, index])` → `_plan_item`, the sequence `__getitem__` consumes) | ✅ support exactly **[2..9]** · **draws at n = 1: 0** · below floor: 0 · per-size share 0.1235–0.1261 vs 0.1250 · **max relative deviation from uniform 1.24 %** (gate < 2 %) · mean n 5.5027 (expected 5.5) |
| 5 | coherent path never entered | ✅ over the same 200,000 draws: draws with ≥ 1 cluster **0** · clusters total **0** · k_declared max 0 · k_realised max 0 · k_target max 0.0 · shortfall reasons {} · cluster records 0 |
| 6 | **G1** exposure flat — on the EXACT launch config | ✅ mean rate 0.6114 vs expected 5.5/9 = 0.6111 · **max deviation 0.28 %** (gate < 2 %) · per class 0.6099–0.6131 (identical to exp004's launch gate, as it must be — same seed, n and S drawn before p) |
| 7 | manifest v2 resolves; the 16 excluded file_ids never appear in the draw pool | ✅ `source_manifest_v2` 16,599 rows; excluded set = v1 − v2 = **16 file_ids** (0714's 대금/아쟁/피리 + all 13 of 0885) · **0 of them in the dataset's actual pool** (pool-level join on `out_path` → `file_id`) · pool 676 songs / 4,250 files (= v2's train pool) · pick-level `unknown_path` 0 · `class_mismatch` 0 · `quarantined` 0 · `wrong_split` 0 · `wrong_dataset` 0 across 200,000 draws |
| 8 | **G4** mixture ≡ Σ(targets) | ✅ 300 items through a real DataLoader (10 workers), max abs error **2.38e-07** (gate ≤ 1e-6, "~1e-7" as expected); silence by tolerance (1e-6) never equality |
| 9 | bf16 with GradScaler disabled | ✅ smoke: `use_amp=True amp_dtype=bfloat16 grad_scaler_enabled=False`; trainer startup prints the same line (`AMP: … grad_scaler_enabled=False`) |
| 10 | STFT/iSTFT pinned fp32; `inference_amp_dtype` float32 | ✅ 8 stft/istft calls recorded, dtypes {`stft:float32`, `istft:complex64`} only; `[amp] inference_amp_dtype=float32` |
| 11 | one clean forward/backward, 0 non-finite | ✅ 4 loader steps = 1 accumulation cycle on GPU 2: losses 0.0148 / 0.0144 / 0.0156 / 0.0152, grad-norm 0.0137 pre-clip, **0 non-finite steps, 0 non-finite params**, peak 36.99 GiB (`smoke_train_step.txt`) |
| 12 | trainer prints lr 1e-4 at ep0, fresh launch, no `--load_*` | ✅ `Train epoch: 0 Learning rate: 0.0001`; running process command line = `--start_check_point …/exp004.1…/start_checkpoint.ckpt --load_only_compatible_weights`, no `--load_optimizer/--load_scheduler/--load_epoch/--load_best_metric/--load_all_*` |
| 13 | GPU 2 idle before launch; GPUs 0 and 1 untouched after | ✅ pre-launch GPU 2 at 0 MiB / 0 % (`gpu_prelaunch.txt`); post-launch GPU 2 at 40.7 GiB / 92–99 % under PID 156730 (`CUDA_VISIBLE_DEVICES=2` in its environ); GPU 0 PID 95210 (exp003.0) and GPU 1 PID 144922 (exp004) still at 48.1 / 40.7 GiB, same PIDs, no restart (`gpu_postlaunch.txt`) |
| 14 | `.env` sourced by the launcher | ✅ `WANDB_ENTITY=maler-gye` present in the training process's environ |
| 15 | HEAD + uncommitted files + MSST pin recorded (tree dirty → recorded, not committed) | ✅ `git_commit.txt` (below) |
| 16 | disk headroom | ✅ storage NVMe 3.2 TB free; `checkpoints/` symlinks there (userdata at 95 %) |

Nothing waived. (exp004 had waived the 200k density check as "verified in the build";
here the density floor is the variable under test, so #4 and #5 ran on this config.)

### Sampler characterisation on the exact config (`launch_gates.txt`, `metrics/launch_gate_summary.json`)

- n ~ U{2..9}: shares 2: 0.1261 · 3: 0.1248 · 4: 0.1235 · 5: 0.1242 · 6: 0.1253 ·
  7: 0.1249 · 8: 0.1259 · 9: 0.1253 (200,000 draws; the same n stream exp004 saw)
- no clusters, no H — every stem is an independent (song, offset, gain, swap) draw
  exactly as in exp002/exp002.4 (the sampler's G2 gate: p = 0 is bit-identical to the
  pre-cluster incoherent path)
- plan cost 100 µs/mix median; audio path 1.90 items/s on 10 workers with three trainers
  on the box

## `git_commit.txt` — launched from a dirty tree (deliberate, recorded)

```
1b27d13544fa300150d826013de977c393aa33d2
 M scripts/verify_coherent_sampler.py
 M src/data/mix_dataset.py
?? configs/exp004.1_htdemucs_incoherent_n2.yaml
?? configs/exp004_htdemucs_coherent_p1_uniform.yaml
?? experiments/260815_anchor_cluster_mixer/
?? experiments/exp004.1_260817_incoherent_n2/
?? experiments/exp004_260815_coherent_p1_uniform/
dd1dae41b755f74f081ada1d24d5143675d83183
```

Line 1 = repo HEAD; middle = uncommitted files at launch (the 08-15 sampler sessions'
`mix_dataset.py` / verifier edits, the build folder, exp004's config + folder, and this
run's config + folder); last line = the MSST submodule pin. ⚠️ Same warning as exp004:
the sampler that trains this run is the **uncommitted** `src/data/mix_dataset.py`. Two
running experiments now depend on it — commit before the tree drifts.

## Ops

- ceiling **150k optimizer steps / 60 epochs** (`num_steps` 10,000 loader iters × batch
  8 / accum 4 = 2,500 optimizer steps per epoch = one eval), effective batch 32
- expected pace **~92 min/epoch** with three trainers on the box (exp004's measured
  pace) → 60 epochs ≈ 3.8 days from 10:13 on 08-17 → ceiling around **2026-08-21
  ~05:00–06:00 KST**. Runs unattended.
- exp002-family ep11–12 collapse window: ~08-18 03:00–06:00 KST
- wandb offline (`--wandb_offline`); ⚠️ project-name trap: MSST hardcodes
  `project='msst'`, sync later with `wandb sync -p gugak_stem_separation
  wandb/offline-run-20260817_101319-1twf15wx`
- checkpoints: `experiments/exp004.1_260817_incoherent_n2/checkpoints` →
  `~/storage/gugak-stemsep-experiments/exp004.1_260817_incoherent_n2/checkpoints`
- `.env` sourced by the launcher (`set -a; source .env; set +a`)

## Intervention rule — LET IT RIDE (unchanged from EXP002.3)

⚠️ **The collapse is the measurement.** If quality drops at any epoch: no stop, no resume
from an earlier checkpoint, no hand-set lr, no scheduler edits. Only a genuine fault —
non-finite steps, crash, disk or CUDA error — justifies stopping. Stop, if ever, with
SIGINT (`kill -INT 156730` or Ctrl-C in the `exp004_1` pane), never `kill -9`.

## First eval — ep0

Landed 11:46 KST (93 min after launch: 10,000 loader steps at ~2.03 it/s + 589 s eval);
**0 non-finite steps**; training loss 0.00994 over the epoch; ep1 started at lr 1e-4.

| ep | 가야금 | 거문고 | 기타 | 대금 | 아쟁 | 양금 | 타악기 | 피리 | 해금 | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | −15.38 | −17.54 | −15.32 | −14.22 | −16.26 | −24.00 | −0.14 | −7.59 | −9.22 | **−13.30** |

Against the seed-45 siblings' ep0 (same head init): **exp004 −13.47 · exp002.4 −14.37**;
wider family: exp002 −19.34 · exp002.1 −14.11 · exp002.3 −14.51 · exp002.2 −18.57.
exp004.1's ep0 sits inside the family's warm-up band, a hair above both seed-45 arms —
sanity confirmed, nothing more claimed at n=1 evals. Trajectory → Notion after the run.

## Context at launch

exp003.0 (BS-RoFormer pilot) live on GPU 0 since 08-09 (ep26 in progress); exp004
(coherent p=1.0 uniform, seed 45) live on GPU 1 since 08-15, at ep24, best +5.85 @ ep23,
no collapse. exp002.4 finished on GPU 2 at 03:39 today (+4.8039 @ ep59, never broke),
freeing the card. Both live runs untouched by this launch.

## Artifacts in this folder

`launch.sh` · `SEEDS.txt` · `git_commit.txt` · `config_diff_vs_exp004.txt` ·
`config_diff_vs_exp002.4.txt` · `verify_start_checkpoint.py` →
`start_checkpoint_check.txt` + `start_checkpoint_sha256.txt` ·
`checkpoint_init_log_from_exp002.4.txt` · `verify_launch_gates.py` → `launch_gates.txt`
+ `metrics/launch_gate_summary.json` + `metrics/launch_gate_exposure.parquet` ·
`smoke_train_step.txt` · `gpu_prelaunch.txt` · `gpu_postlaunch.txt` · `train.log` ·
`checkpoints/` (symlink) · config `configs/exp004.1_htdemucs_incoherent_n2.yaml`
