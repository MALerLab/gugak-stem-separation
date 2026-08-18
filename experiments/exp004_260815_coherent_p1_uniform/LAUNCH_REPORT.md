# exp004_260815_coherent_p1_uniform — launch report

**Launched** 2026-08-15T20:10:57+09:00 · **GPU 1** · tmux `exp004` · python PID 144922
(uv wrapper 144918) · wandb `offline-run-20260815_201100-4932rh1e`
· repo HEAD `1b27d13` (dirty, see below), MSST `dd1dae4`

## What this run is

The **first training use of the anchor-cluster coherent sampler** built earlier today
(`experiments/260815_anchor_cluster_mixer/BUILD_REPORT.md`), at maximum dose:
static `coherent_mix_prob: 1.0`, `anchor_selection: uniform`.

exp002.2 asked "does coherent training help" and could not answer, because drawing the
song first collapsed density (5.00 → 3.24 stems/mix) and per-class exposure (양금 0.10×)
alongside coherence. Today's sampler unlinks them: **density n and the class set S are
drawn uniformly first; only then does p decide which members of S are mutually
coherent** (a cluster = one song, one shared offset). Exposure is flat by construction
at every p (gate G1). exp004 is the first arm where a coherence result is attributable
to coherence.

**p = 1.0 under `uniform` is not exp002.2's one-song-per-mix.** The fill loop uses as
many anchor songs as it needs, so a mix lands as 2–3 real unison clusters mixed
incoherently against each other. On this exact config (200,000 plan draws, below):
H_melodic 0.634, 29.7% of mixes fully coherent, 1.38 clusters/mix, realised
E[k]/declared 0.919. Max dose, fragmented by design — running max first bounds the
whole design space in one run.

⚠️ **Naming.** `exp004` was previously reserved in the Notion legal pad for per-source
loss normalisation; that reservation moves to `exp005` (unbuilt, unspecced). Notion not
edited by this launch — the user's edit.

## The diff from exp002 — exactly eight leaves

Asserted mechanically against the config exp002 *actually logged* at launch
(`wandb/offline-run-20260805_095738-476pciph`), not against exp002's file on disk
(`config_diff.txt`):

```
uv run python scripts/diff_run_config.py \
  --reference-run wandb/offline-run-20260805_095738-476pciph \
  --candidate configs/exp004_htdemucs_coherent_p1_uniform.yaml \
  --expect gugak_mix.coherent_mix_prob --expect gugak_mix.anchor_selection \
  --expect gugak_mix.cluster_min_melodic --expect gugak_mix.density_uniform_min \
  --expect gugak_mix.seed --expect gugak_mix.cluster_shared_channel_swap \
  --expect gugak_mix.cluster_shared_gain --expect training.run_name
```

**119 leaf fields compared · 8 differ · PASS (exit 0).** Every differing leaf is in the
spec table; nothing else differs.

| key | exp002 (logged) | exp004 |
|---|---|---|
| `gugak_mix.coherent_mix_prob` | 0.0 | **1.0** |
| `gugak_mix.anchor_selection` | absent | **uniform** |
| `gugak_mix.cluster_min_melodic` | absent | **2** |
| `gugak_mix.density_uniform_min` | absent (=1) | **2** |
| `gugak_mix.cluster_shared_channel_swap` | absent | true (sampler default, written explicitly) |
| `gugak_mix.cluster_shared_gain` | absent | false (sampler default, deferred decision, written explicitly) |
| `gugak_mix.seed` | 42 | **45** |
| `training.run_name` | `exp002_260805_htdemucs_v2_uniform_n` | `exp004_260815_coherent_p1_uniform` |

Everything else is exp002's, unchanged: manifest v2, `density_mode: uniform`, lr 1e-4,
bf16 from launch, 10 s segments, L1 with silent targets, Adam with clip 5.0, effective
batch 32, eval every 2,500 optimizer steps, Σstem-val selection on 91 songs, EQ off, no
solo pool. The trainer's own startup lines confirm effective batch 32 / 60 epochs /
lr 1e-4.

## Seeds — three surfaces, all 45 (`SEEDS.txt`)

| surface | value | where |
|---|---|---|
| mix draw stream (now also n, S, k, anchors, offsets) | 45 | `gugak_mix.seed` |
| trainer global (torch/np, data order) | 45 | `launch.sh --seed 45` |
| output-head init | 45 | **reused** exp002.4's `start_checkpoint.ckpt` — byte-copied into `checkpoints/`, sha256 `d2f4f940…8d5a65` identical on both sides |

Head-init verification (`verify_start_checkpoint.py` → `start_checkpoint_check.txt`),
in memory, no rebuild: **533 tensors; 529 trunk tensors bit-identical to the official
pretrained htdemucs; 4 head tensors** (`decoder.3.conv_tr.{weight,bias}`,
`tdecoder.3.conv_tr.{weight,bias}`, 9-class shapes) **bit-identical to a
`torch.manual_seed(45)` fresh init through MSST's own model builder** — the exact
computation `init_start_checkpoint.py` performs, so the file *is* seed 45, not merely
labelled so. Provenance chain: exp002.4's live process command line points at that
same file; its `checkpoint_init_log.txt` (copied here as
`checkpoint_init_log_from_exp002.4.txt`) reads `seed (head init): 45`, 529 transferred /
4 fresh, 41,975,216 / 41,996,006 params. The seed-44 cross-check exp002.4 ran is
skipped — that checkpoint has since been pruned; check 3 above is the load-bearing one.

⚠️ Seed 45 is shared with exp002.4 (running on GPU 2). Same head init and trainer seed,
but the anchor-cluster sampler consumes the draw stream differently, so the two runs
do **not** see the same data — holding the seed removes it as an explanatory variable,
nothing more.

## Pre-launch checks — ENFORCED, all pass

| # | check | result |
|---|---|---|
| 1 | config-diff gate vs exp002's logged config | ✅ 119 leaves, 8 differ, all declared (`config_diff.txt`) |
| 2 | start checkpoint: 529 trunk transferred, 4 heads fresh at seed 45 | ✅ 529 bit-identical to pretrained trunk; 4/4 heads == `manual_seed(45)` fresh init; sha256 identical to exp002.4's file (`start_checkpoint_check.txt`) |
| 3 | **G1** exposure flat — on the EXACT launch config, 200,000 plan draws | ✅ mean rate 0.6114 vs expected 5.5/9 = 0.6111 · **max deviation 0.28 %** (gate < 2 %) · per class 0.6099–0.6131 |
| 4 | **G5** zero clusters below `cluster_min_melodic` — exact config | ✅ **0** in 200,000 draws (direct per-cluster assertion) |
| 5 | manifest v2 resolves; excluded files absent from the draw pool | ✅ pool 676 songs / 4,250 files (= v2's train pool); `excluded_file` 0 · `quarantined` 0 · `wrong_split` 0 · `wrong_dataset` 0 · `class_mismatch` 0 across 200,000 draws, joined to `source_manifest_v2` |
| 6 | cluster integrity vs manifest | ✅ `cluster_multi_song` 0 · `cluster_multi_offset` 0 · `cluster_song_vs_plan` 0 · `structure` 0 |
| 7 | **G4** mixture ≡ Σ(targets) on decoded batches | ✅ 300 items through a real DataLoader, max abs error **2.38e-07** (gate ≤ 1e-6); silence by tolerance (1e-6) never equality |
| 8 | bf16 with GradScaler disabled | ✅ smoke: `use_amp=True amp_dtype=bfloat16 grad_scaler_enabled=False`; trainer startup prints the same line |
| 9 | STFT/iSTFT pinned fp32; eval fp32 | ✅ 8 stft/istft calls recorded, dtypes {`stft:float32`, `istft:complex64`} only; `inference_amp_dtype=float32` |
| 10 | one clean forward/backward, 0 non-finite | ✅ 4 loader steps = 1 accumulation cycle on GPU 1: losses 0.0157 / 0.0155 / 0.0162 / 0.0151, grad-norm 0.0148 pre-clip, **0 non-finite steps, 0 non-finite params**, peak 36.99 GiB (`smoke_train_step.txt`) |
| 11 | trainer prints lr 1e-4 at ep0 | ✅ `Train epoch: 0 Learning rate: 0.0001` |
| 12 | fresh launch on the *running* process's command line | ✅ `--start_check_point …/exp004…/start_checkpoint.ckpt --load_only_compatible_weights`; no `--load_optimizer/--load_scheduler/--load_epoch/--load_best_metric/--load_all_*` |
| 13 | GPU 1 idle before launch; GPUs 0 and 2 untouched after | ✅ pre-launch GPU 1 at 0 MiB / 0 % (`gpu_prelaunch.txt`); post-launch GPU 1 at 40.7 GiB / 94 % under PID 144922 (`CUDA_VISIBLE_DEVICES=1` in its environ); GPU 0 PID 95210 (exp003.0) and GPU 2 PID 122869 (exp002.4) still at 48.1 / 40.7 GiB, same PIDs, no restart |
| 14 | HEAD + uncommitted files + MSST pin recorded | ✅ `git_commit.txt` (below) |
| 15 | disk headroom | ✅ storage NVMe 3.2 TB free; `checkpoints/` symlinks there (userdata at 95 %) |

**Waived deliberately** (verified in today's build, unchanged since): the 200,000-draw
density-sampler verification (n support 2–9 and mean 5.5027 were nonetheless re-read off
the G1 run above) · git working-tree cleanliness (recorded instead) · G2 p=0
bit-identity and the 15-cell grid (not this config's business).

### Sampler characterisation on the exact config (`launch_gates.txt`, `metrics/launch_gate_summary.json`)

The numbers below are the *treatment* the model actually receives — the build report's
grid numbers, re-measured on the config that runs:

- n ~ U{2..9}: support exactly 2–9, mean 5.5027 · E[k] declared 5.503, realised
  **5.059 → 0.919** of declared (build report at n_min=1: 0.899)
- shortfall: 40.4 % of draws end short of declared k (r=1 drop 72,770 · no eligible
  anchor 8,013 of 200,000) — higher *rate* than the n_min=1 grid's 36 % because n=1
  draws (which could never fall short) are gone, but a smaller *stem* gap (0.919 vs
  0.899). Every short draw is reason-attributed; 2.75 % of mixes end with 0 clusters
- **H_melodic mean 0.634 · P(H_melodic = 1) 0.297** — matches the build report's
  uniform p=1.0 cell to three digits · raw H 0.634
- clusters/mix 1.38 · size share 2: 38.8 % · 3: 17.8 % · 4–7: 41.4 % · 8–9: 2.0 %
- cluster genre share: 창작국악 33.8 % · 판소리 30.0 % · 풍류음악 19.9 % · 민요 8.2 % ·
  궁중음악 6.3 % · 대풍류 1.8 % · **산조 0** (structural — every 산조 train song holds one
  melodic class, so no 산조 song can seed a melodic pair; 산조 stems reach the model
  only through incoherent draws — build report D4)
- plan cost 107 µs/mix median; audio path 2.59 items/s on 10 workers

## `git_commit.txt` — launched from a dirty tree (deliberate, recorded)

```
1b27d13544fa300150d826013de977c393aa33d2
 M scripts/verify_coherent_sampler.py
 M src/data/mix_dataset.py
?? configs/exp004_htdemucs_coherent_p1_uniform.yaml
?? experiments/260815_anchor_cluster_mixer/
?? experiments/exp004_260815_coherent_p1_uniform/
dd1dae41b755f74f081ada1d24d5143675d83183
```

Line 1 = repo HEAD; middle = uncommitted files at launch (today's two sampler sessions:
the anchor-cluster mixer + melodic-pair rule in `mix_dataset.py` / the verifier, plus the
build folder and this run's config/folder); last line = the MSST submodule pin. ⚠️ The
sampler that trains this run is the **uncommitted** `src/data/mix_dataset.py`; the
committed HEAD does not contain it. Commit before the tree drifts.

## Ops

- ceiling **150k optimizer steps / 60 epochs** (`num_steps` 10,000 loader iters × batch
  8 / accum 4 = 2,500 optimizer steps per epoch = one eval), effective batch 32
- pace at launch: **2.10 it/s → ~79 min/epoch** by tqdm's estimate with three runs on
  the box (load ~13/24 cores); siblings ran ~88 min. Expect the ceiling around
  **2026-08-19** (60 × 79–88 min ≈ 3.3–3.7 days from 20:11 on 08-15 → 08-19 03:00–15:00).
  ep11–12 collapse window: **overnight 08-16, roughly 11:00–14:00 KST at 79 min/epoch,
  ~12:30–15:30 at 88** — i.e. mid-day tomorrow rather than literally overnight.
- wandb offline (`--wandb_offline`); ⚠️ project-name trap: MSST hardcodes
  `project='msst'`, sync later with `wandb sync -p gugak_stem_separation
  wandb/offline-run-20260815_201100-4932rh1e`
- checkpoints: `experiments/exp004_260815_coherent_p1_uniform/checkpoints` →
  `~/storage/gugak-stemsep-experiments/exp004_260815_coherent_p1_uniform/checkpoints`
  (exp002.3-introduced symlink layout)
- `.env` sourced by the launcher (`set -a; source .env; set +a`)

## Intervention rule — LET IT RIDE

⚠️ **The collapse is the measurement.** If heads collapse in the ep11–12 window: no
stop, no resume from an earlier checkpoint, no hand-set lr, no scheduler edits. Only a
genuine fault — non-finite steps, crash, disk or CUDA error — justifies stopping. Quality
going down is not a fault. Stop, if ever, with SIGINT (`kill -INT 144922` or Ctrl-C in
the pane), never `kill -9`.

## First evals — ep0 / ep1 sanity numbers

Landed 21:44 and 23:16 KST (≈ 92 min per epoch + eval); **0 non-finite steps** in both.

| ep | 가야금 | 거문고 | 기타 | 대금 | 아쟁 | 양금 | 타악기 | 피리 | 해금 | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | −16.58 | −17.71 | −17.90 | −15.43 | −17.61 | −21.19 | −0.35 | −13.48 | −0.96 | **−13.47** |
| 1 | −12.93 | −10.94 | −10.54 | −11.00 | −11.94 | −16.83 | +11.12 | −4.47 | +3.20 | **−7.15** |

Against the siblings' first two evals (avg): exp002 −19.34 / −9.54 · exp002.1 −14.11 /
−9.12 · exp002.3 −14.51 / −18.32 · **exp002.4 (same seed-45 head init) −14.37 / −12.23**
· exp002.2 (song-first coherent) −18.57 / −18.89. exp004's ep0 sits in the family's
warm-up band and its ep1 is the best ep1 in the family — sanity confirmed, nothing more
claimed at n=2 evals. Trajectory beyond this point → Notion after the run finishes.

## Context at launch

exp003.0 (BS-RoFormer pilot) live on GPU 0 since 08-09; exp002.4 (seed-45 incoherent
twin) live on GPU 2 since 08-13, past its collapse window and storing improving checkpoints
(ep12 −0.38 → ep14 +0.67 seen during launch prep). GPU 1 idle since exp002.1 finished. Both untouched by this launch.

## Artifacts in this folder

`launch.sh` · `SEEDS.txt` · `git_commit.txt` · `config_diff.txt` ·
`verify_start_checkpoint.py` → `start_checkpoint_check.txt` ·
`checkpoint_init_log_from_exp002.4.txt` · `verify_launch_gates.py` → `launch_gates.txt`
+ `metrics/launch_gate_summary.json` + `metrics/launch_gate_exposure.parquet` ·
`smoke_train_step.txt` · `gpu_prelaunch.txt` · `train.log` · `checkpoints/` (symlink)
· config `configs/exp004_htdemucs_coherent_p1_uniform.yaml`
