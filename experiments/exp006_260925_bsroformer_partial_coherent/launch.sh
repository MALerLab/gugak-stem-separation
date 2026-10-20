#!/usr/bin/env bash
# launch.sh — exp006: BS-RoFormer on PARTIALLY COHERENT mixtures (p = 0.5).
#             Fresh, TWO GPUs (0 and 1) under DDP, torch.compile ON, 35-epoch cap.
#
#   Do NOT start this by hand. scripts/launch_exp006.sh is the entry point: it runs the
#   pre-flight guards (both cards idle, no double launch, config parses, start checkpoint
#   sha256) and then starts THIS file inside `tmux new -d -s exp006`.
#
# WHAT IS DIFFERENT FROM exp003.x, and why each difference is safe:
#   * external/msst/train_ddp.py, not train.py. world_size comes from
#     torch.cuda.device_count(), so CUDA_VISIBLE_DEVICES below is what picks two cards.
#   * batch 4 x accum 4 x 2 ranks = effective batch 32, and num_steps 10000 x 4 x 2 =
#     80,000 items/epoch = 2,500 optimizer steps. Both identical to exp003.x. The config
#     header carries the full derivation.
#   * torch.compile via config.training.use_torch_compile. MEASURED 1.91x at batch 2 and
#     2.22x per item at batch 4 (scripts/bench_compile.py). NOT bit-comparable to
#     exp003.x — fusion reorders reductions. Stated in the config header; do not read
#     a last-digit difference as a result.
#   * The MSST submodule is on branch teo/ddp-fixes, which carries four DDP fixes this
#     run depends on. Two of them are correctness, not speed:
#       - valid.py gathered per-instrument metrics as ZERO-PADDED vectors, so every
#         sparse stem's mean was diluted with literal 0.0 dB entries. exp003.2's
#         10.1379 reads about 6.2 through the unpatched path.
#       - only rank 0 stepped ReduceLROnPlateau, so the ranks' learning rates diverge
#         the moment the reducer fires, and DDP never re-syncs parameters.
#     Running this config against the unpatched submodule produces a plausible-looking
#     run whose val curve is fiction. The pre-flight checks the submodule branch.
#
# CARD ETIQUETTE. This takes GPUs 0 and 1 and deliberately leaves GPU 2 for the rest of
# the lab. GPU 2 is also the card that runs hottest (MEASURED smClock p50 2332 MHz vs
# 2407 on 0/1), so it is the right one to give away for a synchronous two-rank job.
#
# 35 EPOCHS, THEN A DECISION. exp003.0's ladder puts ep35 at 10.2543 against its ep59
# best of 10.7013. This arm stops at 35 and is compared to exp003.0 AT EPOCH 35. If it
# wins, continue to 60 — see CONTINUING TO 60 at the bottom of this file. That is why
# `last_bs_roformer.ckpt` must never be pruned.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. If heads collapse: no stop, no resume, no hand-set
# lr, no scheduler edits. Only a genuine fault — non-finite steps, crash, disk or CUDA
# error — justifies stopping. Quality going down is not a fault.
set -u
cd /home/jae.gye/userdata/repos/gugak-stemsep-teo

RUN=exp006
EXP_DIR=experiments/exp006_260925_bsroformer_partial_coherent
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically. NOTE: WANDB_PROJECT does NOT win — MSST's utils/settings.py passes
# project='msst' as an explicit wandb.init() kwarg, which beats the env var. WANDB_ENTITY
# does apply. So expect this under entity maler-gye, project 'msst', and fix it at sync:
#   wandb sync -p gugak_stem_separation wandb/offline-run-*
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-stemsep-teo
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# one BLAS thread per dataloader worker: 2 ranks x 6 workers on a 24-core cpuset, and
# the loader already has ~5x headroom, so threads here only buy cache thrash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit and branch
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse --abbrev-ref HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst status --short >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — exp006 REUSES exp003.0's seed-42 start checkpoint (byte copy):" | tee -a "$LOG"
  echo "  cp -p /home/jae.gye/storage/gugak-stemsep-experiments/exp003.0_260809_bsroformer_pilot/checkpoints/start_checkpoint.ckpt $START_CKPT" | tee -a "$LOG"
  echo "  then verify: sha256sum -c $EXP_DIR/start_checkpoint_sha256.txt" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path configs/exp006_bsroformer_partial_coherent.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 6 --pin_memory --persistent_workers          # per RANK, 24-core cpuset
  --seed 42 --device_ids 0 1
  --use_standard_loss               # NOT optional — it gates the loss-selection branch
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline                   # streaming only; the local transaction log is written
                                    # either way, so `wandb sync` uploads it later
)

run_fresh() {
  # FRESH: strict load of the plain state_dict the init script wrote. NO --load_* flags.
  # Passing --load_optimizer here would silently restore a HALVED learning rate — this
  # project has been bitten by exactly that twice. The declared lr must be the lr trained.
  uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$START_CKPT" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # CRASH RECOVERY ONLY — restores optimizer/scheduler/epoch so the run continues where it
  # died rather than restarting the experiment.
  uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_bs_roformer.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-5, seed 42, batch 4x4 x 2 ranks, compile on, cap 35) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== ${RUN} finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== ${RUN} stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== ${RUN} CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_bs_roformer.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== ${RUN} ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== ${RUN} CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"

# ---------------------------------------------------------------------------------------
# CONTINUING TO 60 (only if ep35 beats exp003.0's ep35 = 10.2543 by enough to be worth
# another ~2 days of two cards). This is a deliberate human decision, not automatic:
#
#   1. edit configs/exp006_bsroformer_partial_coherent.yaml   training.num_epochs: 35 -> 60
#   2. bash -c 'source .env; ... run_resume'  — i.e. relaunch this file with the resume
#      branch, which passes --load_epoch so training picks up at 35 rather than restarting
#   3. the resume path needs last_bs_roformer.ckpt, so never prune it
#
# Note while reading checkpoints: MSST writes the PREVIOUS best metric into the file it is
# currently saving (verified here — exp003.0's ep59 file is named 10.7013 but records
# best_metric 10.688). Trust the FILENAME for the epoch's own score.
