#!/usr/bin/env bash
# launch.sh — exp003.0: the BS-RoFormer PILOT. Fresh, start-to-finish, GPU 0.
#
#   tmux new -s exp003.0 'bash experiments/exp003.0_260809_bsroformer_pilot/launch.sh'
#
# ⚠️ PILOT, NOT exp003. This run exists to prove the BS-RoFormer pipeline stands up end
# to end and to produce a first sanity curve, and to ask one diagnostic question: does
# the head collapse seen under HTDemucs's shared decoder (가야금 · 기타 · 양금 going quiet
# while 대금 gains, across two seeds) also happen when every source gets its OWN mask
# estimator? Its learning rate is an untuned guess. Its numbers are not results.
#
# FRESH, NOT A RESUME. Starts from the warm-start checkpoint built by
# scripts/init_bsroformer_start_checkpoint.py (pretrained MUSDB trunk + 9 freshly
# initialized per-source mask estimators) with --load_only_compatible_weights and NO
# --load_optimizer / --load_scheduler / --load_epoch. The config's lr 1e-5 is therefore
# the lr the run actually starts at — nothing is inherited through an optimizer state.
#
# Expect the first stretch to look very bad, worse than exp002's warm-up did. 87% of this
# model's parameters are freshly initialized, because BS-RoFormer keeps ~75% of its
# capacity in the per-source mask estimators and we deliberately reinitialized all nine
# so that no output head starts with an advantage. Measured starting point on three val
# songs: SI-SDR about −33 dB average. That is warm-up, not a bug.
#
# --use_standard_loss IS LOAD-BEARING. Without it MSST routes bs_roformer through the
# model's own internal loss (waveform L1 PLUS a multi-resolution STFT term), which would
# silently make the loss differ from exp002's. The spec says L1 on waveforms, unchanged.
#
# STOP POLICY: MSST has no early stopping — only ReduceLROnPlateau. The run goes to its
# 60-epoch ceiling unless stopped by hand. Stop it with SIGINT (Ctrl-C in the pane, or
# `kill -INT <python pid>`), NEVER kill -9: SIGINT raises KeyboardInterrupt so teardown
# runs and wandb closes the run cleanly. Exit 130 is read here as an intentional stop and
# does not trigger the crash-resume. Real crashes get exactly one automatic resume from
# this run's own `last`, then stop with a CRASHED_TWICE marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp003.0_260809_bsroformer_pilot
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically. NOTE: WANDB_PROJECT does NOT win here — MSST's utils/settings.py passes
# project='msst' as an explicit kwarg, which beats the env var. WANDB_ENTITY does apply.
# So expect this run under entity maler-gye, project 'msst', and fix it at sync time:
#   wandb sync -p gugak_stem_separation wandb/offline-run-*
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit.
# The tree is deliberately NOT clean at launch (waived for this run) — the dirty list
# below IS the record of what differed from HEAD when this started.
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst status --short >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — build it first:" | tee -a "$LOG"
  echo "  uv run python scripts/init_bsroformer_start_checkpoint.py \\" | tee -a "$LOG"
  echo "    --config configs/exp003.0_bsroformer_pilot.yaml \\" | tee -a "$LOG"
  echo "    --pretrained ~/storage/gugak-stemsep-experiments/pretrained/model_bs_roformer_ep_17_sdr_9.6568.ckpt \\" | tee -a "$LOG"
  echo "    --out $START_CKPT --log $EXP_DIR/checkpoint_init_log.txt --seed 42 --verify-twice" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path configs/exp003.0_bsroformer_pilot.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
                                    # (no --extension: that is a valid.py-only flag.
                                    # train.py's globber matches mixture.wav AND
                                    # mixture.flac unconditionally, and GT stems fall
                                    # back through [args.extension, flac, wav] — which
                                    # is why exp002 never passed it either.)
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0
  --use_standard_loss               # see the note above — NOT optional
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline                   # streaming only; the local transaction log is
                                    # written either way, so `wandb sync` uploads it
                                    # later (or mid-run) without losing anything
)

run_fresh() {
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$START_CKPT" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # crash recovery only — restores optimizer/scheduler/epoch so the run continues
  # where it died rather than restarting the experiment
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_bs_roformer.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp003.0 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-5, batch 2x16) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp003.0 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp003.0 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp003.0 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_bs_roformer.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp003.0 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp003.0 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
