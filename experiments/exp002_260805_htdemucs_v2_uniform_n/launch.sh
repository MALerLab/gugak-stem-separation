#!/usr/bin/env bash
# launch.sh — exp002: the canonical HTDemucs baseline. Fresh, start-to-finish.
#
#   tmux new -s exp002 'bash experiments/exp002_260805_htdemucs_v2_uniform_n/launch.sh'
#
# FRESH, NOT A RESUME. Starts from the warm-start checkpoint built by
# scripts/init_start_checkpoint.py (pretrained htdemucs trunk + freshly initialized
# 9-source heads) with --load_only_compatible_weights and NO --load_optimizer /
# --load_scheduler / --load_epoch. That is the whole point: exp001 inherited a halved
# learning rate through a resumed optimizer, and exp002 exists so nothing is inherited.
# The config's lr 1e-4 is therefore the lr the run actually starts at.
#
# Expect the first stretch of training to look bad. The output heads are randomly
# initialized, so the model has to learn where to route nine classes before the numbers
# mean anything. That is warm-up, not a bug.
#
# STOP POLICY: MSST has no early stopping — only ReduceLROnPlateau. The run goes to its
# 60-epoch / 150k-step ceiling unless stopped by hand at plateau. Stop it with SIGINT
# (Ctrl-C in the pane, or `kill -INT <python pid>`), NEVER kill -9: SIGINT raises
# KeyboardInterrupt so teardown runs and wandb closes the run cleanly. Exit 130 is read
# here as an intentional stop and does not trigger the crash-resume. Real crashes get
# exactly one automatic resume from this run's own `last`, then stop with a CRASHED_TWICE
# marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp002_260805_htdemucs_v2_uniform_n
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically, and MSST hardcodes project='msst' unless WANDB_PROJECT overrides it
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — build it first:" | tee -a "$LOG"
  echo "  uv run python scripts/init_start_checkpoint.py \\" | tee -a "$LOG"
  echo "    --config configs/exp002_htdemucs_v2_uniform_n.yaml \\" | tee -a "$LOG"
  echo "    --out $START_CKPT --log $EXP_DIR/checkpoint_init_log.txt --seed 42" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp002_htdemucs_v2_uniform_n.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0
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
    --start_check_point "$RESULTS/last_htdemucs.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp002 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp002 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp002 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp002 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
