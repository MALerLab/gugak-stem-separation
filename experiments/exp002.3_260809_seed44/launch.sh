#!/usr/bin/env bash
# launch.sh — exp002.3: the THIRD SEED of exp002. Fresh, start-to-finish.
#
#   tmux new -s exp002_3 'bash experiments/exp002.3_260809_seed44/launch.sh'
#
# IDENTICAL TO exp002's LAUNCHER IN EVERY RESPECT EXCEPT:
#   1. --seed 44 (was 42)          — trainer global: data order + torch RNG
#   2. paths / run name            — run-scoped; a third arm cannot share a results dir
#   3. CUDA_VISIBLE_DEVICES=2      — GPU 0 is exp003.0's, GPU 1 is exp002.1's; both are
#                                    live and must not be perturbed. GPU 2 was freed when
#                                    exp002.2 was SIGINT-stopped at 2026-08-09T15:26:06.
# The config it reads (configs/exp002.3_htdemucs_seed44.yaml) differs from exp002's by
# gugak_mix.seed and training.run_name alone — asserted mechanically against the config
# logged in exp002's own wandb transaction log (scripts/diff_run_config.py), not by eye.
#
# WHY A THIRD SEED. exp002 (42) and exp002.1 (43) collapsed identically at epoch 12 —
# same three heads down (가야금 · 기타 · 양금), same head up (대금) — then diverged
# completely: exp002 never recovered and was stopped, exp002.1 recovered within four
# evals and has improved ever since to +3.81 dB. The failure mode is BIMODAL, so with
# n=2 neither outcome can be called the outlier. n=3 does not estimate a recovery rate;
# it breaks the tie. That is the whole job.
#
# THE THIRD SEED SURFACE lives in the warm-start checkpoint, not here. The 9 output heads
# are randomly initialized when the pretrained 4-source output layer is reshaped, so this
# arm needs its OWN start_checkpoint built at --seed 44. Reusing exp002's (42) or
# exp002.1's (43) would hand this arm someone else's init lottery ticket.
#
# FRESH, NOT A RESUME — same as exp002: --load_only_compatible_weights and NO
# --load_optimizer / --load_scheduler / --load_epoch, so lr 1e-4 is the lr it starts at.
# This project has been bitten TWICE by --load_optimizer silently restoring a halved
# learning rate; the declared lr must be the lr that trains.
#
# CHECKPOINTS LIVE ON STORAGE NVMe. $RESULTS is a symlink to
# ~/storage/gugak-stemsep-experiments/exp002.3_260809_seed44/checkpoints, because
# /home/jae.gye/userdata was 97% full at launch (130 GB free) and MSST keeps every
# improving epoch (~500 MB each; exp002.1 holds 19 GB across 41 of them).
#
# Expect the first stretch of training to look bad. The output heads are randomly
# initialized, so the model has to learn where to route nine classes before the numbers
# mean anything. That is warm-up, not a bug.
#
# ⚠️ INTERVENTION POLICY — LET IT RIDE. THE COLLAPSE IS THE MEASUREMENT.
# If this run collapses around epoch 12, DO NOT stop it, DO NOT resume from an earlier
# checkpoint, DO NOT adjust the learning rate, DO NOT touch the scheduler. exp002.1
# recovered on its own four evals later and whether this one does too is the entire
# question the run exists to answer. Do not intervene on a plateau either —
# ReduceLROnPlateau behaves as configured. Quality going down is NOT a fault.
#
# STOP POLICY: MSST has no early stopping — only ReduceLROnPlateau. The run goes to its
# 60-epoch / 150k-step ceiling unless stopped by hand. The ONLY reasons to stop early are
# genuine faults: non-finite steps, a crash, a disk or CUDA error. Stop it with SIGINT
# (Ctrl-C in the pane, or `kill -INT <python pid>`), NEVER kill -9: SIGINT raises
# KeyboardInterrupt so teardown runs and wandb closes the run cleanly. Exit 130 is read
# here as an intentional stop and does not trigger the crash-resume. Real crashes get
# exactly one automatic resume from this run's own `last`, then stop with a CRASHED_TWICE
# marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp002.3_260809_seed44
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically, and MSST hardcodes project='msst' unless WANDB_PROJECT overrides it
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # GPU 0 = exp003.0, GPU 1 = exp002.1

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — build it first:" | tee -a "$LOG"
  echo "  uv run python scripts/init_start_checkpoint.py \\" | tee -a "$LOG"
  echo "    --config configs/exp002.3_htdemucs_seed44.yaml \\" | tee -a "$LOG"
  echo "    --out $START_CKPT --log $EXP_DIR/checkpoint_init_log.txt --seed 44" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp002.3_htdemucs_seed44.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 44 --device_ids 0          # [exp002.3] seed 44; device 0 = GPU 2 after masking
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

echo "=== exp002.3 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4, seed 44) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp002.3 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp002.3 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002.3 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp002.3 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002.3 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
