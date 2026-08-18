#!/usr/bin/env bash
# launch.sh — exp004.1: the INCOHERENT n≥2 CONTROL for exp004 (static p=0.0, density floor 2).
#
#   tmux new -s exp004_1 'bash experiments/exp004.1_260817_incoherent_n2/launch.sh'
#
# IDENTICAL TO exp004's LAUNCHER IN EVERY RESPECT EXCEPT:
#   1. config                      — configs/exp004.1_htdemucs_incoherent_n2.yaml
#                                    (differs from exp004's LOGGED config in exactly 2
#                                    leaves: coherent_mix_prob 1.0 → 0.0 and run_name —
#                                    asserted mechanically with scripts/diff_run_config.py,
#                                    119 leaves compared; vs exp002.4's logged config it
#                                    differs in density_uniform_min 2, the four INERT
#                                    cluster_* keys, and run_name — 6 leaves, reported)
#   2. paths / run name            — run-scoped
#   3. CUDA_VISIBLE_DEVICES=2      — GPU 0 = exp003.0 (live), GPU 1 = exp004 (live);
#                                    neither may be touched. GPU 2 idle at launch
#                                    (exp002.4 finished there 2026-08-17 03:39).
#   --seed 45 is exp002.4's / exp004's value, kept: all three seed surfaces (head init,
#   mix draw stream, trainer global) sit at 45, and the head init REUSES exp002.4's
#   seed-45 start_checkpoint.ckpt (byte-copied, sha256-verified against exp002.4's AND
#   exp004's copies; 533/533 tensors bit-identical to a fresh seed-45 rebuild —
#   experiments/exp004.1_.../start_checkpoint_check.txt). All three 2×2 arms share the
#   head init — load-bearing for the comparison.
#
# WHY THIS RUN. exp004 (coherent p=1.0, n≥2) moved TWO variables against exp002.4
# (incoherent, n≥1): coherence AND the density floor. Its lead is unattributable.
# exp004.1 is the missing 2×2 corner — incoherent, n ~ U{2..9}, seed 45 — so that
#   exp004   vs exp004.1  isolates coherence
#   exp004.1 vs exp002.4  isolates the density floor
# At p=0 the anchor-cluster path never fires (verified: 0 clusters in 200,000 draws on this
# exact config, launch_gates.txt); n and S are drawn before p, so the (n, S) stream per
# item index is the same one exp004 saw (same seed) — only what follows differs.
#
# FRESH, NOT A RESUME — same as exp002/exp004: --load_only_compatible_weights and NO
# --load_optimizer / --load_scheduler / --load_epoch, so lr 1e-4 is the lr it starts at.
# This project has been bitten TWICE by --load_optimizer silently restoring a halved
# learning rate; the declared lr must be the lr that trains.
#
# CHECKPOINTS LIVE ON STORAGE NVMe. $RESULTS is a symlink to
# ~/storage/gugak-stemsep-experiments/exp004.1_260817_incoherent_n2/checkpoints
# (userdata at 95%; MSST keeps every improving epoch at ~500 MB each).
#
# Expect the first stretch of training to look bad. The output heads are randomly
# initialized, so the model has to learn where to route nine classes before the numbers
# mean anything. That is warm-up, not a bug.
#
# ⚠️ INTERVENTION POLICY — LET IT RIDE. THE COLLAPSE IS THE MEASUREMENT (unchanged from
# EXP002.3). If heads collapse: DO NOT stop it, DO NOT resume from an earlier checkpoint,
# DO NOT adjust the learning rate, DO NOT touch the scheduler. Do not intervene on a
# plateau either — ReduceLROnPlateau behaves as configured. Quality going down is NOT a
# fault.
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

EXP_DIR=experiments/exp004.1_260817_incoherent_n2
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically. ⚠️ WANDB_PROJECT does NOT override MSST's hardcoded project='msst'
# (explicit wandb.init kwarg beats the env var); fix at upload:
#   wandb sync -p gugak_stem_separation wandb/offline-run-<this run>
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # GPU 0 = exp003.0, GPU 1 = exp004 (both live)

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — exp004.1 REUSES exp002.4's seed-45 start checkpoint:" | tee -a "$LOG"
  echo "  cp -p ~/storage/gugak-stemsep-experiments/exp002.4_260813_seed45/checkpoints/start_checkpoint.ckpt $START_CKPT" | tee -a "$LOG"
  echo "  then: uv run python $EXP_DIR/verify_start_checkpoint.py --rebuild <fresh seed-45 rebuild>" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp004.1_htdemucs_incoherent_n2.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 45 --device_ids 0          # seed 45 (all three surfaces); device 0 = GPU 2 after masking
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

echo "=== exp004.1 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4, seed 45, p=0.0 n>=2) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp004.1 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp004.1 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp004.1 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp004.1 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp004.1 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
