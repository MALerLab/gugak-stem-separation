#!/usr/bin/env bash
# launch.sh — exp004: the FIRST anchor-cluster coherent-mixing run (static p=1.0, uniform).
#
#   tmux new -s exp004 'bash experiments/exp004_260815_coherent_p1_uniform/launch.sh'
#
# IDENTICAL TO exp002.4's LAUNCHER IN EVERY RESPECT EXCEPT:
#   1. config                      — configs/exp004_htdemucs_coherent_p1_uniform.yaml
#                                    (differs from exp002's LOGGED config in exactly 8
#                                    leaves: coherent_mix_prob 1.0, anchor_selection
#                                    uniform, cluster_min_melodic 2, density_uniform_min 2,
#                                    cluster_shared_channel_swap true, cluster_shared_gain
#                                    false, seed 45, run_name — asserted mechanically with
#                                    scripts/diff_run_config.py, 119 leaves compared)
#   2. paths / run name            — run-scoped
#   3. CUDA_VISIBLE_DEVICES=1      — GPU 0 = exp003.0 (live), GPU 2 = exp002.4 (live);
#                                    neither may be touched. GPU 1 idle at launch.
#   --seed 45 is exp002.4's value, kept: all three seed surfaces (head init, mix draw
#   stream, trainer global) sit at 45, and the head init REUSES exp002.4's seed-45
#   start_checkpoint.ckpt (byte-copied, sha256-verified, heads bit-identical to a
#   torch.manual_seed(45) fresh init — experiments/exp004_.../start_checkpoint_check.txt).
#
# WHY THIS RUN. exp002.2 (song-first coherent draw, p=1.0) collapsed and could not answer
# "does coherent training help": drawing the song first collapsed density and per-class
# exposure along with coherence. The anchor-cluster sampler (built 2026-08-15) draws
# density and class identity uniformly FIRST, then p decides which members are mutually
# coherent — exposure flat by construction (gate G1 re-verified on THIS config: max dev
# 0.28%; G5 zero clusters below the melodic minimum). exp004 is the first arm where a
# coherence result is attributable to coherence. p=1.0 under `uniform` = 2–3 real unison
# clusters per mix mixed incoherently against each other (H_melodic 0.634, ~30% of mixes
# fully coherent), NOT one song per mix. Max dose first bounds the design space.
#
# FRESH, NOT A RESUME — same as exp002: --load_only_compatible_weights and NO
# --load_optimizer / --load_scheduler / --load_epoch, so lr 1e-4 is the lr it starts at.
# This project has been bitten TWICE by --load_optimizer silently restoring a halved
# learning rate; the declared lr must be the lr that trains.
#
# CHECKPOINTS LIVE ON STORAGE NVMe. $RESULTS is a symlink to
# ~/storage/gugak-stemsep-experiments/exp004_260815_coherent_p1_uniform/checkpoints
# (userdata at 95%; MSST keeps every improving epoch at ~500 MB each).
#
# Expect the first stretch of training to look bad. The output heads are randomly
# initialized, so the model has to learn where to route nine classes before the numbers
# mean anything. That is warm-up, not a bug.
#
# ⚠️ INTERVENTION POLICY — LET IT RIDE. THE COLLAPSE IS THE MEASUREMENT.
# The exp002-family ep11-12 collapse window hits overnight. If heads collapse: DO NOT stop
# it, DO NOT resume from an earlier checkpoint, DO NOT adjust the learning rate, DO NOT
# touch the scheduler. Do not intervene on a plateau either — ReduceLROnPlateau behaves as
# configured. Quality going down is NOT a fault.
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

EXP_DIR=experiments/exp004_260815_coherent_p1_uniform
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically. ⚠️ WANDB_PROJECT does NOT override MSST's hardcoded project='msst'
# (explicit wandb.init kwarg beats the env var); fix at upload:
#   wandb sync -p gugak_stem_separation wandb/offline-run-<this run>
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # GPU 0 = exp003.0, GPU 2 = exp002.4 (both live)

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — exp004 REUSES exp002.4's seed-45 start checkpoint:" | tee -a "$LOG"
  echo "  cp -p ~/storage/gugak-stemsep-experiments/exp002.4_260813_seed45/checkpoints/start_checkpoint.ckpt $START_CKPT" | tee -a "$LOG"
  echo "  then: uv run python $EXP_DIR/verify_start_checkpoint.py" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp004_htdemucs_coherent_p1_uniform.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 45 --device_ids 0          # seed 45 (all three surfaces); device 0 = GPU 1 after masking
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

echo "=== exp004 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4, seed 45, p=1.0 uniform) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp004 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp004 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp004 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp004 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp004 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
