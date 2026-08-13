#!/usr/bin/env bash
# launch.sh — exp002.2: the COHERENT-MIXING arm of exp002. Fresh, start-to-finish.
#
#   tmux new -s exp002_2 'bash experiments/exp002.2_260806_coherent/launch.sh'
#
# IDENTICAL TO exp002's LAUNCHER IN EVERY RESPECT EXCEPT:
#   1. paths / run name            — run-scoped; a separate arm needs its own results dir
#   2. CUDA_VISIBLE_DEVICES=2      — exp002 owns GPU 0, exp002.1 owns GPU 1
# The trainer --seed stays 42, the SAME as exp002 (exp002.1 is the arm that varies it).
# The config it reads (configs/exp002.2_htdemucs_coherent.yaml) differs from exp002's by
# gugak_mix.coherent_mix_prob and training.run_name alone — asserted mechanically against
# the config logged in exp002's own wandb transaction log, 114 leaf fields compared
# (scripts/diff_run_config.py).
#
# THE THIRD SEED SURFACE — the warm-start checkpoint — is ALSO held at 42 here, and that
# is the point. exp002.1 needed its own checkpoint because varying the init lottery was
# its whole purpose; exp002.2 needs the OPPOSITE. Its start_checkpoint.ckpt was rebuilt at
# seed 42 and verified bit-identical to exp002's across all 533 tensors, so both arms
# begin from precisely the same weights and the only difference between them is the
# training data. Rebuilt rather than symlinked so this arm owns its artifact and the
# reproducibility of the seeded init is re-proven rather than assumed.
#
# WHAT ACTUALLY DIFFERS: training mixtures are COHERENT — every example is n stems from
# ONE song at ONE time offset, the real ensemble playing together, instead of stems pasted
# together from unrelated songs. Gugak is heterophonic (all instruments ornamenting one
# melodic line at once), so incoherent mixes are a much easier separation problem than the
# real mixtures val and test are built from. See BUILD_REPORT.md for the verification and
# the two known confounds (n leans sparser, 양금 exposure drops 10x) — both accepted as
# structural, both to be stated in the write-up.
#
# FRESH, NOT A RESUME — same as exp002: --load_only_compatible_weights and NO
# --load_optimizer / --load_scheduler / --load_epoch, so lr 1e-4 is the lr it starts at.
#
# Expect the first stretch of training to look bad. The output heads are randomly
# initialized, so the model has to learn where to route nine classes before the numbers
# mean anything. That is warm-up, not a bug.
#
# STOP POLICY: MSST has no early stopping — only ReduceLROnPlateau. The run goes to its
# 60-epoch / 150k-step ceiling unless stopped by hand at plateau. ⚠️ Do NOT extend past the
# ceiling on a still-falling training loss (decision 2026-08-06: coherent mixing reuses
# musical moments ~32x, so a falling train loss is a weaker signal here than in exp002).
# Stop it with SIGINT (Ctrl-C in the pane, or `kill -INT <python pid>`), NEVER kill -9:
# SIGINT raises KeyboardInterrupt so teardown runs and wandb closes the run cleanly. Exit
# 130 is read here as an intentional stop and does not trigger the crash-resume. Real
# crashes get exactly one automatic resume from this run's own `last`, then stop with a
# CRASHED_TWICE marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp002.2_260806_coherent
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically, and MSST hardcodes project='msst' unless WANDB_PROJECT overrides it
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # exp002 owns GPU 0, exp002.1 GPU 1

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — build it first:" | tee -a "$LOG"
  echo "  uv run python scripts/init_start_checkpoint.py \\" | tee -a "$LOG"
  echo "    --config configs/exp002.2_htdemucs_coherent.yaml \\" | tee -a "$LOG"
  echo "    --out $START_CKPT --log $EXP_DIR/checkpoint_init_log.txt --seed 42" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp002.2_htdemucs_coherent.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0          # seed 42 = exp002's; device 0 = GPU 2 after masking
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

echo "=== exp002.2 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4, seed 42, coherent p=1.0) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp002.2 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp002.2 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002.2 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp002.2 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp002.2 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
