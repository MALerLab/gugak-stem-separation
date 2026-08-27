#!/usr/bin/env bash
# launch.sh — exp003.1: the SEED TWIN of exp003.0 (BS-RoFormer pilot). Fresh, GPU 1.
#
#   tmux new -s exp003.1 'bash experiments/exp003.1_260819_bsroformer_seed46/launch.sh'
#
# IDENTICAL to experiments/exp003.0_260809_bsroformer_pilot/launch.sh except:
#   - config    configs/exp003.1_bsroformer_seed46.yaml   (gugak_mix.seed 42 → 46, run_name)
#   - --seed 46 on the trainer (surface 2) and on the start-checkpoint builder (surface 3)
#   - its own start_checkpoint.ckpt built at seed 46 (same trunk transfer, fresh heads)
#   - default card GPU 1 (exp003.0 keeps GPU 0, exp004.1 keeps GPU 2)
#   - run-scoped names/paths
# Every note in exp003.0's launch.sh — fresh-not-resume, --use_standard_loss being
# load-bearing, "first stretch looks very bad" (87% fresh params), SIGINT-only stop
# policy, one auto-resume then CRASHED_TWICE — applies unchanged.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp003.1_260819_bsroformer_seed46
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

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
  echo "    --config configs/exp003.1_bsroformer_seed46.yaml \\" | tee -a "$LOG"
  echo "    --pretrained ~/storage/gugak-stemsep-experiments/pretrained/model_bs_roformer_ep_17_sdr_9.6568.ckpt \\" | tee -a "$LOG"
  echo "    --out $START_CKPT --log $EXP_DIR/checkpoint_init_log.txt --seed 46 --verify-twice" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path configs/exp003.1_bsroformer_seed46.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
                                    # (no --extension: that is a valid.py-only flag.
                                    # train.py's globber matches mixture.wav AND
                                    # mixture.flac unconditionally, and GT stems fall
                                    # back through [args.extension, flac, wav] — which
                                    # is why exp002 never passed it either.)
  --num_workers 8 --pin_memory --persistent_workers
  --seed 46 --device_ids 0
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

echo "=== exp003.1 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-5, batch 2x16) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp003.1 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp003.1 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp003.1 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_bs_roformer.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp003.1 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp003.1 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
