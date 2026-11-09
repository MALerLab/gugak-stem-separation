#!/usr/bin/env bash
# exp012 launcher — HTDemucs, item-grain coin flip at ratio 0.75 (LBD Q&A mixer test).
#
#   1. config  — configs/exp012_htdemucs_coinflip_r075.yaml (exp004 + coherent_item_ratio 0.75,
#                35-epoch cap, torch.compile; see its header)
#   2. start   — exp004's seed-45 start_checkpoint.ckpt, copied byte-identical
#                (sha256 d2f4f940…5a65, checked 2026-11-03)
#   3. GPU 2   — exp009 (BS-RoFormer, DDP) holds GPUs 0 and 1. Hard-coded, not a `:-`
#                fallback, so an inherited CUDA_VISIBLE_DEVICES can't land this on them.
#   4. THP off from the first attempt (scripts/no_thp_exec.py), as exp009.
#
# Run in tmux:  tmux new -s exp012 'bash experiments/exp012_261103_htdemucs_coinflip_r075/launch.sh'
# Stop:         Ctrl-C in the tmux pane (SIGINT, exit 130 = intentional, never resumed).
set -uo pipefail

cd /home/jae.gye/userdata/repos/gugak-lbd-qna

EXP_DIR=experiments/exp012_261103_htdemucs_coinflip_r075
CONFIG=configs/exp012_htdemucs_coinflip_r075.yaml
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target lives in .env — nothing loads it automatically. ⚠️ WANDB_PROJECT does NOT
# override MSST's hardcoded project='msst'; fix at upload:
#   wandb sync -p gugak_stem_separation wandb/offline-run-<this run>
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-lbd-qna
export CUDA_VISIBLE_DEVICES=2

mkdir -p "$RESULTS"

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

if [ ! -f "$START_CKPT" ]; then
  echo "missing $START_CKPT — exp012 REUSES exp004's seed-45 start checkpoint:" | tee -a "$LOG"
  echo "  cp -p ~/storage/gugak-stemsep-experiments/exp004_260815_coherent_p1_uniform/checkpoints/start_checkpoint.ckpt $START_CKPT" | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path "$CONFIG"
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 16 --pin_memory --persistent_workers   # exp004: 8. Compiled steps want ~31
                                    # items/s; 8 workers sustained ~17 under HTDemucs
                                    # (bench_compile.py docstring). 48 cores, exp009 uses 12.
  --seed 45 --device_ids 0          # seed 45 (all three surfaces); device 0 = GPU 2 after masking
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline                   # streaming only; the local transaction log is written
                                    # either way, so `wandb sync` uploads it later
)

run_fresh() {
  python3 scripts/no_thp_exec.py uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$START_CKPT" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # crash recovery only — restores optimizer/scheduler/epoch so the run continues
  python3 scripts/no_thp_exec.py uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_htdemucs.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp012 START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-4, seed 45, p=1.0, coin r=0.75) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp012 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp012 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp012 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp012 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp012 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
