#!/usr/bin/env bash
# continue.sh — exp006.1: carry exp006 from its 35-epoch cap to 60. TWO GPUs (0 and 1).
#
# This is a CONTINUATION, not a new run. exp006 stopped at 35 because 35 was the budget, not
# because it converged: its ladder rose monotonically through the cap (ep29 11.0852 → ep34
# 11.2352, +0.15 dB over the last five epochs) having already beaten exp003.0's ep59 best
# (10.7013) by +0.53 dB with 25 fewer epochs. This looks for the ceiling and puts exp006 on
# exp003.0's own 60-epoch protocol.
#
# It resumes from exp006's last_bs_roformer.ckpt, which records epoch 34; train.py sets
# start_epoch = epoch + 1, so training picks up at 35. Nothing is recomputed, nothing skipped.
# Checkpoints keep landing in exp006's results directory so the ladder stays ONE continuous
# series — read it from the FILENAMES, which carry each epoch's own score.
#
# ⚠️ --load_optimizer IS CORRECT HERE. It restores optimizer and scheduler state, which is
# exactly what a continuation needs. On a FRESH run the same flag silently reinstates a halved
# learning rate — this project has been bitten by that twice. Do not copy this flag block into
# a fresh launcher.
#
# ⚠️ Stale best bar: MSST writes the PREVIOUS best metric into the file it is saving, so the
# checkpoint carries best_metric 11.2111 while ep34 actually scored 11.2352. The first epoch
# above 11.2111 is therefore filed as "best" even if below 11.2352. Cosmetic only.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. If heads collapse: no stop, no resume, no hand-set lr,
# no scheduler edits. Only a genuine fault — non-finite steps, crash, disk or CUDA error —
# justifies stopping. Quality going down is not a fault.
#
# GPU 2 belongs to exp007 for about another day and a half. Never widen CUDA_VISIBLE_DEVICES.
set -u
cd /home/jae.gye/userdata/repos/gugak-stemsep-teo

RUN=exp006.1
SESSION=exp006_1                  # tmux rewrites '.' to '_' and parses '.' in -t as a window
                                  # separator, so a dotted session name false-negatives
EXP_DIR=experiments/exp006.1_260929_continued_to_60
PREV_RESULTS=experiments/exp006_260925_bsroformer_partial_coherent/checkpoints
RESULTS=$PREV_RESULTS             # one continuous ladder, deliberately
LOG=$EXP_DIR/train.log
RESUME_CKPT=$RESULTS/last_bs_roformer.ckpt

set -a; source .env; set +a
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-stemsep-teo
export CUDA_VISIBLE_DEVICES=0,1   # hard-coded: a `:-` fallback would honour an inherited
                                  # value and could land this on exp007's card
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$EXP_DIR"

fails=0
guard_fail() { echo "  ✗ $1" | tee -a "$LOG"; fails=$((fails + 1)); }
guard_pass() { echo "  ✓ $1"; }
echo "=== ${RUN} pre-flight $(date -Is) ===" | tee -a "$LOG"

for card in 0 1; do
  pids=$(nvidia-smi --id="$card" --query-compute-apps=pid --format=csv,noheader | tr -d ' \n')
  mem=$(nvidia-smi --id="$card" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ -n "$pids" ]; then guard_fail "card ${card} busy — pids ${pids}, ${mem} MiB"
  elif [ "${mem:-0}" -gt 1024 ]; then guard_fail "card ${card} shows ${mem} MiB with no process — ghost context"
  else guard_pass "card ${card} idle (${mem} MiB)"; fi
done

# exp006 must actually have finished, or this is resuming something still moving
if grep -q "=== exp006 finished clean" experiments/exp006_260925_bsroformer_partial_coherent/train.log 2>/dev/null; then
  guard_pass "exp006 finished clean"
else
  guard_fail "exp006 has no 'finished clean' marker — not safe to continue from its last checkpoint"
fi

cur_session=$(tmux display-message -p '#S' 2>/dev/null || true)
if [ -n "$cur_session" ] && [ "$cur_session" = "$SESSION" ]; then
  guard_pass "inside tmux session '${SESSION}' (this launcher's own)"
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  guard_fail "tmux session '${SESSION}' already exists and is not ours"
else guard_pass "no foreign tmux session '${SESSION}'"; fi

if [ -f "$LOG" ] && grep -q "=== ${RUN} START" "$LOG"; then
  guard_fail "${LOG} already carries a START line"
else guard_pass "no prior ${RUN} START line"; fi

[ -f "$RESUME_CKPT" ] && guard_pass "resume checkpoint present ($(du -h "$RESUME_CKPT" | cut -f1))" \
  || guard_fail "missing $RESUME_CKPT"
[ -f .env ] && guard_pass ".env present" || guard_fail ".env missing"

if [ "$fails" -gt 0 ]; then
  echo "REFUSING to continue ${RUN} — ${fails} guard(s) failed. Nothing started." | tee -a "$LOG"
  exit 1
fi

git rev-parse HEAD > $EXP_DIR/git_commit.txt
git -c filter.nbstripout.clean=cat status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse --abbrev-ref HEAD >> $EXP_DIR/git_commit.txt

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path configs/exp006.1_bsroformer_continued_to_60.yaml
  --results_path "$RESULTS"
  --data_path dummy
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val
  --num_workers 6 --pin_memory --persistent_workers
  --seed 42 --device_ids 0 1
  --use_standard_loss
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline
)

run_resume() {
  uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESUME_CKPT" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} START $(date -Is) continuing exp006 from ep34 -> cap 60 (2 ranks, compile on) ===" | tee -a "$LOG"
run_resume
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== ${RUN} finished clean $(date -Is) ===" | tee -a "$LOG"; exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== ${RUN} stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"; exit 0
fi

echo "=== ${RUN} CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
run_resume
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== ${RUN} ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"; exit 0
fi
echo "=== ${RUN} CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
