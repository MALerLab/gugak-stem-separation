#!/usr/bin/env bash
# continue.sh — exp008.1: carry exp008 from its 35-epoch cap to 60. TWO GPUs (0 and 1).
#
# This is a CONTINUATION, not a new run, mirroring exp006 -> exp006.1 exactly (adapted from
# experiments/exp006.1_260929_continued_to_60/continue.sh on teo/training-speedup), so the
# incoherent control and the partially coherent arm are compared at the same 60-epoch cap.
# Normally started by ../exp008_261011_bsroformer_incoherent_n2/run_chain.sh, after it has
# snapshotted last_bs_roformer.ckpt -> last_bs_roformer_ep34.ckpt.
#
# It resumes from exp008's last_bs_roformer.ckpt, which records epoch 34; train.py sets
# start_epoch = epoch + 1, so training picks up at 35. Nothing is recomputed, nothing skipped.
# Checkpoints keep landing in exp008's results directory so the ladder stays ONE continuous
# series — read it from the FILENAMES, which carry each epoch's own score.
#
# ⚠️ --load_optimizer IS CORRECT HERE. It restores optimizer and scheduler state, which is
# exactly what a continuation needs. On a FRESH run the same flag silently reinstates a halved
# learning rate — this project has been bitten by that twice. Do not copy this flag block into
# a fresh launcher.
#
# ⚠️ Stale best bar: MSST writes the PREVIOUS best metric into the file it is saving, so the
# first epoch above that stale bar is filed as "best" even if below ep34's own score. Cosmetic.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. If heads collapse: no stop, no resume, no hand-set lr,
# no scheduler edits. Only a genuine fault — non-finite steps, crash, disk or CUDA error —
# justifies stopping. Quality going down is not a fault.
#
# GPU 2 is reserved for evals. Never widen CUDA_VISIBLE_DEVICES.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation-CLAUDE-exp008

RUN=exp008.1
SESSION=exp008                    # the chain's tmux session (run_chain.sh). tmux rewrites '.'
                                  # to '_' and parses '.' in -t, so no dotted session names
EXP_DIR=experiments/exp008.1_261011_continued_to_60
PREV_EXP_DIR=experiments/exp008_261011_bsroformer_incoherent_n2
PREV_RESULTS=$PREV_EXP_DIR/checkpoints
RESULTS=$PREV_RESULTS             # one continuous ladder, deliberately
LOG=$EXP_DIR/train.log
RESUME_CKPT=$RESULTS/last_bs_roformer.ckpt

set -a; source .env; set +a
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation-CLAUDE-exp008
export CUDA_VISIBLE_DEVICES=0,1   # hard-coded: a `:-` fallback would honour an inherited
                                  # value and could land this on the eval card
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

# exp008 must actually have finished, or this is resuming something still moving
if grep -q "=== exp008 finished clean" "$PREV_EXP_DIR/train.log" 2>/dev/null; then
  guard_pass "exp008 finished clean"
else
  guard_fail "exp008 has no 'finished clean' marker — not safe to continue from its last checkpoint"
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
  --config_path configs/exp008.1_bsroformer_continued_to_60.yaml
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
  # THP disabled for the trainer and everything it spawns — exp008 crashed twice on NCCL
  # watchdog timeouts caused by THP compaction stalls in validation (2026-10-12)
  python3 scripts/no_thp_exec.py uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESUME_CKPT" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} START $(date -Is) continuing exp008 from ep34 -> cap 60 (2 ranks, compile on) ===" | tee -a "$LOG"
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
