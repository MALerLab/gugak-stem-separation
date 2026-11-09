#!/usr/bin/env bash
# resume.sh — exp009 MANUAL RESUME, only after launch.sh stopped with CRASHED_TWICE and Jae
#             approved it. Same run, same results dir, same config; continues from
#             last_bs_roformer.ckpt. TWO GPUs (0 and 1). THP off, as launch.sh.
#
#   Start it in the exp009 tmux session (or a fresh one of that name):
#     tmux new -d -s exp009 'bash experiments/exp009_261103_bsroformer_partial_coherent_p1/resume.sh'
#
# Adapted from experiments/exp008_261011_bsroformer_incoherent_n2/resume.sh. Crash policy as
# launch.sh: resume once, then stop.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. No hand-set lr, no scheduler edits.
set -u
cd /home/jae.gye/userdata/repos/gugak-exp009

RUN=exp009
SESSION=exp009
EXP_DIR=experiments/exp009_261103_bsroformer_partial_coherent_p1
CONFIG=configs/exp009_bsroformer_partial_coherent_p1.yaml
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
RESUME_CKPT=$RESULTS/last_bs_roformer.ckpt
MSST_COMMIT=31110ff54b7af035a096c17289d41dc722cea9d4

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-exp009
export CUDA_VISIBLE_DEVICES=0,1   # hard-coded: GPU 2 is reserved for other jobs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- pre-flight guards: refuse to start if any fails ---
fails=0
guard_fail() { echo "  ✗ $1" | tee -a "$LOG"; fails=$((fails + 1)); }
guard_pass() { echo "  ✓ $1"; }
echo "=== ${RUN} resume pre-flight $(date -Is) ===" | tee -a "$LOG"

for card in 0 1; do
  pids=$(nvidia-smi --id="$card" --query-compute-apps=pid --format=csv,noheader | tr -d ' \n')
  mem=$(nvidia-smi --id="$card" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ -n "$pids" ]; then guard_fail "card ${card} busy — pids ${pids}, ${mem} MiB"
  elif [ "${mem:-0}" -gt 1024 ]; then guard_fail "card ${card} shows ${mem} MiB with no process — ghost context"
  else guard_pass "card ${card} idle (${mem} MiB)"; fi
done

cur_session=$(tmux display-message -p '#S' 2>/dev/null || true)
if [ -n "$cur_session" ] && [ "$cur_session" = "$SESSION" ]; then
  guard_pass "inside tmux session '${SESSION}' (this launcher's own)"
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  guard_fail "tmux session '${SESSION}' already exists and is not ours"
else guard_pass "no foreign tmux session '${SESSION}'"; fi

# only resume a run that actually stopped, and only once by hand
[ -f "$EXP_DIR/CRASHED_TWICE" ] && guard_pass "${RUN} stopped (CRASHED_TWICE present)" \
  || guard_fail "no CRASHED_TWICE marker — ${RUN} may still be running"
if grep -q "=== ${RUN} finished clean" "$LOG"; then guard_fail "${RUN} already finished clean"
else guard_pass "${RUN} not finished"; fi
if grep -q "=== ${RUN} RESUME START" "$LOG"; then guard_fail "${LOG} already carries a RESUME START line"
else guard_pass "no prior manual RESUME START"; fi

[ -f "$RESUME_CKPT" ] && guard_pass "resume checkpoint present ($(du -h "$RESUME_CKPT" | cut -f1))" \
  || guard_fail "missing $RESUME_CKPT"
[ -f .env ] && guard_pass ".env present" || guard_fail ".env missing"
msst_head=$(git -C external/msst rev-parse HEAD 2>/dev/null)
[ "$msst_head" = "$MSST_COMMIT" ] && guard_pass "external/msst at ${MSST_COMMIT:0:7}" \
  || guard_fail "external/msst at ${msst_head:-?}, expected ${MSST_COMMIT}"

if [ "$fails" -gt 0 ]; then
  echo "REFUSING to resume ${RUN} — ${fails} guard(s) failed. Nothing started." | tee -a "$LOG"
  exit 1
fi

set -a; source .env; set +a

{ echo "--- manual resume $(date -Is)"; git rev-parse HEAD; git -c filter.nbstripout.clean=cat status --short
  git -C external/msst rev-parse HEAD; } >> $EXP_DIR/git_commit.txt

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path "$CONFIG"
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
  python3 scripts/no_thp_exec.py uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESUME_CKPT" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} RESUME START $(date -Is) from last_bs_roformer.ckpt -> cap 35, THP off ===" | tee -a "$LOG"
run_resume
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== ${RUN} finished clean (after manual resume) $(date -Is) ===" | tee -a "$LOG"; exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== ${RUN} stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"; exit 0
fi

echo "=== ${RUN} CRASHED after manual resume (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
run_resume
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== ${RUN} ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"; exit 0
fi
echo "=== ${RUN} CRASHED TWICE after manual resume (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
exit "$code"
