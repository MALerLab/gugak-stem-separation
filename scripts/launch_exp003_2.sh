#!/usr/bin/env bash
# launch_exp003_2.sh — the ONE-PASTE launcher for exp003.2 (BS-RoFormer, lr 1e-5 → 2e-5, GPU 2).
#
#   bash ~/userdata/repos/gugak_stem_separation/scripts/launch_exp003_2.sh
#
# Written 2026-08-19 to be fired ~2026-08-21 over SSH from a phone, zero decisions.
# It REFUSES (prints why, exits 1, touches nothing) unless every pre-flight guard passes:
#   1. GPU 2 carries no process and its memory is ~empty (a ghost context after a hard
#      kill shows utilisation with no process — that is also a refusal, not a free card)
#   2. exp004.1 has FINISHED — its train.log carries the "finished clean" (or "ended
#      after resume") marker AND its final `last_htdemucs.ckpt` is on disk
#   3. exp003.2 has not already been launched (no tmux session, no START line in its log)
#   4. the run's inputs resolve: config parses through MSST's own loader, .env exists,
#      inner launch.sh exists, start_checkpoint.ckpt sha256 == exp003.0's seed-42 one
# There is deliberately NO force flag. If a guard is wrong, fix the world, not the guard.
#
# On success it: starts experiments/exp003.2_*/launch.sh inside `tmux new -d -s exp003.2`
# (offline wandb, checkpoints on storage NVMe — both handled by the inner script), adds
# exp003.2 to configs/run_digest.yaml (idempotent — the monitor session reads it),
# appends one dated line to holiday_breadcrumbs.md, waits for the process to appear on
# GPU 2, then prints exactly one closing line: "launched — check tomorrow's digest".
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation || exit 1

RUN=exp003.2
EXP_DIR=experiments/exp003.2_260821_bsroformer_lr2e-5
CONFIG=configs/exp003.2_bsroformer_lr2e-5.yaml
INNER=$EXP_DIR/launch.sh
LOG=$EXP_DIR/train.log
START_CKPT=$EXP_DIR/checkpoints/start_checkpoint.ckpt
START_CKPT_SHA=$EXP_DIR/start_checkpoint_sha256.txt
GPU_INDEX=2                       # nvidia-smi index == CUDA_VISIBLE_DEVICES index here
                                  # (verified 2026-08-19: exp004.1, launched with
                                  # CUDA_VISIBLE_DEVICES=2, shows on nvidia-smi index 2)
GPU_IDLE_MAX_MIB=1024             # an idle Blackwell card reports 0 MiB; anything above
                                  # this with no process is a ghost context
TMUX_SESSION=exp003_2            # NOT exp003.2 — tmux swaps '.' to '_' on creation, and
                                  # '.' in a -t target parses as a window separator, so
                                  # has-session on a dotted name false-negatives. This bug
                                  # made the 2026-08-26 launch look dead when it was fine.
PREV_RUN=exp004.1
PREV_EXP_DIR=experiments/exp004.1_260817_incoherent_n2
PREV_LOG=$PREV_EXP_DIR/train.log
PREV_LAST_CKPT=$PREV_EXP_DIR/checkpoints/last_htdemucs.ckpt   # MSST rewrites `last` every
                                  # epoch; a model_*_ep_59_* file only exists if ep59 was
                                  # the best epoch (exp004 finished with none), so `last`
                                  # + the log marker is the reliable "final" signal
DIGEST=configs/run_digest.yaml
DIGEST_LINE="  ${RUN}: ${EXP_DIR}/train.log"
BREADCRUMBS=holiday_breadcrumbs.md
GPU_WAIT_S=300                    # startup: uv + torch import + 1 GB checkpoint load
                                  # + dataset init; exp003.0 was on-GPU well inside this

failures=0
pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; failures=$((failures + 1)); }

echo "=== ${RUN} pre-flight $(date -Is) ==="

# --- guard 1: GPU must be free ---
gpu_pids=$(nvidia-smi --id="$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ' | sed 's/ *$//')
gpu_mem=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ -n "$gpu_pids" ]; then
  fail "GPU ${GPU_INDEX} busy — process(es) ${gpu_pids}, ${gpu_mem} MiB in use"
elif [ "${gpu_mem:-0}" -gt "$GPU_IDLE_MAX_MIB" ]; then
  fail "GPU ${GPU_INDEX} shows ${gpu_mem} MiB with NO process — ghost context (hard-killed job); card needs a host reboot"
else
  pass "GPU ${GPU_INDEX} idle (no process, ${gpu_mem} MiB)"
fi

# --- guard 2: the previous tenant must have finished ---
if [ ! -f "$PREV_LOG" ]; then
  fail "${PREV_RUN} log missing: ${PREV_LOG}"
elif ! grep -q "=== ${PREV_RUN} finished clean\|=== ${PREV_RUN} ended after resume" "$PREV_LOG"; then
  last_epoch=$(grep "Train epoch" "$PREV_LOG" | tail -1 | tr -d '\r')
  fail "${PREV_RUN} NOT finished — no 'finished clean' marker in ${PREV_LOG} (last: '${last_epoch:-no epoch line}')"
elif [ ! -f "$PREV_LAST_CKPT" ]; then
  fail "${PREV_RUN} final checkpoint missing: ${PREV_LAST_CKPT}"
else
  pass "${PREV_RUN} finished ($(grep -o "=== ${PREV_RUN} [a-z ]*[0-9T:+-]*" "$PREV_LOG" | tail -1)); final ckpt on disk"
fi

# --- guard 3: never launch twice ---
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  fail "tmux session '${TMUX_SESSION}' already exists — ${RUN} is (or was) already launched"
elif [ -f "$LOG" ] && grep -q "=== ${RUN} START" "$LOG"; then
  fail "${LOG} already has a START line — ${RUN} was launched before: $(grep "=== ${RUN} START" "$LOG" | head -1)"
else
  pass "no prior ${RUN} launch (no tmux session, no START line)"
fi

# --- guard 4: inputs resolve ---
[ -f .env ]        && pass ".env present (wandb target)"          || fail ".env missing at repo root"
[ -x "$INNER" ]    && pass "inner runner ${INNER}"                 || fail "inner runner missing/not executable: ${INNER}"
if [ -f "$CONFIG" ]; then
  if lr=$(PYTHONPATH=external/msst uv run python -c "from utils.settings import load_config; c=load_config('bs_roformer','${CONFIG}'); print(c.training.lr, c.training.run_name)" 2>&1); then
    pass "config parses via MSST loader: ${CONFIG} → lr/run_name = ${lr}"
  else
    fail "config failed to parse: ${CONFIG} — ${lr}"
  fi
else
  fail "config missing: ${CONFIG}"
fi
if [ -f "$START_CKPT" ] && [ -f "$START_CKPT_SHA" ]; then
  expected_sha=$(grep -o '^[0-9a-f]\{64\}' "$START_CKPT_SHA" | head -1)   # exp003.0's, recorded 2026-08-19
  actual_sha=$(sha256sum "$START_CKPT" | cut -d' ' -f1)                       # ~1 GB, a few seconds
  if [ -n "$expected_sha" ] && [ "$actual_sha" = "$expected_sha" ]; then
    pass "start_checkpoint.ckpt sha256 matches exp003.0's seed-42 head init"
  else
    fail "start_checkpoint.ckpt sha256 MISMATCH vs ${START_CKPT_SHA} — head init would differ from exp003.0"
  fi
else
  fail "start checkpoint or its sha256 record missing: ${START_CKPT} / ${START_CKPT_SHA}"
fi

if [ "$failures" -gt 0 ]; then
  echo "REFUSING to launch ${RUN} — ${failures} guard(s) failed (✗ above). Nothing started, nothing written."
  exit 1
fi

# --- launch: detached tmux, inner script owns .env, wandb offline, checkpoints, resume policy ---
echo "=== all guards passed — launching ${RUN} in tmux '${TMUX_SESSION}' on GPU ${GPU_INDEX} ==="
tmux new-session -d -s "$TMUX_SESSION" "bash ${INNER}"
sleep 5
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session died within 5 s — tail of ${LOG}:"; tail -20 "$LOG" 2>/dev/null; exit 1
fi
if ! grep -q "=== ${RUN} START" "$LOG" 2>/dev/null; then
  echo "no START line in ${LOG} after 5 s — attach with: tmux attach -t ${TMUX_SESSION}"; exit 1
fi

# --- bookkeeping: digest (idempotent, inserted under `runs:`) + one breadcrumb line ---
if grep -q "^  ${RUN}:" "$DIGEST"; then
  echo "digest: ${RUN} already listed in ${DIGEST} — left as is"
else
  awk -v line="$DIGEST_LINE" '{print} /^runs:/ && !done {print line; done=1}' "$DIGEST" > "${DIGEST}.tmp" && mv "${DIGEST}.tmp" "$DIGEST"
  echo "digest: appended '${DIGEST_LINE}' to ${DIGEST}"
fi
grep -q "^## Launches" "$BREADCRUMBS" || printf '\n## Launches (auto-appended by scripts/launch_exp003_2.sh)\n' >> "$BREADCRUMBS"
printf -- '- launched %s, lr 1e-5→2e-5, GPU %s, %s\n' "$RUN" "$GPU_INDEX" "$(date -Is)" >> "$BREADCRUMBS"

# --- proof of life: a process on the card ---
waited=0
while [ "$waited" -lt "$GPU_WAIT_S" ]; do
  if [ -n "$(nvidia-smi --id="$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "launched — check tomorrow's digest"
    exit 0
  fi
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session died during startup — tail of ${LOG}:"; tail -20 "$LOG"; exit 1
  fi
  sleep 10; waited=$((waited + 10))
done
echo "WARNING: no process on GPU ${GPU_INDEX} after ${GPU_WAIT_S} s — still starting or stuck; attach with: tmux attach -t ${TMUX_SESSION}"
exit 2
