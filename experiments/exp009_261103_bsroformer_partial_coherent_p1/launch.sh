#!/usr/bin/env bash
# launch.sh — exp009: BS-RoFormer on PARTIALLY COHERENT mixtures (p = 1.0), n ~ U{2..9}.
#             Fresh, TWO GPUs (0 and 1) under DDP, torch.compile ON, 35-epoch cap.
#
#   Start it in its own tmux session, never a bare shell:
#     tmux new -d -s exp009 'bash experiments/exp009_261103_bsroformer_partial_coherent_p1/launch.sh'
#   To stop: Ctrl-C the pane (SIGINT) and wait — never kill -9 / tmux kill-session.
#
# WHY. Identical to exp006's and exp008's launchers and configs in every respect except
# gugak_mix.coherent_mix_prob -> 1.0 (see the config header). Adapted from
# experiments/exp008_261011_bsroformer_incoherent_n2/launch.sh, with one launch-only change:
# the trainer runs under scripts/no_thp_exec.py from the FIRST attempt, not only on resume.
# exp008 crashed twice on NCCL watchdog timeouts during validation (2026-10-12) — one rank
# stalled on THP compaction — and resume.sh/continue.sh fixed it with this wrapper. No
# chain into a continuation: exp009 stops at the 35-epoch cap.
#
# CARD ETIQUETTE. GPUs 0 and 1, hard-coded. GPU 2 is reserved for other jobs — never widen
# CUDA_VISIBLE_DEVICES.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. If heads collapse: no stop, no resume, no hand-set
# lr, no scheduler edits. Only a genuine fault — non-finite steps, crash, disk or CUDA
# error — justifies stopping. Quality going down is not a fault.
set -u
cd /home/jae.gye/userdata/repos/gugak-exp009

RUN=exp009
SESSION=exp009
EXP_DIR=experiments/exp009_261103_bsroformer_partial_coherent_p1
CONFIG=configs/exp009_bsroformer_partial_coherent_p1.yaml
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt
MSST_COMMIT=31110ff54b7af035a096c17289d41dc722cea9d4   # = exp006's / exp008's teo/ddp-fixes commit

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-exp009
export CUDA_VISIBLE_DEVICES=0,1   # hard-coded: a `:-` fallback would honour an inherited
                                  # value and could land this on GPU 2
# one BLAS thread per dataloader worker: 2 ranks x 6 workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULTS"

# --- pre-flight guards: refuse to start if any fails ---
fails=0
guard_fail() { echo "  ✗ $1" | tee -a "$LOG"; fails=$((fails + 1)); }
guard_pass() { echo "  ✓ $1"; }
echo "=== ${RUN} pre-flight $(date -Is) ===" | tee -a "$LOG"

# both cards idle; memory with no process = ghost context, use neither
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

if [ -f "$LOG" ] && grep -q "=== ${RUN} START" "$LOG"; then
  guard_fail "${LOG} already carries a START line"
else guard_pass "no prior ${RUN} START line"; fi

if [ ! -f "$START_CKPT" ]; then
  guard_fail "missing $START_CKPT — byte-copy exp003.0's: cp -p /home/jae.gye/storage/gugak-stemsep-experiments/exp003.0_260809_bsroformer_pilot/checkpoints/start_checkpoint.ckpt $START_CKPT"
elif sha256sum --status -c "$EXP_DIR/start_checkpoint_sha256.txt"; then
  guard_pass "start checkpoint sha256 matches"
else guard_fail "start checkpoint sha256 MISMATCH (see $EXP_DIR/start_checkpoint_sha256.txt)"; fi

[ -f .env ] && guard_pass ".env present" || guard_fail ".env missing"
[ -d data/gugak_ensemble_71955/sumstem_9stem/val ] && guard_pass "Σstem val tree present" \
  || guard_fail "data/gugak_ensemble_71955/sumstem_9stem/val missing (worktree data symlink?)"

msst_head=$(git -C external/msst rev-parse HEAD 2>/dev/null)
[ "$msst_head" = "$MSST_COMMIT" ] && guard_pass "external/msst at ${MSST_COMMIT:0:7}" \
  || guard_fail "external/msst at ${msst_head:-?}, expected ${MSST_COMMIT} (DDP fixes)"

if uv run python -c "import sys, yaml; yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)" "$CONFIG" 2>/dev/null; then
  guard_pass "config parses ($CONFIG)"
else guard_fail "config does not parse ($CONFIG)"; fi

if [ "$fails" -gt 0 ]; then
  echo "REFUSING to start ${RUN} — ${fails} guard(s) failed. Nothing started." | tee -a "$LOG"
  exit 1
fi

# wandb target (entity/project/ignore-globs) lives in .env — nothing loads it
# automatically. WANDB_PROJECT does NOT win (MSST hardcodes project='msst'); fix at sync:
#   wandb sync -p gugak_stem_separation wandb/offline-run-*
set -a; source .env; set +a

# provenance: repo commit + dirty-file list + the pinned MSST submodule commit and branch
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git -c filter.nbstripout.clean=cat status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse --abbrev-ref HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst status --short >> $EXP_DIR/git_commit.txt

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path "$CONFIG"
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 6 --pin_memory --persistent_workers          # per RANK
  --seed 42 --device_ids 0 1
  --use_standard_loss               # NOT optional — it gates the loss-selection branch
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline                   # streaming only; the local transaction log is written
                                    # either way, so `wandb sync` uploads it later
)

run_fresh() {
  # FRESH: strict load of the plain state_dict the init script wrote. NO --load_* flags.
  # Passing --load_optimizer here would silently restore a HALVED learning rate — this
  # project has been bitten by exactly that twice. The declared lr must be the lr trained.
  # THP disabled for the trainer and everything it spawns (see header).
  python3 scripts/no_thp_exec.py uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$START_CKPT" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # CRASH RECOVERY ONLY — restores optimizer/scheduler/epoch so the run continues where it
  # died rather than restarting the experiment.
  python3 scripts/no_thp_exec.py uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_bs_roformer.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-5, seed 42, batch 4x4 x 2 ranks, compile on, cap 35, p 1.0, THP off) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== ${RUN} finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== ${RUN} stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== ${RUN} CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_bs_roformer.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== ${RUN} ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== ${RUN} CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"

# Note while reading checkpoints: MSST writes the PREVIOUS best metric into the file it is
# currently saving. Trust the FILENAME for the epoch's own score.
