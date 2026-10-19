#!/usr/bin/env bash
# launch.sh — exp007: BS-RoFormer, partially coherent mixing (p = 0.5) PLUS recording-condition
#             augmentation. Fresh, ONE GPU (card 2), torch.compile ON, 35-epoch cap.
#
# WHAT THIS ARM ASKS. exp006 asks whether partial coherence helps. exp007 holds coherence fixed
# at the same p = 0.5 and adds the capture-path augmentation, so exp007 vs exp006 isolates the
# augmentation. Both are compared to exp003.0 at matching epochs.
#
# WHY THE AUGMENTATION IS COMBINED, AND WHAT THAT COSTS. The arm turns on room + device response
# + stem EQ together. That is deliberate and it is a real limitation: a win cannot be attributed
# to any one of them. The reason is a measurement — the 71955 stems have L/R correlation medians
# of .96-.99 for seven of nine classes, i.e. they are dry, close-miked captures with almost no
# diffuse field. So this corpus's invariant acoustic fingerprint is the MICROPHONE, not the room,
# and convolution being multiplicative in frequency, a room cannot make a fixed spectral
# colouration vary — only random EQ can. One GPU buys one answer, so this arm buys the one most
# likely to move the number. Room-only and EQ-only are the attribution follow-ups.
#
# THE ROOM IS SHARED WITHIN A MIXTURE (scope per_mixture, position_scope shared) and varies
# BETWEEN mixtures. That is load-bearing, not a default: giving each stem its own room would hand
# the model an acoustic separation cue that does not exist in real audio, where every instrument
# sits in one space. What this corpus lacks is inter-RECORDING acoustic variance — MUSDB gets it
# free from many studios; NIA 71955 has one. We add that, and only that.
#
# Targets are reverberant source images: mixture = sum_i (h * s_i), target_i = h * s_i, so the
# sum invariant is exact and the task stays separation, not separation-plus-dereverberation.
#
# ⚠️ THE COLLAPSE IS THE MEASUREMENT. If heads collapse: no stop, no resume, no hand-set lr, no
# scheduler edits. Only a genuine fault — non-finite steps, crash, disk or CUDA error —
# justifies stopping. Quality going down is not a fault.
#
# ⚠️ CARD 2 ONLY. exp006 owns GPUs 0 and 1 for about another two days. Never widen
# CUDA_VISIBLE_DEVICES while that is running.
set -u
cd /home/jae.gye/userdata/repos/gugak-stemsep-aug

RUN=exp007
EXP_DIR=experiments/exp007_260925_bsroformer_room_distance
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
START_CKPT=$RESULTS/start_checkpoint.ckpt

# wandb target lives in .env — nothing loads it automatically. WANDB_PROJECT does NOT win:
# MSST passes project='msst' as an explicit wandb.init() kwarg. Fix it at sync time with
#   wandb sync -p gugak_stem_separation wandb/offline-run-*
set -a; source .env; set +a

export PYTHONPATH=/home/jae.gye/userdata/repos/gugak-stemsep-aug
# HARD-CODED, no ${VAR:-default}. A `:-` fallback honours whatever the invoking shell already
# exported, and an inherited "0,1" would put this run on the two cards exp006 is using — it
# would either OOM instantly against exp006's 75 GiB or take exp006 down with it, and the
# crash handler below would then auto-resume once into the same mistake.
export CUDA_VISIBLE_DEVICES=2
# one BLAS thread per dataloader worker. The augmented loader measures 74 ms p50 per item and
# the trainer asks for 7.6 items/s, so under one core of real work — threads here only thrash
# cache, and exp006's workers share this 24-core cpuset.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULTS"

# provenance: repo commit + dirty list + the pinned MSST submodule commit and branch
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git -c filter.nbstripout.clean=cat status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse --abbrev-ref HEAD >> $EXP_DIR/git_commit.txt

# --- pre-flight. It REFUSES rather than starting something wrong; there is no force flag.
# If a guard is wrong, fix the world, not the guard.
fails=0
guard_fail() { echo "  ✗ $1" | tee -a "$LOG"; fails=$((fails + 1)); }
guard_pass() { echo "  ✓ $1"; }
echo "=== ${RUN} pre-flight $(date -Is) ===" | tee -a "$LOG"

# the card must be free — and a ghost CUDA context (memory held with no process, the residue
# of a kill -9) is a refusal too, not a free card
# NOTE: nvidia-smi does NOT honour CUDA_VISIBLE_DEVICES — it enumerates physical cards. So
# --id must be the PHYSICAL index, which is what CUDA_VISIBLE_DEVICES holds here. Querying
# --id=0 would inspect the wrong card entirely, and card 0 is exp006's.
gpu_pids=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader | tr -d ' \n')
gpu_mem=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ -n "$gpu_pids" ]; then guard_fail "card ${CUDA_VISIBLE_DEVICES} busy — pids ${gpu_pids}, ${gpu_mem} MiB"
elif [ "${gpu_mem:-0}" -gt 1024 ]; then guard_fail "card ${CUDA_VISIBLE_DEVICES} shows ${gpu_mem} MiB with NO process — ghost context"
else guard_pass "card ${CUDA_VISIBLE_DEVICES} idle (${gpu_mem} MiB)"; fi

# never launch twice. tmux rewrites '.' to '_' in session names and parses '.' in a -t target
# as a window separator, so dotted names false-negative — exp007 has no dot, but keep the habit
# Two INDEPENDENT checks. They were a single if/elif chain and that was a bug twice over:
# the chain skipped the log check whenever the session check passed, and the session check
# is self-referential — this script normally runs INSIDE `tmux new -d -s exp007`, so the
# session it looks for is its own and it refused every launch. exp003.2 dodged that by
# putting its guards in an OUTER wrapper that runs before the session exists; the guards are
# inline here, so the check exempts its own session explicitly.
cur_session=$(tmux display-message -p '#S' 2>/dev/null || true)
if [ -n "$cur_session" ] && [ "$cur_session" = "$RUN" ]; then
  guard_pass "inside tmux session '${RUN}' (this launcher's own)"
elif tmux has-session -t "$RUN" 2>/dev/null; then
  guard_fail "tmux session '${RUN}' already exists and is not ours"
else
  guard_pass "no foreign tmux session '${RUN}'"
fi

# the real double-launch guard: a START line means this run already began once
if [ -f "$LOG" ] && grep -q "=== ${RUN} START" "$LOG"; then
  guard_fail "${LOG} already carries a START line: $(grep "=== ${RUN} START" "$LOG" | head -1 | cut -c1-80)"
else
  guard_pass "no prior ${RUN} START line"
fi

if [ ! -f "$START_CKPT" ]; then
  guard_fail "missing $START_CKPT — exp007 REUSES exp003.0's seed-42 start checkpoint (byte copy)"
elif sha256sum -c "$EXP_DIR/start_checkpoint_sha256.txt" >/dev/null 2>&1; then
  guard_pass "start_checkpoint.ckpt sha256 matches exp003.0's seed-42 head init"
else
  guard_fail "start_checkpoint.ckpt sha256 MISMATCH vs $EXP_DIR/start_checkpoint_sha256.txt"
fi

[ -f .env ] && guard_pass ".env present" || guard_fail ".env missing at repo root"

if [ "$fails" -gt 0 ]; then
  echo "REFUSING to launch ${RUN} — ${fails} guard(s) failed. Nothing started." | tee -a "$LOG"
  exit 1
fi

COMMON_FLAGS=(
  --model_type bs_roformer
  --config_path configs/exp007_bsroformer_room_distance.yaml
  --results_path "$RESULTS"
  --data_path dummy                 # unused: training data comes from our custom_dataset
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val   # Σstem variant, 91 songs
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0          # index INTO CUDA_VISIBLE_DEVICES, i.e. physical card 2
  --use_standard_loss               # NOT optional — it gates the loss-selection branch
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline
)

# WHY train_ddp.py ON ONE CARD — this looks wrong and is not. With one visible GPU,
# train_ddp.py runs a one-rank group (world_size = torch.cuda.device_count() = 1) and the
# geometry is untouched: utils/dataset.py builds the trainset at batch x world_size = 4, so
# still 80,000 items and 2,500 optimizer steps per epoch.
#
# The reason is torch.compile. On the PLAIN train.py path with a single card,
# initialize_model_and_device returns a bare module (no DataParallel), so torch.compile wraps
# it into an OptimizedModule that has no `.module`, and save_weights then takes
# `model.state_dict()` — writing every key with an `_orig_mod.` prefix. Two things follow, and
# the second is the dangerous one:
#   * src/render_audio.py and scripts/cross_source_leakage.py do strict loads and would reject
#     the checkpoints outright;
#   * on crash-resume, load_start_checkpoint runs BEFORE compile, so a plain-keyed model meets
#     a prefixed checkpoint, and run_resume's tolerant loader copies only keys it recognises —
#     which is NONE. Verified by running the repo's own loader: "weights changed by load?
#     False". Optimizer, scheduler and epoch all restore fine, so the run would continue from a
#     RANDOMLY INITIALISED model with nothing in the log saying so, under a policy that tells
#     the operator not to intervene when quality drops.
# Under DDP, dist is initialized and save_weights takes `model.module.state_dict()`, which
# forwards through OptimizedModule to the real module and yields clean keys — exp006's
# checkpoints prove it (2595 keys, no prefix). It also puts exp007 on byte-identical trainer
# and eval code paths to exp006, which only helps the comparison.
run_fresh() {
  # FRESH: strict load of the plain state_dict. NO --load_* flags. Passing --load_optimizer here
  # would silently restore a HALVED learning rate — this project has been bitten by that twice.
  uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$START_CKPT" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # CRASH RECOVERY ONLY — restores optimizer/scheduler/epoch.
  uv run python external/msst/train_ddp.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_bs_roformer.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== ${RUN} START $(date -Is) fresh from $(basename "$START_CKPT") (bf16, lr 1e-5, seed 42, batch 4x8 x 1 rank, compile on, room+device+EQ, cap 35) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== ${RUN} finished clean $(date -Is) ===" | tee -a "$LOG"; exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== ${RUN} stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"; exit 0
fi

echo "=== ${RUN} CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_bs_roformer.ckpt" ]; then run_resume; else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"; run_fresh
fi
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== ${RUN} ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"; exit 0
fi

echo "=== ${RUN} CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"

# ---------------------------------------------------------------------------------------
# CONTINUING TO 60, if ep35 justifies it: edit training.num_epochs 35 -> 60 in the config and
# relaunch through run_resume (it passes --load_epoch, so training picks up at 35). Never prune
# last_bs_roformer.ckpt. And note MSST writes the PREVIOUS best metric into the file it is
# currently saving — trust the FILENAME for an epoch's own score.
