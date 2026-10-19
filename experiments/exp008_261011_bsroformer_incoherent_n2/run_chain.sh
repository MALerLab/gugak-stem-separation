#!/usr/bin/env bash
# run_chain.sh — exp008 (fresh, cap 35) -> ep34 snapshot -> exp008.1 (continue to 60).
#
#   Start it in its own tmux session, never a bare shell:
#     tmux new -d -s exp008 'bash experiments/exp008_261011_bsroformer_incoherent_n2/run_chain.sh'
#
# exp008.1 runs ONLY if exp008's train.log carries "=== exp008 finished clean" — the same
# marker exp006.1's continue.sh demands of exp006. Anything else (SIGINT, crash, a resumed
# run) stops the chain here for a human decision. To stop the run: Ctrl-C the pane (SIGINT)
# and wait — never kill -9 / tmux kill-session.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation-CLAUDE-exp008

EXP_DIR=experiments/exp008_261011_bsroformer_incoherent_n2
RESULTS=$EXP_DIR/checkpoints
CHAIN_LOG=$EXP_DIR/chain.log
LAST_CKPT=$RESULTS/last_bs_roformer.ckpt
SNAPSHOT=$RESULTS/last_bs_roformer_ep34.ckpt

log() { echo "=== chain $(date -Is) $1" | tee -a "$CHAIN_LOG"; }

# step 1: exp008, fresh, cap 35
log "running exp008 launch.sh"
bash "$EXP_DIR/launch.sh"
log "exp008 launch.sh exited $?"

if ! grep -q "=== exp008 finished clean" "$EXP_DIR/train.log" 2>/dev/null; then
  log "exp008 did NOT finish clean — not continuing to exp008.1. Stopping for Jae."
  exit 1
fi

# step 2: snapshot the ep34 state before exp008.1 overwrites `last`
if [ -e "$SNAPSHOT" ]; then
  log "REFUSING: $SNAPSHOT already exists — not overwriting it. Stopping for Jae."
  exit 1
fi
cp -p "$LAST_CKPT" "$SNAPSHOT"
if cmp -s "$LAST_CKPT" "$SNAPSHOT"; then
  log "snapshot ok: $SNAPSHOT ($(du -h "$SNAPSHOT" | cut -f1)), byte-identical to last"
else
  log "snapshot MISMATCH between $LAST_CKPT and $SNAPSHOT — not continuing. Stopping for Jae."
  exit 1
fi

# step 3: give the CUDA contexts a moment to release before continue.sh's idle-card guard
sleep 60

log "running exp008.1 continue.sh"
bash experiments/exp008.1_261011_continued_to_60/continue.sh
log "exp008.1 continue.sh exited $?"
