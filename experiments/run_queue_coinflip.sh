#!/usr/bin/env bash
# run_queue_coinflip.sh — exp010 → exp011 → exp012 back to back on GPU 2 (LBD Q&A mixer test).
#
# Each launcher exits 0 for BOTH a clean finish and an intentional Ctrl-C, so the exit code
# can't gate the queue. The gate is the launcher's own log line instead: the next run starts
# only if the previous one logged "finished clean" (or finished after its one crash-resume).
# A Ctrl-C, a double crash, or anything else stops the queue.
#
# Run:  tmux new -s lbd-qna 'bash experiments/run_queue_coinflip.sh'
set -uo pipefail
cd /home/jae.gye/userdata/repos/gugak-lbd-qna

for EXP in exp010_261103_htdemucs_coinflip_r05 \
           exp011_261103_htdemucs_coinflip_r025 \
           exp012_261103_htdemucs_coinflip_r075; do
  echo "=== queue: starting $EXP $(date -Is) ==="
  bash "experiments/$EXP/launch.sh"
  if ! grep "^=== " "experiments/$EXP/train.log" | tail -1 \
       | grep -q -e "finished clean" -e "ended after resume (exit 0)"; then
    echo "=== queue: $EXP did not finish clean — queue stopped $(date -Is) ==="
    exit 1
  fi
done
echo "=== queue: all three done $(date -Is) ==="
