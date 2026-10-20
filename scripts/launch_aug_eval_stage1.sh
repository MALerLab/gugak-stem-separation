#!/usr/bin/env bash
# launch_aug_eval_stage1.sh — augmentation listening test, stage 1
# (docs/aug_listening_test_spec.md): test-split eval of exp007 ep34, master-aligned on
# cuda:0 and Σstem on cuda:1, run in parallel. exp006 ep34's matching evals already exist.
#
# Run inside tmux:
#   tmux new -s aug_eval_stage1 'bash scripts/launch_aug_eval_stage1.sh'
# To stop early: kill -INT the `timeout` pids (Ctrl-C in tmux does not reach them).
set -euo pipefail
cd "$(dirname "$0")/.."

# wandb / env settings, same as every launch script
set -a; source .env; set +a

LOG_DIR=experiments/exp007_260925_bsroformer_room_distance/eval
mkdir -p "$LOG_DIR"

# --signal=INT lets the runner exit cleanly; --kill-after only fires if it hangs after INT.
# no_thp.py disables transparent huge pages (sym8 memory-stall workaround).
for variant in master_aligned sumstem; do
  timeout --signal=INT --kill-after=5m 3h \
    python3 scripts/no_thp.py uv run python -m src.eval.runner \
      --config "configs/eval/exp007_ep34_test_${variant}.yaml" \
      > "$LOG_DIR/exp007_ep34_test_${variant}.log" 2>&1 &
done
wait
echo "stage 1 done"
