#!/usr/bin/env bash
# launch_demo_v3_renders.sh — demo v3 renders through exp006.1 ep59, all on GPU 2, back to back:
#   1. the four real-world clips   (configs/render/demo_v3_exp006.1_ep59.yaml, ~minutes)
#   2. test split, Σstem input     (configs/eval/exp006.1_ep59_test_sumstem_renders.yaml, ~45 min)
#   3. test split, master input    (configs/eval/exp006.1_ep59_test_master_renders.yaml, ~45 min)
# No wandb, so no .env. Each job carries its own time cap (SIGINT first, SIGKILL 5 min later)
# and runs through scripts/no_thp.py (2026-10-13: without it the Σstem job crawled at ~3× the
# 2026-10-11 eval pace, GPU idle, memory pressure ~19% — the sym8 THP-compaction stall).
#
# Stopping early: Ctrl-C in the pane does NOT reach the job — `timeout` runs it in its own
# process group. Send SIGINT to timeout instead (it forwards): kill -INT <timeout pid>.
#
# Run inside tmux, from the repo root:
#   bash experiments/exp006.1_260929_continued_to_60/launch_demo_v3_renders.sh
set -euo pipefail                 # -e stop on first failure · -u unset vars are errors · -o pipefail keep tee'd exit codes

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LOG_DIR=experiments/exp006.1_260929_continued_to_60/eval_renders
mkdir -p "$LOG_DIR"

# 1. real-world clips — render_demo takes --device cuda, so pin the visible card to GPU 2
CUDA_VISIBLE_DEVICES=2 timeout --signal=INT --kill-after=5m 30m \
  python3 scripts/no_thp.py uv run python -m src.render_demo \
  --render-config configs/render/demo_v3_exp006.1_ep59.yaml --device cuda \
  2>&1 | tee "$LOG_DIR/demo_v3_real_world.log"

# 2–3. test-split renders — the eval configs name cuda:2 themselves
for variant in sumstem master; do
  timeout --signal=INT --kill-after=5m 2h \
    python3 scripts/no_thp.py uv run python -m src.eval.runner \
    --config "configs/eval/exp006.1_ep59_test_${variant}_renders.yaml" \
    2>&1 | tee "$LOG_DIR/demo_v3_test_${variant}.log"
done
echo "demo v3 renders done: $(date '+%F %T')"
