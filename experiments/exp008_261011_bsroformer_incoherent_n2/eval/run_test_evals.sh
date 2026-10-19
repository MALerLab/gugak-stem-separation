#!/usr/bin/env bash
# exp008 ep34 TEST evals (Σstem, then lag-aligned master) on GPU 2 — LBD paper v2 numbers.
# Device is fixed in each config (cuda:2). Each eval is capped at 3 h inside the script:
# timeout sends SIGINT first, SIGKILL only 5 min later if it is wedged.
set -euo pipefail
cd /home/jae.gye/userdata/repos/gugak_stem_separation-CLAUDE-exp008
EVAL_DIR=experiments/exp008_261011_bsroformer_incoherent_n2/eval

# Pre-flight: GPU 2 must have no compute processes
GPU2_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$GPU2_UUID"; then
  echo "GPU 2 busy — refusing to start"; exit 1
fi

for variant in sumstem master_aligned; do
  echo "=== eval exp008_ep34_test_${variant} START $(date -Iseconds) ==="
  # --frozen: never re-lock or sync the env; no_thp_exec: THP off for the job (sym8 memory stalls)
  timeout --signal=INT --kill-after=5m 3h \
    uv run --frozen python3 scripts/no_thp_exec.py python -m src.eval.runner \
      --config configs/eval/exp008_ep34_test_${variant}.yaml \
    2>&1 | tee "${EVAL_DIR}/exp008_ep34_test_${variant}.log"
  echo "=== eval exp008_ep34_test_${variant} END $(date -Iseconds) exit ${PIPESTATUS[0]} ==="
done
