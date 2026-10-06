"""bench_recording_aug.py — per-item CPU cost of each recording-condition arm.

Report §3.5 sets the budget: at 24 cores and roughly 5x loader headroom, an arm that pushes
the per-item cost much past ~100 ms starts eating that headroom, and a naive nine-stem
convolution (measured there at 16–20 ms per stereo source, so 145–180 ms per item) does not
fit. This script MEASURES; it asserts nothing. Two numbers per arm:

  * single-process per-item latency, p50/p95 — the honest cost of the stage itself, with
    no worker parallelism hiding it. The delta against the baseline arm is the stage cost.
  * end-to-end DataLoader throughput at the real worker count — what the trainer actually
    sees, which is the number that decides whether the GPU ever waits.

It deliberately reuses the EXPERIMENT CONFIG rather than a synthetic one, so I/O, manifest
filtering and the coherent-cluster draw are all in the measurement.

Run:
    uv run python scripts/bench_recording_aug.py --items 200 --workers 8
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402

# One entry per arm of the experiment matrix report §1 asks for, plus the combinations the
# defaults actually compose into. `block` patches the gugak_mix block; `recording` is the
# recording_aug sub-block (None = the feature stays absent, i.e. the pre-change path).
ARMS = {
    "baseline (exp006, feature absent)": dict(block={}, recording=None),
    "recording_aug enabled, all p=0": dict(block={}, recording={}),
    "EQ stem+mixbus p=.5": dict(block=dict(eq_stem_prob=0.5, eq_mixbus_prob=0.5),
                                recording=None),
    "device response p=1": dict(block={}, recording=dict(device_response=dict(prob=1.0))),
    "room per_mixture p=1": dict(block={}, recording=dict(rir=dict(prob=1.0))),
    "room per_stem p=1 (ablation)": dict(block={}, recording=dict(
        rir=dict(prob=1.0, scope="per_stem", position_scope="per_stem"))),
    "automation shared p=1": dict(block={}, recording=dict(
        level_automation=dict(mixbus_prob=1.0))),
    "LimitAug p=1": dict(block={}, recording=dict(
        limit_aug=dict(prob=1.0, loudness_mode="sampled", normalize_after_bus=False))),
    "stem EQ p=.5 (mic placement proxy)": dict(block=dict(eq_stem_prob=0.5),
                                               recording=None),
    "exp007 arm (EQ + room + device, p=.5)": dict(block=dict(eq_stem_prob=0.5),
                                                  recording=dict(
        rir=dict(prob=0.5), device_response=dict(prob=0.5))),
    "everything on (worst case)": dict(block=dict(eq_stem_prob=0.5, eq_mixbus_prob=0.5),
                                       recording=dict(
        device_response=dict(prob=1.0), rir=dict(prob=1.0),
        level_automation=dict(mixbus_prob=1.0, stem_prob=0.3),
        limit_aug=dict(prob=1.0, loudness_mode="sampled", normalize_after_bus=False))),
}


def build(base_block: dict, spec: dict, num_items: int) -> GugakMixDataset:
    block = copy.deepcopy(base_block)
    block.update(spec["block"])
    if spec["recording"] is not None:
        block["recording_aug"] = {"enable": True, **spec["recording"]}
    return GugakMixDataset(MixDatasetConfig.from_mapping(block), REPO_ROOT, num_items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the recording-condition arms.")
    ap.add_argument("--config", default="configs/exp006_bsroformer_partial_coherent.yaml")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers for the end-to-end pass (0 = skip it)")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as handle:
        base_block = yaml.load(handle, Loader=yaml.FullLoader)["gugak_mix"]

    names = args.arms or list(ARMS)
    total = args.items + args.warmup
    baseline_p50 = None
    print(f"{args.items} items after {args.warmup} warmup · "
          f"{base_block['segment_seconds']} s @ {base_block['sample_rate']} Hz · "
          f"single process, then {args.workers} workers\n")
    header = (f"{'arm':<40s} {'p50':>8s} {'p95':>8s} {'max':>8s} {'Δp50':>8s} "
              f"{'items/s':>9s} {'L/R r':>7s}")
    print(header)
    print("-" * len(header))

    for name in names:
        dataset = build(base_block, ARMS[name], total)
        latencies, correlations = [], []
        for index in range(total):
            started = time.perf_counter()
            _, mixture = dataset[index]
            latencies.append((time.perf_counter() - started) * 1e3)
            # stereo width of the MIXTURE. The stems are close-miked and near-mono
            # (measured per-class L/R correlation .96–.99 for seven of nine classes), so
            # an arm that claims to model a distant stereo capture and leaves this where
            # the dry baseline has it is not doing anything stereophonically.
            mixture = mixture.numpy().astype(np.float64)
            denom = np.sqrt((mixture[0] ** 2).sum() * (mixture[1] ** 2).sum())
            if denom > 0:
                correlations.append(float((mixture[0] * mixture[1]).sum() / denom))
        latencies = np.array(latencies[args.warmup:])
        p50, p95 = float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))
        if baseline_p50 is None:
            baseline_p50 = p50

        throughput = float("nan")
        if args.workers:
            loader = torch.utils.data.DataLoader(
                build(base_block, ARMS[name], args.items), batch_size=1, shuffle=False,
                num_workers=args.workers, persistent_workers=False)
            started = time.perf_counter()
            for _ in loader:
                pass
            throughput = args.items / (time.perf_counter() - started)

        print(f"{name:<40s} {p50:7.1f}ms {p95:7.1f}ms {latencies.max():7.1f}ms "
              f"{p50 - baseline_p50:+7.1f}ms {throughput:8.1f} "
              f"{np.median(correlations):7.3f}")


if __name__ == "__main__":
    main()
