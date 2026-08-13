"""smoke_mix_dataset.py — sanity-check + render the mix dataset before any GPU time.

Builds GugakMixDataset from an experiment config, pulls items through a real
multi-worker DataLoader, and reports the three things that must be true:
  1. the drawn class-count distribution matches the density histogram it targets
  2. mixture ≡ Σ(targets) holds exactly (the training identity)
  3. loudness/peak land where the normalization says they should
Also renders the first few items (mixture + active stems) to disk for listening,
and times throughput — the likeliest shakedown bottleneck.

Run:
    uv run python scripts/smoke_mix_dataset.py
      --config configs/exp001_htdemucs_9stem.yaml
      --items 64        items to pull for statistics
      --render 4        items written to disk as wavs
      --workers 4       DataLoader workers (exercises multiprocessing)
      --out data/mix_smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the incoherent-mix dataset.")
    ap.add_argument("--config", default="configs/exp001_htdemucs_9stem.yaml")
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--render", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/mix_smoke")
    args = ap.parse_args()

    # FullLoader, not safe_load: roformer configs carry !!python/tuple values, and this
    # is the loader MSST itself uses (utils/settings.load_config)
    with open(REPO_ROOT / args.config, encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=yaml.FullLoader)
    cfg = MixDatasetConfig.from_mapping(raw["gugak_mix"])
    dataset = GugakMixDataset(cfg, REPO_ROOT, num_items=args.items)
    print(f"pool: " + " · ".join(f"{c}:{len(dataset.pool[c])}" for c in cfg.classes))
    print("density n:", dataset.density_values.tolist())
    print("density p:", np.round(dataset.density_probs, 3).tolist())

    # pull every item through a real DataLoader (multiprocessing exercised)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=args.workers)
    meter = pyloudnorm.Meter(cfg.sample_rate)
    drawn_counts, loudness_values, peaks, sum_errors = [], [], [], []
    t0 = time.time()
    items = []
    for stems, mixture in loader:
        stems, mixture = stems[0].numpy(), mixture[0].numpy()
        active = (np.abs(stems).sum(axis=(1, 2)) > 0)
        drawn_counts.append(int(active.sum()))
        loudness = meter.integrated_loudness(mixture.T)
        if not np.isinf(loudness):
            loudness_values.append(loudness)
        peaks.append(float(np.abs(mixture).max()))
        sum_errors.append(float(np.abs(mixture - stems.sum(axis=0)).max()))
        if len(items) < args.render:
            items.append((stems, mixture, active))
    elapsed = time.time() - t0

    # --- report ---
    values, counts = np.unique(drawn_counts, return_counts=True)
    print(f"\ndrawn n over {args.items} items (target p in brackets):")
    target = dict(zip(dataset.density_values.tolist(), dataset.density_probs))
    for v, c in zip(values, counts):
        print(f"  n={v}: {c/len(drawn_counts):.3f}  [{target.get(v, 0):.3f}]")
    print(f"\nmixture LUFS: median {np.median(loudness_values):+.1f} "
          f"(target {cfg.target_lufs:+.1f}) · range "
          f"[{min(loudness_values):+.1f}, {max(loudness_values):+.1f}]")
    print(f"mixture peak: max {max(peaks):.3f} (ceiling {cfg.peak_ceiling})")
    print(f"mixture ≡ Σ(targets): max |error| = {max(sum_errors):.2e}")
    print(f"throughput: {args.items / elapsed:.1f} items/s "
          f"({args.workers} workers, {elapsed:.1f}s total)")

    # --- render for listening ---
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (stems, mixture, active) in enumerate(items):
        soundfile.write(out_dir / f"item{i:02d}_mix.wav", mixture.T, cfg.sample_rate)
        for slot in np.flatnonzero(active):
            soundfile.write(out_dir / f"item{i:02d}_{cfg.classes[slot]}.wav",
                            stems[slot].T, cfg.sample_rate)
    print(f"\nrendered {len(items)} items -> {out_dir}")


if __name__ == "__main__":
    main()
