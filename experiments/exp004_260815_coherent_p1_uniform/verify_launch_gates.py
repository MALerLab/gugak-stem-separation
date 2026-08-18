"""verify_launch_gates.py — sampler gates G1 / G5 (+ G4, manifest correctness) on the
EXACT exp004 launch config, not the verifier's (p × mode) grid.

The build report's gates ran on exp002's gugak_mix block with p / mode overridden per
grid cell. This is the sampler's first training use, so the gates are re-run here on
`configs/exp004_htdemucs_coherent_p1_uniform.yaml` as written: no overrides, the
config's own p=1.0 / uniform / cluster_min_melodic=2 / density_uniform_min=2 / seed 45.
It reuses the verifier's own cell runner so the numbers are the same code path.

  G1  per-class exposure flat (< 2% max deviation)      — 200,000 plan draws
  G5  zero clusters below cluster_min_melodic           — direct per-cluster assertion
  +   manifest ground truth: excluded files never drawn, cluster stems share one song
      and one offset, nothing quarantined / mis-split
  G4  mixture ≡ Σ(targets) ≤ 1e-6                        — 300 decoded items, real DataLoader
  M   H_melodic / shortfall / n support / anchor genres — recorded for the launch report

Run from the repo root:
    uv run python experiments/exp004_260815_coherent_p1_uniform/verify_launch_gates.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from verify_coherent_sampler import (  # noqa: E402
    REASON_BITS, aggregate_cells, evaluate_g1, run_audio_cell, run_grid_cell)

CONFIG = "configs/exp004_htdemucs_coherent_p1_uniform.yaml"
OUT_DIR = REPO_ROOT / "experiments/exp004_260815_coherent_p1_uniform"
GRID_DRAWS = 200_000
AUDIO_ITEMS = 300
WORKERS = 10
SILENCE_EPS = 1e-6


def main() -> None:
    raw = yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))
    base = dict(raw["gugak_mix"])
    n_classes = len(base["classes"])
    # overrides ECHO the config (needed only so the cell labels itself); values identical
    echo = {"coherent_mix_prob": base["coherent_mix_prob"],
            "anchor_selection": base["anchor_selection"]}
    cell = {"label": "exp004_launch_config", "base": base, "grid_draws": GRID_DRAWS,
            "exclusions_path": "configs/manifest_exclusions.yaml", "overrides": echo}
    print(f"config {CONFIG}: p={base['coherent_mix_prob']} mode={base['anchor_selection']} "
          f"n_min={base['density_uniform_min']} min_melodic={base['cluster_min_melodic']} "
          f"seed={base['seed']}")

    result = run_grid_cell(cell)
    frames = aggregate_cells([result])
    failures = []

    # --- correctness vs manifest (incl. excluded_file, quarantined, wrong_split) ---
    print("\nCORRECTNESS vs manifest")
    for name, count in sorted(result["violations"].items()):
        print(f"  [{'PASS' if count == 0 else 'FAIL'}] {name}: {count}")
        if count:
            failures.append(name)

    # --- G1 ---
    g1 = evaluate_g1(frames["exposure"], n_classes)
    row = g1.iloc[0]
    print(f"\nG1 exposure: mean rate {row.mean_rate:.4f} vs expected {row.expected_rate:.4f} "
          f"(E[n]/9 with n~U{{2..9}} = 5.5/9) · max deviation {row.max_rel_deviation:.2%} "
          f"· {'PASS' if row.passes else 'FAIL'}")
    per_class = frames["exposure"].sort_values("slot") if "slot" in frames["exposure"] else frames["exposure"]
    for _, r in per_class.iterrows():
        print(f"    {r['class']:<6} {r.rate:.4f}")
    if not row.passes:
        failures.append("G1")

    # --- G5 ---
    g5 = result["violations"]["cluster_below_min_melodic"]
    print(f"\nG5 clusters below melodic minimum: {g5} across {GRID_DRAWS:,} draws "
          f"· {'PASS' if g5 == 0 else 'FAIL'}")

    # --- measurements for the report ---
    n = result["n"]
    support = sorted(np.unique(n).tolist())
    print(f"\nn support {support} (expected 2..9) · mean n {n.mean():.4f} (expected 5.5)")
    if support != list(range(2, n_classes + 1)):
        failures.append("n_support")
    k_decl, k_real = result["k_declared"], result["k_realised"]
    print(f"E[k] declared {k_decl.mean():.4f} · realised {k_real.mean():.4f} · "
          f"ratio {k_real.mean() / k_decl.mean():.3f}")
    short = float((k_real < k_decl).mean())
    reasons = result["reason_counts"]
    print(f"shortfall rate {short:.1%} · reasons {dict(reasons)}")
    h_mel = result["h_mel"]
    h_mel_valid = h_mel[~np.isnan(h_mel)]
    print(f"H_melodic mean {h_mel_valid.mean():.4f} · P(H_melodic=1) "
          f"{(h_mel_valid == 1).mean():.3f} · H mean {np.nanmean(result['h']):.4f}")
    clusters = result["n_clusters"]
    print(f"clusters/draw mean {clusters.mean():.3f} · P(0 clusters) {(clusters == 0).mean():.3%}")
    sizes = Counter()
    for (genre, size), c in result["cluster_genre_size"].items():
        sizes[size] += c
    total = sum(sizes.values())
    print("cluster size share: " + " ".join(f"{s}:{sizes[s] / total:.1%}" for s in sorted(sizes)))
    genres = Counter()
    for (genre, size), c in result["cluster_genre_size"].items():
        genres[genre] += c
    print("cluster genre share: " + " ".join(f"{g}:{c / total:.1%}" for g, c in genres.most_common()))
    print(f"plan time {np.median(result['plan_us']):.0f} µs/mix median · "
          f"pool {result['n_songs']} songs / {result['n_files']} files")

    # --- G4 audio ---
    print(f"\nG4 audio: decoding {AUDIO_ITEMS} items through a real DataLoader ...")
    audio = run_audio_cell(base, {}, "exp004_launch_config", AUDIO_ITEMS, WORKERS, SILENCE_EPS)
    g4_ok = audio["sum_error"] <= 1e-6
    print(f"  mixture ≡ Σ(targets): max abs error {audio['sum_error']:.3e} "
          f"· {'PASS' if g4_ok else 'FAIL'}")
    print(f"  silent slots: {audio['silent_tol']} by tolerance ({audio['silent_eq']} by "
          f"equality) · worst residue {audio['silent_residue']:.3e} · "
          f"{audio['items_per_s']:.2f} items/s")
    if not g4_ok:
        failures.append("G4")

    # --- persist ---
    (OUT_DIR / "metrics").mkdir(exist_ok=True)
    frames["exposure"].to_parquet(OUT_DIR / "metrics/launch_gate_exposure.parquet", index=False)
    summary = {
        "config": CONFIG, "grid_draws": GRID_DRAWS, "audio_items": AUDIO_ITEMS,
        "g1_mean_rate": float(row.mean_rate), "g1_expected_rate": float(row.expected_rate),
        "g1_max_rel_deviation": float(row.max_rel_deviation), "g1_pass": bool(row.passes),
        "g5_below_min_melodic": int(g5), "violations": result["violations"],
        "n_support": support, "n_mean": float(n.mean()),
        "k_declared_mean": float(k_decl.mean()), "k_realised_mean": float(k_real.mean()),
        "shortfall_rate": short, "shortfall_reasons": dict(reasons),
        "h_melodic_mean": float(h_mel_valid.mean()),
        "p_h_melodic_1": float((h_mel_valid == 1).mean()),
        "clusters_per_draw": float(clusters.mean()),
        "cluster_size_share": {int(s): sizes[s] / total for s in sorted(sizes)},
        "cluster_genre_share": {g: c / total for g, c in genres.most_common()},
        "g4_sum_error": audio["sum_error"], "g4_pass": g4_ok,
        "failures": failures,
    }
    (OUT_DIR / "metrics/launch_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLAUNCH SAMPLER GATES:", "PASS" if not failures else f"FAIL {failures}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
