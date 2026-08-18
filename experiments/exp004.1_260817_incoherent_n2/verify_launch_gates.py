"""verify_launch_gates.py — sampler gates on the EXACT exp004.1 launch config. NOT waivable
this time: the density floor (density_uniform_min 2) is the variable under test, and p=0
must provably never enter the coherent path.

Reuses the verifier's own cell runner (scripts/verify_coherent_sampler.py) so the numbers
come from the same code path; every plan draw goes through `GugakMixDataset.plan_item`,
i.e. `np.random.default_rng([seed, index])` → `_plan_item` — the identical per-item RNG
sequence `__getitem__` consumes, so these are the draws the trainer will really see.

  D   200,000-draw density check: n ~ U{2..9}, support exactly 2..9, ZERO draws at n=1,
      max relative deviation from uniform over the 8 sizes reported (gate < 2 %)
  C   coherent path never entered: zero clusters, k_declared 0, k_realised 0, no shortfall
      reasons, no cluster genre/size records — across all 200,000 draws
  G1  per-class exposure flat (< 2 % max deviation)      — same 200,000 draws
  M   manifest v2 resolves; the 16 excluded file_ids (= v1 − v2, asserted 16) never
      appear in the draw POOL (stronger than "never drawn": if they are not in the pool
      no draw can reach them); pool = 676 songs / 4,250 files; manifest joins on the
      picks: unknown_path / class_mismatch / quarantined / wrong_split / wrong_dataset 0
  G4  mixture ≡ Σ(targets) ≤ 1e-6 (report the actual max, expect ~1e-7) — 300 decoded
      items through a real DataLoader; silence by tolerance never equality

Run from the repo root:
    uv run python experiments/exp004.1_260817_incoherent_n2/verify_launch_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from verify_coherent_sampler import (  # noqa: E402
    aggregate_cells, evaluate_g1, run_audio_cell, run_grid_cell)
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402

CONFIG = "configs/exp004.1_htdemucs_incoherent_n2.yaml"
OUT_DIR = REPO_ROOT / "experiments/exp004.1_260817_incoherent_n2"
MANIFEST_V1 = "manifests/parquet/source_manifest.parquet"
GRID_DRAWS = 200_000
AUDIO_ITEMS = 300
WORKERS = 10
SILENCE_EPS = 1e-6
DENSITY_MAX_DEVIATION = 0.02
EXPECTED_EXCLUDED = 16
EXPECTED_POOL_SONGS, EXPECTED_POOL_FILES = 676, 4250


def check_density(n: np.ndarray, n_min: int, n_classes: int) -> tuple[bool, dict]:
    """n must be uniform over n_min..n_classes: exact support, zero below the floor,
    max relative deviation from 1/|support| under the gate.

    Args:
        n: per-draw class counts.
        n_min: density_uniform_min from the config.
        n_classes: number of output classes (support ceiling).
    """
    values, counts = np.unique(n, return_counts=True)
    support = values.tolist()
    expected_support = list(range(n_min, n_classes + 1))
    draws_at_one = int((n == 1).sum())
    below_floor = int((n < n_min).sum())
    expected_share = 1.0 / len(expected_support)
    shares = {int(v): c / len(n) for v, c in zip(values, counts)}
    max_dev = max(abs(s - expected_share) / expected_share for s in shares.values())
    ok = (support == expected_support and draws_at_one == 0 and below_floor == 0
          and max_dev < DENSITY_MAX_DEVIATION)
    return ok, {"support": support, "expected_support": expected_support,
                "draws_at_n1": draws_at_one, "below_floor": below_floor,
                "shares": shares, "expected_share": expected_share,
                "max_rel_deviation": float(max_dev), "mean_n": float(n.mean()),
                "expected_mean_n": float(np.mean(expected_support))}


def check_pool_exclusions(base: dict) -> tuple[bool, dict]:
    """The 16 v1−v2 file_ids must be absent from the dataset's actual draw pool.

    Args:
        base: the config's gugak_mix block.
    """
    v1 = pd.read_parquet(REPO_ROOT / MANIFEST_V1)
    v2 = pd.read_parquet(REPO_ROOT / base["source_manifest"])
    excluded = set(v1.file_id) - set(v2.file_id)
    dataset = GugakMixDataset(MixDatasetConfig.from_mapping(base), REPO_ROOT, num_items=1)
    pool_paths = {entry.out_path for entries in dataset.pool.values() for entry in entries}
    path_to_file = dict(zip(v1.out_path, v1.file_id))
    pool_file_ids = {path_to_file[p] for p in pool_paths}
    excluded_in_pool = sorted(pool_file_ids & excluded)
    n_songs, n_files = len(dataset.songs), sum(len(v) for v in dataset.pool.values())
    ok = (len(excluded) == EXPECTED_EXCLUDED and not excluded_in_pool
          and n_songs == EXPECTED_POOL_SONGS and n_files == EXPECTED_POOL_FILES
          and len(pool_paths) == n_files)
    return ok, {"manifest": base["source_manifest"], "v1_rows": len(v1), "v2_rows": len(v2),
                "excluded_file_ids": len(excluded), "excluded_in_pool": excluded_in_pool,
                "pool_songs": n_songs, "pool_files": n_files}


def main() -> None:
    raw = yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))
    base = dict(raw["gugak_mix"])
    n_classes = len(base["classes"])
    n_min = int(base["density_uniform_min"])
    echo = {"coherent_mix_prob": base["coherent_mix_prob"],
            "anchor_selection": base["anchor_selection"]}
    cell = {"label": "exp004.1_launch_config", "base": base, "grid_draws": GRID_DRAWS,
            "exclusions_path": "configs/manifest_exclusions.yaml", "overrides": echo}
    print(f"config {CONFIG}: p={base['coherent_mix_prob']} mode={base['anchor_selection']} "
          f"n_min={n_min} min_melodic={base['cluster_min_melodic']} seed={base['seed']}")
    failures = []

    # --- M (pool-level): manifest v2 resolves, exclusions absent from the pool ---
    print("\nMANIFEST v2 / exclusions (pool level)")
    m_ok, m = check_pool_exclusions(base)
    print(f"  manifest {m['manifest']}: v1 {m['v1_rows']} rows → v2 {m['v2_rows']} rows · "
          f"excluded file_ids (v1−v2) {m['excluded_file_ids']} (expected {EXPECTED_EXCLUDED})")
    print(f"  pool {m['pool_songs']} songs / {m['pool_files']} files "
          f"(expected {EXPECTED_POOL_SONGS} / {EXPECTED_POOL_FILES})")
    print(f"  excluded file_ids present in pool: {len(m['excluded_in_pool'])} "
          f"· {'PASS' if m_ok else 'FAIL'}")
    if not m_ok:
        failures.append("manifest_exclusions")

    # --- 200,000 plan draws through the real per-item RNG path ---
    result = run_grid_cell(cell)
    frames = aggregate_cells([result])
    n = result["n"]

    # --- D: density ---
    d_ok, d = check_density(n, n_min, n_classes)
    print(f"\nD density (n ~ U{{{n_min}..{n_classes}}}, {GRID_DRAWS:,} draws): support {d['support']} "
          f"(expected {d['expected_support']}) · draws at n=1: {d['draws_at_n1']} · "
          f"below floor: {d['below_floor']}")
    print(f"  per-size share (expected {d['expected_share']:.4f} each): "
          + " ".join(f"{k}:{v:.4f}" for k, v in sorted(d['shares'].items())))
    print(f"  max relative deviation from uniform {d['max_rel_deviation']:.2%} "
          f"(gate < {DENSITY_MAX_DEVIATION:.0%}) · mean n {d['mean_n']:.4f} "
          f"(expected {d['expected_mean_n']}) · {'PASS' if d_ok else 'FAIL'}")
    if not d_ok:
        failures.append("density")

    # --- C: coherent path never entered ---
    clusters = result["n_clusters"]
    c_stats = {"draws_with_clusters": int((clusters > 0).sum()),
               "total_clusters": int(clusters.sum()),
               "k_declared_max": int(result["k_declared"].max()),
               "k_realised_max": int(result["k_realised"].max()),
               "k_target_max": float(result["k_target"].max()),
               "shortfall_reasons": dict(result["reason_counts"]),
               "cluster_records": len(result["cluster_genre_size"]),
               "h_all_nan": bool(np.isnan(result["h"]).all()),
               "h_mel_all_nan": bool(np.isnan(result["h_mel"]).all())}
    c_ok = (c_stats["draws_with_clusters"] == 0 and c_stats["total_clusters"] == 0
            and c_stats["k_declared_max"] == 0 and c_stats["k_realised_max"] == 0
            and c_stats["k_target_max"] == 0.0 and not c_stats["shortfall_reasons"]
            and c_stats["cluster_records"] == 0)
    print(f"\nC coherent path never entered: draws with ≥1 cluster {c_stats['draws_with_clusters']} · "
          f"clusters total {c_stats['total_clusters']} · k_declared max {c_stats['k_declared_max']} · "
          f"k_realised max {c_stats['k_realised_max']} · k_target max {c_stats['k_target_max']} · "
          f"shortfall reasons {c_stats['shortfall_reasons']} · "
          f"H all-NaN {c_stats['h_all_nan']} · {'PASS' if c_ok else 'FAIL'}")
    if not c_ok:
        failures.append("coherent_path_entered")

    # --- correctness vs manifest on the picks (from the cell runner) ---
    print("\nCORRECTNESS vs manifest (pick level, 200,000 draws)")
    for name, count in sorted(result["violations"].items()):
        print(f"  [{'PASS' if count == 0 else 'FAIL'}] {name}: {count}")
        if count:
            failures.append(name)
    print("  (note: the cell runner's own `excluded_file` counter reads a yaml key that "
          "does not exist and is vacuous — the pool-level check above is the real one)")

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
    print(f"plan time {np.median(result['plan_us']):.0f} µs/mix median · "
          f"pool {result['n_songs']} songs / {result['n_files']} files")

    # --- G4 audio ---
    print(f"\nG4 audio: decoding {AUDIO_ITEMS} items through a real DataLoader ...")
    audio = run_audio_cell(base, {}, "exp004.1_launch_config", AUDIO_ITEMS, WORKERS, SILENCE_EPS)
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
        "manifest": m, "density": d, "coherent_path": c_stats,
        "violations": result["violations"],
        "g1_mean_rate": float(row.mean_rate), "g1_expected_rate": float(row.expected_rate),
        "g1_max_rel_deviation": float(row.max_rel_deviation), "g1_pass": bool(row.passes),
        "g4_sum_error": audio["sum_error"], "g4_pass": g4_ok,
        "failures": failures,
    }
    (OUT_DIR / "metrics/launch_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLAUNCH SAMPLER GATES:", "PASS" if not failures else f"FAIL {failures}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
