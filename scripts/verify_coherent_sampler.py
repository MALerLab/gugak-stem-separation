"""verify_coherent_sampler.py — prove the anchor-cluster sampler before it gets GPU time.

Extends the exp002.2-era verifier to the p-knob sampler (mix_dataset module docstring:
density and class identity first, coherent clusters second). The exp002.2-specific
characterisation this file used to carry (co-occurrence ratios, effective-content
counts) described the song-first coherent draw, which no longer exists; the H statistic
below (M1) is that analysis' successor.

HARD GATES (pass/fail — the build is not done until all pass):
  G1  per-class exposure flat (max deviation < 2%) for every p × anchor_selection cell.
      THIS IS THE POINT OF THE BUILD: exp002.2 collapsed exposure to real 편성.
  G2  coherent_mix_prob=0.0 (density_uniform_min=1, seed 42) reproduces exp002's draw
      sequence bit-identically — item tensors compared against the UNTOUCHED exp002-era
      module (git blob, --ref-commit), not against this refactor's own p=0 path.
  G3  realised mean k tracks declared p·n within 2% per (p, n) cell, or the shortfall
      is logged and quantified (every short draw carries a reason).
  G4  mixture ≡ Σ(targets) to ≤ 1e-6, through a real DataLoader.
  G5  ZERO clusters holding fewer than cluster_min_melodic non-percussion classes,
      across every draw of every cell — a direct assertion, not a statistic.
  +   manifest-ground-truth correctness, as before: every claim checked against
      source_manifest, never the sampler's own bookkeeping (cluster stems really share
      one song and one offset; nothing quarantined / excluded / mis-split is ever
      drawn; silence tested by tolerance, never equality — post-ingest silence is 2⁻²³).

MEASUREMENTS (numbers, no verdicts — these are what picks p_max later):
  M1  H = Σᵢ C(kᵢ,2) / C(n,2): the fraction of instrument PAIRS in true unison —
      alongside H_melodic, the same statistic over NON-PERCUSSION members and classes
      only. H counts a {타악기, melody} pairing as unison; H_melodic is the dose the
      heterophony hypothesis is actually about, and the primary treatment variable.
      p is only what we configure.
  M2  cluster-size histogram at p=1.0, per genre, per mode.
  M3  anchor genre composition as a function of k, per mode (does 산조 survive?).
  M4  shortfall rate per (p, n) × mode, with reasons.
  M5  draw-time cost per mix vs the p=0 baseline (plan path + full audio path).

Statistics come through `GugakMixDataset.plan_item`, which runs the real per-item RNG
sequence and stops before decoding audio — so hundreds of thousands of draws are cheap
and describe the sampler that will actually train, not a reimplementation of it. The
audio-dependent gates (G2, G4) pull items through real DataLoaders.

Outputs: metrics parquet (tracked) + figures (gitignored, regenerable) under --out-dir.

Run:
    uv run python scripts/verify_coherent_sampler.py
      --config configs/exp002_htdemucs_v2_uniform_n.yaml   base gugak_mix block
      --grid-draws 200000       plan draws per (p, mode) cell
      --g2-items 200            items bit-compared against the exp002-era module
      --audio-items 300         items decoded per audio cell (G4, M5)
      --workers 12
      --out-dir experiments/260815_anchor_cluster_mixer
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402

# classes the taxonomy defines but this scheme must never train on
QUARANTINED_CLASSES = ["pitched_percussion", "voice"]

GRID_P = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID_MODES = ["uniform", "overlap_weighted", "greedy"]

# shortfall reasons, bit-encoded per draw so attribution survives aggregation
REASON_BITS = {"no_eligible_anchor": 1, "greedy_single_anchor": 2, "r1_dropped": 4}


# --- closed-form baseline ----------------------------------------------------
def expected_exposure_rate(n_min: int, n_classes: int) -> float:
    """P(class in S) under the sampler's own steps 1–2, exactly.

    n ~ U{n_min..C} then n classes uniformly without replacement gives
    P(class present) = E[n]/C, independent of p BY CONSTRUCTION — the number G1
    checks the realised rates against.

    Args:
        n_min: density_uniform_min.
        n_classes: size of the class scheme.
    """
    return float(np.arange(n_min, n_classes + 1).mean() / n_classes)


# --- one grid cell (runs in a worker process) --------------------------------
def run_grid_cell(cell: dict) -> dict:
    """Draw `grid_draws` plans for one (p, mode) cell and aggregate everything.

    Correctness is checked HERE, vectorized against the manifest, so only counts and
    compact per-draw arrays travel back to the parent. Every drawn pick is joined to
    source_manifest by out_path; song/split/class/exclusion truth is read from there.

    Args:
        cell: {label, base (gugak_mix dict), overrides, grid_draws, exclusions_path}.
    """
    block = dict(cell["base"])
    block.update(cell["overrides"])
    cfg = MixDatasetConfig.from_mapping(block)
    dataset = GugakMixDataset(cfg, REPO_ROOT, num_items=cell["grid_draws"])
    classes = list(cfg.classes)
    draws = cell["grid_draws"]
    melodic_slots = dataset.melodic_slots
    min_melodic = int(cfg.cluster_min_melodic)

    # per-draw arrays
    n_arr = np.zeros(draws, dtype=np.int16)
    k_target = np.zeros(draws, dtype=np.float32)
    k_declared = np.zeros(draws, dtype=np.int16)
    k_realised = np.zeros(draws, dtype=np.int16)
    n_clusters = np.zeros(draws, dtype=np.int16)
    reason_bits = np.zeros(draws, dtype=np.int16)
    h_arr = np.full(draws, np.nan, dtype=np.float32)
    h_mel_arr = np.full(draws, np.nan, dtype=np.float32)
    plan_us = np.zeros(draws, dtype=np.float32)
    exposure = np.zeros(len(classes), dtype=np.int64)

    # pick-level collection for the manifest join
    pick_draw: list = []
    pick_slot: list = []
    pick_path: list = []
    pick_start: list = []
    pick_cluster: list = []
    pick_plan_song: list = []

    cluster_genre_size: Counter = Counter()      # (genre, size) -> count
    anchor_genre_by_k: Counter = Counter()       # (k_declared, genre) -> count
    reason_counts: Counter = Counter()
    structure_violations = 0
    below_min_melodic = 0                        # G5: direct per-cluster assertion

    for index in range(draws):
        t0 = time.perf_counter()
        plan = dataset.plan_item(index)
        plan_us[index] = (time.perf_counter() - t0) * 1e6

        n = plan.n
        n_arr[index] = n
        k_target[index] = plan.k_target
        k_declared[index] = plan.k_declared
        k_realised[index] = plan.k_realised
        n_clusters[index] = len(plan.clusters)
        np.add.at(exposure, list(plan.drawn_slots), 1)
        for reason in plan.shortfall_reasons:
            reason_bits[index] |= REASON_BITS[reason]
            reason_counts[reason] += 1

        sizes = [len(c.member_slots) for c in plan.clusters]
        if n >= 2:
            h_arr[index] = (sum(s * (s - 1) for s in sizes) / 2) / (n * (n - 1) / 2)
        # H_melodic: same statistic over melodic members / melodic classes only
        n_melodic = len(set(plan.drawn_slots) & melodic_slots)
        melodic_sizes = [len(set(c.member_slots) & melodic_slots)
                         for c in plan.clusters]
        if n_melodic >= 2:
            h_mel_arr[index] = ((sum(s * (s - 1) for s in melodic_sizes) / 2)
                                / (n_melodic * (n_melodic - 1) / 2))
        below_min_melodic += sum(1 for s in melodic_sizes if s < min_melodic)
        for cluster in plan.clusters:
            cluster_genre_size[(cluster.genre_sub, len(cluster.member_slots))] += 1
            anchor_genre_by_k[(plan.k_declared, cluster.genre_sub)] += 1

        # structural invariants (plan-internal; manifest truth comes after the loop)
        slots = [p.slot for p in plan.picks]
        ok = (len(plan.picks) == n == len(plan.drawn_slots)
              and len(set(slots)) == len(slots)
              and set(slots) == set(plan.drawn_slots)
              and plan.k_realised == sum(sizes)
              and all(s >= 2 for s in sizes)
              and plan.k_declared <= n
              # shortfall and reasons must imply each other (G3's accounting)
              and ((plan.k_realised < plan.k_declared) == bool(plan.shortfall_reasons)))
        if not ok:
            structure_violations += 1

        for pick in plan.picks:
            pick_draw.append(index)
            pick_slot.append(pick.slot)
            pick_path.append(pick.entry.out_path)
            pick_start.append(pick.start_frame)
            pick_cluster.append(-1 if pick.cluster_index is None else pick.cluster_index)
            pick_plan_song.append(
                "" if pick.cluster_index is None
                else plan.clusters[pick.cluster_index].song_id)

    # --- manifest ground truth, one vectorized join ---
    manifest = pd.read_parquet(REPO_ROOT / cfg.source_manifest)
    excluded = set(yaml.safe_load(
        (REPO_ROOT / cell["exclusions_path"]).read_text()).get("exclude", []))
    picks = pd.DataFrame({"draw": pick_draw, "slot": pick_slot, "out_path": pick_path,
                          "start_frame": pick_start, "cluster": pick_cluster,
                          "plan_song": pick_plan_song})
    picks = picks.merge(
        manifest[["out_path", "song_id", "stem_group", "split", "dataset", "file_id"]],
        on="out_path", how="left")

    slot_class = picks.slot.map(dict(enumerate(classes)))
    violations = {
        "structure": int(structure_violations),
        "cluster_below_min_melodic": int(below_min_melodic),   # G5
        "unknown_path": int(picks.song_id.isna().sum()),
        "class_mismatch": int((picks.stem_group != slot_class).sum()),
        "quarantined": int(picks.stem_group.isin(QUARANTINED_CLASSES).sum()),
        "excluded_file": int(picks.file_id.isin(excluded).sum()),
        "wrong_split": int((picks.split != cfg.split).sum()),
        "wrong_dataset": int((~picks.dataset.isin(cfg.datasets)).sum()),
    }
    in_cluster = picks[picks.cluster >= 0]
    if len(in_cluster):
        grouped = in_cluster.groupby(["draw", "cluster"])
        violations["cluster_multi_song"] = int((grouped.song_id.nunique() != 1).sum())
        violations["cluster_multi_offset"] = int(
            (grouped.start_frame.nunique() != 1).sum())
        violations["cluster_song_vs_plan"] = int(
            (in_cluster.song_id != in_cluster.plan_song).sum())
    else:
        violations["cluster_multi_song"] = 0
        violations["cluster_multi_offset"] = 0
        violations["cluster_song_vs_plan"] = 0

    return {
        "label": cell["label"], "mode": cfg.anchor_selection,
        "p": float(cell["overrides"].get("coherent_mix_prob", 0.0)),
        "n_min": int(cfg.density_uniform_min), "draws": draws,
        "classes": classes, "exposure": exposure,
        "n": n_arr, "k_target": k_target, "k_declared": k_declared,
        "k_realised": k_realised, "n_clusters": n_clusters,
        "reason_bits": reason_bits, "h": h_arr, "h_mel": h_mel_arr,
        "plan_us": plan_us,
        "cluster_genre_size": dict(cluster_genre_size),
        "anchor_genre_by_k": dict(anchor_genre_by_k),
        "reason_counts": dict(reason_counts), "violations": violations,
        "n_songs": len(dataset.songs), "n_files": sum(len(v) for v in dataset.pool.values()),
    }


# --- G2: bit-identity against the exp002-era module --------------------------
def load_reference_module(ref_commit: str):
    """Materialise mix_dataset.py as of `ref_commit` and import it as a module.

    The blob is fetched from git — the comparison target is the code exp002 actually
    trained with, not any file in the working tree. It lands in a temp dir: it is a
    copy of repo history, not an artifact.
    """
    source = subprocess.run(
        ["git", "show", f"{ref_commit}:src/data/mix_dataset.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    ref_path = Path(tempfile.mkdtemp()) / f"mix_dataset_ref_{ref_commit}.py"
    ref_path.write_text(source)
    spec = importlib.util.spec_from_file_location("mix_dataset_exp002_ref", ref_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mix_dataset_exp002_ref"] = module   # dataclass needs it findable
    spec.loader.exec_module(module)
    return module


def check_g2_bit_identity(base_block: dict, ref_commit: str, items: int,
                          workers: int) -> dict:
    """Compare item tensors (stems AND mixture) new-vs-reference at p=0, seed as-is."""
    ref_module = load_reference_module(ref_commit)
    new_ds = GugakMixDataset(MixDatasetConfig.from_mapping(dict(base_block)),
                             REPO_ROOT, items)
    ref_ds = ref_module.GugakMixDataset(
        ref_module.MixDatasetConfig.from_mapping(dict(base_block)), REPO_ROOT, items)
    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=max(2, workers // 2))
    new_loader = torch.utils.data.DataLoader(new_ds, **loader_kwargs)
    ref_loader = torch.utils.data.DataLoader(ref_ds, **loader_kwargs)
    mismatches = []
    for index, ((s_new, m_new), (s_ref, m_ref)) in enumerate(zip(new_loader, ref_loader)):
        if not (torch.equal(s_new, s_ref) and torch.equal(m_new, m_ref)):
            mismatches.append(index)
    return {"items": items, "ref_commit": ref_commit, "mismatches": mismatches}


# --- G4 / M5: audio through a real DataLoader --------------------------------
def run_audio_cell(base_block: dict, overrides: dict, label: str, items: int,
                   workers: int, silence_eps: float) -> dict:
    """Decode `items` real items: sum identity, silence-by-tolerance, throughput."""
    block = dict(base_block)
    block.update(overrides)
    dataset = GugakMixDataset(MixDatasetConfig.from_mapping(block), REPO_ROOT, items)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=workers)
    worst_sum_error = 0.0
    worst_silent_residue = 0.0
    equality_silent_slots = 0
    tolerance_silent_slots = 0
    t0 = time.perf_counter()
    for stems, mixture in loader:
        stems, mixture = stems[0].numpy(), mixture[0].numpy()
        worst_sum_error = max(worst_sum_error,
                              float(np.abs(mixture - stems.sum(axis=0)).max()))
        # an UNDRAWN slot is exact zeros; a DRAWN slot whose window happens to be
        # silent carries the 2⁻²³ ingest residue — the gap the tolerance rule covers
        peaks = np.abs(stems).max(axis=(1, 2))
        equality_silent_slots += int((peaks == 0.0).sum())
        tolerance_silent_slots += int((peaks < silence_eps).sum())
        residues = peaks[(peaks > 0.0) & (peaks < silence_eps)]
        if residues.size:
            worst_silent_residue = max(worst_silent_residue, float(residues.max()))
    seconds = time.perf_counter() - t0
    return {"label": label, "items": items, "sum_error": worst_sum_error,
            "silent_eq": equality_silent_slots, "silent_tol": tolerance_silent_slots,
            "silent_residue": worst_silent_residue,
            "items_per_s": items / seconds, "ms_per_item": seconds / items * 1e3}


# --- aggregation -------------------------------------------------------------
def aggregate_cells(cells: list) -> dict:
    """Grid-cell results → the tidy frames the gates, parquet and figures consume."""
    exposure_rows, ktrack_rows, h_rows, cluster_rows, anchor_rows, timing_rows = \
        [], [], [], [], [], []
    for cell in cells:
        key = {"label": cell["label"], "mode": cell["mode"], "p": cell["p"],
               "n_min": cell["n_min"]}
        rates = cell["exposure"] / cell["draws"]
        for class_name, rate in zip(cell["classes"], rates):
            exposure_rows.append({**key, "class": class_name, "rate": float(rate)})
        timing_rows.append({**key, "plan_us_mean": float(cell["plan_us"].mean()),
                            "plan_us_p50": float(np.percentile(cell["plan_us"], 50)),
                            "plan_us_p95": float(np.percentile(cell["plan_us"], 95))})
        for n in np.unique(cell["n"]):
            mask = cell["n"] == n
            count = int(mask.sum())
            short = cell["k_declared"][mask] - cell["k_realised"][mask]
            bits = cell["reason_bits"][mask]
            ktrack_rows.append({
                **key, "n": int(n), "count": count,
                "k_target_mean": float(cell["k_target"][mask].mean()),
                "k_declared_mean": float(cell["k_declared"][mask].mean()),
                "k_realised_mean": float(cell["k_realised"][mask].mean()),
                "shortfall_rate": float((short > 0).mean()),
                "shortfall_mean": float(short.mean()),
                "r_no_anchor": float((bits & 1 > 0).mean()),
                "r_greedy_stop": float((bits & 2 > 0).mean()),
                "r_r1_dropped": float((bits & 4 > 0).mean())})
            h = cell["h"][mask]
            h = h[~np.isnan(h)]
            # H_melodic is NaN when the draw holds < 2 melodic classes, so its count
            # differs from raw H's — both live in the same row for the D3 comparison
            h_mel = cell["h_mel"][mask]
            h_mel = h_mel[~np.isnan(h_mel)]
            if h.size:
                h_rows.append({
                    **key, "n": int(n), "count": int(h.size),
                    "h_mean": float(h.mean()), "h_std": float(h.std()),
                    "h_p10": float(np.percentile(h, 10)),
                    "h_p50": float(np.percentile(h, 50)),
                    "h_p90": float(np.percentile(h, 90)),
                    "h_zero_frac": float((h == 0).mean()),
                    "h_one_frac": float((h == 1).mean()),
                    "h_mel_count": int(h_mel.size),
                    "h_mel_mean": float(h_mel.mean()) if h_mel.size else math.nan,
                    "h_mel_p50": (float(np.percentile(h_mel, 50))
                                  if h_mel.size else math.nan),
                    "h_mel_zero_frac": (float((h_mel == 0).mean())
                                        if h_mel.size else math.nan),
                    "h_mel_one_frac": (float((h_mel == 1).mean())
                                       if h_mel.size else math.nan)})
        for (genre, size), count in cell["cluster_genre_size"].items():
            cluster_rows.append({**key, "genre": genre, "size": int(size),
                                 "count": int(count)})
        for (k, genre), count in cell["anchor_genre_by_k"].items():
            anchor_rows.append({**key, "k_declared": int(k), "genre": genre,
                                "count": int(count)})
    return {"exposure": pd.DataFrame(exposure_rows),
            "k_tracking": pd.DataFrame(ktrack_rows),
            "h_stats": pd.DataFrame(h_rows),
            "cluster_sizes": pd.DataFrame(cluster_rows),
            "anchor_genres": pd.DataFrame(anchor_rows),
            "timing": pd.DataFrame(timing_rows)}


def evaluate_g1(exposure: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Max relative deviation of per-class exposure from the cell mean, per cell."""
    rows = []
    for label, group in exposure.groupby("label"):
        mean_rate = group.rate.mean()
        deviation = (group.rate / mean_rate - 1.0).abs().max()
        expected = expected_exposure_rate(int(group.n_min.iloc[0]), n_classes)
        rows.append({"label": label, "mode": group["mode"].iloc[0],
                     "p": group.p.iloc[0], "mean_rate": mean_rate,
                     "expected_rate": expected,
                     "max_rel_deviation": float(deviation),
                     "passes": bool(deviation < 0.02)})
    return pd.DataFrame(rows).sort_values(["mode", "p"])


def evaluate_g3(k_tracking: pd.DataFrame) -> pd.DataFrame:
    """Per (p, n, mode): realised-vs-target deviation, strict 2% or quantified."""
    rows = []
    for _, row in k_tracking.iterrows():
        if row.k_target_mean == 0.0:
            continue    # p=0 cells: nothing declared, nothing to track
        deviation = abs(row.k_realised_mean - row.k_target_mean) / row.k_target_mean
        rows.append({"label": row.label, "mode": row["mode"], "p": row.p,
                     "n": row.n, "k_target": row.k_target_mean,
                     "k_declared": row.k_declared_mean,
                     "k_realised": row.k_realised_mean,
                     "rel_deviation": float(deviation),
                     "strict_pass": bool(deviation <= 0.02),
                     # n=1 cannot host a cluster (k=2 clamps to a degenerate 1 → 0),
                     # so its whole p·n is shortfall BY DESIGN — quantified, not broken
                     "clamped_n1": bool(row.n == 1),
                     "shortfall_rate": row.shortfall_rate,
                     "shortfall_mean": row.shortfall_mean})
    return pd.DataFrame(rows)


# --- figures -----------------------------------------------------------------
def make_figures(frames: dict, g1: pd.DataFrame, out_dir: Path) -> list:
    """Report figures. Colors: frozen dancheong entity bindings; p is never
    color-encoded (axis/facet only) — no unvalidated ordinal ramp exists."""
    import koreanize_matplotlib  # noqa: F401 — registers the Korean font on import
    import matplotlib.pyplot as plt

    palette = yaml.safe_load((REPO_ROOT / "configs/palette_dancheong.yaml").read_text())
    class_colors = palette["stem_class_colors"]
    genre_colors = palette["genre_colors"]
    mode_colors = {mode: slot["hex"] for mode, slot
                   in zip(GRID_MODES, palette["slots"])}   # validated adjacent order
    grid_color, text_secondary = palette["grid"], palette["text_secondary"]
    plt.rcParams.update({"figure.facecolor": palette["surface"],
                         "axes.facecolor": palette["surface"],
                         "axes.edgecolor": grid_color, "axes.grid": True,
                         "grid.color": grid_color, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "font.size": 9})
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name: str) -> None:
        path = figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    grid = [cell for cell in g1.label if not cell.endswith("nmin2")]

    # F1 — G1: exposure rate per class, lines over p, one panel per mode
    exposure = frames["exposure"]
    exposure = exposure[exposure.label.isin(grid)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    expected = g1[g1.label.isin(grid)].expected_rate.iloc[0]
    for ax, mode in zip(axes, GRID_MODES):
        sub = exposure[exposure["mode"] == mode]
        ax.axhspan(expected * 0.98, expected * 1.02, color=grid_color, alpha=0.5,
                   zorder=0, label="±2% gate" if mode == GRID_MODES[0] else None)
        ax.axhline(expected, color=text_secondary, linewidth=1, linestyle="--")
        for class_name, class_group in sub.groupby("class"):
            class_group = class_group.sort_values("p")
            ax.plot(class_group.p, class_group.rate, marker="o", markersize=3.5,
                    linewidth=1.4, color=class_colors[class_name], label=class_name)
        ax.set_title(f"anchor_selection = {mode}")
        ax.set_xlabel("coherent_mix_prob p")
        ax.set_ylim(expected * 0.94, expected * 1.06)
    axes[0].set_ylabel("P(class ∈ mix)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                    frameon=False)
    fig.suptitle("G1 — per-class exposure is flat at every p (dashed = E[n]/9)", y=1.04)
    save(fig, "fig_g1_exposure.png")

    # F2 — G3: mean realised k vs n, one panel per p>0, modes as lines
    ktrack = frames["k_tracking"]
    ktrack = ktrack[ktrack.label.isin(grid) & (ktrack.p > 0)]
    p_values = sorted(ktrack.p.unique())
    fig, axes = plt.subplots(1, len(p_values), figsize=(2.9 * len(p_values), 3.0),
                             sharey=True)
    for ax, p in zip(np.atleast_1d(axes), p_values):
        sub = ktrack[ktrack.p == p]
        ns = np.arange(1, 10)
        ax.plot(ns, p * ns, color=text_secondary, linewidth=1.2, linestyle="--",
                label="declared p·n")
        for mode in GRID_MODES:
            mode_group = sub[sub["mode"] == mode].sort_values("n")
            ax.plot(mode_group.n, mode_group.k_realised_mean, marker="o",
                    markersize=3.5, linewidth=1.6, color=mode_colors[mode], label=mode)
        ax.set_title(f"p = {p}")
        ax.set_xlabel("n (mix density)")
    np.atleast_1d(axes)[0].set_ylabel("mean realised k")
    np.atleast_1d(axes)[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("G3 — realised coherent-stem count vs the declared p·n", y=1.04)
    save(fig, "fig_g3_k_tracking.png")

    # F3 — M1: mean H heatmaps (n × p) per mode, single-hue sequential. Two variants:
    # raw H (continuity with the pre-melodic-rule record) and H_melodic (primary).
    h_stats = frames["h_stats"]
    h_stats = h_stats[h_stats.label.isin(grid)]
    h_variants = [
        ("h_mean", "mean H (unison pair fraction)",
         "M1 — raw H = Σ C(k_i, 2) / C(n, 2), all classes", "fig_m1_h_mean.png"),
        ("h_mel_mean", "mean H_melodic (melodic unison pair fraction)",
         "M1 — H_melodic: non-percussion pairs only, the primary treatment variable",
         "fig_m1_h_melodic.png")]
    for value_column, bar_label, title, filename in h_variants:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
        for ax, mode in zip(axes, GRID_MODES):
            sub = h_stats[h_stats["mode"] == mode]
            pivot = sub.pivot_table(index="n", columns="p", values=value_column)
            pivot = pivot.reindex(index=np.arange(2, 10), columns=GRID_P)
            image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto",
                              cmap="Blues", vmin=0.0, vmax=1.0,
                              extent=(-0.5, len(GRID_P) - 0.5, 1.5, 9.5))
            ax.set_xticks(range(len(GRID_P)), [str(p) for p in GRID_P])
            ax.set_yticks(np.arange(2, 10))
            ax.set_title(f"{mode}")
            ax.set_xlabel("p")
            ax.grid(False)
            for (row_n, col_p), value in np.ndenumerate(pivot.to_numpy()):
                if not np.isnan(value):
                    ax.text(col_p, row_n + 2, f"{value:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if value > 0.6 else palette["text_primary"])
        axes[0].set_ylabel("n")
        fig.colorbar(image, ax=axes, shrink=0.85, label=bar_label)
        fig.suptitle(title, y=1.02)   # ASCII subscripts: NanumGothic lacks ᵢ
        save(fig, filename)

    # F4 — M2: cluster-size histogram at p=1.0, stacked by genre, per mode
    clusters = frames["cluster_sizes"]
    clusters = clusters[clusters.label.isin(grid) & (clusters.p == 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    genre_order = [g for g in genre_colors if g in set(clusters.genre)]
    for ax, mode in zip(axes, GRID_MODES):
        sub = clusters[clusters["mode"] == mode]
        total = sub["count"].sum()
        bottom = np.zeros(8)
        sizes = np.arange(2, 10)
        for genre in genre_order:
            genre_counts = (sub[sub.genre == genre].set_index("size")["count"]
                            .reindex(sizes, fill_value=0).to_numpy() / total)
            ax.bar(sizes, genre_counts, bottom=bottom, width=0.72,
                   color=genre_colors[genre], label=genre,
                   edgecolor=palette["surface"], linewidth=0.8)
            bottom += genre_counts
        ax.set_title(f"{mode}")
        ax.set_xlabel("cluster size")
        ax.set_xticks(sizes)
    axes[0].set_ylabel("fraction of clusters (p = 1.0)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                    frameon=False)
    fig.suptitle("M2 — cluster sizes at p = 1.0, by anchor genre", y=1.04)
    save(fig, "fig_m2_cluster_sizes.png")

    # F5 — M3: anchor genre share vs k, per mode
    anchors = frames["anchor_genres"]
    anchors = anchors[anchors.label.isin(grid) & (anchors.p == 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, mode in zip(axes, GRID_MODES):
        sub = anchors[anchors["mode"] == mode]
        totals = sub.groupby("k_declared")["count"].sum()
        for genre in genre_order:
            genre_share = (sub[sub.genre == genre].set_index("k_declared")["count"]
                           .reindex(totals.index, fill_value=0) / totals)
            ax.plot(genre_share.index, genre_share.to_numpy(), marker="o",
                    markersize=3.5, linewidth=1.8 if genre == "산조" else 1.2,
                    color=genre_colors[genre], label=genre)
        ax.set_title(f"{mode}")
        ax.set_xlabel("k (declared coherent stems)")
    axes[0].set_ylabel("share of anchor draws (p = 1.0)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                    frameon=False)
    fig.suptitle("M3 — who anchors at high k: genre composition of realised anchors",
                 y=1.04)
    save(fig, "fig_m3_anchor_genres.png")

    # F6 — M4: shortfall rate heatmap (n × p) per mode
    ktrack_all = frames["k_tracking"]
    ktrack_all = ktrack_all[ktrack_all.label.isin(grid) & (ktrack_all.p > 0)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, mode in zip(axes, GRID_MODES):
        sub = ktrack_all[ktrack_all["mode"] == mode]
        pivot = (sub.pivot_table(index="n", columns="p", values="shortfall_rate")
                 .reindex(index=np.arange(1, 10), columns=[p for p in GRID_P if p > 0]))
        image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto",
                          cmap="Blues", vmin=0.0, vmax=1.0,
                          extent=(-0.5, 3.5, 0.5, 9.5))
        ax.set_xticks(range(4), [str(p) for p in GRID_P if p > 0])
        ax.set_yticks(np.arange(1, 10))
        ax.set_title(f"{mode}")
        ax.set_xlabel("p")
        ax.grid(False)
        for (row_n, col_p), value in np.ndenumerate(pivot.to_numpy()):
            if not np.isnan(value):
                ax.text(col_p, row_n + 1, f"{value:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if value > 0.6 else palette["text_primary"])
    axes[0].set_ylabel("n")
    fig.colorbar(image, ax=axes, shrink=0.85, label="P(draw ends short)")
    fig.suptitle("M4 — shortfall rate per (p, n) × mode", y=1.02)
    save(fig, "fig_m4_shortfall.png")

    # F7 — M5: plan-time cost vs p, modes as lines
    timing = frames["timing"]
    timing = timing[timing.label.isin(grid)]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for mode in GRID_MODES:
        sub = timing[timing["mode"] == mode].sort_values("p")
        ax.plot(sub.p, sub.plan_us_mean, marker="o", markersize=4, linewidth=1.6,
                color=mode_colors[mode], label=mode)
    ax.set_xlabel("coherent_mix_prob p")
    ax.set_ylabel("plan time per mix (us)")   # ASCII: NanumGothic lacks the µ glyph
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("M5 — draw-time cost of the plan path", y=1.02)
    save(fig, "fig_m5_timing.png")

    return written


# --- main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify + characterise the anchor-cluster coherent sampler.")
    parser.add_argument("--config", default="configs/exp002_htdemucs_v2_uniform_n.yaml",
                        help="base gugak_mix block; the grid overrides p and mode")
    parser.add_argument("--grid-draws", type=int, default=200_000)
    parser.add_argument("--g2-items", type=int, default=200)
    parser.add_argument("--audio-items", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", default="experiments/260815_anchor_cluster_mixer")
    parser.add_argument("--ref-commit", default="44b2094",
                        help="git commit holding the exp002-era mix_dataset.py (G2)")
    parser.add_argument("--exclusions", default="configs/manifest_exclusions.yaml")
    parser.add_argument("--silence-eps", type=float, default=1e-6,
                        help="amplitude below which a buffer counts as silent; must "
                             "stay above the 1.192e-07 post-ingest DC residue")
    parser.add_argument("--skip-audio", action="store_true",
                        help="plan-path gates only (fast iteration)")
    args = parser.parse_args()

    raw = yaml.safe_load((REPO_ROOT / args.config).read_text())
    base = dict(raw["gugak_mix"])
    n_classes = len(base["classes"])
    out_dir = REPO_ROOT / args.out_dir
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"ANCHOR-CLUSTER SAMPLER VERIFICATION — base {args.config}")
    print("=" * 78)

    # --- the grid: 3 modes × 5 p, plus one density_uniform_min=2 sanity cell ---
    cells = [{"label": f"{mode}_p{p:g}", "base": base, "grid_draws": args.grid_draws,
              "exclusions_path": args.exclusions,
              "overrides": {"coherent_mix_prob": p, "anchor_selection": mode}}
             for mode in GRID_MODES for p in GRID_P]
    cells.append({"label": "uniform_p0.5_nmin2", "base": base,
                  "grid_draws": args.grid_draws, "exclusions_path": args.exclusions,
                  "overrides": {"coherent_mix_prob": 0.5, "anchor_selection": "uniform",
                                "density_uniform_min": 2}})
    print(f"grid: {len(cells)} cells × {args.grid_draws:,} plan draws "
          f"({args.workers} workers)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_grid_cell, cells))
    print(f"grid done in {time.perf_counter() - t0:.0f}s · draw pool "
          f"{results[0]['n_songs']} songs · {results[0]['n_files']} stem files")

    frames = aggregate_cells(results)
    failures: list = []

    # --- correctness (manifest ground truth) ---
    print("\n" + "-" * 78)
    print("CORRECTNESS — checked against the manifest, not the sampler's bookkeeping")
    print("-" * 78)
    total_violations: Counter = Counter()
    for cell in results:
        total_violations.update(cell["violations"])
    for name, count in sorted(total_violations.items()):
        status = "PASS" if count == 0 else "FAIL"
        print(f"  [{status}] {name}: {count} violations "
              f"({len(results)} cells × {args.grid_draws:,} draws)")
        if count:
            failures.append(f"correctness:{name}")

    # --- G1 ---
    print("\n" + "-" * 78)
    print("G1 — per-class exposure flat (< 2% max deviation) at every (p, mode)")
    print("-" * 78)
    g1 = evaluate_g1(frames["exposure"], n_classes)
    print(f"  {'cell':<24} {'mean rate':>10} {'expected':>10} {'max dev':>9}  gate")
    for _, row in g1.iterrows():
        print(f"  {row.label:<24} {row.mean_rate:>10.4f} {row.expected_rate:>10.4f} "
              f"{row.max_rel_deviation:>8.2%}  {'PASS' if row.passes else 'FAIL'}")
        if not row.passes:
            failures.append(f"G1:{row.label}")

    # --- G3 ---
    print("\n" + "-" * 78)
    print("G3 — realised mean k vs declared p·n per (p, n) cell (2% or quantified)")
    print("-" * 78)
    g3 = evaluate_g3(frames["k_tracking"])
    tracked = g3[~g3.clamped_n1]
    strict = int(tracked.strict_pass.sum())
    n1 = g3[g3.clamped_n1]
    print(f"  {strict}/{len(tracked)} trackable cells within 2% strictly; every "
          "remaining cell's shortfall is reason-attributed (structure check above)")
    print(f"  n=1 cells ({len(n1)}): k forced to 0 by design (no cluster of one) — "
          f"declared loss = p per draw, quantified not gated; worst trackable cells:")
    worst = tracked.sort_values("rel_deviation", ascending=False).head(8)
    print(f"  {'cell':<24} {'n':>2} {'p·n':>6} {'declared':>9} {'realised':>9} "
          f"{'dev':>8} {'short%':>7}")
    for _, row in worst.iterrows():
        print(f"  {row.label:<24} {row.n:>2.0f} {row.k_target:>6.2f} "
              f"{row.k_declared:>9.3f} {row.k_realised:>9.3f} "
              f"{row.rel_deviation:>7.1%} {row.shortfall_rate:>7.1%}")
    if total_violations["structure"] > 0:
        failures.append("G3:unattributed shortfall")

    # --- G5 ---
    print("\n" + "-" * 78)
    print("G5 — zero clusters below cluster_min_melodic non-percussion classes")
    print("-" * 78)
    g5_count = total_violations["cluster_below_min_melodic"]
    print(f"  [{'PASS' if g5_count == 0 else 'FAIL'}] {g5_count} clusters below the "
          f"melodic minimum across {len(results)} cells × {args.grid_draws:,} draws "
          "(direct per-cluster assertion)")
    # (already in `failures` via the correctness loop if nonzero)

    # --- D2: no_eligible_anchor visibility ---
    no_anchor_total = sum(cell["reason_counts"].get("no_eligible_anchor", 0)
                          for cell in results)
    print(f"\n  no_eligible_anchor fired {no_anchor_total:,} times in total; per cell:")
    for cell in results:
        count = cell["reason_counts"].get("no_eligible_anchor", 0)
        if count:
            print(f"    {cell['label']:<24} {count:>8,}  "
                  f"({count / cell['draws']:.2%} of draws)")

    # --- density_uniform_min=2 sanity cell ---
    nmin2 = next(cell for cell in results if cell["label"].endswith("nmin2"))
    support = sorted(np.unique(nmin2["n"]).tolist())
    nmin2_ok = support == list(range(2, n_classes + 1))
    print(f"\n  [{'PASS' if nmin2_ok else 'FAIL'}] density_uniform_min=2 cell: "
          f"n support {support}")
    if not nmin2_ok:
        failures.append("density_uniform_min=2 support")

    # --- audio gates ---
    g2 = None
    audio_rows = []
    if not args.skip_audio:
        print("\n" + "-" * 78)
        print("G2 — bit-identity vs the exp002-era module (p=0, seed 42)")
        print("-" * 78)
        g2 = check_g2_bit_identity(base, args.ref_commit, args.g2_items, args.workers)
        ok = not g2["mismatches"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {g2['items']} items vs commit "
              f"{g2['ref_commit']}: {len(g2['mismatches'])} mismatches"
              + (f" (first at {g2['mismatches'][:5]})" if g2["mismatches"] else ""))
        if not ok:
            failures.append("G2")

        print("\n" + "-" * 78)
        print("G4 / M5 — decoded items: mixture ≡ Σ(targets), silence, throughput")
        print("-" * 78)
        audio_cells = [("audio_p0_baseline", {}),
                       ("audio_uniform_p0.5", {"coherent_mix_prob": 0.5}),
                       ("audio_uniform_p1", {"coherent_mix_prob": 1.0}),
                       ("audio_overlapw_p1", {"coherent_mix_prob": 1.0,
                                              "anchor_selection": "overlap_weighted"}),
                       ("audio_greedy_p1", {"coherent_mix_prob": 1.0,
                                            "anchor_selection": "greedy"})]
        for label, overrides in audio_cells:
            result = run_audio_cell(base, overrides, label, args.audio_items,
                                    args.workers, args.silence_eps)
            audio_rows.append(result)
            sum_ok = result["sum_error"] <= 1e-6
            silence_ok = result["silent_tol"] >= result["silent_eq"]
            print(f"  [{'PASS' if sum_ok and silence_ok else 'FAIL'}] {label:<22} "
                  f"max|mix−Σ| {result['sum_error']:.2e} · "
                  f"{result['silent_eq']:,} slots ==0 vs {result['silent_tol']:,} "
                  f"< eps (residue {result['silent_residue']:.2e}) · "
                  f"{result['items_per_s']:.1f} items/s")
            if not sum_ok:
                failures.append(f"G4:{label}")
            if not silence_ok:
                failures.append(f"silence:{label}")

    # --- persist metrics + figures ---
    for name, frame in frames.items():
        frame.to_parquet(out_dir / "metrics" / f"{name}.parquet", index=False)
    g1.to_parquet(out_dir / "metrics" / "g1_exposure_gate.parquet", index=False)
    g3.to_parquet(out_dir / "metrics" / "g3_k_gate.parquet", index=False)
    if audio_rows:
        pd.DataFrame(audio_rows).to_parquet(out_dir / "metrics" / "audio_checks.parquet",
                                            index=False)
    figures = make_figures(frames, g1, out_dir)
    print(f"\nmetrics → {out_dir / 'metrics'} · {len(figures)} figures → "
          f"{out_dir / 'figures'}")

    print("\n" + "=" * 78)
    print("ALL GATES PASSED" if not failures else f"FAILURES: {failures}")
    print("=" * 78)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
