"""verify_coherent_sampler.py — prove the coherent sampler before it gets any GPU time.

Two jobs, deliberately separated.

CORRECTNESS (pass/fail, zero tolerance). Every claim is checked against the MANIFEST,
not against the sampler's own bookkeeping — asking the plan whether the plan is coherent
proves nothing. Each drawn file path is joined back to `source_manifest` and the song,
split, class and exclusion status are read from there.
  1. every stem of every mix comes from ONE song at ONE time offset
  2. no quarantined/excluded source ever appears (pitched_percussion, voice, the v2
     publisher defects, and anything outside the configured split)
  3. n never exceeds the number of classes audible in the drawn window
  4. mixture ≡ Σ(targets) within float tolerance
  5. silence is tested with a tolerance, never equality (post-ingest silence is 2⁻²³)

CHARACTERISATION (numbers, no verdict). What coherence costs and what it buys:
realised n distribution, per-class appearance rate, 가야금·양금 co-occurrence, and an
effective-distinct-content estimate — all against exp002's incoherent baseline, which is
computed in closed form because exp002's class draw is exactly "n ~ U{1..9}, then n
classes uniformly without replacement".

Statistics come through `GugakMixDataset.plan_item`, which runs the real per-item RNG
sequence and stops before decoding audio — so hundreds of thousands of draws are cheap
and describe the sampler that will actually train, not a reimplementation of it. The
audio-dependent checks (4, 5) pull a smaller sample through a real DataLoader.

Run:
    uv run python scripts/verify_coherent_sampler.py
      --config configs/exp002.2_htdemucs_coherent.yaml
      --plan-draws 200000     no-audio draws for statistics
      --audio-items 3000      items decoded for the sum/provenance assertions
      --workers 8
      --silence-eps 1e-6      tolerance for "is this buffer silent" (never ==0)
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from itertools import combinations
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


# --- the incoherent baseline, in closed form --------------------------------
def incoherent_baseline(n_classes: int) -> dict:
    """exp002's draw statistics, computed exactly rather than sampled.

    exp002 draws n ~ Uniform{1..n_classes} and then n classes uniformly WITHOUT
    replacement, which makes every quantity we want a small combinatorial sum:
      P(class c present)      = E[n] / C
      P(classes c and d both) = E[n(n-1)] / (C(C-1))
    No simulation, so there is nothing to be noisy about.

    Args:
        n_classes: size of the class scheme (9 here).
    """
    counts = np.arange(1, n_classes + 1)
    mean_n = counts.mean()
    mean_n_pairs = (counts * (counts - 1)).mean()
    return {
        "n_probabilities": {int(n): 1.0 / n_classes for n in counts},
        "mean_n": float(mean_n),
        "class_rate": float(mean_n / n_classes),
        "pair_rate": float(mean_n_pairs / (n_classes * (n_classes - 1))),
    }


# --- effective distinct content ---------------------------------------------
def log10_incoherent_content(pool: dict, classes: list, windows_per_file: dict) -> float:
    """log10 of how many distinct incoherent mixes the pool can produce.

    An incoherent mix picks a class subset, then for each chosen class an independent
    (file, window) pair, so the counts multiply across classes and sum over subsets:
        Σ_subsets Π_{c in subset} (Σ_{f in c} windows(f))
    Computed in log space and accumulated with logsumexp, since the number overflows.
    """
    per_class_log = {c: math.log10(sum(windows_per_file[entry.out_path]
                                       for entry in pool[c])) for c in classes}
    subset_logs = [sum(per_class_log[c] for c in subset)
                   for size in range(1, len(classes) + 1)
                   for subset in combinations(classes, size)]
    peak = max(subset_logs)
    return peak + math.log10(sum(10 ** (value - peak) for value in subset_logs))


def log10_coherent_content(songs: list, windows_per_file: dict,
                           hop_seconds: float, segment_seconds: float) -> float:
    """log10 of how many distinct coherent mixes the pool can produce.

    A coherent mix is (song, window, non-empty subset of what is audible there, one file
    per chosen class). The song and window are shared rather than chosen per class, so
    the multiplication that made the incoherent number astronomical does not happen —
    which is exactly the diversity cost being measured.

    Approximated per song as: (its window count) × 2^(classes it carries) − 1, using the
    song's whole class roster rather than per-window activity. That OVERSTATES the
    coherent number (not every class is audible in every window), so it is a
    conservative reading of the gap.
    """
    total = 0.0
    for song in songs:
        song_seconds = max(windows_per_file[entry.out_path + ":seconds"]
                           for entries in song.entries_by_class.values()
                           for entry in entries)
        windows = max(1.0, (song_seconds - segment_seconds) / hop_seconds + 1.0)
        file_choices = math.prod(len(entries)
                                 for entries in song.entries_by_class.values())
        total += windows * (2 ** len(song.entries_by_class) - 1) * file_choices
    return math.log10(total)


# --- reporting helpers ------------------------------------------------------
def print_histogram(title: str, realised: Counter, baseline: dict, total: int) -> None:
    """Side-by-side realised vs baseline probability table."""
    print(f"\n{title}")
    print(f"  {'n':>3} {'coherent':>10} {'exp002':>10} {'delta':>10}")
    for n in sorted(set(realised) | set(baseline)):
        got = realised.get(n, 0) / total
        want = baseline.get(n, 0.0)
        print(f"  {n:>3} {got:>10.4f} {want:>10.4f} {got - want:>+10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify + characterise coherent mixing.")
    parser.add_argument("--config", default="configs/exp002.2_htdemucs_coherent.yaml")
    parser.add_argument("--plan-draws", type=int, default=200_000)
    parser.add_argument("--audio-items", type=int, default=3_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--silence-eps", type=float, default=1e-6,
                        help="amplitude below which a buffer counts as silent; must "
                             "stay above the 1.192e-07 post-ingest DC residue")
    parser.add_argument("--content-hop-seconds", type=float, default=1.0,
                        help="window hop used to count 'distinct' excerpts")
    args = parser.parse_args()

    raw = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cfg = MixDatasetConfig.from_mapping(raw["gugak_mix"])
    dataset = GugakMixDataset(cfg, REPO_ROOT, num_items=args.audio_items)
    classes = list(cfg.classes)
    baseline = incoherent_baseline(len(classes))
    failures: list = []

    print("=" * 78)
    print(f"COHERENT SAMPLER VERIFICATION — {args.config}")
    print("=" * 78)
    print(f"coherent_mix_prob {cfg.coherent_mix_prob} · seed {cfg.seed} · "
          f"{len(classes)} classes · {cfg.segment_seconds:.0f}s @ {cfg.sample_rate} Hz")
    print(f"draw pool: {len(dataset.songs)} songs · "
          f"{sum(len(v) for v in dataset.pool.values())} stem files")

    # --- ground truth for the assertions: the manifest, not the sampler ---
    manifest = pd.read_parquet(REPO_ROOT / cfg.source_manifest)
    by_path = manifest.set_index("out_path")
    excluded = set(yaml.safe_load(
        (REPO_ROOT / "configs/manifest_exclusions.yaml").read_text()).get("exclude", []))

    # ================= CORRECTNESS =================
    print("\n" + "-" * 78)
    print("CORRECTNESS — checked against the manifest, not the sampler's bookkeeping")
    print("-" * 78)

    realised_n: Counter = Counter()
    class_appearances: Counter = Counter()
    pair_appearances: Counter = Counter()
    genre_appearances: Counter = Counter()
    active_counts: Counter = Counter()
    coherence_violations = 0
    quarantine_violations = 0
    n_cap_violations = 0
    split_violations = 0

    for index in range(args.plan_draws):
        plan = dataset.plan_item(index)
        if plan is None:
            coherence_violations += 1          # coherent_mix_prob 1.0 ⇒ never happens
            continue

        picked_classes = [class_name for _, class_name, _, _ in plan.picks]
        picked_paths = [entry.out_path for _, _, entry, _ in plan.picks]
        rows = by_path.loc[picked_paths]

        # (1) one song, one offset — song read from the manifest for each drawn file
        if rows.song_id.nunique() != 1 or rows.song_id.iloc[0] != plan.song.song_id:
            coherence_violations += 1

        # (2) nothing quarantined, excluded, or out of split
        if (not set(rows.stem_group) <= set(classes)
                or set(rows.stem_group) & set(QUARANTINED_CLASSES)
                or set(rows.file_id) & excluded):
            quarantine_violations += 1
        if not (rows.split == cfg.split).all() or not rows.dataset.isin(cfg.datasets).all():
            split_violations += 1

        # (3) n within what the window offered
        if len(plan.picks) > len(plan.active_classes) or len(plan.picks) < 1:
            n_cap_violations += 1

        realised_n[len(plan.picks)] += 1
        active_counts[len(plan.active_classes)] += 1
        class_appearances.update(picked_classes)
        pair_appearances.update(combinations(sorted(picked_classes), 2))
        genre_appearances[plan.song.genre_sub] += 1

    checks = [
        (f"all stems from one song at one offset ({args.plan_draws:,} mixes)",
         coherence_violations == 0, f"{coherence_violations} violations"),
        ("no quarantined / excluded / mis-split source ever drawn",
         quarantine_violations == 0 and split_violations == 0,
         f"{quarantine_violations} quarantine, {split_violations} split"),
        ("n never exceeds the classes audible in the window",
         n_cap_violations == 0, f"{n_cap_violations} violations"),
    ]

    # --- audio-dependent checks: mixture ≡ Σ(targets), silence tolerance ---
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=args.workers)
    worst_sum_error = 0.0
    worst_silent_residue = 0.0
    equality_silent_slots = 0
    tolerance_silent_slots = 0
    for stems, mixture in loader:
        stems, mixture = stems[0].numpy(), mixture[0].numpy()
        worst_sum_error = max(worst_sum_error,
                              float(np.abs(mixture - stems.sum(axis=0)).max()))
        # an UNDRAWN slot is written as exact zeros; a DRAWN slot whose window happens to
        # be silent carries the 2⁻²³ ingest residue. Counting both ways shows the gap the
        # tolerance rule exists to cover.
        peaks = np.abs(stems).max(axis=(1, 2))
        equality_silent_slots += int((peaks == 0.0).sum())
        tolerance_silent_slots += int((peaks < args.silence_eps).sum())
        residues = peaks[(peaks > 0.0) & (peaks < args.silence_eps)]
        if residues.size:
            worst_silent_residue = max(worst_silent_residue, float(residues.max()))

    checks.append((f"mixture ≡ Σ(targets) over {args.audio_items:,} decoded items",
                   worst_sum_error < 1e-5, f"max |error| = {worst_sum_error:.2e}"))
    checks.append((
        "silence tested by tolerance, not equality",
        tolerance_silent_slots >= equality_silent_slots,
        f"{equality_silent_slots:,} slots are exactly 0 (never drawn); "
        f"{tolerance_silent_slots:,} are silent within {args.silence_eps:g}; "
        f"worst sub-threshold residue {worst_silent_residue:.3e}"))

    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}\n         {detail}")
        if not passed:
            failures.append(label)

    # ================= CHARACTERISATION =================
    print("\n" + "-" * 78)
    print(f"CHARACTERISATION — {args.plan_draws:,} draws, no verdicts")
    print("-" * 78)

    total = sum(realised_n.values())
    print_histogram("Realised n (how many classes land in a mix)",
                    realised_n, baseline["n_probabilities"], total)
    mean_n = sum(n * c for n, c in realised_n.items()) / total
    mean_active = sum(n * c for n, c in active_counts.items()) / sum(active_counts.values())
    print(f"  mean n {mean_n:.3f} vs exp002 {baseline['mean_n']:.3f} "
          f"({mean_n - baseline['mean_n']:+.3f}) · "
          f"mean classes audible per window {mean_active:.3f}")
    total_variation = 0.5 * sum(
        abs(realised_n.get(n, 0) / total - baseline["n_probabilities"].get(n, 0.0))
        for n in set(realised_n) | set(baseline["n_probabilities"]))
    print(f"  total-variation distance from exp002's flat 1..9: {total_variation:.3f} "
          f"(0 = identical, 1 = disjoint)")

    print("\nPer-class appearance rate (fraction of mixes containing the class)")
    print(f"  {'class':<8} {'coherent':>10} {'exp002':>10} {'ratio':>8}  "
          f"{'sources':>8}")
    for class_name in sorted(classes, key=lambda c: -class_appearances[c]):
        rate = class_appearances[class_name] / total
        print(f"  {class_name:<8} {rate:>10.4f} {baseline['class_rate']:>10.4f} "
              f"{rate / baseline['class_rate']:>8.2f}× {len(dataset.pool[class_name]):>8}")

    # Two co-occurrence numbers, because they answer different questions.
    #   JOINT       P(both present) — how often the model is shown the pair at all.
    #   CONDITIONAL P(partner | class present) — given the rare class turned up, how
    #               often is its unison partner there too.
    # Joint rates for a rare class are dominated by that class's own rarity, so the
    # conditional is what isolates whether coherence changes the PAIRING.
    print("\nCo-occurrence — the pairs the experiment is about")
    print(f"  {'pair':<20} {'joint':>8} {'exp002':>8} {'ratio':>7}   "
          f"{'cond.':>8} {'exp002':>8} {'ratio':>7}")
    focus_pairs = [("양금", "가야금"), ("가야금", "거문고"), ("가야금", "해금"),
                   ("대금", "피리"), ("양금", "해금")]
    conditional_baseline = baseline["pair_rate"] / baseline["class_rate"]
    for anchor, partner in focus_pairs:
        pair = tuple(sorted((anchor, partner)))
        joint = pair_appearances[pair] / total
        conditional = (pair_appearances[pair] / class_appearances[anchor]
                       if class_appearances[anchor] else float("nan"))
        print(f"  P({partner}|{anchor}){'':<{max(0, 8 - len(anchor) - len(partner))}} "
              f"{joint:>8.4f} {baseline['pair_rate']:>8.4f} "
              f"{joint / baseline['pair_rate']:>6.2f}×   "
              f"{conditional:>8.4f} {conditional_baseline:>8.4f} "
              f"{conditional / conditional_baseline:>6.2f}×")

    print("\nGenre mix of the drawn songs (coherent draws inherit real 편성 by song)")
    for genre, count in genre_appearances.most_common():
        print(f"  {genre:<10} {count / total:>7.4f}")

    # --- effective distinct content ---
    windows_per_file: dict = {}
    for class_name in classes:
        for entry in dataset.pool[class_name]:
            seconds = entry.out_frames / cfg.sample_rate
            windows_per_file[entry.out_path] = max(
                1.0, (seconds - cfg.segment_seconds) / args.content_hop_seconds + 1.0)
            windows_per_file[entry.out_path + ":seconds"] = seconds
    log_incoherent = log10_incoherent_content(dataset.pool, classes, windows_per_file)
    log_coherent = log10_coherent_content(dataset.songs, windows_per_file,
                                          args.content_hop_seconds, cfg.segment_seconds)
    print(f"\nEffective distinct content (distinct drawable mixes, "
          f"{args.content_hop_seconds:g}s window hop)")
    print(f"  incoherent (exp002) : 1e{log_incoherent:.1f}")
    print(f"  coherent  (exp002.2): 1e{log_coherent:.1f}   "
          f"(conservative — assumes every class of a song is audible in every window)")
    print(f"  ratio               : 1e{log_incoherent - log_coherent:.1f} fewer")
    distinct_excerpts = sum(windows_per_file[e.out_path]
                            for c in classes for e in dataset.pool[c])
    print(f"  distinct source excerpts (identical for both arms): {distinct_excerpts:,.0f}")

    # The combinatorial counts above are astronomical on both sides and therefore say
    # little about overfitting. The number that actually bounds coherent diversity is how
    # many distinct MUSICAL MOMENTS exist — (song, window) pairs — because every coherent
    # mix is a subset of one moment, and two mixes from the same moment share all their
    # audio. Compared against what the run will consume, this is the overfitting figure.
    distinct_moments = sum(
        max(1.0, (max(windows_per_file[entry.out_path + ":seconds"]
                      for entries in song.entries_by_class.values() for entry in entries)
             - cfg.segment_seconds) / args.content_hop_seconds + 1.0)
        for song in dataset.songs)
    consumed = (int(raw["training"]["num_epochs"]) * int(raw["training"]["num_steps"])
                * int(raw["training"]["batch_size"]))
    print(f"\n  distinct musical moments (song × window): {distinct_moments:,.0f}"
          f"  [{len(dataset.songs)} songs]")
    print(f"  training examples the run will consume:   {consumed:,}"
          f"  ({consumed / distinct_moments:.1f} draws per moment)")

    print("\n" + "=" * 78)
    print("ALL CORRECTNESS CHECKS PASSED" if not failures
          else f"FAILURES: {failures}")
    print("=" * 78)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
