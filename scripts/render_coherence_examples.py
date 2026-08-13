"""render_coherence_examples.py — A/B listening material for the coherent-mixing arm.

For each example it writes a matched PAIR:
  · the coherent mix — n stems from ONE song at ONE time offset, plus each stem solo
  · an incoherent mix of the SAME classes, each stem pulled from a different random song
    at a different offset, drawn through the real incoherent code path

Both go through the identical −19 LUFS mixture normalisation, so the pair is level-matched
by construction and the only audible difference is whether the instruments are playing
together. That is the whole claim of exp002.2, and it should be obvious back to back:
the coherent mix should sound like an ensemble sharing one melodic line (heterophony —
near-unison with different ornamentation per instrument), the incoherent one like several
recordings playing over each other in unrelated modes and tempi.

Examples are spread across genres rather than taken in index order, since 판소리 and 산조
dominate the pool and would otherwise fill the whole set.

Run:
    uv run python scripts/render_coherence_examples.py
      --coherent-config configs/exp002.2_htdemucs_coherent.yaml
      --incoherent-config configs/exp002_htdemucs_v2_uniform_n.yaml
      --examples 10
      --out data/coherence_examples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402


# --- building the two mixes -------------------------------------------------
def render_coherent(dataset: GugakMixDataset, plan) -> tuple[np.ndarray, dict]:
    """Re-read one coherent plan into per-class audio (2, chunk) at its own gains."""
    return {class_name: dataset._read_window(entry, plan.start_frame) * np.float32(gain)
            for _slot, class_name, entry, gain in plan.picks}, {
        "song_id": plan.song.song_id, "genre": plan.song.genre_sub,
        "start_s": plan.start_frame / dataset.cfg.sample_rate}


def render_incoherent(dataset: GugakMixDataset, rng: np.random.Generator,
                      class_names: list, song_to_avoid: str,
                      song_by_path: dict) -> tuple[dict, list]:
    """One excerpt per class from independent random songs, via the real incoherent path.

    Redraws when a class happens to land on the coherent example's own song, so the
    contrast is never accidentally undermined.

    Args:
        dataset: an incoherent-config dataset (its pool is what gets drawn from).
        rng: Generator for the draw.
        class_names: the classes to fill — the coherent mix's class set.
        song_to_avoid: song_id the coherent mix came from.
        song_by_path: out_path → song_id, from the manifest.
    """
    stems, sources = {}, []
    for class_name in class_names:
        for _attempt in range(20):
            excerpt, entry = dataset._draw_excerpt(rng, class_name)
            if song_by_path.get(entry.out_path) != song_to_avoid:
                break
        stems[class_name] = dataset._augment_stem(rng, excerpt, class_name, entry)
        sources.append(f"{class_name}={song_by_path.get(entry.out_path, '?')}")
    return stems, sources


def normalize_like_training(dataset: GugakMixDataset,
                            stems: dict) -> tuple[np.ndarray, dict]:
    """Sum to a mixture and apply the training-time shared-gain normalisation.

    Reuses `GugakMixDataset._normalize`, so the rendered audio is levelled exactly the
    way the trainer's would be — no separate loudness path to drift out of sync.
    """
    stacked = np.stack(list(stems.values()))
    mixture = stacked.sum(axis=0)
    stacked, mixture = dataset._normalize(stacked, mixture)
    return mixture, dict(zip(stems, stacked))


# --- example selection ------------------------------------------------------
def pick_examples_across_genres(dataset: GugakMixDataset, wanted: int,
                                search_limit: int) -> list:
    """Scan item indices and keep coherent plans, round-robin over genres.

    The pool is 30% 판소리 and 23% 산조, so taking the first N indices would render an
    almost mono-genre set. This keeps at most ceil(wanted / genres) per genre, and
    prefers plans with at least two classes — a solo mix has no coherence to hear.
    """
    by_genre: dict = {}
    for index in range(search_limit):
        plan = dataset.plan_item(index)
        if plan is None or len(plan.picks) < 2:
            continue
        bucket = by_genre.setdefault(plan.song.genre_sub, [])
        if len(bucket) < 3:
            bucket.append(plan)

    chosen, round_index = [], 0
    while len(chosen) < wanted and round_index < 3:
        for genre in sorted(by_genre):
            if round_index < len(by_genre[genre]) and len(chosen) < wanted:
                chosen.append(by_genre[genre][round_index])
        round_index += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Render coherent/incoherent A/B pairs.")
    parser.add_argument("--coherent-config",
                        default="configs/exp002.2_htdemucs_coherent.yaml")
    parser.add_argument("--incoherent-config",
                        default="configs/exp002_htdemucs_v2_uniform_n.yaml")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--search-limit", type=int, default=400)
    parser.add_argument("--out", default="data/coherence_examples")
    args = parser.parse_args()

    # --- both datasets: same pool, different mixing mode ---
    coherent_cfg = MixDatasetConfig.from_mapping(
        yaml.safe_load((REPO_ROOT / args.coherent_config).read_text())["gugak_mix"])
    incoherent_cfg = MixDatasetConfig.from_mapping(
        yaml.safe_load((REPO_ROOT / args.incoherent_config).read_text())["gugak_mix"])
    coherent_ds = GugakMixDataset(coherent_cfg, REPO_ROOT, num_items=args.search_limit)
    incoherent_ds = GugakMixDataset(incoherent_cfg, REPO_ROOT, num_items=args.examples)

    manifest = pd.read_parquet(REPO_ROOT / coherent_cfg.source_manifest)
    song_by_path = dict(zip(manifest.out_path, manifest.song_id))

    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    sample_rate = coherent_cfg.sample_rate
    index_lines = ["# Coherent vs incoherent mixing — listening set",
                   "",
                   "Each folder holds one matched pair at identical loudness:",
                   "`coherent_mix.wav` (all stems from one song at one offset) and",
                   "`incoherent_mix.wav` (same classes, unrelated songs). Per-class stems",
                   "sit alongside each mix. Listen to the two mixes back to back.", ""]

    # --- render ---
    plans = pick_examples_across_genres(coherent_ds, args.examples, args.search_limit)
    rng = np.random.default_rng(coherent_cfg.seed)
    for number, plan in enumerate(plans, start=1):
        coherent_stems, info = render_coherent(coherent_ds, plan)
        class_names = list(coherent_stems)
        incoherent_stems, sources = render_incoherent(
            incoherent_ds, rng, class_names, info["song_id"], song_by_path)

        coherent_mix, coherent_stems = normalize_like_training(coherent_ds,
                                                               coherent_stems)
        incoherent_mix, incoherent_stems = normalize_like_training(incoherent_ds,
                                                                    incoherent_stems)

        folder = out_root / (f"{number:02d}_{info['genre']}_{info['song_id']}"
                             f"_at{info['start_s']:.0f}s_{len(class_names)}stem")
        folder.mkdir(exist_ok=True)
        soundfile.write(folder / "coherent_mix.wav", coherent_mix.T, sample_rate)
        soundfile.write(folder / "incoherent_mix.wav", incoherent_mix.T, sample_rate)
        for class_name, audio in coherent_stems.items():
            soundfile.write(folder / f"coherent_stem_{class_name}.wav",
                            audio.T, sample_rate)
        for class_name, audio in incoherent_stems.items():
            soundfile.write(folder / f"incoherent_stem_{class_name}.wav",
                            audio.T, sample_rate)

        index_lines += [
            f"## {folder.name}",
            f"- genre **{info['genre']}** · song `{info['song_id']}` · "
            f"offset {info['start_s']:.1f}s · {len(class_names)} classes",
            f"- classes: {' · '.join(class_names)}",
            f"- incoherent counterpart drew: {' · '.join(sources)}",
            ""]
        print(f"{folder.name}: {' · '.join(class_names)}")

    (out_root / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"\nrendered {len(plans)} pairs -> {out_root}")


if __name__ == "__main__":
    main()
