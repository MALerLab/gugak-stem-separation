"""aug_listening_pick_songs.py — stage 2 of the augmentation listening test.

Picks the test-split master songs and their 30 s listening windows
(docs/aug_listening_test_spec.md, stage 2). d = augmented − baseline per-stem score on the
lag-aligned master eval (stage 1).

  1. Genre-typical: per genre, the song whose song-level d (mean over its present stems)
     is closest to the genre's median d. Its window covers as many stems as possible.
  2. Top-up: while a stem is covered by fewer than `min_songs_per_stem` picked windows,
     add the unpicked song whose per-stem d for that stem is closest to that stem's
     median d, with a window that covers the stem.

A stem counts as covered when it is active for at least `min_window_coverage` of the
window, from the chunk_activities table (activity scan of the true stems).

Run:
    uv run python scripts/aug_listening_pick_songs.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/aug_listening_test.yaml"


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths.

    Layout rule (repo-wide): parquet = source of truth in manifests/parquet/,
    csv = eyeball copy in manifests/csv/. Both dirs are created on demand.

    Args:
        base: table basename, e.g. Path(".../manifests/aug_listening_picks").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


def load_stem_differences(config: dict) -> pd.DataFrame:
    """Per (song, stem) scores of both models and their difference d, present stems only.

    Args:
        config: the parsed aug_listening_test config.
    """
    metric = config["pick"]["metric"]
    frames = {}
    for role in ("baseline", "augmented"):
        model = config["models"][role]
        path = REPO_ROOT / model["eval_dir"] / f"{model['eval_run_id']}.parquet"
        rows = pd.read_parquet(path)
        rows = rows[rows.status == "present"]
        frames[role] = rows.set_index(["song_id", "stem_class"])[[metric, "genre_sub"]]
    merged = frames["baseline"].join(frames["augmented"][[metric]], rsuffix="_augmented",
                                     how="inner")
    merged = merged.rename(columns={metric: "score_baseline",
                                    f"{metric}_augmented": "score_augmented"})
    merged["d"] = merged.score_augmented - merged.score_baseline
    return merged.reset_index()


def window_coverage(config: dict, song_ids: list[str]) -> pd.DataFrame:
    """Per-class activity coverage of every 30 s window (consecutive chunks) per song.

    Returns one row per (song_id, start_s, end_s) with one `cov_<class>` column per
    modeled class: the mean of the chunks' coverage over the window.

    Args:
        config: the parsed aug_listening_test config.
        song_ids: songs to compute windows for.
    """
    pick = config["pick"]
    chunk_len = pick["activity_chunk_len_s"]
    chunks_per_window = int(round(pick["window_s"] / chunk_len))
    chunks = pd.read_parquet(REPO_ROOT / pick["activity_table"])
    chunks = chunks[(chunks.split == "test") & (chunks.chunk_len_s == chunk_len)
                    & chunks.song_id.isin(song_ids)]
    cov_columns = [f"cov_{stem}" for stem in config["classes"]]

    windows = []
    for song_id, song_chunks in chunks.groupby("song_id"):
        song_chunks = song_chunks.sort_values("window_idx")
        coverage = song_chunks[cov_columns].to_numpy()
        starts = song_chunks.start_s.to_numpy()
        for first in range(len(song_chunks) - chunks_per_window + 1):
            row = dict(zip(cov_columns,
                           coverage[first:first + chunks_per_window].mean(axis=0)))
            row.update(song_id=song_id, start_s=float(starts[first]),
                       end_s=float(starts[first]) + pick["window_s"])
            windows.append(row)
    return pd.DataFrame(windows)


def best_window(song_windows: pd.DataFrame, present_stems: list[str],
                required_stem: str | None, min_coverage: float) -> pd.Series | None:
    """The window covering the most present stems, optionally required to cover one stem.

    Ranking: number of present stems covered, then their summed coverage.

    Args:
        song_windows: window_coverage rows of one song.
        present_stems: stems the song actually has.
        required_stem: a stem the window must cover (top-up picks), or None.
        min_coverage: coverage fraction that counts a stem as covered.
    """
    columns = [f"cov_{stem}" for stem in present_stems]
    candidates = song_windows
    if required_stem is not None:
        candidates = candidates[candidates[f"cov_{required_stem}"] >= min_coverage]
    if candidates.empty:
        return None
    num_covered = (candidates[columns] >= min_coverage).sum(axis=1)
    total_coverage = candidates[columns].sum(axis=1)
    ranking = pd.DataFrame({"num_covered": num_covered, "total": total_coverage,
                            "start_s": candidates.start_s})
    # earliest start breaks exact ties, so the pick is deterministic
    best_index = ranking.sort_values(["num_covered", "total", "start_s"],
                                     ascending=[False, False, True]).index[0]
    return candidates.loc[best_index]


def covered_stems(window: pd.Series, present_stems: list[str],
                  min_coverage: float) -> list[str]:
    """Present stems active for at least `min_coverage` of the window."""
    return [stem for stem in present_stems if window[f"cov_{stem}"] >= min_coverage]


def pick_row(song_id: str, genre: str, reason: str, target_stem: str | None,
             window: pd.Series, present_stems: list[str], song_d: float,
             target_d: float | None, min_coverage: float) -> dict:
    """One output row for a picked song."""
    return {"song_id": song_id, "genre_sub": genre, "pick_reason": reason,
            "target_stem": target_stem, "start_s": window.start_s, "end_s": window.end_s,
            "covered_stems": ",".join(covered_stems(window, present_stems, min_coverage)),
            "present_stems": ",".join(present_stems), "song_d": song_d,
            "target_stem_d": target_d}


def pick_genre_typical(differences: pd.DataFrame, windows: pd.DataFrame,
                       min_coverage: float) -> list[dict]:
    """Stage 2 step 1: per genre, the song whose song-level d is closest to the median."""
    song_level = differences.groupby(["song_id", "genre_sub"]).d.mean().reset_index()
    picks = []
    for genre, genre_songs in song_level.groupby("genre_sub"):
        median = genre_songs.d.median()
        ordered = genre_songs.assign(distance=(genre_songs.d - median).abs())
        song = ordered.sort_values(["distance", "song_id"]).iloc[0]
        present = sorted(differences[differences.song_id == song.song_id].stem_class)
        window = best_window(windows[windows.song_id == song.song_id], present, None,
                             min_coverage)
        picks.append(pick_row(song.song_id, genre, "genre_typical", None, window, present,
                              song.d, None, min_coverage))
    return picks


def stem_counts(picks: list[dict], classes: list[str]) -> dict[str, int]:
    """How many picked windows cover each stem."""
    counts = {stem: 0 for stem in classes}
    for pick in picks:
        for stem in filter(None, pick["covered_stems"].split(",")):
            counts[stem] += 1
    return counts


def top_up_stems(picks: list[dict], differences: pd.DataFrame, windows: pd.DataFrame,
                 classes: list[str], min_songs: int, min_coverage: float) -> list[dict]:
    """Stage 2 step 2: add stem-typical songs until every stem has `min_songs` windows."""
    song_level = differences.groupby("song_id").d.mean()
    while True:
        counts = stem_counts(picks, classes)
        short = [stem for stem in classes if counts[stem] < min_songs]
        if not short:
            return picks
        # scarcest stem first, so one added song is never spent on a commoner stem
        stem = min(short, key=lambda name: (counts[name], name))
        picked_ids = {pick["song_id"] for pick in picks}
        stem_rows = differences[differences.stem_class == stem]
        median = stem_rows.d.median()
        candidates = stem_rows[~stem_rows.song_id.isin(picked_ids)]
        candidates = candidates.assign(distance=(candidates.d - median).abs())
        added = False
        for row in candidates.sort_values(["distance", "song_id"]).itertuples():
            present = sorted(differences[differences.song_id == row.song_id].stem_class)
            window = best_window(windows[windows.song_id == row.song_id], present, stem,
                                 min_coverage)
            if window is None:
                continue
            picks.append(pick_row(row.song_id, row.genre_sub, f"top_up_{stem}", stem,
                                  window, present, song_level[row.song_id], row.d,
                                  min_coverage))
            added = True
            break
        if not added:
            raise RuntimeError(f"no unpicked test song has a window covering {stem}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pick = config["pick"]

    # per-stem differences + window coverage for every scored test song
    differences = load_stem_differences(config)
    windows = window_coverage(config, sorted(differences.song_id.unique()))

    # genre-typical picks, then top-ups until every stem is covered enough
    picks = pick_genre_typical(differences, windows, pick["min_window_coverage"])
    picks = top_up_stems(picks, differences, windows, config["classes"],
                         pick["min_songs_per_stem"], pick["min_window_coverage"])

    # write the picks table + report coverage
    table = pd.DataFrame(picks)
    parquet_path, csv_path = table_paths(REPO_ROOT / "manifests" / pick["out_table"])
    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)
    print(table[["song_id", "pick_reason", "start_s", "covered_stems", "song_d"]]
          .to_string(index=False))
    print("\nwindows covering each stem:", stem_counts(picks, config["classes"]))
    print(f"wrote {len(table)} picks -> {parquet_path}")


if __name__ == "__main__":
    main()
