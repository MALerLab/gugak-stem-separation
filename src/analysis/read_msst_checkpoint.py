"""read_msst_checkpoint.py — unpack the metric history MSST stores inside a checkpoint.

MSST checkpoints carry their whole eval history (`all_metrics`: per epoch, per stem
class, one score per scored val song) — but the scores are bare lists with no song
identity attached. Their order follows `valid.py`'s `rglob("mixture.flac")` traversal of
the val tree (filesystem order, NOT sorted), and each class's list holds only the songs
whose folder contains that class's file (absent classes are skipped during eval).

This module recovers usable tables from that structure:

  eval_song_order    reproduce the traversal → song ids in scoring order
  build_per_song     one epoch's scores as (song_id, stem_class, score) rows,
                     count-asserted so a mismatch fails loudly instead of mislabelling
  build_trajectory   one row per eval cycle: per-class means + the overall average
  latest_epoch_key   the newest epoch key in an `all_metrics` dict

Nothing here computes metrics — it only reads what training already froze into the
checkpoint, which guarantees the numbers belong to that exact checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def eval_song_order(valid_root: Path) -> list[str]:
    """Song ids in the exact order MSST's valid.py scored them (rglob traversal, unsorted).

    Args:
        valid_root: Σstem val tree (<song>/mixture.<ext> folders).
    """
    return [p.parent.name for p in valid_root.rglob("mixture.flac")]


def latest_epoch_key(all_metrics: dict) -> str:
    """The highest-numbered epoch key (e.g. "epoch_42") in an MSST metric history."""
    return max(all_metrics, key=lambda key: int(key.split("_")[1]))


def build_per_song(all_metrics: dict, epoch_key: str, song_order: list[str],
                   valid_root: Path, metric: str = "si_sdr",
                   extension: str = "flac") -> pd.DataFrame:
    """Per-(song, class) scores at one epoch, with song identity recovered.

    Args:
        all_metrics: MSST's {epoch_key: {metric: {class: [per-song values]}}}.
        epoch_key: which epoch's scores to unpack (e.g. "epoch_42").
        song_order: song ids in eval traversal order (from `eval_song_order`).
        valid_root: Σstem val tree, used to test which songs hold each class.
        metric: which metric block to unpack.
        extension: audio extension of the val tree stems.
    """
    rows = []
    for stem_class, values in all_metrics[epoch_key][metric].items():
        scored = [song for song in song_order
                  if (valid_root / song / f"{stem_class}.{extension}").exists()]
        if len(scored) != len(values):
            raise ValueError(
                f"{stem_class}: {len(values)} scores vs {len(scored)} songs holding the "
                "class — traversal order or val tree changed; mapping unsafe")
        rows.extend({"song_id": song, "stem_class": stem_class, metric: float(v)}
                    for song, v in zip(scored, values))
    return pd.DataFrame(rows)


def build_trajectory(all_metrics: dict, steps_per_epoch: int,
                     metric: str = "si_sdr") -> pd.DataFrame:
    """One row per eval cycle: per-class means + the average that drives selection.

    Args:
        all_metrics: MSST metric history (see `build_per_song`).
        steps_per_epoch: optimizer steps between evals (for an x-axis in real units).
        metric: which metric block to summarize.
    """
    rows = []
    for key in sorted(all_metrics, key=lambda k: int(k.split("_")[1])):
        epoch = int(key.split("_")[1])
        per_class = all_metrics[key][metric]
        means = {c: float(np.mean(v)) for c, v in per_class.items()}
        rows.append({"epoch": epoch,
                     "optimizer_steps": (epoch + 1) * steps_per_epoch,
                     f"avg_{metric}": float(np.mean(list(means.values()))),
                     **{f"{metric}_{c}": m for c, m in means.items()}})
    return pd.DataFrame(rows)
