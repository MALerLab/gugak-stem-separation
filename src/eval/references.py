"""references.py — per-variant mixture + reference-target access for one eval song.

Two variants, one interface: each song object exposes `mixture_path` plus
`reference(stem_class)` returning a (channels, samples) float array — or None when the
song simply has no stem files for that class. Reference audio is loaded lazily, one
class at a time, so a full song's 9 references are never held in memory together.

sumstem — reads the prebuilt Σstem tree (`<root>/<split>/<song>/mixture.flac` +
  `<class>.flac`), the exact files training-time validation scored. No file = class
  absent from the song.

master — mixture is the publisher master from the ingest store; references are built
  on the fly from the ingested per-instrument stems (source_manifest rows): same-class
  stems summed, every stem trimmed to the song's shortest first (the 판소리 tail rule,
  identical to build_sumstem_eval), mono center-duplicated to stereo — but NO peak
  gain: the master and the raw stems stay at their native ingest-store scale, because
  the master-variant number deliberately measures the model against the actual
  recorded stems, mastering residual included. Absent-class references are never read
  from disk anywhere — the scorer constructs zeros in code.

Audio is read at soundfile's default float64, matching MSST's valid.py reads, so
metric arithmetic is bit-identical to the training-time pipeline (gate G5).

Master ↔ stem alignment (found 2026-09-05, scripts/master_lag_scan.py): in 19 of the
135 test songs the publisher master is a fixed integer number of samples late or early
relative to its own stems. When `master_songs` is given the `master_lag` table, each
MasterSong carries `lag_samples` and the shift is undone before scoring: a late master
(lag > 0) is read from its lag-th frame (`mixture_start_frame`, applied by the runner),
an early master (lag < 0) has the stems' first |lag| frames skipped instead. No table ⇒
lag 0 everywhere, i.e. the pre-2026-09-05 behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile


def _read_transposed(path: Path, frames: int = -1, start: int = 0) -> np.ndarray:
    """Read audio as (channels, samples) float64, mono expanded to one row."""
    audio, _ = soundfile.read(path, frames=frames, start=start, always_2d=True)
    return audio.T


@dataclass(frozen=True)
class SumstemSong:
    """One song of the prebuilt Σstem eval tree."""
    song_dir: Path
    extension: str

    @property
    def mixture_path(self) -> Path:
        return self.song_dir / f"mixture.{self.extension}"

    @property
    def mixture_start_frame(self) -> int:
        """Σstem mixture and references share one timeline by construction."""
        return 0

    def reference(self, stem_class: str) -> np.ndarray | None:
        """The class target as (channels, samples), or None when no file exists."""
        path = self.song_dir / f"{stem_class}.{self.extension}"
        if not path.exists():
            return None
        return _read_transposed(path)


@dataclass(frozen=True)
class MasterSong:
    """One song scored against the publisher master, references summed from the store."""
    master_path: Path
    # per class: [(stem audio path, stem frames), ...] from the source manifest
    stems_by_class: dict[str, list[tuple[Path, int]]]
    min_frames: int          # shortest stem of the song (trim-to-shortest, never pad)
    lag_samples: int = 0     # master delay vs the stems (master_lag table); 0 = aligned

    @property
    def mixture_path(self) -> Path:
        return self.master_path

    @property
    def mixture_start_frame(self) -> int:
        """First master frame to feed the model — skips a late master's leading lag."""
        return max(self.lag_samples, 0)

    def reference(self, stem_class: str) -> np.ndarray | None:
        """Sum of the class's ingested stems at native scale, or None if class absent."""
        stems = self.stems_by_class.get(stem_class)
        if not stems:
            return None
        # an early master (lag < 0) is aligned from the stem side instead
        start = max(-self.lag_samples, 0)
        target: np.ndarray | None = None
        for path, _ in stems:
            audio = _read_transposed(path, frames=self.min_frames - start, start=start)
            if audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)      # centered mono
            target = audio if target is None else target + audio
        return target


def sumstem_songs(sumstem_root: Path, split: str, song_ids: list[str],
                  extension: str) -> dict[str, SumstemSong]:
    """Σstem song objects for a split, existence-checked against the frozen song list.

    Args:
        sumstem_root: the tree root (model config `sumstem_eval.out_root`).
        split: val | test.
        song_ids: frozen song list from eval_manifest — the source of truth; the tree
            is only checked against it, never walked to define the song set.
        extension: audio extension of the tree.
    """
    songs = {}
    missing = []
    for song_id in song_ids:
        song = SumstemSong(song_dir=sumstem_root / split / song_id, extension=extension)
        if not song.mixture_path.exists():
            missing.append(song_id)
        songs[song_id] = song
    if missing:
        raise FileNotFoundError(
            f"Σstem tree {sumstem_root / split} is missing {len(missing)} of "
            f"{len(song_ids)} manifest songs (first: {missing[:3]}) — build it with "
            "src/data/build_sumstem_eval.py before evaluating")
    return songs


def master_songs(source_manifest: pd.DataFrame, split: str, song_ids: list[str],
                 classes: list[str], repo_root: Path,
                 lag_by_song: dict[str, int] | None = None) -> dict[str, MasterSong]:
    """Master-variant song objects for a split, resolved from the source manifest.

    Args:
        source_manifest: the full source_manifest table (dataloaders' one manifest).
        split: val | test.
        song_ids: frozen song list from eval_manifest.
        classes: modeled stem classes — stems outside them (quarantined classes) are
            excluded from the references, mirroring the Σstem build.
        repo_root: prefix for the manifest's repo-relative out_path values.
        lag_by_song: song_id -> lag_samples from the master_lag table. None = no
            alignment (lag 0). When given, every song must have an entry — a missing
            song means the table was scanned for a different split, so fail loudly.
    """
    subset = source_manifest[(source_manifest.dataset == "71955")
                             & (source_manifest.split == split)]
    masters = subset[subset.role == "master"].set_index("song_id")
    stems = subset[(subset.role != "master") & (subset.stem_group.isin(classes))]

    songs = {}
    for song_id in song_ids:
        if song_id not in masters.index:
            raise KeyError(f"{song_id}: no master row in source_manifest for split {split}")
        group = stems[stems.song_id == song_id]
        if group.empty:
            raise KeyError(f"{song_id}: no modeled-class stem rows in source_manifest")
        stems_by_class: dict[str, list[tuple[Path, int]]] = {}
        for row in group.itertuples():
            stems_by_class.setdefault(str(row.stem_group), []).append(
                (repo_root / str(row.out_path), int(row.out_frames)))
        if lag_by_song is not None and song_id not in lag_by_song:
            raise KeyError(f"{song_id}: no row in the master_lag table for split {split} "
                           "— run scripts/master_lag_scan.py for this split")
        songs[song_id] = MasterSong(
            master_path=repo_root / str(masters.loc[song_id, "out_path"]),
            stems_by_class=stems_by_class,
            min_frames=int(group.out_frames.min()),
            lag_samples=int(lag_by_song[song_id]) if lag_by_song is not None else 0)
    return songs
