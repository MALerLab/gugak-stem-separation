"""mix_dataset.py — on-the-fly training mixes (Training Data Strategy recipe).

Two mixing modes, selected by `coherent_mix_prob` (0.0 = fully incoherent, the default
and everything up to exp002; 1.0 = fully coherent, exp002.2; values between mix the two
per item).

INCOHERENT (the base recipe). Draw HOW MANY classes go in the mix (density — uniform
over 1..n_classes, or from the measured audible-편성 distribution of real songs, per
`density_mode`), draw WHICH classes (uniform), draw one activity-aware excerpt per class
from the ingest store — each from a DIFFERENT random song at a DIFFERENT random offset.

COHERENT (exp002.2). Draw one song, one time window in it, and take every selected stem
from THAT song at THAT offset, so the instruments are playing together as recorded.
Motivation: gugak ensembles are heterophonic — instruments play near-unison variants of
one melodic line — so an incoherent mix (가야금 from song A over 거문고 from song B) is a
much easier separation problem than the real mixtures val and test are built from. This
mode trains on the hard case. See `_coherent_item` for the draw, and note the two
consequences it carries by construction: class frequency stops being uniform and starts
following real 편성 (deliberate, not corrected for), and n is capped by how many classes
are actually audible in the drawn window.

Both modes then apply live augmentations, sum to a mixture, and loudness-normalize
mixture and targets by the same gain. Returns (stems, mixture) float32 tensors shaped
[n_classes, 2, chunk] / [2, chunk] — the exact batch contract MSST's trainer consumes,
so this class drops into its DataLoader.

Everything is manifest-driven (source_manifest ⋈ activity_segments ⋈ chunk_activities);
no directory walking, and no audio is decoded at init. Every knob lives in the
experiment YAML's `gugak_mix` block — including RESERVED keys for features that are
designed but deliberately not implemented yet (타악-2× multi-sampling, the pitch-shift
pool, song-base draw units). Setting one of those raises NotImplementedError loudly
rather than silently ignoring it. The nested
`loudness_match` sub-block (→ src/data/loudness_match.py) re-levels solo-pool stems onto
the ensemble loudness distribution; absent or disabled, every stem keeps the stock
random gain and the RNG stream is unchanged.

In measured mode, density is recomputed at init from the per-class coverage columns for
the CONFIGURED class set — never read from the precomputed n_active_gt* columns, which
count all 11 taxonomy groups (an experiment that models fewer classes would inherit
phantom counts). Uniform mode reads no table at all.

Run standalone (smoke test): see scripts/smoke_mix_dataset.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyloudnorm
import soundfile
import torch
from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter, Pedalboard

try:    # imported as a package module (MSST hook: src.data.mix_dataset)
    from src.data.loudness_match import LoudnessMatchConfig, LoudnessTargetSampler
except ModuleNotFoundError:   # imported as a sibling (build_sumstem_eval.py runs so)
    from loudness_match import LoudnessMatchConfig, LoudnessTargetSampler


# --- config -----------------------------------------------------------------
@dataclass
class MixDatasetConfig:
    """The `gugak_mix` block of an experiment YAML (paths repo-relative)."""
    # class scheme: list order = output tensor slot order
    classes: list
    # source pool filters
    datasets: list = field(default_factory=lambda: ["71955"])
    split: str = "train"
    # excerpt geometry
    segment_seconds: float = 10.0
    sample_rate: int = 44100
    # density draw — HOW MANY classes go in a mix.
    #   "measured": n ~ the distribution of how many classes are audible in a random
    #               window of a real song (audible = coverage > threshold in a window of
    #               density_chunk_len_s). exp001's mode.
    #   "uniform":  n ~ uniform over 1..len(classes). exp002 onward — we cannot justify
    #               matching real 편성 statistics when class IDENTITY is already uniform,
    #               so the realism was half-hearted either way (prof, 2026-08-03).
    # The two chunk knobs below are read only in "measured" mode.
    density_mode: str = "measured"
    density_chunk_len_s: float = 10.0
    density_coverage_threshold: float = 0.25
    # coherent mixing — probability that an item is drawn from ONE song at ONE offset
    # instead of from independent songs per class. 0.0 = the incoherent base recipe
    # (default; consumes no randomness, so the draw stream is bit-identical to a run
    # that predates this feature). 1.0 = exp002.2. Under coherent draws n is uniform
    # over 1..(classes audible in the window), so `density_mode` is not consulted.
    coherent_mix_prob: float = 0.0
    # live augmentation knobs (exp001: EQ probs 0.0 — built, off)
    gain_min: float = 0.25
    gain_max: float = 1.25
    # L/R swap. Applied PER STEM in incoherent mixes (the stems are unrelated anyway)
    # and PER MIX in coherent ones — swapping a real ensemble's stems independently
    # would scramble its spatial image, which is part of what makes it coherent.
    channel_swap_prob: float = 0.5
    eq_stem_prob: float = 0.0
    eq_mixbus_prob: float = 0.0
    eq_stem_gain_db: float = 9.0
    eq_mixbus_gain_db: float = 6.0
    # solo-pool loudness pre-conditioning (nested block → LoudnessMatchConfig);
    # absent/empty = disabled = every stem keeps the random-gain treatment
    loudness_match: dict = field(default_factory=dict)
    # mixture normalization: loudnorm mixture+targets by one shared gain, then peak-guard
    target_lufs: float = -19.0
    peak_ceiling: float = 0.99
    # manifests (source of truth — never walk directories)
    source_manifest: str = "manifests/parquet/source_manifest.parquet"
    activity_segments: str = "manifests/parquet/activity_segments.parquet"
    chunk_activities: str = "manifests/parquet/chunk_activities.parquet"
    # reproducibility
    seed: int = 42
    # --- RESERVED knobs (designed, not implemented — nonzero/non-default raises) ---
    multi_sample: dict = field(default_factory=dict)   # {class: n_draws}, 타악-2× seam
    pitch_pool_manifest: str | None = None             # (source × semitone) pool table
    draw_unit: str = "file"                            # "song_base" = summed same-base

    @classmethod
    def from_mapping(cls, mapping: dict) -> "MixDatasetConfig":
        """Build from a plain dict (yaml block); unknown keys error loudly."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(mapping) - known
        if unknown:
            raise KeyError(f"unknown gugak_mix config keys: {sorted(unknown)}")
        return cls(**mapping)


# --- pool entry -------------------------------------------------------------
class SourceEntry(NamedTuple):
    """One drawable source file: where it is, how long, when it plays, where it's from.

    `dataset`, `source_lufs` and `source_channels` exist for loudness matching — the
    treatment applies to solo-pool stems only, and needs the clip's measured level plus
    how it is presented (a mono file is duplicated to stereo on draw, which reads 3 dB
    louder than the file scan's 1-channel measurement) to compute its offset.
    `source_lufs` is nan whenever matching is off or the file has no measurement.
    """
    out_path: str
    out_frames: int
    segments: np.ndarray
    dataset: str
    source_lufs: float
    source_channels: int


# --- song entry (coherent draws only) ---------------------------------------
class SongEntry(NamedTuple):
    """One drawable song: its stems grouped by class, plus the pooled activity it shows.

    `joint_segments` is every active segment of every drawable stem of the song,
    concatenated — the JOINT activity criterion the coherent window draw runs on. Times
    where several instruments play contribute several overlapping segments, so a
    duration-weighted draw over this pool lands on busy passages more often than on
    passages where one instrument is noodling alone. That is the intended behaviour: it
    is the same activity-aware logic the incoherent path applies per stem, evaluated
    across the song instead.

    `max_frames` is the LONGEST stem, not the shortest. The trim-to-shortest rule exists
    to make Σstems line up with the publisher master; no master is involved here, so a
    window in the tail of a long 판소리 가야금 stem is legitimate content and is kept.
    Stems that have already ended simply read as zeros there.
    """
    song_id: str
    genre_sub: str
    entries_by_class: dict
    joint_segments: np.ndarray
    max_frames: int


class CoherentPlan(NamedTuple):
    """Everything one coherent mix does, decided before any audio is read.

    `picks` is [(output slot, class name, source entry, linear gain)] and `start_frame`
    applies to ALL of them — the temporal alignment that makes the mix coherent.
    `active_classes` is what the window offered, so the gap between it and `picks`
    records how much of the ensemble the n-draw left out.
    """
    song: "SongEntry"
    start_frame: int
    active_classes: list
    picks: list
    swap_channels: bool
    eq_boards: list


# --- EQ augmentation (reimplemented from the Embracing Cacophony recipe) -----
def build_random_eq(rng: np.random.Generator, max_gain_db: float,
                    q_max: float) -> Pedalboard:
    """Random EQ chain: low shelf + 0–4 peak filters + high shelf.

    Frequencies are log-spaced with gaussian jitter, gains uniform in ±max_gain_db,
    Q log-uniform in [0.7, q_max] — the parameterization from the Embracing Cacophony
    paper (per-stem: ±9 dB / Q≤5; mixbus: ±6 dB / Q≤3).

    Args:
        rng: numpy Generator driving every random choice.
        max_gain_db: symmetric gain range for every filter.
        q_max: upper bound of the log-uniform Q distribution.
    """
    q_min = 0.7

    def random_q() -> float:
        return ((q_max / q_min) ** rng.random()) * q_min

    log_freq_low = rng.uniform(math.log10(50.0), math.log10(150.0))
    log_freq_high = rng.uniform(math.log10(6000.0), math.log10(12000.0))
    board = Pedalboard([])
    board.append(LowShelfFilter(cutoff_frequency_hz=10 ** log_freq_low,
                                gain_db=rng.uniform(-max_gain_db, max_gain_db),
                                q=random_q()))
    n_peaks = rng.integers(0, 5)
    if n_peaks > 0:
        boundaries = np.logspace(log_freq_low, log_freq_high, num=n_peaks + 1, base=10)
        centers = (boundaries[:-1] + boundaries[1:]) / 2
        for center in centers:
            freq = float(np.clip(rng.normal(center, center ** 0.75), 50.0, 10000.0))
            board.append(PeakFilter(cutoff_frequency_hz=freq,
                                    gain_db=rng.uniform(-max_gain_db, max_gain_db),
                                    q=random_q()))
    board.append(HighShelfFilter(cutoff_frequency_hz=10 ** log_freq_high,
                                 gain_db=rng.uniform(-max_gain_db, max_gain_db),
                                 q=random_q()))
    return board


# --- dataset ----------------------------------------------------------------
class GugakMixDataset(torch.utils.data.Dataset):
    """Incoherent-mix dataset over the ingest store, manifest-driven.

    Args:
        cfg: the experiment's `gugak_mix` block.
        repo_root: absolute repo root all manifest/audio paths resolve against.
        num_items: dataset length (MSST semantics: num_steps × batch_size).
    """

    def __init__(self, cfg: MixDatasetConfig, repo_root: Path, num_items: int) -> None:
        self.cfg = cfg
        self.root = Path(repo_root)
        self.num_items = int(num_items)
        self._reject_unimplemented(cfg)

        self.chunk_frames = int(round(cfg.segment_seconds * cfg.sample_rate))
        self.meter = pyloudnorm.Meter(cfg.sample_rate)

        # solo-pool loudness matching: built before the pool, which asks it for levels
        self.loudness_match_cfg = LoudnessMatchConfig.from_mapping(cfg.loudness_match)
        self.loudness_sampler = (
            LoudnessTargetSampler(self.loudness_match_cfg, self.root, list(cfg.classes),
                                  cfg.segment_seconds)
            if self.loudness_match_cfg.enabled else None)

        self.pool = self._build_source_pool()
        self.density_values, self.density_probs = self._build_density_histogram()
        # the song pool costs a manifest regroup, so it is built only when it can be
        # drawn from — a pure-incoherent run pays nothing for the feature existing
        self.songs = (self._build_song_pool() if cfg.coherent_mix_prob > 0.0 else [])

    # --- reserved-knob guard ---
    @staticmethod
    def _reject_unimplemented(cfg: MixDatasetConfig) -> None:
        """Reserved config keys exist so the vocabulary is stable; using one fails loudly."""
        if not 0.0 <= cfg.coherent_mix_prob <= 1.0:
            raise ValueError(f"coherent_mix_prob={cfg.coherent_mix_prob}: expected [0, 1]")
        if any(int(n) > 1 for n in cfg.multi_sample.values()):
            raise NotImplementedError(
                "multi_sample > 1: 타악-2×-style multi-sampling is a reserved seam")
        if cfg.pitch_pool_manifest is not None:
            raise NotImplementedError(
                "pitch_pool_manifest: the pitch-shift pool is deferred post-exp001")
        if cfg.draw_unit != "file":
            raise NotImplementedError(
                f"draw_unit={cfg.draw_unit!r}: only 'file' (individual tracks) exists; "
                "'song_base' (summed same-base stems) is a reserved seam")

    # --- init-time table work (no audio) ---
    def _load_pool_tables(self) -> tuple[pd.DataFrame, dict]:
        """The two manifest reads both pools need: drawable sources + their activity.

        Kept separate so the per-class pool and the per-song pool are built from ONE
        read of each table and cannot drift apart in their filtering.
        """
        manifest = pd.read_parquet(self.root / self.cfg.source_manifest)
        sources = manifest[(manifest.dataset.isin(self.cfg.datasets))
                           & (manifest.split == self.cfg.split)
                           & (manifest.role != "master")
                           & (manifest.stem_group.isin(self.cfg.classes))]
        segments = pd.read_parquet(self.root / self.cfg.activity_segments)
        segments_by_file = {fid: grp[["start_s", "end_s"]].to_numpy()
                            for fid, grp in segments.groupby("file_id")}
        return sources, segments_by_file

    def _make_entry(self, row) -> SourceEntry | None:
        """One manifest row → a drawable SourceEntry, or None if it has no activity."""
        file_segments = self._segments_by_file.get(row.file_id)
        if file_segments is None or len(file_segments) == 0:
            return None         # fully-silent file: nothing to draw (QC says none exist)
        source_lufs = (self.loudness_sampler.source_loudness(row.file_id)
                       if self.loudness_sampler is not None else math.nan)
        return SourceEntry(row.out_path, int(row.out_frames), file_segments,
                           row.dataset, source_lufs, int(row.out_channels))

    def _build_source_pool(self) -> dict:
        """Per-class draw lists: (paths, frame counts, active segments) from the manifests."""
        self._sources, self._segments_by_file = self._load_pool_tables()

        pool: dict = {}
        for class_name, group in self._sources.groupby("stem_group"):
            entries = [entry for entry in
                       (self._make_entry(row) for row in group.itertuples())
                       if entry is not None]
            pool[class_name] = entries

        missing = [c for c in self.cfg.classes if not pool.get(c)]
        if missing:
            raise ValueError(f"no drawable sources for classes {missing} "
                             f"(datasets={self.cfg.datasets}, split={self.cfg.split})")
        return pool

    def _build_song_pool(self) -> list:
        """Per-song draw list for coherent mixes — the same sources, regrouped by song.

        Only song-keyed rows qualify: the 71470 solo clips are standalone phrases with no
        song to be coherent WITH (`song_id` is null for them), so they cannot take part in
        a coherent draw and are dropped here rather than silently mis-grouped. A run that
        wants both a solo pool and coherent mixes has to say what that means first.

        Songs contributing only one drawable stem are kept: a coherent 1-stem mix is a
        real solo passage, and the incoherent path keeps n=1 for the same reason.
        """
        song_rows = self._sources[self._sources.song_id.notna()]
        dropped = self._sources.song_id.isna().sum()
        if dropped and self.cfg.coherent_mix_prob > 0.0:
            datasets = sorted(self._sources[self._sources.song_id.isna()].dataset.unique())
            raise ValueError(
                f"coherent_mix_prob>0 but {dropped} drawable sources from datasets "
                f"{datasets} have no song_id — coherent draws need songs. Restrict "
                "gugak_mix.datasets to song-keyed sets, or set coherent_mix_prob: 0.0")

        songs = []
        for song_id, group in song_rows.groupby("song_id"):
            entries_by_class: dict = {}
            for row in group.itertuples():
                entry = self._make_entry(row)
                if entry is not None:
                    entries_by_class.setdefault(row.stem_group, []).append(entry)
            if not entries_by_class:
                continue        # every stem silent (QC says this does not happen)
            all_entries = [e for entries in entries_by_class.values() for e in entries]
            songs.append(SongEntry(
                song_id=str(song_id),
                genre_sub=str(group.genre_sub.iloc[0]),
                entries_by_class=entries_by_class,
                joint_segments=np.concatenate([e.segments for e in all_entries]),
                max_frames=max(e.out_frames for e in all_entries)))

        if not songs:
            raise ValueError("coherent_mix_prob>0 but no drawable songs "
                             f"(datasets={self.cfg.datasets}, split={self.cfg.split})")
        return songs

    def _build_density_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """The distribution n is drawn from — (values, probabilities) over class counts.

        Dispatches on cfg.density_mode. Both modes return the same shape, so __getitem__
        is mode-agnostic and the RNG consumes exactly one draw either way — an experiment
        can switch modes without shifting the random stream's structure.
        """
        if self.cfg.density_mode == "uniform":
            return self._uniform_density()
        if self.cfg.density_mode == "measured":
            return self._measured_density()
        raise ValueError(f"density_mode={self.cfg.density_mode!r}: expected "
                         "'uniform' or 'measured'")

    def _uniform_density(self) -> tuple[np.ndarray, np.ndarray]:
        """n uniform over 1..len(classes) — every mix size equally likely.

        n=0 is excluded for the same reason the measured mode drops it: an all-silent
        mixture teaches nothing. The upper end is the class count, since classes are
        drawn without replacement.

        This mode reads no manifest — chunk_activities is a measured-mode input only.
        """
        values = np.arange(1, len(self.cfg.classes) + 1)
        return values, np.full(len(values), 1.0 / len(values))

    def _measured_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Audible-class-count distribution recomputed for the configured class set.

        Counts per window how many of cfg.classes exceed the coverage threshold —
        deliberately NOT the precomputed n_active_gt* columns (11-group counts).
        n=0 windows are dropped (all-silent mixes teach nothing); n=1 stays (real
        solo passages, ~2% — easy anchor examples).
        """
        chunks = pd.read_parquet(self.root / self.cfg.chunk_activities)
        chunks = chunks[(chunks.split == self.cfg.split)
                        & (chunks.chunk_len_s == self.cfg.density_chunk_len_s)]
        if chunks.empty:
            available = sorted(pd.read_parquet(
                self.root / self.cfg.chunk_activities).chunk_len_s.unique())
            raise ValueError(
                f"no chunk_activities rows at chunk_len_s={self.cfg.density_chunk_len_s} "
                f"(available: {available}) — add the length to configs/activity_scan.yaml "
                "and re-run stage 2 (seconds of compute)")

        coverage_columns = [f"cov_{c}" for c in self.cfg.classes]
        absent = [c for c in coverage_columns if c not in chunks.columns]
        if absent:
            raise ValueError(f"chunk_activities lacks coverage columns {absent}")

        audible_counts = (chunks[coverage_columns].to_numpy()
                          > self.cfg.density_coverage_threshold).sum(axis=1)
        audible_counts = audible_counts[audible_counts >= 1]
        values, counts = np.unique(audible_counts, return_counts=True)
        return values, counts / counts.sum()

    # --- per-item sampling (audio) ---
    def _draw_excerpt(self, rng: np.random.Generator,
                      class_name: str) -> tuple[np.ndarray, SourceEntry]:
        """One activity-aware excerpt of a random source of `class_name` → (2, chunk).

        The incoherent draw: a random file of the class, then a random activity-anchored
        window inside it (→ `_draw_window_start`). The entry it came from is returned
        alongside, since the gain treatment depends on which pool that is.
        """
        entry = self.pool[class_name][rng.integers(len(self.pool[class_name]))]
        start_frame = self._draw_window_start(rng, entry.segments, entry.out_frames)
        return self._read_window(entry, start_frame), entry

    def _draw_window_start(self, rng: np.random.Generator, segments: np.ndarray,
                           total_frames: int) -> int:
        """Activity-aware window start, in frames, from a pool of active segments.

        One segment is chosen duration-weighted, an anchor point drawn inside it, and the
        window placed uniformly at random over positions containing the anchor — so every
        window overlaps real activity, but silence around short segments stays in
        (silence is a valid signal when deliberate).

        The incoherent path passes ONE stem's segments; the coherent path passes the
        pooled segments of a whole song. Identical arithmetic either way, which is what
        makes "the same activity-aware logic, evaluated jointly" literally true.

        Args:
            rng: the item's Generator.
            segments: (n, 2) array of [start_s, end_s] active spans.
            total_frames: length of the source (or, for a song, its longest stem).
        """
        durations = segments[:, 1] - segments[:, 0]
        segment = segments[rng.choice(len(segments), p=durations / durations.sum())]
        anchor_s = rng.uniform(segment[0], segment[1])
        start_s = anchor_s - rng.uniform(0.0, self.cfg.segment_seconds)
        max_start = max(0, total_frames - self.chunk_frames)
        return int(np.clip(round(start_s * self.cfg.sample_rate), 0, max_start))

    def _read_window(self, entry: SourceEntry, start_frame: int) -> np.ndarray:
        """Read one (2, chunk) excerpt at a given frame offset.

        Sources shorter than the window are zero-padded at the tail; mono sources are
        center-duplicated to stereo (never naive-summed — anti-phase rule). A start
        beyond the file's end returns silence outright: in a coherent draw every stem
        reads the SAME offset, and a stem that has already finished genuinely has no
        content there.
        """
        if start_frame >= entry.out_frames:
            return np.zeros((2, self.chunk_frames), dtype=np.float32)
        audio, sample_rate = soundfile.read(
            self.root / entry.out_path, start=start_frame, frames=self.chunk_frames,
            dtype="float32", always_2d=True, fill_value=0.0)   # fill pads short reads
        if sample_rate != self.cfg.sample_rate:
            raise ValueError(f"{entry.out_path}: sr {sample_rate} != "
                             f"{self.cfg.sample_rate} (ingest store contract broken)")
        audio = audio.T                                        # -> (channels, frames)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)                # centered mono
        return audio

    def _draw_stem_gain(self, rng: np.random.Generator, class_name: str,
                        entry: SourceEntry) -> float:
        """Linear gain for one stem: loudness matching and random gain, COMPOSED.

        Two different jobs, applied in order. Matching is pool pre-conditioning: a stem
        from a matched pool (the 71470 solo clips) is first moved onto a level drawn from
        the class's ensemble distribution, so the two pools' level distributions look
        alike. The random gain is then augmentation, applied to EVERY stem regardless of
        pool. Replacing rather than composing — the earlier design — would hand the
        augmentation to ensemble stems only, and since the configured range is not
        centred (0.25–1.25 linear averages about −3.2 dB) that alone would push ensemble
        stems systematically quieter: a fresh pool-correlated loudness cue of exactly the
        kind matching exists to remove.

        The range deliberately stays 0.25–1.25 rather than being re-centred, so exp002
        differs from exp001 only in the solo pool and the comparison stays single-
        variable. The asymmetry is harmless once every stem sees it equally.

        With matching disabled the first branch is skipped entirely and no extra
        randomness is consumed, so the RNG stream is identical to a stock run.

        Args:
            rng: the item's Generator.
            class_name: the stem class being drawn.
            entry: the pool entry the excerpt came from (its pool and measured level).
        """
        matched_gain = 1.0
        if (self.loudness_sampler is not None
                and entry.dataset in self.loudness_match_cfg.matched_datasets):
            drawn = self.loudness_sampler.draw_gain(rng, class_name, entry.dataset,
                                                    entry.source_lufs,
                                                    entry.source_channels)
            if drawn is not None:
                matched_gain = drawn
        return matched_gain * float(rng.uniform(self.cfg.gain_min, self.cfg.gain_max))

    def _augment_stem(self, rng: np.random.Generator, audio: np.ndarray,
                      class_name: str, entry: SourceEntry) -> np.ndarray:
        """Live per-stem chain: gain (random or loudness-matched) · L/R swap · EQ.

        Args:
            rng: the item's Generator.
            audio: the drawn excerpt, (2, chunk).
            class_name: the stem class being drawn.
            entry: the pool entry the excerpt came from.
        """
        # np.float32 cast: a float64 scalar would silently promote the whole chain
        audio = audio * np.float32(self._draw_stem_gain(rng, class_name, entry))
        if rng.random() < self.cfg.channel_swap_prob:
            audio = audio[::-1].copy()
        if rng.random() < self.cfg.eq_stem_prob:
            board = build_random_eq(rng, self.cfg.eq_stem_gain_db, q_max=5.0)
            audio = board(audio, self.cfg.sample_rate)
        return audio

    def _normalize(self, stems: np.ndarray,
                   mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Loudnorm mixture AND targets by one shared gain, then peak-guard both.

        Normalizing the mixture (never per-stem) keeps stem balance intact; applying
        the identical gain to the targets keeps mixture ≡ Σ(targets). The -inf guard
        covers near-silent draws (a lone sparse stem's quiet window).
        """
        loudness = self.meter.integrated_loudness(mixture.T)
        if not math.isinf(loudness):
            gain = np.float32(10 ** ((self.cfg.target_lufs - loudness) / 20))
            stems, mixture = stems * gain, mixture * gain
        peak = float(np.abs(mixture).max())
        if peak > self.cfg.peak_ceiling:
            scale = np.float32(self.cfg.peak_ceiling / peak)
            stems, mixture = stems * scale, mixture * scale
        return stems, mixture

    # --- coherent draw (one song, one offset) ---
    @staticmethod
    def _window_coverage(segments: np.ndarray, window_start_s: float,
                         window_end_s: float) -> float:
        """Fraction of the window this source's active segments cover, in [0, 1]."""
        overlap = np.minimum(segments[:, 1], window_end_s) - np.maximum(segments[:, 0],
                                                                       window_start_s)
        return float(np.clip(overlap, 0.0, None).sum()) / (window_end_s - window_start_s)

    def _active_classes_in_window(self, song: SongEntry,
                                  start_frame: int) -> dict:
        """Which of the song's classes are audible in this window → {class: [entries]}.

        "Audible" reuses the project's existing definition — activity covering more than
        `density_coverage_threshold` of the window, the same test `chunk_activities` and
        the measured density mode apply. A class qualifies if at least one of its files
        clears the bar, and only the files that clear it are drawable, so a selected
        class is always one that actually plays here.

        Fallback: if nothing clears the threshold (possible — the window is anchored on
        activity, but a short segment can cover less than a quarter of 10 s), the single
        best-covered file is used. That keeps every mix non-silent, matching the standing
        "skip n=0, keep n=1" rule, rather than emitting a training example of nothing.
        """
        window_start_s = start_frame / self.cfg.sample_rate
        window_end_s = window_start_s + self.cfg.segment_seconds

        active: dict = {}
        best_class, best_entry, best_coverage = None, None, -1.0
        for class_name, entries in song.entries_by_class.items():
            for entry in entries:
                coverage = self._window_coverage(entry.segments, window_start_s,
                                                 window_end_s)
                if coverage > self.cfg.density_coverage_threshold:
                    active.setdefault(class_name, []).append(entry)
                if coverage > best_coverage:
                    best_class, best_entry, best_coverage = class_name, entry, coverage
        if not active:
            active = {best_class: [best_entry]}
        return active

    def _plan_coherent(self, rng: np.random.Generator) -> CoherentPlan:
        """Decide everything about one coherent mix, without touching a single audio file.

        The draw, in order: song uniform over the train songs · window from the song's
        JOINT activity · which classes are audible there · n uniform over
        1..len(audible) · that many classes without replacement · one file per class ·
        per-stem gain · one per-mix L/R swap decision · per-stem EQ.

        Split out from the audio work so that verification and characterisation can pull
        hundreds of thousands of draws through the REAL sampling code rather than a
        reimplementation of it — the statistics reported for this arm describe the
        sampler that trains it, by construction (→ scripts/verify_coherent_sampler.py).

        Two properties this deliberately does NOT have. Class frequency is not uniform —
        it follows real 편성, so 해금 appears far more than 양금; correcting for it would
        undo the coherence. And n does not follow exp002's flat 1..9 — songs carrying
        few classes cap it low, so the realised distribution leans sparse. Both are
        measured and reported rather than engineered away.
        """
        song = self.songs[rng.integers(len(self.songs))]
        start_frame = self._draw_window_start(rng, song.joint_segments, song.max_frames)
        active = self._active_classes_in_window(song, start_frame)

        # n uniform over 1..audible — the min(n_classes, ...) cap is automatic, since a
        # song can never make more classes audible than the scheme models.
        # sorted() rather than dict order: the draw must not depend on insertion order.
        active_classes = sorted(active)
        n_classes = int(rng.integers(1, len(active_classes) + 1))
        chosen = rng.choice(len(active_classes), size=n_classes, replace=False)

        picks = []
        for pick in chosen:
            class_name = active_classes[pick]
            candidates = active[class_name]
            # one file per class slot, matching exp002's draw_unit: file — a song with
            # 피리1+피리2 contributes one of them, not their sum (→ launch report)
            entry = candidates[rng.integers(len(candidates))]
            picks.append((self.cfg.classes.index(class_name), class_name, entry,
                          float(self._draw_stem_gain(rng, class_name, entry))))

        # L/R swap PER MIX, not per stem: one decision for the whole ensemble preserves
        # its spatial image
        swap_channels = bool(rng.random() < self.cfg.channel_swap_prob)
        eq_boards = [build_random_eq(rng, self.cfg.eq_stem_gain_db, q_max=5.0)
                     if rng.random() < self.cfg.eq_stem_prob else None
                     for _ in picks]
        return CoherentPlan(song=song, start_frame=start_frame,
                            active_classes=active_classes, picks=picks,
                            swap_channels=swap_channels, eq_boards=eq_boards)

    def _coherent_item(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One coherent mix: every stem read from one song at one time offset."""
        plan = self._plan_coherent(rng)

        stems = np.zeros((len(self.cfg.classes), 2, self.chunk_frames), dtype=np.float32)
        for slot, _class_name, entry, gain in plan.picks:
            # the SAME start_frame for every stem — this is the whole point of the arm
            stems[slot] = self._read_window(entry, plan.start_frame) * np.float32(gain)
        if plan.swap_channels:
            # applied to all slots at once; undrawn slots are zeros, so it is a no-op
            stems = stems[:, ::-1].copy()
        for (slot, _class_name, _entry, _gain), board in zip(plan.picks, plan.eq_boards):
            if board is not None:
                stems[slot] = board(stems[slot], self.cfg.sample_rate)
        return stems, np.array([slot for slot, _, _, _ in plan.picks])

    def plan_item(self, index: int) -> CoherentPlan | None:
        """Provenance of item `index` — the coherent plan, or None if it drew incoherent.

        Runs the identical RNG sequence `__getitem__` runs, so the plan it returns is the
        one that item really uses; it just stops before any audio is read. Verification
        and characterisation entry point.

        Args:
            index: the dataset index, exactly as the DataLoader would pass it.
        """
        rng = np.random.default_rng([self.cfg.seed, index])
        if self.cfg.coherent_mix_prob > 0.0 and rng.random() < self.cfg.coherent_mix_prob:
            return self._plan_coherent(rng)
        return None

    def _incoherent_item(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One incoherent mix: n classes, each from an independent song and offset."""
        n_classes = int(rng.choice(self.density_values, p=self.density_probs))
        drawn = rng.choice(len(self.cfg.classes), size=n_classes, replace=False)

        stems = np.zeros((len(self.cfg.classes), 2, self.chunk_frames), dtype=np.float32)
        for slot in drawn:
            class_name = self.cfg.classes[slot]
            excerpt, entry = self._draw_excerpt(rng, class_name)
            stems[slot] = self._augment_stem(rng, excerpt, class_name, entry)
        return stems, drawn

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # independent, reproducible stream per (seed, item) — worker-count-agnostic
        rng = np.random.default_rng([self.cfg.seed, index])

        # short-circuit: at p=0.0 no random number is drawn here, so a pure-incoherent
        # run's stream is bit-identical to one that predates coherent mixing
        if self.cfg.coherent_mix_prob > 0.0 and rng.random() < self.cfg.coherent_mix_prob:
            stems, drawn = self._coherent_item(rng)
        else:
            stems, drawn = self._incoherent_item(rng)

        if rng.random() < self.cfg.eq_mixbus_prob:
            # one shared EQ over every stem: linear, so the mixture hears the same EQ
            # and mixture ≡ Σ(targets) survives
            board = build_random_eq(rng, self.cfg.eq_mixbus_gain_db, q_max=3.0)
            for slot in drawn:
                stems[slot] = board(stems[slot], self.cfg.sample_rate)

        mixture = stems.sum(axis=0)
        stems, mixture = self._normalize(stems, mixture)
        return torch.from_numpy(stems), torch.from_numpy(mixture)

    def __len__(self) -> int:
        return self.num_items


# --- MSST adapter -----------------------------------------------------------
def create_msst_dataset(config, batch_size: int) -> GugakMixDataset:
    """Factory the MSST fork calls when `config.training.custom_dataset` points here.

    Reads the experiment YAML's `gugak_mix` block; dataset length follows MSST's
    epoch semantics (num_steps × batch_size). Training must be launched from the
    repo root — manifest paths resolve against the current working directory.

    Args:
        config: full MSST config object (ml_collections ConfigDict or OmegaConf).
        batch_size: per-process batch size, passed by the fork's prepare_data.
    """
    block = config["gugak_mix"]
    if hasattr(block, "to_dict"):          # ml_collections ConfigDict
        block = block.to_dict()
    else:
        try:                               # OmegaConf container
            from omegaconf import OmegaConf
            if OmegaConf.is_config(block):
                block = OmegaConf.to_container(block, resolve=True)
        except ImportError:
            pass
    mix_cfg = MixDatasetConfig.from_mapping(dict(block))

    # MSST's losses/metrics/logging are keyed by training.instruments — the two
    # class lists must be identical AND identically ordered, or stems misalign
    instruments = list(config["training"]["instruments"])
    if instruments != list(mix_cfg.classes):
        raise ValueError(
            "config.training.instruments must equal gugak_mix.classes (same order) — "
            f"got {instruments} vs {list(mix_cfg.classes)}")

    repo_root = Path.cwd()
    if not (repo_root / mix_cfg.source_manifest).exists():
        raise FileNotFoundError(
            f"{mix_cfg.source_manifest} not found under cwd={repo_root} — launch "
            "training from the gugak_stem_separation repo root")

    num_items = int(config["training"]["num_steps"]) * int(batch_size)
    return GugakMixDataset(mix_cfg, repo_root, num_items)
