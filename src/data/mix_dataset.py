"""mix_dataset.py — on-the-fly incoherent training mixes (Training Data Strategy recipe).

Each item is built, not loaded: draw HOW MANY classes go in the mix (density — uniform
over 1..n_classes, or from the measured audible-편성 distribution of real songs, per
`density_mode`), draw WHICH classes (uniform), draw one
activity-aware excerpt per class from the ingest store, apply live augmentations, sum to
a mixture, then loudness-normalize mixture and targets by the same gain. Returns
(stems, mixture) float32 tensors shaped [n_classes, 2, chunk] / [2, chunk] — the exact
batch contract MSST's trainer consumes, so this class drops into its DataLoader.

Everything is manifest-driven (source_manifest ⋈ activity_segments ⋈ chunk_activities);
no directory walking, and no audio is decoded at init. Every knob lives in the
experiment YAML's `gugak_mix` block — including RESERVED keys for features that are
designed but deliberately not implemented yet (coherent mixes, 타악-2× multi-sampling,
the pitch-shift pool, song-base draw units). Setting one of those raises
NotImplementedError loudly rather than silently ignoring it. The nested
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
    # live augmentation knobs (exp001: EQ probs 0.0 — built, off)
    gain_min: float = 0.25
    gain_max: float = 1.25
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
    coherent_mix_prob: float = 0.0        # p>0 = coherent same-song draws (+ 판소리 trim)
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

    # --- reserved-knob guard ---
    @staticmethod
    def _reject_unimplemented(cfg: MixDatasetConfig) -> None:
        """Reserved config keys exist so the vocabulary is stable; using one fails loudly."""
        if cfg.coherent_mix_prob > 0.0:
            raise NotImplementedError(
                "coherent_mix_prob > 0: coherent same-song draws (incl. 판소리 "
                "trim-to-shortest) are designed but not implemented yet")
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
    def _build_source_pool(self) -> dict:
        """Per-class draw lists: (paths, frame counts, active segments) from the manifests."""
        manifest = pd.read_parquet(self.root / self.cfg.source_manifest)
        sources = manifest[(manifest.dataset.isin(self.cfg.datasets))
                           & (manifest.split == self.cfg.split)
                           & (manifest.role != "master")
                           & (manifest.stem_group.isin(self.cfg.classes))]

        segments = pd.read_parquet(self.root / self.cfg.activity_segments)
        segments_by_file = {fid: grp[["start_s", "end_s"]].to_numpy()
                            for fid, grp in segments.groupby("file_id")}

        pool: dict = {}
        for class_name, group in sources.groupby("stem_group"):
            entries = []
            for row in group.itertuples():
                file_segments = segments_by_file.get(row.file_id)
                if file_segments is None or len(file_segments) == 0:
                    continue    # fully-silent file: nothing to draw (QC says none exist)
                source_lufs = (self.loudness_sampler.source_loudness(row.file_id)
                               if self.loudness_sampler is not None else math.nan)
                entries.append(SourceEntry(row.out_path, int(row.out_frames),
                                           file_segments, row.dataset, source_lufs,
                                           int(row.out_channels)))
            pool[class_name] = entries

        missing = [c for c in self.cfg.classes if not pool.get(c)]
        if missing:
            raise ValueError(f"no drawable sources for classes {missing} "
                             f"(datasets={self.cfg.datasets}, split={self.cfg.split})")
        return pool

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

        A random active segment is chosen duration-weighted, an anchor point drawn
        inside it, and the window placed uniformly at random over positions containing
        the anchor — so every excerpt overlaps real activity, but silence around short
        segments stays in (silence is a valid signal when deliberate). Sources shorter
        than the window are zero-padded at the tail; mono sources are center-duplicated
        to stereo (never naive-summed — anti-phase rule). The entry it came from is
        returned alongside, since the gain treatment depends on which pool that is.
        """
        entry = self.pool[class_name][rng.integers(len(self.pool[class_name]))]
        out_path, total_frames, segments = entry.out_path, entry.out_frames, entry.segments

        durations = segments[:, 1] - segments[:, 0]
        segment = segments[rng.choice(len(segments), p=durations / durations.sum())]
        anchor_s = rng.uniform(segment[0], segment[1])
        start_s = anchor_s - rng.uniform(0.0, self.cfg.segment_seconds)
        max_start = max(0, total_frames - self.chunk_frames)
        start_frame = int(np.clip(round(start_s * self.cfg.sample_rate), 0, max_start))

        audio, sample_rate = soundfile.read(
            self.root / out_path, start=start_frame, frames=self.chunk_frames,
            dtype="float32", always_2d=True, fill_value=0.0)   # fill pads short reads
        if sample_rate != self.cfg.sample_rate:
            raise ValueError(f"{out_path}: sr {sample_rate} != {self.cfg.sample_rate} "
                             "(ingest store contract broken)")
        audio = audio.T                                        # -> (channels, frames)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)                # centered mono
        return audio, entry

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # independent, reproducible stream per (seed, item) — worker-count-agnostic
        rng = np.random.default_rng([self.cfg.seed, index])

        n_classes = int(rng.choice(self.density_values, p=self.density_probs))
        drawn = rng.choice(len(self.cfg.classes), size=n_classes, replace=False)

        stems = np.zeros((len(self.cfg.classes), 2, self.chunk_frames), dtype=np.float32)
        for slot in drawn:
            class_name = self.cfg.classes[slot]
            excerpt, entry = self._draw_excerpt(rng, class_name)
            stems[slot] = self._augment_stem(rng, excerpt, class_name, entry)

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
