"""mix_dataset.py — on-the-fly training mixes (Training Data Strategy recipe).

One sampler, one knob: `coherent_mix_prob` (p) sets HOW MUCH of each mix is coherent —
0.0 = fully incoherent (exp002 and everything before it), 1.0 = every drawable stem
coherent, values between interpolate at stem grain rather than item grain.

THE DRAW ORDER IS THE DESIGN. exp002.2 drew the song first, so density and class
identity followed real 편성 — realised density fell 5.00 → 3.24 and rare-class exposure
collapsed (양금 0.10×). Coherence and exposure moved together, and the run's collapse
became uninterpretable. This sampler decides density and class identity FIRST, uniformly,
and only then decides which members of that class set are mutually coherent — so
per-class exposure is uniform BY CONSTRUCTION at every value of p:

  1. n ~ density draw (uniform over density_uniform_min..len(classes), or measured)
  2. S = n classes, uniform without replacement            ← exposure fixed here
  3. k = ⌊p·n⌋ + Bernoulli(frac(p·n)), so E[k] = p·n       ← how many are coherent
     (k=1 is degenerate — one stem is coherent with nothing — and snaps to 0 or 2
     by fair coin, preserving E[k])
  4. fill loop: anchor songs are drawn (per `anchor_selection`) and each contributes
     one CLUSTER — the classes of S it can make jointly audible at ONE shared offset,
     capped by the remaining need. A cluster is valid only if it holds at least
     `cluster_min_melodic` NON-PERCUSSION classes: unison is a melodic phenomenon, and
     a {타악기, one melody} cluster is a drum over a single line — it inflates k while
     teaching nothing about heterophony. 타악기 may JOIN a cluster, but never counts
     toward the minimum. Shortfall is accepted and recorded, never repaired by
     resampling S — that would leak exposure bias back in.
  5. everything of S not in a cluster is an ordinary incoherent draw (independent
     song, independent offset)

Coherence holds WITHIN a cluster; between clusters (and against the incoherent stems)
it is ordinary incoherent mixing. `anchor_selection` spans the design space:
`greedy` = one anchor of maximum overlap, shortfall accepted (Method A) · `uniform` =
anchors drawn uniformly until filled (Method B) · `overlap_weighted` = the interpolation.

Motivation for coherence at all: gugak ensembles are heterophonic — instruments play
near-unison variants of one melodic line — so an incoherent mix (가야금 from song A over
거문고 from song B) is a much easier separation problem than the real mixtures val and
test are built from. p controls how much of the hard case training sees.

Every item is PLANNED before any audio is read (`_plan_item`: all RNG, no I/O;
`__getitem__`: audio realisation of the plan). Verification pulls hundreds of thousands
of draws through the REAL sampling code via `plan_item` rather than a reimplementation
of it (→ scripts/verify_coherent_sampler.py). `coherent_mix_prob` is read per draw, not
cached at init, so a curriculum module can drive it during training.

Both mix halves then share the live augmentations, sum to a mixture, and
loudness-normalize mixture and targets by the same gain. Returns (stems, mixture)
float32 tensors shaped [n_classes, 2, chunk] / [2, chunk] — the exact batch contract
MSST's trainer consumes, so this class drops into its DataLoader.

RECORDING-CONDITION AUGMENTATION (`gugak_mix.recording_aug` → src/data/recording_aug.py)
sits between the per-stem chain and the normalizer: capture response · room convolution ·
level automation · mix-bus loudness/limiting · an opt-in dirty-input branch. It exists
because 71955 was recorded in ONE studio, so the corpus has no inter-recording acoustic
variance for cross-song mixing to exploit, while deployment audio is picked up at a
distance in an unknown room. The room is drawn PER MIXTURE and SHARED WITHIN one — giving
each stem its own room would hand the separator an acoustic grouping cue that real
recordings do not contain. The block is off unless a config says otherwise, and off means
zero RNG draws, so exp003.x/exp004.x/exp006 are bit-identical to the pre-feature module.

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
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyloudnorm
import scipy.signal
import soundfile
import torch
import yaml
from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter, Pedalboard

try:    # imported as a package module (MSST hook: src.data.mix_dataset)
    from src.data.loudness_match import LoudnessMatchConfig, LoudnessTargetSampler
    from src.data import recording_aug
except ModuleNotFoundError:   # imported as a sibling (build_sumstem_eval.py runs so)
    from loudness_match import LoudnessMatchConfig, LoudnessTargetSampler
    import recording_aug


# Safety margin added to the qualifying joint-interval length. A cluster window is
# placed to CONTAIN a joint-activity interval longer than coverage_threshold × segment,
# which guarantees both of the interval's instruments clear the audibility test — but
# window starts round to whole frames, so an interval at exactly the threshold could
# lose half a sample of coverage and fail by 1e-5. Ten milliseconds dwarfs that.
JOINT_INTERVAL_MARGIN_S = 0.01

ANCHOR_SELECTION_MODES = ("uniform", "overlap_weighted", "greedy")


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
    #   "uniform":  n ~ uniform over density_uniform_min..len(classes). exp002 onward —
    #               we cannot justify matching real 편성 statistics when class IDENTITY
    #               is already uniform, so the realism was half-hearted either way
    #               (prof, 2026-08-03).
    # The two chunk knobs below are read only in "measured" mode.
    density_mode: str = "measured"
    density_chunk_len_s: float = 10.0
    density_coverage_threshold: float = 0.25
    # lower end of the uniform density draw. 1 preserves exp002's stream exactly
    # (n ~ U{1..9}); new runs set 2 (prof meeting 2026-08-10: n=1 mixes teach the
    # trivial identity mapping). Read only in "uniform" mode. DO NOT change the default.
    density_uniform_min: int = 1
    # coherent mixing — p, the expected FRACTION of each mix's stems that are drawn
    # mutually coherent (E[k] = p·n, see module docstring). 0.0 = the incoherent base
    # recipe (default; consumes no extra randomness, so the draw stream is bit-identical
    # to a run that predates this feature). Read PER DRAW, never cached, so a curriculum
    # module may mutate it between steps.
    coherent_mix_prob: float = 0.0
    # how the fill loop picks anchor songs (→ module docstring):
    #   "greedy"           single anchor, maximum overlap with the unfilled classes,
    #                      shortfall accepted (Method A)
    #   "uniform"          uniform over eligible anchors, loop until filled (Method B)
    #   "overlap_weighted" P(song) ∝ |overlap|, loop until filled (the interpolation)
    anchor_selection: str = "uniform"
    # live augmentation knobs (exp001: EQ probs 0.0 — built, off)
    gain_min: float = 0.25
    gain_max: float = 1.25
    # L/R swap. Applied PER STEM to incoherent stems (they are unrelated anyway); ONE
    # shared decision per CLUSTER when cluster_shared_channel_swap is true — swapping a
    # real ensemble's stems independently would scramble its spatial image, which is
    # part of what makes it coherent. (exp002.2 applied per-mix swap with no config
    # field; this is that behaviour's field, at cluster grain.)
    channel_swap_prob: float = 0.5
    cluster_shared_channel_swap: bool = True
    # random gain inside a cluster. False (default, current behaviour) = per-stem, which
    # augments balance but DESTROYS the anchor's real inter-instrument balance; true =
    # one shared gain per cluster, preserving it. ⚠️ OPEN DESIGN DECISION — wired but
    # unstudied; do not flip without an experiment (→ build report).
    cluster_shared_gain: bool = False
    # minimum NON-PERCUSSION classes a cluster must contain (→ module docstring). The
    # percussion set is resolved from the taxonomy's per-instrument `pitched` field —
    # a class is melodic iff it holds at least one pitched instrument — never from the
    # class name, so an 11-class scheme's pitched_percussion (편종·편경·방향) counts as
    # melodic. Only 0..2 exists: clusters are seeded from PAIRWISE joint-activity
    # spans, which can guarantee at most 2 jointly-audible melodic classes.
    cluster_min_melodic: int = 2
    # taxonomy the resolution reads (canonical mapping — path in config, not code)
    stem_taxonomy: str = "configs/stem_taxonomy.yaml"
    eq_stem_prob: float = 0.0
    eq_mixbus_prob: float = 0.0
    eq_stem_gain_db: float = 9.0
    eq_mixbus_gain_db: float = 6.0
    # solo-pool loudness pre-conditioning (nested block → LoudnessMatchConfig);
    # absent/empty = disabled = every stem keeps the random-gain treatment
    loudness_match: dict = field(default_factory=dict)
    # RECORDING-CONDITION augmentation (nested block → src/data/recording_aug.py): the
    # capture-side stages — device response, room convolution, level automation, mix-bus
    # loudness/limiting, and the opt-in dirty-input branch. Absent/empty/`enable: false`
    # consumes NOT ONE RNG draw and runs NOT ONE line of the realisation path, so every
    # config that predates it stays bit-identical (proved in tests/test_recording_aug.py).
    # It lives in its own module for the same reason loudness_match does: it is a nested
    # sub-block with its own config surface and its own DSP.
    recording_aug: dict = field(default_factory=dict)
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


# --- song entry (anchor draws only) -----------------------------------------
class SongEntry(NamedTuple):
    """One drawable anchor song: its stems grouped by class, and how long it runs.

    `max_frames` is the LONGEST stem, not the shortest. The trim-to-shortest rule exists
    to make Σstems line up with the publisher master; no master is involved here, so a
    window in the tail of a long 판소리 가야금 stem is legitimate content and is kept.
    Stems that have already ended simply read as zeros there.
    """
    song_id: str
    genre_sub: str
    entries_by_class: dict
    max_frames: int


# --- plan objects (all sampling decided before any audio is read) -----------
class StemPick(NamedTuple):
    """One stem of one mix, fully decided: what to read, where, and how to treat it.

    Cluster members share their cluster's `start_frame` (the temporal alignment that
    makes them coherent); incoherent stems each carry their own. `cluster_index` links
    back into MixPlan.clusters, None = incoherent.
    """
    slot: int
    class_name: str
    entry: SourceEntry
    gain: float
    start_frame: int
    swap_channels: bool
    eq_board: Pedalboard | None
    cluster_index: int | None


@dataclass
class ClusterPlan:
    """One anchor song's contribution: which classes it plays, aligned at one offset.

    Mutable (a dataclass, unlike the other plan objects) because the r==1 rule may grow
    the last cluster by one member after it is built. `shared_gain`/`shared_swap` are
    None when the corresponding `cluster_shared_*` knob is off and the members drew
    independently.
    """
    song_id: str
    genre_sub: str
    start_frame: int
    member_slots: list
    shared_gain: float | None
    shared_swap: bool | None


class MixPlan(NamedTuple):
    """Everything one mix does, decided before any audio is read.

    `k_declared` is what step 3 asked for (post snap/clamp), `k_realised` what the fill
    loop achieved — the gap is the shortfall, itemised in `shortfall_reasons` (tuple of
    "no_eligible_anchor" · "greedy_single_anchor" · "r1_dropped"). Exposure lives in
    `drawn_slots` and is untouched by any of that: shortfall demotes stems to
    incoherent, never removes or replaces them.

    `recording` carries the five recording-condition stages and DEFAULTS TO None — the
    field is last and optional so that every existing caller that builds or unpacks a
    MixPlan positionally keeps working, and so a provenance dump of a pre-recording_aug
    config reads exactly as it did before.
    """
    p: float
    n: int
    drawn_slots: tuple
    k_target: float
    k_declared: int
    k_realised: int
    clusters: tuple
    picks: tuple
    shortfall_reasons: tuple
    mixbus_board: Pedalboard | None
    recording: "recording_aug.RecordingPlan | None" = None


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


# --- segment arithmetic (anchor index build) --------------------------------
def merge_segments(segments: np.ndarray) -> np.ndarray:
    """Sort [start, end] spans and merge overlaps → disjoint, ascending spans.

    Activity segments arrive sorted and disjoint from the scan, but the intersection
    below is only correct under that invariant, so it is enforced rather than assumed.

    Args:
        segments: (n, 2) array of [start_s, end_s] spans, any order.
    """
    if len(segments) == 0:
        return segments.reshape(0, 2)
    ordered = segments[np.argsort(segments[:, 0])]
    merged = [ordered[0].copy()]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.vstack(merged)


def intersect_segments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of two disjoint-ascending span lists → the spans active in BOTH.

    Args:
        a: (n, 2) merged span list.
        b: (m, 2) merged span list.
    """
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i, 0], b[j, 0])
        hi = min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append((lo, hi))
        # advance whichever span ends first
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out) if out else np.empty((0, 2))


# --- dataset ----------------------------------------------------------------
class GugakMixDataset(torch.utils.data.Dataset):
    """Mix dataset over the ingest store, manifest-driven (→ module docstring).

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

        # recording-condition augmentation: parsed and cross-validated up front (a
        # dry-target or dirty-input arm must fail at construction, never halfway through
        # an epoch), and the RIR/noise banks are attached only when their probability is
        # actually nonzero — a missing bank under prob > 0 is a hard error, never a
        # silent no-op.
        self.recording_cfg = recording_aug.ResolvedRecordingConfig.from_mapping(
            cfg.recording_aug)
        self.rir_pool = (
            recording_aug.RirPool(self.recording_cfg.rir, self.root, cfg.sample_rate,
                                  self.chunk_frames)
            if self.recording_cfg.enable and self.recording_cfg.rir.prob > 0.0 else None)
        self.noise_pool = (
            recording_aug.NoisePool(self.recording_cfg.dirty, self.root, cfg.sample_rate)
            if self.recording_cfg.enable and self.recording_cfg.dirty.noise_prob > 0.0
            else None)

        self.pool = self._build_source_pool()
        self.melodic_slots = self._resolve_melodic_slots()
        self.density_values, self.density_probs = self._build_density_histogram()
        # anchor pool (songs + joint-activity index): a manifest regroup plus pairwise
        # segment intersections. Built eagerly when the config already asks for
        # coherence (fail fast, pay up front); built lazily on the first coherent draw
        # otherwise, so a pure-incoherent run pays nothing for the feature existing —
        # but a curriculum ramping p from 0 still works without a restart.
        self._songs: list | None = None
        if cfg.coherent_mix_prob > 0.0:
            self._ensure_anchor_pool()

    # --- config guard ---
    @staticmethod
    def _reject_unimplemented(cfg: MixDatasetConfig) -> None:
        """Value checks + reserved-key guard (reserved keys fail loudly, never no-op)."""
        if not 0.0 <= cfg.coherent_mix_prob <= 1.0:
            raise ValueError(f"coherent_mix_prob={cfg.coherent_mix_prob}: expected [0, 1]")
        if cfg.anchor_selection not in ANCHOR_SELECTION_MODES:
            raise ValueError(f"anchor_selection={cfg.anchor_selection!r}: expected one "
                             f"of {ANCHOR_SELECTION_MODES}")
        if not 1 <= int(cfg.density_uniform_min) <= len(cfg.classes):
            raise ValueError(f"density_uniform_min={cfg.density_uniform_min}: expected "
                             f"1..{len(cfg.classes)} (the class count)")
        if not 0 <= int(cfg.cluster_min_melodic) <= 2:
            raise NotImplementedError(
                f"cluster_min_melodic={cfg.cluster_min_melodic}: only 0..2 exists — "
                "clusters are seeded from pairwise joint-activity spans, which can "
                "guarantee at most 2 jointly-audible melodic classes; ≥3 would need a "
                "triple-wise span index (reserved seam)")
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

    def _resolve_melodic_slots(self) -> frozenset:
        """Which output slots are melodic (non-percussion), from the taxonomy.

        A class is melodic iff at least one of its instruments has `pitched: true` in
        `configs/stem_taxonomy.yaml`. This is a factual property read from the canonical
        mapping, never a name match — 타악기 (drums, clappers: all unpitched) resolves
        percussion, while an 11-class scheme's pitched_percussion (편종·편경·방향: tuned
        idiophones) resolves melodic. Drives the cluster_min_melodic rule only; the
        incoherent path never consults it.
        """
        taxonomy = yaml.safe_load(
            (self.root / self.cfg.stem_taxonomy).read_text())["instruments"]
        group_has_pitched: dict = {}
        for info in taxonomy.values():
            group = info.get("stem_group")
            if group is not None:
                group_has_pitched[group] = (group_has_pitched.get(group, False)
                                            or bool(info["pitched"]))
        missing = [c for c in self.cfg.classes if c not in group_has_pitched]
        if missing:
            raise ValueError(f"classes {missing} not present as stem_group in "
                             f"{self.cfg.stem_taxonomy} — cannot resolve percussion")
        return frozenset(slot for slot, class_name in enumerate(self.cfg.classes)
                         if group_has_pitched[class_name])

    def _build_song_pool(self) -> list:
        """Per-song draw list for anchor draws — the same sources, regrouped by song.

        Only song-keyed rows qualify: the 71470 solo clips are standalone phrases with no
        song to be coherent WITH (`song_id` is null for them), so they cannot take part in
        a coherent draw and are dropped here rather than silently mis-grouped. A run that
        wants both a solo pool and coherent mixes has to say what that means first.

        Songs contributing only one drawable class are kept in the SONG list (they are
        legitimate incoherent sources) but can never become anchors: anchor eligibility
        requires a jointly-audible class PAIR, which the index below simply never
        records for them.
        """
        song_rows = self._sources[self._sources.song_id.notna()]
        dropped = self._sources.song_id.isna().sum()
        if dropped:
            datasets = sorted(self._sources[self._sources.song_id.isna()].dataset.unique())
            raise ValueError(
                f"coherent mixing engaged but {dropped} drawable sources from datasets "
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
                max_frames=max(e.out_frames for e in all_entries)))

        if not songs:
            raise ValueError("coherent mixing engaged but no drawable songs "
                             f"(datasets={self.cfg.datasets}, split={self.cfg.split})")
        return songs

    @property
    def songs(self) -> list:
        """The anchor song pool (built on first access)."""
        self._ensure_anchor_pool()
        return self._songs

    def _ensure_anchor_pool(self) -> None:
        """Build the song pool + joint-activity anchor index, once."""
        if self._songs is not None:
            return
        self._songs = self._build_song_pool()
        self._build_anchor_index()

    def _build_anchor_index(self) -> None:
        """Precompute which songs can anchor which class pairs, and where.

        For every song and every unordered pair of its classes, intersect the two
        classes' active segments AT ENTRY LEVEL (each file of a vs each file of b) and
        keep the contiguous joint spans longer than coverage_threshold × segment (+ a
        rounding margin). Entry level matters: a class-level union could show joint
        activity that no single drawable FILE of the class sustains (타악기 is several
        instruments in separate files), and the cluster draw guarantees rest on one
        concrete file per class clearing the audibility bar.

        Products, all indexed by song position in `self._songs`:
          _pair_row        (a, b) slot pair → row in the two matrices below
          _pair_eligible   bool (36, n_songs): song has ≥1 qualifying span for the pair
          _pair_intervals  per song, dict pair → (m, 2) qualifying spans in seconds
          _song_classes    bool (n_songs, n_classes): song has ≥1 drawable file of class

        Spans from different entry pairs of the same class pair may overlap and are
        deliberately NOT re-merged: the window draw weights spans by duration, so a
        passage where two 피리 files both meet the 대금 counts twice — the same
        busy-passages-drawn-more behaviour the incoherent window draw already has.
        """
        n_classes = len(self.cfg.classes)
        slot_of = {name: slot for slot, name in enumerate(self.cfg.classes)}
        min_span_s = (self.cfg.density_coverage_threshold * self.cfg.segment_seconds
                      + JOINT_INTERVAL_MARGIN_S)

        self._pair_row = {pair: row for row, pair
                          in enumerate(combinations(range(n_classes), 2))}
        self._pair_eligible = np.zeros((len(self._pair_row), len(self._songs)),
                                       dtype=bool)
        self._pair_intervals: list[dict] = []
        self._song_classes = np.zeros((len(self._songs), n_classes), dtype=bool)

        for song_index, song in enumerate(self._songs):
            merged_by_slot = {slot_of[name]: [merge_segments(e.segments) for e in entries]
                             for name, entries in song.entries_by_class.items()}
            self._song_classes[song_index, list(merged_by_slot)] = True

            pair_intervals: dict = {}
            for slot_a, slot_b in combinations(sorted(merged_by_slot), 2):
                spans = [kept
                         for merged_a in merged_by_slot[slot_a]
                         for merged_b in merged_by_slot[slot_b]
                         for joint in [intersect_segments(merged_a, merged_b)]
                         for kept in [joint[(joint[:, 1] - joint[:, 0]) >= min_span_s]]
                         if len(kept)]
                if spans:
                    pair_intervals[(slot_a, slot_b)] = np.vstack(spans)
                    self._pair_eligible[self._pair_row[(slot_a, slot_b)],
                                        song_index] = True
            self._pair_intervals.append(pair_intervals)

    def _build_density_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """The distribution n is drawn from — (values, probabilities) over class counts.

        Dispatches on cfg.density_mode. Both modes return the same shape, so the plan
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
        """n uniform over density_uniform_min..len(classes) — every size equally likely.

        n=0 is excluded for the same reason the measured mode drops it: an all-silent
        mixture teaches nothing. The upper end is the class count, since classes are
        drawn without replacement. The lower end is configurable: 1 reproduces exp002;
        2 (the go-forward value, prof 2026-08-10) drops the single-stem mixes whose
        target is the identity mapping.

        This mode reads no manifest — chunk_activities is a measured-mode input only.
        """
        values = np.arange(int(self.cfg.density_uniform_min), len(self.cfg.classes) + 1)
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

    # --- shared draw arithmetic ---
    def _draw_window_start(self, rng: np.random.Generator, segments: np.ndarray,
                           total_frames: int) -> int:
        """Activity-aware window start, in frames, from a pool of active segments.

        One segment is chosen duration-weighted, an anchor point drawn inside it, and the
        window placed uniformly at random over positions containing the anchor — so every
        window overlaps real activity, but silence around short segments stays in
        (silence is a valid signal when deliberate). This is the INCOHERENT window draw;
        cluster windows use `_cluster_window_start`, whose containment rule additionally
        guarantees joint audibility.

        Args:
            rng: the item's Generator.
            segments: (n, 2) array of [start_s, end_s] active spans.
            total_frames: length of the source.
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
        beyond the file's end returns silence outright: in a cluster every stem reads
        the SAME offset, and a stem that has already finished genuinely has no content
        there.
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

    # --- listening-example helpers (scripts/render_coherence_examples.py) ---
    def _draw_excerpt(self, rng: np.random.Generator,
                      class_name: str) -> tuple[np.ndarray, SourceEntry]:
        """One activity-aware excerpt of a random source of `class_name` → (2, chunk).

        The incoherent draw as a single audio-returning call. Training goes through
        `_plan_incoherent_stem` instead (plan first, read later); this stays for the
        rendering script, and both consume the item RNG identically.
        """
        entry = self.pool[class_name][rng.integers(len(self.pool[class_name]))]
        start_frame = self._draw_window_start(rng, entry.segments, entry.out_frames)
        return self._read_window(entry, start_frame), entry

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

    def _loudnorm(self, stems: np.ndarray, mixture: np.ndarray,
                  target_lufs: float) -> tuple[np.ndarray, np.ndarray]:
        """Measure the mixture and scale mixture AND targets by the one shared gain.

        Split out of `_normalize` so the LimitAug arm can normalize BEFORE the mix bus to
        a per-item sampled loudness (→ recording_aug.LimitAugConfig: the old fixed −19 LUFS
        pass run after the limiter erases exactly the loudness diversity the technique
        exists to create). The arithmetic and its order are untouched, so the default path
        is bit-identical.
        """
        loudness = self.meter.integrated_loudness(mixture.T)
        if not math.isinf(loudness):
            gain = np.float32(10 ** ((target_lufs - loudness) / 20))
            stems, mixture = stems * gain, mixture * gain
        return stems, mixture

    def _peak_guard(self, stems: np.ndarray,
                    mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One shared scale that brings the mixture peak under the ceiling, if it is over."""
        peak = float(np.abs(mixture).max())
        if peak > self.cfg.peak_ceiling:
            scale = np.float32(self.cfg.peak_ceiling / peak)
            stems, mixture = stems * scale, mixture * scale
        return stems, mixture

    def _normalize(self, stems: np.ndarray,
                   mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Loudnorm mixture AND targets by one shared gain, then peak-guard both.

        Normalizing the mixture (never per-stem) keeps stem balance intact; applying
        the identical gain to the targets keeps mixture ≡ Σ(targets). The -inf guard
        covers near-silent draws (a lone sparse stem's quiet window).
        """
        stems, mixture = self._loudnorm(stems, mixture, self.cfg.target_lufs)
        return self._peak_guard(stems, mixture)

    # --- cluster draw (the coherent half of a mix) ---
    @staticmethod
    def _window_coverage(segments: np.ndarray, window_start_s: float,
                         window_end_s: float) -> float:
        """Fraction of the window this source's active segments cover, in [0, 1]."""
        overlap = np.minimum(segments[:, 1], window_end_s) - np.maximum(segments[:, 0],
                                                                       window_start_s)
        return float(np.clip(overlap, 0.0, None).sum()) / (window_end_s - window_start_s)

    def _draw_coherent_count(self, rng: np.random.Generator, p: float, n: int) -> int:
        """k — how many of the n stems will be drawn mutually coherent, E[k] = p·n.

        ⌊p·n⌋ plus a Bernoulli on the fraction gives E[k] = p·n exactly. k=1 is
        degenerate (one stem is coherent with nothing) and snaps to 0 or 2 by fair
        coin, which preserves the expectation. The n-clamp can only engage at n=1
        (raw k never exceeds n otherwise), where it forces k=0 — that cell's realised
        E[k] is 0 against a declared p·1, a quantified shortfall, not a bug.

        No randomness is consumed when p·n is an integer (in particular at p=0, which
        keeps the exp002 stream bit-identical, and at p=1, where k=n deterministically).

        Args:
            rng: the item's Generator.
            p: coherent_mix_prob as read for THIS draw.
            n: the mix's class count.
        """
        if p <= 0.0:
            return 0
        target = p * n
        k = int(math.floor(target))
        fraction = target - k
        if fraction > 0.0 and rng.random() < fraction:
            k += 1
        if k == 1:
            k = 2 if rng.random() < 0.5 else 0
        k = min(k, n)
        return k if k >= 2 else 0

    def _seed_pair_valid(self, pair: tuple) -> bool:
        """Can this class pair seed a cluster? It must carry the melodic minimum.

        A cluster's guaranteed-audible members are its seed pair, so the pair itself
        must hold ≥ cluster_min_melodic melodic classes — a {타악기, melody} span can
        never be the reason a cluster exists. Percussion still joins clusters, but
        only ones a melodic pair already justified.
        """
        melodic = ((pair[0] in self.melodic_slots) + (pair[1] in self.melodic_slots))
        return melodic >= int(self.cfg.cluster_min_melodic)

    def _eligible_anchor_songs(self, remaining: list) -> np.ndarray:
        """Song indices that can anchor a cluster for the unfilled classes.

        Eligible = the song has, for at least one VALID pair of classes in `remaining`
        (→ `_seed_pair_valid`: the pair carries the melodic minimum), a contiguous
        joint-activity span long enough that a window containing it makes both classes
        clear the audibility bar (the precomputed index). Requiring a pair is the ≥2
        rule: a cluster of one is not coherence.

        Can now be empty even with anchors on disk — e.g. R = {타악기, one melody}
        offers no valid seed pair. The caller logs that as no_eligible_anchor.

        Args:
            remaining: sorted unfilled class slots (R).
        """
        rows = [self._pair_row[pair] for pair in combinations(remaining, 2)
                if self._seed_pair_valid(pair)]
        if not rows:
            return np.array([], dtype=np.intp)
        return np.flatnonzero(self._pair_eligible[rows].any(axis=0))

    def _select_anchor(self, rng: np.random.Generator, eligible: np.ndarray,
                       remaining: list) -> int:
        """One anchor song index from the eligible set, per `anchor_selection`.

        Overlap (for the weighted and greedy rules) counts the song's drawable classes
        in R — the upper bound on what the song could contribute — not the subset any
        particular window makes audible, which is not known until the window is drawn.

        Args:
            rng: the item's Generator.
            eligible: song indices from `_eligible_anchor_songs` (non-empty).
            remaining: sorted unfilled class slots (R).
        """
        mode = self.cfg.anchor_selection
        if mode == "uniform":
            return int(eligible[rng.integers(eligible.size)])
        overlap = self._song_classes[eligible][:, remaining].sum(axis=1)
        if mode == "overlap_weighted":
            return int(eligible[rng.choice(eligible.size, p=overlap / overlap.sum())])
        if mode == "greedy":
            tied = eligible[overlap == overlap.max()]
            return int(tied[rng.integers(tied.size)])
        raise ValueError(f"anchor_selection={mode!r}")   # unreachable after init guard

    def _cluster_window_start(self, rng: np.random.Generator, span: np.ndarray,
                              max_frames: int) -> int:
        """Window start (frames) that keeps the chosen joint span fully in play.

        If the span is longer than the window, the window lands uniformly INSIDE it
        (both instruments active wall-to-wall). Otherwise the window is placed uniformly
        among positions CONTAINING the whole span — since qualifying spans are longer
        than coverage_threshold × segment, containment alone guarantees both of the
        span's instruments clear the audibility test. No rejection sampling anywhere.

        Args:
            rng: the item's Generator.
            span: [start_s, end_s] of the chosen joint-activity span.
            max_frames: the anchor song's longest stem, in frames.
        """
        segment_s = self.cfg.segment_seconds
        span_start, span_end = float(span[0]), float(span[1])
        max_start_s = max(0.0, max_frames / self.cfg.sample_rate - segment_s)
        if span_end - span_start >= segment_s:
            start_s = rng.uniform(span_start, span_end - segment_s)
        else:
            low = max(0.0, span_end - segment_s)
            high = min(span_start, max_start_s)
            start_s = rng.uniform(low, max(low, high))
        max_start = max(0, max_frames - self.chunk_frames)
        return int(np.clip(round(start_s * self.cfg.sample_rate), 0, max_start))

    def _audible_in_window(self, song: SongEntry, start_frame: int,
                           remaining: list) -> dict:
        """Which unfilled classes this song makes audible here → {slot: [entries]}.

        "Audible" reuses the project's one definition — activity covering more than
        `density_coverage_threshold` of the window, the same test `chunk_activities`
        and the measured density mode apply. A class qualifies if at least one of its
        files clears the bar, and only the clearing files are drawable, so a cluster
        member is always a concrete file that actually plays here.

        Args:
            song: the anchor song.
            start_frame: the cluster window's start.
            remaining: unfilled class slots to test (others are not this cluster's
                business — they are already clustered or will draw incoherently).
        """
        window_start_s = start_frame / self.cfg.sample_rate
        window_end_s = window_start_s + self.cfg.segment_seconds
        audible: dict = {}
        for slot in remaining:
            entries = song.entries_by_class.get(self.cfg.classes[slot])
            if not entries:
                continue
            clearing = [e for e in entries
                        if self._window_coverage(e.segments, window_start_s,
                                                 window_end_s)
                        > self.cfg.density_coverage_threshold]
            if clearing:
                audible[slot] = clearing
        return audible

    def _draw_cluster(self, rng: np.random.Generator, song_index: int, remaining: list,
                      need: int, cluster_index: int) -> tuple:
        """One anchor song → one cluster: members, shared offset, per-member picks.

        Draw order (fixed; every step below consumes the item RNG deterministically):
        joint span duration-weighted over the song's VALID seed pairs within R (melodic
        minimum, → `_seed_pair_valid`) → window start (containment rule) → members (all
        audible R-classes incl. percussion, thinned to `need` by `_thin_members`, which
        protects the melodic minimum) → shared gain / shared swap if configured → per
        member in slot order: file among clearing candidates, gain if not shared, swap
        if not shared, EQ. Guaranteed ≥2 members satisfying the minimum: the chosen
        span's own pair is audible by construction.

        Args:
            rng: the item's Generator.
            song_index: the anchor, from `_select_anchor`.
            remaining: sorted unfilled class slots (R).
            need: how many members are still wanted (r).
            cluster_index: position this cluster will take in the plan.

        Returns:
            (cluster, picks, spares) — spares = audible-but-unchosen {slot: [entries]},
            kept so the r==1 rule can grow this cluster by one member.
        """
        song = self._songs[song_index]
        pair_intervals = self._pair_intervals[song_index]
        # only VALID seed pairs pick the window: joint-span geometry is computed on the
        # melodic members, so a percussion class can never be why the cluster exists
        available = [pair for pair in combinations(remaining, 2)
                     if pair in pair_intervals and self._seed_pair_valid(pair)]
        spans = np.vstack([pair_intervals[pair] for pair in available])
        durations = spans[:, 1] - spans[:, 0]
        span = spans[rng.choice(len(spans), p=durations / durations.sum())]
        start_frame = self._cluster_window_start(rng, span, song.max_frames)

        audible = self._audible_in_window(song, start_frame, remaining)
        slots = sorted(audible)
        members = self._thin_members(rng, slots, min(len(slots), need))
        spares = {slot: audible[slot] for slot in slots if slot not in members}

        shared_gain = (float(rng.uniform(self.cfg.gain_min, self.cfg.gain_max))
                       if self.cfg.cluster_shared_gain else None)
        shared_swap = (bool(rng.random() < self.cfg.channel_swap_prob)
                       if self.cfg.cluster_shared_channel_swap else None)

        picks = []
        for slot in members:
            candidates = audible[slot]
            entry = candidates[rng.integers(len(candidates))]
            class_name = self.cfg.classes[slot]
            gain = (shared_gain if shared_gain is not None
                    else float(self._draw_stem_gain(rng, class_name, entry)))
            swap = (shared_swap if shared_swap is not None
                    else bool(rng.random() < self.cfg.channel_swap_prob))
            board = (build_random_eq(rng, self.cfg.eq_stem_gain_db, q_max=5.0)
                     if rng.random() < self.cfg.eq_stem_prob else None)
            picks.append(StemPick(slot, class_name, entry, gain, start_frame, swap,
                                  board, cluster_index))

        cluster = ClusterPlan(song_id=song.song_id, genre_sub=song.genre_sub,
                              start_frame=start_frame, member_slots=list(members),
                              shared_gain=shared_gain, shared_swap=shared_swap)
        return cluster, picks, spares

    def _thin_members(self, rng: np.random.Generator, slots: list, m: int) -> list:
        """m of the audible classes, never dropping below the melodic minimum.

        The window's seed pair guarantees ≥ cluster_min_melodic melodic classes are in
        `slots`; the required melodic members are drawn first, the rest uniformly from
        everything left. This is NOT uniform over all valid subsets — melodic-heavier
        subsets are slightly favoured (each weighted by how many ways its melodic part
        could have been the required draw) — a documented, rule-conservative bias.
        A direct consequence at m == cluster_min_melodic: the cluster is all-melodic,
        so 타악기 only ever joins clusters of at least three.

        Args:
            rng: the item's Generator.
            slots: sorted audible class slots.
            m: target member count (≥ cluster_min_melodic whenever ≥ 2).
        """
        if len(slots) <= m:
            return list(slots)
        melodic = [s for s in slots if s in self.melodic_slots]
        required = min(int(self.cfg.cluster_min_melodic), m)
        picked = ([melodic[i] for i in rng.choice(len(melodic), size=required,
                                                  replace=False)]
                  if required else [])
        rest = [s for s in slots if s not in picked]
        if m - required:
            picked += [rest[i] for i in rng.choice(len(rest), size=m - required,
                                                   replace=False)]
        return sorted(picked)

    def _grow_last_cluster(self, rng: np.random.Generator, cluster: ClusterPlan,
                           spares: dict, cluster_index: int) -> StemPick | None:
        """The r==1 rule: extend the last cluster by one audible spare class, if any.

        A leftover need of one cannot start a new cluster (one stem is coherent with
        nothing), so it either joins the last cluster — the spare is audible in that
        cluster's own window, so coherence is genuine — or is dropped to the incoherent
        pool. Shared gain/swap are inherited so the cluster stays internally uniform.

        Args:
            rng: the item's Generator.
            cluster: the last cluster built.
            spares: audible-but-unchosen {slot: [entries]} from that cluster's window.
            cluster_index: that cluster's position in the plan.

        Returns:
            The grown member's StemPick, or None if the window had no spare.
        """
        if not spares:
            return None
        slots = sorted(spares)
        slot = slots[int(rng.integers(len(slots)))]
        candidates = spares[slot]
        entry = candidates[rng.integers(len(candidates))]
        class_name = self.cfg.classes[slot]
        gain = (cluster.shared_gain if cluster.shared_gain is not None
                else float(self._draw_stem_gain(rng, class_name, entry)))
        swap = (cluster.shared_swap if cluster.shared_swap is not None
                else bool(rng.random() < self.cfg.channel_swap_prob))
        board = (build_random_eq(rng, self.cfg.eq_stem_gain_db, q_max=5.0)
                 if rng.random() < self.cfg.eq_stem_prob else None)
        cluster.member_slots.append(slot)
        return StemPick(slot, class_name, entry, gain, cluster.start_frame, swap,
                        board, cluster_index)

    def _fill_clusters(self, rng: np.random.Generator, drawn_slots: np.ndarray,
                       k_declared: int) -> tuple[list, list, list]:
        """Step 4 — cover k of the drawn classes with coherent clusters.

        R starts as all drawn classes and r as k. Each iteration draws one eligible
        anchor (per `anchor_selection`) and takes one cluster from it; clusters always
        hold ≥2 members, so r strictly falls. Exits: filled (r=0), r=1 (grow-or-drop
        rule), no eligible anchor (shortfall — NEVER resample S, that is the exposure
        leak this sampler exists to close), or greedy's single-anchor stop.

        Args:
            rng: the item's Generator.
            drawn_slots: S, in draw order.
            k_declared: k from `_draw_coherent_count` (≥2 when called).

        Returns:
            (clusters, picks, shortfall_reasons).
        """
        remaining = sorted(int(slot) for slot in drawn_slots)
        r = k_declared
        clusters: list = []
        picks: list = []
        reasons: list = []
        last_spares: dict = {}
        while r >= 2:
            eligible = self._eligible_anchor_songs(remaining)
            if eligible.size == 0:
                reasons.append("no_eligible_anchor")
                break
            song_index = self._select_anchor(rng, eligible, remaining)
            cluster, cluster_picks, last_spares = self._draw_cluster(
                rng, song_index, remaining, r, cluster_index=len(clusters))
            clusters.append(cluster)
            picks.extend(cluster_picks)
            for pick in cluster_picks:
                remaining.remove(pick.slot)
            r -= len(cluster_picks)
            if self.cfg.anchor_selection == "greedy":
                if r >= 2:
                    reasons.append("greedy_single_anchor")   # Method A accepts this
                break
        if r == 1 and clusters:
            grown = self._grow_last_cluster(rng, clusters[-1], last_spares,
                                            cluster_index=len(clusters) - 1)
            if grown is not None:
                picks.append(grown)
                remaining.remove(grown.slot)
            else:
                reasons.append("r1_dropped")
        return clusters, picks, reasons

    # --- the per-item plan (all RNG, no audio) ---
    def _plan_item(self, rng: np.random.Generator) -> MixPlan:
        """Decide everything about one mix, without touching a single audio file.

        The module docstring's five steps, in order, plus the trailing mixbus-EQ gate.
        Incoherent stems are planned in S's draw order with exactly the per-stem RNG
        sequence the pre-cluster code used, so a p=0 item is bit-identical to exp002's
        (verified against the exp002-era module — scripts/verify_coherent_sampler.py).

        Split out from the audio work so that verification and characterisation can pull
        hundreds of thousands of draws through the REAL sampling code rather than a
        reimplementation of it.
        """
        p = float(self.cfg.coherent_mix_prob)   # per-draw read — the curriculum seam
        n = int(rng.choice(self.density_values, p=self.density_probs))
        drawn = rng.choice(len(self.cfg.classes), size=n, replace=False)

        k_declared = self._draw_coherent_count(rng, p, n)
        if k_declared >= 2:
            self._ensure_anchor_pool()
            clusters, picks, reasons = self._fill_clusters(rng, drawn, k_declared)
        else:
            clusters, picks, reasons = [], [], []

        clustered_slots = {pick.slot for pick in picks}
        for slot in drawn:
            if int(slot) not in clustered_slots:
                picks.append(self._plan_incoherent_stem(rng, int(slot)))

        mixbus_board = (build_random_eq(rng, self.cfg.eq_mixbus_gain_db, q_max=3.0)
                        if rng.random() < self.cfg.eq_mixbus_prob else None)
        # recording-condition stages LAST in the stream, so adding them cannot shift a
        # single draw of the mixing sampler above — that is what makes an existing config
        # bit-identical rather than merely statistically equivalent
        recording = recording_aug.plan_recording(
            rng, self.recording_cfg, self.rir_pool, self.noise_pool,
            slots=tuple(sorted(pick.slot for pick in picks)),
            cluster_of_slot={pick.slot: pick.cluster_index for pick in picks
                             if pick.cluster_index is not None},
            sample_rate=self.cfg.sample_rate, segment_seconds=self.cfg.segment_seconds,
            chunk_frames=self.chunk_frames)
        return MixPlan(p=p, n=n, drawn_slots=tuple(int(s) for s in drawn),
                       k_target=p * n, k_declared=k_declared,
                       k_realised=sum(len(c.member_slots) for c in clusters),
                       clusters=tuple(clusters), picks=tuple(picks),
                       shortfall_reasons=tuple(reasons), mixbus_board=mixbus_board,
                       recording=recording)

    def _plan_incoherent_stem(self, rng: np.random.Generator, slot: int) -> StemPick:
        """One incoherent stem, fully decided: independent song, offset, augmentation.

        RNG order is the contract (file → window → gain → swap → EQ): it must match the
        pre-cluster incoherent path draw for draw, or exp002 stops being reproducible.

        Args:
            rng: the item's Generator.
            slot: output slot = index into cfg.classes.
        """
        class_name = self.cfg.classes[slot]
        entries = self.pool[class_name]
        entry = entries[rng.integers(len(entries))]
        start_frame = self._draw_window_start(rng, entry.segments, entry.out_frames)
        gain = float(self._draw_stem_gain(rng, class_name, entry))
        swap = bool(rng.random() < self.cfg.channel_swap_prob)
        board = (build_random_eq(rng, self.cfg.eq_stem_gain_db, q_max=5.0)
                 if rng.random() < self.cfg.eq_stem_prob else None)
        return StemPick(slot, class_name, entry, gain, start_frame, swap, board, None)

    def plan_item(self, index: int) -> MixPlan:
        """Provenance of item `index` — the full plan, before any audio is read.

        Runs the identical RNG sequence `__getitem__` runs, so the plan it returns is
        the one that item really uses; it just stops before decoding. Verification and
        characterisation entry point.

        Args:
            index: the dataset index, exactly as the DataLoader would pass it.
        """
        rng = np.random.default_rng([self.cfg.seed, index])
        return self._plan_item(rng)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # independent, reproducible stream per (seed, item) — worker-count-agnostic
        rng = np.random.default_rng([self.cfg.seed, index])
        plan = self._plan_item(rng)

        stems = np.zeros((len(self.cfg.classes), 2, self.chunk_frames), dtype=np.float32)
        for pick in plan.picks:
            # np.float32 cast: a float64 scalar would silently promote the whole chain
            audio = self._read_window(pick.entry, pick.start_frame) * np.float32(pick.gain)
            if pick.swap_channels:
                audio = audio[::-1].copy()
            if pick.eq_board is not None:
                audio = pick.eq_board(audio, self.cfg.sample_rate)
            stems[pick.slot] = audio

        if plan.mixbus_board is not None:
            # one shared EQ over every stem: linear, so the mixture hears the same EQ
            # and mixture ≡ Σ(targets) survives
            for pick in plan.picks:
                stems[pick.slot] = plan.mixbus_board(stems[pick.slot],
                                                     self.cfg.sample_rate)

        if plan.recording is not None:
            stems, mixture = self._realise_recording(plan, stems)
        else:
            mixture = stems.sum(axis=0)
            stems, mixture = self._normalize(stems, mixture)
        return torch.from_numpy(stems), torch.from_numpy(mixture)

    # --- recording-condition realisation (stages 1–5) ---
    def _realise_recording(self, plan: MixPlan,
                           stems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the planned recording stages and return (stems, mixture).

        Order is the report's §5.1 pipeline: capture chain → room → level → sum → mix bus
        → degradation → normalization. Only the DRAWN slots are processed; the rest of the
        tensor is silence and convolving nine zeros would be pure cost (n averages 5.5
        under n ~ U{2..9}, so this is a 40% saving on every FFT in the item).

        Stages 1–3 are linear and are applied to the TARGETS, with the mixture rebuilt as
        their sum afterwards, so mixture ≡ Σ(targets) is true by construction rather than
        by cancellation. Only stage 4 (nonlinear bus) and stage 5 (mixture-only
        degradation) need the explicit target policies in recording_aug.

        Args:
            plan: the item's MixPlan, whose `recording` field is not None.
            stems: (n_classes, 2, chunk) with the drawn slots already filled.
        """
        rec = plan.recording
        cfg = self.recording_cfg
        slots = sorted(pick.slot for pick in plan.picks)
        active = stems[slots]                                    # (n, 2, chunk) copy
        position_of = {slot: index for index, slot in enumerate(slots)}

        # --- stage 1: capture chain (minimum-phase biquads; linear → targets follow) ---
        for group, sos in rec.device:
            rows = [position_of[slot] for slot in group]
            # sosfilt promotes to the SOS dtype (float64); cast back so the chain stays
            # float32 end to end, as the gain path's np.float32 casts already ensure
            active[rows] = scipy.signal.sosfilt(sos, active[rows],
                                                axis=-1).astype(np.float32)

        # --- stage 2: room ---
        dry = None
        if rec.rir is not None:
            if rec.rir.target_mode == "dry_source":
                dry = active.copy()      # the dereverberation arm keeps the dry targets
            rows_per_stem = [0] * len(slots)
            for assignment in rec.rir.assignments:
                for slot in assignment.slots:
                    rows_per_stem[position_of[slot]] = assignment.row
            active = self.rir_pool.convolve(active, rows_per_stem)

        # --- stage 3: level automation (before the sum, so the sum stays exact) ---
        for envelope in rec.envelopes:
            curve = recording_aug.realise_envelope(envelope, self.chunk_frames,
                                                   self.cfg.sample_rate)
            rows = (slice(None) if envelope.slots is None
                    else [position_of[slot] for slot in envelope.slots])
            active[rows] = active[rows] * curve
            if dry is not None:
                dry[rows] = dry[rows] * curve     # shared level rides both task views

        stems[slots] = active if dry is None else dry
        mixture = active.sum(axis=0)

        # --- stage 4: mix bus ---
        if rec.bus is not None:
            stems, mixture = self._apply_bus(rec.bus, stems, mixture, slots)
        else:
            stems, mixture = self._normalize(stems, mixture)

        # --- stage 5: degrade the MIXTURE only (targets stay clean, sum breaks) ---
        if rec.dirty is not None:
            mixture = self._apply_dirty(rec.dirty, mixture)

        if cfg.assert_invariant and cfg.coherence_mode == "strict_sum":
            error = float(np.abs(mixture - stems.sum(axis=0)).max())
            if error >= cfg.silence_eps:
                raise AssertionError(
                    f"strict_sum violated: max|mixture - Σstems| = {error:.3e} "
                    f">= silence_eps {cfg.silence_eps:.3e}")
        return np.ascontiguousarray(stems), np.ascontiguousarray(mixture)

    def _apply_bus(self, bus, stems: np.ndarray, mixture: np.ndarray,
                   slots: list) -> tuple[np.ndarray, np.ndarray]:
        """Stage 4 — pre-bus loudness, limiter, and the §4.3 target allocation.

        The pre-bus normalization is what carries the loudness diversity: `target_lufs`
        is nan when `loudness_mode: fixed`, in which case the dataset's own −19 LUFS is
        used and only the limiter's waveform shaping varies. After the limiter the targets
        are bus-affected source images r·s_i, and the float residual is pushed onto the
        locally dominant stem so that mixture ≡ Σ(targets) is restored EXACTLY.
        """
        target_lufs = (self.cfg.target_lufs if math.isnan(bus.target_lufs)
                       else bus.target_lufs)
        stems, mixture = self._loudnorm(stems, mixture, target_lufs)
        if bus.release_ms is not None:
            gain = recording_aug.limiter_gain(mixture, bus.threshold_dbfs, bus.attack_ms,
                                              bus.release_ms, self.cfg.sample_rate)
            limited = mixture * gain
            ratio = recording_aug.bus_ratio(mixture, limited,
                                            self.recording_cfg.silence_eps)
            # fancy indexing returns a COPY, so the allocation is edited and written back
            # in one place rather than through a view that does not exist
            allocated = stems[slots] * ratio
            if bus.residual_correction:
                recording_aug.add_residual(allocated, limited - allocated.sum(axis=0))
            stems[slots] = allocated
            mixture = allocated.sum(axis=0)
        if bus.normalize_after_bus:
            stems, mixture = self._loudnorm(stems, mixture, self.cfg.target_lufs)
        return self._peak_guard(stems, mixture)

    def _apply_dirty(self, dirty, mixture: np.ndarray) -> np.ndarray:
        """Stage 5 — noise → resample → quantize → clip, on the mixture alone.

        Report §5.1's order. Every one of these is a mixture-only operation and the
        targets deliberately stay clean, which is why it is unreachable outside
        `coherence_mode: clean_under_degraded_input`.
        """
        if dirty.noise_row is not None:
            noise = self.noise_pool.read(dirty.noise_row, dirty.noise_start,
                                         self.chunk_frames)
            signal_rms = float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2)))
            noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
            if noise_rms > 0 and signal_rms > 0:
                wanted = signal_rms / (10 ** (dirty.noise_snr_db / 20.0))
                mixture = mixture + noise * np.float32(wanted / noise_rms)
        if dirty.resample_hz is not None:
            mixture = recording_aug.apply_resample_roundtrip(
                mixture, self.cfg.sample_rate, dirty.resample_hz)
        if dirty.quantize_bits is not None:
            mixture = recording_aug.apply_quantize(mixture, dirty.quantize_bits)
        if dirty.clip_dbfs is not None:
            mixture = recording_aug.apply_clip(mixture, dirty.clip_dbfs)
        return mixture

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
