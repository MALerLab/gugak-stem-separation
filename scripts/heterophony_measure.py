"""Quantify heterophony in the ensemble dataset — pairwise pitch agreement between stems.

For every song in a genre-stratified sample, every pair of MELODIC stems (피리 · 대금 ·
해금 · 아쟁 · 가야금 · 거문고 · 양금; percussion and 기타 excluded; every part counted, so
피리/피리2/피리3 form within-class pairs) is compared frame by frame over the frames where
BOTH stems are active:

  U_pc100   fraction with pitch-class distance ≤ 100 cents   (headline)
  U_pc50    fraction with pitch-class distance ≤ 50 cents
  U_abs100  fraction with unfolded |Δcents| ≤ 100            (true unison, no octave fold)
  hist      folded Δcents histogram, 10-cent bins, 0–600

f0 = librosa.pyin on the ingested 44.1 kHz stems (mono downmix), hop 10 ms, 60–2000 Hz.
active = pyin voiced flag AND frame RMS > −60 dBFS.

Chance baseline: for every real pair (class A, class B) one INCOHERENT pair of the same
class pair is drawn from two different songs' stems (within-class pairs → two different
songs' stems of that same class), at a random time offset. Same pair count as the real
set, so the two tables are directly comparable.

CPU-only, read-only on the data. Outputs → paper/ismir2026_lbd/numbers/:
  heterophony.csv        per (song, stem pair): frames_coactive, U_pc100, U_pc50, U_abs100
  heterophony.md         overall / per genre / per class pair, real vs chance,
                         frame-weighted and song-mean
  heterophony_hist.csv   folded Δcents histogram, real and chance
  provenance.md          appended: sample list, params, runtime, commit

Run:
    uv run python scripts/heterophony_measure.py --pilot 5
    uv run python scripts/heterophony_measure.py --target 60
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# one BLAS thread per worker — pyin's cost is numba Viterbi, not BLAS, and 24 workers ×
# 24 threads would thrash
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "manifests/parquet/source_manifest.parquet"
NUMBERS_DIR = REPO_ROOT / "paper/ismir2026_lbd/numbers"

MELODIC_CLASSES = ("피리", "대금", "해금", "아쟁", "가야금", "거문고", "양금")
ZERO_PAIR_GENRE = "산조"  # one melodic instrument by construction → zero pairs, expected

# --- f0 / activity parameters (task spec) ---
SAMPLE_RATE = 44100
HOP_SECONDS = 0.010
HOP_LENGTH = int(round(SAMPLE_RATE * HOP_SECONDS))  # 441 samples
FRAME_LENGTH = 2048  # 46 ms analysis window; ≥ 2 periods of fmin at 44.1 kHz
F0_MIN_HZ = 60.0
F0_MAX_HZ = 2000.0
ACTIVE_RMS_DBFS = -60.0

# --- agreement thresholds / histogram ---
PC_THRESHOLD_HEADLINE = 100.0
PC_THRESHOLD_TIGHT = 50.0
ABS_THRESHOLD = 100.0
HIST_BIN_CENTS = 10
HIST_EDGES = np.arange(0, 600 + HIST_BIN_CENTS, HIST_BIN_CENTS)  # folded distance ∈ [0, 600]

SEED = 42
MIN_PER_GENRE = 3


@dataclass(frozen=True)
class Stem:
    """One melodic stem of one song, as read from the source manifest."""
    file_id: str
    song_id: str
    genre: str
    stem_class: str
    instrument_raw: str
    path: Path


# ----------------------------------------------------------------------------------
# sample selection
# ----------------------------------------------------------------------------------
def load_melodic_stems() -> pd.DataFrame:
    """Ensemble-dataset stems in the seven melodic classes, one row per stem file."""
    manifest = pd.read_parquet(MANIFEST_PATH)
    stems = manifest[
        (manifest.dataset == "71955")
        & (manifest.role == "stem")
        & (manifest.stem_group.isin(MELODIC_CLASSES))
    ]
    columns = ["file_id", "song_id", "genre_sub", "stem_group", "instrument_raw", "out_path"]
    return stems[columns].sort_values(["song_id", "stem_group", "instrument_raw"]).reset_index(drop=True)


def song_table(stems: pd.DataFrame) -> pd.DataFrame:
    """One row per song: genre + number of melodic stems."""
    return stems.groupby("song_id").agg(genre=("genre_sub", "first"), num_melodic=("file_id", "size")).reset_index()


def stratified_sample(songs: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Genre-stratified song sample: ≥ MIN_PER_GENRE per genre, the rest proportional.

    산조 is fixed at MIN_PER_GENRE (single melodic instrument → zero pairs). For every other
    genre the pool is restricted to songs with ≥ 2 melodic stems so no slot is spent on a
    song that cannot form a pair.
    """
    rng = np.random.default_rng(seed)
    pool = songs[(songs.num_melodic >= 2) | (songs.genre == ZERO_PAIR_GENRE)]
    counts = pool.genre.value_counts()
    quota: dict[str, int] = {ZERO_PAIR_GENRE: MIN_PER_GENRE}
    remaining = target - MIN_PER_GENRE
    others = counts.drop(ZERO_PAIR_GENRE, errors="ignore")
    for genre, count in others.items():
        quota[genre] = max(MIN_PER_GENRE, int(round(remaining * count / others.sum())))
    picked = []
    for genre, num in quota.items():
        candidates = pool[pool.genre == genre]
        take = min(num, len(candidates))
        picked.append(candidates.iloc[rng.choice(len(candidates), size=take, replace=False)])
    return pd.concat(picked).sort_values("song_id").reset_index(drop=True)


def pilot_sample(songs: pd.DataFrame, num: int, seed: int) -> pd.DataFrame:
    """Small pilot: `num` songs with ≥ 2 melodic stems, spread over genres round-robin."""
    rng = np.random.default_rng(seed)
    pool = songs[songs.num_melodic >= 2]
    by_genre = {g: d.iloc[rng.permutation(len(d))] for g, d in pool.groupby("genre")}
    picked, cursor = [], 0
    while len(picked) < num:
        for genre in sorted(by_genre):
            if cursor < len(by_genre[genre]) and len(picked) < num:
                picked.append(by_genre[genre].iloc[cursor])
        cursor += 1
    return pd.DataFrame(picked).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# f0 extraction (one process per stem)
# ----------------------------------------------------------------------------------
def cache_path(cache_dir: Path, file_id: str) -> Path:
    return cache_dir / (file_id.replace("/", "__").replace(":", "_") + ".npz")


def extract_f0(stem_path: str, out_path: str) -> float:
    """pyin f0 + voiced flag + frame RMS for one stem → npz. Returns wall seconds."""
    import librosa
    import soundfile as sf

    started = time.perf_counter()
    audio, sr = sf.read(stem_path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, f"{stem_path}: sr {sr} ≠ {SAMPLE_RATE}"
    mono = audio.mean(axis=1)  # ensemble ingest verified: no anti-phase survivors
    f0, voiced_flag, _ = librosa.pyin(
        mono, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True,
    )
    rms = librosa.feature.rms(y=mono, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    num_frames = min(len(f0), len(rms))
    np.savez_compressed(out_path, f0=f0[:num_frames].astype(np.float32),
                        voiced=voiced_flag[:num_frames], rms=rms[:num_frames].astype(np.float32))
    return time.perf_counter() - started


def compute_all_f0(stems: list[Stem], cache_dir: Path, workers: int) -> tuple[float, int]:
    """Run pyin over every uncached stem in parallel. Returns (wall s, num computed)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not cache_path(cache_dir, s.file_id).exists()]
    print(f"f0: {len(stems)} stems, {len(todo)} to compute on {workers} workers", flush=True)
    started = time.perf_counter()
    per_stem_seconds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_f0, str(s.path), str(cache_path(cache_dir, s.file_id))): s for s in todo}
        for done, future in enumerate(as_completed(futures), 1):
            per_stem_seconds.append(future.result())
            if done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} stems, mean {np.mean(per_stem_seconds):.1f} s/stem", flush=True)
    return time.perf_counter() - started, len(todo)


def load_track(cache_dir: Path, stem: Stem) -> tuple[np.ndarray, np.ndarray]:
    """(cents, active) per frame. cents = 1200·log2(f0/440) (nan when unvoiced)."""
    blob = np.load(cache_path(cache_dir, stem.file_id))
    f0 = blob["f0"].astype(np.float64)
    rms_dbfs = 20 * np.log10(np.maximum(blob["rms"].astype(np.float64), 1e-12))
    active = blob["voiced"] & (rms_dbfs > ACTIVE_RMS_DBFS) & np.isfinite(f0) & (f0 > 0)
    cents = np.full_like(f0, np.nan)
    cents[active] = 1200.0 * np.log2(f0[active] / 440.0)
    return cents, active


# ----------------------------------------------------------------------------------
# pairwise agreement
# ----------------------------------------------------------------------------------
def pair_metrics(cents_a: np.ndarray, active_a: np.ndarray,
                 cents_b: np.ndarray, active_b: np.ndarray) -> dict:
    """Agreement over co-active frames of two aligned tracks (already same length)."""
    coactive = active_a & active_b
    delta = np.abs(cents_a[coactive] - cents_b[coactive])
    folded = np.mod(delta, 1200.0)
    folded = np.minimum(folded, 1200.0 - folded)
    hist, _ = np.histogram(folded, bins=HIST_EDGES)
    num = int(coactive.sum())
    return {
        "frames_coactive": num,
        "frames_a_active": int(active_a.sum()),
        "frames_b_active": int(active_b.sum()),
        "hits_pc100": int((folded <= PC_THRESHOLD_HEADLINE).sum()),
        "hits_pc50": int((folded <= PC_THRESHOLD_TIGHT).sum()),
        "hits_abs100": int((delta <= ABS_THRESHOLD).sum()),
        "U_pc100": (folded <= PC_THRESHOLD_HEADLINE).mean() if num else np.nan,
        "U_pc50": (folded <= PC_THRESHOLD_TIGHT).mean() if num else np.nan,
        "U_abs100": (delta <= ABS_THRESHOLD).mean() if num else np.nan,
        "hist": hist,
    }


def class_pair_key(class_a: str, class_b: str) -> str:
    return "×".join(sorted((class_a, class_b)))


def real_pairs(stems_by_song: dict[str, list[Stem]], tracks: dict[str, tuple]) -> list[dict]:
    """All stem pairs within each song, trimmed to the shorter track (stems start-aligned)."""
    rows = []
    for song_id, stems in stems_by_song.items():
        for a, b in itertools.combinations(stems, 2):
            cents_a, active_a = tracks[a.file_id]
            cents_b, active_b = tracks[b.file_id]
            length = min(len(cents_a), len(cents_b))
            metrics = pair_metrics(cents_a[:length], active_a[:length], cents_b[:length], active_b[:length])
            rows.append({"song_id": song_id, "genre": a.genre, "source": "real",
                         "stem_a": a.instrument_raw, "stem_b": b.instrument_raw,
                         "class_a": a.stem_class, "class_b": b.stem_class,
                         "class_pair": class_pair_key(a.stem_class, b.stem_class),
                         "pair_kind": "within" if a.stem_class == b.stem_class else "cross",
                         "chance_song_a": "", "chance_song_b": "", "offset_frames": 0, **metrics})
    return rows


def chance_pairs(real_rows: list[dict], all_stems: list[Stem], tracks: dict[str, tuple], seed: int) -> list[dict]:
    """One incoherent twin per real pair: same class pair, two different songs, random offset."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Stem]] = {}
    for stem in all_stems:
        by_class.setdefault(stem.stem_class, []).append(stem)
    rows = []
    for real in real_rows:
        class_a, class_b = real["class_a"], real["class_b"]
        while True:
            a = by_class[class_a][rng.integers(len(by_class[class_a]))]
            b = by_class[class_b][rng.integers(len(by_class[class_b]))]
            if a.song_id != b.song_id:
                break
        cents_a, active_a = tracks[a.file_id]
        cents_b, active_b = tracks[b.file_id]
        # slide the shorter track to a random position inside the longer one
        if len(cents_a) < len(cents_b):
            cents_a, active_a, cents_b, active_b = cents_b, active_b, cents_a, active_a
            a, b = b, a
        length = len(cents_b)
        offset = int(rng.integers(0, len(cents_a) - length + 1))
        metrics = pair_metrics(cents_a[offset:offset + length], active_a[offset:offset + length], cents_b, active_b)
        rows.append({**{k: real[k] for k in ("song_id", "genre", "class_a", "class_b", "class_pair", "pair_kind")},
                     "source": "chance", "stem_a": a.instrument_raw, "stem_b": b.instrument_raw,
                     "chance_song_a": a.song_id, "chance_song_b": b.song_id, "offset_frames": offset, **metrics})
    return rows


# ----------------------------------------------------------------------------------
# aggregation + reporting
# ----------------------------------------------------------------------------------
METRICS = ("U_pc100", "U_pc50", "U_abs100")


def frame_weighted(rows: pd.DataFrame) -> pd.Series:
    """Σhits / Σcoactive frames over the given pair rows."""
    total = rows.frames_coactive.sum()
    out = {m: rows[f"hits_{m[2:]}"].sum() / total if total else np.nan for m in METRICS}
    out["frames_coactive"] = total
    out["num_pairs"] = len(rows)
    return pd.Series(out)


def song_mean(rows: pd.DataFrame) -> pd.Series:
    """Per-song frame-weighted value, then the unweighted mean over songs."""
    per_song = rows.groupby("song_id").apply(frame_weighted, include_groups=False)
    out = per_song[list(METRICS)].mean()
    out["num_songs"] = per_song[list(METRICS)].dropna().shape[0]
    return out


def side_by_side(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Real vs chance table, frame-weighted and song-mean, grouped by `group_cols`."""
    parts = []
    for (source, *keys), block in rows.groupby(["source", *group_cols]):
        fw = frame_weighted(block)
        sm = song_mean(block)
        parts.append({**dict(zip(group_cols, keys)), "source": source,
                      "num_pairs": int(fw.num_pairs), "frames_coactive": int(fw.frames_coactive),
                      "num_songs": int(sm.num_songs),
                      **{f"{m}_frame": fw[m] for m in METRICS}, **{f"{m}_song": sm[m] for m in METRICS}})
    table = pd.DataFrame(parts)
    order = {"real": 0, "chance": 1}
    return table.sort_values([*group_cols, "source"], key=lambda s: s.map(order) if s.name == "source" else s)


def md_table(table: pd.DataFrame, group_cols: list[str]) -> str:
    """Markdown with real/chance as adjacent rows and Δ (real − chance) on the chance row."""
    head = [*group_cols, "src", "pairs", "frames", "U_pc100 fw", "U_pc50 fw", "U_abs100 fw",
            "U_pc100 song", "U_pc50 song", "U_abs100 song", "Δpc100 fw"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, group in table.groupby(group_cols, sort=False) if group_cols else [((), table)]:
        real = group[group.source == "real"].iloc[0]
        for _, row in group.iterrows():
            delta = "" if row.source == "real" else f"{real.U_pc100_frame - row.U_pc100_frame:+.3f}"
            cells = [str(row[c]) for c in group_cols] + [row.source, str(row.num_pairs), f"{row.frames_coactive:,}"]
            cells += [f"{row[f'{m}_frame']:.3f}" for m in METRICS] + [f"{row[f'{m}_song']:.3f}" for m in METRICS] + [delta]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(rows: pd.DataFrame, sample: pd.DataFrame, out_path: Path, runtime: dict) -> None:
    rows = rows[rows.frames_coactive > 0]
    sections = [
        "# Heterophony — pairwise pitch agreement between melodic stems (ensemble dataset)\n",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/heterophony_measure.py`. "
        f"Sample: {len(sample)} songs, {len(rows[rows.source == 'real'])} real stem pairs + the same number "
        "of chance pairs. Pairs with zero co-active frames are dropped from every mean.\n",
        "**Definitions.** `U_pc100` = fraction of co-active frames whose pitch-class distance "
        "(|Δcents| folded mod 1200, then min(d, 1200−d)) is ≤ 100 cents; `U_pc50` same at ≤ 50; "
        "`U_abs100` = unfolded |Δcents| ≤ 100 (true unison, no octave fold). "
        "*fw* = frame-weighted (Σhits / Σco-active frames over all pairs in the cell); "
        "*song* = per-song frame-weighted value, then unweighted mean over songs. "
        "`Δpc100 fw` = real − chance. Chance = same class pair, stems from two different songs, random offset.\n",
        "## Overall (all pairs pooled)\n", md_table(side_by_side(rows, []), []),
        "\n## By pair kind (within-class = same instrument class, e.g. 피리×피리2)\n",
        md_table(side_by_side(rows, ["pair_kind"]), ["pair_kind"]),
        "\n## Per genre (all pairs)\n", md_table(side_by_side(rows, ["genre"]), ["genre"]),
        "\n## Per genre × pair kind\n", md_table(side_by_side(rows, ["genre", "pair_kind"]), ["genre", "pair_kind"]),
        "\n## Per class pair\n", md_table(side_by_side(rows, ["class_pair"]), ["class_pair"]),
        "\n## Runtime\n",
        "\n".join(f"- {k}: {v}" for k, v in runtime.items()),
    ]
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def write_histogram(rows: pd.DataFrame, out_path: Path) -> None:
    """Folded Δcents histogram, pooled over frames, per (source, pair_kind) + 'all'."""
    records = []
    for source, block in rows.groupby("source"):
        for kind in ("all", "within", "cross"):
            sub = block if kind == "all" else block[block.pair_kind == kind]
            counts = np.sum(np.stack(sub["hist"].tolist()), axis=0) if len(sub) else np.zeros(len(HIST_EDGES) - 1, int)
            for low, count in zip(HIST_EDGES[:-1], counts):
                records.append({"source": source, "pair_kind": kind, "bin_low_cents": int(low),
                                "bin_high_cents": int(low + HIST_BIN_CENTS), "frames": int(count),
                                "fraction": count / counts.sum() if counts.sum() else np.nan})
    pd.DataFrame(records).to_csv(out_path, index=False)


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def append_provenance(sample: pd.DataFrame, rows: pd.DataFrame, runtime: dict, workers: int, out_path: Path) -> None:
    real = rows[(rows.source == "real") & (rows.frames_coactive > 0)]
    genre_counts = sample.genre.value_counts().sort_index()
    text = [
        "\n## Heterophony measure (`heterophony.csv` / `.md` / `_hist.csv`)\n",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/heterophony_measure.py` "
        f"(untracked) at git commit `{git_commit()}`. CPU only, read-only on the data.",
        f"- **Sample:** {len(sample)} songs from `manifests/parquet/source_manifest.parquet` (ensemble dataset, "
        f"all splits), genre-stratified, seed {SEED}, ≥ {MIN_PER_GENRE} per genre; non-산조 pool restricted "
        "to songs with ≥ 2 melodic stems. Per genre: "
        + ", ".join(f"{g} {n}" for g, n in genre_counts.items()) + ".",
        f"- **Stems:** every stem in the seven melodic classes ({' · '.join(MELODIC_CLASSES)}); multi-part "
        "instruments (피리2, 가야금3 …) are distinct stems and form within-class pairs. 타악기 · 기타 · "
        "pitched_percussion excluded.",
        f"- **f0:** `librosa.pyin` {__import__('librosa').__version__}, ingested 44.1 kHz stems downmixed to mono "
        f"(L/R mean), hop {HOP_LENGTH} samples ({HOP_SECONDS * 1000:.0f} ms), frame {FRAME_LENGTH}, "
        f"fmin {F0_MIN_HZ:.0f} Hz, fmax {F0_MAX_HZ:.0f} Hz, default pyin priors/resolution (10 cents).",
        f"- **Active frame:** pyin voiced flag AND frame RMS > {ACTIVE_RMS_DBFS:.0f} dBFS (same frame/hop).",
        f"- **Pair metrics:** over frames where both stems are active; pitch-class distance = |Δcents| mod 1200 "
        f"folded to [0, 600]; thresholds {PC_THRESHOLD_HEADLINE:.0f} / {PC_THRESHOLD_TIGHT:.0f} cents (folded) and "
        f"{ABS_THRESHOLD:.0f} cents unfolded; histogram bins {HIST_BIN_CENTS} cents on the folded distance.",
        "- **Chance baseline:** for each real pair one pair of the same class pair drawn (seed "
        f"{SEED}) from two different songs of the same sample, shorter track placed at a uniform random offset "
        "inside the longer; within-class chance pairs = two different songs' stems of that class.",
        f"- **Counts:** {len(real)} real pairs with co-active frames ({int((real.pair_kind == 'within').sum())} "
        f"within-class, {int((real.pair_kind == 'cross').sum())} cross-class), "
        f"{int(real.frames_coactive.sum()):,} co-active frames.",
        "- **Runtime:** " + "; ".join(f"{k} {v}" for k, v in runtime.items()) + f"; {workers} worker processes.",
        "- **Songs:** " + ", ".join(sample.song_id.tolist()) + ".",
    ]
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(text) + "\n")


# ----------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=int, default=0, help="run on N songs only, write nothing to numbers/")
    parser.add_argument("--target", type=int, default=60, help="stratified sample size")
    parser.add_argument("--all", action="store_true", help="every song in the ensemble dataset")
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    parser.add_argument("--cache-dir", type=Path, required=True, help="npz cache for f0 tracks")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stems_table = load_melodic_stems()
    songs = song_table(stems_table)
    if args.pilot:
        sample = pilot_sample(songs, args.pilot, args.seed)
    elif args.all:
        sample = songs
    else:
        sample = stratified_sample(songs, args.target, args.seed)
    print(f"sample: {len(sample)} songs — " + ", ".join(f"{g} {n}" for g, n in sample.genre.value_counts().sort_index().items()))

    chosen = stems_table[stems_table.song_id.isin(sample.song_id)]
    stems = [Stem(r.file_id, r.song_id, r.genre_sub, r.stem_group, r.instrument_raw, REPO_ROOT / r.out_path)
             for r in chosen.itertuples()]
    total_audio_hours = chosen.merge(pd.read_parquet(MANIFEST_PATH)[["file_id", "out_duration"]], on="file_id").out_duration.sum() / 3600

    f0_seconds, num_computed = compute_all_f0(stems, args.cache_dir, args.workers)
    print(f"f0 done: {f0_seconds / 60:.1f} min wall for {num_computed} stems ({total_audio_hours:.2f} h audio)", flush=True)

    started = time.perf_counter()
    tracks = {s.file_id: load_track(args.cache_dir, s) for s in stems}
    stems_by_song: dict[str, list[Stem]] = {}
    for stem in stems:
        stems_by_song.setdefault(stem.song_id, []).append(stem)
    real = real_pairs(stems_by_song, tracks)
    chance = chance_pairs(real, stems, tracks, args.seed)
    rows = pd.DataFrame(real + chance)
    pair_seconds = time.perf_counter() - started
    runtime = {"f0 wall": f"{f0_seconds / 60:.1f} min ({num_computed} stems computed, {total_audio_hours:.2f} h audio)",
               "pairing wall": f"{pair_seconds:.0f} s", "songs": len(sample), "stems": len(stems)}

    print(f"pairs: {len(real)} real / {len(chance)} chance, {pair_seconds:.0f} s")
    print(md_table(side_by_side(rows[rows.frames_coactive > 0], ["pair_kind"]), ["pair_kind"]))
    print(md_table(side_by_side(rows[rows.frames_coactive > 0], []), []))
    zero = rows[(rows.source == "real") & (rows.frames_coactive == 0)]
    print(f"real pairs with zero co-active frames: {len(zero)}")

    if args.pilot:
        print("pilot — nothing written to numbers/")
        return
    NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    csv_cols = ["song_id", "genre", "source", "pair_kind", "class_pair", "class_a", "class_b", "stem_a", "stem_b",
                "chance_song_a", "chance_song_b", "offset_frames", "frames_a_active", "frames_b_active",
                "frames_coactive", "U_pc100", "U_pc50", "U_abs100"]
    rows[csv_cols].to_csv(NUMBERS_DIR / "heterophony.csv", index=False)
    write_markdown(rows, sample, NUMBERS_DIR / "heterophony.md", runtime)
    write_histogram(rows, NUMBERS_DIR / "heterophony_hist.csv")
    append_provenance(sample, rows, runtime, args.workers, NUMBERS_DIR / "provenance.md")
    print(f"wrote {NUMBERS_DIR}/heterophony.{{csv,md}}, heterophony_hist.csv, provenance.md (appended)")


if __name__ == "__main__":
    main()
