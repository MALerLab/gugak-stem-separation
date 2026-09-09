"""master_lag_scan.py — measure the master ↔ Σstem time offset for every eval song.

Found 2026-09-05: for 19 of the 135 test songs the publisher master is a fixed integer
number of samples late (or early) relative to its own stems — 59 samples in the three
대풍류 songs and nine 판소리 songs, 338 samples in four 판소리 songs, −2138 in two
창작국악 songs — with no duration difference. Almost certainly a latency bug in the
publisher's mastering chain. It is invisible to the ear (1.34 ms) but fatal to
sample-wise metrics: SI-SDR / uSDR against references summed from the stems collapse to
about −16 dB on those songs, however good the separation is (see the maths in
docs/eval_pipeline_build_report.md if it gets written up; in short, with zero-lag
correlation ρ the best possible SI-SDR is 10·log10(ρ² / (1 − ρ²)), and ρ ≈ 0.1 there).

This script writes the `master_lag` manifest table, one row per (split, song):

    lag_samples   > 0  ⇒ the master is DELAYED relative to the stems, i.e.
                         master[t] ≈ Σstem[t − lag_samples]. To align, skip the master's
                         first lag_samples frames.
                  < 0  ⇒ the master is EARLY; skip the stems' first |lag_samples| frames.
    corr_at_lag   normalized cross-correlation at the detected lag (≈ 1 = clean sum)
    corr_at_zero  the same at lag 0 — what an unaligned eval effectively scores against

src/eval/references.py applies the shift when an eval config names this table
(`master_lag_table`). The Σstem variant is unaffected: its mixture is the sum of the
reference files by construction, so both sides share one timeline.

Run:
    uv run python scripts/master_lag_scan.py [--config configs/analysis/master_lag_scan.yaml]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/analysis/master_lag_scan.yaml"


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths.

    Layout rule (repo-wide): parquet = source of truth in manifests/parquet/,
    csv = eyeball copy in manifests/csv/. Both dirs are created on demand.

    Args:
        base: table basename, e.g. Path(".../manifests/master_lag").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


def load_settings(config_path: Path) -> dict:
    """Read the `master_lag_scan` block of the analysis config."""
    return yaml.safe_load(config_path.read_text())["master_lag_scan"]


def read_mono(path: Path, frames: int = -1) -> np.ndarray:
    """Read a file as a float64 mono downmix (mean of channels), optionally truncated."""
    audio, _ = soundfile.read(path, frames=frames, always_2d=True)
    return audio.mean(axis=1)


def sum_stems_mono(stem_paths: list[Path], frames: int) -> np.ndarray:
    """Σstem mono downmix at native scale, every stem trimmed to `frames` (never padded)."""
    total = np.zeros(frames)
    for path in stem_paths:
        total += read_mono(path, frames=frames)
    return total


def cross_correlation_lag(master: np.ndarray, sumstem: np.ndarray) -> tuple[int, float, float]:
    """Lag of the master relative to Σstem, by full-range FFT cross-correlation.

    Args:
        master: mono master, DC-removed.
        sumstem: mono Σstem, DC-removed, same length as master.

    Returns:
        (lag_samples, corr_at_lag, corr_at_zero) — lag > 0 means master[t] ≈ sumstem[t − lag].
    """
    n = len(master)
    fft_length = 1 << int(np.ceil(np.log2(2 * n)))
    correlation = np.fft.irfft(np.fft.rfft(master, fft_length)
                               * np.conj(np.fft.rfft(sumstem, fft_length)), fft_length)
    # circular → linear: index 0 is lag 0, the tail holds the negative lags
    correlation = np.concatenate([correlation[-(n - 1):], correlation[:n]])
    lags = np.arange(-(n - 1), n)
    normalizer = np.linalg.norm(master) * np.linalg.norm(sumstem) + 1e-12
    peak = int(np.argmax(np.abs(correlation)))
    return (int(lags[peak]), float(correlation[peak] / normalizer),
            float(correlation[n - 1] / normalizer))


def scan_song(song_id: str, split: str, genre_sub: str, master_path: Path,
              stem_paths: list[Path], stem_frames: list[int], sample_rate: int) -> dict:
    """Measure one song: lag, correlations and the master/stem duration bookkeeping."""
    min_frames, max_frames = min(stem_frames), max(stem_frames)
    master = read_mono(master_path)
    sumstem = sum_stems_mono(stem_paths, min_frames)
    common = min(len(master), len(sumstem))
    master_trimmed = master[:common] - master[:common].mean()
    sumstem_trimmed = sumstem[:common] - sumstem[:common].mean()
    lag, corr_at_lag, corr_at_zero = cross_correlation_lag(master_trimmed, sumstem_trimmed)
    return {"song_id": song_id, "split": split, "genre_sub": genre_sub,
            "lag_samples": lag, "lag_ms": lag / sample_rate * 1000.0,
            "corr_at_lag": corr_at_lag, "corr_at_zero": corr_at_zero,
            "master_frames": int(len(master)), "min_stem_frames": int(min_frames),
            "max_stem_frames": int(max_frames), "num_stems": len(stem_paths)}


def song_jobs(source_manifest: pd.DataFrame, eval_manifest: pd.DataFrame,
              splits: list[str]) -> list[dict]:
    """One job dict per eval song: master path + every stem path/frame count."""
    ensemble = source_manifest[source_manifest.dataset == "71955"]
    jobs = []
    for split in splits:
        songs = eval_manifest[eval_manifest.split == split].sort_values("song_id")
        for row in songs.itertuples():
            group = ensemble[ensemble.song_id == row.song_id]
            master = group[group.role == "master"]
            stems = group[group.role != "master"]
            if len(master) != 1 or stems.empty:
                raise KeyError(f"{row.song_id}: expected 1 master + ≥1 stems in source_manifest")
            jobs.append({"song_id": row.song_id, "split": split, "genre_sub": row.genre_sub,
                         "master_path": REPO_ROOT / str(master.out_path.iloc[0]),
                         "stem_paths": [REPO_ROOT / str(p) for p in stems.out_path],
                         "stem_frames": [int(f) for f in stems.out_frames],
                         "sample_rate": int(master.out_sr.iloc[0])})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Master ↔ Σstem lag scan.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_settings(REPO_ROOT / args.config)

    source_manifest = pd.read_parquet(REPO_ROOT / settings["source_manifest"])
    eval_manifest = pd.read_parquet(REPO_ROOT / settings["eval_manifest"])
    jobs = song_jobs(source_manifest, eval_manifest, list(settings["splits"]))
    print(f"scanning {len(jobs)} songs with {settings['workers']} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=int(settings["workers"])) as pool:
        futures = {pool.submit(scan_song, **job): job["song_id"] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"  {done}/{len(jobs)} {row['song_id']:26s} lag={row['lag_samples']:6d} "
                  f"({row['lag_ms']:7.2f} ms) corr@lag={row['corr_at_lag']:+.3f} "
                  f"corr@0={row['corr_at_zero']:+.3f}", flush=True)

    table = pd.DataFrame(rows).sort_values(["split", "song_id"]).reset_index(drop=True)
    parquet_path, csv_path = table_paths(REPO_ROOT / settings["out_table"])
    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)
    nonzero = table[table.lag_samples != 0]
    print(f"\nwrote {len(table)} rows -> {parquet_path}\n"
          f"{len(nonzero)} songs with non-zero lag:\n"
          f"{nonzero[['song_id', 'lag_samples', 'lag_ms', 'corr_at_lag', 'corr_at_zero']].to_string(index=False)}")


if __name__ == "__main__":
    main()
