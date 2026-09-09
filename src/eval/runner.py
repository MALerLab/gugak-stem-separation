"""runner.py — the streaming evaluation loop: infer one song → score → keep rows only.

Per song: read the variant's mixture, separate it with MSST's own chunked inference
(`demix`, fp32 per the experiment config), then classify every modeled class as
PRESENT or ABSENT and score:

  PRESENT  reference file(s) exist and the reference RMS clears the silence tolerance
           (configs/silence.yaml `silence.absent_rms_dbfs`) → si_sdr + usdr, computed
           after an MSST-style trim of reference and estimate to their common length.
  ABSENT   no reference file, OR reference RMS below the tolerance. The reference is
           hard-zeroed IN CODE (silent files are never read as targets) and no SDR is
           reported — against a zero reference it degenerates to a constant. What is
           reported instead is `pred_energy_dbfs`, the model's false-activation level.

Separated audio is discarded song by song — nothing is written to disk unless the
config names songs in its optional `render` block. Output is ONE long-format parquet
(+ csv twin), one row per (song, class), carrying raw scores and run metadata; every
aggregation lives in src/eval/aggregate.py, reading that parquet.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import torch
import yaml

from src.eval.metrics import MetricFn, build_metrics, rms_dbfs
from src.eval.references import (MasterSong, SumstemSong, master_songs,
                                 sumstem_songs)
from src.render_audio import REPO_ROOT, load_msst_model, model_sample_rate

VARIANTS = ("sumstem", "master")
SPLITS = ("val", "test")


@dataclass(frozen=True)
class EvalJob:
    """One scoring run, loaded from a configs/eval/*.yaml file."""
    run_id: str
    checkpoint: str                    # repo-relative, recorded verbatim in every row
    model_config: str
    model_type: str
    targets: tuple[dict, ...]          # [{split, variant}, ...]
    metrics: tuple[str, ...]
    usdr_eps: float
    metric_cap_db: float
    absent_rms_dbfs: float
    eval_manifest: Path
    out_dir: Path
    seed: int
    device: str
    render_songs: tuple[str, ...] | str  # song ids, "all", or empty = render nothing
    render_preds_only: bool            # skip mixture/truth copies (already on disk)
    render_out_dir: Path | None
    class_map: tuple[tuple[str, str], ...]  # model head -> taxonomy class (zero-shot:
                                       # e.g. drums -> 타악기); unmapped heads find no
                                       # reference and score as absent
    master_lag_table: Path | None      # master_lag manifest (scripts/master_lag_scan.py);
                                       # aligns the master to its stems before scoring.
                                       # None = unaligned (the pre-2026-09-05 numbers)
    config_hash: str                   # sha256 of the eval config file, first 12 hex


def load_eval_job(config_path: Path) -> EvalJob:
    """Parse and validate an eval-job YAML into an EvalJob."""
    raw_bytes = config_path.read_bytes()
    raw = yaml.safe_load(raw_bytes)

    for target in raw["targets"]:
        if target["split"] not in SPLITS or target["variant"] not in VARIANTS:
            raise ValueError(f"target {target}: split must be one of {SPLITS}, "
                             f"variant one of {VARIANTS}")

    # the silence tolerance has exactly one home — configs/silence.yaml
    silence = yaml.safe_load((REPO_ROOT / raw["silence_config"]).read_text())
    absent_rms_dbfs = float(silence["silence"]["absent_rms_dbfs"])

    render = raw.get("render", {})
    render_enabled = bool(render.get("enabled", False))
    if render_enabled and not render.get("out_dir"):
        raise ValueError("render.enabled requires render.out_dir")
    # songs: a list of song ids, or the string "all" = persist every scored song
    songs_raw = render.get("songs", []) if render_enabled else []
    render_songs = "all" if songs_raw == "all" else tuple(songs_raw)

    return EvalJob(
        run_id=str(raw["run_id"]),
        checkpoint=str(raw["checkpoint"]),
        model_config=str(raw["model_config"]),
        model_type=str(raw["model_type"]),
        targets=tuple(raw["targets"]),
        metrics=tuple(raw["metrics"]),
        usdr_eps=float(raw["usdr_eps"]),
        metric_cap_db=float(raw["metric_cap_db"]),
        absent_rms_dbfs=absent_rms_dbfs,
        eval_manifest=REPO_ROOT / raw["eval_manifest"],
        out_dir=REPO_ROOT / raw["out_dir"],
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        render_songs=render_songs,
        render_preds_only=bool(render.get("preds_only", False)),
        render_out_dir=(REPO_ROOT / render["out_dir"]) if render_enabled else None,
        class_map=tuple((raw.get("class_map") or {}).items()),
        master_lag_table=((REPO_ROOT / raw["master_lag_table"])
                          if raw.get("master_lag_table") else None),
        config_hash=hashlib.sha256(raw_bytes).hexdigest()[:12],
    )


def git_commit_hash() -> str:
    """Current repo HEAD, recorded into every row for provenance."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def classify_and_score(reference: np.ndarray | None, estimate: np.ndarray,
                       metric_fns: dict[str, MetricFn],
                       absent_rms_dbfs: float) -> dict:
    """Classify one (song, class) pair as PRESENT/ABSENT and score accordingly.

    Args:
        reference: ground-truth waveform (channels, samples), or None when the song
            has no stem file for the class.
        estimate: the model's predicted waveform for the class (channels, samples).
        metric_fns: active metrics from build_metrics.
        absent_rms_dbfs: silence tolerance — reference RMS below this ⇒ ABSENT.

    Returns:
        Row fragment: status, ref_rms_dbfs, pred_energy_dbfs, and one key per metric
        (NaN for ABSENT rows — no SDR is defined against a zero reference).
    """
    ref_rms = float("nan") if reference is None else rms_dbfs(reference)
    present = reference is not None and ref_rms >= absent_rms_dbfs

    row = {"status": "present" if present else "absent",
           "ref_rms_dbfs": ref_rms,
           "pred_energy_dbfs": rms_dbfs(estimate)}
    if present:
        # MSST-style common-length trim (판소리 stem tails / master length mismatches)
        n = min(reference.shape[-1], estimate.shape[-1])
        reference, estimate = reference[..., :n], estimate[..., :n]
        for name, fn in metric_fns.items():
            row[name] = fn(reference, estimate)
    else:
        # absent: the reference is zeros BY CONSTRUCTION (never read from a silent
        # file); the only meaningful number is the model's false-activation energy
        for name in metric_fns:
            row[name] = float("nan")
    return row


def separate_song(model, msst_config, device: torch.device, mixture_path: Path,
                  model_type: str, start_frame: int = 0) -> tuple[dict[str, np.ndarray], float]:
    """Run MSST chunked inference on one full song.

    Args:
        start_frame: first mixture frame to read — a late master's lag (see
            references.MasterSong.mixture_start_frame); 0 for every other case.

    Returns:
        (class -> estimate (channels, samples) float32, song duration in seconds).
    """
    from utils.model_utils import demix   # importable after load_msst_model's path edit

    # float64 read, transposed — byte-identical to MSST valid.py's read path
    mixture, sample_rate = soundfile.read(mixture_path, start=start_frame, always_2d=True)
    expected_rate = model_sample_rate(msst_config)
    if expected_rate != int(sample_rate):
        raise ValueError(f"{mixture_path}: sample rate {sample_rate} != config "
                         f"{expected_rate} — resampling is not part of this pipeline")
    mixture = mixture.T
    with torch.inference_mode():
        separated = demix(msst_config, model, mixture.copy(), device,
                          model_type=model_type)
    return separated, mixture.shape[-1] / sample_rate


def build_song_index(job: EvalJob, msst_config, split: str,
                     variant: str) -> tuple[pd.DataFrame, dict]:
    """The frozen song list for one target, plus per-song variant accessors.

    Returns:
        (eval_manifest rows [song_id, genre_sub] for the split, song_id -> song object).
    """
    manifest = pd.read_parquet(job.eval_manifest)
    rows = (manifest[manifest.split == split][["song_id", "genre_sub"]]
            .sort_values("song_id").reset_index(drop=True))
    song_ids = rows.song_id.tolist()

    classes = list(msst_config.training.instruments)
    # FullLoader, not safe_load: BS-RoFormer configs carry !!python/tuple tags
    # (established pattern, see src/analysis/compare_runs.py)
    model_cfg = yaml.load((REPO_ROOT / job.model_config).read_text(),
                          Loader=yaml.FullLoader)
    if variant == "sumstem":
        # tree root + extension from the model config: eval reads the same tree
        # training-time validation scored
        eval_cfg = model_cfg["sumstem_eval"]
        songs = sumstem_songs(REPO_ROOT / eval_cfg["out_root"], split, song_ids,
                              eval_cfg["extension"])
    else:
        source_manifest = pd.read_parquet(
            REPO_ROOT / model_cfg["gugak_mix"]["source_manifest"])
        lag_by_song = None
        if job.master_lag_table is not None:
            lag_table = pd.read_parquet(job.master_lag_table)
            lag_table = lag_table[lag_table.split == split]
            lag_by_song = dict(zip(lag_table.song_id, lag_table.lag_samples.astype(int)))
        songs = master_songs(source_manifest, split, song_ids, classes, REPO_ROOT,
                             lag_by_song=lag_by_song)
    return rows, songs


def render_listening_copy(out_dir: Path, mixture_path: Path,
                          separated: dict[str, np.ndarray],
                          song: SumstemSong | MasterSong, classes: list[str],
                          preds_only: bool = False) -> None:
    """Write mixture / predictions / existing truths for one rendered song.

    Args:
        preds_only: write only pred_* files — for full-run persistence, where the
            mixture and truths already live on disk (Σstem tree / ingest store).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = soundfile.info(mixture_path).samplerate
    if not preds_only:
        mixture, sample_rate = soundfile.read(
            mixture_path, start=song.mixture_start_frame, always_2d=True)
        soundfile.write(out_dir / "mixture.flac", mixture, sample_rate,
                        subtype="PCM_24")
    for stem_class in classes:
        soundfile.write(out_dir / f"pred_{stem_class}.flac",
                        np.asarray(separated[stem_class]).T, sample_rate,
                        subtype="PCM_24")
        if not preds_only:
            truth = song.reference(stem_class)
            if truth is not None:
                soundfile.write(out_dir / f"truth_{stem_class}.flac", truth.T,
                                sample_rate, subtype="PCM_24")


def run(job: EvalJob, overwrite: bool = False) -> Path:
    """Execute the full scoring run and persist the raw rows parquet (+ csv twin).

    Args:
        job: parsed eval-job config.
        overwrite: allow replacing an existing rows file of the same run_id.
    """
    parquet_path = job.out_dir / f"{job.run_id}.parquet"
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"{parquet_path} exists — pass --overwrite to replace")

    torch.manual_seed(job.seed)
    np.random.seed(job.seed)

    device = torch.device(job.device)
    model, msst_config, _ = load_msst_model(
        job.model_type, REPO_ROOT / job.model_config, REPO_ROOT / job.checkpoint,
        device)
    if getattr(msst_config.inference, "normalize", False):
        raise ValueError("config.inference.normalize is on — the pipeline assumes the "
                         "un-normalized eval path training-time validation used")
    classes = list(msst_config.training.instruments)
    metric_fns = build_metrics(list(job.metrics), job.usdr_eps, job.metric_cap_db)

    commit = git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    all_rows: list[dict] = []
    for target in job.targets:
        split, variant = target["split"], target["variant"]
        song_rows, songs = build_song_index(job, msst_config, split, variant)
        print(f"[{job.run_id}] {split}/{variant}: {len(song_rows)} songs, "
              f"{len(classes)} classes", flush=True)

        for position, entry in enumerate(song_rows.itertuples(), 1):
            song = songs[entry.song_id]
            started = time.time()
            separated, duration_sec = separate_song(
                model, msst_config, device, song.mixture_path, job.model_type,
                start_frame=song.mixture_start_frame)

            class_map = dict(job.class_map)
            for stem_class in classes:
                estimate = np.asarray(separated[stem_class])
                # zero-shot: a mapped head is scored against its taxonomy class's
                # reference and its row carries the taxonomy name; unmapped heads
                # find no reference file and fall through the absent path
                target_class = class_map.get(stem_class, stem_class)
                reference = song.reference(target_class)
                scored = classify_and_score(reference, estimate, metric_fns,
                                            job.absent_rms_dbfs)
                all_rows.append({
                    "run_id": job.run_id, "checkpoint": job.checkpoint,
                    "split": split, "variant": variant,
                    "song_id": entry.song_id, "genre_sub": entry.genre_sub,
                    "stem_class": target_class, **scored,
                    "duration_sec": duration_sec,
                    "lag_samples": getattr(song, "lag_samples", 0),
                    "config_hash": job.config_hash, "git_commit": commit,
                    "seed": job.seed, "timestamp": timestamp,
                })

            if job.render_songs == "all" or entry.song_id in job.render_songs:
                render_listening_copy(
                    job.render_out_dir / f"{split}_{variant}" / entry.song_id,
                    song.mixture_path, separated, song, classes,
                    preds_only=job.render_preds_only)

            print(f"  {position}/{len(song_rows)} {entry.song_id} "
                  f"({duration_sec:.0f}s audio, {time.time() - started:.1f}s)",
                  flush=True)
            del separated                      # stream: song audio never accumulates

    frame = pd.DataFrame(all_rows)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(parquet_path.with_suffix(".csv"), index=False)

    present = int((frame.status == "present").sum())
    print(f"\nwrote {len(frame)} rows ({present} present, {len(frame) - present} "
          f"absent) -> {parquet_path}", flush=True)
    return parquet_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Config-driven evaluation runner.")
    parser.add_argument("--config", required=True,
                        help="eval-job YAML (configs/eval/*.yaml)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing rows file for this run_id")
    args = parser.parse_args()
    job = load_eval_job(REPO_ROOT / args.config)
    run(job, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
