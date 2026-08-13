"""render_audio.py — render separation audio (mixture / predictions / truth) from a checkpoint.

Config-driven: a render-job YAML (see configs/render/) names the checkpoint, model config,
val tree and song list, plus the excerpt policy — window length and how the window is
chosen (busiest or quietest stretch via the chunk_activities manifest, uniform random, the
full song, or an explicit per-song start). The CLI supplies only what varies per
invocation: output directory, device, threads.

For each song: pick the excerpt, separate it with MSST chunked inference on a padded read
(pad trimmed afterwards, so chunking edge effects never land inside the excerpt), and
write self-describing FLACs — the input mixture, every predicted stem, and the
ground-truth stems that exist. Absent classes deliberately get a predicted file and NO
truth file: hearing near-silence where an instrument is absent is part of the demo.
Per-stem SI-SDR goes to a CSV per song plus one combined table; full-song reference
scores are unpacked from the checkpoint's own eval history (never a stored table, so the
numbers can't belong to a different checkpoint).

Read-only w.r.t. checkpoints, manifests and the val tree. Writes only under --out-dir.

Run:
    uv run python -m src.render_audio --out-dir <dir> [--render-config configs/render/default.yaml]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import torch
import yaml

from src.analysis.read_msst_checkpoint import (build_per_song, eval_song_order,
                                               latest_epoch_key)

REPO_ROOT = Path(__file__).resolve().parents[1]     # assumes this file sits in src/<here>

SELECTION_MODES = ("most_dense", "least_dense", "uniform_random", "full_song")


def scale_invariant_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR in dB for one stem, mirroring MSST's metric (channels pooled).

    Args:
        reference: ground-truth waveform, shape (channels, samples).
        estimate: predicted waveform, same shape.
    """
    eps = 1e-8
    scale = np.sum(estimate * reference + eps) / np.sum(reference ** 2 + eps)
    projection = reference * scale
    noise = projection - estimate
    return float(10 * np.log10((np.sum(projection ** 2) + eps) /
                               (np.sum(noise ** 2) + eps)))


def load_msst_model(model_type: str, model_config_path: Path, checkpoint_path: Path,
                    device: torch.device) -> tuple:
    """Load an MSST model + config and its checkpoint dict, ready for inference.

    Args:
        model_type: MSST model key (htdemucs, bs_roformer, ...).
        model_config_path: the experiment's MSST-format YAML.
        checkpoint_path: training checkpoint (full save dict or bare state dict).
        device: target device.

    Returns:
        (model in eval mode on device, MSST config, raw checkpoint dict).
    """
    # MSST is import-path-dependent; keep the sys.path edit scoped to this loader so
    # importing this module never mutates the caller's import state.
    msst_root = str(REPO_ROOT / "external" / "msst")
    if msst_root not in sys.path:
        sys.path.insert(0, msst_root)
    from utils.settings import get_model_from_config

    model, config = get_model_from_config(model_type, str(model_config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (checkpoint["model_state_dict"]
                  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                  else checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config, checkpoint


def pick_excerpt_start(chunks: pd.DataFrame, song_id: str, classes: list[str],
                       selection: str, excerpt_seconds: float, chunk_len_s: float,
                       coverage_threshold: float,
                       rng: np.random.Generator) -> float:
    """Start time (s) of the excerpt, chosen per the configured selection mode.

    Scores every non-overlapping activity window of the song by summed per-class
    coverage (only classes audible above the threshold count), then scores each run of
    consecutive windows long enough to hold the excerpt. Summed coverage is continuous,
    so it separates passages that a bare audible-class count would tie.

    Args:
        chunks: `chunk_activities` rows (any splits) with cov_* columns.
        song_id: song to search.
        classes: modeled stem classes (coverage columns to read).
        selection: most_dense | least_dense | uniform_random.
        excerpt_seconds: excerpt length the run must cover.
        chunk_len_s: activity window length to read from the manifest.
        coverage_threshold: a class counts as audible above this window fraction.
        rng: seeded generator (consumed only by uniform_random).
    """
    song = (chunks[(chunks.song_id == song_id) & (chunks.chunk_len_s == chunk_len_s)]
            .sort_values("window_idx"))
    if song.empty:
        raise ValueError(f"{song_id}: no chunk_activities rows at {chunk_len_s} s")
    coverage = song[[f"cov_{c}" for c in classes]].to_numpy()
    n_windows = max(1, math.ceil(excerpt_seconds / chunk_len_s))
    if len(coverage) < n_windows:
        return 0.0
    window_score = np.where(coverage > coverage_threshold, coverage, 0.0).sum(axis=1)
    run_scores = np.convolve(window_score, np.ones(n_windows), mode="valid")
    if selection == "most_dense":
        run_index = int(np.argmax(run_scores))
    elif selection == "least_dense":
        run_index = int(np.argmin(run_scores))
    elif selection == "uniform_random":
        run_index = int(rng.integers(0, len(run_scores)))
    else:
        raise ValueError(f"unknown selection mode: {selection}")
    return float(song.start_s.to_numpy()[run_index])


def render_song(model, config, device: torch.device, song_dir: Path, out_dir: Path,
                classes: list[str], start_seconds: float | None,
                excerpt_seconds: float, pad_seconds: float, model_type: str,
                extension: str = "flac") -> list[dict]:
    """Separate one excerpt (or the full song) and write mixture / predicted / truth files.

    Args:
        model: loaded MSST model in eval mode.
        config: MSST config (drives chunked inference).
        device: inference device.
        song_dir: Σstem val folder holding mixture + per-class truth files.
        out_dir: destination folder for this song.
        classes: modeled stem classes, in head order.
        start_seconds: excerpt start within the song; None renders the full song.
        excerpt_seconds: excerpt length (ignored when start_seconds is None).
        pad_seconds: padding separated-then-trimmed on each side of the excerpt.
        model_type: MSST model key (drives demix's chunking strategy).
        extension: audio extension of the val tree.
    """
    from utils.model_utils import demix

    info = soundfile.info(song_dir / f"mixture.{extension}")
    sample_rate = info.samplerate

    if start_seconds is None:      # full song: no excerpting, edges are real edges
        read_start, lead_pad = 0, 0
        excerpt_frames = read_frames = info.frames
    else:
        excerpt_frames = int(excerpt_seconds * sample_rate)
        pad_frames = int(pad_seconds * sample_rate)
        # padded read so chunked-inference edges fall outside the kept excerpt
        read_start = max(0, int(start_seconds * sample_rate) - pad_frames)
        lead_pad = int(start_seconds * sample_rate) - read_start
        read_frames = min(lead_pad + excerpt_frames + pad_frames,
                          info.frames - read_start)
        excerpt_frames = min(excerpt_frames, read_frames - lead_pad)

    mixture, _ = soundfile.read(song_dir / f"mixture.{extension}", start=read_start,
                                frames=read_frames, dtype="float32", always_2d=True)
    separated = demix(config, model, mixture.T, device, model_type=model_type)

    keep = slice(lead_pad, lead_pad + excerpt_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    soundfile.write(out_dir / "mixture.flac", mixture[keep], sample_rate, subtype="PCM_24")

    rows = []
    for stem_class in classes:
        prediction = np.asarray(separated[stem_class]).T[keep]     # -> (frames, ch)
        soundfile.write(out_dir / f"pred_{stem_class}.flac", prediction, sample_rate,
                        subtype="PCM_24")

        truth_path = song_dir / f"{stem_class}.{extension}"
        if truth_path.exists():
            truth, _ = soundfile.read(truth_path, start=read_start + lead_pad,
                                      frames=excerpt_frames, dtype="float32",
                                      always_2d=True)
            soundfile.write(out_dir / f"truth_{stem_class}.flac", truth, sample_rate,
                            subtype="PCM_24")
            excerpt_sdr = scale_invariant_sdr(truth.T, prediction.T)
            present = True
        else:
            excerpt_sdr = float("nan")     # absent class: silence is the correct answer
            present = False
        rows.append({"stem_class": stem_class, "present_in_song": present,
                     "excerpt_si_sdr_db": round(excerpt_sdr, 2),
                     "pred_rms": round(float(np.sqrt((prediction ** 2).mean())), 6),
                     "pred_peak": round(float(np.abs(prediction).max()), 4)})
    return rows


def full_song_reference_scores(checkpoint: dict, val_root: Path) -> pd.DataFrame | None:
    """Per-(song, class) SI-SDR from the checkpoint's own eval history, if it has one.

    Args:
        checkpoint: raw checkpoint dict (needs MSST's `all_metrics`).
        val_root: Σstem val tree the run evaluated on.
    """
    if not (isinstance(checkpoint, dict) and "all_metrics" in checkpoint):
        print("checkpoint carries no eval history — full-song scores skipped")
        return None
    history = checkpoint["all_metrics"]
    epoch_key = latest_epoch_key(history)
    print(f"full-song reference scores from {epoch_key} of the checkpoint's own history")
    return build_per_song(history, epoch_key, eval_song_order(val_root), val_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render separation audio from a checkpoint.")
    ap.add_argument("--render-config", default="configs/render/default.yaml",
                    help="render-job YAML (checkpoint, songs, excerpt policy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu is usable (slower than realtime) when GPUs are busy")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads; kept modest so co-tenant GPU jobs keep their cores")
    args = ap.parse_args()

    job = yaml.safe_load((REPO_ROOT / args.render_config).read_text(encoding="utf-8"))
    excerpt = job["excerpt"]
    selection = excerpt["selection"]
    if selection not in SELECTION_MODES:
        raise ValueError(f"excerpt.selection must be one of {SELECTION_MODES}")
    val_root = REPO_ROOT / job["val_root"]
    extension = job.get("extension", "flac")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, config, checkpoint = load_msst_model(
        job["model_type"], REPO_ROOT / job["model_config"],
        REPO_ROOT / job["checkpoint"], device)
    classes = list(config.training.instruments)

    # window selection inputs, loaded only if some song actually needs them
    needs_activity = any("start_s" not in song for song in job["songs"]) \
        and selection != "full_song"
    chunks = (pd.read_parquet(REPO_ROOT / job["chunk_activities"])
              if needs_activity else None)
    rng = np.random.default_rng(excerpt.get("seed", 42))

    song_scores = (full_song_reference_scores(checkpoint, val_root)
                   if job.get("full_song_scores", True) else None)

    all_rows = []
    for song in job["songs"]:
        song_id = song["song_id"]
        if "start_s" in song:
            start = float(song["start_s"])
        elif selection == "full_song":
            start = None
        else:
            start = pick_excerpt_start(
                chunks, song_id, classes, selection, excerpt["seconds"],
                excerpt.get("activity_chunk_len_s", 10.0),
                excerpt.get("coverage_threshold", 0.25), rng)
        span = ("full song" if start is None
                else f"excerpt {start:.0f}-{start + excerpt['seconds']:.0f}s")
        print(f"{song_id}: {span}")

        rows = render_song(model, config, device, val_root / song_id,
                           out_root / song_id, classes, start,
                           excerpt["seconds"], excerpt.get("pad_seconds", 5.0),
                           job["model_type"], extension)
        # attach the authoritative full-song eval numbers next to the excerpt numbers
        full = ({} if song_scores is None else
                song_scores[song_scores.song_id == song_id]
                .set_index("stem_class")["si_sdr"].round(2).to_dict())
        for row in rows:
            row.update(song_id=song_id, note=song.get("note", ""),
                       excerpt_start_s=(round(start, 1) if start is not None else 0.0),
                       full_song_si_sdr_db=full.get(row["stem_class"], float("nan")))
        pd.DataFrame(rows).to_csv(out_root / song_id / "si_sdr.csv", index=False)
        all_rows.extend(rows)

    columns = ["song_id", "note", "excerpt_start_s", "stem_class", "present_in_song",
               "excerpt_si_sdr_db", "full_song_si_sdr_db", "pred_rms", "pred_peak"]
    combined = pd.DataFrame(all_rows)[columns]
    combined.to_csv(out_root / "si_sdr_all_songs.csv", index=False)
    print(f"\nwrote {len(job['songs'])} songs + si_sdr_all_songs.csv -> {out_root}")


if __name__ == "__main__":
    main()
