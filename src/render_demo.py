"""render_demo.py — separate real-world demo clips (no ground truth) into listenable stems.

Sibling of render_audio.py for clips that have NO reference stems: the real-world demo
set (YouTube excerpts + a couple of ensemble-set excerpts, all pre-normalised to the
−19 LUFS the models trained on). Nothing here scores anything — it loads a checkpoint,
runs MSST's chunked inference over each clip, and writes the mixture plus one file per
predicted stem in a playback-friendly format (MP3 by default; size and compatibility win
for meeting playback).

Config-driven: a render-job YAML (see configs/render/real_world_demo_*.yaml) names the
checkpoint, model config, input glob and output root. The CLI supplies only what varies
per invocation: device and threads.

Output layout:  <output_root>/<run_label>_ep<N>/<clip>/{mixture,pred_<stem>}.mp3
                <output_root>/<run_label>_ep<N>/render_manifest.csv   (one row per stem)

Read-only w.r.t. checkpoints and inputs. Writes only under output_root.

Run:
    uv run python -m src.render_demo --device cpu --threads 8
    uv run python -m src.render_demo --render-config configs/render/real_world_demo_default.yaml
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import torch
import yaml

from src.render_audio import REPO_ROOT, load_msst_model, model_sample_rate


def checkpoint_epoch(checkpoint_path: Path) -> int:
    """The `ep_N` number embedded in an MSST best-model checkpoint filename."""
    match = re.search(r"_ep_(\d+)_", checkpoint_path.name)
    if match is None:
        raise ValueError(f"no `_ep_N_` in checkpoint name: {checkpoint_path.name}")
    return int(match.group(1))


def git_commit_hash() -> str:
    """Short HEAD hash, for the manifest (so a render is traceable to code)."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def clip_name(path: Path, strip_suffix: str) -> str:
    """Output folder name for an input clip: its stem minus the configured suffix."""
    name = path.stem
    return name[: -len(strip_suffix)] if strip_suffix and name.endswith(strip_suffix) else name


def read_clip_as_stereo(path: Path, expected_sample_rate: int) -> tuple[np.ndarray, int]:
    """Read a clip as (channels=2, frames) float32, duplicating mono to stereo.

    Args:
        path: input audio file.
        expected_sample_rate: the model's training sample rate; mismatch is an error
            (inputs are pre-ingested to it, so a mismatch means the wrong file).
    """
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"{path.name}: {sample_rate} Hz, model expects {expected_sample_rate}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)     # mono → identical L/R, model needs 2 ch
    elif audio.shape[1] != 2:
        raise ValueError(f"{path.name}: {audio.shape[1]} channels, expected 1 or 2")
    return audio.T, sample_rate


def write_audio(path: Path, audio_channels_first: np.ndarray, sample_rate: int,
                audio_settings: dict) -> None:
    """Write (channels, frames) audio in the configured format (MP3 / FLAC / WAV …).

    Args:
        path: destination without extension; the format's lowercase name is appended.
        audio_channels_first: (channels, frames) float32.
        sample_rate: Hz.
        audio_settings: `audio` block of the render job (format, compression_level).
    """
    fmt = audio_settings.get("format", "MP3").upper()
    out = path.with_suffix(f".{fmt.lower()}")
    kwargs = {}
    if "compression_level" in audio_settings and fmt in ("MP3", "OGG", "FLAC"):
        kwargs["compression_level"] = float(audio_settings["compression_level"])
    soundfile.write(out, audio_channels_first.T, sample_rate, format=fmt, **kwargs)


def render_clip(model, config, device: torch.device, clip_path: Path, out_dir: Path,
                classes: list[str], model_type: str, audio_settings: dict) -> list[dict]:
    """Separate one clip and write mixture + predicted stems; return per-stem rows.

    Args:
        model: loaded MSST model in eval mode.
        config: MSST config (drives chunked inference).
        device: inference device.
        clip_path: input clip.
        out_dir: destination folder for this clip.
        classes: modeled stem classes, in head order.
        model_type: MSST model key (drives demix's chunking strategy).
        audio_settings: `audio` block of the render job.
    """
    from utils.model_utils import demix   # MSST import (path set up by load_msst_model)

    mixture, sample_rate = read_clip_as_stereo(clip_path, model_sample_rate(config))
    separated = demix(config, model, mixture, device, model_type=model_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    if audio_settings.get("write_mixture", True):
        write_audio(out_dir / "mixture", mixture, sample_rate, audio_settings)

    rows = []
    for stem_class in classes:
        prediction = np.asarray(separated[stem_class], dtype=np.float32)
        write_audio(out_dir / f"pred_{stem_class}", prediction, sample_rate, audio_settings)
        rows.append({"stem_class": stem_class,
                     "pred_rms": round(float(np.sqrt((prediction ** 2).mean())), 6),
                     "pred_peak": round(float(np.abs(prediction).max()), 4)})
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render demo-clip stems from a checkpoint.")
    parser.add_argument("--render-config", default="configs/render/real_world_demo_default.yaml",
                        help="render-job YAML (checkpoint, inputs, output format)")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"],
                        help="cpu is the default: six 30 s clips need no GPU")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads; kept modest so co-tenant GPU jobs keep their cores")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    job = yaml.safe_load((REPO_ROOT / args.render_config).read_text(encoding="utf-8"))
    audio_settings = job.get("audio", {})

    checkpoint_path = REPO_ROOT / job["checkpoint"]
    epoch = checkpoint_epoch(checkpoint_path)
    out_root = REPO_ROOT / job["output_root"] / f"{job['run_label']}_ep{epoch}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, config, _ = load_msst_model(job["model_type"], REPO_ROOT / job["model_config"],
                                       checkpoint_path, device)
    classes = list(config.training.instruments)

    clip_paths = sorted((REPO_ROOT / Path(job["input_glob"]).parent).glob(
        Path(job["input_glob"]).name))
    if not clip_paths:
        raise FileNotFoundError(f"no clips match {job['input_glob']}")
    strip_suffix = job.get("clip_name_strip_suffix", "")
    commit = git_commit_hash()

    all_rows = []
    for clip_path in clip_paths:
        name = clip_name(clip_path, strip_suffix)
        print(f"{name}: separating …", flush=True)
        with torch.inference_mode():
            rows = render_clip(model, config, device, clip_path, out_root / name, classes,
                               job["model_type"], audio_settings)
        for row in rows:
            row.update(clip=name, input_file=str(clip_path.relative_to(REPO_ROOT)),
                       checkpoint=str(checkpoint_path.relative_to(REPO_ROOT)),
                       epoch=epoch, git_commit=commit)
        all_rows.extend(rows)

    columns = ["clip", "stem_class", "pred_rms", "pred_peak", "input_file", "checkpoint",
               "epoch", "git_commit"]
    pd.DataFrame(all_rows)[columns].to_csv(out_root / "render_manifest.csv", index=False)
    print(f"\nwrote {len(clip_paths)} clips × {len(classes)} stems + render_manifest.csv "
          f"-> {out_root}")


if __name__ == "__main__":
    main()
