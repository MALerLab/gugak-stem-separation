"""aug_listening_build_page.py — stage 4 of the augmentation listening test.

Assembles the A/B page (docs/aug_listening_test_spec.md, stage 4) into page.out_dir:
  index.html                    copied from scripts/aug_listening_page/
  rows.json                     items → rows (one per stem) with the blind A/B mapping
  audio/<clip>/mixture.mp3      the input clip
  audio/<clip>/<stem>_a.mp3     version A of a row (file names never name the model)
  audio/<clip>/<stem>_b.mp3     version B
  audio/<clip>/<stem>_true.mp3  true stem (master items only)

A row is skipped when every version of it is quieter than `silent_row_dbfs`. A/B is
assigned per row from `page.seed`, so a rebuild reproduces the same mapping.

Run (after stage 3 renders exist):
    uv run python scripts/aug_listening_build_page.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/aug_listening_test.yaml"
PAGE_TEMPLATE = REPO_ROOT / "scripts" / "aug_listening_page" / "index.html"

# ASCII file stems for the Korean class names, so audio URLs stay plain
STEM_FILE_NAMES = {"가야금": "gayageum", "거문고": "geomungo", "기타": "gita", "대금": "daegeum",
                   "아쟁": "ajaeng", "양금": "yanggeum", "타악기": "taak", "피리": "piri",
                   "해금": "haegeum"}


def rms_dbfs(path: Path) -> float:
    """RMS level of a file in dBFS (−inf for digital silence)."""
    audio, _ = soundfile.read(path, always_2d=True)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def encode_mp3(source: Path, destination: Path, bitrate_kbps: int) -> None:
    """Encode one file to MP3 with ffmpeg, always overwriting (A/B files must track the
    current mapping, never a stale one)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
                    "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", str(destination)],
                   check=True)


def item_meta(clip) -> str:
    """One-line description under an item's title."""
    if clip.item_type == "master":
        reason = ("typical for its genre" if clip.pick_reason == "genre_typical"
                  else f"added for {clip.pick_reason.removeprefix('top_up_')}")
        return f"{clip.genre_sub} · window {clip.start_s:.0f}–{clip.start_s + 30:.0f} s · {reason}"
    return f"window starts {clip.start_s:.0f} s into the recording"


def build_rows(config: dict, clip, render_dirs: dict[str, Path],
               rng: random.Random, audio_dir: Path) -> list[dict]:
    """Rows of one item: per stem, A/B (shuffled) + true stem; silent rows skipped.

    Args:
        config: the parsed aug_listening_test config.
        clip: one clip_index row (itertuples namedtuple; a Series would shadow .clip).
        render_dirs: role (baseline / augmented) → that model's render folder.
        rng: seeded generator for the A/B assignment.
        audio_dir: page audio root.
    """
    page = config["page"]
    true_dir = REPO_ROOT / config["clips"]["out_root"] / "true" / clip.clip
    rows = []
    for stem in config["classes"]:
        predictions = {role: render_dirs[role] / clip.clip / f"pred_{stem}.flac"
                       for role in ("baseline", "augmented")}
        true_path = true_dir / f"{stem}.wav" if clip.item_type == "master" else None
        if true_path is not None and not true_path.exists():
            true_path = None
        sources = list(predictions.values()) + ([true_path] if true_path else [])
        if max(rms_dbfs(path) for path in sources) < page["silent_row_dbfs"]:
            continue

        # blind assignment: which model plays as A
        roles = ["baseline", "augmented"]
        rng.shuffle(roles)
        file_stem = STEM_FILE_NAMES[stem]
        row = {"stem": stem}
        for letter, role in zip(("a", "b"), roles):
            relative = f"audio/{clip.clip}/{file_stem}_{letter}.{page['audio_format']}"
            encode_mp3(predictions[role], audio_dir.parent / relative, page["bitrate_kbps"])
            row[letter] = relative
            row[f"{letter}_model"] = config["models"][role]["run"]
        row["true"] = None
        if true_path is not None:
            relative = f"audio/{clip.clip}/{file_stem}_true.{page['audio_format']}"
            encode_mp3(true_path, audio_dir.parent / relative, page["bitrate_kbps"])
            row["true"] = relative
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    page = config["page"]

    out_dir = REPO_ROOT / page["out_dir"]
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = REPO_ROOT / config["clips"]["out_root"]
    clip_index = pd.read_csv(inputs_dir.parent / "clip_index.csv")
    render_root = REPO_ROOT / config["render_root"]
    render_dirs = {role: render_root / f"{config['models'][role]['render_label']}_ep{config['epoch']}"
                   for role in ("baseline", "augmented")}

    # masters first (sorted by song), then YouTube items in manifest order
    clip_index["order"] = (clip_index.item_type != "master").astype(int)
    clip_index = clip_index.sort_values(["order", "clip"], kind="stable")
    rng = random.Random(page["seed"])

    items = []
    for clip in clip_index.itertuples(index=False):
        mixture = f"audio/{clip.clip}/mixture.{page['audio_format']}"
        encode_mp3(inputs_dir.parent / "inputs" / f"{clip.clip}.wav", out_dir / mixture,
                   page["bitrate_kbps"])
        items.append({"clip": clip.clip, "item_type": clip.item_type, "title": clip.title,
                      "meta": item_meta(clip), "mixture": mixture,
                      "rows": build_rows(config, clip, render_dirs, rng, audio_dir)})

    # page data + template, then a size report against the artifact limits
    (out_dir / "rows.json").write_text(json.dumps({"items": items}, ensure_ascii=False,
                                                  indent=1))
    shutil.copy(PAGE_TEMPLATE, out_dir / "index.html")
    audio_files = sorted(audio_dir.rglob(f"*.{page['audio_format']}"))
    total_mb = sum(path.stat().st_size for path in audio_files) / 1e6
    num_rows = sum(len(item["rows"]) for item in items)
    print(f"{len(items)} items, {num_rows} A/B rows, {len(audio_files)} audio files, "
          f"{total_mb:.0f} MB -> {out_dir}")
    print("artifact limits: 511 files / 256 MB per version, 255 files / 64 MB per publish")


if __name__ == "__main__":
    main()
