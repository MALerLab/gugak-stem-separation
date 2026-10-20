"""aug_listening_cut_clips.py — stage 3a of the augmentation listening test.

Builds the render inputs and the true-stem references (docs/aug_listening_test_spec.md):
  - master windows: for each stage-2 pick, the lag-aligned 30 s window of the publisher
    master → inputs/<clip>.wav, and each present class's true stem for the same window →
    true/<clip>/<stem>.wav. Both at native ingest scale, exactly as the master eval
    scored them (src/eval/references.py does the lag alignment and class summing).
  - YouTube items: symlinks to the demo set v2 normalised clips → inputs/yt_<item>.wav,
    minus the excluded items.

Both models are then rendered over inputs/ with src.render_demo.

Run:
    uv run python scripts/aug_listening_cut_clips.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.eval.references import master_songs  # noqa: E402

DEFAULT_CONFIG = "configs/aug_listening_test.yaml"
SAMPLE_RATE = 44100


def master_clip_name(song_id: str, start_s: float) -> str:
    """Clip name for a master window: song number + window start in centiseconds."""
    return f"m{song_id.split('_')[0]}_t{int(round(start_s * 100)):06d}"


def cut_master_windows(config: dict, picks: pd.DataFrame, inputs_dir: Path,
                       true_dir: Path) -> list[dict]:
    """Write each pick's master window + true stems; return one row per clip.

    Args:
        config: the parsed aug_listening_test config.
        picks: the stage-2 picks table.
        inputs_dir: destination for the mixture clips (render inputs).
        true_dir: destination root for the per-clip true stems.
    """
    clips_config = config["clips"]
    source_manifest = pd.read_parquet(REPO_ROOT / clips_config["source_manifest"])
    lag_table = pd.read_parquet(REPO_ROOT / clips_config["master_lag_table"])
    lag_by_song = dict(zip(lag_table.song_id, lag_table.lag_samples))
    songs = master_songs(source_manifest, "test", list(picks.song_id), config["classes"],
                         REPO_ROOT, lag_by_song)
    num_frames = int(round(config["pick"]["window_s"] * SAMPLE_RATE))

    rows = []
    for pick in picks.itertuples():
        song = songs[pick.song_id]
        name = master_clip_name(pick.song_id, pick.start_s)
        start_frame = int(round(pick.start_s * SAMPLE_RATE))

        # the master window, read from the lag-corrected master timeline
        mixture, sample_rate = soundfile.read(
            song.mixture_path, start=song.mixture_start_frame + start_frame,
            frames=num_frames, always_2d=True)
        assert sample_rate == SAMPLE_RATE, f"{song.mixture_path}: {sample_rate} Hz"
        soundfile.write(inputs_dir / f"{name}.wav", mixture, SAMPLE_RATE, subtype="FLOAT")

        # the true stems for the same window (references are already lag-aligned)
        clip_true_dir = true_dir / name
        clip_true_dir.mkdir(parents=True, exist_ok=True)
        for stem in config["classes"]:
            reference = song.reference(stem)
            if reference is None:
                continue
            window = reference[:, start_frame:start_frame + num_frames]
            soundfile.write(clip_true_dir / f"{stem}.wav", window.T, SAMPLE_RATE,
                            subtype="FLOAT")
        rows.append({"clip": name, "item_type": "master", "song_id": pick.song_id,
                     "genre_sub": pick.genre_sub, "pick_reason": pick.pick_reason,
                     "start_s": pick.start_s, "covered_stems": pick.covered_stems,
                     "title": pick.song_id})
    return rows


def link_youtube_clips(config: dict, inputs_dir: Path) -> list[dict]:
    """Symlink the in-vocabulary YouTube clips of demo set v2 into inputs/."""
    youtube = config["youtube"]
    items = pd.read_parquet(REPO_ROOT / youtube["demo_set_manifest"])
    items = items[~items.item_id.isin(youtube["exclude_items"])
                  & ~items.source_kind.isin(youtube["exclude_source_kinds"])]
    rows = []
    for item in items.itertuples():
        source = REPO_ROOT / youtube["clip_dir"] / item.normalised_file
        if not source.exists():
            raise FileNotFoundError(source)
        name = f"yt_{item.item_id}"
        link = inputs_dir / f"{name}.wav"
        if not link.exists():
            link.symlink_to(source)
        rows.append({"clip": name, "item_type": "youtube", "song_id": None,
                     "genre_sub": None, "pick_reason": None, "start_s": item.start_s,
                     "covered_stems": None, "title": item.short_title})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text())

    out_root = REPO_ROOT / config["clips"]["out_root"]
    inputs_dir = out_root.parent / "inputs"
    true_dir = out_root / "true"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    true_dir.mkdir(parents=True, exist_ok=True)

    # master windows + true stems, then the YouTube clips
    picks = pd.read_parquet(REPO_ROOT / "manifests" / "parquet"
                            / f"{config['pick']['out_table']}.parquet")
    rows = cut_master_windows(config, picks, inputs_dir, true_dir)
    rows += link_youtube_clips(config, inputs_dir)

    # clip index for the page builder (audio-side, lives next to the clips)
    index = pd.DataFrame(rows)
    index.to_csv(out_root.parent / "clip_index.csv", index=False)
    print(index[["clip", "item_type", "title"]].to_string(index=False))
    print(f"\n{len(index)} clips -> {inputs_dir}")


if __name__ == "__main__":
    main()
