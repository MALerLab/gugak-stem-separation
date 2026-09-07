"""build_demo_web.py — build the static listening page (demo_web/site) from one manifest.

Everything about WHICH songs appear flows from demo_web/manifest.yaml; everything about HOW
(paths, encoder, thresholds, class order) from configs/demo_web.yaml. For every manifest item:

  1. cut the excerpt window from the model input and the nine predicted stems
     (test songs: full-song FLAC renders; real-world: the 30 s demo-set-v2 renders),
  2. gate stems by predicted-stem RMS over the window (config threshold, tolerance-based),
     union over the item's input variants, honouring the manifest's stems_override,
  3. encode MP3 (as the 2026-08-18 render) + write a wavesurfer peaks JSON per file,
  4. attach per-song SI-SDR (mean over present classes) for both input variants to test items,
  5. write demo_web/site/data.json: sections → items → files, peaks, scores,
     plus demo_web/site/README.md with provenance and a copy of the manifest.

Idempotent: existing MP3s are kept unless --force; item folders no longer in the manifest are
pruned from site/audio and site/peaks. Silent stems are never written.

Run:
    uv run python scripts/build_demo_web.py
    uv run python scripts/build_demo_web.py --force        # re-encode every file
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_ORDER = ("genre", "real_world", "master_gap")
TEST_VARIANT_ORDER = ("sumstem", "master")
INPUT_FILE_STEM = "input"


# ----------------------------------------------------------------------------- helpers
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path_like: str) -> Path:
    """Repo-relative or ~-prefixed config path → absolute Path."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def existing_korean_path(path: Path) -> Path:
    """Return `path`, falling back to its NFD spelling (Korean filenames may be either)."""
    if path.exists():
        return path
    alternative = path.with_name(unicodedata.normalize("NFD", path.name))
    if alternative.exists():
        return alternative
    raise FileNotFoundError(path)


def git_commit_hash() -> str:
    """Short HEAD hash plus a `-dirty` marker when the working tree has changes."""
    short = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return f"{short}-dirty" if dirty else short


# ----------------------------------------------------------------------------- audio
def audio_duration_s(path: Path) -> float:
    info = soundfile.info(str(path))
    return info.frames / info.samplerate


def read_window(path: Path, start_s: float, length_s: float) -> tuple[np.ndarray, int]:
    """Read [start_s, start_s + length_s) as (frames, channels) float32 via ffmpeg; raises if it
    overruns. ffmpeg, not libsndfile: the render FLACs carry no seek table, and libsndfile's
    FLAC path is both slow to seek and slow to decode (≈20× real time on dense stems)."""
    info = soundfile.info(str(path))
    start_frame = int(round(start_s * info.samplerate))
    num_frames = int(round(length_s * info.samplerate))
    if start_frame < 0 or start_frame + num_frames > info.frames:
        raise ValueError(f"window {start_s:.2f}+{length_s:.2f} s overruns {path} "
                         f"({info.frames / info.samplerate:.2f} s)")
    command = ["ffmpeg", "-v", "error", "-nostdin",
               "-ss", f"{start_frame / info.samplerate:.6f}",     # input-side seek, sample-accurate (accurate_seek default)
               "-i", str(path),
               "-frames:a", "1000000",                             # no frame cap in practice; -t rounds at packet edges
               "-f", "f32le", "-ac", str(info.channels), "-ar", str(info.samplerate), "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, info.channels)
    if len(audio) < num_frames:
        raise ValueError(f"ffmpeg returned {len(audio)} < {num_frames} frames for {path}")
    return np.ascontiguousarray(audio[:num_frames]), info.samplerate


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS over all samples and channels (floored, never −inf)."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-12))


def is_audible(audio: np.ndarray, threshold_dbfs: float) -> bool:
    """Tolerance test against the configured floor — silence is a ±1-LSB residue, not zero."""
    return rms_dbfs(audio) > threshold_dbfs


def compute_peaks(audio: np.ndarray, sample_rate: int, points: int) -> dict:
    """Mono min/max envelope in the audiowaveform v2 JSON layout wavesurfer's `peaks` accepts."""
    mono = audio.mean(axis=1)
    samples_per_pixel = max(1, math.ceil(len(mono) / points))
    padded_length = samples_per_pixel * math.ceil(len(mono) / samples_per_pixel)
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: len(mono)] = mono
    buckets = padded.reshape(-1, samples_per_pixel)
    interleaved = np.empty(buckets.shape[0] * 2, dtype=np.float64)
    interleaved[0::2] = buckets.min(axis=1)
    interleaved[1::2] = buckets.max(axis=1)
    return {"version": 2, "channels": 1, "sample_rate": int(sample_rate),
            "samples_per_pixel": samples_per_pixel, "length": int(buckets.shape[0]),
            "data": [round(float(value), 3) for value in interleaved]}


def encode_audio(audio: np.ndarray, sample_rate: int, out_path: Path, audio_cfg: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out_path), audio, sample_rate, format=audio_cfg["format"].upper(),
                    compression_level=float(audio_cfg["compression_level"]))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------------- window rule
def default_window_start(song_id: str, chunks: pd.DataFrame, class_keys: list[str],
                         window_cfg: dict, duration_s: float, length_s: float) -> float:
    """v1 median-audible-class rule: the 30 s window whose mean audible-class count sits closest
    to the song's own median, margins respected, ties broken by proximity to ⅓ of the song.
    Falls back to a centred window when the song is too short for the margins."""
    song_chunks = chunks[(chunks.song_id == song_id)
                         & (chunks.chunk_len_s == float(window_cfg["chunk_len_s"]))]
    song_chunks = song_chunks.sort_values("start_s").reset_index(drop=True)
    coverage = song_chunks[[f"cov_{key}" for key in class_keys]].to_numpy()
    num_audible = (coverage > float(window_cfg["coverage_threshold"])).sum(axis=1)
    per_window = int(round(length_s / float(window_cfg["chunk_len_s"])))
    margin = float(window_cfg["edge_margin_s"])
    candidates = []
    for index in range(len(song_chunks) - per_window + 1):
        start = float(song_chunks.start_s.iloc[index])
        if start < margin or start + length_s > duration_s - margin:
            continue
        candidates.append((start, float(num_audible[index:index + per_window].mean())))
    if not candidates:
        return max(0.0, min(duration_s - length_s, (duration_s - length_s) / 2))
    median = float(np.median(num_audible))
    best_distance = min(abs(mean - median) for _, mean in candidates)
    near_best = [start for start, mean in candidates if abs(mean - median) <= best_distance + 0.1]
    anchor = duration_s * float(window_cfg["anchor_fraction"])
    return min(near_best, key=lambda start: abs(start + length_s / 2 - anchor))


# ----------------------------------------------------------------------------- scores
def song_scores(eval_rows: pd.DataFrame) -> dict[str, dict]:
    """song_id → {si_sdr_db: mean over present classes, num_present}."""
    present = eval_rows[eval_rows.status == "present"]
    grouped = present.groupby("song_id").si_sdr.agg(["mean", "count"])
    return {nfc(song_id): {"si_sdr_db": round(float(row["mean"]), 2), "num_present": int(row["count"])}
            for song_id, row in grouped.iterrows()}


# ----------------------------------------------------------------------------- item resolution
class BuildContext:
    """Everything the per-item builders need, loaded once."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.classes: list[dict] = cfg["classes"]
        self.class_keys = [entry["key"] for entry in self.classes]
        self.slug_of = {entry["key"]: entry["slug"] for entry in self.classes}
        test_cfg = cfg["test"]
        eval_manifest = pd.read_parquet(resolve(test_cfg["eval_manifest"]))
        eval_manifest["song_id"] = eval_manifest.song_id.map(nfc)
        self.test_songs = eval_manifest[eval_manifest.split == "test"].set_index("song_id")
        metadata = pd.read_csv(resolve(test_cfg["metadata_csv"]))
        self.song_titles = {nfc(row.project_name): row.title for row in metadata.itertuples()}
        lag = pd.read_parquet(resolve(test_cfg["master_lag"]))
        self.master_lag_ms = {nfc(row.song_id): round(float(row.lag_ms), 1) for row in lag.itertuples()
                              if int(row.lag_samples) != 0}
        self.chunks = pd.read_parquet(resolve(test_cfg["chunk_activities"]))
        self.chunks["song_id"] = self.chunks.song_id.map(nfc)
        self.scores = {variant: song_scores(pd.read_parquet(resolve(spec["scores"])))
                       for variant, spec in test_cfg["variants"].items()}
        demo_set = load_yaml(resolve(cfg["real_world"]["demo_set_config"]))
        self.real_world_items = demo_set["items"]
        self.real_world_folder = {item: Path(pick).stem for item, pick in demo_set["freeze"]["picks"].items()}

    def is_test_song(self, source: str) -> bool:
        return nfc(source) in self.test_songs.index


def test_variant_paths(ctx: BuildContext, song_id: str, variant: str) -> dict[str, Path]:
    """input + pred_<class> paths for one test song under one input variant."""
    spec = ctx.cfg["test"]["variants"][variant]
    preds_dir = resolve(spec["preds"].format(song_id=song_id))
    paths = {INPUT_FILE_STEM: existing_korean_path(resolve(spec["input"].format(song_id=song_id)))}
    for key in ctx.class_keys:
        paths[key] = existing_korean_path(preds_dir / f"pred_{key}.flac")
    return paths


def real_world_paths(ctx: BuildContext, item_id: str) -> dict[str, Path]:
    folder = resolve(ctx.cfg["real_world"]["renders"]) / ctx.real_world_folder[item_id]
    paths = {INPUT_FILE_STEM: existing_korean_path(folder / "mixture.flac")}
    for key in ctx.class_keys:
        paths[key] = existing_korean_path(folder / f"pred_{key}.flac")
    return paths


def resolve_item(item: dict, section: str, ctx: BuildContext) -> dict:
    """Manifest item → kind, title, genre, variants (key, label, file paths), window, scores."""
    source = nfc(str(item["source"]))
    window_cfg = ctx.cfg["window"]
    length_s = float(item.get("window", {}).get("length_s", window_cfg["default_length_s"]))
    if ctx.is_test_song(source):
        variant_keys = list(TEST_VARIANT_ORDER) if section == "master_gap" else [TEST_VARIANT_ORDER[0]]
        variants = [{"key": key, "label": ctx.cfg["test"]["variants"][key]["label"],
                     "paths": test_variant_paths(ctx, source, key)} for key in variant_keys]
        duration_s = min(audio_duration_s(path) for variant in variants for path in variant["paths"].values())
        start_s = item.get("window", {}).get("start_s")
        if start_s is None:
            start_s = default_window_start(source, ctx.chunks, ctx.class_keys, window_cfg, duration_s, length_s)
        genre_ko = ctx.test_songs.loc[source, "genre_sub"]
        resolved = {"kind": "test", "genre": item.get("genre", ctx.cfg["genres"].get(genre_ko, genre_ko)),
                    "genre_ko": genre_ko, "title": item.get("display_title", ctx.song_titles.get(source, source)),
                    "scores": {key: ctx.scores[key].get(source) for key in TEST_VARIANT_ORDER},
                    "master_lag_ms": ctx.master_lag_ms.get(source)}
    else:
        if section == "master_gap":
            raise ValueError(f"master_gap item {item['id']!r} must reference a test song, got {source!r}")
        if source not in ctx.real_world_folder:
            raise ValueError(f"unknown source {source!r} for item {item['id']!r}: neither a test song "
                             f"nor a demo-set v2 item")
        variants = [{"key": "recording", "label": ctx.cfg["real_world"]["label"],
                     "paths": real_world_paths(ctx, source)}]
        duration_s = min(audio_duration_s(path) for path in variants[0]["paths"].values())
        start_s = float(item.get("window", {}).get("start_s", 0.0))
        length_s = min(length_s, duration_s - start_s)
        resolved = {"kind": "real_world", "genre": item.get("genre"), "genre_ko": None,
                    "title": item.get("display_title", ctx.real_world_items[source]["short_title"]),
                    "scores": None, "master_lag_ms": None}
    resolved.update({"id": str(item["id"]), "source": source, "variants": variants,
                     "window": {"start_s": round(float(start_s), 3), "length_s": round(float(length_s), 3)},
                     "note": item.get("note"), "stems_override": item.get("stems_override") or {}})
    return resolved


# ----------------------------------------------------------------------------- per-item build
def decide_stems(rms_by_variant: dict[str, dict[str, float]], override: dict, class_keys: list[str],
                 threshold_dbfs: float) -> list[str]:
    """Audible set = union over variants of stems above the RMS floor, ± manifest overrides."""
    include = {nfc(key) for key in override.get("include", [])}
    exclude = {nfc(key) for key in override.get("exclude", [])}
    unknown = (include | exclude) - set(class_keys)
    if unknown:
        raise ValueError(f"stems_override names unknown classes: {sorted(unknown)}")
    audible = {key for per_class in rms_by_variant.values() for key, level in per_class.items()
               if level > threshold_dbfs}
    return [key for key in class_keys if (key in audible or key in include) and key not in exclude]


def build_item(resolved: dict, ctx: BuildContext, site_dir: Path, force: bool) -> dict:
    """Cut, gate, encode and describe one item; returns its data.json record."""
    cfg = ctx.cfg
    window = resolved["window"]
    extension = cfg["audio"]["extension"]
    # read every window once: input + all nine predictions per variant
    windows: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for variant in resolved["variants"]:
        windows[variant["key"]] = {name: read_window(path, window["start_s"], window["length_s"])
                                   for name, path in variant["paths"].items()}
    rms_by_variant = {key: {name: rms_dbfs(audio) for name, (audio, _) in files.items() if name != INPUT_FILE_STEM}
                      for key, files in windows.items()}
    stems = decide_stems(rms_by_variant, resolved["stems_override"], ctx.class_keys,
                         float(cfg["audible"]["rms_threshold_dbfs"]))
    # write input + audible stems only
    variant_records = []
    for variant in resolved["variants"]:
        files_record = {}
        for name in [INPUT_FILE_STEM, *stems]:
            audio, sample_rate = windows[variant["key"]][name]
            file_stem = INPUT_FILE_STEM if name == INPUT_FILE_STEM else ctx.slug_of[name]
            audio_rel = Path("audio") / resolved["id"] / variant["key"] / f"{file_stem}.{extension}"
            peaks_rel = Path("peaks") / resolved["id"] / variant["key"] / f"{file_stem}.json"
            if force or not (site_dir / audio_rel).exists():
                encode_audio(audio, sample_rate, site_dir / audio_rel, cfg["audio"])
            write_json(site_dir / peaks_rel, compute_peaks(audio, sample_rate, int(cfg["peaks"]["points"])))
            files_record[file_stem] = {"audio": audio_rel.as_posix(), "peaks": peaks_rel.as_posix()}
        variant_records.append({
            "key": variant["key"], "label": variant["label"],
            "input": files_record.pop(INPUT_FILE_STEM), "stems": files_record,
            "stem_rms_dbfs": {ctx.slug_of[key]: round(level, 1) for key, level in rms_by_variant[variant["key"]].items()},
        })
    return {"id": resolved["id"], "title": resolved["title"], "genre": resolved["genre"],
            "genre_ko": resolved["genre_ko"], "kind": resolved["kind"], "source": resolved["source"],
            "window": window, "note": resolved["note"], "scores": resolved["scores"],
            "master_lag_ms": resolved["master_lag_ms"],
            "stems": [ctx.slug_of[key] for key in stems], "variants": variant_records}


def prune_orphans(site_dir: Path, keep_ids: set[str]) -> list[str]:
    """Remove audio/ and peaks/ folders for items no longer in the manifest."""
    removed = []
    for subdir in ("audio", "peaks"):
        root = site_dir / subdir
        if not root.exists():
            continue
        for folder in root.iterdir():
            if folder.is_dir() and folder.name not in keep_ids:
                shutil.rmtree(folder)
                removed.append(f"{subdir}/{folder.name}")
    return removed


# ----------------------------------------------------------------------------- outputs
def directory_size_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e6


def write_readme(site_dir: Path, cfg: dict, manifest_path: Path, data: dict, commit: str) -> None:
    model = cfg["model"]
    with open(manifest_path, encoding="utf-8") as handle:
        manifest_text = handle.read()
    num_items = sum(len(section["items"]) for section in data["sections"])
    section_counts = ", ".join(f"{section['id']} {len(section['items'])}" for section in data["sections"])
    lines = [
        f"# {cfg['page']['title']}",
        "",
        f"Listening demo for the {cfg['page']['venue']} — static page, served from this repo via GitHub Pages: "
        f"{cfg['publish']['pages_url']}",
        "",
        "## Provenance",
        "",
        f"- Model: {model['label']}",
        f"- Checkpoint: `{model['checkpoint']}`",
        f"- Render date: {model['render_date']}",
        f"- Built: {data['meta']['built_at']} from `gugak-stem-separation` commit `{commit}`",
        f"- Items: {num_items} ({section_counts})",
        f"- Audible-stem gate: predicted-stem RMS over the window > {cfg['audible']['rms_threshold_dbfs']} dBFS; "
        "silent stems are not shipped",
        f"- Encoding: {cfg['audio']['format']} via libsndfile, compression_level {cfg['audio']['compression_level']}",
        "- Scores: SI-SDR per song, mean over present classes; the master-input score uses the lag-aligned rows",
        "",
        "Everything under `audio/`, `peaks/` and `data.json` is generated by `scripts/build_demo_web.py` "
        "in the code repo from the manifest below; `index.html` renders `data.json`.",
        "",
        "## Manifest (copy)",
        "",
        "```yaml",
        manifest_text.rstrip(),
        "```",
        "",
    ]
    (site_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(cfg: dict, force: bool) -> dict:
    ctx = BuildContext(cfg)
    manifest_path = resolve(cfg["manifest"])
    manifest = load_yaml(manifest_path)
    site_dir = resolve(cfg["site_dir"])
    site_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    sections = []
    for section_key in SECTION_ORDER:
        items = manifest.get("sections", {}).get(section_key) or []
        records = []
        for item in items:
            if item["id"] in seen_ids:
                raise ValueError(f"duplicate item id {item['id']!r}")
            seen_ids.add(item["id"])
            resolved = resolve_item(item, section_key, ctx)
            record = build_item(resolved, ctx, site_dir, force)
            print(f"  [{section_key}] {record['id']}: {record['window']['start_s']:.1f}+"
                  f"{record['window']['length_s']:.0f} s · stems {','.join(record['stems']) or '—'}")
            records.append(record)
        section_cfg = cfg["sections"][section_key]
        sections.append({"id": section_key, "title": section_cfg["title"], "blurb": section_cfg["blurb"],
                         "items": records})
    removed = prune_orphans(site_dir, seen_ids)
    for path in removed:
        print(f"  pruned {path}")
    commit = git_commit_hash()
    data = {
        "meta": {
            "page": cfg["page"], "model": cfg["model"], "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_commit": commit, "audible_threshold_dbfs": cfg["audible"]["rms_threshold_dbfs"],
            "classes": [{"slug": entry["slug"], "en": entry["en"], "ko": entry["key"]} for entry in cfg["classes"]],
            "variant_labels": {**{key: spec["label"] for key, spec in cfg["test"]["variants"].items()},
                               "recording": cfg["real_world"]["label"]},
        },
        "sections": sections,
    }
    write_json(site_dir / "data.json", data)
    write_readme(site_dir, cfg, manifest_path, data, commit)
    return data


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/demo_web.yaml")
    parser.add_argument("--force", action="store_true", help="re-encode audio even when the MP3 exists")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    cfg = load_yaml(resolve(args.config))
    print(f"building {cfg['site_dir']} from {cfg['manifest']}")
    data = build(cfg, force=args.force)
    site_dir = resolve(cfg["site_dir"])
    num_items = sum(len(section["items"]) for section in data["sections"])
    print(f"done: {num_items} items · site {directory_size_mb(site_dir):.1f} MB "
          f"(audio {directory_size_mb(site_dir / 'audio'):.1f} MB, peaks {directory_size_mb(site_dir / 'peaks'):.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
