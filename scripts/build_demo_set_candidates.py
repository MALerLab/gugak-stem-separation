"""build_demo_set_candidates.py — stage 1 of the real-world demo pool (`demo_set_v1`).

Acquire → ingest → QC → propose 30 s candidate excerpts for LISTENING. Nothing here is a
score: the external items have no ground-truth stems, so no SDR exists, and the dataset
items (slot 1) are chosen from manifest content only — never from a model output.

Part A — external items (slots 2-5, YouTube):
    raw/<slot>.<ext>  (fetched separately by yt-dlp, best audio-only stream, no re-encode)
    → ffprobe (container / codec / bitrate / sr / channels, BEFORE ingest)
    → decode → the SAME per-file ops as src/data/ingest.py
        (channel rule from L/R corr → DC removal → resample 44.1 kHz → peak clamp → PCM_24)
    → ingest/<slot>.wav (NOT loudness-normalised — that happens at inference time)
    → post-ingest QC (sr / channels / L-R corr / integrated LUFS) with loud flags
    → provenance/<slot>.txt (description verbatim) + parsed programme + credited players
    → candidate windows by EXCLUSION (dead air) + tonality ranking (spectral flatness)
Part B — dataset items (slot 1, in-domain control, publisher MASTER, frozen test split):
    shortlist 5 songs per genre from eval_manifest ⋈ source_manifest_v2 ⋈ chunk_activities,
    2 windows per song at the song's own median audible-class count.
Output: candidates/<self-describing>.wav + manifests/demo_sets/v1_candidates.{parquet,csv}.

Run:
    uv run python scripts/build_demo_set_candidates.py --config configs/demo_set_v1.yaml
      --part {A,B,both}   default both
      --skip-render       tables only
      --freeze            stage 2: take `freeze.picks` from the config → final/ (as-is +
                          _normalised twins) + manifests/demo_sets/v1.{parquet,csv}
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from src.data.ingest import (  # noqa: E402  (the standard ingest ops, unchanged)
    IngestConfig, apply_channel, decide_channel_action, peak_normalize, remove_dc, resample_to,
)

# instrument names we know how to recognise in prose descriptions (Korean names, incl. the
# 대취타 winds our data barely covers). Order matters only for readability.
KNOWN_INSTRUMENTS = [
    "피리", "대금", "소금", "단소", "해금", "아쟁", "가야금", "거문고", "양금", "대쟁", "생황",
    "태평소", "나발", "나각", "장구", "좌고", "북", "소리북", "용고", "징", "자바라", "꽹과리",
    "소고", "박", "편종", "편경", "방향", "집박", "등채", "고수", "소리", "병창",
]
_TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")   # h:mm:ss | m:ss


# ---------------------------------------------------------------------------- helpers
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["storage_root"] = Path(cfg["storage_root"]).expanduser()
    return cfg


def storage_dirs(cfg: dict) -> dict[str, Path]:
    """raw/ ingest/ candidates/ provenance/ under storage_root (created on demand)."""
    dirs = {k: cfg["storage_root"] / k for k in ("raw", "ingest", "candidates", "provenance")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def mmss(seconds: float) -> str:
    """Seconds → zero-padded MMSS for filenames (minutes may exceed 59: 65:44 → '6544')."""
    total = int(round(seconds))
    return f"{total // 60:02d}{total % 60:02d}"


def hms(seconds: float) -> str:
    total = int(round(seconds))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def table_paths(base: Path) -> tuple[Path, Path]:
    """demo_sets tables live side by side: <base>.parquet + <base>.csv (per the spec)."""
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".parquet"), base.with_suffix(".csv")


# ---------------------------------------------------------------------------- Part A: acquire / probe
def find_raw_file(raw_dir: Path, slot: str) -> Path:
    """The media file yt-dlp wrote for this slot (raw/<slot>.<ext>, not the sidecars)."""
    hits = [p for p in raw_dir.glob(f"{slot}.*")
            if p.suffix not in (".description", ".json", ".log", ".txt")]
    if len(hits) != 1:
        raise FileNotFoundError(f"{slot}: expected exactly one raw media file, found {hits}")
    return hits[0]


def probe_raw(path: Path) -> dict:
    """ffprobe the acquired stream: container / codec / bitrate / sr / channels (pre-ingest)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    fmt = info["format"]
    bitrate = audio.get("bit_rate") or fmt.get("bit_rate")
    return {
        "raw_container": fmt.get("format_name"),
        "raw_codec": audio.get("codec_name"),
        "raw_bitrate_kbps": round(float(bitrate) / 1000, 1) if bitrate else None,
        "raw_sr": int(audio.get("sample_rate")),
        "raw_channels": int(audio.get("channels")),
        "raw_channel_layout": audio.get("channel_layout"),
        "raw_duration_s": round(float(fmt.get("duration")), 3),
    }


def decode_raw(path: Path, scratch_dir: Path) -> tuple[np.ndarray, int]:
    """Decode the stream losslessly (float32 PCM WAV) at its native rate/channels via ffmpeg.

    Goes through a temp file rather than a pipe: libsndfile's virtual-IO on a BytesIO is
    ~10× slower than a real file, and a piped WAV has no valid length header anyway.
    """
    tmp = scratch_dir / f"{path.stem}.decoded.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-acodec", "pcm_f32le", str(tmp)],
                   check=True)
    audio, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    tmp.unlink()
    return audio, sr


def channel_metrics(audio: np.ndarray) -> dict:
    """peak / DC / L-R correlation of a (frames, channels) array — the QC quantities the
    channel rule and the flags need. Same maths as scripts/audio_qc.content_metrics, but
    per-channel column views instead of axis-1 reductions over (N, 2), which numpy runs
    row by row (76 s per 7 min of audio; the same numbers here take under a second)."""
    channels = audio.shape[1]
    peak = max(float(np.abs(audio[:, c]).max()) for c in range(channels)) if audio.size else 0.0
    dc = float(audio.mean(dtype=np.float64))
    if channels != 2:
        return {"peak": peak, "dc_offset": dc, "lr_corr": float("nan"), "lr_identical": None}
    left, right = audio[:, 0], audio[:, 1]
    identical = bool(np.array_equal(left, right))
    if left.std() < 1e-9 or right.std() < 1e-9:      # constant channel => corr undefined
        corr = 1.0 if identical else 0.0
    else:
        corr = float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])
    return {"peak": peak, "dc_offset": dc, "lr_corr": corr, "lr_identical": identical}


def yt_metadata(raw_dir: Path, slot: str) -> dict:
    info = json.loads((raw_dir / f"{slot}.info.json").read_text())
    return {"title": info.get("title"), "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"), "yt_format_id": info.get("format_id"),
            "yt_duration_s": info.get("duration")}


# ---------------------------------------------------------------------------- Part A: ingest + QC
def ingest_external(audio: np.ndarray, src_sr: int, ingest_cfg: IngestConfig, out_path: Path) -> dict:
    """Run the standard per-file ingest chain on decoded audio and write PCM_24 at 44.1 kHz.

    Mirrors ingest.process_task step for step; the channel rule reads L/R correlation
    measured on the decoded raw audio (there is no QC parquet row for external items).
    """
    pre = channel_metrics(audio)                     # pre-ingest L/R corr etc.
    action = decide_channel_action(audio.shape[1], pre["lr_corr"], pre["lr_identical"], ingest_cfg)
    ops = []
    audio = apply_channel(audio, action, ingest_cfg.keep_channel)
    if action == "pick_channel":
        ops.append(f"pick_ch{ingest_cfg.keep_channel}")
    audio = remove_dc(audio); ops.append("dc")
    if src_sr != ingest_cfg.target_sr:
        audio = resample_to(audio, src_sr, ingest_cfg.target_sr, ingest_cfg.resampler_quality)
        ops.append(f"resample({src_sr}->{ingest_cfg.target_sr})")
    audio, pk_before, pk_after = peak_normalize(audio, ingest_cfg.peak_ceiling, ingest_cfg.peak_target)
    if pk_after != pk_before:
        ops.append(f"peak_norm({pk_before:.3f}->{pk_after:.3f})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, ingest_cfg.target_sr, subtype=ingest_cfg.output_subtype)
    return {"raw_lr_corr": pre["lr_corr"], "raw_peak": pre["peak"], "raw_dc_offset": pre["dc_offset"],
            "channel_action": action, "peak_before": pk_before, "peak_after": pk_after,
            "ops_applied": ";".join(ops)}


def post_ingest_qc(path: Path) -> dict:
    """sr / channels / L-R corr / peak / integrated LUFS of the stored ingest file."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    m = channel_metrics(audio)
    lufs = float(pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64)))
    return {"ingest_sr": sr, "ingest_channels": audio.shape[1],
            "ingest_duration_s": round(audio.shape[0] / sr, 3),
            "ingest_lr_corr": m["lr_corr"], "ingest_peak": m["peak"], "ingest_lufs": round(lufs, 2)}


def qc_flags(row: dict, flags_cfg: dict) -> list[str]:
    """Loud, human-readable flags for anything outside the ensemble-dataset norm."""
    out = []
    if row["ingest_channels"] != flags_cfg["expect_channels"]:
        out.append(f"NOT_{flags_cfg['expect_channels']}CH(ch={row['ingest_channels']})")
    lo, hi = flags_cfg["lr_corr_typical"]
    corr = row["ingest_lr_corr"]
    if corr is None or (isinstance(corr, float) and np.isnan(corr)):
        out.append("LR_CORR_UNDEFINED")
    elif not (lo <= corr <= hi):
        out.append(f"LR_CORR_ATYPICAL({corr:.3f} vs typical {lo}-{hi})")
    lo, hi = flags_cfg["lufs_typical"]
    if not (lo <= row["ingest_lufs"] <= hi):
        out.append(f"LUFS_ATYPICAL({row['ingest_lufs']:.1f} vs typical {lo}..{hi})")
    if row["channel_action"] != "keep_stereo":
        out.append(f"CHANNEL_ACTION={row['channel_action']}")
    return out


# ---------------------------------------------------------------------------- Part A: provenance parsing
def parse_programme(description: str) -> list[dict]:
    """Every line carrying a timestamp → {start_s, title}. Works for both '01. 수제천 (0:20)'
    and '02:10 해금 협주곡 …' layouts."""
    items = []
    for line in description.splitlines():
        m = _TS.search(line)
        if not m:
            continue
        h, mnt, sec = m.groups()
        start = (int(h) if h else 0) * 3600 + int(mnt) * 60 + int(sec)
        title = (line[:m.start()] + line[m.end():]).strip(" ()|-–\t")
        title = re.sub(r"^\d{1,2}\.\s*", "", title).strip()
        items.append({"start_s": float(start), "title": title})
    return items


def parse_credited_players(block: str) -> list[dict]:
    """'피리/황규상·이건회, 대금/이상원' style credits → [{instrument, count, players}]."""
    out = []
    for m in re.finditer(r"([가-힣]+)\s*/\s*([가-힣·,\s()준단원]+?)(?=(?:,\s*[가-힣]+\s*/)|$|\n)", block):
        instrument, names = m.group(1), m.group(2)
        players = [p.strip() for p in re.split(r"[·,]", names) if p.strip()]
        out.append({"instrument": instrument, "count": len(players), "players": players})
    return out


def parse_expected_instrumentation(slot: str, description: str, programme: list[dict]) -> dict:
    """Credited instrumentation from the description — per piece when the description
    credits per piece (slot 2), otherwise item-level. Counts are null when the text names
    the instrument but not the players."""
    lines = description.splitlines()
    if slot == "slot2":                                     # per-piece '○ instr/names' credits
        pieces, current = {}, None
        for line in lines:
            if _TS.search(line):
                current = parse_programme(line)[0]["title"]; pieces[current] = []
            elif current and line.strip().startswith("○"):
                pieces[current].extend(parse_credited_players(line.strip("○ ").strip()))
        return {"granularity": "per_piece",
                "pieces": {k: [p for p in v if p["instrument"] not in ("무용", "작곡")] for k, v in pieces.items()},
                "notes": "무용 (dancer) and 작곡 (composer) credits dropped; 집박 = clapper/conductor."}
    if slot == "slot5":                                     # '소리: a, b / 고수: c'
        m = re.search(r"소리:\s*([^/]+)/\s*고수:\s*(\S+)", description)
        singers = [s.strip() for s in m.group(1).split(",")] if m else []
        drummer = m.group(2).strip() if m else None
        return {"granularity": "item", "credits": [
            {"instrument": "소리 (voice)", "count": len(singers), "players": singers},
            {"instrument": "북 (고수)", "count": 1 if drummer else None, "players": [drummer] if drummer else []}],
            "notes": "voice class does not exist in the 9-class scheme."}
    if slot == "slot4":
        return {"granularity": "item", "credits": [
            {"instrument": "해금 (협연 solo)", "count": 1, "players": ["변아영"]},
            {"instrument": "국악관현악단 (orchestra, roster not credited)", "count": None, "players": []}],
            "notes": "section = 해금 협주곡 '끝없이 하늘 끝으로' 02:10–03:06 only; roster not printed."}
    # slot 3 and any prose description: keyword mentions, counts unknown
    mentioned = [i for i in KNOWN_INSTRUMENTS if i in description and i not in ("소리", "박", "북", "등채")]
    return {"granularity": "item",
            "credits": [{"instrument": i, "count": None, "players": []} for i in mentioned]
            + [{"instrument": "등채 (conductor's baton, cues only)", "count": 1, "players": ["김기동"]}],
            "notes": "prose description — instruments named, players not credited per part; "
                     "ensemble = 국립국악원 정악단."}


def write_provenance(prov_dir: Path, raw_dir: Path, slot: str, meta: dict,
                     programme: list[dict], instrumentation: dict) -> None:
    """provenance/<slot>.txt = description VERBATIM; sidecars carry the parsed views."""
    shutil.copyfile(raw_dir / f"{slot}.description", prov_dir / f"{slot}.txt")
    shutil.copyfile(raw_dir / f"{slot}.info.json", prov_dir / f"{slot}.info.json")
    (prov_dir / f"{slot}.parsed.json").write_text(json.dumps(
        {"slot": slot, **meta, "programme": programme, "expected_instrumentation": instrumentation},
        ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------- Part A: candidate windows
def block_features(mono: np.ndarray, sr: int, heur: dict) -> pd.DataFrame:
    """Per analysis block: RMS dBFS + mean spectral flatness in the configured band."""
    block = int(heur["block_s"] * sr)
    n_fft = int(heur["fft_size"])
    hop = n_fft // 2
    lo_hz, hi_hz = heur["flatness_band_hz"]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    window = np.hanning(n_fft)
    n_blocks = len(mono) // block
    rms_db = np.empty(n_blocks); flat = np.empty(n_blocks)
    for b in range(n_blocks):
        seg = mono[b * block:(b + 1) * block].astype(np.float64)
        rms_db[b] = 20 * np.log10(max(np.sqrt(np.mean(seg ** 2)), 1e-12))
        starts = np.arange(0, len(seg) - n_fft + 1, hop)
        frames = np.stack([seg[s:s + n_fft] * window for s in starts])
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        power = power[:, band] + 1e-20
        flat[b] = float(np.mean(np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)))
    return pd.DataFrame({"t": np.arange(n_blocks) * heur["block_s"], "rms_db": rms_db, "flatness": flat})


def window_table(blocks: pd.DataFrame, duration_s: float, excerpt_s: float, heur: dict,
                 lo: float | None = None, hi: float | None = None) -> pd.DataFrame:
    """Every candidate 30 s window (stride hop_s) with its summary features + dead-air verdict."""
    lo = heur["edge_margin_s"] if lo is None else lo
    hi = duration_s - heur["edge_margin_s"] if hi is None else hi
    dead_blocks = max(1, int(round(heur["dead_air_block_s"] / heur["block_s"])))
    rows = []
    for start in np.arange(lo, hi - excerpt_s + 1e-9, heur["hop_s"]):
        sel = blocks[(blocks.t >= start) & (blocks.t < start + excerpt_s)]
        if len(sel) < 2:
            continue
        # dead air = any run of `dead_blocks` consecutive blocks below the floor
        below = (sel.rms_db.values < heur["dead_air_dbfs"]).astype(int)
        run = np.convolve(below, np.ones(dead_blocks, dtype=int), mode="valid")
        rows.append({"start_s": float(start), "end_s": float(start + excerpt_s),
                     "mean_dbfs": round(float(sel.rms_db.mean()), 2),
                     "min_block_dbfs": round(float(sel.rms_db.min()), 2),
                     "std_dbfs": round(float(sel.rms_db.std()), 2),
                     "mean_flatness": round(float(sel.flatness.mean()), 4),
                     "max_flatness": round(float(sel.flatness.max()), 4),
                     "dead_air": bool((run >= dead_blocks).any())})
    return pd.DataFrame(rows)


def rank_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Exclude dead air; keep the tonal half (flatness ≤ median of the pool — applause / crowd
    noise / ambience are the flat tail); then loudest first as the tutti proxy. Ranking by
    flatness alone favoured the quietest, sparsest passages over full-ensemble playing."""
    ok = windows[~windows.dead_air].copy()
    if len(ok) == 0:
        return ok
    tonal = ok[ok.mean_flatness <= ok.mean_flatness.median()]
    return tonal.sort_values(["mean_dbfs", "mean_flatness"], ascending=[False, True]).reset_index(drop=True)


def pick_non_overlapping(ranked: pd.DataFrame, n: int, min_gap_s: float) -> pd.DataFrame:
    picks = []
    for r in ranked.itertuples(index=False):
        if all(abs(r.start_s - p.start_s) >= min_gap_s for p in picks):
            picks.append(r)
        if len(picks) == n:
            break
    return pd.DataFrame(picks)


def spread_by_piece(windows: pd.DataFrame, programme: list[dict], duration_s: float, n: int,
                    skip_titles: tuple[str, ...] = ("병창",)) -> pd.DataFrame:
    """One best window per programme piece (window fully inside the piece), instrumental
    pieces first; if pieces < n, fill with the next-best non-overlapping windows."""
    bounds = [(p["title"], p["start_s"], (programme[i + 1]["start_s"] if i + 1 < len(programme) else duration_s))
              for i, p in enumerate(programme)]
    picks = []
    for title, lo, hi in bounds:
        if any(k in title for k in skip_titles):
            continue
        inside = windows[(windows.start_s >= lo) & (windows.end_s <= hi)]
        ranked = rank_windows(inside)
        if len(ranked):
            picks.append({**ranked.iloc[0].to_dict(), "piece": title})
    picks = pd.DataFrame(picks).sort_values("mean_dbfs", ascending=False).head(n)
    return picks.sort_values("start_s").reset_index(drop=True)


def evenly_spaced_windows(lo: float, hi: float, excerpt_s: float, n: int, windows: pd.DataFrame) -> pd.DataFrame:
    """Section shorter than n non-overlapping excerpts → n evenly spaced (overlapping) starts,
    each snapped to the nearest analysed window so it still carries features."""
    starts = np.linspace(lo, hi - excerpt_s, n)
    rows = []
    for s in starts:
        nearest = windows.iloc[(windows.start_s - s).abs().argmin()].to_dict()
        rows.append(nearest)
    return pd.DataFrame(rows).drop_duplicates("start_s").reset_index(drop=True)


def piece_at(t: float, programme: list[dict]) -> str | None:
    cur = None
    for p in programme:
        if p["start_s"] <= t:
            cur = p["title"]
    return cur


def render_excerpt(src: Path, start_s: float, excerpt_s: float, out: Path) -> None:
    info = sf.info(str(src))
    audio, sr = sf.read(str(src), start=int(start_s * info.samplerate),
                        frames=int(excerpt_s * info.samplerate), dtype="float32", always_2d=True)
    sf.write(str(out), audio, sr, subtype="PCM_24")


def build_part_a(cfg: dict, dirs: dict[str, Path], render: bool) -> tuple[list[dict], list[dict]]:
    """Returns (candidate rows, per-item QC/provenance rows)."""
    ingest_cfg = IngestConfig.load(REPO_ROOT / cfg["ingest_config"], REPO_ROOT)
    heur, excerpt_s = cfg["heuristic"], float(cfg["excerpt_s"])
    cand_rows, item_rows = [], []
    for slot, item in cfg["external"].items():
        print(f"\n=== {slot}: {item['short_title']}")
        raw = find_raw_file(dirs["raw"], slot)
        meta = {**yt_metadata(dirs["raw"], slot), **probe_raw(raw), "raw_file": raw.name}
        print(f"  raw: {meta['raw_container']} / {meta['raw_codec']} {meta['raw_bitrate_kbps']} kbps "
              f"{meta['raw_sr']} Hz {meta['raw_channels']}ch ({meta['raw_channel_layout']}) "
              f"{hms(meta['raw_duration_s'])}")

        # ingest through the standard chain (idempotent: reuse an existing ingest file)
        ingest_path = dirs["ingest"] / f"{slot}.wav"
        if ingest_path.exists():
            ingest_row = json.loads((dirs["provenance"] / f"{slot}.ingest.json").read_text())
            print("  ingest: cached")
        else:
            audio, sr = decode_raw(raw, dirs["raw"])
            ingest_row = ingest_external(audio, sr, ingest_cfg, ingest_path)
            del audio
            (dirs["provenance"] / f"{slot}.ingest.json").write_text(json.dumps(ingest_row, indent=2))
        qc = post_ingest_qc(ingest_path)
        flags = qc_flags({**ingest_row, **qc}, cfg["qc_flags"])
        print(f"  ingest: {qc['ingest_sr']} Hz {qc['ingest_channels']}ch  L/R corr {qc['ingest_lr_corr']:.4f}  "
              f"LUFS {qc['ingest_lufs']}  peak {qc['ingest_peak']:.3f}  ops={ingest_row['ops_applied']}")
        print(f"  flags: {flags or 'none'}")

        # provenance: verbatim description + parsed programme / credits
        description = (dirs["raw"] / f"{slot}.description").read_text()
        programme = parse_programme(description)
        instrumentation = parse_expected_instrumentation(slot, description, programme)
        write_provenance(dirs["provenance"], dirs["raw"], slot, meta, programme, instrumentation)

        # candidate windows
        blocks_cache = dirs["provenance"] / f"{slot}.blocks.parquet"
        if blocks_cache.exists():
            blocks = pd.read_parquet(blocks_cache)
        else:
            mono, sr = sf.read(str(ingest_path), dtype="float32", always_2d=True)
            blocks = block_features(mono.mean(axis=1), sr, heur)
            del mono
            blocks.to_parquet(blocks_cache, index=False)
        duration = qc["ingest_duration_s"]
        section = item.get("section")
        lo, hi = (section["start_s"], section["end_s"]) if section else (None, None)
        windows = window_table(blocks, duration, excerpt_s, heur, lo, hi)
        n = int(item["num_candidates"])
        if item.get("spread_across_pieces"):
            picks = spread_by_piece(windows, programme, duration, n)
        elif section and heur["overlap_allowed_if_section_short"] and (hi - lo) < n * excerpt_s:
            picks = evenly_spaced_windows(lo, hi, excerpt_s, n, windows)
            print(f"  ⚠️ section {hms(lo)}–{hms(hi)} is only {hi - lo:.0f} s → {len(picks)} overlapping windows")
        else:
            picks = pick_non_overlapping(rank_windows(windows), n, heur["min_gap_between_picks_s"]).sort_values("start_s")
        n_dead = int(windows.dead_air.sum())
        print(f"  windows analysed {len(windows)} · rejected for dead air {n_dead} · picked {len(picks)}")

        item_row = {"slot": slot, "item_id": slot, "source_kind": "youtube", "url": item["url"],
                    "video_id": item["video_id"], "short_title": item["short_title"], "setting": item["setting"],
                    **meta, **ingest_row, **qc, "qc_flags": ";".join(flags),
                    "expected_instrumentation": json.dumps(instrumentation, ensure_ascii=False),
                    "programme": json.dumps(programme, ensure_ascii=False)}
        item_rows.append(item_row)
        for i, w in enumerate(picks.itertuples(index=False), start=1):
            fname = f"{slot}_c{i:02d}_{mmss(w.start_s)}-{mmss(w.end_s)}.wav"
            piece = getattr(w, "piece", None) or piece_at(w.start_s, programme)
            piece_instr = (instrumentation["pieces"].get(piece) if instrumentation.get("granularity") == "per_piece"
                           else instrumentation.get("credits"))
            cand_rows.append({
                "slot": slot, "item_id": slot, "candidate_idx": i, "candidate_file": fname,
                "start_s": w.start_s, "end_s": w.end_s, "start_hms": hms(w.start_s), "end_hms": hms(w.end_s),
                "piece": piece, "expected_instrumentation": json.dumps(piece_instr, ensure_ascii=False),
                "mean_dbfs": w.mean_dbfs, "min_block_dbfs": w.min_block_dbfs, "std_dbfs": w.std_dbfs,
                "mean_flatness": w.mean_flatness, "max_flatness": w.max_flatness,
                "selection_note": "heuristic proposal — listener gate pending"})
            if render:
                render_excerpt(ingest_path, w.start_s, excerpt_s, dirs["candidates"] / fname)
            print(f"    c{i:02d} {hms(w.start_s)}–{hms(w.end_s)}  {w.mean_dbfs:6.1f} dBFS  flat {w.mean_flatness:.3f}"
                  f"  {('[' + piece + ']') if piece else ''}")
    return cand_rows, item_rows


# ---------------------------------------------------------------------------- Part B: slot 1
def eligible_songs(s1: dict) -> pd.DataFrame:
    """Test-split songs of the two genres, minus exclusions, with master path + stem classes."""
    ev = pd.read_parquet(REPO_ROOT / s1["eval_manifest"])
    sm = pd.read_parquet(REPO_ROOT / s1["source_manifest"])
    sm = sm[(sm.dataset == "71955") & (sm.split == s1["split"])]
    excluded = set(s1["exclude_num_ids"]) | set(s1["exclude_pinned_listening"])
    songs = ev[(ev.split == s1["split"]) & ev.genre_sub.isin(list(s1["genres"]))].copy()
    songs["excluded_by_rule"] = songs.num_id.isin(excluded)
    masters = sm[sm.role == "master"].set_index("song_id")
    stems = sm[sm.role == "stem"].groupby("song_id").agg(
        stem_classes=("stem_group", lambda s: sorted(set(s))),
        num_stem_files=("file_id", "size"))
    songs = songs.join(masters[["out_path", "out_duration"]], on="song_id").join(stems, on="song_id")
    songs = songs.rename(columns={"out_path": "master_path", "out_duration": "duration_s"})
    return songs.reset_index(drop=True)


def audible_per_chunk(ca: pd.DataFrame, s1: dict) -> pd.DataFrame:
    """chunk_activities (one chunk length) → per chunk: number + list of audible MODELLED classes."""
    cols = [f"cov_{c}" for c in s1["modelled_classes"]]
    sub = ca[ca.chunk_len_s == float(s1["chunk_len_s"])].copy()
    cov = sub[cols].values
    audible = cov > float(s1["coverage_threshold"])
    sub["n_audible"] = audible.sum(axis=1)                 # binary count (class-count criterion)
    sub["sum_coverage"] = cov.sum(axis=1)                  # continuous: simultaneously-active classes
    sub["audible_classes"] = [tuple(c for c, a in zip(s1["modelled_classes"], row) if a) for row in audible]
    return sub[["song_id", "genre_sub", "split", "start_s", "n_audible", "sum_coverage", "audible_classes"]]


def song_density(chunks: pd.DataFrame, s1: dict) -> pd.DataFrame:
    """Per song: audible density = mean simultaneously-active modelled classes (Σ per-class
    coverage per chunk, averaged — continuous, so 'closest to median' can discriminate),
    the median binary count, and the classes audible in ≥25 % of chunks (song-level set)."""
    rows = []
    for song, g in chunks.groupby("song_id"):
        counts: dict[str, int] = {}
        for classes in g.audible_classes:
            for c in classes:
                counts[c] = counts.get(c, 0) + 1
        song_audible = sorted(c for c, k in counts.items() if k / len(g) >= 0.25)
        rows.append({"song_id": song, "num_chunks": len(g),
                     "audible_density": round(float(g.sum_coverage.mean()), 3),
                     "mean_n_audible": round(float(g.n_audible.mean()), 3),
                     "median_n_audible": float(g.n_audible.median()),
                     "audible_classes_song": song_audible, "num_audible_classes_song": len(song_audible)})
    return pd.DataFrame(rows)


def shortlist_genre(songs: pd.DataFrame, genre: str, gcfg: dict, s1: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the per-genre criterion; returns (ranked shortlist, reference stats)."""
    pool = songs[songs.genre_sub == genre].copy()
    ref = {"genre": genre, "num_test_songs": len(pool),
           "genre_median_density_all_test": float(pool.audible_density.median())}
    elig = pool[(~pool.excluded_by_rule) & (pool.duration_s >= float(s1["min_song_duration_s"]))].copy()
    ref["num_eligible"] = len(elig)
    ref["genre_median_density_eligible"] = float(elig.audible_density.median()) if len(elig) else None
    if gcfg["criterion"] == "closest_to_genre_median_density":
        elig["distance_to_target"] = (elig.audible_density - ref["genre_median_density_all_test"]).abs()
        elig["in_target_range"] = True
        ranked = elig.sort_values(["distance_to_target", "song_id"])
    elif gcfg["criterion"] == "audible_class_count_in_range":
        lo, hi = gcfg["target_range"]
        ref["target_range"] = [lo, hi]
        in_range = elig[elig.num_audible_classes_song.between(lo, hi)].copy()
        ref["num_in_range"] = len(in_range)
        centre = (lo + hi) / 2
        elig["in_target_range"] = elig.num_audible_classes_song.between(lo, hi)
        elig["distance_to_target"] = np.where(
            elig.in_target_range, (elig.audible_density - centre).abs(),
            # out of range: whole-class distance to the range edge (never mistaken for in-range)
            elig.num_audible_classes_song.apply(lambda k: 1 + min(abs(k - lo), abs(k - hi))))
        # report, don't widen silently: in-range songs first, extras flagged in_target_range=False
        ranked = elig.sort_values(["in_target_range", "distance_to_target", "song_id"], ascending=[False, True, True])
    else:
        raise ValueError(gcfg["criterion"])
    return ranked.head(int(s1["shortlist_per_genre"])).reset_index(drop=True), ref


def song_windows(chunks: pd.DataFrame, song: pd.Series, s1: dict, excerpt_s: float) -> pd.DataFrame:
    """30 s windows (3 consecutive chunks) whose mean audible count sits closest to the song's
    own median; head/tail margins respected; the two picks kept ≥ 60 s apart when possible."""
    g = chunks[chunks.song_id == song.song_id].sort_values("start_s").reset_index(drop=True)
    per_win = int(round(excerpt_s / float(s1["chunk_len_s"])))
    margin, dur = float(s1["edge_margin_s"]), float(song.duration_s)
    median = float(g.n_audible.median())
    rows = []
    for i in range(len(g) - per_win + 1):
        w = g.iloc[i:i + per_win]
        start, end = float(w.start_s.iloc[0]), float(w.start_s.iloc[0]) + excerpt_s
        if start < margin or end > dur - margin:
            continue
        classes = sorted(set().union(*[set(c) for c in w.audible_classes]))
        rows.append({"start_s": start, "end_s": end, "window_mean_n_audible": round(float(w.n_audible.mean()), 2),
                     "window_min_n_audible": int(w.n_audible.min()), "song_median_n_audible": median,
                     "window_audible_classes": "·".join(classes),
                     "distance_to_song_median": abs(float(w.n_audible.mean()) - median)})
    wins = pd.DataFrame(rows)
    # ties are the norm (most windows sit exactly at the median): among near-best windows
    # take the one nearest ⅓ of the song, then the one nearest ⅔ — spread, deterministic
    near_best = wins[wins.distance_to_song_median <= wins.distance_to_song_median.min() + 0.1]
    picks = []
    for anchor in np.linspace(dur / 3, 2 * dur / 3, int(s1["windows_per_song"])):
        free = lambda df: df[[all(abs(s - p["start_s"]) >= excerpt_s for p in picks) for s in df.start_s]]
        pool = free(near_best)
        if len(pool) == 0:                       # near-best pool exhausted → best remaining window overall
            pool = free(wins).sort_values("distance_to_song_median").head(5)
        if len(pool) == 0:
            continue
        picks.append(pool.iloc[(pool.start_s + excerpt_s / 2 - anchor).abs().argmin()].to_dict())
    return pd.DataFrame(picks).sort_values("start_s").reset_index(drop=True)


def build_part_b(cfg: dict, dirs: dict[str, Path], render: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (candidate rows, shortlist rows, per-genre reference stats)."""
    s1, excerpt_s = cfg["slot1"], float(cfg["excerpt_s"])
    songs = eligible_songs(s1)
    ca = pd.read_parquet(REPO_ROOT / s1["chunk_activities"])
    chunks = audible_per_chunk(ca[ca.song_id.isin(songs.song_id)], s1)
    songs = songs.merge(song_density(chunks, s1), on="song_id", how="left")
    cand_rows, short_rows, refs = [], [], []
    for genre, gcfg in s1["genres"].items():
        ranked, ref = shortlist_genre(songs, genre, gcfg, s1)
        refs.append(ref)
        print(f"\n=== slot1 / {genre}: {ref}")
        for rank, song in enumerate(ranked.itertuples(index=False), start=1):
            short_rows.append({
                "slot": "slot1", "genre_sub": genre, "rank": rank, "song_id": song.song_id, "num_id": song.num_id,
                "duration_s": song.duration_s, "stem_classes": "·".join(song.stem_classes),
                "audible_classes_song": "·".join(song.audible_classes_song),
                "num_audible_classes_song": song.num_audible_classes_song,
                "audible_density": song.audible_density, "mean_n_audible": song.mean_n_audible,
                "median_n_audible": song.median_n_audible, "in_target_range": bool(song.in_target_range),
                "genre_median_density_all_test": ref["genre_median_density_all_test"],
                "distance_to_target": round(float(song.distance_to_target), 3),
                "master_path": song.master_path})
            print(f"  #{rank} {song.song_id}  {song.duration_s:6.1f} s  stems {'·'.join(song.stem_classes)}  "
                  f"audible {'·'.join(song.audible_classes_song)}  density {song.audible_density:.2f}  "
                  f"Δ {song.distance_to_target:.2f}{'' if song.in_target_range else '  ⚠️ OUT OF TARGET RANGE'}")
            wins = song_windows(chunks, song, s1, excerpt_s)
            for i, w in enumerate(wins.itertuples(index=False), start=1):
                fname = f"slot1_{gcfg['tag']}_{song.num_id}_c{i:02d}_{mmss(w.start_s)}-{mmss(w.end_s)}.wav"
                cand_rows.append({
                    "slot": "slot1", "item_id": f"slot1_{gcfg['tag']}_{song.num_id}", "candidate_idx": i,
                    "candidate_file": fname, "start_s": w.start_s, "end_s": w.end_s,
                    "start_hms": hms(w.start_s), "end_hms": hms(w.end_s), "piece": None,
                    "song_id": song.song_id, "genre_sub": genre, "variant": "master",
                    "window_mean_n_audible": w.window_mean_n_audible, "song_median_n_audible": w.song_median_n_audible,
                    "window_audible_classes": w.window_audible_classes,
                    "expected_instrumentation": json.dumps(
                        {"stem_classes": song.stem_classes, "audible_classes_song": song.audible_classes_song},
                        ensure_ascii=False),
                    "selection_note": "manifest-only selection — no model output or score referenced"})
                if render:
                    render_excerpt(REPO_ROOT / song.master_path, w.start_s, excerpt_s, dirs["candidates"] / fname)
                print(f"      c{i:02d} {hms(w.start_s)}–{hms(w.end_s)}  audible {w.window_mean_n_audible} "
                      f"(song median {w.song_median_n_audible})  {w.window_audible_classes}")
    return cand_rows, short_rows, refs


# ---------------------------------------------------------------------------- Stage 2: freeze
def normalise_excerpt(audio: np.ndarray, sr: int, target_lufs: float, peak_ceiling: float) -> tuple[np.ndarray, dict]:
    """The inference-time rule (mix_dataset._normalize): one gain to target_lufs, then a
    peak guard. Returns (audio, {measured_lufs, gain_db, peak_guard_applied, realised_lufs})."""
    meter = pyloudnorm.Meter(sr)
    measured = float(meter.integrated_loudness(audio.astype(np.float64)))
    gain = 10 ** ((target_lufs - measured) / 20)
    out = audio * np.float32(gain)
    peak = float(np.abs(out).max())
    guarded = peak > peak_ceiling
    if guarded:
        out = out * np.float32(peak_ceiling / peak)
    realised = float(meter.integrated_loudness(out.astype(np.float64)))
    return out, {"measured_lufs_as_is": round(measured, 2), "gain_db": round(20 * np.log10(gain), 2),
                 "peak_guard_applied": bool(guarded), "realised_lufs": round(realised, 2),
                 "peak_normalised": round(float(np.abs(out).max()), 4)}


def build_freeze(cfg: dict, dirs: dict[str, Path]) -> pd.DataFrame:
    """Stage 2: resolve the listened picks against the candidate tables, render as-is +
    normalised twins into final/, and write the frozen v1 manifest."""
    fz = cfg["freeze"]
    final_dir = cfg["storage_root"] / fz["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    base = REPO_ROOT / cfg["out_candidates"]
    cands = pd.read_parquet(base.with_suffix(".parquet")).set_index("candidate_file")
    items = pd.read_parquet(base.parent / f"{base.name}_external_items.parquet").set_index("slot")
    shortlist = pd.read_parquet(base.parent / f"{base.name}_slot1_shortlist.parquet").set_index("song_id")
    masters = pd.read_parquet(REPO_ROOT / cfg["slot1"]["source_manifest"])
    masters = masters[masters.role == "master"].set_index("song_id")

    rows = []
    for item_key, cand_file in fz["picks"].items():
        c = cands.loc[cand_file]
        slot = c["slot"]
        stem = Path(cand_file).stem
        as_is = final_dir / f"{stem}.wav"
        normalised = final_dir / f"{stem}{fz['normalised_suffix']}.wav"
        shutil.copyfile(dirs["candidates"] / cand_file, as_is)
        audio, sr = sf.read(str(as_is), dtype="float32", always_2d=True)
        out, norm = normalise_excerpt(audio, sr, float(fz["target_lufs"]), float(fz["peak_ceiling"]))
        sf.write(str(normalised), out, sr, subtype="PCM_24")

        row = {"item_key": item_key, "slot": slot, "source_kind": "youtube" if slot != "slot1" else "dataset_master",
               "start_s": c["start_s"], "end_s": c["end_s"], "start_hms": c["start_hms"], "end_hms": c["end_hms"],
               "excerpt_s": float(cfg["excerpt_s"]), "candidate_file": cand_file,
               "as_is_file": as_is.name, "normalised_file": normalised.name,
               "target_lufs": float(fz["target_lufs"]), **norm,
               "channels": int(audio.shape[1]), "sample_rate": sr,
               "expected_instrumentation": c["expected_instrumentation"],
               "selection_note": fz["notes"].get(item_key, "")}
        if slot == "slot1":
            song = c["song_id"]; sl = shortlist.loc[song]; m = masters.loc[song]
            row.update(url=None, video_id=None, video_title=None, uploader=None, upload_date=None,
                       song_id=song, genre_sub=c["genre_sub"], variant="master", piece=None,
                       stem_classes=sl["stem_classes"], audible_classes_song=sl["audible_classes_song"],
                       audible_density=sl["audible_density"], window_audible_classes=c["window_audible_classes"],
                       ingest_lr_corr=m["lr_corr"], raw_container="wav", raw_codec=m["src_subtype"],
                       raw_bitrate_kbps=None, raw_sr=int(m["src_sr"]), raw_channels=int(m["src_channels"]),
                       ingest_lufs_full_file=None, qc_flags="")
        else:
            it = items.loc[slot]
            row.update(url=it["url"], video_id=it["video_id"], video_title=it["title"], uploader=it["uploader"],
                       upload_date=it["upload_date"], song_id=None, genre_sub=None, variant=None, piece=c["piece"],
                       stem_classes=None, audible_classes_song=None, audible_density=None, window_audible_classes=None,
                       ingest_lr_corr=it["ingest_lr_corr"], raw_container=it["raw_container"], raw_codec=it["raw_codec"],
                       raw_bitrate_kbps=it["raw_bitrate_kbps"], raw_sr=int(it["raw_sr"]), raw_channels=int(it["raw_channels"]),
                       ingest_lufs_full_file=it["ingest_lufs"], qc_flags=it["qc_flags"])
        rows.append(row)
        print(f"  {item_key:14s} {cand_file:40s} as-is {norm['measured_lufs_as_is']:6.1f} LUFS → gain {norm['gain_db']:+5.1f} dB"
              f" → {norm['realised_lufs']:6.1f} LUFS{'  ⚠️ peak-guarded' if norm['peak_guard_applied'] else ''}")

    cols = ["item_key", "slot", "source_kind", "url", "video_title", "video_id", "uploader", "upload_date",
            "song_id", "genre_sub", "variant", "piece", "start_s", "end_s", "start_hms", "end_hms", "excerpt_s",
            "as_is_file", "normalised_file", "candidate_file", "channels", "sample_rate",
            "target_lufs", "measured_lufs_as_is", "gain_db", "peak_guard_applied", "realised_lufs", "peak_normalised",
            "raw_container", "raw_codec", "raw_bitrate_kbps", "raw_sr", "raw_channels", "ingest_lr_corr",
            "ingest_lufs_full_file", "qc_flags", "expected_instrumentation", "stem_classes", "audible_classes_song",
            "audible_density", "window_audible_classes", "selection_note"]
    df = pd.DataFrame(rows)[cols]
    out_parquet, out_csv = table_paths(REPO_ROOT / fz["out_manifest"])
    df.to_parquet(out_parquet, index=False); df.to_csv(out_csv, index=False)
    print(f"\nfrozen {len(df)} items → {out_parquet} (+csv) · audio → {final_dir}")
    return df


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="demo_set_v1 stage 1: candidates for listening")
    ap.add_argument("--config", default="configs/demo_set_v1.yaml")
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="stage 2: freeze the listened picks → v1 manifest")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    dirs = storage_dirs(cfg)
    render = not args.skip_render
    if args.freeze:
        build_freeze(cfg, dirs)
        return

    cand_a, items_a = build_part_a(cfg, dirs, render) if args.part in ("A", "both") else ([], [])
    cand_b, short_b, refs_b = build_part_b(cfg, dirs, render) if args.part in ("B", "both") else ([], [], [])

    out_parquet, out_csv = table_paths(REPO_ROOT / cfg["out_candidates"])
    cands = pd.DataFrame(cand_a + cand_b)
    if len(cands):
        cands.to_parquet(out_parquet, index=False); cands.to_csv(out_csv, index=False)
    side = REPO_ROOT / cfg["out_candidates"]
    if items_a:
        pd.DataFrame(items_a).to_parquet(side.parent / f"{side.name}_external_items.parquet", index=False)
        pd.DataFrame(items_a).to_csv(side.parent / f"{side.name}_external_items.csv", index=False)
    if short_b:
        pd.DataFrame(short_b).to_parquet(side.parent / f"{side.name}_slot1_shortlist.parquet", index=False)
        pd.DataFrame(short_b).to_csv(side.parent / f"{side.name}_slot1_shortlist.csv", index=False)
        (side.parent / f"{side.name}_slot1_reference.json").write_text(json.dumps(refs_b, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(cands)} candidate rows → {out_parquet} (+csv)"
          f"{' · external_items' if items_a else ''}{' · slot1_shortlist' if short_b else ''}")


if __name__ == "__main__":
    main()
