"""build_source_manifest.py — merge the ingest manifest with the ingest-store QC scan.

Produces `manifests/parquet/source_manifest.parquet` (+ csv twin in `manifests/csv/`):
ONE ROW PER INGESTED SOURCE FILE, carrying everything a dataloader or an offline stage
needs about that file —

    provenance + ops   (from manifests/parquet/ingest_manifest.parquet)
    split              (frozen 71955 song-level split, already denormalized by ingest)
    content metrics    (from manifests/parquet/audio_qc_ingest_*.parquet — active_frac, rms,
                        lr_corr, dead, clip_frac, dc_offset, peak)
    taxonomy           (from configs/stem_taxonomy.yaml — canonical instrument name,
                        working stem group, pitched flag, publisher 9-class group)

Why this table exists
---------------------
The two inputs are split by concern, not by grain: `ingest_manifest` records what was
DONE to each file, `audio_qc_ingest` records what each file IS. The mixing pipeline
needs both at once — e.g. sampling a random excerpt requires `active_frac` (some stems
like 박 are active <1% of the time, so a naive random start lands on silence), while
assigning that excerpt to a stem class requires the instrument taxonomy.

Grain discipline
----------------
This table is the SOURCE grain (one row per ingested file) and stays that way. The
pitch-shift pool is a DERIVED grain (one row per source × semitone) and belongs in a
separate `shift_pool_manifest` that foreign-keys back here via `file_id` — it must not
re-copy split/instrument/content columns, or they can drift per shift copy.

`file_id` is a stable primary key derived from `dataset` + the source path relative to
the dataset dir, so it survives a re-ingest or a re-cut split (it is NOT a row index).

Versions
--------
The table is versioned by OUTPUT NAME, never by overwriting: `source_manifest` is v1
(what exp001 trained against, kept as the historical record), `source_manifest_v2` is
v1 minus the publisher defects declared in `configs/manifest_exclusions.yaml`. An
experiment selects its version through `gugak_mix.source_manifest` in its YAML — no
path is ever edited in code.

Run:
    uv run python src/data/build_source_manifest.py [options]
      --taxonomy PATH   taxonomy yaml (default: configs/stem_taxonomy.yaml)
      --exclusions PATH exclusion rules yaml (default: none -> v1, nothing dropped)
      --out BASE        output basename (default: manifests/source_manifest)
      --dry-run         build + report, write nothing
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

# --- inputs (repo-relative; every path overridable on the CLI) ---
INGEST_MANIFEST = "manifests/parquet/ingest_manifest.parquet"
QC_INGEST = ["manifests/parquet/audio_qc_ingest_71955.parquet",
             "manifests/parquet/audio_qc_ingest_71470.parquet"]

# --- QC columns worth carrying; the rest (sr/subtype/channels/frames/duration) already
#     live in the ingest manifest as out_* and would just duplicate ---
QC_CONTENT_COLUMNS = ["peak", "rms", "dc_offset", "clip_frac", "active_frac", "dead",
                      "lr_corr", "lr_identical", "lr_max_diff"]


def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths.

    Layout rule (repo-wide): parquet = source of truth in manifests/parquet/,
    csv = eyeball copy in manifests/csv/. Both dirs are created on demand.

    Args:
        base: table basename, e.g. Path(".../manifests/source_manifest").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


def nfc(value: object) -> object:
    """NFC-normalize Korean strings so disk names and label tables join correctly."""
    return unicodedata.normalize("NFC", value) if isinstance(value, str) else value


# ---------- taxonomy ----------
def load_taxonomy(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read stem_taxonomy.yaml into a lookup frame plus a 71470 code -> name map.

    Args:
        path: path to configs/stem_taxonomy.yaml.

    Returns:
        (taxonomy frame indexed by canonical instrument name, {instrument_cd: name}).
    """
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = []
    for name, entry in spec["instruments"].items():
        rows.append({"instrument_canonical": nfc(name),
                     "stem_group": nfc(entry.get("stem_group")),
                     "pitched": bool(entry["pitched"]),
                     "instrument_group_71955": nfc(entry.get("group_71955")),
                     "code_71470": entry.get("code_71470")})
    taxonomy = pd.DataFrame(rows)
    code_to_name = {r.code_71470: r.instrument_canonical
                    for r in taxonomy.itertuples() if r.code_71470}
    return taxonomy, code_to_name


def canonical_instrument(frame: pd.DataFrame, code_to_name: dict[str, str]) -> pd.Series:
    """Resolve each row to a canonical instrument name across both datasets.

    71955 already stores Korean base names (digit-stripped at ingest); 71470 stores
    publisher instrument codes, which map through the taxonomy.

    Args:
        frame: merged manifest rows (needs `dataset` and `instrument`).
        code_to_name: {instrument_cd: canonical name} from the taxonomy.

    Returns:
        Series of canonical instrument names (NaN for masters, which have no instrument).
    """
    instrument = frame["instrument"].map(nfc)
    is_solo = frame["dataset"].eq("71470")
    return instrument.where(~is_solo, instrument.map(code_to_name))


# ---------- build ----------
def make_file_id(frame: pd.DataFrame) -> pd.Series:
    """Stable PK: '<dataset>:<source path relative to the dataset dir>'.

    Derived from provenance rather than row order, so re-ingesting or re-cutting the
    split leaves ids unchanged and the shift pool's foreign keys stay valid.
    """
    relative = frame["src_path"].astype(str).map(
        lambda p: "/".join(Path(p).parts[2:]) if len(Path(p).parts) > 2 else Path(p).name)
    return frame["dataset"].astype(str) + ":" + relative.map(nfc)


def build(root: Path, taxonomy_path: Path) -> pd.DataFrame:
    """Join ingest manifest + QC scans + taxonomy into the source manifest."""
    ingest = pd.read_parquet(root / INGEST_MANIFEST)
    qc = pd.concat([pd.read_parquet(root / p) for p in QC_INGEST], ignore_index=True)

    # --- join on the ingested output path (1:1 by construction; verified below) ---
    qc_slim = qc[["path", *QC_CONTENT_COLUMNS]].rename(columns={"path": "out_path"})
    merged = ingest.merge(qc_slim, on="out_path", how="left", validate="one_to_one")
    missing = merged[QC_CONTENT_COLUMNS[0]].isna().sum()
    if missing:
        raise ValueError(f"{missing} ingest rows have no QC row — re-run scripts/audio_qc.py "
                         f"over the ingest store before building the source manifest")

    # --- taxonomy: canonical name, then stem group / pitched / publisher group ---
    taxonomy, code_to_name = load_taxonomy(taxonomy_path)
    merged["instrument_canonical"] = canonical_instrument(merged, code_to_name)
    merged = merged.merge(taxonomy.drop(columns=["code_71470"]),
                          on="instrument_canonical", how="left")

    # --- `pitched` is set for every taxonomy entry, so NA here == absent from the yaml.
    #     (stem_group is legitimately null for 71470-only instruments, so it can't be
    #     used as the presence check.) ---
    unmapped = merged[merged["instrument_canonical"].notna() & merged["pitched"].isna()]
    if len(unmapped):
        names = sorted(unmapped["instrument_canonical"].dropna().unique())
        raise ValueError(f"instruments missing from the taxonomy: {names}")

    # --- nullable boolean, NOT object: masters carry NA (no instrument), and an object
    #     column silently makes `~frame.pitched` truthy for BOTH values ---
    merged["pitched"] = merged["pitched"].astype("boolean")

    # --- stable primary key, then a readable column order ---
    merged.insert(0, "file_id", make_file_id(merged))
    if merged["file_id"].duplicated().any():
        raise ValueError("file_id is not unique — source paths collide")

    lead = ["file_id", "dataset", "role", "split", "song_id", "clip_id", "genre_sub",
            "instrument_canonical", "instrument_raw", "stem_group", "pitched",
            "instrument_group_71955"]
    return merged[[*lead, *[c for c in merged.columns if c not in lead]]]


# ---------- exclusions ----------
def apply_exclusions(frame: pd.DataFrame,
                     exclusions_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Drop publisher-defective files declared in an exclusions yaml.

    A rule targets one song of one dataset. With an `instruments` list it drops exactly
    those stems; without one it drops the whole song, master included. Every rule must
    match at least one row — a rule matching nothing means the song name or an instrument
    name is wrong, and silently excluding nothing would leave the defect in the training
    pool while the provenance record claims it was removed.

    Args:
        frame: the built source manifest, before exclusion.
        exclusions_path: path to configs/manifest_exclusions.yaml.

    Returns:
        (frame without the excluded rows, one audit record per rule).
    """
    spec = yaml.safe_load(exclusions_path.read_text(encoding="utf-8"))
    song_ids = frame["song_id"].map(nfc)
    instruments = frame["instrument_canonical"].map(nfc)

    drop_mask = pd.Series(False, index=frame.index)
    audit: list[dict] = []
    for rule in spec["exclusions"]:
        matches = frame["dataset"].eq(str(rule["dataset"])) & song_ids.eq(nfc(rule["song_id"]))
        wanted = rule.get("instruments")
        if wanted is not None:
            matches &= instruments.isin([nfc(name) for name in wanted])
        if not matches.any():
            raise ValueError(
                f"exclusion rule matched no rows: song_id={rule['song_id']!r} "
                f"instruments={wanted!r} — check the song name (NFC) against the manifest")
        if wanted is not None and len(set(instruments[matches])) != len(set(wanted)):
            found = sorted(set(instruments[matches]))
            raise ValueError(
                f"exclusion rule for {rule['song_id']!r} asked for {sorted(set(wanted))} "
                f"but only matched {found}")
        drop_mask |= matches
        audit.append({"song_id": rule["song_id"], "instruments": wanted,
                      "rows_dropped": int(matches.sum()),
                      "file_ids": sorted(frame.loc[matches, "file_id"])})

    return frame[~drop_mask].reset_index(drop=True), audit


def report(frame: pd.DataFrame) -> None:
    """Print a short sanity summary of the built manifest."""
    print(f"\nsource_manifest: {len(frame):,} rows × {len(frame.columns)} cols")
    print("\n-- rows by dataset/role --")
    print(frame.groupby(["dataset", "role"]).size().to_string())
    print("\n-- 71955 songs by split --")
    ensemble = frame[frame.dataset.eq("71955")]
    print(ensemble.groupby("split")["song_id"].nunique().to_string())
    print("\n-- working stem groups: 71955 stems vs 71470 clips --")
    sources = frame[frame.role.isin(["stem", "clip"])]
    pivot = (sources.pivot_table(index="stem_group", columns="dataset", values="file_id",
                                 aggfunc="size", fill_value=0)
             .assign(hours=sources.groupby("stem_group")["out_duration"].sum() / 3600)
             .round(1).sort_values("71955", ascending=False))
    print(pivot.to_string())
    print(f"\nmasters (no instrument): {frame.instrument_canonical.isna().sum():,}")

    # --- held-aside rows are a deliberate state, so surface them explicitly rather than
    #     letting them hide as silent nulls ---
    aside = frame[frame.instrument_canonical.notna() & frame.stem_group.isna()]
    print(f"HELD ASIDE (stem_group deliberately undecided): {len(aside):,} clips")
    if len(aside):
        detail = (aside.groupby("instrument_canonical")
                  .agg(clips=("file_id", "size"),
                       minutes=("out_duration", lambda s: s.sum() / 60))
                  .round(1).sort_values("clips", ascending=False))
        print(detail.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge ingest manifest + ingest-store QC + taxonomy.")
    ap.add_argument("--taxonomy", default="configs/stem_taxonomy.yaml")
    ap.add_argument("--exclusions", default=None,
                    help="exclusion rules yaml; omit for v1 (nothing dropped)")
    ap.add_argument("--out", default="manifests/source_manifest")
    ap.add_argument("--dry-run", action="store_true", help="build + report, write nothing")
    args = ap.parse_args()

    root = find_root()
    frame = build(root, root / args.taxonomy)

    if args.exclusions:
        rows_before = len(frame)
        frame, audit = apply_exclusions(frame, root / args.exclusions)
        print(f"\n-- exclusions ({args.exclusions}) --")
        for record in audit:
            scope = record["instruments"] or "ENTIRE SONG"
            print(f"  {record['song_id']}  {scope}  -> {record['rows_dropped']} rows")
        print(f"  rows {rows_before:,} -> {len(frame):,} "
              f"({rows_before - len(frame)} dropped)")

    report(frame)

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    out_parquet, out_csv = table_paths(root / args.out)
    frame.to_parquet(out_parquet, index=False)
    frame.to_csv(out_csv, index=False)
    print(f"\nwrote {out_parquet} + {out_csv}")


if __name__ == "__main__":
    main()
