"""verify_manifest_v2.py — prove source_manifest_v2 is v1 minus exactly the declared defects.

The exclusion path is the only thing standing between a known-bad publisher file and the
training pool, so it gets checked rather than trusted. Four assertions, each of which
would catch a different way of getting this wrong:

  1. EXACT REMOVALS   — the set of vanished file_ids equals the set the exclusion rules
                        say should vanish. Catches over- and under-exclusion alike.
  2. SURVIVORS UNTOUCHED — every remaining row is identical to its v1 counterpart on every
                        column. Catches a rebuild that silently changed something else
                        (taxonomy edit, re-ingest, QC re-scan) while we were looking at
                        the exclusions.
  3. KEEP-LIST INTACT — songs that must survive in full (0886, the kept twin of the looped
                        pair; 0905, the benign val song) have the same row count and the
                        same file_ids as in v1.
  4. POOL SANITY      — the nine modeled classes still have drawable train-split sources,
                        so no exclusion accidentally emptied a class.

Run:
    uv run python scripts/verify_manifest_v2.py
      --v1 manifests/parquet/source_manifest.parquet
      --v2 manifests/parquet/source_manifest_v2.parquet
      --exclusions configs/manifest_exclusions.yaml
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# the nine classes exp002 models (publisher's 9; pitched percussion quarantined)
MODELED_CLASSES = ["가야금", "거문고", "기타", "대금", "아쟁", "양금", "타악기", "피리", "해금"]

# songs that MUST survive whole — the reason each one is easy to delete by mistake
KEEP_WHOLE = {
    "0886_창작국악_창작국악": "the kept twin of the 0885 loop pair — excluding it too would "
                              "delete the music entirely instead of de-duplicating it",
    "0905_창작국악_창작국악": "benign master-as-stem (genuine solo 해금 piece), val split, "
                              "frozen — must be left completely alone",
}


def nfc(value: object) -> object:
    """NFC-normalize a Korean string so config names join against manifest names.

    Non-strings pass through untouched — song_id and instrument_canonical are null for
    71470 clips and for masters respectively.
    """
    return unicodedata.normalize("NFC", value) if isinstance(value, str) else value


def expected_removals(v1: pd.DataFrame, exclusions_path: Path) -> set[str]:
    """The file_ids the exclusion rules claim to remove, resolved against v1.

    Deliberately re-derived here from the rules rather than imported from the builder:
    a verification that reuses the code under test proves only self-consistency.

    Args:
        v1: the pre-exclusion manifest.
        exclusions_path: configs/manifest_exclusions.yaml.
    """
    spec = yaml.safe_load(exclusions_path.read_text(encoding="utf-8"))
    song_ids = v1["song_id"].map(lambda s: nfc(s) if isinstance(s, str) else s)
    instruments = v1["instrument_canonical"].map(lambda s: nfc(s) if isinstance(s, str) else s)

    expected: set[str] = set()
    for rule in spec["exclusions"]:
        matches = v1["dataset"].eq(str(rule["dataset"])) & song_ids.eq(nfc(rule["song_id"]))
        if rule.get("instruments") is not None:
            matches &= instruments.isin([nfc(name) for name in rule["instruments"]])
        expected |= set(v1.loc[matches, "file_id"])
    return expected


def check_exact_removals(v1: pd.DataFrame, v2: pd.DataFrame, expected: set[str]) -> list[str]:
    """Assertion 1: removed set == declared set, and nothing was added."""
    removed = set(v1["file_id"]) - set(v2["file_id"])
    added = set(v2["file_id"]) - set(v1["file_id"])
    failures = []
    if removed != expected:
        failures.append(f"removed set != declared set; "
                        f"unexpectedly removed={sorted(removed - expected)}, "
                        f"declared but still present={sorted(expected - removed)}")
    if added:
        failures.append(f"v2 contains file_ids absent from v1: {sorted(added)}")
    print(f"  [1] removals: {len(removed)} rows, matches declared set: {removed == expected}")
    print(f"      rows {len(v1):,} -> {len(v2):,}")
    for file_id in sorted(removed):
        print(f"        - {file_id}")
    return failures


def check_survivors_untouched(v1: pd.DataFrame, v2: pd.DataFrame) -> list[str]:
    """Assertion 2: every surviving row is column-for-column identical to v1."""
    left = v1[v1["file_id"].isin(set(v2["file_id"]))].sort_values("file_id").reset_index(drop=True)
    right = v2.sort_values("file_id").reset_index(drop=True)
    failures = []
    if list(left.columns) != list(right.columns):
        failures.append(f"column sets differ: v1={list(left.columns)} v2={list(right.columns)}")
        return failures
    differing = [column for column in left.columns if not left[column].equals(right[column])]
    if differing:
        failures.append(f"surviving rows differ from v1 in columns: {differing}")
    print(f"  [2] survivors identical to v1 on all {len(left.columns)} columns: {not differing}")
    return failures


def check_keep_list(v1: pd.DataFrame, v2: pd.DataFrame) -> list[str]:
    """Assertion 3: the must-survive songs kept every row they had in v1."""
    failures = []
    for song_id, why in KEEP_WHOLE.items():
        before = set(v1.loc[v1["song_id"].map(nfc).eq(nfc(song_id)), "file_id"])
        after = set(v2.loc[v2["song_id"].map(nfc).eq(nfc(song_id)), "file_id"])
        intact = bool(before) and before == after
        print(f"  [3] {song_id}: {len(before)} -> {len(after)} rows, intact: {intact}")
        if not intact:
            failures.append(f"{song_id} not intact ({why}); lost {sorted(before - after)}")
    return failures


def check_pool_sanity(v2: pd.DataFrame) -> list[str]:
    """Assertion 4: every modeled class still has train-split ensemble sources."""
    pool = v2[v2["dataset"].eq("71955") & v2["split"].eq("train") & v2["role"].ne("master")]
    counts = pool["stem_group"].value_counts()
    empty = [c for c in MODELED_CLASSES if counts.get(c, 0) == 0]
    print("  [4] train-split ensemble sources per modeled class: "
          + ", ".join(f"{c} {counts.get(c, 0)}" for c in MODELED_CLASSES))
    return [f"classes with no drawable train sources: {empty}"] if empty else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify source_manifest_v2 against v1.")
    ap.add_argument("--v1", default="manifests/parquet/source_manifest.parquet")
    ap.add_argument("--v2", default="manifests/parquet/source_manifest_v2.parquet")
    ap.add_argument("--exclusions", default="configs/manifest_exclusions.yaml")
    args = ap.parse_args()

    v1 = pd.read_parquet(REPO_ROOT / args.v1)
    v2 = pd.read_parquet(REPO_ROOT / args.v2)
    expected = expected_removals(v1, REPO_ROOT / args.exclusions)

    print(f"verifying {args.v2} against {args.v1}")
    failures = [*check_exact_removals(v1, v2, expected),
                *check_survivors_untouched(v1, v2),
                *check_keep_list(v1, v2),
                *check_pool_sanity(v2)]

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
