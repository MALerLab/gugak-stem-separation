"""aug_listening_tally.py — stage 5 of the augmentation listening test.

Unblinds the A/B answers and counts, per stem and item type (master / YouTube), how often
the augmented model (exp007) or the baseline (exp006) was preferred, next to the master
eval's per-stem SDR difference (docs/aug_listening_test_spec.md, stage 5).

Input: one JSON per listener in <page.out_dir>/../answers/ — the `answers/<id>` documents
of the artifact's db, saved there with ArtifactData (or a listener's "Copy answers" text).
Blind answers (no reveal before answering) are counted separately from all answers.

Run:
    uv run python scripts/aug_listening_tally.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/aug_listening_test.yaml"


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a <dir>/<name> basename to its (parquet, csv) twin paths, dirs created."""
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


def unblinded_answers(config: dict, page_dir: Path) -> pd.DataFrame:
    """One row per (listener, item, stem) answer, with the choice mapped to a model role.

    Args:
        config: the parsed aug_listening_test config.
        page_dir: the built page folder (holds rows.json).
    """
    items = json.loads((page_dir / "rows.json").read_text())["items"]
    rows_by_key = {(item["clip"], row["stem"]): (item, row)
                   for item in items for row in item["rows"]}
    run_to_role = {model["run"]: role for role, model in config["models"].items()}

    records = []
    for path in sorted((page_dir.parent / "answers").glob("*.json")):
        answers = json.loads(path.read_text())["answers"]
        for key, answer in answers.items():
            if not answer.get("choice"):
                continue
            clip, stem = key.split("|")
            item, row = rows_by_key[(clip, stem)]
            preferred = {"A": run_to_role[row["a_model"]], "B": run_to_role[row["b_model"]],
                         "same": "no_difference"}[answer["choice"]]
            records.append({"listener": path.stem, "clip": clip, "item_type": item["item_type"],
                            "stem": stem, "preferred": preferred,
                            "blind": not answer.get("revealed_before_answer", False),
                            "tags": ",".join(answer.get("tags", [])),
                            "note": answer.get("note", "")})
    return pd.DataFrame(records)


def preference_counts(answers: pd.DataFrame) -> pd.DataFrame:
    """Per (item_type, stem): counts of augmented / baseline / no_difference, all + blind."""
    tables = []
    for label, subset in (("all", answers), ("blind", answers[answers.blind])):
        counts = (subset.groupby(["item_type", "stem", "preferred"]).size()
                  .unstack(fill_value=0))
        counts.columns = [f"{label}_{column}" for column in counts.columns]
        tables.append(counts)
    return pd.concat(tables, axis=1).fillna(0).astype(int).reset_index()


def master_stem_differences(config: dict) -> pd.DataFrame:
    """Mean per-stem master-eval difference (augmented − baseline) over all test songs."""
    metric = config["pick"]["metric"]
    frames = []
    for role in ("baseline", "augmented"):
        model = config["models"][role]
        rows = pd.read_parquet(REPO_ROOT / model["eval_dir"] / f"{model['eval_run_id']}.parquet")
        rows = rows[rows.status == "present"].set_index(["song_id", "stem_class"])[metric]
        frames.append(rows.rename(role))
    merged = pd.concat(frames, axis=1, join="inner")
    merged["d"] = merged.augmented - merged.baseline
    return (merged.groupby("stem_class").d.agg(["mean", "median"])
            .rename(columns={"mean": "sdr_d_mean", "median": "sdr_d_median"})
            .reset_index().rename(columns={"stem_class": "stem"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    page_dir = REPO_ROOT / config["page"]["out_dir"]

    answers = unblinded_answers(config, page_dir)
    if answers.empty:
        raise SystemExit(f"no answers under {page_dir.parent / 'answers'}")
    tally = preference_counts(answers).merge(master_stem_differences(config), on="stem",
                                             how="left")

    # tracked metrics live with the augmented run's folder
    out_base = (REPO_ROOT / config["models"]["augmented"]["eval_dir"]).parent / "listening"
    for name, table in (("aug_listening_tally", tally), ("aug_listening_answers", answers)):
        parquet_path, csv_path = table_paths(out_base / name)
        table.to_parquet(parquet_path, index=False)
        table.to_csv(csv_path, index=False)
    print(tally.to_string(index=False))
    print(f"\n{len(answers)} answers from {answers.listener.nunique()} listener(s) -> {out_base}")


if __name__ == "__main__":
    main()
