"""aggregate.py — turn raw eval rows into reported numbers, WITHOUT re-inference.

Reads the long-format rows parquet written by src/eval/runner.py and computes, per
(run_id, split, variant) and per metric:

  summary   both headline aggregations side by side —
              class_balanced_mean : mean of per-class means (the conservative number
                                    reported so far; every class weighs equally)
              pooled_mean         : mean over all (song, class) PRESENT pairs
  per_stem  per-class mean / median / song count
  per_genre per-genre pooled mean AND class-balanced mean (column names mirror
            experiments/analysis training_curves_per_genre)
  absent    per-class false-activation levels on ABSENT pairs (pred_energy_dbfs)

(An si_sdr >= usdr invariant check lived here during the 2026-08-30 build and was
dropped by decision — the ordering is not mathematically guaranteed; see
docs/eval_pipeline_build_report.md, G6 section.)

Run:
    uv run python -m src.eval.aggregate --rows <rows.parquet> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ("si_sdr", "usdr", "csdr")
GROUP_COLUMNS = ["run_id", "split", "variant"]


def active_metrics(rows: pd.DataFrame) -> list[str]:
    """Metric columns actually present (and not all-NaN) in the rows table."""
    return [c for c in METRIC_COLUMNS if c in rows.columns and rows[c].notna().any()]


def summarize(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Both headline aggregations per (run, split, variant, metric)."""
    present = rows[rows.status == "present"]
    out = []
    for keys, group in present.groupby(GROUP_COLUMNS):
        for metric in metrics:
            per_class = group.groupby("stem_class")[metric].mean()
            out.append({**dict(zip(GROUP_COLUMNS, keys)), "metric": metric,
                        "class_balanced_mean": float(per_class.mean()),
                        "pooled_mean": float(group[metric].mean()),
                        "n_songs": int(group.song_id.nunique()),
                        "n_pairs": int(len(group))})
    return pd.DataFrame(out)


def per_stem(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-class means/medians/counts over PRESENT pairs."""
    present = rows[rows.status == "present"]
    aggregations = {f"{m}_{stat}": (m, stat) for m in metrics
                    for stat in ("mean", "median")}
    table = (present.groupby(GROUP_COLUMNS + ["stem_class"])
             .agg(**aggregations, n_songs=("song_id", "nunique"))
             .reset_index())
    return table


def per_genre(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-genre pooled AND class-balanced means (never global-mean-only)."""
    present = rows[rows.status == "present"]
    out = []
    for keys, group in present.groupby(GROUP_COLUMNS + ["genre_sub"]):
        record = dict(zip(GROUP_COLUMNS + ["genre_sub"], keys))
        for metric in metrics:
            record[f"{metric}_pooled_mean"] = float(group[metric].mean())
            record[f"{metric}_class_balanced_mean"] = float(
                group.groupby("stem_class")[metric].mean().mean())
        record.update(num_songs=int(group.song_id.nunique()),
                      num_scores=int(len(group)))
        out.append(record)
    return pd.DataFrame(out)


def absent_levels(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-class false-activation energy on ABSENT pairs."""
    absent = rows[rows.status == "absent"]
    if absent.empty:
        return absent.iloc[0:0]
    return (absent.groupby(GROUP_COLUMNS + ["stem_class"])
            .agg(n_absent=("song_id", "nunique"),
                 pred_energy_dbfs_mean=("pred_energy_dbfs", "mean"),
                 pred_energy_dbfs_median=("pred_energy_dbfs", "median"),
                 pred_energy_dbfs_max=("pred_energy_dbfs", "max"))
            .reset_index())


def write_table(table: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Persist one aggregate as parquet + csv twin and echo it."""
    table.to_parquet(out_dir / f"{name}.parquet", index=False)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"\n== {name} ==")
    print(table.round(2).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw eval rows.")
    parser.add_argument("--rows", required=True,
                        help="rows parquet from src.eval.runner")
    parser.add_argument("--out-dir", default=None,
                        help="destination (default: alongside the rows file)")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    rows = pd.read_parquet(rows_path)
    metrics = active_metrics(rows)
    out_dir = Path(args.out_dir) if args.out_dir else rows_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rows_path.stem

    write_table(summarize(rows, metrics), out_dir, f"{stem}_summary")
    write_table(per_stem(rows, metrics), out_dir, f"{stem}_per_stem")
    write_table(per_genre(rows, metrics), out_dir, f"{stem}_per_genre")
    absent = absent_levels(rows)
    if len(absent):
        write_table(absent, out_dir, f"{stem}_absent")
    else:
        print("\nno absent pairs — absent-levels table skipped")


if __name__ == "__main__":
    main()
