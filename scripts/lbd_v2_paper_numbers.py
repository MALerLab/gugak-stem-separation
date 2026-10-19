"""ISMIR 2026 LBD camera-ready, paper draft v2_bsr_ep35 — ledger of every number the draft quotes.

Reads the raw rows parquets written by src/eval/runner.py (one row per (song, stem
class), status present/absent) for the four Table 1 runs × two input variants, and
re-derives each quoted number with one aggregation code path:

  class-balanced mean = mean over the 9 stem classes of (mean over PRESENT songs of
  that class) — the same rule as scripts/lbd_numbers_and_figure1.py.
  per-genre pooled mean = mean over all PRESENT (song, class) pairs of the genre —
  the rule behind docs/master_eval_lag_report.md's per-genre drop table.

Every class-balanced mean is cross-checked against the run's own `_summary.parquet`
(written by src/eval/aggregate.py at eval time); any disagreement aborts.

Deliverables (into --out-dir, e.g. paper/ismir2026_lbd_camera_ready/paper/numbers/v2_bsr_ep35):
  paper_numbers_v2.csv   one row per quoted number: id, where it appears, formula,
                         full-precision value, the string as printed (1 decimal for
                         per-genre drops, 2 otherwise)
  paper_numbers_v2.md    the same ledger, plus the inputs table (run → rows parquet,
                         checkpoint, eval git commit) and the per-genre drop table

No inference, no GPU. Run:
    uv run python scripts/lbd_v2_paper_numbers.py --out-dir <numbers dir>
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT = "test"
STEM_ORDER = ("타악기", "피리", "대금", "해금", "아쟁", "가야금", "거문고", "기타", "양금")
# the paper's English stem names (Data section of the draft)
STEM_ENGLISH = {"타악기": "percussion", "피리": "Piri", "대금": "Daegeum", "해금": "Haegeum",
                "아쟁": "Ajaeng", "가야금": "Gayageum", "거문고": "Geomungo", "기타": "other",
                "양금": "Yanggeum"}
GENRE_ENGLISH = {"민요": "Minyo", "산조": "Sanjo", "판소리": "Pansori", "궁중음악": "court music",
                 "풍류음악": "Pungnyu", "창작국악": "contemporary Gugak", "대풍류": "Daepungnyu"}


@dataclass(frozen=True)
class Run:
    """One Table 1 row: a training run's eval rows parquet stem, per input variant."""
    key: str           # exp id
    label: str         # model + mixing, as Table 1 names it
    eval_stem: str     # repo-relative path prefix; + `_test_sumstem` / `_test_master_aligned`
    sumstem_suffix: str = "_test_sumstem"

    def rows_parquet(self, variant: str) -> Path:
        """Rows parquet of this run for one input variant (sumstem | master)."""
        suffix = self.sumstem_suffix if variant == "sumstem" else "_test_master_aligned"
        return REPO_ROOT / f"{self.eval_stem}{suffix}.parquet"


RUNS: dict[str, Run] = {run.key: run for run in (
    Run("exp004.1", "HTDemucs, incoherent",
        "experiments/exp004.1_260817_incoherent_n2/eval/eval_exp004.1_ep57",
        sumstem_suffix="_val_test_sumstem"),
    Run("exp004", "HTDemucs, partially coherent",
        "experiments/exp004_260815_coherent_p1_uniform/eval/eval_exp004_ep58"),
    Run("exp008", "BS-RoFormer, incoherent",
        "experiments/exp008_261011_bsroformer_incoherent_n2/eval/eval_exp008_ep34"),
    Run("exp006", "BS-RoFormer, partially coherent",
        "experiments/exp006_260925_bsroformer_partial_coherent/eval/eval_exp006_ep34"),
)}
VARIANTS = ("sumstem", "master")
# per-genre drop: averaged over the partially coherent run of each model, as v0 did
# with exp004 + exp003.0 (docs/master_eval_lag_report.md)
GENRE_DROP_RUNS = ("exp004", "exp006")


# ----------------------------------------------------------------------------- loading
def load_present_rows(run: Run, variant: str) -> pd.DataFrame:
    """PRESENT test rows of one run and variant."""
    rows = pd.read_parquet(run.rows_parquet(variant))
    rows = rows[(rows.split == SPLIT) & (rows.variant == variant)]
    if rows.empty:
        raise ValueError(f"{run.key}: no {SPLIT}/{variant} rows in {run.rows_parquet(variant)}")
    return rows[rows.status == "present"]


def constant_value(rows: pd.DataFrame, column: str) -> str:
    """The single value of a column that must be constant within one eval run."""
    values = rows[column].unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within a run: {values}")
    return str(values[0])


# ------------------------------------------------------------------------- aggregation
def class_means(present: pd.DataFrame, metric: str = "si_sdr") -> pd.Series:
    """Per-stem-class mean over present songs, in taxonomy order."""
    return present.groupby("stem_class")[metric].mean().reindex(STEM_ORDER)


def class_balanced_mean(present: pd.DataFrame, metric: str = "si_sdr") -> float:
    """Mean of the 9 class means — the headline aggregation."""
    return float(class_means(present, metric).mean())


def genre_pooled_means(present: pd.DataFrame) -> pd.Series:
    """SI-SDR mean over all present (song, class) pairs, per genre."""
    return present.groupby("genre_sub")["si_sdr"].mean()


def cross_check_against_run_summary(run: Run, variant: str, present: pd.DataFrame) -> None:
    """Abort unless our class-balanced means equal the run's own eval-time summary."""
    summary_path = run.rows_parquet(variant).with_name(
        run.rows_parquet(variant).stem + "_summary.parquet")
    summary = pd.read_parquet(summary_path)
    summary = summary[(summary.split == SPLIT) & (summary.variant == variant)]
    for metric in ("si_sdr", "usdr"):
        theirs = float(summary.loc[summary.metric == metric, "class_balanced_mean"].iloc[0])
        ours = class_balanced_mean(present, metric)
        if abs(ours - theirs) > 1e-9:
            raise AssertionError(f"{run.key} {variant} {metric}: ours {ours} != summary {theirs}")


# ------------------------------------------------------------------------------ ledger
def ledger_row(number_id: str, where: str, formula: str, value: float | str,
               decimals: int = 2) -> dict:
    """One quoted number: id, location in the draft, formula, value, printed string."""
    printed = value if isinstance(value, str) else f"{value:.{decimals}f}"
    return {"id": number_id, "where": where, "formula": formula,
            "value": value if isinstance(value, str) else repr(value), "printed": printed}


def build_ledger(present: dict[tuple[str, str], pd.DataFrame]) -> tuple[list[dict], pd.DataFrame]:
    """Every number draft v2 quotes for the BS-RoFormer runs and master gap."""
    means = {(key, variant): class_balanced_mean(rows) for (key, variant), rows in present.items()}
    usdr = {(key, variant): class_balanced_mean(rows, "usdr") for (key, variant), rows in present.items()}
    ledger = []

    # Table 1 cells for the two BS-RoFormer rows
    for key in ("exp008", "exp006"):
        for variant in VARIANTS:
            ledger.append(ledger_row(
                f"table1.{key}.{variant}", f"Table 1, {RUNS[key].label} row, "
                f"{'Stem sum' if variant == 'sumstem' else 'Master'} column",
                f"class-balanced SI-SDR, {key} ep34, {variant}", means[(key, variant)]))

    # coherence gain for BS-RoFormer, on each input
    for variant in VARIANTS:
        ledger.append(ledger_row(
            f"coherence_gain.bsroformer.{variant}",
            "abstract + Results ¶1 (sumstem); Results master ¶ (both)",
            f"exp006 − exp008, class-balanced SI-SDR, {variant}",
            means[("exp006", variant)] - means[("exp008", variant)]))
    # HTDemucs coherence gain, recomputed only to show the BS-RoFormer sentence's neighbours
    for variant in VARIANTS:
        ledger.append(ledger_row(
            f"coherence_gain.htdemucs.{variant}", "reference (already printed in v0)",
            f"exp004 − exp004.1, class-balanced SI-SDR, {variant}",
            means[("exp004", variant)] - means[("exp004.1", variant)]))

    # exp006 headline, margin over the best HTDemucs run, per-stem extremes
    best_htdemucs = max(("exp004.1", "exp004"), key=lambda key: means[(key, "sumstem")])
    ledger.append(ledger_row("exp006.margin_over_best_htdemucs", "Results ¶3",
                             f"exp006 − {best_htdemucs} (best HTDemucs), sumstem",
                             means[("exp006", "sumstem")] - means[(best_htdemucs, "sumstem")]))
    exp006_classes = class_means(present[("exp006", "sumstem")])
    for end, stem_class in (("min", exp006_classes.idxmin()), ("max", exp006_classes.idxmax())):
        ledger.append(ledger_row(f"exp006.per_stem_{end}", "Results ¶3",
                                 f"{end} over 9 class means, exp006 ep34 sumstem",
                                 float(exp006_classes[stem_class])))
        ledger.append(ledger_row(f"exp006.per_stem_{end}_stem", "Results ¶3",
                                 f"argument of the {end}", STEM_ENGLISH[stem_class]))

    # master drop range over all four Table 1 runs
    drops = {key: means[(key, "sumstem")] - means[(key, "master")] for key in RUNS}
    for end, pick in (("min", min), ("max", max)):
        key = pick(drops, key=drops.get)
        ledger.append(ledger_row(f"master_drop.{end}", "Results master ¶ + Discussion ¶1",
                                 f"{end} over Table 1 runs of (sumstem − master); here {key}",
                                 drops[key]))

    # per-genre pooled drop, averaged over the partially coherent run of each model
    genre_drop = pd.DataFrame({
        key: genre_pooled_means(present[(key, "sumstem")]) - genre_pooled_means(present[(key, "master")])
        for key in RUNS}).loc[:, list(RUNS)]
    genre_drop["mean_" + "_".join(GENRE_DROP_RUNS)] = genre_drop[list(GENRE_DROP_RUNS)].mean(axis=1)
    genre_drop = genre_drop.sort_values("mean_" + "_".join(GENRE_DROP_RUNS))
    average_column = "mean_" + "_".join(GENRE_DROP_RUNS)
    for genre in ("민요", "산조"):
        ledger.append(ledger_row(
            f"genre_drop.{GENRE_ENGLISH[genre]}", "Results master ¶",
            f"mean over {' and '.join(GENRE_DROP_RUNS)} of pooled (sumstem − master), {genre}",
            float(genre_drop.loc[genre, average_column]), decimals=1))
    for key in RUNS:
        order = " < ".join(genre_drop[key].sort_values().index)
        ledger.append(ledger_row(f"genre_drop.order.{key}", "Results master ¶ (ordering claim)",
                                 f"genres by ascending pooled drop, {key}", order))
    for key in ("exp008", "exp006"):
        ledger.append(ledger_row(f"usdr.{key}.sumstem", "not printed (numbers/summary twin)",
                                 f"class-balanced uSDR, {key} ep34, sumstem", usdr[(key, "sumstem")]))
    return ledger, genre_drop


def inputs_table(present: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Which rows parquet, checkpoint and eval commit each number came from."""
    return pd.DataFrame([{
        "run": key, "variant": variant,
        "rows_parquet": str(RUNS[key].rows_parquet(variant).relative_to(REPO_ROOT)),
        "checkpoint": Path(constant_value(rows, "checkpoint")).name,
        "eval_git_commit": constant_value(rows, "git_commit")[:12],
        "n_present_pairs": len(rows),
    } for (key, variant), rows in present.items()])


def markdown_table(table: pd.DataFrame, decimals: int = 2) -> str:
    """Pipe-table rendering with fixed decimals (pandas' to_markdown needs tabulate)."""
    frame = table.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda v: f"{v:.{decimals}f}")
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_outputs(out_dir: Path, ledger: list[dict], genre_drop: pd.DataFrame,
                  inputs: pd.DataFrame) -> None:
    """Ledger CSV + a markdown twin with inputs and the per-genre table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_frame = pd.DataFrame(ledger)
    ledger_frame.to_csv(out_dir / "paper_numbers_v2.csv", index=False)
    text = (
        "# Paper draft v2_bsr_ep35 — quoted numbers ledger\n\n"
        "Generated by `scripts/lbd_v2_paper_numbers.py` (main repo) from the eval rows parquets "
        "below; every class-balanced mean is asserted equal to its run's eval-time summary. "
        "Test split, 135 songs, SI-SDR in dB unless named. `master` = publisher master aligned "
        "to its stems via `manifests/parquet/master_lag.parquet`.\n\n"
        "## Ledger\n\n" + markdown_table(ledger_frame.drop(columns="value")) +
        "\n\n## Inputs\n\n" + markdown_table(inputs) +
        "\n\n## Per-genre drop, sumstem − master (dB), pooled over present (song, class) pairs\n\n" +
        markdown_table(genre_drop.reset_index()) + "\n")
    (out_dir / "paper_numbers_v2.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="numbers folder of the paper draft (paper repo)")
    args = parser.parse_args()

    # load + cross-check every (run, variant)
    present = {}
    for key, run in RUNS.items():
        for variant in VARIANTS:
            rows = load_present_rows(run, variant)
            cross_check_against_run_summary(run, variant, rows)
            present[(key, variant)] = rows

    ledger, genre_drop = build_ledger(present)
    write_outputs(args.out_dir, ledger, genre_drop, inputs_table(present))
    print(pd.DataFrame(ledger)[["id", "printed", "formula"]].to_string(index=False))


if __name__ == "__main__":
    main()
