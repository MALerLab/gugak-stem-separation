"""ISMIR 2026 LBD — reported numbers + Figure 1, aggregated from existing eval rows.

Reads the raw long-format rows parquets written by src/eval/runner.py (one row per
(song, stem class), status present/absent) for the five reported arms and re-derives
every number the paper quotes with ONE aggregation code path:

  class-balanced mean = mean over the 9 stem classes of (mean over PRESENT songs of
  that class). Absent (song, class) pairs are gated out upstream by the silence
  tolerance (configs/silence.yaml) and never enter any mean.

Deliverables (paper/ismir2026_lbd/numbers/ + figures/):
  summary.csv/.md    5 arms × {SI-SDR, uSDR}, test split
  per_stem.csv/.md   4 Σstem arms × 9 stem classes (+ mean), one table per metric
  deltas.csv         per stem: density_floor = exp004.1 − exp002.4,
                               coherence     = exp004   − exp004.1   (SI-SDR)
  per_genre.csv      4 Σstem arms × 7 genres, SI-SDR
  provenance.md      checkpoints, config hashes, tolerance, counts — reviewer answers
  fig1_per_stem_delta.pdf/.png   grouped delta bars, Korean stem labels, NanumGothic
                                 embedded (TrueType, pdf.fonttype 42)

The re-aggregation is cross-checked against each run's own `<run>_summary.parquet`
(written by src/eval/aggregate.py at eval time) and aborts on any disagreement.

No inference, no GPU. Run:
    uv run python scripts/lbd_numbers_and_figure1.py
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import koreanize_matplotlib  # noqa: F401 — registers NanumGothic with matplotlib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.plot_training_curves import load_palette  # noqa: E402

PAPER_DIR = REPO_ROOT / "paper/ismir2026_lbd"
NUMBERS_DIR = PAPER_DIR / "numbers"
FIGURES_DIR = PAPER_DIR / "figures"
PALETTE_PATH = REPO_ROOT / "configs/palette_dancheong.yaml"
SILENCE_PATH = REPO_ROOT / "configs/silence.yaml"

METRICS = ("si_sdr", "usdr")
METRIC_LABELS = {"si_sdr": "SI-SDR", "usdr": "uSDR"}
# taxonomy display order (matches configs/palette_dancheong.yaml stem_class_colors)
STEM_ORDER = ("타악기", "피리", "대금", "해금", "아쟁", "가야금", "거문고", "기타", "양금")
SPLIT = "test"


@dataclass(frozen=True)
class Arm:
    """One reported row: an eval rows parquet plus its eval-job config."""
    key: str                 # exp id used in every output table
    label: str               # human description for .md tables
    variant: str             # sumstem | master
    rows_parquet: str        # repo-relative path to the runner's rows parquet
    eval_config: str         # repo-relative path to the configs/eval/*.yaml it ran from

    @property
    def row_name(self) -> str:
        """Table row id: exp key, plus the variant when it is not the Σstem default."""
        return self.key if self.variant == "sumstem" else f"{self.key} ({self.variant})"


ARMS: tuple[Arm, ...] = (
    Arm("exp002.4", "HTDemucs, incoherent, n≥1", "sumstem",
        "experiments/exp002.4_260813_seed45/eval/eval_exp002.4_ep59_val_test_sumstem.parquet",
        "configs/eval/exp002.4_val_test_sumstem.yaml"),
    Arm("exp004.1", "HTDemucs, incoherent, n≥2", "sumstem",
        "experiments/exp004.1_260817_incoherent_n2/eval/eval_exp004.1_ep57_val_test_sumstem.parquet",
        "configs/eval/exp004.1_val_test_sumstem.yaml"),
    Arm("exp004", "HTDemucs, partially coherent p=1.0, n≥2", "sumstem",
        "experiments/exp004_260815_coherent_p1_uniform/eval/eval_exp004_ep58_test_sumstem.parquet",
        "configs/eval/exp004_test_sumstem.yaml"),
    Arm("exp003.0", "BS-RoFormer, incoherent, n≥1", "sumstem",
        "experiments/exp003.0_260809_bsroformer_pilot/eval/eval_exp003.0_ep59_test_sumstem.parquet",
        "configs/eval/exp003.0_test_sumstem.yaml"),
    Arm("exp003.0", "BS-RoFormer, incoherent, n≥1 — publisher master as mixture", "master",
        "experiments/exp003.0_260809_bsroformer_pilot/eval/eval_exp003.0_ep59_test_master.parquet",
        "configs/eval/exp003.0_test_master.yaml"),
)
SUMSTEM_ARMS = tuple(arm for arm in ARMS if arm.variant == "sumstem")
# the 2×2 spine: which arm pairs define the two reported deltas
DELTA_PAIRS = {"density_floor": ("exp002.4", "exp004.1"), "coherence": ("exp004.1", "exp004")}


# ----------------------------------------------------------------------------- loading
def load_present_rows(arm: Arm) -> pd.DataFrame:
    """PRESENT rows of one arm on the test split, with the arm's row_name attached."""
    rows = pd.read_parquet(REPO_ROOT / arm.rows_parquet)
    rows = rows[(rows.split == SPLIT) & (rows.variant == arm.variant)]
    if rows.empty:
        raise ValueError(f"{arm.key}: no {SPLIT}/{arm.variant} rows in {arm.rows_parquet}")
    return rows[rows.status == "present"].assign(arm=arm.row_name)


def constant_column(rows: pd.DataFrame, column: str) -> str:
    """The single value of a column that must be constant within one eval run."""
    values = rows[column].unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within a run: {values}")
    return str(values[0])


def config_hash_of(config_path: Path) -> str:
    """Runner's config fingerprint: sha256 of the eval config file, first 12 hex."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


def checkpoint_epoch(checkpoint_path: str) -> int:
    """Epoch parsed from an MSST checkpoint name `model_<type>_ep_<N>_si_sdr_<v>.ckpt`."""
    name = Path(checkpoint_path).name
    return int(name.split("_ep_")[1].split("_")[0])


# ------------------------------------------------------------------------- aggregation
def class_means(present: pd.DataFrame, metric: str) -> pd.Series:
    """Per-stem-class mean over present songs, in taxonomy order."""
    return present.groupby("stem_class")[metric].mean().reindex(STEM_ORDER)


def class_balanced_mean(present: pd.DataFrame, metric: str) -> float:
    """Mean of the 9 class means — the headline aggregation."""
    return float(class_means(present, metric).mean())


def cross_check_against_run_summary(arm: Arm, present: pd.DataFrame) -> None:
    """Abort if our re-aggregation disagrees with the run's own summary parquet."""
    summary_path = REPO_ROOT / arm.rows_parquet.replace(".parquet", "_summary.parquet")
    summary = pd.read_parquet(summary_path)
    summary = summary[(summary.split == SPLIT) & (summary.variant == arm.variant)]
    for metric in METRICS:
        theirs = float(summary.loc[summary.metric == metric, "class_balanced_mean"].iloc[0])
        ours = class_balanced_mean(present, metric)
        if abs(theirs - ours) > 1e-9:
            raise AssertionError(f"{arm.row_name} {metric}: ours {ours} != summary {theirs}")


def build_summary(present_by_arm: dict[Arm, pd.DataFrame]) -> pd.DataFrame:
    """One row per arm: both metrics' class-balanced means + counts + checkpoint."""
    records = []
    for arm, present in present_by_arm.items():
        records.append({
            "arm": arm.row_name, "model_and_sampling": arm.label, "variant": arm.variant,
            **{metric: class_balanced_mean(present, metric) for metric in METRICS},
            "epoch": checkpoint_epoch(constant_column(present, "checkpoint")),
            "n_songs": int(present.song_id.nunique()), "n_present_pairs": int(len(present)),
        })
    return pd.DataFrame(records)


def build_per_stem(present_by_arm: dict[Arm, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Rows = Σstem arms, columns = 9 stem classes + mean, for one metric."""
    table = pd.DataFrame({arm.row_name: class_means(present, metric)
                          for arm, present in present_by_arm.items()
                          if arm.variant == "sumstem"}).T
    table["mean"] = table[list(STEM_ORDER)].mean(axis=1)
    table.index.name = "arm"
    return table


def present_counts_per_class(present_by_arm: dict[Arm, pd.DataFrame]) -> pd.Series:
    """n present songs per class — identical across Σstem arms (same references)."""
    counts = {arm.row_name: present.groupby("stem_class").size().reindex(STEM_ORDER)
              for arm, present in present_by_arm.items() if arm.variant == "sumstem"}
    frame = pd.DataFrame(counts)
    if not frame.nunique(axis=1).eq(1).all():
        raise AssertionError(f"present counts differ across Σstem arms:\n{frame}")
    return frame.iloc[:, 0].rename("n_songs")


def build_deltas(per_stem_si_sdr: pd.DataFrame) -> pd.DataFrame:
    """Per stem (+ mean): the three SI-SDR levels and the two 2×2 deltas."""
    columns = list(STEM_ORDER) + ["mean"]
    levels = per_stem_si_sdr.loc[["exp002.4", "exp004.1", "exp004"], columns].T
    levels.columns = [f"si_sdr_{key}" for key in levels.columns]
    for name, (before, after) in DELTA_PAIRS.items():
        levels[name] = levels[f"si_sdr_{after}"] - levels[f"si_sdr_{before}"]
    levels.index.name = "stem_class"
    return levels.reset_index()


def build_per_genre(present_by_arm: dict[Arm, pd.DataFrame]) -> pd.DataFrame:
    """Σstem arms × genre: SI-SDR pooled over (song, class) pairs, plus class-balanced."""
    records = []
    for arm, present in present_by_arm.items():
        if arm.variant != "sumstem":
            continue
        for genre, group in present.groupby("genre_sub"):
            records.append({
                "arm": arm.row_name, "genre_sub": genre,
                "si_sdr_pooled_mean": float(group.si_sdr.mean()),
                "si_sdr_class_balanced_mean": float(group.groupby("stem_class").si_sdr.mean().mean()),
                "n_songs": int(group.song_id.nunique()), "n_present_pairs": int(len(group)),
                "n_classes_present": int(group.stem_class.nunique()),
            })
    return pd.DataFrame(records)


# ------------------------------------------------------------------------------ writing
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


def write_summary(summary: pd.DataFrame) -> None:
    """summary.csv (all columns) + summary.md (the two headline metrics)."""
    summary.to_csv(NUMBERS_DIR / "summary.csv", index=False)
    display = summary[["arm", "model_and_sampling", "epoch", "si_sdr", "usdr"]].rename(
        columns={"model_and_sampling": "model / sampling", **METRIC_LABELS})
    text = ("# Summary — test split, mean of class means (dB)\n\n"
            "Σstem variant unless marked `(master)`. Absent stems gated out "
            "(reference RMS < −80 dBFS). 135 test songs, 9 stem classes.\n\n"
            + markdown_table(display) + "\n")
    (NUMBERS_DIR / "summary.md").write_text(text)


def write_per_stem(per_stem: dict[str, pd.DataFrame], counts: pd.Series) -> None:
    """per_stem.csv (long over metric) + per_stem.md (one table per metric + n row)."""
    long = pd.concat([table.assign(metric=metric) for metric, table in per_stem.items()])
    long = long.reset_index()[["metric", "arm", *STEM_ORDER, "mean"]]
    long.to_csv(NUMBERS_DIR / "per_stem.csv", index=False)
    sections = ["# Per-stem class means — test split, Σstem variant (dB)\n"]
    counts_row = pd.DataFrame([["n (test songs with stem present)", *counts.tolist(), ""]],
                              columns=["arm", *STEM_ORDER, "mean"])
    sections.append(markdown_table(counts_row) + "\n")
    for metric, table in per_stem.items():
        sections.append(f"## {METRIC_LABELS[metric]}\n\n" + markdown_table(table.reset_index()) + "\n")
    (NUMBERS_DIR / "per_stem.md").write_text("\n".join(sections))


def write_provenance(present_by_arm: dict[Arm, pd.DataFrame], counts: pd.Series,
                     genre_counts: pd.Series, gating_table: pd.DataFrame) -> None:
    """provenance.md — every fact a reviewer could ask about the reported numbers."""
    silence = yaml.safe_load(SILENCE_PATH.read_text())["silence"]["absent_rms_dbfs"]
    arm_rows = []
    for arm, present in present_by_arm.items():
        config_path = REPO_ROOT / arm.eval_config
        recorded_hash = constant_column(present, "config_hash")
        current_hash = config_hash_of(config_path)
        hash_note = "matches file on disk" if recorded_hash == current_hash else \
            f"**MISMATCH — file now hashes {current_hash}**"
        arm_rows.append({
            "arm": arm.row_name, "run_id": constant_column(present, "run_id"),
            "checkpoint": constant_column(present, "checkpoint"),
            "epoch": checkpoint_epoch(constant_column(present, "checkpoint")),
            "eval config": arm.eval_config, "config_hash": f"{recorded_hash} ({hash_note})",
            "eval git commit": constant_column(present, "git_commit")[:12],
            "eval timestamp (UTC)": constant_column(present, "timestamp"),
            "n songs": int(present.song_id.nunique()), "n present pairs": int(len(present)),
        })
    arms_table = pd.DataFrame(arm_rows)
    counts_table = counts.reset_index().rename(columns={"index": "stem_class"})
    genre_table = genre_counts.reset_index()
    genre_table.columns = ["genre_sub", "n_test_songs"]
    text = f"""# Provenance — ISMIR 2026 LBD reported numbers

Generated by `scripts/lbd_numbers_and_figure1.py` from the eval rows parquets listed
below. Nothing here re-runs inference; the script re-aggregates the raw per-(song,
class) rows and asserts equality with each run's own `_summary.parquet`.

## Evaluation protocol (as run)

- **Runner:** `src/eval/runner.py` — MSST chunked inference (`demix`, fp32 per the
  experiment's model config), one pass per song, scores kept, audio discarded (except
  where the eval config persisted preds for listening).
- **Split:** `test` = 135 songs from the frozen `manifests/parquet/eval_manifest.parquet`
  (song-level split, stratified by genre_sub; the val split has 91 songs and is used for
  model selection only). No test song is used for training or checkpoint choice.
- **Variants.** `sumstem`: our Σstem mixture (stems trimmed to the shortest per song,
  summed, peak-normalised), references = the same processed stems, so mixture ≡ Σ
  targets. `master`: the publisher's mastered mix as input, references summed on the
  fly from ingested stems at native scale — carries the irreducible mastering residual,
  reported as the real-world reference only.
- **Metrics.** `si_sdr`: scale-invariant SDR (Le Roux et al. 2019), copied verbatim from
  MSST's `utils/metrics.py` (eps 1e-8, whole song, channels pooled). `usdr`: global SDR
  as in MDX'21 / SDX'23, `10·log10((Σs² + eps)/(Σ(s−ŝ)² + eps))`, eps 1e-7, whole
  song, channels pooled, not scale-invariant. Both clamped at +100 dB (`metric_cap_db`).
  Neither is chunked museval BSSEval; no cSDR is reported.
- **Common-length trim:** reference and estimate are trimmed to their common length
  before scoring (판소리 stem tails / master length mismatches).
- **Silence tolerance / absent gating:** `configs/silence.yaml` →
  `silence.absent_rms_dbfs = {silence}` dBFS. A (song, class) pair is ABSENT when no
  reference file exists or the reference's full-song RMS (channels pooled) falls below
  this; absent pairs get no SDR (only the model's false-activation energy is logged)
  and never enter any mean. Background: post-ingest silence is a ±1-LSB DC-removal
  residue (~−138 dBFS peak), so the −80 dBFS gate sits ~58 dB above it.
- **Aggregation rule (every reported mean):** *mean of class means* — for each of the 9
  stem classes take the mean over test songs in which that class is PRESENT, then the
  unweighted mean over the 9 classes. Every class weighs equally regardless of how many
  songs contain it (양금: 13 songs; 타악기: 128). Pooled means over all present pairs are
  in `summary.csv` twins (`src/eval/aggregate.py`) but are not the reported number.
- **Deltas** (`deltas.csv`, SI-SDR): `density_floor = exp004.1 − exp002.4` (adding the
  n≥2 floor to incoherent sampling) and `coherence = exp004 − exp004.1` (switching to
  partially coherent sampling at the same floor). Mean deltas are differences of the
  class-balanced means, which equal the mean of the per-class deltas.
- **Per genre** (`per_genre.csv`): `si_sdr_pooled_mean` = mean over all present
  (song, class) pairs of the genre (the reported column — several genre × class cells
  hold 1–3 songs, so a class-balanced mean within genre is noisy; it is included as
  `si_sdr_class_balanced_mean` for completeness).
- **Eval seed:** 42 for every run (inference is deterministic; seed recorded for
  completeness). Eval code at git commit `{constant_column(next(iter(present_by_arm.values())), "git_commit")}`.

## Arms → checkpoints

Checkpoint epochs are those the eval configs pointed at (the run's best-val-SI-SDR
checkpoint under MSST's `model_<type>_ep_<N>_si_sdr_<val>.ckpt` naming). Each arm's
`config_hash` in the rows was re-derived from the config file now on disk.

{markdown_table(arms_table)}

Training-side facts for the four HTDemucs/BS-RoFormer arms (sampling regime, seeds,
warm-start checkpoints) are in each experiment folder's `LAUNCH_REPORT.md` and its
`configs/exp*.yaml`; the eval config's `model_config` field names the one used.

## Item counts

Test songs per stem class with the class PRESENT (identical across the four Σstem arms;
the master variant differs by one 타악기 song — see note):

{markdown_table(counts_table)}

Test songs per genre (all arms score the same 135 songs):

{markdown_table(genre_table)}

Note on the master variant: `exp003.0 (master)` has 766 present pairs vs 765 for every
Σstem arm. One 타악기 reference straddles the −80 dBFS gate under the two variants'
reference scaling (Σstem references are peak-normalised together with the mixture;
master references are raw ingested stems):

{markdown_table(gating_table)}

That pair enters the master-variant means only; no Σstem number is affected.
"""
    (NUMBERS_DIR / "provenance.md").write_text(text)


def master_gating_differences() -> pd.DataFrame:
    """(song, class) pairs whose present/absent status differs between exp003.0's two
    variants, with each variant's reference RMS and the master-side scores."""
    by_variant = {}
    for arm in ARMS:
        if arm.key == "exp003.0":
            rows = pd.read_parquet(REPO_ROOT / arm.rows_parquet)
            by_variant[arm.variant] = rows[rows.split == SPLIT].set_index(["song_id", "stem_class"])
    joined = by_variant["sumstem"].join(by_variant["master"], lsuffix="_sumstem", rsuffix="_master")
    differing = joined[joined.status_sumstem != joined.status_master]
    columns = ["status_sumstem", "ref_rms_dbfs_sumstem", "status_master", "ref_rms_dbfs_master",
               "si_sdr_master", "usdr_master"]
    return differing[columns].reset_index()


# ------------------------------------------------------------------------------ figure
FIGURE_WIDTH_CM = 8.5
FIGURE_HEIGHT_CM = 5.6
FONT_SIZE_PT = 7.5
# series colours: deltas are transitions, not run entities — one saturated palette hue
# for the primary (coherence) and the reserved neutral for the secondary (density floor)
COHERENCE_COLOR = "#2f62c4"      # 군청 ultramarine, validated slot
DENSITY_FLOOR_COLOR = "#b9b9b4"  # lightened neutral_other, recessive by design


def configure_figure_fonts() -> None:
    """NanumGothic (from koreanize_matplotlib) everywhere, embedded as TrueType."""
    matplotlib.rcParams.update({
        "font.family": "NanumGothic",
        "font.size": FONT_SIZE_PT,
        "axes.unicode_minus": False,   # NanumGothic lacks U+2212; use ASCII hyphen-minus
        "pdf.fonttype": 42,            # embed TrueType (Type 42), keeps text selectable
        "ps.fonttype": 42,
    })


def draw_figure1(deltas: pd.DataFrame, palette: dict) -> plt.Figure:
    """Grouped SI-SDR delta bars per stem, ordered by coherence delta descending."""
    stems = deltas[deltas.stem_class != "mean"].sort_values("coherence", ascending=False)
    positions = list(range(len(stems)))
    width, gap = 0.36, 0.04

    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54))
    axis.set_facecolor("white")
    axis.grid(axis="y", color=palette["grid"], linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(palette["text_secondary"])
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(colors=palette["text_secondary"], labelcolor=palette["text_primary"],
                     width=0.6, length=2.5)

    # secondary series first so the primary sits on top in the legend order we want
    axis.bar([p - (width + gap) / 2 for p in positions], stems["density_floor"],
             width=width, color=DENSITY_FLOOR_COLOR, zorder=2,
             label="density floor  (n≥1 → n≥2, incoherent)")
    axis.bar([p + (width + gap) / 2 for p in positions], stems["coherence"],
             width=width, color=COHERENCE_COLOR, zorder=3,
             label="coherence  (incoherent → partially coherent, n≥2)")
    axis.axhline(0, color=palette["text_primary"], linewidth=0.7, zorder=4)

    axis.set_xticks(positions, stems["stem_class"].tolist())
    axis.set_xlim(-0.6, len(stems) - 0.4)
    axis.set_ylabel("Δ SI-SDR (dB, test, mean over songs)")
    axis.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE_PT - 0.5,
                handlelength=1.0, handletextpad=0.5, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    return fig


def save_figure1(figure: plt.Figure) -> None:
    """Vector PDF (fonts embedded) + 300 dpi PNG under paper/ismir2026_lbd/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        path = FIGURES_DIR / f"fig1_per_stem_delta.{suffix}"
        figure.savefig(path, facecolor="white", **kwargs)
        print(f"figure -> {path}")


# -------------------------------------------------------------------------------- main
def main() -> None:
    NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    present_by_arm = {arm: load_present_rows(arm) for arm in ARMS}
    for arm, present in present_by_arm.items():
        cross_check_against_run_summary(arm, present)
    print("cross-check vs per-run _summary.parquet: all arms agree")

    summary = build_summary(present_by_arm)
    write_summary(summary)
    print("\n== summary ==\n" + summary.round(2).to_string(index=False))

    per_stem = {metric: build_per_stem(present_by_arm, metric) for metric in METRICS}
    counts = present_counts_per_class(present_by_arm)
    write_per_stem(per_stem, counts)

    deltas = build_deltas(per_stem["si_sdr"])
    deltas.to_csv(NUMBERS_DIR / "deltas.csv", index=False)
    print("\n== deltas (SI-SDR) ==\n" + deltas.round(2).to_string(index=False))

    per_genre = build_per_genre(present_by_arm)
    per_genre.to_csv(NUMBERS_DIR / "per_genre.csv", index=False)
    genre_counts = (present_by_arm[SUMSTEM_ARMS[0]].groupby("genre_sub").song_id.nunique()
                    .rename("n_test_songs"))

    gating_table = master_gating_differences()
    print("\n== Σstem vs master gating differences (exp003.0) ==\n"
          + gating_table.round(2).to_string(index=False))
    write_provenance(present_by_arm, counts, genre_counts, gating_table)

    configure_figure_fonts()
    palette = load_palette(PALETTE_PATH)
    save_figure1(draw_figure1(deltas, palette))


if __name__ == "__main__":
    main()
