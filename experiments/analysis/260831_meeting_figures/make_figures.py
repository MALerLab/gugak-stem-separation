"""One-off figures for the 2026-08-31 individual meeting.

Figure 1 — BS-RoFormer headline + seed twin: exp003.0 (finished, +10.70@59) and
exp003.1 (seed 46, running) val curves on one axis, exp004 (best HTDemucs) as context.
Figure 2 — where coherence's +1.55 dB lives: per-stem final val SI-SDR, exp004.1
(incoherent baseline, gray) vs exp004 (coherent, gold), sorted by gain, Δ labelled.

Figures 3a/3b — si_sdr vs usdr for the baseline model (exp003.0 ep59), per stem and
per genre, from the standalone eval pipeline's aggregate tables; class-balanced
averages carried in the legend.

Data: tracked training_curves.parquet (exp003.0, exp004) + train.log parses via
scripts/run_digest.py (exp003.1, exp004.1 — not yet in the curves run list) +
experiments/exp003.0_*/eval/ aggregate parquets (figures 3a/3b).
Colours: dancheong palette; exp003.0/exp004 keep their frozen slots, exp003.1 takes
the free 8th slot (appended, never repainted), the baseline run wears reference gray.

Run: uv run python experiments/analysis/260831_meeting_figures/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import koreanize_matplotlib  # noqa: F401 — Korean-capable font for stem names
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # NanumGothic lacks U+2212

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_digest import parse_train_log  # noqa: E402
from src.analysis.plot_training_curves import load_palette, style_axis  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"
CURVES_PARQUET = (REPO_ROOT / "experiments/analysis/260817_training_curves"
                  / "metrics/training_curves.parquet")

# frozen entity bindings (RUNS colour order) + the one append this figure introduces
RUN_COLORS = {"exp003.0": "#9c4f9b", "exp004": "#d99a1e", "exp003.1": "#c23a5c"}
BASELINE_GRAY = "#8a8a84"

LOGS = {
    "exp003.1": "experiments/exp003.1_260819_bsroformer_seed46/train.log",
    "exp004.1": "experiments/exp004.1_260817_incoherent_n2/train.log",
    "exp004": "experiments/exp004_260815_coherent_p1_uniform/train.log",
}


def curve_from_parquet(curves: pd.DataFrame, run: str) -> pd.Series:
    """Epoch-indexed average val SI-SDR for one run from the tracked table."""
    rows = curves[curves["run"] == run].sort_values("epoch")
    return rows.set_index("epoch")["avg_si_sdr"]


def figure_curves(palette: dict) -> plt.Figure:
    """Figure 1: exp003.0 + exp003.1 curves, exp004 context, best-epoch markers."""
    curves = pd.read_parquet(CURVES_PARQUET)
    series = {"exp003.0": curve_from_parquet(curves, "exp003.0"),
              "exp004": curve_from_parquet(curves, "exp004")}
    log = parse_train_log(REPO_ROOT / LOGS["exp003.1"])
    series["exp003.1"] = pd.Series(log.epoch_avg_vals,
                                   index=range(len(log.epoch_avg_vals)))

    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    style_axis(axis, palette)
    labels = {
        "exp003.0": "exp003.0 BS-RoFormer (s:42) · best ep59 +10.70",
        "exp003.1": "exp003.1 seed twin (s:46) · running",
        "exp004": "exp004 HTDemucs coherent (s:45) · best ep58 +6.70",
    }
    for run, values in series.items():
        linestyle = "--" if run == "exp003.1" else "-"  # twin overlaps the pilot exactly
        axis.plot(values.index, values.values, color=RUN_COLORS[run], linewidth=2.0,
                  linestyle=linestyle, label=labels[run])
        best_epoch = values.idxmax()
        axis.plot(best_epoch, values[best_epoch], marker="^", markersize=9,
                  color=RUN_COLORS[run], markeredgecolor=palette["surface"])
        label_dy = 10 if run == "exp003.1" else 0  # clear of the best marker
        axis.annotate(f"{values.iloc[-1]:+.2f}", (values.index[-1], values.iloc[-1]),
                      xytext=(6, label_dy), textcoords="offset points", va="center",
                      color=palette["text_primary"], fontsize=9)
    axis.set_xlabel("epoch", color=palette["text_primary"])
    axis.set_ylabel("avg val SI-SDR (dB, Σstem, 91 songs)", color=palette["text_primary"])
    axis.set_title("BS-RoFormer result is seed-robust — twin tracks the pilot within 0.1 dB",
                   color=palette["text_primary"], loc="left")
    axis.legend(loc="lower right", frameon=False, fontsize=9,
                labelcolor=palette["text_primary"])
    fig.tight_layout()
    return fig


def figure_coherence(palette: dict) -> plt.Figure:
    """Figure 2: per-stem final val SI-SDR, incoherent (gray) vs coherent (gold)."""
    final = {run: parse_train_log(REPO_ROOT / LOGS[run]).epoch_stem_vals[-1]
             for run in ("exp004.1", "exp004")}
    # baseline-ascending: weakest incoherent stem on the left, strongest on the right
    stems = sorted(final["exp004"], key=lambda s: final["exp004.1"][s])
    gains = [final["exp004"][s] - final["exp004.1"][s] for s in stems]

    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    style_axis(axis, palette)
    positions = range(len(stems))
    width, gap = 0.38, 0.03
    axis.bar([p - (width + gap) / 2 for p in positions],
             [final["exp004.1"][s] for s in stems], width=width, color=BASELINE_GRAY,
             label="exp004.1 incoherent n≥2 (s:45) · +5.15 avg", zorder=2)
    axis.bar([p + (width + gap) / 2 for p in positions],
             [final["exp004"][s] for s in stems], width=width, color=RUN_COLORS["exp004"],
             label="exp004 coherent p=1 n≥2 (s:45) · +6.70 avg", zorder=2)
    for position, stem, gain in zip(positions, stems, gains):
        top = max(final["exp004"][stem], final["exp004.1"][stem])
        axis.annotate(f"{gain:+.1f}", (position, top), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=9,
                      color=palette["text_primary"])
    axis.set_xticks(list(positions), stems)
    axis.set_ylabel("final val SI-SDR (dB, Σstem, ep59)", color=palette["text_primary"])
    axis.set_title("Coherent mixing rescues weak heads — 기타 carries a third of the +1.55 dB",
                   color=palette["text_primary"], loc="left")
    axis.legend(loc="upper left", frameon=False, fontsize=9,
                labelcolor=palette["text_primary"])
    fig.tight_layout()
    return fig


EVAL_DIR = (REPO_ROOT / "experiments/exp003.0_260809_bsroformer_pilot/eval")
EVAL_STEM = "eval_exp003.0_ep59_val_sumstem"
# metric colours: adjacent validated palette pair (석록 / 군청); not run entities
METRIC_COLORS = {"si_sdr": "#1f9e78", "usdr": "#2f62c4"}


def metric_legend_labels() -> dict[str, str]:
    """Legend text carrying the class-balanced average of each metric."""
    summary = pd.read_parquet(EVAL_DIR / f"{EVAL_STEM}_summary.parquet")
    means = summary.set_index("metric")["class_balanced_mean"]
    return {"si_sdr": f"SI-SDR (training metric) · avg {means['si_sdr']:+.2f}",
            "usdr": f"uSDR (new, not scale-invariant) · avg {means['usdr']:+.2f}"}


def figure_metrics_bars(table: "pd.DataFrame", index_column: str, value_columns: dict,
                        palette: dict, title: str, y_label: str,
                        tick_suffix: "pd.Series | None" = None) -> plt.Figure:
    """Paired si_sdr/usdr bars over stems or genres, sorted by si_sdr descending."""
    table = table.sort_values(value_columns["si_sdr"], ascending=False)
    labels = metric_legend_labels()
    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    style_axis(axis, palette)
    positions = range(len(table))
    width, gap = 0.38, 0.03
    for offset, metric in ((-1, "si_sdr"), (1, "usdr")):
        axis.bar([p + offset * (width + gap) / 2 for p in positions],
                 table[value_columns[metric]], width=width,
                 color=METRIC_COLORS[metric], label=labels[metric], zorder=2)
    for position, (_, row) in zip(positions, table.iterrows()):
        top = max(row[value_columns["si_sdr"]], row[value_columns["usdr"]])
        delta = row[value_columns["usdr"]] - row[value_columns["si_sdr"]]
        axis.annotate(f"{delta:+.1f}", (position, top), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=9,
                      color=palette["text_primary"])
    ticks = table[index_column]
    if tick_suffix is not None:
        ticks = ticks + tick_suffix.loc[table.index].map(lambda n: f"\n(n={n})")
    axis.set_xticks(list(positions), ticks)
    axis.set_ylabel(y_label, color=palette["text_primary"])
    axis.set_title(title, color=palette["text_primary"], loc="left")
    axis.legend(loc="upper right", frameon=False, fontsize=9,
                labelcolor=palette["text_primary"])
    fig.tight_layout()
    return fig


def figure_metrics_per_stem(palette: dict) -> plt.Figure:
    """Figure 3a: per-stem si_sdr vs usdr for exp003.0 ep59 (Δ = usdr − si_sdr)."""
    table = pd.read_parquet(EVAL_DIR / f"{EVAL_STEM}_per_stem.parquet")
    return figure_metrics_bars(
        table, "stem_class", {"si_sdr": "si_sdr_mean", "usdr": "usdr_mean"}, palette,
        "SI-SDR vs uSDR — exp003.0 ep59 baseline, per stem (Δ = uSDR - SI-SDR)",
        "val mean over songs (dB, Σstem, ep59)")


def figure_metrics_per_genre(palette: dict) -> plt.Figure:
    """Figure 3b: per-genre pooled si_sdr vs usdr for exp003.0 ep59."""
    table = pd.read_parquet(EVAL_DIR / f"{EVAL_STEM}_per_genre.parquet")
    return figure_metrics_bars(
        table, "genre_sub",
        {"si_sdr": "si_sdr_pooled_mean", "usdr": "usdr_pooled_mean"}, palette,
        "SI-SDR vs uSDR — exp003.0 ep59 baseline, per genre (Δ = uSDR - SI-SDR)",
        "val pooled mean over (song, stem) (dB, Σstem, ep59)",
        tick_suffix=table["num_songs"])


def main() -> None:
    palette = load_palette(REPO_ROOT / "configs/palette_dancheong.yaml")
    OUT_DIR.mkdir(exist_ok=True)
    figures = {"meeting_curves_bsroformer_twin": figure_curves(palette),
               "meeting_coherence_per_stem": figure_coherence(palette),
               "meeting_metrics_per_stem": figure_metrics_per_stem(palette),
               "meeting_metrics_per_genre": figure_metrics_per_genre(palette)}
    for name, figure in figures.items():
        path = OUT_DIR / f"{name}.png"
        figure.savefig(path, dpi=160, facecolor=palette["surface"])
        print(f"figure -> {path}")


if __name__ == "__main__":
    main()
