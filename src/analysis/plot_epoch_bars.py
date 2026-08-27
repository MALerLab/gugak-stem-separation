"""plot_epoch_bars.py — one run's val SI-SDR at its best epoch, as bars per stem / per genre.

WHAT IT DRAWS. Two figures, both snapshots (no time axis).
  1. Per-stem bars: one bar per stem class, value = that class's mean val SI-SDR at the
     run's best-average epoch (MSST's own per-class number).
  2. Per-genre bars: one bar per genre, value = the pooled mean over every (song, stem)
     score in the genre at the same epoch; the number of val songs is printed under each
     genre label so thin genres (대풍류 n=2) read as sketches.
A second, optional run is drawn as a reference bar beside each primary bar (same colours as
the training-curve figures — colour follows the run, never repainted). Bars are sorted by
the primary run's value, highest left → lowest right. A black dotted line marks the mean
over every shown bar (the same "average shown run × average panel" anchor as the
small-multiples figures — an intuition anchor, not a statistic to cite).

DATA SOURCE. The tables written by `plot_training_curves.py` (`training_curves.parquet` +
`training_curves_per_genre.parquet`) — this script never touches a checkpoint. Best epoch
per run is recomputed from the curves table (via `summarize_runs`), never read from the
stored summary parquet, so the two can't drift.

CONFIG. `configs/analysis/plot_epoch_bars.yaml` — every figure knob plus
`curves_config`, the curves YAML this script shares its run list and metric with.
All settings keys are required; `load_config` is the only reader.

USE AS A CLI:
    uv run python -m src.analysis.plot_epoch_bars
    uv run python -m src.analysis.plot_epoch_bars --metrics-dir <dir with the two parquets>
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import koreanize_matplotlib  # noqa: F401 — registers a Korean-capable font with matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.compare_runs import find_repo_root
from src.analysis.plot_training_curves import (Settings as CurveSettings,
                                               assign_run_colors, legend_label,
                                               load_config as load_curves_config,
                                               load_palette, style_axis, summarize_runs)

matplotlib.use("Agg")


@dataclass(frozen=True)
class Settings:
    """Every knob that decides how the figures are built. Loaded 1:1 from the YAML —
    no field has an in-code default, so the YAML stays the single source of values."""
    curves_config: Path  # run list + metric live in the curves YAML (single source)
    metric: str
    metrics_dir: Path
    output_dir: Path
    palette_path: Path
    primary_run: str  # the run the figure is about; bars sorted by its values
    reference_run: str | None  # drawn beside each primary bar; None = omit
    per_stem_figure_name: str
    per_stem_title: str
    per_stem_y_label: str
    per_genre_figure_name: str
    per_genre_title: str
    per_genre_y_label: str
    bar_width: float  # per bar; two bars per group when a reference run is drawn
    bar_gap: float  # surface gap between the two bars of a group (axis units)
    value_label_font_size: float
    figure_size: tuple[float, float]
    dpi: int


DEFAULT_CONFIG_PATH = Path("configs/analysis/plot_epoch_bars.yaml")

SETTINGS_PATH_FIELDS = ("metrics_dir", "output_dir", "palette_path")


def load_config(config_path: Path) -> Settings:
    """The YAML's settings as a typed object. Paths stay repo-relative."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = dict(raw["settings"])
    for field_name in SETTINGS_PATH_FIELDS:
        values[field_name] = Path(values[field_name])
    values["figure_size"] = tuple(values["figure_size"])
    return Settings(curves_config=Path(raw["curves_config"]), **values)


# --- reading -----------------------------------------------------------------------

def load_curve_tables(settings: Settings, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-epoch tables written by plot_training_curves.py (curves, per-genre curves)."""
    metrics_dir = root / settings.metrics_dir
    return (pd.read_parquet(metrics_dir / "training_curves.parquet"),
            pd.read_parquet(metrics_dir / "training_curves_per_genre.parquet"))


def shown_runs(settings: Settings) -> list[str]:
    """Run labels drawn, primary first (legend + bar order follow this)."""
    return [settings.primary_run] + ([settings.reference_run] if settings.reference_run else [])


def best_epoch_of(summary: pd.DataFrame, run_label: str) -> int:
    """The run's best-average epoch as recomputed from the curves table."""
    return int(summary.loc[summary["run"] == run_label, "best_epoch"].iloc[0])


def per_stem_at_best(curves: pd.DataFrame, summary: pd.DataFrame, runs: list[str],
                     settings: Settings) -> pd.DataFrame:
    """Wide table: index = stem class, one column per shown run (value at its best epoch)."""
    prefix = f"{settings.metric}_"
    stem_columns = [c for c in curves.columns
                    if c.startswith(prefix) and c != f"avg_{settings.metric}"]
    columns = {}
    for run_label in runs:
        row = curves[(curves["run"] == run_label)
                     & (curves["epoch"] == best_epoch_of(summary, run_label))].iloc[0]
        columns[run_label] = {c[len(prefix):]: float(row[c]) for c in stem_columns}
    return pd.DataFrame(columns)


def per_genre_at_best(genre_curves: pd.DataFrame, summary: pd.DataFrame, runs: list[str],
                      settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Wide table: index = genre, one column per shown run (pooled mean at its best epoch),
    plus the number of val songs per genre (identical across runs — same val tree)."""
    value_column = f"{settings.metric}_pooled_mean"
    columns, num_songs = {}, None
    for run_label in runs:
        rows = genre_curves[(genre_curves["run"] == run_label)
                            & (genre_curves["epoch"] == best_epoch_of(summary, run_label))]
        columns[run_label] = rows.set_index("genre_sub")[value_column].astype(float)
        if num_songs is None:
            num_songs = rows.set_index("genre_sub")["num_songs"].astype(int)
    return pd.DataFrame(columns), num_songs


def sort_by_primary(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Highest primary-run value on the left, lowest on the right."""
    return table.sort_values(settings.primary_run, ascending=False)


# --- drawing -----------------------------------------------------------------------

def draw_grouped_bars(axis: plt.Axes, table: pd.DataFrame, colors: dict[str, str],
                      palette: dict, settings: Settings) -> None:
    """One group per row of `table`, one bar per column (run), value labels above bars.

    Args:
        axis: target axes.
        table: sorted wide table (index = category, columns = shown runs, primary first).
        colors: run label -> hex.
        palette: surface / text tokens.
        settings: bar geometry + label size.
    """
    runs = list(table.columns)
    positions = np.arange(len(table))
    total_width = len(runs) * settings.bar_width + (len(runs) - 1) * settings.bar_gap
    for index, run_label in enumerate(runs):
        offset = -total_width / 2 + settings.bar_width / 2 + index * (settings.bar_width
                                                                     + settings.bar_gap)
        values = table[run_label].to_numpy()
        axis.bar(positions + offset, values, width=settings.bar_width, color=colors[run_label],
                 edgecolor=palette["surface"], linewidth=0.8, zorder=3)
        # value labels: primary ink for the primary run, muted for the reference
        label_color = palette["text_primary"] if index == 0 else palette["text_secondary"]
        for x, value in zip(positions + offset, values):
            # surface-coloured backing so a label stays legible where it crosses the
            # dotted reference line or a grid line
            axis.annotate(f"{value:+.1f}", (x, value), xytext=(0, 3 if value >= 0 else -3),
                          textcoords="offset points", ha="center",
                          va="bottom" if value >= 0 else "top",
                          fontsize=settings.value_label_font_size, color=label_color,
                          bbox={"facecolor": palette["surface"], "edgecolor": "none",
                                "pad": 1.0}, zorder=4)
    axis.set_xticks(positions)


def draw_reference_mean(axis: plt.Axes, table: pd.DataFrame, palette: dict) -> float:
    """Black dotted line at the mean over every shown bar; returns the value for the legend."""
    reference = float(table.to_numpy().mean())
    axis.axhline(reference, color=palette["text_primary"], linewidth=1.2, linestyle=":",
                 zorder=2)
    return reference


def legend_entries(summary: pd.DataFrame, runs: list[str], colors: dict[str, str],
                   reference_mean: float, palette: dict,
                   curve_settings: CurveSettings) -> list:
    """Legend handles: one swatch per shown run (`exp003.0 · best ep28 +10.07 · running`)
    plus the dotted reference line with its value."""
    handles = []
    for run_label in runs:
        summary_row = summary[summary["run"] == run_label].iloc[0]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[run_label],
                                     label=legend_label(summary_row, curve_settings)))
    handles.append(plt.Line2D([], [], color=palette["text_primary"], linestyle=":",
                              linewidth=1.2,
                              label=f"mean of shown bars {reference_mean:+.2f}"))
    return handles


def plot_bars(table: pd.DataFrame, tick_labels: list[str], summary: pd.DataFrame,
              colors: dict[str, str], palette: dict, settings: Settings,
              curve_settings: CurveSettings, title: str, y_label: str) -> plt.Figure:
    """One bar figure: grouped bars sorted by the primary run, reference line, legend.

    Args:
        table: sorted wide table (index = category, columns = shown runs).
        tick_labels: x tick text per row of `table` (genre labels carry n songs).
        summary: per-run best/last (legend text).
        colors: run label -> hex.
        palette: palette dict.
        settings: figure knobs.
        curve_settings: passed to `legend_label` (metric name).
        title: descriptive figure title.
        y_label: y-axis text.
    """
    figure, axis = plt.subplots(figsize=settings.figure_size, dpi=settings.dpi)
    figure.patch.set_facecolor(palette["surface"])
    draw_grouped_bars(axis, table, colors, palette, settings)
    reference_mean = draw_reference_mean(axis, table, palette)
    style_axis(axis, palette)
    axis.set_xticklabels(tick_labels, color=palette["text_primary"])
    axis.set_title(title, color=palette["text_primary"], loc="left", fontsize=13)
    axis.set_ylabel(y_label, color=palette["text_primary"])
    axis.legend(handles=legend_entries(summary, list(table.columns), colors, reference_mean,
                                       palette, curve_settings),
                frameon=False, loc="upper right", fontsize=9,
                labelcolor=palette["text_primary"])
    figure.tight_layout()
    return figure


# --- orchestration -----------------------------------------------------------------

def build_figures(settings: Settings, root: Path) -> dict[str, plt.Figure]:
    """Read the curve tables once and return {figure_name: figure} for both bar charts."""
    curve_runs, curve_settings = load_curves_config(root / settings.curves_config)
    curves, genre_curves = load_curve_tables(settings, root)
    summary = summarize_runs(curves, curve_settings)
    runs = shown_runs(settings)
    palette = load_palette(root / settings.palette_path)
    colors = assign_run_colors(curve_runs, palette)  # over the FULL list: never repaint a run

    stem_table = sort_by_primary(per_stem_at_best(curves, summary, runs, settings), settings)
    genre_table, num_songs = per_genre_at_best(genre_curves, summary, runs, settings)
    genre_table = sort_by_primary(genre_table, settings)
    genre_labels = [f"{genre}\n(n={num_songs[genre]})" for genre in genre_table.index]

    return {
        settings.per_stem_figure_name: plot_bars(
            stem_table, list(stem_table.index), summary, colors, palette, settings,
            curve_settings, settings.per_stem_title, settings.per_stem_y_label),
        settings.per_genre_figure_name: plot_bars(
            genre_table, genre_labels, summary, colors, palette, settings,
            curve_settings, settings.per_genre_title, settings.per_genre_y_label),
    }


def write_figures(figures: dict[str, plt.Figure], settings: Settings, root: Path) -> list[Path]:
    """Save every figure as PNG under output_dir (gitignored via experiments/**/*.png)."""
    output_dir = root / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, facecolor=figure.get_facecolor())
        paths.append(path)
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None,
                        help=f"config YAML (default {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--metrics-dir", default=None, help="override Settings.metrics_dir")
    parser.add_argument("--output-dir", default=None, help="override Settings.output_dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    root = find_repo_root()
    config_path = Path(args.config) if args.config else root / DEFAULT_CONFIG_PATH
    settings = load_config(config_path)
    if args.metrics_dir:
        settings = replace(settings, metrics_dir=Path(args.metrics_dir))
    if args.output_dir:
        settings = replace(settings, output_dir=Path(args.output_dir))
    figures = build_figures(settings, root)
    for path in write_figures(figures, settings, root):
        print(f"figure -> {path}")


if __name__ == "__main__":
    main()
