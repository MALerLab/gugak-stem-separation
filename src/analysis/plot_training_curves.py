"""plot_training_curves.py — val SI-SDR against training epoch, one line per run.

WHAT IT DRAWS. Three figures from one checkpoint pass.
  1. Average curve: for every run in RUNS, the per-eval average val SI-SDR (MSST's own
     selection metric: the unweighted mean of the nine per-class means, on the Σstem val
     tree) from the run's first eval to its most recent one.
  2. Per-stem small multiples: one panel per stem class (palette order), one line
     per run in `Settings.panel_runs`, shared y-axis.
  3. Per-genre small multiples: one panel per genre (palette order), same runs; the value
     is the pooled mean over every (song, stem) score in the genre. Song identity is
     recovered from the val tree's scoring order (see read_msst_checkpoint.py) and genre
     joined from the source manifest.
In both, a triangle marks each run's best AVERAGE epoch and running runs are labelled
`running` in the legend.

DATA SOURCE. MSST persists its whole eval history inside every checkpoint it writes as
`all_metrics`: {epoch_N: {metric: {class: [one value per scored val song]}}}. Both the
rolling `last_<model>.ckpt` (written BEFORE an epoch's eval) and the highest-epoch
best-model `model_*_ep_N_*.ckpt` (written AFTER it) are read, and the longer history wins;
a run that kept only best-model checkpoints simply ends its curve at the last one it
kept. Reading is strictly read-only, so it is safe against a run that is mid-training.

X-AXIS. Epoch index, because that is what every launch report and Notion entry talks
about. Every run here evaluates every 2,500 optimizer steps, so epoch and optimizer steps
are the same axis up to a constant; both columns are written to the parquet, and
`Settings.x_axis` switches the figure between them.

CONFIG. `configs/analysis/plot_training_curves.yaml` — the run list (colour order:
append only, reorder = repaint) and every figure knob. All settings keys are required;
`load_config` is the only reader.

USE AS A CLI:
    uv run python -m src.analysis.plot_training_curves
    uv run python -m src.analysis.plot_training_curves --x-axis optimizer_steps
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path

import koreanize_matplotlib  # noqa: F401 — registers a Korean-capable font with matplotlib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.analysis.compare_runs import find_repo_root, steps_per_eval
from src.analysis.read_msst_checkpoint import build_trajectory, eval_song_order

matplotlib.use("Agg")


# --- config loading ----------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    """One training run drawn as one line."""
    label: str
    experiment_dir: Path
    config_path: Path
    status: str = "finished"  # "finished" | "running" — running is flagged in the legend
    seed: int | None = None  # shown as `(s:N)` in the legend for seed-family runs


@dataclass(frozen=True)
class Settings:
    """Every knob that decides how the figure is built. Loaded 1:1 from the YAML —
    no field has an in-code default, so the YAML stays the single source of values."""
    metric: str
    x_axis: str  # "epoch" | "optimizer_steps"
    rolling_checkpoint_glob: str
    best_checkpoint_glob: str
    palette_path: Path
    output_dir: Path
    figure_name: str
    title: str
    y_label: str
    panel_runs: tuple[str, ...]
    per_stem_figure_name: str
    per_stem_title: str
    per_stem_y_label: str
    per_stem_grid: tuple[int, int]
    per_stem_figure_size: tuple[float, float]
    valid_root: Path
    source_manifest_path: Path
    per_genre_figure_name: str
    per_genre_title: str
    per_genre_y_label: str
    per_genre_grid: tuple[int, int]
    per_genre_figure_size: tuple[float, float]
    x_labels: dict
    legend_max_columns: int
    line_width: float
    marker_size: float
    figure_size: tuple[float, float]
    dpi: int


DEFAULT_CONFIG_PATH = Path("configs/analysis/plot_training_curves.yaml")

# yaml scalars needing a typed container: repo-relative paths and fixed-size tuples
SETTINGS_PATH_FIELDS = ("palette_path", "output_dir", "valid_root", "source_manifest_path")
SETTINGS_TUPLE_FIELDS = ("panel_runs", "per_stem_grid", "per_stem_figure_size",
                         "per_genre_grid", "per_genre_figure_size", "figure_size")


def load_config(config_path: Path) -> tuple[tuple[Run, ...], Settings]:
    """The YAML's run list and settings as typed objects. Paths stay repo-relative."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = tuple(Run(label=entry["label"],
                     experiment_dir=Path(entry["experiment_dir"]),
                     config_path=Path(entry["config_path"]),
                     status=entry.get("status", "finished"),
                     seed=entry.get("seed"))
                 for entry in raw["runs"])
    values = dict(raw["settings"])
    for field_name in SETTINGS_PATH_FIELDS:
        values[field_name] = Path(values[field_name])
    for field_name in SETTINGS_TUPLE_FIELDS:
        values[field_name] = tuple(values[field_name])
    return runs, Settings(**values)


# --- reading -----------------------------------------------------------------------

def epoch_in_filename(checkpoint_path: Path) -> int:
    """The `ep_N` number embedded in an MSST best-model checkpoint filename."""
    match = re.search(r"_ep_(\d+)_", checkpoint_path.name)
    if match is None:
        raise ValueError(f"no `_ep_N_` in checkpoint name: {checkpoint_path.name}")
    return int(match.group(1))


def candidate_history_checkpoints(run: Run, settings: Settings) -> list[Path]:
    """The checkpoints that could hold the run's longest metric history.

    MSST writes the rolling `last_*.ckpt` BEFORE an epoch's eval and the best-model
    `model_*_ep_N_*.ckpt` AFTER it, so whichever is newer can be one eval ahead of the
    other; both are returned and the longer history wins after loading. Damaged/renamed
    checkpoints never match either glob because their names start with neither prefix.

    Args:
        run: the run whose checkpoints/ folder is searched.
        settings: supplies both globs.
    """
    checkpoint_dir = run.experiment_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob(settings.rolling_checkpoint_glob))
    best_models = sorted(checkpoint_dir.glob(settings.best_checkpoint_glob))
    if best_models:
        candidates.append(max(best_models, key=epoch_in_filename))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {checkpoint_dir}")
    return candidates


def load_longest_history(candidates: list[Path]) -> tuple[Path, dict]:
    """Load each candidate's `all_metrics`; return the (path, history) with most evals."""
    import torch  # local import: keeps `--help` and palette-only paths torch-free

    histories: list[tuple[Path, dict]] = []
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "all_metrics" not in checkpoint:
            raise KeyError(f"{path} carries no all_metrics")
        histories.append((path, checkpoint["all_metrics"]))
    return max(histories, key=lambda item: len(item[1]))


def load_run_history(run: Run, settings: Settings) -> tuple[Path, dict]:
    """The run's longest `all_metrics` history and the checkpoint it came from."""
    return load_longest_history(candidate_history_checkpoints(run, settings))


def build_run_trajectory(run: Run, checkpoint_path: Path, all_metrics: dict,
                         settings: Settings) -> pd.DataFrame:
    """One row per eval for one run: epoch, optimizer_steps, avg metric, per-class means.

    Args:
        run: the run being read (label / status / seed are stamped onto every row).
        checkpoint_path: where `all_metrics` came from (recorded for auditability).
        all_metrics: MSST's persisted metric history.
        settings: metric key.
    """
    trajectory = build_trajectory(all_metrics, steps_per_eval(run.config_path),
                                  settings.metric)
    trajectory.insert(0, "run", run.label)
    trajectory.insert(1, "status", run.status)
    trajectory.insert(2, "seed", run.seed)
    trajectory.insert(3, "source_checkpoint", str(checkpoint_path))
    return trajectory


def songs_holding_class(song_order: list[str], valid_root: Path,
                        classes: list[str], extension: str = "flac") -> dict[str, list[str]]:
    """Per stem class, the val songs (in scoring order) whose folder holds that stem.

    MSST skips absent classes during eval, so each class's score list is only as long as
    this list; computed once and reused for every epoch.
    """
    return {stem_class: [song for song in song_order
                         if (valid_root / song / f"{stem_class}.{extension}").exists()]
            for stem_class in classes}


def load_song_genres(source_manifest_path: Path) -> pd.Series:
    """song_id -> genre_sub for the ensemble dataset, from the source manifest."""
    manifest = pd.read_parquet(source_manifest_path, columns=["dataset", "song_id", "genre_sub"])
    return (manifest[manifest["dataset"] == "71955"][["song_id", "genre_sub"]]
            .drop_duplicates().set_index("song_id")["genre_sub"])


def build_run_genre_trajectory(run: Run, all_metrics: dict, scored_songs: dict[str, list[str]],
                               song_genres: pd.Series, settings: Settings) -> pd.DataFrame:
    """One row per (eval, genre) for one run: pooled and class-balanced mean SI-SDR.

    pooled_mean = mean over every (song, stem) score in the genre — every stem of every
    song counts once (the number plotted). class_balanced_mean = mean of the genre's
    per-class means, mirroring how MSST's overall average is built; kept in the table
    for the record, noisier where a genre holds one or two songs of a class.

    Args:
        run: the run being read.
        all_metrics: MSST's persisted metric history.
        scored_songs: class -> songs in scoring order (`songs_holding_class`).
        song_genres: song_id -> genre_sub.
        settings: metric key.
    """
    rows: list[dict] = []
    for epoch_key in sorted(all_metrics, key=lambda k: int(k.split("_")[1])):
        epoch = int(epoch_key.split("_")[1])
        # unpack this epoch into (song, class, score) with genre attached
        scores = []
        for stem_class, values in all_metrics[epoch_key][settings.metric].items():
            songs = scored_songs[stem_class]
            if len(songs) != len(values):
                raise ValueError(f"{run.label} {epoch_key} {stem_class}: {len(values)} scores "
                                 f"vs {len(songs)} songs holding the class — mapping unsafe")
            scores.extend((song, stem_class, float(v)) for song, v in zip(songs, values))
        per_score = pd.DataFrame(scores, columns=["song_id", "stem_class", "score"])
        per_score["genre_sub"] = per_score["song_id"].map(song_genres)
        if per_score["genre_sub"].isna().any():
            raise ValueError(f"{run.label}: some val songs did not join to a genre")
        # aggregate per genre
        for genre, group in per_score.groupby("genre_sub", sort=False):
            rows.append({"run": run.label, "status": run.status, "seed": run.seed,
                         "epoch": epoch, "genre_sub": genre,
                         f"{settings.metric}_pooled_mean": float(group["score"].mean()),
                         f"{settings.metric}_class_balanced_mean":
                             float(group.groupby("stem_class")["score"].mean().mean()),
                         "num_songs": int(group["song_id"].nunique()),
                         "num_scores": len(group)})
    return pd.DataFrame(rows)


def build_curves_tables(runs: tuple[Run, ...], settings: Settings,
                        root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every run's (per-class trajectory, per-genre trajectory), each stacked long.

    Checkpoints are loaded once per run and both tables are built from the same history.
    """
    valid_root = root / settings.valid_root
    song_order = eval_song_order(valid_root)
    song_genres = load_song_genres(root / settings.source_manifest_path)
    scored_songs: dict[str, list[str]] | None = None

    trajectories, genre_trajectories = [], []
    for run in runs:
        checkpoint_path, all_metrics = load_run_history(run, settings)
        if scored_songs is None:
            classes = list(next(iter(all_metrics.values()))[settings.metric].keys())
            scored_songs = songs_holding_class(song_order, valid_root, classes)
        trajectories.append(build_run_trajectory(run, checkpoint_path, all_metrics, settings))
        genre_trajectories.append(build_run_genre_trajectory(
            run, all_metrics, scored_songs, song_genres, settings))
    return (pd.concat(trajectories, ignore_index=True),
            pd.concat(genre_trajectories, ignore_index=True))


def summarize_runs(curves: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per run: best epoch + value, last epoch + value, status — the legend's numbers."""
    avg_column = f"avg_{settings.metric}"
    rows = []
    for run_label, group in curves.groupby("run", sort=False):
        best_row = group.loc[group[avg_column].idxmax()]
        last_row = group.loc[group["epoch"].idxmax()]
        rows.append({"run": run_label,
                     "status": group["status"].iloc[0],
                     "seed": group["seed"].iloc[0],
                     "best_epoch": int(best_row["epoch"]),
                     f"best_{avg_column}": float(best_row[avg_column]),
                     "last_epoch": int(last_row["epoch"]),
                     f"last_{avg_column}": float(last_row[avg_column]),
                     "num_evals": len(group)})
    return pd.DataFrame(rows)


# --- drawing -----------------------------------------------------------------------

def load_palette(palette_path: Path) -> dict:
    """The project's canonical chart palette (단청) as a dict."""
    return yaml.safe_load(palette_path.read_text())


def assign_run_colors(runs: tuple[Run, ...], palette: dict) -> dict[str, str]:
    """Run label -> hex, walking the palette's validated slot order without cycling."""
    slots = [slot["hex"] for slot in palette["slots"]]
    if len(runs) > len(slots):
        raise ValueError(f"{len(runs)} runs but only {len(slots)} palette slots — "
                         "fold runs or facet; never generate a 9th hue")
    return {run.label: slots[index] for index, run in enumerate(runs)}


def legend_label(summary_row: pd.Series, settings: Settings) -> str:
    """`exp002.1 (s:43) · best ep59 +3.99` (+ ` · running` while the run is live)."""
    best_value = summary_row[f"best_avg_{settings.metric}"]
    label = str(summary_row["run"])
    if pd.notna(summary_row["seed"]):
        label += f" (s:{int(summary_row['seed'])})"
    label += f" · best ep{summary_row['best_epoch']} {best_value:+.2f}"
    if summary_row["status"] == "running":
        label += " · running"
    return label


def style_axis(axis: plt.Axes, palette: dict) -> None:
    """Recessive axes: dashed zero line, light horizontal grid, no top/right spines."""
    axis.set_facecolor(palette["surface"])
    axis.axhline(0, color=palette["text_secondary"], linewidth=0.8, linestyle="--")
    axis.grid(axis="y", color=palette["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(palette["grid"])
    axis.tick_params(colors=palette["text_secondary"], labelcolor=palette["text_primary"])


def draw_run_line(axis: plt.Axes, group: pd.DataFrame, value_column: str,
                  best_epoch: int, color: str, label: str | None,
                  palette: dict, settings: Settings) -> None:
    """One run's line on one axis plus a triangle at `best_epoch` (the run's overall best).

    Args:
        axis: target axes.
        group: that run's rows of the curves table.
        value_column: which column to draw (avg or one per-class mean).
        best_epoch: epoch to mark — the run's best AVERAGE epoch, so the marker means the
            same thing on every panel (the checkpoint the headline number came from).
        color: the run's hex.
        label: legend label, or None to keep the line out of the legend.
        palette: surface token for the marker ring.
        settings: axis choice + mark sizes.
    """
    group = group.sort_values(settings.x_axis)
    axis.plot(group[settings.x_axis], group[value_column], color=color,
              linewidth=settings.line_width, label=label)
    best_row = group.loc[group["epoch"] == best_epoch].iloc[0]
    axis.plot(best_row[settings.x_axis], best_row[value_column], marker="^",
              markersize=settings.marker_size, color=color,
              markeredgecolor=palette["surface"], markeredgewidth=1.2, linestyle="none")


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame, colors: dict[str, str],
                palette: dict, settings: Settings) -> plt.Figure:
    """Figure 1: one line per run (average metric), triangle at each run's best epoch.

    Args:
        curves: long table from `build_curves_table`.
        summary: per-run best/last from `summarize_runs` (drives legend + markers).
        colors: run label -> hex.
        palette: the palette dict, for surface / grid / text tokens.
        settings: axis choice, labels, mark sizes.
    """
    avg_column = f"avg_{settings.metric}"
    figure, axis = plt.subplots(figsize=settings.figure_size, dpi=settings.dpi)
    figure.patch.set_facecolor(palette["surface"])

    # one line per run, in RUNS order so legend order = chronology
    for _, summary_row in summary.iterrows():
        run_label = summary_row["run"]
        draw_run_line(axis, curves[curves["run"] == run_label], avg_column,
                      int(summary_row["best_epoch"]), colors[run_label],
                      legend_label(summary_row, settings), palette, settings)

    style_axis(axis, palette)
    axis.set_title(settings.title, color=palette["text_primary"], loc="left", fontsize=13)
    axis.set_xlabel(settings.x_labels[settings.x_axis], color=palette["text_primary"])
    axis.set_ylabel(settings.y_label, color=palette["text_primary"])
    axis.legend(frameon=False, loc="lower right", fontsize=9,
                labelcolor=palette["text_primary"], title="▲ = best epoch",
                title_fontsize=9)
    figure.tight_layout()
    return figure


def panel_reference_mean(curves: pd.DataFrame, summary: pd.DataFrame,
                         panels: list[tuple[str, str]], value_column_for: dict[str, str],
                         settings: Settings) -> float:
    """The 'average shown run × average panel' value: mean over every (run, panel) cell,
    each read at that run's best epoch. Not a statistic to cite — an intuition anchor so a
    panel reads as above/below the middle of the figure.

    Args:
        curves: long table (with a `facet` column when panels are row filters).
        summary: per-run best epochs.
        panels: (facet_key, title) pairs being drawn.
        value_column_for: facet_key -> value column.
        settings: which runs are shown.
    """
    selected = summary[summary["run"].isin(settings.panel_runs)]
    cells: list[float] = []
    for _, summary_row in selected.iterrows():
        at_best = curves[(curves["run"] == summary_row["run"])
                         & (curves["epoch"] == summary_row["best_epoch"])]
        for facet_key, _ in panels:
            rows = at_best[at_best["facet"] == facet_key] if "facet" in at_best else at_best
            cells.append(float(rows[value_column_for[facet_key]].iloc[0]))
    return float(sum(cells) / len(cells))


def plot_small_multiples(curves: pd.DataFrame, summary: pd.DataFrame, colors: dict[str, str],
                         palette: dict, settings: Settings, panels: list[tuple[str, str]],
                         value_column_for: dict[str, str], grid: tuple[int, int],
                         figure_size: tuple[float, float], title: str,
                         y_label: str, reference_label: str) -> plt.Figure:
    """Small multiples: one panel per facet, one line per selected run, shared y-axis.

    The triangle marks each run's overall best epoch (same epoch on every panel); a black
    dotted line at `panel_reference_mean` is repeated identically in every panel; one
    figure-level legend serves all panels.

    Args:
        curves: long table holding one row per (run, epoch[, facet]).
        summary: per-run best/last from `summarize_runs`.
        colors: run label -> hex, assigned over the FULL run list.
        palette: palette dict (tokens).
        settings: which runs, axis choice, mark sizes.
        panels: ordered (facet_key, panel_title) pairs.
        value_column_for: facet_key -> column of `curves` to draw; when the facet is a
            row filter rather than a column, `curves` must carry a `facet` column already
            restricted per panel by the caller (see `plot_per_genre_panels`).
        grid: (rows, columns).
        figure_size: inches.
        title: figure suptitle.
        y_label: shared y label.
        reference_label: legend wording for the dotted reference line (value appended).
    """
    rows, columns = grid
    if len(panels) > rows * columns:
        raise ValueError(f"{len(panels)} panels do not fit a {rows}x{columns} grid")
    selected = summary[summary["run"].isin(settings.panel_runs)]
    missing = set(settings.panel_runs) - set(selected["run"])
    if missing:
        raise ValueError(f"panel_runs not in RUNS: {sorted(missing)}")

    reference = panel_reference_mean(curves, summary, panels, value_column_for, settings)
    figure, axes = plt.subplots(rows, columns, figsize=figure_size, dpi=settings.dpi,
                                sharex=True, sharey=True, squeeze=False)
    figure.patch.set_facecolor(palette["surface"])

    # one panel per facet; legend entries only from the first panel
    for panel_index, (axis, (facet_key, panel_title)) in enumerate(zip(axes.flat, panels)):
        panel_curves = curves[curves["facet"] == facet_key] if "facet" in curves else curves
        for _, summary_row in selected.iterrows():
            run_label = summary_row["run"]
            label = legend_label(summary_row, settings) if panel_index == 0 else None
            draw_run_line(axis, panel_curves[panel_curves["run"] == run_label],
                          value_column_for[facet_key], int(summary_row["best_epoch"]),
                          colors[run_label], label, palette, settings)
        # identical reference line in every panel: the figure-wide mean at best epochs
        axis.axhline(reference, color=palette["text_primary"], linewidth=1.2, linestyle=":",
                     label=f"{reference_label} {reference:+.2f}" if panel_index == 0 else None)
        style_axis(axis, palette)
        axis.set_title(panel_title, color=palette["text_primary"], loc="left", fontsize=12)

    # hide any unused panels
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)

    # shared axis labels + one legend for the whole figure
    for axis in axes[-1, :]:
        axis.set_xlabel(settings.x_labels[settings.x_axis], color=palette["text_primary"])
    for axis in axes[:, 0]:
        axis.set_ylabel(y_label, color=palette["text_primary"])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center",
                  ncol=min(len(labels), settings.legend_max_columns), fontsize=9,
                  labelcolor=palette["text_primary"],
                  title="▲ = run's best epoch (average metric)", title_fontsize=9)
    figure.suptitle(title, color=palette["text_primary"], x=0.01, ha="left", fontsize=14)
    figure.tight_layout(rect=(0, 0.08, 1, 0.97))
    return figure


def plot_per_stem_panels(curves: pd.DataFrame, summary: pd.DataFrame,
                          colors: dict[str, str], palette: dict,
                          settings: Settings) -> plt.Figure:
    """Figure 2: one panel per stem (palette order), drawn from the wide table."""
    class_order = list(palette["stem_class_colors"].keys())
    return plot_small_multiples(
        curves, summary, colors, palette, settings,
        panels=[(stem_class, stem_class) for stem_class in class_order],
        value_column_for={c: f"{settings.metric}_{c}" for c in class_order},
        grid=settings.per_stem_grid, figure_size=settings.per_stem_figure_size,
        title=settings.per_stem_title, y_label=settings.per_stem_y_label,
        reference_label="mean of shown runs × stems (best epochs)")


def plot_per_genre_panels(genre_curves: pd.DataFrame, summary: pd.DataFrame,
                          colors: dict[str, str], palette: dict,
                          settings: Settings) -> plt.Figure:
    """Figure 3: one panel per genre (palette order), pooled mean over the genre's stems.

    Panel titles carry the genre's val song count so thin genres read as thin.
    """
    genre_order = [g for g in palette["genre_colors"] if g in set(genre_curves["genre_sub"])]
    songs_per_genre = genre_curves.groupby("genre_sub")["num_songs"].max()
    value_column = f"{settings.metric}_pooled_mean"
    return plot_small_multiples(
        genre_curves.rename(columns={"genre_sub": "facet"}), summary, colors, palette,
        settings,
        panels=[(g, f"{g} (n={songs_per_genre[g]} songs)") for g in genre_order],
        value_column_for={g: value_column for g in genre_order},
        grid=settings.per_genre_grid, figure_size=settings.per_genre_figure_size,
        title=settings.per_genre_title, y_label=settings.per_genre_y_label,
        reference_label="mean of shown runs × genres (best epochs)")


# --- outputs -----------------------------------------------------------------------

def write_outputs(curves: pd.DataFrame, genre_curves: pd.DataFrame, summary: pd.DataFrame,
                  figures: dict[str, plt.Figure], settings: Settings,
                  root: Path) -> tuple[Path, list[Path]]:
    """Tracked parquet (+ csv twin) under metrics/, gitignored pngs under figures/.

    Args:
        figures: figure name -> matplotlib figure, saved as `<name>.png`.
    """
    metrics_dir = root / settings.output_dir / "metrics"
    figures_dir = root / settings.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    curves.to_parquet(metrics_dir / "training_curves.parquet", index=False)
    curves.to_csv(metrics_dir / "training_curves.csv", index=False)
    genre_curves.to_parquet(metrics_dir / "training_curves_per_genre.parquet", index=False)
    genre_curves.to_csv(metrics_dir / "training_curves_per_genre.csv", index=False)
    summary.to_parquet(metrics_dir / "training_curves_summary.parquet", index=False)
    summary.to_csv(metrics_dir / "training_curves_summary.csv", index=False)
    figure_paths = []
    for name, figure in figures.items():
        figure_path = figures_dir / f"{name}.png"
        figure.savefig(figure_path, facecolor=figure.get_facecolor())
        figure_paths.append(figure_path)
    return metrics_dir, figure_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None,
                        help=f"config YAML (default {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--x-axis", choices=("epoch", "optimizer_steps"), default=None,
                        help="override Settings.x_axis")
    parser.add_argument("--output-dir", default=None, help="override Settings.output_dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    root = find_repo_root()
    config_path = Path(args.config) if args.config else root / DEFAULT_CONFIG_PATH
    loaded_runs, settings = load_config(config_path)
    if args.x_axis:
        settings = replace(settings, x_axis=args.x_axis)
    if args.output_dir:
        settings = replace(settings, output_dir=Path(args.output_dir))

    runs = tuple(replace(run, experiment_dir=root / run.experiment_dir,
                         config_path=root / run.config_path) for run in loaded_runs)
    palette = load_palette(root / settings.palette_path)

    curves, genre_curves = build_curves_tables(runs, settings, root)
    summary = summarize_runs(curves, settings)
    colors = assign_run_colors(runs, palette)
    figures = {
        settings.figure_name: plot_curves(curves, summary, colors, palette, settings),
        settings.per_stem_figure_name: plot_per_stem_panels(
            curves, summary, colors, palette, settings),
        settings.per_genre_figure_name: plot_per_genre_panels(
            genre_curves, summary, colors, palette, settings),
    }
    metrics_dir, figure_paths = write_outputs(curves, genre_curves, summary, figures,
                                              settings, root)

    print(summary.to_string(index=False))
    print(f"\nmetrics -> {metrics_dir}")
    for figure_path in figure_paths:
        print(f"figure  -> {figure_path}")


if __name__ == "__main__":
    main()
