"""compare_runs.py — compare two or more training arms at matched optimizer steps.

THE QUESTION THIS ANSWERS. A raw side-by-side table invites the eye to read any difference
as a finding. It usually isn't: exp001.1 measured this val metric's epoch-to-epoch swing at
roughly ±0.5–1.4 dB, so a 0.4 dB gap between two arms is not evidence of anything. The
headline output here is therefore not the gap — it is the gap DIVIDED by each arm's own
volatility over the same window. If that ratio is below 1 the report says the gap sits
inside the noise, in those words, instead of printing a number that looks like a result.

ALIGNMENT IS BY OPTIMIZER STEP, never by wall-clock and never by epoch index. Wall-clock is
meaningless on a shared machine (arms contend for dataloader I/O). Epoch index is
meaningless across model families, because "epoch" here is just `num_steps` loader
iterations and BS-RoFormer arms use a different value. Each arm's steps-per-eval is
computed from its OWN config as num_steps / gradient_accumulation_steps, and arms are then
intersected on the step counts they actually share.

DATA SOURCE. MSST persists its full metric history inside every checkpoint as
`all_metrics`: {epoch_N: {metric: {class: [one value per scored val song]}}}. That is the
authoritative record, and it is live, because `last_<model>.ckpt` is rewritten every eval.
Reading it is strictly read-only and safe against an arm that is still training.

USE AS A LIBRARY:
    from src.analysis.compare_runs import Arm, load_settings, compare_arms, format_report

    settings = load_settings("configs/analysis/compare_runs.yaml")
    arms = [Arm("exp002", Path("experiments/exp002_..."), Path("configs/exp002_....yaml")),
            Arm("exp002.1", Path("experiments/exp002.1_..."), Path("configs/exp002.1_....yaml"))]
    result = compare_arms(arms, settings)
    print(format_report(result, settings))

USE AS A CLI:
    uv run python -m src.analysis.compare_runs \
      --arm exp002=experiments/exp002_260805_htdemucs_v2_uniform_n=configs/exp002_htdemucs_v2_uniform_n.yaml \
      --arm exp002.1=experiments/exp002.1_260806_twin=configs/exp002.1_htdemucs_twin.yaml \
      --out-dir experiments/exp002.1_260806_twin/metrics
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

DEFAULT_SETTINGS_PATH = "configs/analysis/compare_runs.yaml"


# --- value objects -----------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One training run being compared."""
    label: str
    experiment_dir: Path
    config_path: Path


@dataclass(frozen=True)
class Settings:
    """Every knob that decides how a comparison is read. Loaded from YAML, CLI-overridable."""
    warmup_steps: int = 25000
    min_evals_for_volatility: int = 3
    volatility_window: int | None = None
    inside_noise_ratio: float = 1.0
    exceeds_noise_ratio: float = 2.0
    highlight_classes: tuple[str, ...] = ()
    overall_label: str = "OVERALL"
    metric: str = "si_sdr"
    checkpoint_glob: str = "last_*.ckpt"


@dataclass
class ComparisonResult:
    """Everything a report or a caller needs, already computed."""
    arms: list[Arm]
    trajectories: list[pd.DataFrame]
    aligned: pd.DataFrame
    summaries: pd.DataFrame
    class_names: list[str] = field(default_factory=list)


# --- configuration -----------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repo root (the directory holding pyproject.toml)."""
    current = Path.cwd() if start is None else start
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def load_settings(config_path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Build Settings from the analysis YAML.

    Args:
        config_path: path to the compare_runs config, absolute or repo-relative.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = find_repo_root() / path
    block = OmegaConf.load(path).compare_runs
    volatility_window = block.get("volatility_window", None)
    return Settings(
        warmup_steps=int(block.warmup_steps),
        min_evals_for_volatility=int(block.min_evals_for_volatility),
        volatility_window=None if volatility_window is None else int(volatility_window),
        inside_noise_ratio=float(block.inside_noise_ratio),
        exceeds_noise_ratio=float(block.exceeds_noise_ratio),
        highlight_classes=tuple(block.highlight_classes),
        overall_label=str(block.overall_label),
        metric=str(block.metric),
        checkpoint_glob=str(block.checkpoint_glob),
    )


def parse_arm(spec: str) -> Arm:
    """Parse a LABEL=EXPERIMENT_DIR=CONFIG_PATH argument into an Arm.

    Args:
        spec: three '='-separated fields, e.g.
            'exp002=experiments/exp002_260805_.../=configs/exp002_....yaml'.
    """
    parts = spec.split("=")
    if len(parts) != 3:
        raise ValueError(f"--arm needs LABEL=EXPERIMENT_DIR=CONFIG_PATH, got {spec!r}")
    label, experiment_dir, config_path = parts
    return Arm(label=label,
               experiment_dir=Path(experiment_dir),
               config_path=Path(config_path))


# --- loading -----------------------------------------------------------------------

def steps_per_eval(config_path: Path) -> int:
    """Optimizer steps between evals, from the arm's own config.

    MSST evaluates once per 'epoch', where an epoch is `num_steps` loader iterations;
    with gradient accumulation those are not the same thing as optimizer steps.

    Args:
        config_path: the arm's experiment YAML.
    """
    # OmegaConf, not yaml.safe_load — MSST loads htdemucs configs through OmegaConf, and
    # plain YAML 1.1 misreads unpunctuated scientific notation (`1e-3` becomes a string).
    config = OmegaConf.load(config_path)
    loader_steps = int(config.training.num_steps)
    accumulation = int(config.training.get("gradient_accumulation_steps", 1))
    return loader_steps // accumulation


def load_all_metrics(experiment_dir: Path, checkpoint_glob: str) -> dict:
    """Read MSST's persisted metric history out of an arm's rolling checkpoint.

    Args:
        experiment_dir: the arm's experiment folder (must contain checkpoints/).
        checkpoint_glob: which checkpoint holds the rolling history.
    """
    candidates = sorted((experiment_dir / "checkpoints").glob(checkpoint_glob))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoints/{checkpoint_glob} under {experiment_dir}")
    checkpoint = torch.load(candidates[0], map_location="cpu", weights_only=False)
    if "all_metrics" not in checkpoint:
        raise KeyError(f"{candidates[0]} carries no all_metrics")
    return checkpoint["all_metrics"]


def build_trajectory(arm: Arm, settings: Settings) -> pd.DataFrame:
    """One row per eval cycle: optimizer_steps + per-class means + the overall mean.

    'Overall' reproduces MSST's own selection metric — the unweighted mean of the
    per-class means, NOT a mean over songs — so it matches the checkpoint filenames.

    Args:
        arm: the run to read.
        settings: supplies the metric key and the checkpoint glob.
    """
    all_metrics = load_all_metrics(arm.experiment_dir, settings.checkpoint_glob)
    per_eval_steps = steps_per_eval(arm.config_path)

    rows: list[dict] = []
    for epoch_key in sorted(all_metrics, key=lambda k: int(k.split("_")[1])):
        epoch = int(epoch_key.split("_")[1])
        class_means = {name: float(np.mean(values))
                       for name, values in all_metrics[epoch_key][settings.metric].items()}
        rows.append({"arm": arm.label,
                     "epoch": epoch,
                     "optimizer_steps": (epoch + 1) * per_eval_steps,
                     settings.overall_label: float(np.mean(list(class_means.values()))),
                     **class_means})
    return pd.DataFrame(rows)


# --- analysis ----------------------------------------------------------------------

def mean_absolute_successive_difference(series: np.ndarray) -> float:
    """An arm's own eval-to-eval swing: the average size of a step between consecutive evals.

    This is the noise yardstick. A standard deviation would also absorb the run's genuine
    upward trend and overstate the wobble; successive differences measure only the jitter
    around wherever the curve currently is.

    Args:
        series: one arm's metric values at consecutive matched evals.
    """
    if series.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(series))))


def apply_window(series: np.ndarray, window: int | None) -> np.ndarray:
    """Restrict a series to its trailing `window` entries (None = keep everything).

    Args:
        series: values at consecutive matched evals, oldest first.
        window: how many trailing evals to keep.
    """
    return series if window is None else series[-window:]


def between_arm_gap(stacked: np.ndarray) -> np.ndarray:
    """Per-eval gap across arms: signed for a pair, max-minus-min for three or more.

    Args:
        stacked: (n_arms, n_evals) metric values.
    """
    if stacked.shape[0] == 2:
        return stacked[0] - stacked[1]
    return stacked.max(axis=0) - stacked.min(axis=0)


def verdict_for(ratio: float, settings: Settings) -> str:
    """Turn a gap-to-noise ratio into the word the report prints."""
    if not np.isfinite(ratio):
        return "insufficient evals"
    if ratio < settings.inside_noise_ratio:
        return "INSIDE NOISE"
    if ratio < settings.exceeds_noise_ratio:
        return "marginal"
    return "EXCEEDS NOISE"


def compare_class(aligned: pd.DataFrame, arms: list[Arm], class_name: str,
                  settings: Settings) -> dict:
    """Gap-vs-noise summary for one class over the (windowed) matched region.

    The gap and the swing are computed over the SAME trailing window, or their ratio
    would not be apples-to-apples.

    Args:
        aligned: matched evals indexed by optimizer_steps, columns '<label>::<class>'.
        arms: the arms being compared, in report order.
        class_name: the class (or the overall label) to summarise.
        settings: window and threshold configuration.
    """
    series = {arm.label: apply_window(
        aligned[f"{arm.label}::{class_name}"].to_numpy(), settings.volatility_window)
        for arm in arms}
    stacked = np.vstack(list(series.values()))

    per_eval_gap = between_arm_gap(stacked)
    mean_absolute_gap = float(np.mean(np.abs(per_eval_gap)))
    max_absolute_gap = float(np.max(np.abs(per_eval_gap)))

    # within-arm volatility over the same window, pooled across arms
    per_arm_swing = {label: mean_absolute_successive_difference(values)
                     for label, values in series.items()}
    finite_swings = [v for v in per_arm_swing.values() if np.isfinite(v)]
    pooled_swing = float(np.mean(finite_swings)) if finite_swings else float("nan")

    ratio = (mean_absolute_gap / pooled_swing
             if pooled_swing and np.isfinite(pooled_swing) and pooled_swing > 0
             else float("nan"))

    summary = {"class": class_name,
               "mean_abs_gap_db": mean_absolute_gap,
               "max_abs_gap_db": max_absolute_gap,
               "within_arm_swing_db": pooled_swing,
               "gap_to_noise_ratio": ratio,
               "verdict": verdict_for(ratio, settings)}
    for arm in arms:
        summary[f"final_{arm.label}"] = float(series[arm.label][-1])
        summary[f"swing_{arm.label}"] = per_arm_swing[arm.label]
    return summary


def align_arms(trajectories: list[pd.DataFrame], arms: list[Arm],
               class_names: list[str], warmup_steps: int) -> pd.DataFrame:
    """Intersect arms on shared optimizer-step counts, dropping the warm-up window.

    Args:
        trajectories: one per arm, in the same order as `arms`.
        arms: the arms being compared.
        class_names: classes to carry through (the overall label included).
        warmup_steps: evals at or below this step count are discarded.
    """
    shared_steps = set.intersection(*(set(t["optimizer_steps"]) for t in trajectories))
    shared_steps = sorted(s for s in shared_steps if s > warmup_steps)

    frame = pd.DataFrame({"optimizer_steps": shared_steps}).set_index("optimizer_steps")
    for arm, trajectory in zip(arms, trajectories):
        indexed = trajectory.set_index("optimizer_steps")
        for class_name in class_names:
            frame[f"{arm.label}::{class_name}"] = indexed.loc[shared_steps, class_name]
    return frame


def compare_arms(arms: list[Arm], settings: Settings) -> ComparisonResult:
    """Load every arm, align them, and summarise every shared class.

    Args:
        arms: two or more runs to compare.
        settings: reading rules.
    """
    if len(arms) < 2:
        raise ValueError("need at least two arms to compare")

    trajectories = [build_trajectory(arm, settings) for arm in arms]

    # only classes every arm actually scored can be compared
    reserved = {"arm", "epoch", "optimizer_steps", settings.overall_label}
    shared_classes = set.intersection(
        *(set(t.columns) - reserved for t in trajectories))
    class_names = [settings.overall_label] + sorted(shared_classes)

    aligned = align_arms(trajectories, arms, class_names, settings.warmup_steps)
    summaries = (pd.DataFrame([compare_class(aligned, arms, name, settings)
                               for name in class_names])
                 if not aligned.empty else pd.DataFrame())
    return ComparisonResult(arms=arms, trajectories=trajectories, aligned=aligned,
                            summaries=summaries, class_names=class_names)


# --- reporting ---------------------------------------------------------------------

def format_report(result: ComparisonResult, settings: Settings) -> str:
    """Render the human-readable report.

    Args:
        result: output of compare_arms.
        settings: reading rules (thresholds, window, highlights).
    """
    arms, aligned, summaries = result.arms, result.aligned, result.summaries
    labels = [arm.label for arm in arms]

    if aligned.empty:
        lines = [f"No matched evals past warm-up ({settings.warmup_steps:,} steps)."]
        for arm, trajectory in zip(arms, result.trajectories):
            lines.append(f"  {arm.label}: {len(trajectory)} evals, "
                         f"latest {trajectory['optimizer_steps'].max():,} steps")
        return "\n".join(lines)

    window_note = ("all matched evals" if settings.volatility_window is None
                   else f"trailing {settings.volatility_window} evals")
    lines = [
        "=" * 84,
        f"ARM COMPARISON — {' vs '.join(labels)}",
        "=" * 84,
        f"matched evals: {len(aligned)}  "
        f"(optimizer steps {aligned.index.min():,} → {aligned.index.max():,}; "
        f"warm-up ≤{settings.warmup_steps:,} dropped)",
        f"gap & swing computed over: {window_note}",
        "",
    ]

    if len(aligned) < settings.min_evals_for_volatility:
        lines += [
            f"⚠️  TOO EARLY TO READ. Fewer than {settings.min_evals_for_volatility} matched",
            "    evals past warm-up, so the within-arm volatility estimate is not yet",
            "    meaningful. Numbers below are for monitoring only — do not conclude.",
            "",
        ]

    lines += ["GAP vs NOISE — the question. Ratio = mean |between-arm gap| ÷ each arm's",
              f"own eval-to-eval swing over the same window. Below "
              f"{settings.inside_noise_ratio} = inside the noise.",
              "-" * 84,
              f"{'class':<10}{'gap dB':>9}{'swing dB':>10}{'ratio':>8}  {'verdict':<16}"
              + "".join(f"{'final ' + l:>16}" for l in labels)]
    for _, row in summaries.iterrows():
        marker = " ←" if row["class"] in settings.highlight_classes else ""
        lines.append(
            f"{row['class']:<10}{row['mean_abs_gap_db']:>9.2f}"
            f"{row['within_arm_swing_db']:>10.2f}{row['gap_to_noise_ratio']:>8.2f}  "
            f"{row['verdict']:<16}"
            + "".join(f"{row['final_' + l]:>16.2f}" for l in labels) + marker)

    if settings.highlight_classes:
        lines += ["", "HIGHLIGHTED — the classes this comparison was built to interrogate",
                  "-" * 84]
        for class_name in settings.highlight_classes:
            row = summaries[summaries["class"] == class_name]
            if row.empty:
                lines.append(f"  {class_name}: not present in these arms")
                continue
            row = row.iloc[0]
            per_arm = " · ".join(
                f"{l} {row['final_' + l]:+.2f} (own swing {row['swing_' + l]:.2f})"
                for l in labels)
            lines.append(f"  {class_name}: {per_arm}")
            lines.append(f"      mean gap {row['mean_abs_gap_db']:.2f} dB vs noise "
                         f"{row['within_arm_swing_db']:.2f} dB → {row['verdict']}")

    lines += ["", f"PER-EVAL DETAIL ({settings.overall_label}) — matched optimizer steps",
              "-" * 84,
              f"{'steps':>10}" + "".join(f"{l:>14}" for l in labels) + f"{'gap':>10}"]
    for steps, row in aligned.iterrows():
        values = [row[f"{l}::{settings.overall_label}"] for l in labels]
        gap = values[0] - values[1] if len(values) == 2 else max(values) - min(values)
        lines.append(f"{steps:>10,}" + "".join(f"{v:>14.3f}" for v in values)
                     + f"{gap:>10.3f}")

    lines += ["", "=" * 84,
              "Read the ratio, not the gap. n=2 arms give a RANGE, not a standard",
              "deviation — a ratio above the 'exceeds' cutoff says 'worth interpreting',",
              "never 'significant'.",
              "=" * 84]
    return "\n".join(lines)


def write_outputs(result: ComparisonResult, report: str, out_dir: Path) -> Path:
    """Persist the comparison. Parquet is the tracked artifact, the text is its readable twin.

    Args:
        result: output of compare_arms.
        report: rendered text report.
        out_dir: destination directory (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "_vs_".join(arm.label for arm in result.arms)
    result.summaries.to_parquet(out_dir / f"arm_comparison_{stem}.parquet", index=False)
    result.aligned.reset_index().to_parquet(
        out_dir / f"arm_matched_evals_{stem}.parquet", index=False)
    (out_dir / f"arm_comparison_{stem}.txt").write_text(report + "\n")
    return out_dir / f"arm_comparison_{stem}.parquet"


# --- entry point -------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """CLI surface. Every config value has an override so one-off runs need no file edit."""
    parser = argparse.ArgumentParser(
        description="Compare training arms at matched optimizer steps.")
    parser.add_argument("--arm", action="append", required=True,
                        help="LABEL=EXPERIMENT_DIR=CONFIG_PATH (repeat per arm)")
    parser.add_argument("--config", default=DEFAULT_SETTINGS_PATH,
                        help="analysis settings YAML")
    parser.add_argument("--out-dir", default=None, help="write tables here")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--volatility-window", type=int, default=None,
                        help="use only the trailing N matched evals for gap and swing")
    parser.add_argument("--min-evals-for-volatility", type=int, default=None)
    parser.add_argument("--inside-noise-ratio", type=float, default=None)
    parser.add_argument("--exceeds-noise-ratio", type=float, default=None)
    parser.add_argument("--highlight", action="append", default=None,
                        help="class to call out (repeatable; replaces the config list)")
    parser.add_argument("--metric", default=None)
    return parser


def settings_with_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply any CLI overrides on top of the loaded settings.

    Args:
        settings: values from the YAML.
        args: parsed CLI namespace; None means 'not overridden'.
    """
    overrides = {}
    for name in ("warmup_steps", "volatility_window", "min_evals_for_volatility",
                 "inside_noise_ratio", "exceeds_noise_ratio", "metric"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if args.highlight is not None:
        overrides["highlight_classes"] = tuple(args.highlight)
    return replace(settings, **overrides) if overrides else settings


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    settings = settings_with_overrides(load_settings(args.config), args)
    arms = [parse_arm(spec) for spec in args.arm]

    result = compare_arms(arms, settings)
    report = format_report(result, settings)
    print(report)

    if args.out_dir and not result.aligned.empty:
        written = write_outputs(result, report, Path(args.out_dir))
        print(f"\nwrote {written.parent}/{written.stem}.{{parquet,txt}}")


if __name__ == "__main__":
    main()
