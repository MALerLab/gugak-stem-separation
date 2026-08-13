"""cross_source_leakage.py — what is each output head actually emitting?

THE QUESTION. A head can score badly on SI-SDR for two very different reasons: it went
DEAD (emits noise or near-silence, containing nothing in particular), or it got REASSIGNED
(emits a different instrument's content, i.e. the outputs effectively permuted). SI-SDR
scores a prediction against its intended target only, so it cannot tell these apart — both
look like "bad". A cross-source table can.

WHAT IT BUILDS. For one checkpoint, a 9x9 table over the Σstem val set:
  · one ROW per model output    (predicted 가야금, predicted 거문고, …)
  · one COLUMN per ground truth (true 가야금, true 거문고, …)
  · cell (i, j) = the fraction of predicted source i's energy explained by true source j

Cell (i, j) is the squared cosine between predicted i and true j — project pred_i onto
true_j, take the projected energy, divide by pred_i's own energy. It reads as a percentage
breakdown of a row: "predicted 가야금 is 70% true 가야금, 20% true 대금, 10% other".

⚠️ ROWS DO NOT SUM TO EXACTLY 1. The true stems are not orthogonal to each other — gugak
ensembles are heterophonic, so instruments genuinely share spectral content — and each
column is projected independently. A row summing well above 1 is itself informative: it
means that output is explained about equally well by several true sources, i.e. it is
generic rather than specific. Read row sums as a diagnostic, not as a bug.

SI-SDR IS DELIBERATELY NOT USED for the cells. It is designed to score a prediction
against its intended target and misbehaves off-target.

Read-only with respect to any training run: it loads a checkpoint and the frozen val
renders, and writes only to its own output directory. Pass a checkpoint COPY if the source
is a `last_*.ckpt` that a live run is still overwriting.

Run:
    uv run python scripts/cross_source_leakage.py \
      --checkpoint experiments/.../checkpoints/model_htdemucs_ep_11_si_sdr_-1.1442.ckpt \
      --experiment-config configs/exp002_htdemucs_v2_uniform_n.yaml \
      --label ep11 \
      --out-dir experiments/.../leakage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repo root (the directory holding pyproject.toml)."""
    current = Path.cwd() if start is None else start
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def load_model(checkpoint_path: Path, experiment_config: Path, device: torch.device,
               inference_batch_size: int):
    """Build the model from the experiment config and load a checkpoint into it.

    Accepts both checkpoint layouts MSST writes: a bare state_dict (our warm-start
    checkpoints) and a training checkpoint dict carrying 'model_state_dict'.

    Args:
        checkpoint_path: .ckpt to evaluate.
        experiment_config: the experiment YAML that defines the architecture.
        device: where to run inference.
        inference_batch_size: overrides the config's inference.batch_size.
    """
    from utils.settings import get_model_from_config

    model, config = get_model_from_config("htdemucs", str(experiment_config))
    config.inference.batch_size = inference_batch_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(
        checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def song_directories(valid_root: Path, extension: str) -> list[Path]:
    """Val song folders, in the same rglob traversal order MSST's valid.py uses."""
    return [p.parent for p in valid_root.rglob(f"mixture.{extension}")]


def energy(signal: np.ndarray) -> float:
    """Total energy of a (channels, time) signal, summed over channels."""
    return float(np.sum(signal.astype(np.float64) ** 2))


def leakage_for_song(predicted: dict[str, np.ndarray], truth: dict[str, np.ndarray],
                     classes: list[str], silence_energy_eps: float) -> tuple[dict, dict]:
    """One song's projection ratios and per-head energy ratios.

    Args:
        predicted: class -> (channels, time), every class (the model always emits all 9).
        truth: class -> (channels, time), only the classes actually present in this song.
        classes: the class list, defining row/column order.
        silence_energy_eps: mean-square floor below which a source counts as silent.

    Returns:
        (ratios, energy_ratios) where ratios[(row, column)] is the fraction of the row
        output's energy explained by that column's true stem, and energy_ratios[c] is the
        predicted-to-true energy ratio in dB for class c.
    """
    ratios: dict[tuple[str, str], float] = {}
    energy_ratios: dict[str, float] = {}

    for row_class in classes:
        prediction = predicted[row_class]
        prediction_energy = energy(prediction)
        # a silent prediction has no energy to apportion; leave its row empty
        if prediction_energy / prediction.size < silence_energy_eps:
            continue

        for column_class, true_signal in truth.items():
            true_energy = energy(true_signal)
            if true_energy / true_signal.size < silence_energy_eps:
                continue
            # squared cosine: the share of the prediction explained by this true stem
            inner = float(np.sum(prediction.astype(np.float64)
                                 * true_signal.astype(np.float64)))
            ratios[(row_class, column_class)] = (inner ** 2 / true_energy
                                                 / prediction_energy)

    # how loud each head is against the stem it is supposed to be producing
    for class_name in classes:
        if class_name not in truth:
            continue
        true_energy = energy(truth[class_name])
        prediction_energy = energy(predicted[class_name])
        if true_energy / truth[class_name].size < silence_energy_eps:
            continue
        energy_ratios[class_name] = 10.0 * np.log10(
            max(prediction_energy, 1e-30) / true_energy)

    return ratios, energy_ratios


def analyze_checkpoint(checkpoint_path: Path, experiment_config: Path, valid_root: Path,
                       settings, device: torch.device,
                       limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the whole val set through one checkpoint and aggregate the leakage table.

    Args:
        checkpoint_path: .ckpt to evaluate.
        experiment_config: experiment YAML defining architecture + class list.
        valid_root: Σstem val root (folders of mixture + per-class renders).
        settings: the leakage config block.
        device: inference device.

    Returns:
        (table, energy_table) — long-form cell means with song counts, and per-class
        predicted-vs-true energy in dB.
    """
    from utils.audio_utils import read_audio_transposed
    from utils.model_utils import demix
    from tqdm import tqdm

    extension = str(settings.extension)
    silence_energy_eps = float(settings.silence_energy_eps)

    model, config = load_model(checkpoint_path, experiment_config, device,
                               int(settings.inference_batch_size))
    classes = list(config.training.instruments)

    ratio_sums: dict[tuple[str, str], float] = {}
    ratio_counts: dict[tuple[str, str], int] = {}
    energy_sums: dict[str, float] = {}
    energy_counts: dict[str, int] = {}

    folders = song_directories(valid_root, extension)
    if limit is not None:
        folders = folders[:limit]
    if not folders:
        raise FileNotFoundError(
            f"no mixture.{extension} found under {valid_root} "
            "(note: rglob does not descend into symlinked directories)")
    for folder in tqdm(folders, desc=checkpoint_path.stem[:40]):
        mixture, _ = read_audio_transposed(str(folder / f"mixture.{extension}"))

        with torch.inference_mode():
            predicted = demix(config, model, mixture.copy(), device,
                              model_type="htdemucs")

        # ground truth: only the classes this song actually contains
        truth: dict[str, np.ndarray] = {}
        for class_name in classes:
            path = folder / f"{class_name}.{extension}"
            if not path.exists():
                continue
            track, _ = read_audio_transposed(str(path), class_name, skip_err=True)
            if track is not None:
                truth[class_name] = track

        # guard: trim everything to a common length before any inner product
        lengths = [mixture.shape[-1]]
        lengths += [v.shape[-1] for v in truth.values()]
        lengths += [v.shape[-1] for v in predicted.values()]
        common_length = min(lengths)
        truth = {k: v[..., :common_length] for k, v in truth.items()}
        predicted = {k: np.asarray(v)[..., :common_length]
                     for k, v in predicted.items()}

        ratios, energy_ratios = leakage_for_song(predicted, truth, classes,
                                                 silence_energy_eps)
        for key, value in ratios.items():
            ratio_sums[key] = ratio_sums.get(key, 0.0) + value
            ratio_counts[key] = ratio_counts.get(key, 0) + 1
        for key, value in energy_ratios.items():
            energy_sums[key] = energy_sums.get(key, 0.0) + value
            energy_counts[key] = energy_counts.get(key, 0) + 1

    table = pd.DataFrame(
        [{"predicted": row, "truth": column,
          "ratio": ratio_sums[(row, column)] / ratio_counts[(row, column)],
          "n_songs": ratio_counts[(row, column)]}
         for (row, column) in sorted(ratio_sums)])
    energy_table = pd.DataFrame(
        [{"predicted": name,
          "energy_ratio_db": energy_sums[name] / energy_counts[name],
          "n_songs": energy_counts[name]}
         for name in sorted(energy_sums)])
    return table, energy_table


def pivot(table: pd.DataFrame, classes: list[str], value: str = "ratio") -> pd.DataFrame:
    """Long-form cells -> a classes x classes matrix in canonical order."""
    wide = table.pivot(index="predicted", columns="truth", values=value)
    return wide.reindex(index=classes, columns=classes)


def render(matrix: pd.DataFrame, title: str, as_percent: bool = True) -> str:
    """Format a matrix as a fixed-width table (rows = predicted, columns = true)."""
    lines = [title, "-" * (12 + 8 * len(matrix.columns))]
    corner = "pred \\ true"
    lines.append(f"{corner:<12}" + "".join(f"{c:>8}" for c in matrix.columns))
    for row_name, row in matrix.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append(f"{'·':>8}")
            elif as_percent:
                cells.append(f"{100 * value:>8.1f}")
            else:
                cells.append(f"{value:>+8.1f}")
        lines.append(f"{row_name:<12}" + "".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-source leakage table for one checkpoint.")
    parser.add_argument("--checkpoint", required=True, help=".ckpt to analyze")
    parser.add_argument("--experiment-config", required=True,
                        help="experiment YAML defining the architecture + classes")
    parser.add_argument("--label", required=True,
                        help="short name for this checkpoint, e.g. 'ep11'")
    parser.add_argument("--out-dir", required=True, help="where to write tables")
    parser.add_argument("--valid-root", default=None,
                        help="Σstem val root (default: from the experiment config)")
    parser.add_argument("--config", default="configs/leakage_analysis.yaml",
                        help="analysis tunables")
    parser.add_argument("--device", default="cuda",
                        help="'cuda', 'cuda:1', or 'cpu'")
    parser.add_argument("--limit", type=int, default=None,
                        help="analyze only the first N val songs (smoke-testing)")
    args = parser.parse_args()

    repo_root = find_repo_root()
    settings = OmegaConf.load(repo_root / args.config).leakage
    experiment_config = Path(args.experiment_config)
    experiment = OmegaConf.load(experiment_config)
    classes = list(experiment.training.instruments)

    valid_root = Path(args.valid_root) if args.valid_root else Path(
        experiment.sumstem_eval.out_root) / "val"
    device = torch.device(args.device)

    table, energy_table = analyze_checkpoint(
        Path(args.checkpoint), experiment_config, valid_root, settings, device,
        limit=args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"leakage_{args.label}.parquet", index=False)
    energy_table.to_parquet(out_dir / f"leakage_energy_{args.label}.parquet", index=False)
    pivot(table, classes).to_csv(out_dir / f"leakage_{args.label}_matrix.csv")

    print()
    print(render(pivot(table, classes),
                 f"CROSS-SOURCE LEAKAGE — {args.label} "
                 f"(% of each ROW's predicted energy explained by each COLUMN's true stem)"))
    print()
    print("predicted-vs-true energy (dB; 0 = same loudness as its target stem)")
    for _, row in energy_table.iterrows():
        print(f"  {row['predicted']:<8} {row['energy_ratio_db']:>+7.1f} dB "
              f"(n={int(row['n_songs'])})")
    print(f"\nwrote {out_dir}/leakage_{args.label}.{{parquet,csv}}")


if __name__ == "__main__":
    main()
