"""verify_gates.py — the verification gates for the evaluation pipeline.

Synthetic self-tests with known answers (G1–G4) and a regression against the
training-time pipeline (G5):

  G1  perfect reconstruction     -> both metrics saturate at the documented cap
                                    (config metric_cap_db; eps keeps raw values
                                    finite, the cap makes "perfect" one number)
  G2  estimate = 0.5 × reference -> usdr == 6.02 dB ± 0.01, si_sdr saturates
  G3  estimate = zeros           -> usdr == 0.00 dB ± 0.01
  G4  reference below tolerance  -> ABSENT path: no SDR, pred_energy_dbfs populated
  G5  one real val song's si_sdr reproduces the number frozen in the training
      checkpoint's own eval history (side-by-side report; disagreement = STOP)

(A sixth gate asserting si_sdr >= usdr per row existed during the 2026-08-30 build
and was dropped by decision: the ordering is not mathematically guaranteed and the
model's under-scaled estimates (β < 1) violate it on ~96 % of real rows — see
docs/eval_pipeline_build_report.md, G6 section.)

G1–G4 exercise the REAL scoring path (classify_and_score + build_metrics), not a
re-implementation. Gates print PASS/FAIL and the script exits non-zero on any FAIL.

Run:
    uv run python -m src.eval.verify_gates --config configs/eval/<job>.yaml
        [--song <song_id>] [--skip-g5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.read_msst_checkpoint import (build_per_song, eval_song_order,  # noqa: E402
                                               latest_epoch_key)
from src.eval.metrics import build_metrics, rms_dbfs  # noqa: E402
from src.eval.runner import (EvalJob, classify_and_score, load_eval_job,  # noqa: E402
                             separate_song)

TOLERANCE_DB = 0.01


def report(gate: str, passed: bool, detail: str) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {gate}: {detail}")
    return passed


def run_synthetic_gates(job: EvalJob) -> list[bool]:
    """G1–G4 on seeded synthetic signals, through the real scoring path."""
    rng = np.random.default_rng(job.seed)
    metric_fns = build_metrics(list(job.metrics), job.usdr_eps, job.metric_cap_db)
    reference = rng.standard_normal((2, 10 * 44100)) * 0.1     # ~-20 dBFS stereo noise
    results = []

    # G1: estimate == reference exactly
    row = classify_and_score(reference, reference.copy(), metric_fns,
                             job.absent_rms_dbfs)
    ok = row["si_sdr"] == job.metric_cap_db and row["usdr"] == job.metric_cap_db
    results.append(report(
        "G1 perfect reconstruction", ok,
        f"si_sdr={row['si_sdr']:.2f}, usdr={row['usdr']:.2f} "
        f"(cap {job.metric_cap_db:.1f} dB — raw eps-limited values sit above it)"))

    # G2: estimate = 0.5 × reference — usdr 6.02, si_sdr scale-invariant → cap
    row = classify_and_score(reference, 0.5 * reference, metric_fns,
                             job.absent_rms_dbfs)
    ok = abs(row["usdr"] - 6.0206) <= TOLERANCE_DB and row["si_sdr"] == job.metric_cap_db
    results.append(report(
        "G2 half-scale estimate", ok,
        f"usdr={row['usdr']:.4f} (expect 6.0206±0.01), si_sdr={row['si_sdr']:.2f}"))

    # G3: estimate = zeros, reference nonzero — usdr exactly 0 dB
    row = classify_and_score(reference, np.zeros_like(reference), metric_fns,
                             job.absent_rms_dbfs)
    ok = abs(row["usdr"]) <= TOLERANCE_DB
    results.append(report(
        "G3 zero estimate", ok,
        f"usdr={row['usdr']:.4f} (expect 0.00±0.01); si_sdr={row['si_sdr']:.1f} "
        "(unconstrained here, reported for context)"))

    # G4: reference at the post-ingest DC-residue level → ABSENT, and the
    # missing-file path (reference=None) must land in the same place
    residue = rng.standard_normal((2, 10 * 44100)) * 1.19e-7
    estimate = rng.standard_normal((2, 10 * 44100)) * 0.01
    row_quiet = classify_and_score(residue, estimate, metric_fns, job.absent_rms_dbfs)
    row_nofile = classify_and_score(None, estimate, metric_fns, job.absent_rms_dbfs)
    ok = all(r["status"] == "absent" and np.isnan(r["si_sdr"]) and np.isnan(r["usdr"])
             and np.isfinite(r["pred_energy_dbfs"])
             for r in (row_quiet, row_nofile))
    results.append(report(
        "G4 absent routing", ok,
        f"quiet-file ref_rms={row_quiet['ref_rms_dbfs']:.1f} dBFS < "
        f"{job.absent_rms_dbfs:.0f} → status={row_quiet['status']}, "
        f"pred_energy={row_quiet['pred_energy_dbfs']:.1f} dBFS; "
        f"no-file → status={row_nofile['status']}"))
    return results


def run_regression_gate(job: EvalJob, song_id: str | None) -> bool:
    """G5: our si_sdr on one real val song vs the checkpoint's own frozen number."""
    import torch
    from src.render_audio import load_msst_model

    device = torch.device(job.device)
    model, msst_config, checkpoint = load_msst_model(
        job.model_type, REPO_ROOT / job.model_config, REPO_ROOT / job.checkpoint,
        device)

    model_cfg = yaml.load((REPO_ROOT / job.model_config).read_text(),
                          Loader=yaml.FullLoader)
    eval_cfg = model_cfg["sumstem_eval"]
    valid_root = REPO_ROOT / eval_cfg["out_root"] / "val"
    extension = eval_cfg["extension"]

    if "all_metrics" not in checkpoint:
        print("  FAIL  G5: checkpoint carries no eval history (all_metrics)")
        return False
    history = checkpoint["all_metrics"]
    epoch_key = latest_epoch_key(history)
    song_order = eval_song_order(valid_root)
    frozen = build_per_song(history, epoch_key, song_order, valid_root,
                            extension=extension)
    if song_id is None:
        song_id = song_order[0]        # deterministic default: first scored song
    frozen_song = frozen[frozen.song_id == song_id].set_index("stem_class")["si_sdr"]
    if frozen_song.empty:
        print(f"  FAIL  G5: {song_id} not found in checkpoint history")
        return False

    from src.eval.references import SumstemSong
    song = SumstemSong(song_dir=valid_root / song_id, extension=extension)
    metric_fns = build_metrics(list(job.metrics), job.usdr_eps, job.metric_cap_db)
    separated, _ = separate_song(model, msst_config, device, song.mixture_path,
                                 job.model_type)

    print(f"  G5 regression on {song_id} vs checkpoint {epoch_key}:")
    print(f"    {'stem_class':<12} {'checkpoint':>12} {'pipeline':>12} {'delta':>10}")
    worst = 0.0
    for stem_class in msst_config.training.instruments:
        reference = song.reference(stem_class)
        if reference is None:
            continue
        row = classify_and_score(reference, np.asarray(separated[stem_class]),
                                 metric_fns, job.absent_rms_dbfs)
        frozen_value = float(frozen_song[stem_class])
        delta = row["si_sdr"] - frozen_value
        worst = max(worst, abs(delta))
        print(f"    {stem_class:<12} {frozen_value:>12.4f} {row['si_sdr']:>12.4f} "
              f"{delta:>+10.4f}")
    ok = worst <= TOLERANCE_DB
    verdict = "PASS" if ok else "FAIL — STOP: do not adjust either side, investigate"
    print(f"  {verdict}  G5: max |delta| = {worst:.4f} dB (tolerance {TOLERANCE_DB})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval-pipeline gates.")
    parser.add_argument("--config", required=True, help="eval-job YAML")
    parser.add_argument("--song", default=None,
                        help="val song for G5 (default: first in traversal order)")
    parser.add_argument("--skip-g5", action="store_true",
                        help="skip the GPU regression gate")
    args = parser.parse_args()

    job = load_eval_job(REPO_ROOT / args.config)
    print(f"gates for {job.run_id} (cap {job.metric_cap_db} dB, usdr_eps "
          f"{job.usdr_eps}, absent < {job.absent_rms_dbfs} dBFS RMS):")

    results = run_synthetic_gates(job)
    if not args.skip_g5:
        results.append(run_regression_gate(job, args.song))
    else:
        print("  SKIP  G5 (--skip-g5)")

    if not all(results):
        sys.exit(1)
    print("all executed gates passed")


if __name__ == "__main__":
    main()
