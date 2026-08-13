"""init_bsroformer_start_checkpoint.py — BS-RoFormer warm-start checkpoint (4 → N stems).

The BS-RoFormer sibling of scripts/init_start_checkpoint.py. A separate script rather
than a branch in that one, because the two surgeries are structurally different:

  HTDemucs has ONE shared decoder and the source count only shows up in the output
  channel count of the final convolution of each branch. The 4 -> 9 change is a shape
  mismatch on two tensors, and "transfer everything name+shape-compatible" handles it
  by itself.

  BS-RoFormer has num_stems SEPARATE mask estimators — `mask_estimators.{i}` — that are
  all the SAME shape. A name+shape transfer would therefore silently succeed on
  estimators 0..3, handing four arbitrary gugak classes the pretrained drums/bass/other/
  vocals heads while the other five start from scratch. Shapes agreeing is exactly why
  this needs its own code: the naive load is not an error, it is a wrong experiment.

POLICY IMPLEMENTED HERE (decided 2026-08-09, exp003.0): transfer the shared trunk, and
freshly initialise ALL N mask estimators. Every output head starts statistically
identical, which is a precondition for the question exp003.0 exists to ask — whether
per-source estimators stop the head collapse seen under HTDemucs's shared decoder. A
4-warm / 5-fresh split would bake a systematic advantage into exactly the axis being
measured. The cost is accepted and stated: the pretrained mask estimators are ~75% of
the checkpoint's parameters, so only the trunk actually transfers.

Output is a PLAIN state_dict: MSST's `--load_only_compatible_weights` path performs a
strict `model.load_state_dict(torch.load(path))`, so training starts from precisely this
tensor set — no tolerant-load ambiguity.

The fresh estimators are the ONE piece of randomness in an otherwise deterministic
transfer, so `--seed` fixes it and the log records it. `--verify-twice` rebuilds the
whole checkpoint in-process a second time and asserts every tensor is bit-identical
(spec gate). Never delete a start_checkpoint.ckpt once a run has used it.

Run:
    uv run python scripts/init_bsroformer_start_checkpoint.py
      --config configs/exp003.0_bsroformer_pilot.yaml
      --pretrained ~/storage/.../model_bs_roformer_ep_17_sdr_9.6568.ckpt
      --out experiments/exp003.0_260809_bsroformer_pilot/checkpoints/start_checkpoint.ckpt
      --log experiments/exp003.0_260809_bsroformer_pilot/checkpoint_init_log.txt
      --seed 42 --verify-twice
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))

# every parameter under this prefix is a per-source mask estimator -> fresh by policy
HEAD_PREFIX = "mask_estimators."


def group_of(param_name: str) -> str:
    """Collapse a parameter name to its reporting group (e.g. 'layers.3', 'band_split')."""
    parts = param_name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]


def build_state_dict(config_path: str, pretrained_path: Path, seed: int) -> tuple[dict, dict]:
    """Build the warm-start tensor set once. Returns (state_dict, provenance).

    Args:
        config_path: repo-relative experiment YAML — defines the N-stem target model.
        pretrained_path: the 4-stem MUSDB checkpoint to transfer the trunk from.
        seed: seeds the fresh initialisation of the N mask estimators.
    """
    # the mask estimators are randomly initialized; seed BEFORE the model is built
    torch.manual_seed(seed)

    # target: the N-stem model, built by MSST itself -> param names match training
    from utils.settings import get_model_from_config
    target_model, config = get_model_from_config("bs_roformer",
                                                 str(REPO_ROOT / config_path))
    target_state = target_model.state_dict()

    # source: the pretrained 4-stem MUSDB checkpoint (a plain state_dict on disk)
    source_state = torch.load(pretrained_path, map_location="cpu", weights_only=True)

    # transfer the trunk; hold every mask estimator back regardless of shape agreement
    transferred, held_back, shape_mismatch, absent_in_source = [], [], [], []
    for name, tensor in target_state.items():
        if name.startswith(HEAD_PREFIX):
            held_back.append(name)
        elif name not in source_state:
            absent_in_source.append(name)
        elif source_state[name].shape != tensor.shape:
            shape_mismatch.append(
                f"{name}  target{tuple(tensor.shape)} vs "
                f"pretrained{tuple(source_state[name].shape)}")
        else:
            target_state[name] = source_state[name].clone()
            transferred.append(name)

    provenance = {
        "config": config_path,
        "pretrained": str(pretrained_path),
        "seed": seed,
        "instruments": list(config.training.instruments),
        "num_stems": int(config.model.num_stems),
        "source_tensors": len(source_state),
        "transferred": transferred,
        "held_back": held_back,
        "shape_mismatch": shape_mismatch,
        "absent_in_source": absent_in_source,
        "target_state": target_state,
    }
    return target_state, provenance


def diff_state_dicts(first: dict, second: dict) -> list[str]:
    """Every tensor that differs between two builds — empty list means bit-identical."""
    differences = []
    if first.keys() != second.keys():
        differences.append(f"key sets differ: {set(first) ^ set(second)}")
        return differences
    for name, tensor in first.items():
        other = second[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            differences.append(f"{name}: {tensor.shape}/{tensor.dtype} vs "
                               f"{other.shape}/{other.dtype}")
        elif not torch.equal(tensor, other):
            differences.append(f"{name}: values differ "
                               f"(max abs diff {(tensor - other).abs().max().item():.3e})")
    return differences


def write_report(provenance: dict, verify_note: str) -> str:
    """The init report — which parameter groups transferred, which started fresh."""
    target_state = provenance["target_state"]
    transferred = provenance["transferred"]
    held_back = provenance["held_back"]

    def summarize(names: list[str]) -> dict:
        counts: dict = defaultdict(int)
        for name in names:
            counts[group_of(name.split("  ")[0])] += 1
        return dict(sorted(counts.items()))

    def params(names: list[str]) -> int:
        return sum(target_state[n].numel() for n in names)

    total_params = sum(v.numel() for v in target_state.values())
    fresh = held_back + provenance["absent_in_source"] + [
        d.split("  ")[0] for d in provenance["shape_mismatch"]]

    lines = [
        "BS-RoFormer warm-start checkpoint — trunk transferred, all mask estimators fresh",
        f"config: {provenance['config']}",
        f"pretrained source: {provenance['pretrained']}",
        f"seed (mask-estimator init): {provenance['seed']}",
        f"num_stems: {provenance['num_stems']}",
        f"instruments ({len(provenance['instruments'])}): {provenance['instruments']}",
        "",
        f"tensors: {len(target_state)} target | {provenance['source_tensors']} pretrained",
        f"  {len(transferred):5d} transferred from pretrained",
        f"  {len(held_back):5d} mask-estimator tensors HELD BACK (fresh by policy, "
        f"not by shape)",
        f"  {len(provenance['shape_mismatch']):5d} shape mismatch outside the heads "
        f"(fresh init)",
        f"  {len(provenance['absent_in_source']):5d} absent in pretrained (fresh init)",
        "",
        f"parameters: {total_params:,} total",
        f"  transferred: {params(transferred):,} "
        f"({100 * params(transferred) / total_params:.1f}%)",
        f"  freshly initialised: {params(fresh):,} "
        f"({100 * params(fresh) / total_params:.1f}%)",
        f"    of which mask estimators: {params(held_back):,} "
        f"({100 * params(held_back) / total_params:.1f}%)",
        "",
        f"determinism check: {verify_note}",
        "", "TRANSFERRED groups (tensor count):",
        *(f"  {g}: {c}" for g, c in summarize(transferred).items()),
        "", "HELD BACK — mask estimators, fresh by policy (tensor count):",
        *(f"  {g}: {c}" for g, c in summarize(held_back).items()),
        "", "SHAPE MISMATCH outside the heads (should be empty):",
        *(f"  {d}" for d in provenance["shape_mismatch"]),
        "", "ABSENT in pretrained (fresh init):",
        *(f"  {n}" for n in provenance["absent_in_source"]),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="BS-RoFormer warm-start checkpoint (4 -> N).")
    ap.add_argument("--config", required=True, help="experiment YAML (defines the model)")
    ap.add_argument("--pretrained", required=True, help="4-stem MUSDB .ckpt state_dict")
    ap.add_argument("--out", required=True, help="output .ckpt path")
    ap.add_argument("--log", required=True, help="init report path")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the fresh init of the per-source mask estimators")
    ap.add_argument("--verify-twice", action="store_true",
                    help="rebuild and assert every tensor is bit-identical (spec gate)")
    args = ap.parse_args()

    pretrained = Path(args.pretrained).expanduser().resolve()
    state, provenance = build_state_dict(args.config, pretrained, args.seed)

    verify_note = "NOT RUN (--verify-twice not passed)"
    if args.verify_twice:
        second, _ = build_state_dict(args.config, pretrained, args.seed)
        differences = diff_state_dicts(state, second)
        if differences:
            print("DETERMINISM CHECK FAILED — two builds differ:", file=sys.stderr)
            for d in differences[:20]:
                print(f"  {d}", file=sys.stderr)
            raise SystemExit(1)
        verify_note = (f"PASS — two independent builds at seed {args.seed} are "
                       f"bit-identical across all {len(state)} tensors")

    report = write_report(provenance, verify_note)
    print(report)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    log_path = REPO_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report + "\n")
    print(f"\nwrote {out_path}\nlog   {log_path}")


if __name__ == "__main__":
    main()
