"""diff_run_config.py — compare a new experiment config against a run's LOGGED config.

The house rule for deriving an arm is "diff against what the reference run actually ran,
not against a file that claims to describe it". A config file on disk can drift after
launch (or have been edited by the launcher); the config inside the reference run's wandb
transaction log is the record of what the trainer was really handed. This script reads
that record and diffs it, field by leaf field, against a candidate YAML.

It exists because the comparison has now been run three times by hand — exp002 vs
exp001.2, exp002.1 vs exp002, exp002.2 vs exp002 — and each time the point was to be able
to say "N leaves compared, M differ, here they are" rather than "looks the same to me".

⚠️ The candidate YAML is loaded with **OmegaConf**, the loader MSST itself uses, not
`yaml.safe_load`. YAML 1.1 reads unpunctuated scientific notation (`1e-3`) as a *string*
while OmegaConf reads it as a float, so safe_load manufactures differences that do not
exist (this bit us once already on `htdemucs.dconv_init`).

Run:
    uv run python scripts/diff_run_config.py
      --reference-run wandb/offline-run-20260805_095738-476pciph
      --candidate configs/exp002.2_htdemucs_coherent.yaml
      [--expect gugak_mix.coherent_mix_prob --expect training.run_name]

`--expect` names a field that is ALLOWED to differ. Any differing field not listed, or
any listed field that turns out to be identical, makes the script exit non-zero — so
"only what I meant to change changed" becomes a check rather than a claim.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2 as wandb_pb
from wandb.sdk.internal import datastore

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- reading the reference run's logged config ------------------------------
def read_logged_config(run_dir: Path) -> dict:
    """Pull the training config out of a wandb offline run's transaction log.

    MSST logs the whole config object as a single `config` entry whose value is the
    Python repr of the resolved dict, so it is recovered with `ast.literal_eval` (safe:
    literals only, no code execution).

    Args:
        run_dir: a `wandb/offline-run-*` directory.
    """
    log_files = sorted(run_dir.glob("*.wandb"))
    if not log_files:
        raise FileNotFoundError(f"no .wandb transaction log under {run_dir}")

    store = datastore.DataStore()
    store.open_for_scan(str(log_files[0]))
    while True:
        data = store.scan_data()
        if data is None:
            raise ValueError(f"{log_files[0]}: no run record with a config")
        record = wandb_pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "run":
            continue
        for item in record.run.config.update:
            if item.key == "config":
                return ast.literal_eval(json.loads(item.value_json))


# --- flattening + diffing ---------------------------------------------------
def flatten(node, prefix: str = "") -> dict:
    """Nested mapping → {dotted.path: leaf}. Lists are compared whole, as one leaf."""
    if not isinstance(node, dict):
        return {prefix: node}
    flat: dict = {}
    for key, value in node.items():
        flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def comparable(value):
    """Normalize a leaf so equal values compare equal across the two loaders.

    Lists become tuples (hashable, order-preserving) and numbers are floated, so
    `1e-4` and `0.0001` and `1.0e-04` are one value rather than three.
    """
    if isinstance(value, (list, tuple)):
        return tuple(comparable(v) for v in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff an experiment config against a reference run's logged config.")
    parser.add_argument("--reference-run", required=True,
                        help="wandb/offline-run-* directory of the reference arm")
    parser.add_argument("--candidate", required=True,
                        help="candidate experiment YAML (repo-relative)")
    parser.add_argument("--expect", action="append", default=[],
                        help="dotted field allowed to differ (repeatable)")
    args = parser.parse_args()

    # --- load both sides ---
    reference = flatten(read_logged_config(REPO_ROOT / args.reference_run))
    candidate = flatten(OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / args.candidate), resolve=True))

    # --- compare every leaf on either side ---
    all_fields = sorted(set(reference) | set(candidate))
    missing_marker = object()
    differing = []
    for field in all_fields:
        left = reference.get(field, missing_marker)
        right = candidate.get(field, missing_marker)
        if left is missing_marker or right is missing_marker:
            differing.append((field, left, right))
        elif comparable(left) != comparable(right):
            differing.append((field, left, right))

    # --- report ---
    print(f"reference : {args.reference_run}  (config as logged at launch)")
    print(f"candidate : {args.candidate}")
    print(f"\n{len(all_fields)} leaf fields compared · {len(differing)} differ\n")
    for field, left, right in differing:
        marker = "expected" if field in args.expect else "UNEXPECTED"
        left_text = "<absent>" if left is missing_marker else repr(left)
        right_text = "<absent>" if right is missing_marker else repr(right)
        print(f"  [{marker:10s}] {field}\n      reference: {left_text}"
              f"\n      candidate: {right_text}")
    if not differing:
        print("  (configs are identical)")

    # --- verdict: exactly the declared set changed, nothing more, nothing less ---
    differing_fields = {field for field, _, _ in differing}
    unexpected = sorted(differing_fields - set(args.expect))
    unchanged = sorted(set(args.expect) - differing_fields)
    print()
    if unexpected:
        print(f"❌ FAIL — undeclared differences: {unexpected}")
    if unchanged:
        print(f"❌ FAIL — declared as changed but identical: {unchanged}")
    if not unexpected and not unchanged:
        print(f"✅ PASS — exactly the {len(args.expect)} declared field(s) differ")
    sys.exit(1 if (unexpected or unchanged) else 0)


if __name__ == "__main__":
    main()
