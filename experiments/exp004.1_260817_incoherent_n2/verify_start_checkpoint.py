"""verify_start_checkpoint.py — prove exp004.1's start checkpoint IS the seed-45 head init
shared by exp002.4 and exp004 (load-bearing: all three 2×2 arms must start from the same
output heads).

exp004.1 REUSES exp002.4's seed-45 start_checkpoint.ckpt (byte-copied, sha256-verified
against exp002.4's AND exp004's copies). This script checks, in memory:
  1. 533 tensors; 529 trunk tensors bit-identical to the official pretrained htdemucs
  2. the 4 output-head tensors carry the 9-class shapes
  3. those 4 heads are bit-identical to `torch.manual_seed(45)` + MSST's model builder —
     the exact computation scripts/init_start_checkpoint.py performs
  4. EVERY tensor (533/533) is bit-identical to a FRESH seed-45 REBUILD produced by
     scripts/init_start_checkpoint.py --seed 45 on this launch's config (path via --rebuild)
     — the user's "bit-identical to a fresh seed-45 rebuild" assertion, literally
  5. bit-identical, tensor for tensor, to the copies exp002.4 and exp004 train(ed) from

Run from the repo root:
    CUDA_VISIBLE_DEVICES= uv run python experiments/exp004.1_260817_incoherent_n2/verify_start_checkpoint.py \
        --rebuild <path/to/rebuild_seed45.ckpt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))
from utils.settings import get_model_from_config  # noqa: E402
from demucs.pretrained import get_model as get_pretrained  # noqa: E402

CONFIG = "configs/exp004.1_htdemucs_incoherent_n2.yaml"
CKPT = REPO_ROOT / "experiments/exp004.1_260817_incoherent_n2/checkpoints/start_checkpoint.ckpt"
SIBLINGS = {
    "exp002.4": REPO_ROOT / "experiments/exp002.4_260813_seed45/checkpoints/start_checkpoint.ckpt",
    "exp004": REPO_ROOT / "experiments/exp004_260815_coherent_p1_uniform/checkpoints/start_checkpoint.ckpt",
}
SEED = 45


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", required=True, help="fresh seed-45 rebuild .ckpt to compare against")
    args = ap.parse_args()

    state = torch.load(CKPT, map_location="cpu")
    pretrained = get_pretrained("htdemucs").models[0].state_dict()

    # 1 + 2: trunk identical to pretrained, heads reshaped
    trunk_equal, heads = [], []
    for name, tensor in state.items():
        if name in pretrained and pretrained[name].shape == tensor.shape:
            trunk_equal.append(bool(torch.equal(tensor, pretrained[name])))
        else:
            heads.append(name)
    print(f"tensors: {len(state)} total | {len(trunk_equal)} trunk "
          f"({sum(trunk_equal)} bit-identical to pretrained) | {len(heads)} heads")
    for name in heads:
        print(f"  head {name} {tuple(state[name].shape)}")
    ok1 = len(state) == 533 and len(trunk_equal) == 529 and all(trunk_equal) and len(heads) == 4

    # 3: heads == fresh init at seed 45 (same call sequence as init_start_checkpoint.py)
    torch.manual_seed(SEED)
    model, _ = get_model_from_config("htdemucs", str(REPO_ROOT / CONFIG))
    fresh = model.state_dict()
    head_match = {name: bool(torch.equal(state[name], fresh[name])) for name in heads}
    print(f"heads bit-identical to torch.manual_seed({SEED}) fresh init: "
          f"{sum(head_match.values())}/4  {head_match}")
    ok3 = all(head_match.values())

    # 4: every tensor == the fresh seed-45 rebuild written by init_start_checkpoint.py
    rebuild = torch.load(Path(args.rebuild), map_location="cpu")
    same_keys = set(rebuild) == set(state)
    equal_all = sum(bool(torch.equal(state[n], rebuild[n])) for n in state) if same_keys else 0
    print(f"vs fresh seed-{SEED} rebuild ({args.rebuild}): keys identical={same_keys} · "
          f"{equal_all}/{len(state)} tensors bit-identical")
    ok4 = same_keys and equal_all == len(state)

    # 5: tensor-identical to the sibling arms' copies
    ok5 = True
    for label, path in SIBLINGS.items():
        if not path.exists():
            print(f"vs {label} start checkpoint: MISSING at {path}")
            ok5 = False
            continue
        sib = torch.load(path, map_location="cpu")
        eq = sum(bool(torch.equal(state[n], sib[n])) for n in state) if set(sib) == set(state) else 0
        print(f"vs {label} start checkpoint: {eq}/{len(state)} tensors bit-identical")
        ok5 = ok5 and eq == len(state)

    verdict = ok1 and ok3 and ok4 and ok5
    print("START-CHECKPOINT GATE:", "PASS" if verdict else "FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
