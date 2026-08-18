"""verify_start_checkpoint.py — prove exp004's start checkpoint is the seed-45 head init.

exp004 REUSES exp002.4's seed-45 start_checkpoint.ckpt (byte-copied, sha256-verified)
rather than rebuilding it. This script checks, in memory and without writing anything:
  1. 533 tensors; 529 trunk tensors bit-identical to the official pretrained htdemucs
     (the transfer is deterministic, so the trunk is a controlled constant)
  2. the 4 output-head tensors carry the 9-class shapes
  3. those 4 heads are bit-identical to what `torch.manual_seed(45)` + MSST's model
     builder produce — the exact computation scripts/init_start_checkpoint.py performs —
     i.e. the head init IS seed 45, not merely labelled so
  4. (optional) vs the seed-44 start checkpoint (exp002.3): 529 identical, 4/4 differ —
     skipped when that file has been pruned; check 3 is the load-bearing one anyway

Run from the repo root:
    CUDA_VISIBLE_DEVICES= uv run python experiments/exp004_260815_coherent_p1_uniform/verify_start_checkpoint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))
from utils.settings import get_model_from_config  # noqa: E402
from demucs.pretrained import get_model as get_pretrained  # noqa: E402

CONFIG = "configs/exp004_htdemucs_coherent_p1_uniform.yaml"
CKPT = REPO_ROOT / "experiments/exp004_260815_coherent_p1_uniform/checkpoints/start_checkpoint.ckpt"
SEED44 = REPO_ROOT / "experiments/exp002.3_260809_seed44/checkpoints/start_checkpoint.ckpt"
SEED = 45


def main() -> None:
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

    # 4 (optional): independence vs seed 44
    ok4 = True
    if SEED44.exists():
        s44 = torch.load(SEED44, map_location="cpu")
        same = sum(bool(torch.equal(state[n], s44[n])) for n in state if n not in heads)
        diff_heads = sum(not torch.equal(state[n], s44[n]) for n in heads)
        print(f"vs seed-44 start checkpoint: {same}/529 trunk identical, "
              f"{diff_heads}/4 heads differ")
        ok4 = same == 529 and diff_heads == 4
    else:
        print("vs seed-44 start checkpoint: SKIPPED (file pruned since exp002.4's check)")

    verdict = ok1 and ok3 and ok4
    print("START-CHECKPOINT GATE:", "PASS" if verdict else "FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
