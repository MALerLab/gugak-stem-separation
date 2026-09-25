"""bench_compile.py — does torch.compile actually speed up this BS-RoFormer step?

WHY THIS EXISTS. The DCGM telemetry from exp003.x says the training step is GPU-bound but
INEFFICIENT: smActive ~87% while pipeTensorActive is ~12% and smOccupancy ~44%. That is the
signature of many small sequential kernels, not of a saturated matmul pipeline — and
BS-RoFormer emits 550+ of them per forward (BandSplit's 62-band Python loop, plus 62 bands
x 9 mask estimators). Kernel fusion is the textbook answer to that shape. But nobody has
measured it on this fork, at sm_120, with the fork's fp32-pinned spectral path, so the
honest thing is to measure before spending days of GPU on a guess.

WHAT IT MEASURES. Median wall-clock of a full forward+backward at the REAL training
geometry (batch 2 x 2ch x 8 s @ 44.1 kHz, bf16 autocast, standard-loss path), for:

    eager          the baseline the 6.85 h/epoch number came from
    compile        torch.compile(model) as the code stands today
    compile+sdpa   same, after routing attention through the modern
                   torch.nn.attention.sdpa_kernel context manager instead of the
                   deprecated torch.backends.cuda.sdp_kernel that attend.py still uses
                   (the old one is a guaranteed graph break at every attention call)

Synthetic tensors are used on purpose: the dataloader was measured to have ~5x headroom
(BS-RoFormer asks 3.4 items/s; the same loader sustained 17+ items/s under HTDemucs), so
including it would only add noise to a model-side question.

NOTHING IS WRITTEN. No checkpoint, no wandb, no results dir. Read-only apart from stdout.

Run (on a free card):
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/bench_compile.py \
        --config configs/exp003.0_bsroformer_pilot.yaml --steps 12 --warmup 5
"""
from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))

from utils.settings import get_model_from_config  # noqa: E402


def patch_attention_to_modern_sdpa() -> int:
    """Route BS-RoFormer's attention through torch.nn.attention.sdpa_kernel.

    attend.py wraps every scaled_dot_product_attention call in
    `torch.backends.cuda.sdp_kernel(...)`, a deprecated context manager that Dynamo
    cannot trace through — it breaks the graph at each of the 16 attention calls, which
    is exactly where fusion would otherwise pay. This swaps in the supported API while
    preserving the fork's fp32 fallback rule (no flash kernel exists for fp32, so a
    non-half dtype must fall back to math / mem-efficient).

    Returns:
        the number of Attend modules patched.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import models.bs_roformer.attend as attend_mod

    def _backends(cfg, dtype: torch.dtype, is_cuda: bool):
        if is_cuda and dtype not in (torch.float16, torch.bfloat16):
            return [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        picked = []
        if cfg.enable_flash:
            picked.append(SDPBackend.FLASH_ATTENTION)
        if cfg.enable_mem_efficient:
            picked.append(SDPBackend.EFFICIENT_ATTENTION)
        if cfg.enable_math:
            picked.append(SDPBackend.MATH)
        return picked or [SDPBackend.MATH]

    def flash_attn(self, q, k, v):
        is_cuda = q.is_cuda
        cfg = self.cuda_config if is_cuda else self.cpu_config
        with sdpa_kernel(_backends(cfg, q.dtype, is_cuda)):
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

    attend_mod.Attend.flash_attn = flash_attn
    return 1


def build(config_path: str, device: torch.device):
    """Construct the configured BS-RoFormer on `device`, in train mode."""
    model, config = get_model_from_config("bs_roformer", config_path)
    model = model.to(device)
    model.train()
    return model, config


def run_variant(name: str, model, x: torch.Tensor, y: torch.Tensor,
                amp_dtype: torch.dtype, steps: int, warmup: int) -> dict:
    """Time `steps` forward+backward passes after `warmup` untimed ones.

    A fresh Adam is used so the optimizer state allocation is inside the measurement the
    same way it is during training. Gradients are zeroed every step: this benchmarks a
    micro-step, which is what accumulation repeats.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    for i in range(warmup + steps):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            out = model(x)
            loss = F.l1_loss(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        tag = "warmup" if i < warmup else "timed "
        print(f"    [{name}] {tag} {i:2d}  {dt:8.3f} s", flush=True)

    med = statistics.median(times)
    return {
        "name": name,
        "batch": int(x.shape[0]),
        "median_s": med,
        "min_s": min(times),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="torch.compile benchmark for BS-RoFormer.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=12, help="timed steps per variant")
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps (compile lands here)")
    ap.add_argument("--variants", default="eager,compile,compile_sdpa")
    ap.add_argument("--batch", type=int, default=None,
                    help="override config batch_size (effective batch is held elsewhere "
                         "by moving gradient_accumulation_steps the other way)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    device = torch.device("cuda:0")
    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"torch        : {torch.__version__}  cuda {torch.version.cuda}")

    _, config = get_model_from_config("bs_roformer", args.config)
    tr = config.training
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        str(getattr(tr, "amp_dtype", "bfloat16"))
    ]
    batch = int(args.batch) if args.batch else int(tr.batch_size)
    chunk = int(config.audio.chunk_size)
    n_stems = int(config.model.num_stems)
    print(f"geometry     : batch {batch} x 2ch x {chunk} samples "
          f"({chunk / config.audio.sample_rate:.1f} s), {n_stems} stems, amp={amp_dtype}")
    print(f"accum        : {getattr(tr, 'gradient_accumulation_steps', 1)} "
          f"(effective batch {batch * int(getattr(tr, 'gradient_accumulation_steps', 1))})")

    torch.manual_seed(args.seed)
    x = torch.randn(batch, 2, chunk, device=device)
    y = torch.randn(batch, n_stems, 2, chunk, device=device)

    results = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"\n=== {variant} ===", flush=True)
        torch.manual_seed(args.seed)

        if variant == "compile_sdpa":
            n = patch_attention_to_modern_sdpa()
            print(f"    patched Attend.flash_attn -> torch.nn.attention.sdpa_kernel ({n})")

        model, _ = build(args.config, device)
        if variant.startswith("compile"):
            t0 = time.perf_counter()
            model = torch.compile(model)
            print(f"    torch.compile() returned in {time.perf_counter() - t0:.2f} s "
                  f"(graph build happens on first call)")
        try:
            results.append(run_variant(variant, model, x, y, amp_dtype, args.steps, args.warmup))
        except Exception as e:  # a failed compile must not hide the eager number
            print(f"    !! {variant} FAILED: {type(e).__name__}: {e}")
            results.append({"name": variant, "batch": batch, "median_s": float("nan"),
                            "min_s": float("nan"), "peak_gib": float("nan")})
        del model
        torch.cuda.empty_cache()

    # An epoch is a fixed number of ITEMS (num_steps x config batch), not of steps, so
    # h/epoch must be derived per item — otherwise a bigger physical batch looks free.
    items_per_epoch = int(getattr(tr, "num_steps", 40000)) * int(tr.batch_size)
    print("\n" + "=" * 80)
    print(f"{'variant':<16}{'batch':>6}{'s/step':>10}{'ms/item':>10}"
          f"{'peak GiB':>10}{'vs eager':>11}{'h/epoch':>10}")
    print("=" * 80)
    base = next((r["median_s"] / r["batch"] for r in results if r["name"] == "eager"), float("nan"))
    for r in results:
        per_item = r["median_s"] / r["batch"]
        speedup = base / per_item if per_item == per_item else float("nan")
        h = per_item * items_per_epoch / 3600
        print(f"{r['name']:<16}{r['batch']:>6}{r['median_s']:>10.4f}{per_item * 1000:>10.1f}"
              f"{r['peak_gib']:>10.1f}{speedup:>10.2f}x{h:>10.2f}")
    print("=" * 80)
    print(f"({items_per_epoch} items/epoch, single GPU, training only — validation excluded)")


if __name__ == "__main__":
    main()
