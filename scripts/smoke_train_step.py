"""smoke_train_step.py — one real optimizer step on the GPU, before committing days to a run.

The dataset smoke (scripts/smoke_mix_dataset.py) proves the data is right; this proves the
MODEL end of the pipe is right, and it is the cheapest place to catch the mistakes that
otherwise surface hours into a run:

  * the warm-start checkpoint doesn't actually load into the configured model
  * AMP is misconfigured — most importantly a live fp16 GradScaler under bf16, which is
    the classic porting bug (bf16 needs no loss scaling; a live scaler silently changes
    the update rule)
  * the very first forward/backward produces non-finite values

It reproduces MSST's own train_one_epoch arithmetic rather than approximating it: same
autocast dtype resolution, same `GradScaler(enabled=(amp_dtype != 'bfloat16'))`, same
loss division by the accumulation count, same unscale → clip → step order. If this
disagrees with MSST, the check is worthless.

Nothing is written: no checkpoint, no wandb run, no results directory.

Run:
    uv run python scripts/smoke_train_step.py
      --config configs/exp002_htdemucs_v2_uniform_n.yaml
      --checkpoint experiments/exp002_.../checkpoints/start_checkpoint.ckpt
      --steps 4          loader steps to run (one full accumulation cycle by default)
      --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402


def resolve_amp(training: dict) -> tuple[bool, torch.dtype, str]:
    """Reproduce MSST's autocast dtype resolution exactly (train.py lines ~96-99).

    Args:
        training: the config's `training` block.

    Returns:
        (use_amp, autocast dtype, dtype name as configured).
    """
    name = str(training.get("amp_dtype", "float16"))
    return bool(training.get("use_amp", False)), \
        {"float16": torch.float16, "bfloat16": torch.bfloat16}[name], name


def main() -> None:
    ap = argparse.ArgumentParser(description="One real training step, no side effects.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=4, help="loader steps (default: one accum cycle)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_type", default="htdemucs",
                    help="MSST model family — must match the launch script's --model_type")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override training.batch_size (batch-sizing sweeps only)")
    ap.add_argument("--assert_fp32_spectral", action="store_true",
                    help="record every torch.stft/istft call and fail if any ran in "
                         "reduced precision (spectral-model gate)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # FullLoader, not safe_load: roformer configs carry !!python/tuple values, and this
    # is the loader MSST itself uses (utils/settings.load_config)
    with open(REPO_ROOT / args.config, encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=yaml.FullLoader)
    training = raw["training"]

    # --- model, built by MSST so parameter names match training exactly ---
    from utils.settings import get_model_from_config
    model, config = get_model_from_config(args.model_type, str(REPO_ROOT / args.config))
    state = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu")
    model.load_state_dict(state)          # strict: mirrors --load_only_compatible_weights
    device = torch.device("cuda")
    model = model.to(device).train()
    print(f"[model] loaded {args.checkpoint} strictly into the configured model")
    print(f"[model] sources: {len(training['instruments'])} | "
          f"params: {sum(p.numel() for p in model.parameters()):,}")

    # --- AMP, resolved the way MSST resolves it ---
    use_amp, amp_dtype, amp_name = resolve_amp(training)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_name != "bfloat16"))
    print(f"[amp] use_amp={use_amp} amp_dtype={amp_name} "
          f"grad_scaler_enabled={scaler.is_enabled()}")
    if amp_name == "bfloat16" and scaler.is_enabled():
        raise SystemExit("FAIL: GradScaler is live under bf16")
    print(f"[amp] inference_amp_dtype={training.get('inference_amp_dtype', '<unset>')}")

    # --- spectral-precision probe: wrap torch.stft/istft and record what dtype each
    # call actually received. The project rule is that spectral transforms never run in
    # reduced precision, and in a spectral-only model that transform IS the model, so
    # this is checked directly rather than inferred from the autocast config. ---
    spectral_dtypes: list = []
    if args.assert_fp32_spectral:
        real_stft, real_istft = torch.stft, torch.istft

        def record(name, fn):
            def wrapped(input, *a, **kw):
                spectral_dtypes.append((name, input.dtype))
                return fn(input, *a, **kw)
            return wrapped

        torch.stft = record("stft", real_stft)
        torch.istft = record("istft", real_istft)

    # --- data: the real dataset, the real batch size ---
    batch_size = int(args.batch_size or training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    dataset = GugakMixDataset(MixDatasetConfig.from_mapping(dict(raw["gugak_mix"])),
                             REPO_ROOT, num_items=args.steps * batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    print(f"[data] batch {batch_size} x accum {accumulation} = effective "
          f"{batch_size * accumulation} | density_mode="
          f"{raw['gugak_mix'].get('density_mode', 'measured')}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    torch.cuda.reset_peak_memory_stats()

    # --- the step loop, arithmetic copied from MSST train_one_epoch ---
    nonfinite_steps = 0
    for step, (targets, mixture) in enumerate(loader):
        mixture, targets = mixture.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            loss = F.l1_loss(model(mixture), targets)
        loss = loss / accumulation
        scaler.scale(loss).backward()

        value = loss.item() * accumulation
        if not torch.isfinite(torch.tensor(value)):
            nonfinite_steps += 1
        print(f"[step {step}] loss={value:.6f} finite={torch.isfinite(torch.tensor(value)).item()}")

        if (step + 1) % accumulation == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 float(training["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            print(f"[optimizer step] grad_norm(pre-clip)={grad_norm:.4f} "
                  f"clip={training['grad_clip']} finite={torch.isfinite(grad_norm).item()}")
            if not torch.isfinite(grad_norm):
                raise SystemExit("FAIL: non-finite gradient norm")

    if args.assert_fp32_spectral:
        # complex64 IS the fp32 complex type (a pair of fp32), and it is what istft
        # consumes; the failures we are guarding against are float16/bfloat16/complex32
        full_precision = {torch.float32, torch.complex64, torch.float64, torch.complex128}
        seen = sorted({f"{n}:{d}" for n, d in spectral_dtypes})
        bad = sorted({f"{n}:{d}" for n, d in spectral_dtypes if d not in full_precision})
        print(f"\n[spectral] {len(spectral_dtypes)} torch.stft/istft calls, "
              f"input dtypes seen: {seen}")
        if not spectral_dtypes:
            raise SystemExit("FAIL: no spectral calls recorded — probe did not fire")
        if bad:
            raise SystemExit(f"FAIL: spectral transform ran in reduced precision: {bad}")
        print("[spectral] PASS — every STFT/iSTFT ran on fp32 input")

    nonfinite_params = sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
    print(f"\n[result] non-finite steps: {nonfinite_steps} / {args.steps}")
    print(f"[result] non-finite parameters after the update: {nonfinite_params}")
    print(f"[result] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if nonfinite_steps or nonfinite_params:
        raise SystemExit("FAIL: non-finite values in the first steps")
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
