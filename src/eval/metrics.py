"""metrics.py — the evaluation metric registry: si_sdr, usdr, and the csdr stub.

Both implemented metrics take (reference, estimate) as (channels, samples) float
arrays, pool channels AND time into one scalar per (song, stem class), and return dB.
Neither chunks: one number per whole song. Values are clamped to a saturation cap
(config `metric_cap_db`, default +100 dB) so a perfect reconstruction reports the cap,
never inf — the eps terms already make every value finite, but their eps-driven
saturation level depends on signal energy, so the cap gives "perfect" one defined
number.

si_sdr — scale-invariant SDR (Le Roux et al. 2019), copied VERBATIM from MSST
  (external/msst/utils/metrics.py::si_sdr, eps = 1e-8 including its per-element eps
  placement inside the sums). Verbatim because gate G5 requires reproducing the
  training-time validation numbers exactly; do not "clean up" the eps placement.

usdr — global SDR, the MDX'21 / SDX'23 ranking metric ("uSDR"), per song per stem:

      uSDR = 10 * log10( (Σ_n s(n)^2 + eps) / (Σ_n (s(n) - ŝ(n))^2 + eps) )

  summed over channels and the whole song. eps = 1e-7 (MDX'21 paper default), supplied
  from config. NOT scale-invariant; no gain matching, no filtering, no alignment.

csdr — museval chunked BSSEval v4 SDR. Registered but NOT implemented (raises).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Metric callables built by `build_metrics`: (reference, estimate) -> dB float, where
# both arrays are (channels, samples) and already trimmed to a common length.
MetricFn = Callable[[np.ndarray, np.ndarray], float]

MSST_SI_SDR_EPS = 1e-8   # MSST's internal eps — fixed, NOT the config usdr_eps


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB, bit-matching MSST's implementation.

    Args:
        reference: ground-truth waveform, shape (channels, samples).
        estimate: predicted waveform, same shape.
    """
    eps = MSST_SI_SDR_EPS
    # verbatim MSST math (external/msst/utils/metrics.py): eps is added per element
    # inside the sums, the optimal gain is applied to the REFERENCE, and a final eps
    # sits both on the denominator and outside the ratio
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference ** 2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))
    reference = reference * scale
    value = np.mean(10 * np.log10(
        np.sum(reference ** 2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))
    return float(value)


def usdr(reference: np.ndarray, estimate: np.ndarray, eps: float) -> float:
    """Global SDR ("uSDR", MDX'21) in dB — not scale-invariant, whole-song, no chunking.

    Args:
        reference: ground-truth waveform, shape (channels, samples).
        estimate: predicted waveform, same shape.
        eps: numerator/denominator regularizer (config `usdr_eps`).
    """
    signal_energy = float(np.sum(np.square(reference), dtype=np.float64))
    error_energy = float(np.sum(np.square(reference - estimate), dtype=np.float64))
    return float(10 * np.log10((signal_energy + eps) / (error_energy + eps)))


def csdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Chunked BSSEval v4 SDR — registered stub, deliberately unimplemented."""
    # Intended backend: museval (bss_eval v4, 1 s chunks) for literature-comparable
    # chunked SDR. Not implemented yet and museval is deliberately NOT a dependency.
    raise NotImplementedError(
        "csdr is a registered stub: implement via museval (BSSEval v4) when the "
        "chunked-SDR milestone lands")


def build_metrics(names: list[str], usdr_eps: float, cap_db: float) -> dict[str, MetricFn]:
    """Resolve metric names into ready-to-call, cap-clamped functions.

    Args:
        names: metric names to activate, e.g. ["si_sdr", "usdr"].
        usdr_eps: eps for the usdr formula (from config).
        cap_db: saturation cap in dB applied to every metric's output.
    """
    registry: dict[str, MetricFn] = {
        "si_sdr": si_sdr,
        "usdr": lambda reference, estimate: usdr(reference, estimate, usdr_eps),
        "csdr": csdr,
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise KeyError(f"unknown metrics {unknown}; registry has {sorted(registry)}")

    def capped(fn: MetricFn) -> MetricFn:
        return lambda reference, estimate: min(fn(reference, estimate), cap_db)

    return {name: capped(registry[name]) for name in names}


def rms_dbfs(audio: np.ndarray, floor_dbfs: float = -200.0) -> float:
    """RMS level in dBFS, channels and time pooled; digital silence reports the floor.

    Args:
        audio: waveform, any shape (all elements pooled).
        floor_dbfs: value returned for exact digital zero (log guard only — every real
            post-ingest signal sits at or above the −138 dBFS DC residue).
    """
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if rms <= 10 ** (floor_dbfs / 20):
        return floor_dbfs
    return float(20 * np.log10(rms))
