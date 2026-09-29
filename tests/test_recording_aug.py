"""test_recording_aug.py — the recording-condition augmentation's safety net.

Five properties, in descending order of how expensive it would be to get them wrong:

  1. OFF IS BYTE-FOR-BYTE OFF. Every config that predates the feature must produce the
     identical tensors. This is checked against the REAL pre-change module, extracted from
     git at a pinned commit and imported side by side, not against a remembered value or a
     hand-copied snapshot — so it cannot rot into a tautology.
  2. THE SUM INVARIANT. max|mixture − Σ targets| stays under the project's silence
     tolerance for every stage and combination the report says is sum-preserving, and the
     two modes that deliberately break it are unreachable without naming them.
  3. DETERMINISM. One (seed, index) gives one item, whatever the worker count.
  4. TARGET COHERENCE FOR THE ROOM STAGE. In source_image mode the targets really are
     h * s_i and the mixture really is their sum — proved by convolving the dry targets
     from the twin dereverberation arm with the planned kernel.
  5. THE LIMITER'S CONTRACT. It never overshoots its threshold and it is exactly
     transparent below it, because the mix-bus target allocation divides by its output.

Run:
    uv run pytest tests/test_recording_aug.py -v
CPU cost lives in scripts/bench_recording_aug.py — measured, never asserted.
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data import recording_aug                                       # noqa: E402
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig       # noqa: E402

# The commit this feature was written on top of: "[exp] add exp006 — BS-RoFormer on
# partially coherent mixtures". The bit-identity test imports src/data/mix_dataset.py AS OF
# THIS COMMIT and diffs the tensors. If you rebase the branch, repin this to whatever the
# parent of the recording_aug commit becomes — and the guard below will shout if you
# accidentally point it at a commit that already has the feature.
BASELINE_REF = "54ccba4"
NO_FILTER = ["-c", "filter.nbstripout.clean=cat", "-c", "filter.nbstripout.smudge=cat"]

# configs that must not move a bit. exp004 is htdemucs at p = 1.0, so the fully coherent
# cluster path is covered as well as exp003.0's pure-incoherent and exp006's p = 0.5.
FROZEN_CONFIGS = ("configs/exp003.0_bsroformer_pilot.yaml",
                  "configs/exp004_htdemucs_coherent_p1_uniform.yaml",
                  "configs/exp006_bsroformer_partial_coherent.yaml")
BASE_CONFIG = "configs/exp006_bsroformer_partial_coherent.yaml"
SILENCE_EPS = 1e-6          # CLAUDE.md: the ±1-LSB DC-removal residue is 1.192e-07
RIR_MANIFEST = REPO_ROOT / "manifests/parquet/rir_pool_v1.parquet"


# --- fixtures ----------------------------------------------------------------
def load_block(config_path: str) -> dict:
    """The `gugak_mix` block of an experiment YAML (FullLoader: !!python/tuple values)."""
    with open(REPO_ROOT / config_path, encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)["gugak_mix"]


def build(block: dict, num_items: int = 32) -> GugakMixDataset:
    return GugakMixDataset(MixDatasetConfig.from_mapping(block), REPO_ROOT, num_items)


def with_recording(**overrides) -> dict:
    """exp006's block plus a recording_aug block assembled from `overrides`."""
    block = copy.deepcopy(load_block(BASE_CONFIG))
    recording = {"enable": True, "assert_invariant": False}
    recording.update(overrides)
    block["recording_aug"] = recording
    return block


def needs_bank() -> None:
    if not RIR_MANIFEST.exists():
        pytest.skip(f"no RIR bank at {RIR_MANIFEST} — build it with "
                    "scripts/build_rir_pool.py")


@pytest.fixture(scope="session")
def baseline_module(tmp_path_factory):
    """src/data/mix_dataset.py as of BASELINE_REF, imported under its own module name."""
    source = subprocess.run(["git", *NO_FILTER, "show",
                             f"{BASELINE_REF}:src/data/mix_dataset.py"],
                            cwd=REPO_ROOT, capture_output=True, text=True,
                            check=True).stdout
    assert "recording_aug" not in source, (
        f"BASELINE_REF={BASELINE_REF} already contains recording_aug — the bit-identity "
        "test would be comparing the new code against itself. Repin it to the commit "
        "BEFORE the feature landed.")
    path = tmp_path_factory.mktemp("baseline") / "baseline_mix_dataset.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("baseline_mix_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_mix_dataset"] = module
    spec.loader.exec_module(module)
    return module


# --- 1. off by default -------------------------------------------------------
@pytest.mark.parametrize("config_path", FROZEN_CONFIGS)
def test_off_by_default_is_bit_identical(baseline_module, config_path):
    """Existing configs must produce BYTE-IDENTICAL tensors, not merely close ones.

    Bytes, not np.allclose: the point of the guarantee is that a run launched from an old
    config sees the exact same stream it would have seen before, so a 1-ULP difference is
    a failure, not a rounding detail.
    """
    block = load_block(config_path)
    assert "recording_aug" not in block, f"{config_path} must not set recording_aug"
    old = baseline_module.GugakMixDataset(
        baseline_module.MixDatasetConfig.from_mapping(copy.deepcopy(block)),
        REPO_ROOT, 16)
    new = build(copy.deepcopy(block), 16)
    for index in range(16):
        old_stems, old_mixture = old[index]
        new_stems, new_mixture = new[index]
        assert old_stems.numpy().tobytes() == new_stems.numpy().tobytes(), \
            f"{config_path} item {index}: stems differ"
        assert old_mixture.numpy().tobytes() == new_mixture.numpy().tobytes(), \
            f"{config_path} item {index}: mixture differs"


def test_off_by_default_draws_no_randomness(baseline_module):
    """Beyond the tensors: the PLAN must match too, so the RNG stream really is untouched.

    A same-tensor/different-stream failure is possible in principle (a draw consumed and
    then not used), and it would only surface later as an unreproducible curriculum arm.
    """
    block = load_block(BASE_CONFIG)
    old = baseline_module.GugakMixDataset(
        baseline_module.MixDatasetConfig.from_mapping(copy.deepcopy(block)), REPO_ROOT, 64)
    new = build(copy.deepcopy(block), 64)
    for index in range(64):
        old_plan, new_plan = old.plan_item(index), new.plan_item(index)
        assert new_plan.recording is None
        assert (old_plan.n, old_plan.drawn_slots, old_plan.k_declared,
                old_plan.k_realised, old_plan.shortfall_reasons) == \
               (new_plan.n, new_plan.drawn_slots, new_plan.k_declared,
                new_plan.k_realised, new_plan.shortfall_reasons)
        assert [(p.slot, p.entry.out_path, p.start_frame, p.gain, p.swap_channels)
                for p in old_plan.picks] == \
               [(p.slot, p.entry.out_path, p.start_frame, p.gain, p.swap_channels)
                for p in new_plan.picks]


def test_enabled_but_all_probabilities_zero_is_still_identical(baseline_module):
    """`enable: true` with every stage at 0 must ALSO cost nothing.

    This is the property that lets an experiment matrix turn stages on one at a time
    without the arms silently desynchronising through an unused Bernoulli draw.
    """
    block = load_block(BASE_CONFIG)
    old = baseline_module.GugakMixDataset(
        baseline_module.MixDatasetConfig.from_mapping(copy.deepcopy(block)), REPO_ROOT, 8)
    new = build(with_recording(), 8)
    for index in range(8):
        old_stems, old_mixture = old[index]
        new_stems, new_mixture = new[index]
        assert old_stems.numpy().tobytes() == new_stems.numpy().tobytes()
        assert old_mixture.numpy().tobytes() == new_mixture.numpy().tobytes()


# --- 2. the sum invariant ----------------------------------------------------
def sum_error(dataset: GugakMixDataset, count: int = 12) -> float:
    worst = 0.0
    for index in range(count):
        stems, mixture = dataset[index]
        worst = max(worst, float((mixture - stems.sum(axis=0)).abs().max()))
    return worst


SUM_PRESERVING = {
    "eq_only": dict(block=dict(eq_stem_prob=0.5, eq_mixbus_prob=0.5), recording=None),
    "device": dict(block={}, recording=dict(device_response=dict(prob=1.0))),
    "room_per_mixture": dict(block={}, recording=dict(rir=dict(prob=1.0))),
    "room_per_cluster": dict(block={}, recording=dict(
        rir=dict(prob=1.0, scope="per_cluster", position_scope="per_cluster"))),
    "room_per_stem": dict(block={}, recording=dict(
        rir=dict(prob=1.0, scope="per_stem", position_scope="per_stem"))),
    "automation_shared": dict(block={}, recording=dict(
        level_automation=dict(mixbus_prob=1.0))),
    "automation_per_stem": dict(block={}, recording=dict(
        level_automation=dict(mixbus_prob=1.0, stem_prob=1.0))),
    "limit_aug": dict(block={}, recording=dict(
        limit_aug=dict(prob=1.0, loudness_mode="sampled", normalize_after_bus=False))),
    "limit_aug_no_residual": dict(block={}, recording=dict(
        limit_aug=dict(prob=1.0, loudness_mode="sampled", normalize_after_bus=False,
                       residual_correction=False))),
    "everything_linear_plus_bus": dict(block=dict(eq_stem_prob=0.5, eq_mixbus_prob=0.5),
                                       recording=dict(
        device_response=dict(prob=1.0), rir=dict(prob=1.0),
        level_automation=dict(mixbus_prob=1.0, stem_prob=0.3),
        limit_aug=dict(prob=1.0, loudness_mode="sampled", normalize_after_bus=False))),
}


@pytest.mark.parametrize("name", sorted(SUM_PRESERVING))
def test_sum_invariant(name):
    """Report §4: every linear stage and the ratio-allocated bus keep mixture ≡ Σ targets."""
    spec = SUM_PRESERVING[name]
    if spec["recording"] is not None and "rir" in spec["recording"]:
        needs_bank()
    block = copy.deepcopy(load_block(BASE_CONFIG))
    block.update(spec["block"])
    if spec["recording"] is not None:
        block["recording_aug"] = {"enable": True, **spec["recording"]}
    error = sum_error(build(block))
    assert error < SILENCE_EPS, f"{name}: max|mixture - Σstems| = {error:.3e}"


def test_assert_invariant_flag_is_wired():
    """`assert_invariant: true` must actually run the check, not just parse."""
    needs_bank()
    dataset = build(with_recording(assert_invariant=True, rir=dict(prob=1.0),
                                   device_response=dict(prob=1.0),
                                   level_automation=dict(mixbus_prob=1.0)))
    for index in range(6):
        dataset[index]          # raises AssertionError inside _realise_recording if broken


def test_dirty_input_breaks_the_sum_on_purpose():
    """Stage 5 is supposed to violate the invariant — confirm it is the mixture that moved.

    Exercises resample (soxr, not librosa — the lab convention), quantize and clip
    together, so the whole implemented degradation chain runs, not just the arithmetic
    ones. The targets must be untouched: it is separation PLUS restoration, by definition.
    """
    block = with_recording(coherence_mode="clean_under_degraded_input",
                           dirty_input=dict(resample_prob=1.0, intermediate_rates=[16000],
                                            quantize_prob=1.0, bit_depths=[8],
                                            clip_prob=1.0, clip_dbfs=[-6.0, -6.0]))
    dirty = build(block)
    clean = build(load_block(BASE_CONFIG))
    for index in range(4):
        dirty_stems, dirty_mixture = dirty[index]
        clean_stems, _ = clean[index]
        residual = float((dirty_mixture - dirty_stems.sum(axis=0)).abs().max())
        assert residual > SILENCE_EPS, ("dirty_input left the mixture untouched — the "
                                        "degradation did not reach the output")
        # not bytes: the recording path sums only the DRAWN slots while the untouched path
        # sums all nine (the rest are exact zeros), and numpy reassociates a 5-term
        # reduction differently from a 9-term one. The difference is last-bit, and that is
        # the claim being checked — the degradation did not reach the targets.
        assert np.abs(dirty_stems.numpy() - clean_stems.numpy()).max() < SILENCE_EPS, \
            "dirty_input must degrade the MIXTURE ONLY; the targets moved"
        assert float(dirty_mixture.abs().max()) <= 10 ** (-6.0 / 20.0) + 1e-6


# --- 3. determinism ----------------------------------------------------------
@pytest.mark.parametrize("workers", [0, 2, 4])
def test_determinism_across_worker_counts(workers):
    """Same (seed, index) → same audio, whatever the DataLoader does with processes."""
    needs_bank()
    block = with_recording(rir=dict(prob=0.7), device_response=dict(prob=0.7),
                           level_automation=dict(mixbus_prob=0.5),
                           limit_aug=dict(prob=0.5, loudness_mode="sampled",
                                          normalize_after_bus=False))
    dataset = build(block, num_items=8)
    reference = [dataset[index] for index in range(8)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=workers)
    for index, (stems, mixture) in enumerate(loader):
        assert stems[0].numpy().tobytes() == reference[index][0].numpy().tobytes(), \
            f"workers={workers} item {index}: stems differ"
        assert mixture[0].numpy().tobytes() == reference[index][1].numpy().tobytes(), \
            f"workers={workers} item {index}: mixture differs"


def test_plan_is_reproducible_and_index_dependent():
    needs_bank()
    dataset = build(with_recording(rir=dict(prob=1.0), device_response=dict(prob=1.0)))
    first = dataset.plan_item(3)
    assert first.recording.rir.room_id == dataset.plan_item(3).recording.rir.room_id
    rooms = {dataset.plan_item(index).recording.rir.room_id for index in range(40)}
    assert len(rooms) > 1, "every item drew the same room — the bank is not being sampled"


# --- 4. target coherence for the room stage ----------------------------------
def test_rir_image_targets_are_h_convolved_with_s():
    """source_image mode: target_i == h * s_i, and the mixture == Σ targets.

    The dry stems come from the TWIN dereverberation arm, which differs from the image arm
    only in which array is written to the output — the RNG stream, the room draw and the
    mixture (and therefore the normalisation gain) are identical by construction. So
    convolving the dry arm's targets with the planned kernel must reproduce the image
    arm's targets exactly, up to FFT rounding.
    """
    needs_bank()
    import scipy.signal
    common = dict(prob=1.0, scope="per_mixture", position_scope="shared")
    image = build(with_recording(rir=dict(**common, target_mode="source_image")), 8)
    dry = build(with_recording(coherence_mode="dereverberation",
                               rir=dict(**common, target_mode="dry_source")), 8)
    for index in range(4):
        plan = image.plan_item(index)
        assert plan.recording.rir is not None
        assert len(plan.recording.rir.assignments) == 1, \
            "per_mixture + shared must produce exactly ONE kernel for the whole mixture"
        kernel = image.rir_pool.kernel(plan.recording.rir.assignments[0].row)

        wet_stems, wet_mixture = (t.numpy() for t in image[index])
        dry_stems, dry_mixture = (t.numpy() for t in dry[index])
        # the two arms must agree on the MIXTURE: only the target view differs
        assert np.abs(wet_mixture - dry_mixture).max() < 1e-6

        slots = [slot for slot in range(dry_stems.shape[0])
                 if np.abs(dry_stems[slot]).max() > 0]
        assert slots
        for slot in slots:
            expected = np.stack([
                scipy.signal.fftconvolve(dry_stems[slot, channel], kernel[channel])
                [:wet_stems.shape[-1]] for channel in (0, 1)])
            error = np.abs(expected - wet_stems[slot]).max()
            scale = max(np.abs(wet_stems[slot]).max(), 1e-12)
            assert error / scale < 1e-4, (f"slot {slot}: target is not h*s "
                                          f"(rel error {error / scale:.2e})")
        assert np.abs(wet_mixture - wet_stems.sum(axis=0)).max() < SILENCE_EPS


def test_per_mixture_scope_shares_one_room_and_per_stem_does_not():
    """The default must hand every stem the SAME response; the ablation must not."""
    needs_bank()
    shared = build(with_recording(rir=dict(prob=1.0)), 8)
    split = build(with_recording(rir=dict(prob=1.0, scope="per_stem",
                                          position_scope="per_stem")), 8)
    for index in range(6):
        assignments = shared.plan_item(index).recording.rir.assignments
        assert len(assignments) == 1
    varied = max(len(split.plan_item(index).recording.rir.assignments)
                 for index in range(6))
    assert varied > 1, "per_stem scope did not produce more than one response"


def lr_correlation(mixture: np.ndarray) -> float:
    left, right = mixture[0].astype(np.float64), mixture[1].astype(np.float64)
    denom = np.sqrt((left ** 2).sum() * (right ** 2).sum())
    return float((left * right).sum() / denom) if denom > 0 else np.nan


def test_the_room_actually_decorrelates_the_stereo_pair():
    """The bank must be a TWO-MICROPHONE room, not one response used twice.

    Measured on the ingested store, per-class L/R correlation is .96–.99 for seven of the
    nine classes: the stems are close-miked with essentially no diffuse field. Convolving
    both channels with the SAME response would leave that correlation where it is and give
    a "mono room", which is not what a distant stereo capture sounds like — its late field
    is decorrelated. So each bank entry is a stereo pair of responses from one room and the
    loader convolves L with h_L and R with h_R. This is the check that it really happens;
    a regression to a mono kernel would sail past every other test in this file.
    """
    needs_bank()
    dry = build(load_block(BASE_CONFIG), 16)
    wet = build(with_recording(rir=dict(prob=1.0)), 16)
    dry_corr = np.nanmedian([lr_correlation(dry[i][1].numpy()) for i in range(16)])
    wet_corr = np.nanmedian([lr_correlation(wet[i][1].numpy()) for i in range(16)])
    assert wet_corr < dry_corr - 0.2, (
        f"the room left the stereo image alone: median L/R correlation {dry_corr:.3f} "
        f"dry vs {wet_corr:.3f} wet — is the bank mono?")


def test_distance_is_sampled_and_logged():
    """Distance/DRR must vary across mixtures — it is the parameter the experiment sweeps."""
    needs_bank()
    dataset = build(with_recording(rir=dict(prob=1.0)), 64)
    distances = [dataset.plan_item(index).recording.rir.assignments[0].distance_m
                 for index in range(64)]
    drr = [dataset.plan_item(index).recording.rir.assignments[0].drr_db
           for index in range(64)]
    assert max(distances) - min(distances) > 1.0, f"distance barely moved: {distances[:8]}"
    assert max(drr) - min(drr) > 5.0, f"DRR barely moved: {drr[:8]}"


# --- 5. the limiter's contract ----------------------------------------------
@pytest.mark.parametrize("release_ms", [30.0, 200.0])
def test_limiter_never_overshoots(release_ms):
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal((2, 44100)) * 0.4).astype(np.float32)
    signal[:, 20000:20200] *= 8.0                       # a transient well over the ceiling
    for threshold_dbfs in (0.0, -6.0):
        gain = recording_aug.limiter_gain(signal, threshold_dbfs, 1.5, release_ms, 44100)
        peak = float(np.abs(signal * gain).max())
        ceiling = 10.0 ** (threshold_dbfs / 20.0)
        assert peak <= ceiling + 1e-6, f"overshoot {peak:.6f} > {ceiling:.6f}"


def test_limiter_is_exactly_transparent_below_threshold():
    """The target allocation divides by the limiter's output, so unity below threshold
    must be EXACT unity — a 0.1% error would show up as a 0.1% error on every target."""
    rng = np.random.default_rng(1)
    quiet = (rng.standard_normal((2, 44100)) * 0.01).astype(np.float32)
    gain = recording_aug.limiter_gain(quiet, 0.0, 1.5, 100.0, 44100)
    assert np.array_equal(gain, np.ones_like(gain))


def test_limit_aug_sampled_loudness_is_not_flattened():
    """The whole point of LimitAug is loudness DIVERSITY; normalize_after_bus must kill it
    and leaving it off must preserve it (report §5.1's normalization-order trap)."""
    import pyloudnorm
    meter = pyloudnorm.Meter(44100)

    def spread(normalize_after_bus: bool) -> float:
        dataset = build(with_recording(limit_aug=dict(
            prob=1.0, loudness_mode="sampled", target_lufs_mean=-14.0,
            target_lufs_std=2.0, normalize_after_bus=normalize_after_bus)), 16)
        values = [meter.integrated_loudness(dataset[i][1].numpy().T) for i in range(16)]
        values = [v for v in values if np.isfinite(v)]
        return float(np.std(values))

    assert spread(normalize_after_bus=False) > 1.0
    assert spread(normalize_after_bus=True) < 0.5


# --- configuration guards ----------------------------------------------------
def test_dry_targets_need_the_dereverberation_mode():
    with pytest.raises(ValueError, match="dry_source"):
        build(with_recording(rir=dict(prob=1.0, target_mode="dry_source")))


def test_dirty_input_needs_its_own_coherence_mode():
    with pytest.raises(ValueError, match="dirty_input"):
        build(with_recording(dirty_input=dict(quantize_prob=0.5)))


def test_codec_is_a_reserved_seam():
    with pytest.raises(NotImplementedError, match="codec"):
        build(with_recording(coherence_mode="clean_under_degraded_input",
                             dirty_input=dict(codec_prob=0.01)))


def test_unknown_keys_fail_loudly():
    with pytest.raises(KeyError, match="recording_aug.rir"):
        build(with_recording(rir=dict(prob=1.0, rt60_bandz_s=[[0.1, 0.2]])))
    with pytest.raises(KeyError, match="recording_aug"):
        build(with_recording(reverb=dict(prob=1.0)))


def test_missing_rir_bank_is_an_error_not_a_no_op():
    with pytest.raises(FileNotFoundError, match="build_rir_pool"):
        build(with_recording(rir=dict(prob=0.3,
                                      manifest="manifests/parquet/does_not_exist.parquet")))


def test_noise_without_a_corpus_is_an_error():
    with pytest.raises(FileNotFoundError, match="noise_manifest"):
        build(with_recording(coherence_mode="clean_under_degraded_input",
                             dirty_input=dict(noise_prob=0.5)))
