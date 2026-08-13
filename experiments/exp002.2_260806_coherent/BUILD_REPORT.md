# exp002.2 — coherent mixing · build & verification report

**Built 2026-08-06; launched the same day after review (GPU 2). Stopped by SIGINT
2026-08-09 after 49 epochs** — best −9.11 at ep20, broke at ep21, never recovered.
Results and post-mortem → Notion Experiments page (EXP002.2). This document is the
pre-launch build & verification record.

---

## 1. What this arm is

Every training mix up to and including exp002 has been **incoherent**: 가야금 lifted from
song A, 거문고 from song B, pasted together. Standard practice in Western music source
separation, and it works well there.

Gugak ensemble music is **heterophonic** — the instruments play essentially the same
melody at the same time, differing mainly in idiomatic ornamentation. So an incoherent mix
sets two different melodies, modes and tempi against each other, which is a far easier
separation problem than the real mixtures val and test are built from. The suspicion:
the model has been trained almost entirely on the easy case and evaluated on the hard one,
and has never had to learn unison disentanglement because unison barely occurs in training.
exp001 is consistent with that — 가야금 (the most unison-exposed melodic instrument) and
양금 (a 풍류 doubling voice) were its two worst classes.

exp002.2 trains on **coherent** mixes: one song, one time offset, real ensemble.

---

## 2. What was implemented

`src/data/mix_dataset.py` gains a coherent path behind the previously-reserved
`coherent_mix_prob` knob. `0.0` = the incoherent base recipe, still the default;
`1.0` = exp002.2; values in between mix the two per item, which is the "coherency ratio p"
knob Notion's Training Data Strategy already names.

**The coherent draw, in order:**

1. **Song** — uniform over the 676 train songs.
2. **Window** — the song's stems' active segments are pooled into one list and a segment
   drawn duration-weighted, an anchor taken inside it, and the 10 s window placed
   uniformly over positions containing the anchor. This is the *same arithmetic* as the
   existing per-stem excerpt-start logic (`_draw_window_start` is shared by both paths);
   passing a whole song's segments instead of one stem's is exactly what "evaluated
   jointly across the song's stems" means. Because overlapping segments contribute
   independently, busy passages are drawn more often than solo noodling.
3. **Active set** — a stem counts as audible if its activity covers more than
   `density_coverage_threshold` (0.25) of the window: the project's existing definition,
   the same test `chunk_activities` and the measured density mode use. A class qualifies
   if at least one of its files clears the bar, and only clearing files are drawable.
4. **n** — uniform over `1 .. len(active)`. The `min(9, …)` cap is automatic.
5. **Selection** — n classes without replacement, then one file per class.
6. **Mixture** = sum of the selected stems. Never the publisher master.

**Deliberate design points**

| point | decision | why |
|---|---|---|
| class balance | **not corrected** | class frequency now follows real 편성; correcting it would undo coherence. Measured and reported instead (§5) |
| per-stem random gain | **kept**, 0.25–1.25 | varying balance does not disturb the melodic relationships that make the mix coherent |
| L/R swap | **per mix**, not per stem | swapping a real ensemble's stems independently scrambles its spatial image. ⚠️ deliberate deviation from exp002 |
| pitched percussion | excluded, as everywhere | 편종·편경·방향 quarantine unchanged |
| manifest | `source_manifest_v2` | same file the other two arms read |
| loudness | −19 LUFS, one gain on mixture + all targets | unchanged |
| 판소리 trim-to-shortest | **not applied** | that rule exists to make Σstems line up with the publisher master; no master is involved here, so the tail of a long 가야금 stem is legitimate content. Windows are bounded by the song's *longest* stem; stems that have ended read as zeros |
| 71470 solo pool | structurally excluded | a standalone phrase clip has no song to be coherent *with*. The builder raises rather than silently mis-grouping. Moot for this arm — the solo pool is parked |

**Structural note.** The coherent draw is split into `_plan_coherent` (all randomness, no
disk) and `_coherent_item` (the reads). That is why the statistics in §5 come through the
real sampling code rather than a reimplementation of it — `plan_item()` runs the identical
RNG sequence `__getitem__` runs and stops before decoding audio.

**Nothing else changed.** `chunk_activities` is not read by the coherent path, so its
known v1 scope limit (it still contains 0885's and 0714's windows) does not reach this arm.

---

## 3. Config

`configs/exp002.2_htdemucs_coherent.yaml`, derived from exp002's config **as logged in its
wandb transaction log** (`offline-run-20260805_095738-476pciph`), not from a reconstructed
file. The comparison is now a tracked script, `scripts/diff_run_config.py`, which loads
the candidate through **OmegaConf** — the loader MSST itself uses — because `yaml.safe_load`
reads `1e-3` as a string and manufactures differences that do not exist.

```
$ uv run python scripts/diff_run_config.py \
    --reference-run wandb/offline-run-20260805_095738-476pciph \
    --candidate configs/exp002.2_htdemucs_coherent.yaml \
    --expect gugak_mix.coherent_mix_prob --expect training.run_name

114 leaf fields compared · 2 differ

  [expected  ] gugak_mix.coherent_mix_prob
      reference: 0.0
      candidate: 1.0
  [expected  ] training.run_name
      reference: 'exp002_260805_htdemucs_v2_uniform_n'
      candidate: 'exp002.2_260806_coherent'

✅ PASS — exactly the 2 declared field(s) differ
```

Both differences are the mixing mode or run-scoped. The **L/R-swap scope change carries no
config field** — it is intrinsic to the coherent path, since a per-stem swap would undo
the coherence the arm exists to create. That is the one behavioural change not visible in
the diff, and it is recorded here deliberately.

Everything else matches exp002: bf16, lr 1e-4, EQAug off, 9 classes, 10 s segments,
batch 8 × accum 4, eval every 2,500 steps, Σstem-val, 150k ceiling, manifest v2.

**Seed 42 — same as exp002, deliberately.** ⚠️ This conflicts with the arm-index rule
written in exp002.1's launch report ("leaving 44 for exp002.2"); the brief overrides it and
the reason is sound — exp002.2 is a single-variable comparison against exp002, so seed is
held fixed rather than varied. But note plainly what holding it can and cannot do: a
different sampling algorithm consumes randomness differently, so seed 42 does **not** make
the two arms see matched data. It removes seed as an explanatory variable; exp002.1 is
what measures how much seed alone is worth. Read the three arms as:
exp002.1 − exp002 = run-to-run noise floor; exp002.2 − exp002 = coherence + that noise.

---

## 4. Verification

`scripts/verify_coherent_sampler.py`. Every correctness claim is checked against the
**manifest**, not against the sampler's own bookkeeping — asking the plan whether the plan
is coherent proves nothing. Each drawn file path is joined back to `source_manifest_v2`
and song, split, class and exclusion status read from there.

| check | result |
|---|---|
| all stems from one song at one time offset | ✅ **0 violations in 200,000 mixes** |
| no quarantined (pitched_percussion, voice), excluded (v2 defects) or mis-split source | ✅ 0 quarantine, 0 split |
| n never exceeds the classes audible in the window | ✅ 0 violations |
| mixture ≡ Σ(targets) | ✅ max abs error **2.38e-07** over 3,000 decoded items |
| silence tested by tolerance, never equality | ✅ 17,278 slots exactly zero (never drawn); same count silent within 1e-6; worst sub-threshold residue 0.0 |
| **incoherent path unchanged** | ✅ **64/64 items bit-identical**, hashed before and after the change |

The regression baseline was captured from the pre-edit code and compared afterwards, twice
(once after the feature, once after the plan/execute refactor). At `coherent_mix_prob: 0.0`
the mode decision short-circuits and draws no random number at all, so exp002's stream is
untouched by construction as well as by measurement.

⚠️ **Silence-check caveat, stated honestly.** No drawn stem in the sample was silent — all
17,278 silent slots were slots that were never drawn, which are written as exact zeros. So
the tolerance test *passed* but was never *exercised* by a real 2⁻²³ residue. That is
expected: draws are activity-anchored, so a drawn stem essentially always has content. The
tolerance is correct to use; this run just did not stress it.

⚠️ **`SILENCE_EPS` gap.** `CLAUDE.md` requires the silence tolerance to come from config,
not inline. No config key implements it anywhere in the repo. I put it on the verification
script's own surface (`--silence-eps`, default 1e-6) rather than add a key to the training
config, which would have widened the 2-field diff. **Adding a real shared key is an open
item.**

---

## 5. Characterisation — 200,000 draws, no verdicts

exp002's baseline is computed in **closed form**, not sampled: exp002 draws n ~ U{1..9}
then n classes uniformly without replacement, so P(class) = E[n]/9 and P(pair) =
E[n(n−1)]/72 exactly.

### 5.1 Realised n — leans much sparser

| n | coherent | exp002 | delta |
|---|---|---|---|
| 1 | 0.2445 | 0.1111 | **+0.1334** |
| 2 | 0.2408 | 0.1111 | **+0.1297** |
| 3 | 0.1213 | 0.1111 | +0.0102 |
| 4 | 0.1149 | 0.1111 | +0.0038 |
| 5 | 0.0993 | 0.1111 | −0.0118 |
| 6 | 0.0864 | 0.1111 | −0.0248 |
| 7 | 0.0746 | 0.1111 | −0.0365 |
| 8 | 0.0140 | 0.1111 | **−0.0971** |
| 9 | 0.0041 | 0.1111 | **−0.1070** |

**mean n 3.24 vs 5.00** · mean classes audible per window 5.46 · total-variation distance
**0.277**.

Nearly half of all coherent mixes are 1- or 2-stem, and dense 8–9 stem mixes essentially
vanish (1.8% vs 22.2%). The cause is the data, not the sampler: 160 of the 676 train songs
carry only two classes (산조 = solo instrument + 장구), and only 39 carry all nine. Drawing
n uniformly *within* what a song offers cannot reach a flat 1..9 marginal.

### 5.2 Per-class appearance rate

| class | coherent | exp002 | ratio | sources |
|---|---|---|---|---|
| 타악기 | 0.5847 | 0.5556 | **1.05×** | 819 |
| 대금 | 0.4468 | 0.5556 | 0.80× | 566 |
| 해금 | 0.4266 | 0.5556 | 0.77× | 539 |
| 피리 | 0.4253 | 0.5556 | 0.77× | 568 |
| 가야금 | 0.4112 | 0.5556 | 0.74× | 508 |
| 거문고 | 0.4026 | 0.5556 | 0.72× | 470 |
| 아쟁 | 0.3758 | 0.5556 | 0.68× | 523 |
| 기타 | 0.1100 | 0.5556 | **0.20×** | 185 |
| **양금** | **0.0528** | 0.5556 | **0.10×** | 72 |

타악기 is essentially unaffected (it is in almost every ensemble). The six main melodic
classes lose about a quarter of their exposure. **기타 and 양금 fall off a cliff.**

### 5.3 Co-occurrence — and the finding that complicates the premise

| pair | joint | exp002 | ratio | conditional | exp002 | ratio |
|---|---|---|---|---|---|---|
| P(가야금 \| 양금) | 0.0339 | 0.3704 | **0.09×** | **0.6417** | 0.6667 | **0.96×** |
| P(거문고 \| 가야금) | 0.2288 | 0.3704 | 0.62× | 0.5563 | 0.6667 | 0.83× |
| P(해금 \| 가야금) | 0.2269 | 0.3704 | 0.61× | 0.5518 | 0.6667 | 0.83× |
| P(피리 \| 대금) | 0.2610 | 0.3704 | 0.70× | 0.5841 | 0.6667 | 0.88× |
| P(해금 \| 양금) | 0.0263 | 0.3704 | 0.07× | 0.4973 | 0.6667 | 0.75× |

**This is the number the brief called "close to the point of the whole experiment", and it
does not say what the premise assumed.**

Read the two columns together. *Jointly*, 가야금 and 양금 co-occur **11× less often** under
coherent mixing. But *conditionally* — given 양금 turned up at all — 가야금 is there
**64.2% of the time versus 66.7%, essentially unchanged**. Coherent mixing does not make
가야금 and 양금 meet more often. They already met, constantly, in incoherent mixes; what
they never did was play *together*. The change is entirely in the **quality** of the
pairing, and it is paid for by a **10× cut in how often 양금 is seen at all**.

So for 양금 specifically the arm trades "56% of mixes contain 양금, none in unison" for
"5% of mixes contain 양금, essentially all in unison". Whether that helps the 양금 head is
genuinely open, and could plausibly go the wrong way. For 가야금 the trade is far gentler
(0.74× exposure) and the hypothesis test is clean.

### 5.4 Genre mix

판소리 0.298 · 산조 0.223 · 창작국악 0.205 · 풍류음악 0.115 · 민요 0.083 · 궁중음악 0.052 ·
대풍류 0.024 — tracks the train split's song counts, as expected under uniform song draw.

### 5.5 Effective distinct content

| measure | value |
|---|---|
| distinct source excerpts (identical for both arms) | 957,194 |
| distinct incoherent mixes (exp002) | ~1e44.5 |
| distinct coherent mixes (exp002.2) | ~1e8.3 |
| **distinct musical moments (song × window, 1 s hop)** | **149,573** across 676 songs |
| training examples the run consumes (60 × 10,000 × 8) | 4,800,000 → **32 draws per moment** |

The 1e44 vs 1e8 gap is real but not the number to reason about — both are far beyond what
any run consumes. The meaningful figure is **musical moments**: every coherent mix is a
subset of one, and two mixes from the same moment share all their audio. At a 1 s hop that
is 149,573 moments and 32 draws each; counting genuinely non-overlapping 10 s windows
instead, roughly 15,000 moments and ~320 passes.

---

## 6. Listening material

`data/coherence_examples/` (gitignored, regenerable via
`scripts/render_coherence_examples.py`). Ten matched pairs, one per folder, spread across
**all seven genres**:

- `coherent_mix.wav` — n stems from one song at one offset, plus each stem solo
- `incoherent_mix.wav` — the **same classes**, each pulled from a different random song
  through the real incoherent code path, with the coherent example's own song excluded
- per-class stems for both, and a `README.md` index naming song, genre, offset and the
  songs the incoherent counterpart drew from

Both mixes in a pair go through the identical −19 LUFS normalisation, so they are
level-matched by construction (measured: every pair at −19.0 LUFS) and the only audible
difference is whether the instruments are playing together.

| folder | genre | classes |
|---|---|---|
| 01_궁중음악_0159 | 궁중음악 | 대금 · 피리 · 타악기 |
| 02_대풍류_0415 | 대풍류 | 아쟁 · 타악기 · 대금 · 피리 · 해금 |
| 03_민요_0796 | 민요 | 거문고 · 가야금 · 대금 · 피리 |
| 04_산조_0342 | 산조 | 아쟁 · 타악기 |
| 05_창작국악_0880 | 창작국악 | 8 classes |
| 06_판소리_0646 | 판소리 | 대금 · 타악기 · 해금 · 피리 |
| 07_풍류음악_0065 | 풍류음악 | 가야금 · 대금 · 해금 · 피리 · 타악기 |
| 08_궁중음악_0130 | 궁중음악 | **양금** · 대금 · 가야금 · 해금 · 아쟁 · 타악기 |
| 09_대풍류_0400 | 대풍류 | 타악기 · 대금 · 해금 · 피리 · 아쟁 |
| 10_민요_0793 | 민요 | 7 classes |

**08 is the one to listen to first** — it is the 양금 + 가야금 unison case the whole
experiment is about, in a 궁중음악 ensemble.

---

## 7. Concerns, plainly

**1. The arm is not single-variable, and cannot be made so.** exp002.2 differs from exp002
in coherence *and* in the realised n distribution (mean 3.24 vs 5.00, TV distance 0.277).
This is structural: gugak's real 편성 caps n, and no coherent sampler can produce a flat
1..9 marginal from songs that mostly carry 2 or 7 classes. Two consequences:

- Training loss is **not comparable** between the arms, and less comparable than exp002 vs
  exp001 already was — more sparse mixes means more all-zero targets, which are cheap.
  Notion already flags this for exp002; it is worse here.
- Any per-class SDR gap between the arms is confounded by exposure. 양금 at 0.10× exposure
  cannot be read as a coherence effect.

  *The alternative I did not build*, since the brief specified the song-first order: draw
  n ~ U{1..9} first and reject songs until one supports it. That preserves the n marginal
  but heavily skews song selection toward the 39 nine-class songs and would exclude 산조
  from all but n≤2. I think the brief's choice is the right one — but the confound should
  be stated in the write-up rather than discovered later.

**2. The 양금 result may invert the hypothesis.** §5.3 is the thing I most want a second
opinion on. If the theory is "양금 fails because it is never trained in unison", coherent
mixing supplies unison but removes 90% of its training signal. A worse 양금 in exp002.2
would then be uninterpretable — starvation or coherence, no way to tell. If 양금 is a
primary target of this experiment, it may need its own treatment (a 타악-2×-style
multi-sampling knob for rare classes, or a mixed `coherent_mix_prob` around 0.5 so both
regimes are seen). **Worth deciding before launch.**

**3. Overfitting risk — my read: moderate, and acceptable.** 149,573 musical moments at
~32 draws each, or ~15,000 non-overlapping ones at ~320 passes. Two things make me
comfortable: window starts are drawn at *frame* resolution, so literal tensor repeats
essentially never occur, and each moment yields up to 2^k−1 different class subsets at
different gains. And for calibration, the field trains MUSDB18 models for hundreds of
epochs over **100** songs; we have 676. I would launch this without an overfitting-specific
mitigation, but I would watch the train/val gap more closely than in exp002, and I would
not extend the run past the 150k ceiling on the strength of a still-falling training loss.

**4. Deviations from the brief, stated rather than rounded off:**

- Config knob is `coherent_mix_prob: 1.0`, not `mixing.mode: coherent` — the brief said
  "e.g.", the reserved knob already existed under this name, and Notion calls it "coherency
  ratio p". A mode key alongside it would be two names for one concept. *(Your call,
  confirmed before I wrote it.)*
- Multi-file classes (피리1/피리2 — 164 train songs) contribute **one** file per slot, not
  their sum, to keep parity with exp002's `draw_unit: file`. A coherent mix of a 3-피리
  song therefore hears one 피리: a partial ensemble, not the full one. Summing them is the
  musically truer choice and is what `draw_unit: song_base` is reserved for, but it would
  add a second variable. *(Your call, confirmed before I wrote it.)*
- 판소리 trim-to-shortest not applied — see §2.
- No `SILENCE_EPS` config key added — see §4.

**5. Under-specified in the brief, resolved by me:** what happens when *nothing* clears the
0.25 coverage threshold in the drawn window (possible — the window is anchored on activity
but a short segment can cover under a quarter of 10 s). I fall back to the single
best-covered file, so every mix has at least one audible stem, matching the standing
"skip n=0, keep n=1" rule. Frequency is not separately instrumented; it is bounded above by
the n=1 rate.

---

## 8. State at end of session

- **Not launched.** Nothing committed. Notion not edited.
- **Live arms verified undisturbed:** exp002 PID 61603 on GPU 0 and exp002.1 PID 65008 on
  GPU 1 both alive and logging; `configs/exp002_htdemucs_v2_uniform_n.yaml` (mtime 08-05
  09:40), `configs/exp002.1_htdemucs_twin.yaml` (08-06 07:51) and
  `source_manifest_v2.parquet` (08-05 09:36) all predate this session; 13 and 2 checkpoint
  files respectively, untouched. They import `mix_dataset` from memory and never re-read it
  from disk — and even if they did, the incoherent path is bit-identical.
- **GPU 2 is idle** and is the card this arm would use.

**New files:** `configs/exp002.2_htdemucs_coherent.yaml` ·
`scripts/verify_coherent_sampler.py` · `scripts/diff_run_config.py` ·
`scripts/render_coherence_examples.py` · this report.
**Modified:** `src/data/mix_dataset.py` · `.gitignore`.
**Built after this report, before launch:** `launch.sh` and the seeded
`start_checkpoint.ckpt` (verified bit-identical to exp002's across all 533 tensors).
