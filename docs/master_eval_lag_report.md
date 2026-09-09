# Master-input test scores, lag-corrected (2026-09-05)

## Lag finding
- In 19 of 135 test songs the publisher master is offset from its own stems by a fixed integer number of samples, with no duration difference: 59 samples (1.34 ms) in all 3 대풍류 songs and 9 판소리 songs; 338 samples (7.7 ms) in 4 판소리 songs; −2138 samples (−48 ms) in 2 창작국악 songs; 41 in one 창작국악 song. Consistent with a latency bug in the publisher's mastering chain.
- Effect: SI-SDR/uSDR compare sample-by-sample, so a 1.34 ms shift collapses the score to about −16 dB regardless of separation quality (master–Σstem correlation ≈ 0.1 at zero lag vs ≈ 0.9 at the true lag). The Σstem variant is unaffected: its mixture is the sum of the reference files by construction.
- Fix: measured per song by full-song cross-correlation (`scripts/master_lag_scan.py` → `manifests/parquet/master_lag.parquet`); the master eval now reads the master from its lag-th sample (or the stems from theirs for a negative lag) before inference. No change to metrics, silence handling, or aggregation.
- Previously reported exp003.0 master SI-SDR of **4.89** was unaligned; aligned it is **7.59**.

## Table 1 — test SI-SDR (dB), class-balanced mean of per-class means, absent classes excluded
| run | Σstem | master (aligned) | drop |
|---|---|---|---|
| exp002.4 HTDemucs incoherent n≥1 | 4.91 | 2.77 | −2.14 |
| exp004.1 HTDemucs incoherent n≥2 | 5.27 | 2.81 | −2.46 |
| exp004 HTDemucs coherent p1 | 6.77 | 4.12 | −2.66 |
| exp003.0 BS-RoFormer | 10.97 | 7.59 | −3.38 |

Coherence gap (exp004 − exp004.1): +1.50 dB on Σstem, +1.31 dB on master.

## Per-genre drop, master − Σstem (dB), pooled over present (song, class) pairs
| genre | songs | exp004 | exp003.0 |
|---|---|---|---|
| 민요 | 11 | −0.17 | −0.35 |
| 궁중음악 | 7 | −1.22 | −2.05 |
| 판소리 | 40 | −1.37 | −2.00 |
| 풍류음악 | 16 | −1.78 | −2.14 |
| 창작국악 | 28 | −2.49 | −4.10 |
| 대풍류 | 3 | −3.86 | −6.00 |
| 산조 | 30 | −6.50 | −11.60 |

Genre ordering is identical for both models. 산조's large drop is a ceiling effect: it scores 22 dB on Σstem (a two-source problem) while its master sits ≈10 dB from the stem sum; its absolute master score (10.6) is still second-highest.

## Outputs
`experiments/<run>/eval/eval_<run>_test_master_aligned{,_summary,_per_stem,_per_genre,_absent}.parquet` for exp002.4 ep59, exp004.1 ep57, exp004 ep58, exp003.0 ep59. Uncommitted.
