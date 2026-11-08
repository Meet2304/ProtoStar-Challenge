# Plan A: graph-based spatio-temporal learning for the Placeholder series (v1.x)

This plan treats the 210 cities as nodes of a fixed geographic graph. It forecasts the part of each city's recent anomaly that
its neighbours do not share: the city minus a Gaussian-kernel (600 km) average of its neighbours. That relative series has
27-44% less noise than the raw anomaly. Before 1983 its recent level persisted, with fitted coefficients between 0.45 and 0.80
across fit origins 1940-1975.

There are three versions, all on the 30-year average, with no learners and no gate:
- **v1.1** adds a centred graph-local term.
- **v1.2** adds the same term plus a long-window relative offset.
- **v1.3** is v1.1 plus a pre-declared, damped and capped **national warm increment** for temperature and CDD.

The graph term has zero cross-city median adjustment, so by construction v1.1 and v1.2 add nothing for a continental warm
shift. v1.3 exists because 2019-2025 is expected to be warmer than the 30-year mean. Pre-1983 data cannot estimate how much of
a sustained warm shift to carry: the mean share of cities with a positive floored 30-year warming rate is 84%, 15% and 9% over
the three blocks. The increment's shrink (0.5) and cap (1.0 F on the national gap) are therefore a conservative prior, fixed
before computing any v1.3 number.

Chronological evidence (pre-1983, training targets <= F asserted in code, `docs/plans/creative/analysis/planA_walkforward.py`):

| Final points against average30 | 1940-49 | 1950-62 | 1963-75 |
|---|---:|---:|---:|
| v1.1 | +0.66 | +2.37 | +1.10 |
| v1.2 | +1.06 | +2.49 | +1.07 |
| v1.3 | +0.19 | +2.38 | +1.10 |
| v0.5-like warming base | -9.22 | +3.75 | +2.66 |

- The increment fires only in the 1940-49 block, where it costs 0.47 points against v1.1. That is the cost side; pre-1983 data
  never shows its benefit.
- Under `reward.py`'s exact bootstrap on 22 held-out triplets, v1.1 promotes on all 22, but **only just**: the weakest triplet
  has P(better) = 0.950. Its 90% Monte Carlo interval is [0.942, 0.958], and P fell below 0.95 in 2 of 30 seeds.
- v1.3 promotes on 20 of 22, and its weakest triplet fails in 26 of 30 seeds.

The trade-off is explicit (section 4.4). v1.3 is the better-motivated forecast for 2019-2025. It is less likely to promote on
the development folds, which cannot test sustained warming. A promoted v1.1 is not treated as the challenge submission as it
stands.

Everything is numpy. Runs read no feature or enhancement release, so they are `official`. One origin takes about 7 s,
deterministically. No official development ablations are run, because the ladder would count them (section 4.3).

Status: draft r5

Author: strategy planner A (Claude Opus 5.5), 2026-11-08. Brief: `docs/plans/creative/brief.md` (sections 3-6). Reward and
insights: `evaluation/REWARD.md`, `evaluation/INSIGHTS.md`, `code/evaluation/reward.py`. Evidence scripts:
`docs/plans/creative/analysis/planA_walkforward.py` (sections 3-4) and `docs/plans/creative/analysis/planA_section1.py`
(section 1).

---

## Changes in r5

Grok's review of r4 (86/100), with Sol's points from the plan B review, and the data and insights updates.

| Item | What changed |
|---|---|
| 1. National warm increment (Grok MUST) | New **v1.3 = v1.1 + I**, which replaces the deferred learner stack. The formula, the shrink `KAPPA = 0.5`, the cap `CAP_F = 1.0` F and the floor at 0 were written into the script before it first computed a v1.3 number, and were not changed afterwards. The plan states that pre-1983 data cannot estimate a sustained-warming coefficient (mean share of cities with a positive floored rate 84% / 15% / 9%), and explains how the shrink and cap were chosen instead (section 1.3). The pre-1983 cost is reported: -0.47 points in 1940-49, never the -9 of W, because the increment is at most 0.5 F. Grok's development numbers (+0.79 / +0.14 / +0.73 F residuals; +13 / -5 temperature points) and the insights bias figures are disclosed as reused development evidence that shaped the design. v1.3 has its own walk-forward rows and keep decision. The trade-off is in section 4.4. |
| 2. Bootstrap exactly as `reward.py` (SHOULD) | `compare()` in the script resets seed 20261108 for every triplet and resamples only cities with a scored row (205-208 per triplet). The weakest v1.1 triplet (1960, 1967, 1974) gives P = 0.9500, a binomial 90% interval of [0.9420, 0.9580], and P < 0.95 in 2 of 30 seeds (seeds 1-30: 0.9475-0.9690), matching Grok. Promotion is described as being at the bar. |
| 3. Split equality check (SHOULD) | Section 5: on rows with a 30-year mean, beta = 0 (and `KAPPA = 0` for v1.3) must equal `control:average30`. Fallback rows (cities without a 30-year mean: 16, 3 and 5 at F = 1940, 1960, 1975, with G = 0 there verified in the script) are checked against `reference(..., "average")`'s fallbacks separately: shorter-window mean, then full-history mean, then the metric-wide training median. The increment is not added on fallback rows. The "no NaN" wording is removed. |
| 4. Section 1 code (SHOULD) | New `docs/plans/creative/analysis/planA_section1.py` reproduces every section 1 table: distances, correlation by distance, relative noise, decomposition, in-era tables, the kernel horse-race, and the empirical-Bayes, break-aware and correlation-graph rejections. All are recomputed with the declared centred estimator. CDD kernel values and two decomposition coefficients moved in the second or third decimal; the table shows the script's values. |
| 5. Overlapping examples (SHOULD) | Section 3.1: at F = 1940 there are 22,671 examples on 5,936 unique city-years; at F = 1975, 30,640 on 7,963 (temperature). Each target city-year is used a median of 4 times. |
| 6. Wording (SHOULD) | `load()` parses the whole CSV, then keeps years <= 1982. 84% / 15% / 9% is the mean share of cities with a positive floored 30-year warming rate. |
| 7. Diagnostic exclusion (Sol MUST) | `reward.py ladder` counts every official development run, whatever its name. No official development ablations are run. The diagnostics that r4 planned as ablations are pre-1983 walk-forward rows instead (25-NN graph, `precip=off`, +0.5 W, raw national shift). Before each version, the champion is recorded with `reward.py ladder`, and the run is compared with it explicitly (`--against`). If the team wants development ablations, the plan asks the eval station for a diagnostic flag that `reward.candidates` skips (section 4.3). |
| Data update (features-v4/v5) | No version reads any feature or enhancement release, only `data/city_info.csv` (static). Section 2 says so per version. |
| Insights (eval station) | Each version names its finding. v1.1 and v1.2 target finding 1 (persistent per-city temperature offsets, honest +0.98 on v0.9). v1.3 targets finding 2 (regime sensitivity; bias flips sign by origin). The increment is damped and falls back to the 30-year average whenever the national gap is <= 0. Insights output is tracked per run (section 8). |

## Changes in r4 and r3 (for the record)

- **r4:** Regenerated the walk-forward with the target cutoff `t + h <= F` asserted. The r3 script had used `t + h <= 1982`.
  Froze v1.2 as the joint model, deferred the learner-based v1.3, corrected the zero-median wording, measured runtime and
  determinism, and added the spatial-block bootstrap.
- **r3:** Removed post-origin station information. Made the champion verdict the keep rule. Added the `average30 + G` route and
  one deploy path. Moved to the centred estimator. Added the walk-forward and the design-overlap table.

---

## 1. Strategy and model-selection logic

### 1.1 What the data say (years <= 1982; `planA_section1.py`)

Both scripts parse the whole `data/annual_valid.csv` and `data/city_info.csv`, then keep only years <= 1982 (asserted) before
computing anything. No `private/` file and no run output was read. Every table in this section fits and scores on the same
pre-1983 era. These tables are descriptive; the out-of-sample evidence is section 3.4.

**Spatial correlation** (median pairwise correlation of annual anomalies, 1941-1982, pairs with >= 25 shared years):

| Distance (km) | 0-250 | 250-500 | 500-750 | 750-1000 | 1000-1500 | 1500-2500 | >2500 | e-folding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| annual_avg_temp | 0.77 | 0.71 | 0.64 | 0.54 | 0.38 | 0.10 | -0.19 | 1176 km |
| cdd | 0.76 | 0.67 | 0.57 | 0.45 | 0.28 | 0.01 | -0.15 | 1026 km |
| hottest_day_temp | 0.67 | 0.52 | 0.40 | 0.30 | 0.18 | 0.00 | -0.13 | 859 km |
| total_precip_in | 0.60 | 0.42 | 0.27 | 0.18 | 0.10 | 0.04 | 0.00 | 746 km |

Median distance to the 1st, 5th, 10th, 15th and 30th nearest city: 104, 234, 346, 423 and 634 km.

**Relative anomaly** `R = anomaly - mean of 15 nearest`, 1941-1982:

| Metric | sd(anomaly) | sd(R) | lag-1 anomaly | lag-1 R | lag-5 R |
|---|---:|---:|---:|---:|---:|
| annual_avg_temp | 1.257 F | 0.702 F | 0.30 | 0.57 | 0.15 |
| cdd | 183.6 | 113.8 | 0.25 | 0.46 | 0.11 |
| hottest_day_temp | 2.990 F | 2.078 F | 0.14 | 0.14 | 0.00 |
| total_precip_in | 7.260 in | 5.296 in | 0.05 | 0.06 | 0.01 |

**Decomposition.** OLS without intercept of the lead-4-7 departure on three parts of the 10-year gap: national, regional
(15-NN) and local. The national part is removed from the target for the era columns.

| Metric | 1950-75 [national, regional, local] | regional 1950-62 / 1963-75 | local 1950-62 / 1963-75 |
|---|---|---|---|
| annual_avg_temp | 1.324, 0.371, 0.602 | +0.11 / +0.69 | +0.65 / +0.52 |
| cdd | 1.079, 0.722, 0.610 | +0.58 / +0.82 | +0.69 / +0.50 |
| hottest_day_temp | 1.380, 0.695, 0.495 | +0.25 / +1.41 | +0.53 / +0.43 |
| total_precip_in | -0.361, -0.402, 0.332 | -0.37 / -0.53 | +0.38 / +0.29 |

- The local coefficient is the era-stable one.
- The national coefficient for temperature, 1.32 over 1950-75, shows that recent national cold persisted in the cooling era.
  That is lesson 1's effect, and it is why the national increment in v1.3 is floored at 0 (section 1.3).

**Declared centred estimator, in-era best coefficient** (skill of `ref30 + b centre(L)` against ref30, leads 4-7, clipped). The
last column adds the best `b centre(L)` to the v0.5-like proxy `0.5 (ref30 + W) + 0.5 combo`:

| Metric | 1915-35: b, skill | 1950-62 | 1963-75 | gain over the proxy (3 eras) |
|---|---|---|---|---|
| annual_avg_temp | 0.65, +0.00865 | 1.00, +0.05054 | 0.65, +0.01433 | 0.00000 / +0.00600 / 0.00000 |
| total_precip_in | 0.70, +0.00670 | 0.75, +0.00685 | 0.40, +0.00219 | same (proxy = ref30) |
| hottest_day_temp | 0.65, +0.01348 | 0.65, +0.01483 | 0.55, +0.01045 | same (proxy = ref30) |
| cdd | 0.60, +0.00937 | 0.90, +0.03534 | 0.70, +0.01901 | +0.00011 / +0.00372 / +0.00022 |

v1.2 in-era, skill of `ref30 + centre(b L + g L_long)` on a 0.1 grid:

| Metric | 1915-35: b, g, skill | 1950-62 | 1963-75 |
|---|---|---|---|
| annual_avg_temp | 0.7, 0.4, +0.00930 | 1.0, 0.4, +0.05352 | 0.7, 0.3, +0.01599 |
| total_precip_in | 0.8, 0.4, +0.00797 | 0.9, 0.5, +0.01012 | 0.5, 0.2, +0.00307 |
| hottest_day_temp | 0.6, 0.0, +0.01343 | 0.6, 0.5, +0.01723 | 0.6, 0.2, +0.01157 |
| cdd | 0.6, 0.5, +0.00997 | 1.0, 0.5, +0.03815 | 0.8, 0.3, +0.02092 |

### 1.2 Chosen models and the finding each targets

```
v1.1:  forecast(c, m, O+h) = clip_m( ref30 + G ),          G = beta_m * centre(L_raw)
v1.2:  forecast            = clip_m( ref30 + G ),          G = centre(beta_m * L_raw + gamma_m * L_long_raw)
v1.3:  forecast            = clip_m( ref30 + I_m + G ),    G as v1.1, beta refitted with I in the base
       I_temp  = KAPPA * min(max(N, 0), CAP_F)             (same for every city with a 30-year mean)
       I_cdd,c = I_temp * rho_c                            I = 0 for precipitation and hottest day
clip_m = max(0, .) for total_precip_in and cdd. G and I are the same for all seven leads.
```

| Version | Insight finding targeted (`evaluation/INSIGHTS.md`, v0.9) | Hypothesis, written before scoring |
|---|---|---|
| v1.1 | 1. Persistent per-city temperature offsets (32% of squared error, honest +0.98) | A city's recent departure from its neighbours persists (local coefficient stable across eras), so `ref30 + beta centre(L)` removes part of the persistent city error. This holds for all four metrics, without the shared-warming exposure. |
| v1.2 | 1, the same finding, via a robust city level | The 30-year mean carries about `sd(R)/sqrt(30)` of local sampling noise (0.97 in precipitation, 0.38 F hottest day). A 60-year relative offset estimates the stable city level better. |
| v1.3 | 2. Regime sensitivity: the level bias flips sign by origin, so a constant correction fails | A level correction computed at each origin from its own recent national decade, damped by 0.5, capped, and falling back to the 30-year average whenever the gap is <= 0, captures part of a sustained warm shift without the -9-point failure of an extrapolated trend. |

Not targeted:
- **Finding 3** (northern zones about -0.5 points each): the zone contributions of v1.1-v1.3 are read from insights after
  scoring, and no zone-specific change is planned.
- **Finding 4** (CDD departures too large, best scaling 0.72): v1.x has no W departures. Its CDD graph coefficient is already
  fitted in-fold, between 0.45 and 0.75.
- **Finding 5** (precipitation): low priority, as insights says.

Why the 30-year average base:
- The champion is `control:average30`, so v1.1's paired comparison differs from it only by G.
- The W-based base loses -9.22 points in 1940-49 (section 3.4).
- Shared weather still enters each city's absolute error in both runs, so the paired difference is less noisy, not noise-free.

### 1.3 The national warm increment (v1.3)

**Why it is needed (Grok).**
- G has zero cross-city median, and the neighbour subtraction removes any uniform anomaly, so v1.1 matches climatology on the
  continental warm shift.
- REWARD.md says the development periods had weaker warming than 2019-2025.
- A model with no level term is biased cold for the challenge if warming continues.

**Reused development evidence, disclosed.** These numbers come from development years. They shaped the decision to add v1.3
and its fallback design. They were not used to set KAPPA or CAP_F, and this plan did not compute them:
- Grok, final-lead cross-city temperature residuals against the 30-year mean: +0.79 F (1987-1990), +0.14 F (1994-1997),
  +0.73 F (2001-2004).
- Grok: the undamped shift "cross-city median of the last-10 mean minus the 30-year mean" was worth about +13 temperature points
  at origin 1997 and about -5 at 1990.
- Insights: v0.9's level bias flips sign (temperature -0.98 / +0.52 / -0.20 F at 1983 / 1990 / 1997; CDD -84 / +62 / -2).

**What pre-1983 data can and cannot say.**
- The mean share of cities with a positive floored 30-year warming rate is 84% in 1940-49, 15% in 1950-62 and 9% in 1963-75.
- The national 10-year gap N was positive only in 1940-49 (mean +0.19 F, max +0.65) and negative throughout 1950-75 (-0.13 to
  -0.54 F).
- There is no pre-1983 era of sustained warming, so the data cannot estimate the coefficient of a warm shift. They can only
  measure the cost when warming stalls (1940-49).

**Formula (fixed).** At origin O:
1. N = the cross-city median, over cities with a 30-year mean and >= 5 valid years in O-9..O, of
   `mean(valid years in O-9..O) - ref30`.
2. `I_temp = 0.5 * min(max(N, 0), 1.0 F)`, added to every city's annual_avg_temp forecast at all leads (cities with a 30-year
   mean only).
3. `I_cdd,c = I_temp * rho_c`. `rho_c` is v0.5's regional CDD response: the OLS slope of annual CDD on annual mean temperature
   over O-29..O (>= 15 joint valid years), averaged over the city and its 15 nearest, NaN to 0, floored at 0.

**How the parameters were chosen, in place of estimation.**
- **Floor at 0.** This is the physical prior that the shift being forecast is warming. It also follows lesson 1: the national
  cold persistence of 1950-75 (coefficient 1.32) is the era-specific signal that cost the learners in v0.1, and a symmetric
  increment would bring it back. The diagnostic row "raw national shift" (unfloored, uncapped, KAPPA 1) shows this. It gains
  +5.16 and +2.29 points in the cooling blocks by persisting cooling, and loses -1.21 in 1940-49. It is not a candidate.
- **KAPPA = 0.5.** Write N as persistent signal plus decadal noise:
  - coefficient 0 is climatology (all noise);
  - 1 is full persistence of the recent decade's level;
  - 2 is a linear trend extrapolated to leads 4-7 (the leads sit about 20 years after the centre of the 30-year window, the
    last decade's centre about 10);
  - in 1940-49 the coefficient that would have helped was <= 0.

  0.5 is halfway between climatology and persistence, below the trend value. It is the same 50/50 shrinkage the series already
  uses (v0.5's combo weight), adopted as a convention, not fitted.
- **CAP_F = 1.0 F on N** (so the increment is at most 0.5 F). It bounds the damage from an exceptional national decade
  independent of the data. It was set as a round value above the pre-1983 range of N, so it binds only in exceptional decades.
  It never bound in 1940-1975 (max N +0.65). Whether it binds at the development origins or at 2018 is logged, not assumed.
- **CDD through rho**, not a uniform degree-day shift. Warming of x F adds about x times the city's local response, which is
  near 0 in cool cities and large in hot ones.

**Pre-1983 cost (walk-forward, section 3.4):**
- v1.3 minus v1.1 is -0.47 / +0.01 / +0.00 final points by block.
- In 1940-49 the temperature points fall from +1.38 to +0.04, CDD from +0.42 to -0.13, and the positive share from 70% to 50%.
- It does not repeat W's -9.22, because the increment is at most 0.5 F and is not extrapolated.

### 1.4 Alternatives considered and rejected

Kernel horse-race (`planA_section1.py` (f); mean over three eras of in-era best skill, centred L, clipped):

| Graph | temp | precip | hottest | cdd | mean |
|---|---:|---:|---:|---:|---:|
| 5-NN equal | +0.0217 | +0.0037 | +0.0115 | +0.0163 | +0.0133 |
| 15-NN equal (v0.3's graph) | +0.0240 | +0.0052 | +0.0112 | +0.0190 | +0.0149 |
| 25-NN equal | **+0.0274** | **+0.0055** | +0.0122 | **+0.0215** | **+0.0166** |
| Gaussian 300 km | +0.0239 | +0.0049 | +0.0116 | +0.0181 | +0.0146 |
| **Gaussian 600 km (declared in r1, kept)** | +0.0245 | +0.0052 | **+0.0129** | +0.0212 | +0.0160 |
| ordinary kriging (exp. cov., e-fold per metric, nugget 0.25, 30 nearest) | +0.0222 | +0.0046 | +0.0107 | +0.0164 | +0.0135 |
| correlation-learned (top 10 of 30 nearest by first-difference corr., corr^2) | +0.0259 | +0.0046 | +0.0105 | +0.0191 | +0.0150 |

25-NN leads by 0.0006 mean skill, and in the walk-forward by 0.00 / +0.06 / +0.11 points per block (section 3.4). I keep the
kernel declared in r1:
- The gap is below the triplet-level noise.
- Switching after seeing two comparisons on the same years is the kind of repeated choice the ladder is meant to prevent.

The 25-NN result is reported as a diagnostic.

Rejected alternatives against centred L (`planA_section1.py` (g)-(i); in-era best b on 0-1.5, skill against ref30):

| Metric | 1915-35: L / EB / break | 1950-62 | 1963-75 |
|---|---|---|---|
| annual_avg_temp | +0.0086 / +0.0083 / +0.0058 | +0.0507 / +0.0470 / +0.0597 | +0.0143 / +0.0133 / +0.0148 |
| total_precip_in | +0.0067 / +0.0000 / +0.0056 | +0.0069 / +0.0000 / +0.0044 | +0.0022 / +0.0000 / +0.0017 |
| hottest_day_temp | +0.0135 / +0.0104 / +0.0128 | +0.0148 / +0.0123 / +0.0145 | +0.0105 / +0.0100 / +0.0101 |
| cdd | +0.0094 / +0.0055 / +0.0087 | +0.0353 / +0.0349 / +0.0397 | +0.0190 / +0.0162 / +0.0176 |

| Candidate | Why not |
|---|---|
| Graph neural network (`uv add torch`) | The signal is one linear high-pass operator and a scalar. Learned messages over 210 nodes and about 40 overlapping snapshots would fit the era-unstable regional part. It also needs a lock change and risks GPU nondeterminism. |
| Graph-Laplacian per-city coefficients | The closest cheap test, per-city empirical-Bayes reliability weights (EB above), is below L in every era for every metric, and 0 for precipitation, where the white-noise assumption sends every weight to 0. |
| Break-aware local level (most significant single split of R, t > 4) | Better for temperature in 2 of 3 eras and for CDD in 1 of 3; worse for hottest day and precipitation in all three. Mixed and not walk-forward tested. Insights finding 1 suggests break-aware levels, so it is the natural next candidate after this round, as one version on its own. |
| Kriging, correlation-learned graph | Below the Gaussian kernel on the mean. |
| Regional precipitation mean reversion (-0.37 / -0.53) | Both eras sit in one drought-to-wet cycle; likely era-specific. |
| Smoothing or damping W | W's error is shared across nodes. The increment in v1.3 replaces W with a bounded level term. |

---

## 2. Targets and inputs

**Predicted:** the level `predicted_value`. Internally it is a shift from `ref30(c, m, O)`, as produced by
`evaluation.adapters.reference(..., method="average")`.

**Inputs, per version:** suite `train` and `data/city_info.csv` (lat/lon, `static_context`), with the manifest
`notebooks/Meet/placeholder-v0.1-assets.json`.
- **v1.1, v1.2 and v1.3 read no feature or enhancement release:** no `features-v4`, `features-v5`, `clean-v*` or
  `enhance-v*`, and no daily data.
- The `*_qc` / `_fill` / `_homog` / `_adj` columns in the feature releases are fitted on data up to the outer origin, so they
  would not be causal at pseudo-origins anyway.
- The SST indices in `enhance-v2` and `features-v5` are low-pass and would make runs exploratory. If a later version used them,
  it would load them through `code/enhance_ocean.load_ocean_for_origin` and score with the release's assets template, outside
  the champion route.

**Graph terms at a (pseudo-)origin t, from years <= t only:**

- `K_ij = exp(-(d_ij / 600 km)^2)`, with `K_ii = 0` and haversine distance d.
- `A = X - ref30(t)` on t-59..t. `N_c(y)` is the kernel mean of the neighbours' valid A, NaN if the valid mass is < 1.0.
  `R = A - N`.
- `L_raw = EWMA_hl(R, t-29..t) - mean(R, t-29..t)`, with half-life 3 (temperature, hottest day, CDD) or 6 (precipitation). It
  needs >= 10 valid years.
- `L_long_raw = mean(R, t-59..t) - mean(R, t-29..t)`. It needs >= 15 valid years in t-59..t-30 and >= 10 in t-29..t.
- `centre(v)`: subtract the median over defined cities, and set undefined cities to 0. The median adjustment is 0, but the
  median forecast level can still move (baseline [0, 10, 20] + [20, 0, -1] moves the median from 10 to 19), and G can correlate
  with any spatial pattern.
- National increment (v1.3): section 1.3. It is computed from the same cut table. It is the only term that moves every city's
  level in the same direction.

---

## 3. Training procedure

### 3.1 Coefficient fit

At the outer origin O:
1. Take pseudo-origins t in [O-45, O-4], leads h in 4..7, with **t + h <= O** (asserted).
2. Fit `beta_m` (v1.2: `(beta_m, gamma_m)`) as the argmin over the grid 0.00-1.00 (step 0.05, ascending, first minimum kept) of
   `mean |y - deploy(base, G)|`. The base is ref30 for v1.1/v1.2 and `ref30 + I(t)` for v1.3, with the increment computed at
   each pseudo-origin from its own data. With fewer than 2,000 rows the coefficients are 0.

**The fit sample is overlapping examples, not independent city-years.** For temperature, F = 1940 gives 22,671 examples on
5,936 unique target city-years, and F = 1975 gives 30,640 on 7,963. Precipitation is 22,965 / 6,012 and 30,644 / 7,957. A
target city-year appears a median of 4 times: four leads from four pseudo-origins.

### 3.2 One assembly path

```python
def centre(v):
    return np.where(np.isfinite(v), v - np.nanmedian(v), 0.0)

def graph_shift(beta, gamma, L_raw, L_long_raw, use_long): ...      # as planA_walkforward.graph_shift

def national_increment(tables, O, rho):                             # v1.3; 0 for precipitation and hottest day
    N = median over cities with ref30 and >= 5 valid in O-9..O of (mean(O-9..O) - ref30)
    I_temp = KAPPA * min(max(N, 0.0), CAP_F)                        # KAPPA = 0.5, CAP_F = 1.0, fixed
    return {"annual_avg_temp": I_temp, "cdd": I_temp * rho, "total_precip_in": 0, "hottest_day_temp": 0}

def deploy(metric, ref30, G, I=0.0):                                # the only forecast formula (fit and forecast)
    f = ref30 + I + G
    return np.maximum(f, 0.0) if metric in NON_NEGATIVE else f
```

- The script implements these functions: `graph_shift`, `national_gap`, `national_shift`, `deploy`, `coef_rows`, `fit_coefs`.
- The adapter copies them, takes `load_coords` and `_check_coords` from v0.5, and uses `reference(..., "average")` for ref30
  and its fallbacks.
- On rows without a 30-year mean, G = 0, I = 0, and the forecast is `reference()`'s fallback.
- There is no gate and no inner origin.

### 3.3 Determinism and runtime (measured)

- No randomness. Grid scans run in fixed order, and kernel sums use broadcasting rather than BLAS `@`.
- One complete origin from a table cut at F (CSV load, pivot, 46 pseudo-origins, v1.1 + v1.2 + v1.3 fits, forecast), twice
  each:

| F | run 1 | run 2 | SHA-256 v1.1+v1.2 | SHA-256 v1.3 | identical | equals walk-forward |
|---|---:|---:|---|---|---|---|
| 1940 | 6.5 s | 6.6 s | d2654cef2645705a | 17e785718f01c097 | yes | yes |
| 1960 | 7.6 s | 6.4 s | f4b94357cdafa19f | a59a425d96ee5df4 | yes | yes |
| 1975 | 6.8 s | 7.2 s | 75f65a152199aef2 | 1cf734bd8393feb9 | yes | yes |

The v1.1+v1.2 digests are unchanged from r4. The adapter is not written yet; its first checks are in section 5. The harness
limit is 900 s.

### 3.4 Chronological evidence (cutoff enforced)

For each fit origin F = 1940..1975, the coefficients are fitted with targets <= F (asserted). The forecast at F is scored on
F+1..F+7 against the 30-year average with the reward formula. Origins 1940, 1960 and 1975 recomputed from a table with every
year > F removed reproduce the walk-forward exactly.

**Held-out points against average30, by block of fit origins:**

| Candidate | 1940-49 final | 1950-62 final | 1963-75 final | origins > 0 | live by block |
|---|---:|---:|---:|---|---|
| **v1.1** | **+0.6582** | **+2.3694** | **+1.0970** | 70% / 100% / 85% | +2.32 / +3.00 / +2.56 |
| **v1.2** | **+1.0576** | **+2.4880** | **+1.0703** | 80% / 100% / 85% | +2.71 / +2.97 / +2.46 |
| **v1.3** | **+0.1861** | **+2.3823** | **+1.0989** | 50% / 100% / 85% | +1.87 / +3.02 / +2.57 |
| diag: v1.1 with 25-NN graph | +0.6621 | +2.4299 | +1.2032 | 70% / 100% / 85% | +2.40 / +3.10 / +2.52 |
| diag: v1.1 with precip=off | +0.7751 | +2.2058 | +1.0399 | 80% / 100% / 85% | +2.35 / +2.87 / +2.40 |
| diag: v1.1 + raw national shift (KAPPA 1, unfloored, uncapped) | -1.2126 | +5.1629 | +2.2861 | 40% / 100% / 77% | +0.87 / +3.90 / +4.85 |
| diag: v1.1 + 0.5 W | -2.8193 | +2.1311 | +1.3292 | 40% / 100% / 92% | -0.66 / +2.91 / +2.62 |
| ref: v0.5-like proxy, no G | -9.2177 | +3.7541 | +2.6594 | 0% / 100% / 100% | -7.20 / +3.08 / +3.83 |

**Final points by metric** (temp / precip / hottest / cdd):

| Candidate | 1940-49 | 1950-62 | 1963-75 | worst origin |
|---|---|---|---|---:|
| v1.1 | +1.38 / -0.47 / +1.30 / +0.42 | +4.40 / +0.65 / +1.49 / +2.94 | +1.37 / +0.23 / +1.03 / +1.76 | -1.31 |
| v1.2 | +1.69 / +0.03 / +1.76 / +0.75 | +4.49 / +0.81 / +1.62 / +3.05 | +1.52 / +0.16 / +0.84 / +1.76 | -0.83 |
| v1.3 | +0.04 / -0.47 / +1.30 / -0.13 | +4.40 / +0.65 / +1.49 / +2.99 | +1.36 / +0.23 / +1.03 / +1.78 | -2.32 |

**National increment, by block** (N is the national 10-year gap; I = 0.5 min(max(N, 0), 1)):

| Block | N mean (range), F | I mean, F | origins with I > 0 | cap binding |
|---|---|---:|---:|---:|
| 1940-49 | +0.194 (-0.216 to +0.651) | 0.112 | 80% | 0% |
| 1950-62 | -0.241 (-0.335 to -0.134) | 0.000 | 0% | 0% |
| 1963-75 | -0.428 (-0.539 to -0.300) | 0.000 | 0% | 0% |

From 1950 onwards, v1.3 equals v1.1 up to the small change in refitted beta. The pre-1983 test measures only the increment's
cost when warming stalls, never its benefit.

**Coefficients (beta, gamma) by fit origin** (data <= F only):

| Metric | v1.1 beta 1940 / 1950 / 1960 / 1970 / 1975 (median of 36) | v1.2 (beta, gamma) 1940 / 1950 / 1960 / 1970 / 1975 (median) | v1.3 beta (median) |
|---|---|---|---|
| annual_avg_temp | 0.80 / 0.65 / 0.70 / 0.80 / 0.80 (0.725) | (0.80,0.45) (0.65,0.35) (0.75,0.40) (0.90,0.35) (0.85,0.40) (0.75, 0.40) | 0.80 / 0.65 / 0.70 / 0.85 / 0.80 (0.75) |
| total_precip_in | 0.80 / 0.60 / 0.55 / 0.50 / 0.45 (0.55) | (0.80,0.40) (0.60,0.75) (0.60,0.60) (0.55,0.55) (0.60,0.50) (0.60, 0.65) | same as v1.1 |
| hottest_day_temp | 0.60 / 0.60 / 0.65 / 0.65 / 0.60 (0.65) | (0.60,0.05) (0.60,0.65) (0.65,0.70) (0.65,0.50) (0.65,0.50) (0.60, 0.65) | same as v1.1 |
| cdd | 0.70 / 0.45 / 0.55 / 0.65 / 0.70 (0.55) | (0.70,0.55) (0.50,0.40) (0.60,0.45) (0.65,0.50) (0.75,0.25) (0.60, 0.45) | 0.65 / 0.45 / 0.55 / 0.65 / 0.70 (0.55) |

The coefficients are stable for temperature, hottest day and CDD. Precipitation's beta falls from 0.80 to 0.45, and it is the
weakest component: `precip=off` is 0.12 points better in 1940-49 and 0.06-0.16 worse after.

**Promotion analogue, exactly as `reward.py`.** 22 held-out triplets (F, F+7, F+14), F = 1940..1961. Each comparison:
- 2,000 multinomial draws, with seed 20261108 reset per triplet;
- only cities with a scored row resampled (205-208);
- the same draw for every origin, split and metric;
- `reward.py`'s four guards.

Dependence diagnostic: whole spatial blocks resampled instead, 5 x 5 degrees (42 blocks) or 10 x 10 (14 blocks).

| Candidate vs champion | resampling | median reward | median P | min P (triplet) | median 90% LB | min LB | worst floor | promote |
|---|---|---:|---:|---|---:|---:|---:|---:|
| **v1.1 vs average30** | cities (official) | +1.8532 | 1.000 | **0.9500** (1960, 1967, 1974) | +1.1659 | +0.0010 | -0.1058 | 22 / 22 |
| v1.1 vs average30 | 5-degree blocks | +1.8532 | 1.000 | 0.9555 | +1.0889 | +0.0291 | -0.1058 | 22 / 22 |
| v1.1 vs average30 | 10-degree blocks | +1.8532 | 1.000 | 0.9555 | +1.0805 | +0.0202 | -0.1058 | 22 / 22 |
| **v1.2 vs average30** | cities (official) | +1.8073 | 1.000 | 0.9765 (1961, 1968, 1975) | +1.1497 | +0.1104 | -0.0344 | 22 / 22 |
| v1.2 vs average30 | 10-degree blocks | +1.8073 | 1.000 | 0.9445 | +1.1165 | -0.0219 | -0.0344 | 21 / 22 |
| **v1.3 vs average30** | cities (official) | +1.8609 | 1.000 | **0.9390** (1941, 1948, 1955) | +1.1747 | -0.0451 | -0.1058 | 20 / 22 |
| v1.3 vs average30 | 10-degree blocks | +1.8609 | 1.000 | 0.9085 | +1.0707 | -0.1567 | -0.1058 | 20 / 22 |
| v1.2 vs v1.1 | cities | +0.1019 | 0.708 | 0.2560 | -0.1905 | -0.5223 | -0.0344 | 4 / 22 |
| v1.3 vs v1.1 | cities | +0.0000 | 0.148 | 0.0000 | -0.0099 | -0.8578 | -0.1058 | 3 / 22 |
| diag 25-NN vs average30 | cities | +1.9090 | 1.000 | 0.9695 | +1.2267 | +0.1118 | -0.0758 | 22 / 22 |
| diag precip=off vs average30 | cities | +1.7577 | 1.000 | 0.9385 | +1.0849 | -0.0432 | +0.0000 | 21 / 22 |
| diag raw national shift vs average30 | cities | +3.7372 | 1.000 | 0.4600 | +2.9899 | -0.8688 | -0.8909 | 19 / 22 |
| ref v0.5-like proxy vs average30 | cities | +3.0465 | 1.000 | 0.0000 | +2.2895 | -7.8926 | -16.0311 | 16 / 22 |
| ref v0.5-like proxy vs average30 | 10-degree blocks | +3.0465 | 0.998 | 0.0000 | +1.5381 | -8.5361 | -16.0311 | 16 / 22 |

**Monte Carlo error of the verdict at the bar:**
- v1.1, triplet (1960, 1967, 1974): P = 0.9500 at seed 20261108, with binomial 90% interval [0.9420, 0.9580]. Seeds 1-30 give
  0.9475-0.9690 (mean 0.9577), and P < 0.95 in 2 of 30.
- v1.2's weakest triplet (1961, 1968, 1975) is 0.9765, and P < 0.95 in 0 of 30 seeds.
- v1.3's weakest triplet (1941, 1948, 1955) is 0.9390, and P < 0.95 in 26 of 30 seeds.

What this means:
- **v1.1 promotes in the analogue only at the bar.** A development promote is plausible, not assured; v1.2 has more margin.
- Spatial-block resampling lowers v1.1's median lower bound by 0.08-0.09 points, against 0.75 for the W-based proxy. The graph
  term's gains are spatially spread out; the warming term's are clustered.
- **v1.3 vs v1.1** is identical or nearly identical on most triplets: the increment is 0 at every fit origin after 1949, and
  only the refitted beta differs. Exact ties count as neither better nor worse in `reward.py`, hence the median P of 0.148. It promotes on only 3 of 22. If v1.1 promotes first, v1.3 will very likely be `within_noise` against it.
- The bootstrap cannot show four things: triplet overlap; that the kernel and half-lives were chosen on these years
  (section 4.2); 1984-2004 behaving differently; and anything about sustained warming.

---

## 4. Selection without peeking

### 4.1 Fixed in advance vs fitted in-fold

**Fixed:**
- Gaussian 600 km kernel, mass floor 1.0.
- Half-lives {3, 3, 3, 6}.
- Windows of 30 and 60 years, minimum valid years 10 / 15.
- Combined centring.
- Grid 0-1, step 0.05, ascending ties. Fit leads 4-7 with `t + h <= O`. `MIN_COEF_ROWS = 2000`.
- G and I on all leads.
- Base = 30-year average.
- v1.3: `KAPPA = 0.5`, `CAP_F = 1.0`, floor at 0, `GAP_MIN_VALID = 5`, CDD conversion through v0.5's rho.

**Fitted in each fold:** beta (and gamma) at the outer origin. N and rho are computed at the origin from its own data. No value
is chosen after looking at a development score.

### 4.2 Overlap between design years and the development folds

Design used pseudo-origins 1895-1975 with targets <= 1982.

| Outer origin | coefficient-fit pseudo-origins | overlap with design pseudo-origins | targets in fit |
|---|---|---|---|
| 1983 | 1938-1979 | 1938-1975 (38 of 42) | 1942-1983 |
| 1990 | 1945-1986 | 1945-1975 (31 of 42) | 1949-1990 |
| 1997 | 1952-1993 | 1952-1975 (24 of 42) | 1956-1997 |

- The fixed hyperparameters were chosen on years that also train every development fold, which is allowed.
- The in-fold fit and the walk-forward are not independent evidence for the kernel or half-lives.
- This plan computed no number on 1983-2018. Development-year numbers from reviewers and insights shaped v1.3 (section 1.3)
  and are reused evidence.

### 4.3 Keep rule, champion record and diagnostic exclusion

**Before each version:**
1. Run `uv run python code/evaluation/reward.py ladder` and record the champion's run id and model in
   `docs/models/placeholder.md`. Other stations' official runs can change it between versions.
2. Score the version.
3. Run `reward.py score <run_id> --against <recorded champion> --json`.

**Keep rule** (brief sections 5-6):
1. **Promote** against the recorded champion, with all four `reward.py` guards. A promoted version becomes the champion.
2. **While nothing from this plan has promoted:** a non-promoting version is carried forward over `Placeholder_v0.5` only if
   both its 90% lower bound and its reward beat v0.5's, against the same champion. v0.5's values come from
   `reward.py score 20261108T141546Z-d0ac8f8b36 --json`, read before v1.1 and recorded. After a promotion, a version is carried
   forward only by promoting against the new champion.
3. `worse` is never kept.
4. Secondary evidence, reported but never deciding: the paired comparison against v0.5, and the comparison against
   `control:average30` when the champion is something else. Shared errors do not cancel exactly in a paired MAE comparison.

There is no fixed point threshold.

**Diagnostic exclusion (Sol MUST).** `reward.candidates` puts every complete official run on the locked suite into the ladder,
whatever its name or tags. An official ablation that happened to clear the bar would become champion, and every extra official
run is another try at the same three folds. So:
- **No official development ablations in this round.** The three official runs are v1.1, v1.2 and v1.3, each a real candidate.
- The questions r4's ablations asked are answered by the pre-1983 diagnostic rows in section 3.4: 25-NN graph, `precip=off`,
  +0.5 W, and the raw national shift.
- I rejected the two mechanisms that would make development ablations exploratory:
  - Scoring from an uncommitted checkout: the label is honest, but the run loses its commit pin and cannot be reproduced.
  - Declaring an unused non-gate-eligible release: the label would misstate the inputs.
- If the team wants development ablations, the plan asks the eval station for a `diagnostic` flag in `run.json` that
  `reward.candidates` skips. Until that exists, none are run.

**Pre-registered expectations (development):**
- v1.1: +0.5 to +2.0 points against average30, positive at >= 2 origins including 1990. Hottest day and temperature carry the
  gain; precipitation is within ±0.5. If v1.1 is negative at >= 2 origins, the pre-1983 persistence does not hold in 1984-2004.
- v1.3: within ±1.5 points of v1.1 at every origin, and below v1.1 at 1990 if the national gap there is positive (reused
  evidence suggests so). Likely `within_noise` against v1.1 if v1.1 has promoted.

### 4.4 The challenge forecast and the trade-off

A promoted v1.1 is **not** the 2018-origin submission as it stands. Its forecast has the 30-year average's cross-city level,
and every source available here points to 2019-2025 being warmer (REWARD.md, and Grok's reused residuals). The trade-off:

- **v1.3 helps the challenge forecast**, if warming continued, by up to 0.5 F on temperature and the matching CDD response.
- **v1.3 costs the development promote.** On the folds it adds a shared shift, which is spread across all cities and cannot be
  diversified by city resampling. Its sign depends on the origin, and 1990-type origins lose.
  - In the analogue it promotes on 20 of 22 triplets against average30, against 22 for v1.1.
  - Its weakest triplet fails in 26 of 30 seeds.
  - Against a v1.1 champion it promotes on 3 of 22.
- **The development folds cannot settle this.** Their warming is weaker than 2019-2025, and pre-1983 has no sustained-warming
  era. A non-promoting v1.3 is not evidence against the increment.

**Pre-declared hand-off recommendation** (it does not change the ladder or the keep rule; the integrator and team decide). For
the frozen candidate and the 2004/2011 assessment, recommend v1.3 over v1.1 if all of these hold:
- v1.3's verdict against the champion recorded before it is not `worse`.
- Every metric's final points are >= -2.
- Its live points are not clearly worse than average30's.

Otherwise recommend the ladder champion, and state in the README that it carries no national warm shift. The assessment
origins (2004 targets 2005-2011, 2011 targets 2012-2018) are later and warmer than the development folds, so they are the
first clean test of the increment.

---

## 5. Versions

Each version is one single-file adapter in `notebooks/Meet/`, committed, then scored from a clean commit:

```powershell
uv run python scripts/line.py status
uv run python code/evaluation/reward.py ladder                     # record the champion first
uv run python scripts/line.py score --name Placeholder_v1.1 --adapter notebooks/Meet/placeholder_v1_1.py:forecast `
  --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3
uv run python code/evaluation/reward.py score <run_id> --against <recorded champion> --json
uv run python code/evaluation/insights.py <run_id> --json
uv run python scripts/line.py sync
```

| Version | Built on | Reads | Change | Finding |
|---|---|---|---|---|
| **Placeholder_v1.1** | `control:average30` | `data/city_info.csv` only | `G = beta_m centre(L_raw)`, all metrics, clipped | 1 |
| **Placeholder_v1.2** | `control:average30` (sibling, run unconditionally) | same | `G = centre(beta_m L_raw + gamma_m L_long_raw)`, joint fit | 1 |
| **Placeholder_v1.3** | v1.1's definition | same | + national warm increment I (section 1.3), beta refitted with I in the base | 2 |

- **Order:** v1.1, v1.2, v1.3. Each is compared with the champion recorded just before it. No other official runs, and no
  version tuned on a development score.
- **The learner-based v0.5 + G** stays deferred (r4). Validating it needs residual learners refitted at every walk-forward and
  inner origin.
- **Implementation:** v1.1 2-3 h (copy the script's functions); v1.2 and v1.3 about 30 min each on top.
- **Record per run:** beta (gamma) per metric, mean `|G|`, share of cities with G != 0, the centring median, the median forecast
  shift, and for v1.3 N, I, whether the cap bound, and the mean CDD increment.

**Checks before scoring:**
- **Rows with a 30-year mean:** with beta = 0 (and `KAPPA = 0` for v1.3), the forecast equals `control:average30` exactly.
- **Rows without a 30-year mean** (pre-1983: 16 cities at F = 1940, 3 at 1960, 5 at 1975; the development counts are measured
  at this check): G = 0 and I = 0 there. The forecast equals `reference(..., "average")`'s fallback exactly: the mean of the
  valid years in the 30-year window, else the full-history mean, else the metric-wide training median.
- G has zero cross-city median adjustment at every origin.
- No year > fit origin is read: cut the table at t and compare.
- The adapter's 1975 forecast on a pre-1983 cut equals the script's, to 1e-9.
- Two runs give identical SHA-256 digests. Then read `runtime_s` from the first scorecard.

---

## 6. How each lesson and finding is addressed

1. **Shared warming (lesson 1).**
   - v1.1 and v1.2 carry none: the median adjustment is 0, and the level effect is not claimed to be zero.
   - v1.3 carries a bounded, floored national level term, not an extrapolated trend.
   - The floor keeps out the mid-century cooling persistence that the learners absorbed in v0.1.
2. **The 1990 overshoot (lesson 2, finding 2).**
   - v1.1 and v1.2 have no exposure: in 1940-49 they gain +0.66 / +1.06, against -9.22 for W.
   - v1.3's exposure is bounded at 0.5 F and costs 0.47 points in 1940-49. It falls back to the 30-year average whenever the
     national gap is <= 0.
   - Nothing is tuned on 1990.
3. **Zero precipitation and hottest-day skill (lesson 3).** Hottest day gains +1.03 to +1.49 points in every block.
   Precipitation is weak: -0.47 / +0.65 / +0.23 (v1.1) and +0.03 / +0.81 / +0.16 (v1.2), with a falling beta.
4. **Noise (lesson 4).** At most two scalars per metric, fitted on overlapping examples (median reuse 4). The verdict is at the
   bar for v1.1, with Monte Carlo error reported. Spatial-block resampling shows that the city bootstrap is mildly optimistic
   for G.
5. **Reused years (lesson 5).** Design used years <= 1982. Three official runs, and no official ablations. Reused development
   numbers are disclosed (section 1.3). Only the 2004/2011 assessment is clean.

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Measured (pre-1983, cutoff enforced) | Sign in development results | Response |
|---|---|---|---|
| v1.1 just misses the promotion bar | weakest triplet P 0.950, below 0.95 in 2 of 30 seeds | P(better) 0.93-0.96 | Rule 4.3.2 decides; no reruns or seed changes to push P |
| Local persistence weaker in 1984-2004 | beta stable except precipitation (0.80 to 0.45) | v1.1 negative at >= 2 origins | Report a negative result; no retuning |
| Increment wrong-signed at a development origin | 1940-49 cost -0.47 points | v1.3 below v1.1 at 1990 | Expected; section 4.4 hand-off rule |
| Precipitation below -2 points | worst block -0.47; worst triplet floor -0.11 | `reward.py` floor | Checked per run |
| Live clearly worse | v1.1 live +2.32 to +3.00; v1.3 +1.87 to +3.02 | live P(worse) | Measured per run |
| City bootstrap optimistic | spatial blocks lower median LB by 0.08-0.09 | | Official verdict unchanged; diagnostic reported |
| Hyperparameters chosen on evidence years | not measurable | promotes on development, fails assessment | One-shot assessment; 25-NN diagnostic shows the kernel sensitivity is small |
| Champion changes between versions | | | Champion recorded before each version |
| Adapter differs from the script | | 1975 equality check fails | Fix before scoring |

**Dependencies:** none new (numpy, pandas), and no data release. **Compute:** about 7 s per origin; 3 official runs.

---

## 8. Tracking

Every scored run is mirrored with `uv run python scripts/line.py sync`.

- **Names:** `Placeholder_v1.1`, `Placeholder_v1.2`, `Placeholder_v1.3`.
- **Tags:** `series=Placeholder`, `version`, `plan=A-graph`, `built_on`, `finding=<1|2>`, `label=official`, and
  `champion_before=<run id>`.
- **Reward block, against the recorded champion and, secondary, against v0.5:**
  - the verdict and the four guard booleans;
  - the final delta, 90% interval, P(better) and noise sd;
  - points by origin and by metric;
  - the live delta and live P(worse).
- **Insights per run** (`insights.py --json`):
  - the city-offset share for temperature (finding 1: should fall for v1.1/v1.2);
  - the bias by origin for temperature and CDD (finding 2: v1.3);
  - Central-North and East-North zone points (finding 3);
  - CDD best scaling (finding 4).
- **Scorecard metrics:** `skill.origin.*`, `skill.final.*`, `skill.live.*`, `bias.*`, `win_rate`, `worst_decile_city_skill`,
  `stability.range`, `runtime_s`.
- **Artifacts:** `placeholder_selection.json` with the coefficients, `|G|`, centring median, median forecast shift and, for
  v1.3, N, I, the cap flag and the CDD increment. Results go in `docs/models/placeholder.md` with the champion recorded before
  each version.

---

## Appendix A. How the numbers were computed

- **Section 3.4, the coefficients, fit-sample counts, fallback counts, national increment, bootstraps, Monte Carlo, runtime and
  digests:** `docs/plans/creative/analysis/planA_walkforward.py` (about 5 minutes).
  - It parses the whole CSV, then keeps years <= 1982 (asserted), and asserts `max(training target year) <= F` before every
    fit.
  - Its `compare()` mirrors `reward.py`'s `compare()`: seed reset per call, only cities with a scored row, the same draws for
    all cells, and the four guards. The champion's skill is 0 when the champion is average30.
  - Spatial blocks are cells of `floor(lat / 5), floor(lon / 5)` and `floor(lat / 10), floor(lon / 10)`.
- **Section 1:** `docs/plans/creative/analysis/planA_section1.py` (about 3 minutes). It imports the estimator from the
  walk-forward script.
  - Correlations use anomalies against each city's 1941-1982 mean, with pairs needing >= 25 shared valid years. The e-folding
    length is a least-squares fit of `log corr = -d / L` (corr > 0.05, d < 2000 km).
  - The decomposition is OLS without intercept on pseudo-origins 1950-1975, leads 4-7.
  - The in-era and kernel tables pick the best coefficient on the same era, with the declared centred estimator and clipping.
    Kriging and the correlation graph use row-normalised weights with a valid-mass floor of 0.5.
  - Empirical-Bayes weights are `tau^2 / (tau^2 + noise)` per city, with noise = `var(R) sum(w^2) / (sum w)^2` and tau^2 from
    the cross-city mean of `L^2`.
  - The break level is the mean of R after the most significant single split (>= 3 years each side) when t > 4, else L.
- **W and the v0.5-like proxy** (diagnostic rows only) follow v0.5's definitions. rho is the same regional CDD response that W
  uses.
