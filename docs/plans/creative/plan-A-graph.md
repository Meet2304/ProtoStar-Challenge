# Plan A: graph-based spatio-temporal learning for the Placeholder series (v1.x)

This plan treats the 210 cities as nodes of a fixed geographic graph. It forecasts the part of each city's recent anomaly that
its neighbours do not share: the city minus a Gaussian-kernel (600 km) average of its neighbours. That relative series has
27-44% less noise than the raw anomaly. Before 1983 its recent level persisted, with fitted coefficients between 0.45 and 0.80
across fit origins 1940-1975.

There are three versions, all on the 30-year average, with no learners and no gate:
- **v1.1** adds a centred graph-local term.
- **v1.2** adds the same term plus a long-window relative offset.
- **v1.3** is v1.1 plus a damped, floored national level increment for temperature and CDD.

The graph term has zero cross-city median adjustment, and the neighbour subtraction removes any uniform anomaly, so v1.1 and
v1.2 cannot respond to a nationwide departure of the recent decade from the 30-year mean. v1.3 measures that departure at each
origin, from that origin's data only: N is the cross-city median of the last-10-year mean minus the 30-year mean. It carries
`0.5 * min(max(N, 0), 1.0 F)` forward. The parameters are unchanged since r5.

The case for v1.3 rests only on the frozen-parameter walk-forward on pre-1983 data (training targets <= F asserted in code,
`docs/plans/creative/analysis/planA_walkforward.py`):

| Fit-origin block | 1915-35 | 1925-39 | 1940-49 | 1950-62 | 1963-75 | all 61 fit origins 1915-1975 |
|---|---:|---:|---:|---:|---:|---:|
| national gap N > 0 at | 95% of origins | 100% | 80% | 0% | 0% | |
| v1.3 minus v1.1 | +1.7341 | +1.9924 | -0.4721 | +0.0130 | +0.0018 | +0.4957 |

When N was positive and the warm decades continued (1915-1939), the increment paid about +1.7 to +2.0 points. When warming
stalled (1940-49) it cost 0.47. When N was negative (1950-75) the increment itself was 0; final points still moved by +0.0130
(1950-62) and +0.0018 (1963-75) because beta is refitted. The early-era rows reuse design-era years, so they are reused
design-era evidence, not a clean test. The 1915-35 and 1925-39 blocks overlap in 1925-1935, and the 11 early triplets
overlap each other, so they are supporting descriptions of one stretch of history, not independent replications.

| Final points against average30 | 1915-35 | 1925-39 | 1940-49 | 1950-62 | 1963-75 |
|---|---:|---:|---:|---:|---:|
| v1.1 | +0.90 | +0.51 | +0.66 | +2.37 | +1.10 |
| v1.2 | +0.87 | +0.56 | +1.06 | +2.49 | +1.07 |
| v1.3 | +2.63 | +2.50 | +0.19 | +2.38 | +1.10 |
| v0.5-like warming base | +4.14 | +1.38 | -9.22 | +3.75 | +2.66 |

Under `reward.py`'s exact bootstrap on 22 held-out triplets from 1940-1961, v1.1 promotes on all 22, but only at the bar: the
weakest triplet has P(better) = 0.950, and P fell below 0.95 in 2 of 30 seeds. v1.3 promotes on 20 of 22 there. On the 11
early triplets (1915-1925 starts), v1.3 promotes on 11 of 11 against average30 and on 11 of 11 against v1.1, while v1.1
promotes on 7 of 11.

**One decision procedure** (section 4.3) governs champion retention, nomination for the integrator's assessment, and the
submission:
- Every version is judged against the champion recorded just before it, and only a `promote` changes the champion.
- The nominee is the latest champion.
- A non-promoting v1.3 goes to the integrator as a disclosed alternative, not a recommendation.

Everything is numpy. Runs read no feature or enhancement release, so they are `official`. One origin takes 7-9 s,
deterministically. No development diagnostics are run, even though the `--diagnostic` flag now exists (section 4.3).

Status: draft r6.2 (r6 plus wording fixes from Sol's r6 review (93/100) and Grok's r6.1 review (94/100), and the pre-declared conditional v1.4; edits by the training station)

Author: strategy planner A (Claude Opus 5.5), 2026-11-08. Brief: `docs/plans/creative/brief.md` (sections 3-7). Reward and
insights: `evaluation/REWARD.md`, `evaluation/INSIGHTS.md`, `code/evaluation/reward.py`. Evidence scripts:
`docs/plans/creative/analysis/planA_walkforward.py` (sections 3-4) and `docs/plans/creative/analysis/planA_section1.py`
(section 1).

---

## Changes in r6

Sol's re-review of r5 (88/100): three MUSTs and three SHOULDs, plus the eval station's diagnostic-run mechanism.

| Item | What changed |
|---|---|
| 1. Causal rationale (MUST) | Removed from the selection argument: every statement about realised 2019-2025 conditions; every assertion that the assessment periods or the challenge years are warmer, including the REWARD.md line; and the reviewers' and insights' development-year numbers. The case for v1.3 now rests on information available at each origin: the origin-local gap N, and the frozen-parameter walk-forward over 61 pre-1983 fit origins (sections 1.3 and 3.4). Section 1.3 keeps a provenance note: the author saw development-year numbers before r5, and they are not part of the argument. |
| 2. One decision procedure (MUST) | Sections 4.3 and 4.4 merged into one mechanical procedure. Each version is compared with the champion recorded before it (v1.3 against v1.1 or v1.2 if either promoted). Only a promote changes the champion. The nominee for the assessment and the submission is the latest champion. A non-promoting v1.3 goes to the integrator as a disclosed alternative with its evidence, not a recommendation. The r5 hand-off recommendation is withdrawn. |
| 3. Historical warming claim and early-era diagnostic (MUST) | Withdrawn: "no pre-1983 era of sustained warming" and "never shows its benefit". N is positive at 95% of fit origins 1915-1935 and at every origin 1925-1939. The script now also fits origins 1915-1939, with the same `t + h <= F` assert and unchanged parameters (at least 8,882 temperature and 9,758 precipitation training examples per fit). It adds a cut-table equality check at F = 1925. v1.3 minus v1.1 is **+1.7341** (1915-35) and **+1.9924** (1925-39), matching Sol. These rows are labelled reused design-era evidence. The trade-off discussion is updated. |
| 4. Parameter provenance (SHOULD) | The freeze of KAPPA, CAP_F and the floor before the first v1.3 number is described as **author-reported**: no timestamped pre-run declaration exists. The constants and the first v1.3 results were committed together in r5. |
| 5. Wording (SHOULD) | v1.2's equality check sets gamma = 0 as well as beta = 0 (beta 0 with gamma 0.4 gives G = [-0.8, 0, 0.8] in Sol's check). Fallback counts are per metric: precipitation 16 / 4 / 6, the other metrics 16 / 3 / 5 at F = 1940 / 1960 / 1975. N <= 0 removes I at that origin; v1.3 is then `ref30 + G` (not average30), with a beta that can differ slightly from v1.1's (r6.1 follow-up; see the section 5 equality checks). The "same cross-city level" wording is dropped. |
| 6. Cap as an amplitude bound (SHOULD) | CAP_F is described only as an amplitude bound. It never bound in 1915-1975 (max N +0.743 F), so it does not explain the 1940-49 result. That cost is reported on its own (-0.4721 points). It is small because N was small (mean +0.19 F, increment mean 0.11 F) and the increment is a level, not an extrapolated slope. |
| Diagnostic runs (eval station, brief section 7) | None pre-declared. The questions they would answer are covered by the pre-1983 diagnostic rows. Any ablation that looked better would have to be re-scored as a candidate, adding attempts on the same three folds (section 4.3). |

## Changes in r5 and earlier (for the record)

- **r5:** Added v1.3 (national increment), the `reward.py`-exact bootstrap with Monte Carlo seeds, the section 1 script,
  overlap counts, the split equality check, and the no-official-ablations policy. The development-year numbers r5 quoted are
  withdrawn from the argument in r6.
- **r4:** Regenerated the walk-forward with `t + h <= F` asserted, froze v1.2 as the joint model, deferred the learner-based
  version, and measured runtime and determinism.
- **r3:** Removed post-origin station information, made the champion verdict the keep rule, added the `average30 + G` route,
  moved to the centred estimator, and added the walk-forward.

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
| v1.3 | 2. Regime sensitivity: the level bias flips sign by origin, so a constant correction fails | A level correction computed at each origin from its own recent national decade (N), damped by 0.5, bounded at 0.5 F, and absent whenever N <= 0 (I = 0 then; beta is still refitted) |

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

### 1.3 The national level increment (v1.3)

**What it responds to.**
- G has zero cross-city median adjustment, and the neighbour subtraction removes any uniform anomaly. So v1.1 and v1.2 forecast
  the 30-year mean's national level even when the most recent decade, observed at the origin, sits above it across the country.
- N measures that departure at each origin, from that origin's data only.
- The hypothesis is that a damped part of a positive N persists into leads 4-7. It is tested only with information available
  at each origin: the walk-forward below, which uses fit origins <= 1975 and targets <= 1982.

**Formula (unchanged since r5).** At origin O:
1. N = the cross-city median, over cities with a 30-year mean and >= 5 valid years in O-9..O, of
   `mean(valid years in O-9..O) - ref30`.
2. `I_temp = KAPPA * min(max(N, 0), CAP_F)`, with `KAPPA = 0.5` and `CAP_F = 1.0` F. It is added to annual_avg_temp at all
   leads, for cities with a 30-year mean.
3. `I_cdd,c = I_temp * rho_c`. `rho_c` is v0.5's regional CDD response: the OLS slope of annual CDD on annual mean temperature
   over O-29..O (>= 15 joint valid years), averaged over the city and its 15 nearest, NaN to 0, floored at 0.
4. I = 0 for precipitation and hottest day. When N <= 0, I = 0 at that origin and v1.3 is `ref30 + G` (not average30); its
   beta is refitted with the earlier increments in the training targets, so it can differ slightly from v1.1's.

**Provenance (author-reported).** I report that KAPPA, CAP_F and the floor were written into the script before it first
computed a v1.3 number, and that I have not changed them since. There is no timestamped pre-run declaration: the constants and
the first v1.3 results were committed together (r5, `07aae4a`).

Before r5 I had also seen development-year numbers passed on by reviewers and by insights: level residuals and biases by origin,
and the value of an undamped national shift at two origins. They influenced the decision to try a national level term. They are
not part of the argument below, and no parameter was set from them.

**Why these values, rather than estimated ones.**
- **KAPPA = 0.5.** Write N as persistent signal plus decadal noise. Then:
  - coefficient 0 is climatology;
  - 1 is full persistence of the recent decade's level;
  - about 2 is a linear trend extrapolated to leads 4-7 (the leads sit about 20 years after the centre of the 30-year window,
    the last decade's centre about 10).

  0.5 is halfway between climatology and persistence. It is the same 50/50 shrinkage convention the series uses for its combo
  weight, and it was not fitted.
- **Floor at 0.** This is a design choice, not a result of the pre-1983 data. Those data show national persistence in both
  directions: the decomposition's national coefficient is 1.32 in 1950-75. The unfloored diagnostic "raw national shift"
  (KAPPA 1, no floor, no cap) gains in the cooling blocks by carrying cooling forward. The floor makes v1.3 depart from v1.1
  only when N is positive. That keeps its exposure one-sided and keeps out the persistence of mid-century cooling, which cost
  the learners in v0.1 (lesson 1). It is kept as declared; switching to the unfloored form after seeing the walk-forward would
  be selection on it.
- **CAP_F = 1.0 F** is an amplitude bound: the increment can never exceed 0.5 F, whatever N is. It never bound in 1915-1975
  (max N +0.743 F). Whether it binds at the development origins or at 2018 is logged.
- **CDD through rho**, not a uniform degree-day shift. A uniform x F warming adds about x times each city's local response.

**Frozen-parameter walk-forward** (section 3.4). Coefficients are refitted at every fit origin with targets <= F; KAPPA, CAP_F
and the floor are unchanged. The 1915-1939 rows are **reused design-era evidence**: those years were also used for the section 1
tables.

| Fit origins | N > 0 at | N mean (range), F | I mean, F | v1.3 minus v1.1, final points | v1.3 > v1.1 at | v1.3 minus v1.1, live |
|---|---:|---|---:|---:|---:|---:|
| 1915-35 | 95% | +0.245 (-0.010 to +0.593) | 0.123 | **+1.7341** | 95% | +1.19 |
| 1925-39 | 100% | +0.442 (+0.123 to +0.743) | 0.221 | **+1.9924** | 87% | +2.32 |
| 1940-49 | 80% | +0.194 (-0.216 to +0.651) | 0.112 | **-0.4721** | | |
| 1950-62 | 0% | -0.241 (-0.335 to -0.134) | 0.000 | +0.0130 | | |
| 1963-75 | 0% | -0.428 (-0.539 to -0.300) | 0.000 | +0.0018 | | |
| all 61, 1915-1975 | | | | +0.4957 | | |

- The increment paid when a positive national gap persisted (1915-1939). It cost when warming stalled (1940-49). It did
  nothing when N was negative (1950-75).
- The 1940-49 cost is small because N was small (mean +0.19 F) and the increment is a level held constant over the leads. W
  extrapolates a 30-year slope roughly 15-25 years forward, and lost 9.22 points there.
- The early-era gain is mostly temperature and CDD. In 1915-35, v1.3's temperature points are +4.63 against +0.78 for v1.1,
  and CDD +3.88 against +0.79.

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
target city-year appears a median of 4 times: four leads from four pseudo-origins. The early-era fits (F = 1915-1939, r6) have
at least 8,882 temperature and 9,758 precipitation examples per fit.

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
  each, measured in the r6 run:

| F | run 1 | run 2 | SHA-256 v1.1+v1.2 | SHA-256 v1.3 | identical | equals walk-forward |
|---|---:|---:|---|---|---|---|
| 1925 | 8.2 s | 8.0 s | 1ff648444381413f | 5e962b59c0d607c4 | yes | yes |
| 1940 | 8.5 s | 8.5 s | d2654cef2645705a | 17e785718f01c097 | yes | yes |
| 1960 | 8.6 s | 7.3 s | f4b94357cdafa19f | a59a425d96ee5df4 | yes | yes |
| 1975 | 8.7 s | 7.5 s | 75f65a152199aef2 | 1cf734bd8393feb9 | yes | yes |

- The 1940-1975 digests are unchanged from r5, and the v1.1+v1.2 ones from r4.
- Times are 1-2 s longer than r5's 6.4-7.6 s, from machine load during this run.
- The adapter is not written yet; its first checks are in section 5. The harness limit is 900 s.

### 3.4 Chronological evidence (cutoff enforced)

For each fit origin F = 1915..1975, the coefficients are fitted with targets <= F (asserted before every fit). The forecast at
F is scored on F+1..F+7 against the 30-year average with the reward formula. Origins 1925, 1940, 1960 and 1975, recomputed from
a table with every year > F removed, reproduce the walk-forward exactly.

- **Main blocks (1940-1975):** as in r4/r5.
- **Early blocks (1915-1939, added in r6):** reused design-era evidence. They are the only pre-1983 period in which the
  national gap stayed positive.
- **Fit size:** at least 8,882 temperature examples and 9,758 precipitation examples per fit in 1915-35, and 14,810 / 15,775 in
  1925-39.

**Held-out final points against average30, by block of fit origins** (positive share in brackets):

| Candidate | 1915-35 | 1925-39 | 1940-49 | 1950-62 | 1963-75 |
|---|---:|---:|---:|---:|---:|
| **v1.1** | +0.8983 (95%) | +0.5109 (80%) | **+0.6582** (70%) | **+2.3694** (100%) | **+1.0970** (85%) |
| **v1.2** | +0.8731 (95%) | +0.5611 (80%) | **+1.0576** (80%) | **+2.4880** (100%) | **+1.0703** (85%) |
| **v1.3** | +2.6324 (100%) | +2.5033 (87%) | **+0.1861** (50%) | **+2.3823** (100%) | **+1.0989** (85%) |
| diag: v1.1 with 25-NN graph | | | +0.6621 (70%) | +2.4299 (100%) | +1.2032 (85%) |
| diag: v1.1 with precip=off | | | +0.7751 (80%) | +2.2058 (100%) | +1.0399 (85%) |
| diag: v1.1 + raw national shift (KAPPA 1, unfloored, uncapped) | +4.0087 (100%) | +3.2372 (73%) | -1.2126 (40%) | +5.1629 (100%) | +2.2861 (77%) |
| diag: v1.1 + 0.5 W | +3.7637 (95%) | +3.0636 (73%) | -2.8193 (40%) | +2.1311 (100%) | +1.3292 (92%) |
| ref: v0.5-like proxy, no G | +4.1415 (95%) | +1.3833 (73%) | -9.2177 (0%) | +3.7541 (100%) | +2.6594 (100%) |

**Live points by block:**

| Candidate | 1915-35 | 1925-39 | 1940-49 | 1950-62 | 1963-75 |
|---|---:|---:|---:|---:|---:|
| v1.1 | +1.79 | +1.59 | +2.32 | +3.00 | +2.56 |
| v1.2 | +1.78 | +1.51 | +2.71 | +2.97 | +2.46 |
| v1.3 | +2.98 | +3.91 | +1.87 | +3.02 | +2.57 |

**Final points by metric** (temp / precip / hottest / cdd):

| Candidate | 1915-35 | 1925-39 | 1940-49 | 1950-62 | 1963-75 | worst origin, 1940-75 / 1915-39 |
|---|---|---|---|---|---|---:|
| v1.1 | +0.78 / +0.66 / +1.37 / +0.79 | +0.55 / +0.22 / +1.17 / +0.10 | +1.38 / -0.47 / +1.30 / +0.42 | +4.40 / +0.65 / +1.49 / +2.94 | +1.37 / +0.23 / +1.03 / +1.76 | -1.31 / -1.40 |
| v1.2 | +0.76 / +0.71 / +1.26 / +0.77 | +0.54 / +0.42 / +1.17 / +0.12 | +1.69 / +0.03 / +1.76 / +0.75 | +4.49 / +0.81 / +1.62 / +3.05 | +1.52 / +0.16 / +0.84 / +1.76 | -0.83 / -0.98 |
| v1.3 | +4.63 / +0.66 / +1.37 / +3.88 | +4.38 / +0.22 / +1.17 / +4.24 | +0.04 / -0.47 / +1.30 / -0.13 | +4.40 / +0.65 / +1.49 / +2.99 | +1.36 / +0.23 / +1.03 / +1.78 | -2.32 / -3.16 |

The national increment by block, and v1.3 minus v1.1, are in section 1.3.

**Coefficients (beta, gamma) by fit origin** (data <= F only; 1940-1975 as in r4/r5):

| Metric | v1.1 beta 1940 / 1950 / 1960 / 1970 / 1975 (median of 36) | v1.2 (beta, gamma) 1940 / 1950 / 1960 / 1970 / 1975 (median) | v1.3 beta (median) |
|---|---|---|---|
| annual_avg_temp | 0.80 / 0.65 / 0.70 / 0.80 / 0.80 (0.725) | (0.80,0.45) (0.65,0.35) (0.75,0.40) (0.90,0.35) (0.85,0.40) (0.75, 0.40) | 0.80 / 0.65 / 0.70 / 0.85 / 0.80 (0.75) |
| total_precip_in | 0.80 / 0.60 / 0.55 / 0.50 / 0.45 (0.55) | (0.80,0.40) (0.60,0.75) (0.60,0.60) (0.55,0.55) (0.60,0.50) (0.60, 0.65) | same as v1.1 |
| hottest_day_temp | 0.60 / 0.60 / 0.65 / 0.65 / 0.60 (0.65) | (0.60,0.05) (0.60,0.65) (0.65,0.70) (0.65,0.50) (0.65,0.50) (0.60, 0.65) | same as v1.1 |
| cdd | 0.70 / 0.45 / 0.55 / 0.65 / 0.70 (0.55) | (0.70,0.55) (0.50,0.40) (0.60,0.45) (0.65,0.50) (0.75,0.25) (0.60, 0.45) | 0.65 / 0.45 / 0.55 / 0.65 / 0.70 (0.55) |

Precipitation's beta falls from 0.80 to 0.45, and it is the weakest component: `precip=off` is 0.12 points better in 1940-49
and 0.06-0.16 worse after.

**Promotion analogue, exactly as `reward.py`.** Each comparison uses:
- 2,000 multinomial draws, with seed 20261108 reset per triplet;
- only cities with a scored row resampled (205-208);
- the same draw for every origin, split and metric;
- `reward.py`'s four guards.

Main set: 22 triplets (F, F+7, F+14), F = 1940..1961. Dependence diagnostic: whole spatial blocks resampled, 5 x 5 degrees
(42 blocks) or 10 x 10 (14).

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

Early set (r6, reused design-era evidence): 11 triplets, F = 1915..1925, city resampling:

| Candidate vs champion | median reward | median P | min P | median 90% LB | worst floor | promote |
|---|---:|---:|---:|---:|---:|---:|
| v1.1 vs average30 | +0.6817 | 0.960 | 0.5605 | +0.0375 | -1.0176 | 7 / 11 |
| v1.3 vs average30 | +2.3770 | 1.000 | 0.9725 | +1.7685 | -0.1101 | 11 / 11 |
| v1.3 vs v1.1 | +1.3881 | 1.000 | 0.9790 | +1.2544 | -0.1101 | 11 / 11 |

**Monte Carlo error of the verdict at the bar:**
- v1.1, triplet (1960, 1967, 1974): P = 0.9500 at seed 20261108, binomial 90% interval [0.9420, 0.9580]. Seeds 1-30 give
  0.9475-0.9690, and P < 0.95 in 2 of 30.
- v1.2's weakest triplet: 0.9765, below 0.95 in 0 of 30 seeds.
- v1.3's weakest triplet (1941, 1948, 1955): 0.9390, below 0.95 in 26 of 30 seeds.

What this means:
- **v1.1 promotes in the 1940-61 analogue only at the bar**, and on 7 of 11 early triplets. A development promote is plausible,
  not assured. v1.2 has more margin in 1940-61.
- **Whether v1.3 promotes against v1.1 depends on the regime.** It promotes on 11 of 11 early triplets (positive N that
  persisted) and on 3 of 22 in 1940-61 (N mostly 0 or negative, or warming stalling).
  - In the 1940-61 set the two forecasts are identical or nearly so on most triplets: the increment is 0 at every fit origin
    after 1949, and exact ties count as neither better nor worse in `reward.py`.
  - This plan does not compute N at the development origins (design analysis stays <= 1982). It is logged in the first run.
- Spatial-block resampling lowers v1.1's median lower bound by 0.08-0.09 points, against 0.75 for the W-based proxy.
- The bootstrap cannot show:
  - the effect of triplet overlap;
  - that the kernel and half-lives were chosen on these years (section 4.2);
  - whether 1984-2004 behaves like any pre-1983 block.

---

## 4. Selection without peeking

### 4.1 Fixed in advance vs fitted in-fold

**Fixed:**
- Gaussian 600 km kernel, mass floor 1.0.
- Half-lives {3, 3, 3, 6}.
- Windows of 30 and 60 years, with minimum valid years 10 / 15.
- Combined centring.
- Grid 0-1, step 0.05, ascending ties. Fit leads 4-7 with `t + h <= O`. `MIN_COEF_ROWS = 2000`.
- G and I on all leads.
- Base = 30-year average.
- v1.3: `KAPPA = 0.5`, `CAP_F = 1.0`, floor at 0, `GAP_MIN_VALID = 5`, CDD conversion through v0.5's rho. Their freeze timing
  is author-reported (section 1.3).

**Fitted in each fold:** beta (and gamma) at the outer origin. N and rho are computed at the origin from its own data. No value
is chosen after looking at a development score.

### 4.2 Overlap between design years and the development folds

Design used pseudo-origins 1870-1975 with targets <= 1982. The early-era fits reach back to pseudo-origin 1870.

| Outer origin | coefficient-fit pseudo-origins | overlap with design pseudo-origins | targets in fit |
|---|---|---|---|
| 1983 | 1938-1979 | 1938-1975 (38 of 42) | 1942-1983 |
| 1990 | 1945-1986 | 1945-1975 (31 of 42) | 1949-1990 |
| 1997 | 1952-1993 | 1952-1975 (24 of 42) | 1956-1997 |

- The fixed hyperparameters were chosen on years that also train every development fold, which is allowed.
- The in-fold fit and the walk-forward are not independent evidence for the kernel, the half-lives or the v1.3 constants.
- This plan computed no number on 1983-2018. Development-year numbers seen by the author before r5 are disclosed in section
  1.3 and are not part of the argument.

### 4.3 The decision procedure (champion, nomination, submission)

One mechanical procedure governs which version is the champion, which is nominated for the integrator's one-shot assessment,
and which produces the submission.

1. **Record.** Before each version, run `uv run python code/evaluation/reward.py ladder`. Record the champion's run id and model
   in `docs/models/placeholder.md` as `C_before`. Other stations' official runs can change it between versions.
2. **Score and compare.** Score the version, then run `reward.py score <run_id> --against <C_before> --json`.
3. **Promote or not.**
   - The version becomes the champion only if the verdict is `promote`, with all four `reward.py` guards: final P(better)
     >= 0.95 with a positive delta, better on >= 2 of 3 origins, live P(worse) < 0.95, and every metric's final points >= -2.
   - Any other verdict leaves the champion unchanged.
4. **Order.** v1.1, then v1.2, then v1.3, each against the champion current at that moment. If v1.1 promoted, v1.2 and v1.3 are
   compared with v1.1. If v1.2 also promoted, v1.3 is compared with v1.2.
   - v1.3 always uses v1.1's graph definition, whatever v1.1's verdict. Against a v1.1 champion its comparison measures the
     increment including the coefficient refit it causes; against a v1.2 champion it also removes the long-window term, so it
     then measures both changes together, not the increment alone (see v1.4 in section 5).
5. **Carried forward** (brief sections 5-6). While nothing from this plan has promoted, a non-promoting version is recorded as
   "carried forward" over `Placeholder_v0.5` only if both its 90% lower bound and its reward beat v0.5's, against the same
   champion. That status only names the reference for a later round of this series. It never nominates. v0.5's values come from
   `reward.py score 20261108T141546Z-d0ac8f8b36 --json`, read before v1.1.
6. **Nomination and submission.** The nominee for the integrator's freeze and assessment, and the model behind the submission,
   is the **latest ladder champion** when the round ends. This plan nominates a v1.x version only if it is that champion.
7. **Disclosed alternative.** If v1.3 does not promote, its verdict, reward, interval, per-origin points, the N and I it used,
   and the walk-forward rows in sections 1.3 and 3.4 go to the integrator as a disclosed alternative. They are not a
   recommendation. If the champion is v1.1 or v1.2, the nomination note states as a fact that the model has zero cross-city
   median adjustment and no national level term.
8. **Never kept:** a `worse` verdict, and any diagnostic run.

The secondary comparisons (against v0.5, and against `control:average30` when the champion is something else) are reported and
never decide. Shared errors do not cancel exactly in a paired MAE comparison. There is no fixed point threshold.

**Diagnostic runs.** `line.py score --diagnostic` now seals runs that the ladder lists but never promotes (brief section 7). This
plan pre-declares **no** development diagnostics, for these reasons:
- The questions they would answer (kernel choice, the precipitation component, W exposure, an unfloored increment) are covered
  by the pre-1983 rows in section 3.4.
- Any ablation that looked better would have to be re-scored as a new candidate, which is another try at the same three reused
  folds.
- Seeing development scores for KAPPA or floor variants would invite tuning them on those folds.

So the plan has exactly three official development runs: v1.1, v1.2 and v1.3.

**Pre-registered expectations (development):**
- v1.1: +0.5 to +2.0 points against average30, positive at >= 2 origins. Hottest day and temperature carry the gain;
  precipitation is within ±0.5. If v1.1 is negative at >= 2 origins, the pre-1983 persistence does not hold in 1984-2004.
- v1.3 against v1.1 follows the sign of N at the development origins, as in the walk-forward:
  - where N <= 0 the two are nearly identical;
  - where N > 0 the difference has the sign of whether the national warm departure persisted over the next 4-7 years.

  No direction is predicted, because N at those origins has not been computed.

### 4.4 Trade-off

- **v1.1 and v1.2** have no national level term. In the walk-forward they never lose to average30 by much in any block (worst
  block +0.51), but they forgo the +1.7 to +2.0 points the increment earned when a positive national gap persisted.
- **v1.3** adds that level term:
  - it gains when a positive N persists (1915-1939: 11 of 11 early triplets promote against v1.1);
  - it costs when warming stalls (1940-49: -0.47 points; its weakest 1940-61 triplet fails in 26 of 30 seeds against
    average30);
  - it is neutral when N <= 0.

  Across all 61 pre-1983 fit origins it is +0.50 points above v1.1.
  Head to head against v1.1 on the 22 triplets of 1940-61, v1.3 promotes on 3/22 under the official city bootstrap, and
  on 2/22 and 0/22 when whole 5-degree and 10-degree spatial blocks are resampled (Grok's r6.1 check). The city figure is
  the one the ladder uses; the block figures show it is optimistic.
- **The ladder decides between them** (section 4.3). If v1.1 promotes first and the development origins look like 1940-61,
  v1.3 will probably be `within_noise` against it, and v1.1 stays the nominee with the disclosure in step 7. If they look like
  1915-1939, v1.3 can promote against v1.1.

---

## 5. Versions

Each version is one single-file adapter in `notebooks/Meet/`, committed, then scored from a clean commit:

```powershell
uv run python scripts/line.py status
uv run python code/evaluation/reward.py ladder                     # record C_before
uv run python scripts/line.py score --name Placeholder_v1.1 --adapter notebooks/Meet/placeholder_v1_1.py:forecast `
  --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3
uv run python code/evaluation/reward.py score <run_id> --against <C_before> --json
uv run python code/evaluation/insights.py <run_id> --json
uv run python scripts/line.py sync
```

| Version | Built on | Reads | Change | Finding |
|---|---|---|---|---|
| **Placeholder_v1.1** | `control:average30` | `data/city_info.csv` only | `G = beta_m centre(L_raw)`, all metrics, clipped | 1 |
| **Placeholder_v1.2** | `control:average30` (sibling, run unconditionally) | same | `G = centre(beta_m L_raw + gamma_m L_long_raw)`, joint fit | 1 |
| **Placeholder_v1.3** | v1.1's definition (whatever v1.1's verdict) | same | + national level increment I (section 1.3), beta refitted with I in the base | 2 |
| **Placeholder_v1.4** (conditional) | v1.2's definition | same | + the same national increment I as v1.3 (same KAPPA, cap and floor), beta and gamma refitted with I in the base | 2 |

- **Order:** v1.1, v1.2, v1.3, then v1.4 **only if v1.2 is the champion after v1.3** (r6.2, from Grok's r6.1 review). v1.4 is
  pre-declared here, before any development score, so that the increment is also tested on the version with more margin under
  the bar (v1.2's weakest pre-1983 triplet has minimum P 0.9765, 0 of 30 seeds below 0.95) without mixing in the removal of
  the long-window term. No `--diagnostic` runs, and no version is tuned on a development score.
- **Frozen constants:** KAPPA = 0.5, CAP_F = 1.0 F and the floor at 0 stay unchanged after development scores, for v1.3 and
  v1.4 alike. Their freeze before the first v1.3 number is author-reported (no older timestamp exists).
- **The learner-based v0.5 + G** stays deferred (r4).
- **Implementation:** v1.1 2-3 h (copy the script's functions); v1.2 and v1.3 about 30 min each on top.
- **Record per run:** beta (gamma) per metric, mean `|G|`, share of cities with G != 0, the centring median, the median forecast
  shift, and for v1.3 N, I, whether the cap bound, and the mean CDD increment.

**Checks before scoring:**
- **Rows with a 30-year mean:**
  - v1.1 with beta = 0 equals `control:average30` exactly.
  - v1.2 with beta = 0 **and gamma = 0** equals it exactly. With beta = 0 and gamma != 0, G is not zero: Sol's example
    gamma 0.4 gives G = [-0.8, 0, 0.8].
  - v1.3 with beta = 0 and `KAPPA = 0` equals it exactly.
  - v1.3 with `KAPPA = 0.5` at an origin with N <= 0 has I = 0 at that origin, but it is **not** guaranteed to equal v1.1:
    the increments at earlier pseudo-origins enter v1.3's coefficient fit, so its beta can differ (Sol's check at F = 1970:
    N = -0.464 F, temperature beta 0.80 for v1.1 against 0.85 for v1.3, forecasts differing by up to 0.067 F). The check
    there is I = 0 exactly, and that v1.3 equals `ref30 + G` with its own fitted beta.
- **Rows without a 30-year mean:** the counts are per metric. Pre-1983 they are precipitation 16 / 4 / 6 and the other metrics
  16 / 3 / 5 at F = 1940 / 1960 / 1975; development counts are measured at this check.
  - G = 0 and I = 0 there.
  - The forecast equals `reference(..., "average")`'s fallback exactly: the mean of the valid years in the 30-year window, else
    the full-history mean, else the metric-wide training median.
- G has zero cross-city median adjustment at every origin.
- No year > fit origin is read: cut the table at t and compare.
- The adapter's 1975 forecast on a pre-1983 cut equals the script's, to 1e-9.
- Two runs give identical SHA-256 digests. Then read `runtime_s` from the first scorecard.

---

## 6. How each lesson and finding is addressed

1. **Shared warming (lesson 1).**
   - v1.1 and v1.2 carry none: the median adjustment is 0, and the level effect is not claimed to be zero.
   - v1.3 carries a damped, floored national level term computed at each origin, not an extrapolated trend.
2. **The 1990 overshoot (lesson 2, finding 2).**
   - v1.1 and v1.2 have no level exposure: in 1940-49 they gain +0.66 / +1.06, against -9.22 for W.
   - v1.3 costs 0.47 there and gains +1.7 to +2.0 in 1915-1939.
   - Nothing is tuned on 1990, or on any development origin.
3. **Zero precipitation and hottest-day skill (lesson 3).** Hottest day gains +1.03 to +1.49 points in every block from 1915 to
   1975. Precipitation is weak (-0.47 to +0.66 by block for v1.1), with a falling beta.
4. **Noise (lesson 4).** At most two scalars per metric, fitted on overlapping examples (median reuse 4).
   - The v1.1 verdict is at the bar, with its Monte Carlo error reported.
   - Spatial-block resampling shows the city bootstrap is mildly optimistic for G.
5. **Reused years (lesson 5).**
   - Design used years <= 1982 only.
   - Exactly three official development runs, and no diagnostics.
   - Development-year numbers seen before r5 are disclosed and not used.
   - Only the 2004/2011 assessment is clean.

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Measured (pre-1983, cutoff enforced) | Sign in development results | Response |
|---|---|---|---|
| v1.1 just misses the promotion bar | weakest 1940-61 triplet P 0.950 (below 0.95 in 2 of 30 seeds); 7 of 11 early triplets | P(better) 0.93-0.96 | Section 4.3 step 3 decides; no reruns or seed changes |
| Local persistence weaker in 1984-2004 | beta stable except precipitation (0.80 to 0.45) | v1.1 negative at >= 2 origins | Report a negative result; no retuning |
| Increment on the wrong side at a development origin | 1940-49 cost -0.47 points; worst v1.3 origin -3.16 (1925-39) | v1.3 below v1.1 where N > 0 | Section 4.3 step 7 discloses it |
| Precipitation below -2 points | worst block -0.47; worst 1940-61 triplet floor -0.11; early triplets -1.02 (v1.1) | `reward.py` floor | Checked per run |
| Live clearly worse | v1.1 live +1.59 to +3.00; v1.3 +1.87 to +3.91 | live P(worse) | Measured per run |
| City bootstrap optimistic | spatial blocks lower median LB by 0.08-0.09 | | Official verdict unchanged; diagnostic reported |
| Hyperparameters chosen on evidence years | not measurable | promotes on development, fails assessment | One-shot assessment; 25-NN diagnostic shows small kernel sensitivity |
| Champion changes between versions | | | `C_before` recorded each time |
| Adapter differs from the script | | 1975 equality check fails | Fix before scoring |

**Dependencies:** none new (numpy, pandas), and no data release. **Compute:** 7-9 s per origin; 3 official runs.

---

## 8. Tracking

Every scored run is mirrored with `uv run python scripts/line.py sync`.

- **Names:** `Placeholder_v1.1`, `Placeholder_v1.2`, `Placeholder_v1.3`.
- **Tags:** `series=Placeholder`, `version`, `plan=A-graph`, `built_on`, `finding=<1|2>`, `label=official`, and
  `champion_before=<C_before run id>`.
- **Reward block against `C_before`** (secondary: against v0.5):
  - the verdict and the four guard booleans;
  - the final delta, 90% interval, P(better) and noise sd;
  - points by origin and by metric;
  - the live delta and live P(worse).
- **Insights per run** (`insights.py --json`):
  - the temperature city-offset share (finding 1);
  - the bias by origin for temperature and CDD (finding 2);
  - Central-North and East-North zone points (finding 3);
  - CDD best scaling (finding 4).
- **Scorecard metrics:** `skill.origin.*`, `skill.final.*`, `skill.live.*`, `bias.*`, `win_rate`, `worst_decile_city_skill`,
  `stability.range`, `runtime_s`.
- **Artifacts:** `placeholder_selection.json` with the coefficients, `|G|`, the centring median, the median forecast shift and,
  for v1.3, N, I, the cap flag and the CDD increment.
- **Model card:** `docs/models/placeholder.md` records `C_before`, the verdict and the section 4.3 outcome for each version.

---

## Appendix A. How the numbers were computed

- **Sections 1.3 and 3.4, coefficients, fit-sample and fallback counts, national increment, bootstraps, Monte Carlo, runtime
  and digests:** `docs/plans/creative/analysis/planA_walkforward.py` (about 6 minutes).
  - It parses the whole CSV, then keeps years <= 1982 (asserted), and asserts `max(training target year) <= F` before every
    fit, for all 61 fit origins 1915-1975.
  - Its `compare()` mirrors `reward.py`'s `compare()`: seed reset per call, only cities with a scored row, the same draws for
    all cells, and the four guards. The champion's skill is 0 when the champion is average30.
  - Spatial blocks are cells of `floor(lat / 5), floor(lon / 5)` and `floor(lat / 10), floor(lon / 10)`.
  - Early triplets are (F, F+7, F+14) for F = 1915..1925.
- **Section 1:** `docs/plans/creative/analysis/planA_section1.py` (about 3 minutes). It imports the estimator from the
  walk-forward script.
  - Correlations use anomalies against each city's 1941-1982 mean, with pairs needing >= 25 shared valid years. The e-folding
    length is a least-squares fit of `log corr = -d / L` (corr > 0.05, d < 2000 km).
  - The decomposition is OLS without intercept on pseudo-origins 1950-1975, leads 4-7.
  - The in-era and kernel tables pick the best coefficient on the same era, with the declared centred estimator and clipping.
    Kriging and the correlation graph use row-normalised weights with a valid-mass floor of 0.5.
  - Empirical-Bayes weights are `tau^2 / (tau^2 + noise)` per city, with noise = `var(R) sum(w^2) / (sum w)^2`.
  - The break level is the mean of R after the most significant single split (>= 3 years each side) when t > 4, else L.
- **W and the v0.5-like proxy** (diagnostic rows only) follow v0.5's definitions. rho is the same regional CDD response that W
  uses.
