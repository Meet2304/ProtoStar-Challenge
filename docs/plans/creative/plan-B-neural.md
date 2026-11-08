# Plan B: a regional state-space core with a gated city-departure learner

**Summary.** Plan B adds a regional Kalman local-level filter to the series. One numpy filter per metric runs over each city's neighbourhood (the city plus its 15 nearest cities). Its signal-to-noise ratio is estimated by bounded maximum likelihood on years up to each pseudo-origin. With a drift term phi, chosen strictly chronologically, the filter can also carry the regional warming slope. The core corrects temperature and CDD. At half weight it also gives the hottest day its first regional signal. Precipitation gets a longer-memory climatology, because the data show it is close to white noise. The Huber/boosting learners stay as a gated model of city departures. All selection (phi, then lambda) now scores one shared assembly function, the same one that makes the deployed forecast. A retrospective, fully nested test on five outer origins 1948-1976 (all targets ≤ 1983) ran this procedure. The stacked candidate (core, precipitation, hottest day, combo kept) scored +2.40 final points against the 30-year average and +0.93 against v0.5 (90% [+0.73, +1.14], better on 3 of 5 origins). Most of that gain comes from one fold (1962). On the development origins the city-bootstrap noise is about three times larger, so promotion there is uncertain, not likely. The neural deep ensemble is now a deferred version: with purged chronological early stopping it beats Huber on temperature and CDD city departures (+0.048 against +0.043 of departure error), but the gain is small.

Status: draft r2

Author: strategy planner B (Kiro, Claude Opus 5.5). r1 2026-11-08; r2 2026-11-08 after Sol's review (71/100). Family: neural and probabilistic temporal models trained globally across cities.
Scope: plan only. No adapter was implemented or scored, no development or assessment run was opened, and nothing was committed. The only sealed-run read was `reward.py score` on v0.5's existing run (E7).

---

## Changes in r2

| # | Sol's item | How it is handled |
|---|---|---|
| 1 | Strictly chronological lambda gate | Each gate origin I gets its own `phi(I)`, chosen on inner origins I−21..I−7, whose targets end ≤ I (§3.3). The outer phi is never reused. The same boundary holds for every other quantity (§2.2). It was run this way in the retrospective test (E8). |
| 2 | One assembly function | `assemble()` (§3.2) is the only path from components to a forecast. It is used for phi selection (lambda = 0), the lambda gate and deployment. The v0.5 split between `assemble()` and `blend_with_combo()` is removed as version `v2.0`. |
| 3 | Chronological evidence for the complete procedure | Computed: E8 runs the full nested procedure (MLE, phi, learners, lambda gate, combo blend, clipping) on outer origins 1948/1955/1962/1969/1976. It is labelled retrospective, because E1-E6 used the same years. The first implementation step must reproduce it, with a stop rule (§5.1). |
| 4 | E6 chronology | Redone as E6 r2: every training target ≤ 1965, test origins 1966-1976. The MLP uses purged chronological early stopping. The r1 table is withdrawn. The MLP now leads for temperature and CDD, which changes §1.3. |
| 5 | Kalman fallback and safeguards | The variance ratio is bounded inside the optimiser, rho = q/r ∈ [1e-5, 0.25], so the steady-state gain is ≤ 0.39. A concentrated likelihood is used, with a convergence and finiteness check. Every fallback is listed exactly (§3.1), and tests are specified (§5.2). The false r1 claim "q → 0 gives W = 0" is corrected. |
| 6 | Dependency tree | §5.3 gives every version its parent and the decision after it. The conditional core-only candidate now runs right after the core version. The exact per-metric effect of dropping combo is stated, including the doubled learner weight for the hottest day and precipitation. |
| 7 | Ablation-retention policy | Ablations are diagnostic only, never kept, and never trigger anything. The one conditional candidate (v2.3) has a trigger that does not depend on any ablation. The keep rule still compares with v0.5's reward (+0.834) and 90% lower bound (−0.603) (§4.3). |
| 8 | Promotion prospects from paired errors | E8 reports candidate-specific bootstrap bounds, P(better), origin wins and metric floors, computed with `reward.py`'s arithmetic. "+1.5 points" is now only an illustration of the noise at development sd 0.89 (§4.4). |
| 9 | Priorities | Order: assembly fix, then precipitation, then core, with an early benchmark of one complete nested origin and a determinism check. The neural version is last and runs only if the candidate is reproducible and well inside the runtime limit (§5.1, §5.3). |
| — | Factual corrections | The q → 0 limit with drift is v0.3-like extrapolation (§3.1, Sol's check reproduced in closed form). The old variance clip did not prevent last-year-like forecasts; the new bound does. The 1.6% precipitation figure is an iid-Gaussian calculation, not a general MAE ceiling. Clipping targets does not turn squared-error MLP training into a Huber loss. Sol's reproduction of E5 (precipitation +0.003654, 42/62 origins) is recorded. |

---

## 0. Evidence used, and how it was obtained

All numbers come from throwaway scripts that read only `data/annual_valid.csv` and `data/city_info.csv`, filtered to `year <= 1983`. The scripts were not committed, because only this file may be edited. E8 also imports, unchanged, `notebooks/Meet/placeholder_v0_5.py` and the harness functions `evaluation.core.build_fold`, `evaluation.adapters.reference` and `evaluation.official_score.score`. Every pseudo-origin has its targets at or before 1983, so nothing overlaps the development targets (1984-2004). Nothing under `private/`, nothing from origin 2004/2011 and nothing after 1983 was read. Skill uses the official arithmetic: `1 - pooled MAE / pooled MAE(30-year mean)`, over city-years with a valid target and a valid 30-year mean. "% origins" is the share of pseudo-origins where a method beats the 30-year mean.

**Evidence status.** E1-E6 informed the design, so E8, which uses the same years, is **retrospective** evidence, not independent validation. The only clean tests left are the development origins (already reused) and the integrator's one-shot assessment.

Regional series, used everywhere (§3.1): `c[i, y]` is the mean over city i and its 15 nearest cities (haversine, ties by `city_id`) of `value[j, y] − base[j]`, with at least 4 valid members. `base[j]` is city j's mean over `[t−59, t]`, with at least 20 valid years.

**E1. Where the variance is (1951-1983, anomaly vs each city's 1951-1980 mean).**

| Metric | sd | national mean explains | 15-NN mean (city excluded) explains |
|---|---:|---:|---:|
| annual_avg_temp | 1.23 F | 24% | 69% |
| total_precip_in | 7.29 in | 15% | 48% |
| hottest_day_temp | 3.00 F | 14% | 52% |
| cdd | 185 | 18% | 62% |

The shared signal is regional, not national.

**E2. Persistence (pooled lag correlations at lags 1, 3 and 7, 1931-1983; anomaly vs the trailing 30-year mean).**

| Metric | anomaly | city departure from region | national mean |
|---|---|---|---|
| annual_avg_temp | +0.36 +0.24 +0.22 | +0.64 +0.41 +0.20 | +0.37 +0.34 +0.40 |
| total_precip_in | +0.08 +0.05 −0.04 | +0.12 +0.09 +0.02 | +0.08 −0.05 −0.11 |
| hottest_day_temp | +0.24 +0.16 +0.07 | +0.23 +0.14 +0.09 | +0.37 +0.47 +0.20 |
| cdd | +0.34 +0.30 +0.15 | +0.53 +0.36 +0.18 | +0.41 +0.47 +0.19 |

**E3. Trend method in two eras (annual_avg_temp, pseudo-origins 1915-1976, leads 4-7).** The 1915-1945 era (early-century warming, then the 1940s plateau) is an analogue of the 1990 overshoot.

| Method | 1915-1945 | 1946-1976 | all |
|---|---:|---:|---:|
| v0.3 W: regional OLS-30 slope, floored, from the window centre | −0.0495 | −0.0062 | −0.0272 |
| combo blend (half trend-30, half mean-10) | −0.0321 | +0.0377 | +0.0038 |
| mean of last 10 years | +0.0092 | +0.0448 | +0.0275 |
| Kalman local linear trend (MLE), slope floored at 0 | −0.0040 | +0.0313 | +0.0142 |
| Kalman local level, windowed, no drift | +0.0047 | +0.0555 | +0.0308 |
| local level plus the floored OLS-30 slope as drift (phi = 1) | −0.0465 | +0.0485 | +0.0024 |

For CDD: v0.3 W −0.0306, half-weight windowed level +0.0214. The pooled MLE of a local linear trend put the slope variance at 0 (median of 62 origins, temperature and CDD), so the bend has to come from the level. The fitted level q/r for temperature was about 0 (1930), 0.018 (1960) and 0.032 (1976), a steady-state gain of 0 to 0.16. *Correction to r1:* r1 said q/r = 0 returns the window mean and so the 30-year average. That holds only with zero drift and identical masks (see §3.1).

**E4. Nested drift selection, core only (outer 1950-1976, inner O−21..O−7).** The rule always chose phi = 0. Temperature +0.0587, CDD +0.0339; fixed phi = 1 gave +0.0571 / +0.0286. Before 1983, phi is hardly identifiable. (E8 below repeats this inside the complete procedure: phi(I) = 1 was chosen at gate origins 1934 and 1941, so the rule does turn the drift on after a warming spell.)

**E5. Central values for precipitation and the hottest day (pseudo-origins 1915-1976 unless stated).**

| Estimator | precip | % origins | hottest day | % origins |
|---|---:|---:|---:|---:|
| median of 30 years (1940-1976) | −0.0097 | 24% | +0.0082 | 70% |
| median of last 10 years (1940-1976) | −0.0480 | | +0.0147 | |
| Hodges-Lehmann of 30 years | −0.0014 | 39% | +0.0026 | 71% |
| mean30 + kappa·sd30, kappa = pooled median z | −0.0007 | 47% | +0.0007 | 56% |
| **half 30-year mean + half all-history mean (≥ 40 valid years)** | **+0.0037** | **68%** | +0.0087 | 48% |
| windowed regional level, full weight | −0.0095 | 26% | +0.0151 | 69% |
| **windowed regional level, half weight** | −0.0030 | 34% | **+0.0143** | **71%** |

Sol reproduced the precipitation row independently: +0.003654, 42 of 62 origins. Median-type centres do not help precipitation (skew +0.30, and a 30-point median is noisier than a mean). *Qualified:* if yearly totals were iid Gaussian, a perfectly known mean would improve on the 30-year mean's MAE by at most `1 − 1/sqrt(1 + 1/30) ≈ 1.6%`. That is a reference scale for this idealised case, not a general MAE ceiling: skew, drift and station changes can move the bound either way. For the hottest day the half-weight level scores −0.0010 in 1915-1945 and +0.0290 in 1946-1976.

**E6 r2. City-departure learners with a purged chronology (replaces r1's E6).** Target: `y(t+h) − ref30 − (mean over the 15 neighbours of the same quantity)`, leads 4-7. Training rows: origins 1930-1961 with **target year ≤ 1965**. Test: origins 1966-1976 (targets ≤ 1983), so every training target precedes every test origin. *r1's version trained on origins through 1962 with 7-year labels through 1969, later than test origins 1966-1968, and used sklearn's random validation split. It is withdrawn.* MLP early stopping is chronological and purged: fit rows have target ≤ 1953, validation rows have origins 1954-1958 and target ≤ 1965, and the stopping rule is patience 10 on validation MAE. The best epoch count (3-12 across seeds and metrics) is then refit on all training rows under `threadpool_limits(1)`. Skill is relative to predicting 0:

| Metric | Huber | GBM (v0.5 settings) | MLP ensemble, 5 seeds (32-16) | half 15-year mean departure |
|---|---:|---:|---:|---:|
| annual_avg_temp | +0.0425 | +0.0394 | **+0.0476** | +0.0170 |
| total_precip_in | **+0.0030** | +0.0020 | −0.0008 | +0.0010 |
| hottest_day_temp | +0.0037 | +0.0010 | +0.0000 | **+0.0115** |
| cdd | +0.0431 | +0.0356 | **+0.0474** | +0.0141 |

With honest stopping, the MLP no longer trails: it beats Huber by about 0.005 of departure error on temperature and CDD. The MLP target is the robust-scaled departure clipped to ±3. *Correction to r1:* the loss is still squared error, which estimates a mean of the clipped target. Clipping only limits the influence of outliers. It does not make the loss Huber.

**E7. Current reward of v0.5** (`reward.py score 20261108T141546Z-d0ac8f8b36`, read-only). Final +0.834 points, 90% [−0.603, +2.308], P(better) 0.81, noise sd 0.890. Metrics: +2.41 / 0.00 / +0.37 / +0.56. Origins: 1983 +0.27, 1990 −2.35, 1997 +4.58. Live +2.90.

**E8. Retrospective, fully nested evaluation of complete candidates (new).**

*Procedure.* Outer origins O ∈ {1948, 1955, 1962, 1969, 1976}, targets O+1..O+7 ≤ 1983, from `build_fold` on the ≤ 1983 table (208 scored cities). Inside each fold, everything is chosen from data ≤ O exactly as §3 specifies: rho by bounded MLE at anchors ≤ t, `phi(F)` at each fit origin F from inner origins F−21..F−7 scored with `assemble(lambda=0)`, learners on pseudo-origins F−45..F−1, and lambda from gate origins O−21/−14/−7, each with its own phi(I) and the full `assemble()` including the combo blend and clipping. Paired city bootstrap with `reward.py`'s arithmetic: 2,000 multinomial city draws, seed 20261108, pooled MAE ratios per origin×split×metric, points = 100 × mean over metrics, then mean over origins. The `v0.5` row is the unchanged v0.5 code, including its gate/assembly mismatch.

| Candidate (all nested) | final pts vs average30 | 90% interval | P | sd | live | origins won | by origin 1948 / 1955 / 1962 / 1969 / 1976 | metric pts temp / prcp / hot / cdd |
|---|---:|---|---:|---:|---:|---:|---|---|
| v0.5 (as is) | +1.46 | [+0.93, +1.97] | 1.00 | 0.32 | +2.49 | 4/5 | +0.06 / +2.12 / +7.40 / +0.18 / −2.44 | +2.37 / 0.00 / +0.32 / +3.17 |
| precipitation base only, no learners | +0.08 | [−0.01, +0.16] | 0.94 | 0.05 | +0.10 | 5/5 | +0.16 / +0.05 / +0.06 / +0.13 / +0.01 | 0 / +0.33 / 0 / 0 |
| core (temp, CDD) + combo 0.5 | +1.83 | [+1.29, +2.37] | 1.00 | 0.33 | +2.88 | 2/5 | −0.24 / +2.45 / +10.35 / −0.86 / −2.56 | +3.19 / 0.00 / +0.34 / +3.78 |
| + precipitation base | +1.91 | [+1.34, +2.46] | 1.00 | 0.34 | +2.97 | 2/5 | −0.08 / +2.50 / +10.41 / −0.73 / −2.56 | +3.19 / +0.33 / +0.34 / +3.78 |
| **+ hottest-day level (main candidate)** | **+2.40** | **[+1.80, +2.96]** | 1.00 | 0.35 | +3.27 | 4/5 | +0.12 / +2.67 / +11.99 / +0.08 / −2.88 | +3.19 / +0.33 / +2.29 / +3.78 |
| same, combo weight 0 | +1.95 | [+1.61, +2.29] | 1.00 | 0.21 | +2.22 | 4/5 | +0.33 / +0.88 / +8.38 / +1.40 / −1.23 | +3.12 / +0.33 / +2.46 / +1.90 |
| core + precipitation + hottest, no learners, no combo | +1.75 | [+1.41, +2.09] | 1.00 | 0.20 | +1.94 | 3/5 | −0.08 / +0.75 / +8.38 / +0.89 / −1.18 | +2.81 / +0.33 / +2.21 / +1.65 |
| core (temp, CDD), combo 0 | +1.38 | [+1.06, +1.69] | 1.00 | 0.19 | +1.85 | 3/5 | −0.07 / +0.59 / +6.80 / +0.46 / −0.89 | +3.12 / 0.00 / +0.49 / +1.90 |
| learners only (no core, no combo) | +0.18 | [−0.05, +0.39] | 0.89 | 0.14 | +0.46 | 4/5 | +0.57 / +0.14 / +0.14 / +0.77 / −0.74 | +0.25 / 0.00 / +0.49 / −0.04 |

Paired against v0.5 (same draws):

| Candidate | Δ final pts | 90% interval | P(better) | origins won |
|---|---:|---|---:|---:|
| core + combo 0.5 | +0.36 | [+0.21, +0.51] | 1.00 | 2/5 |
| + precipitation | +0.44 | [+0.28, +0.61] | 1.00 | 2/5 |
| **+ hottest day (main candidate)** | **+0.93** | **[+0.73, +1.14]** | 1.00 | 3/5 |
| same, combo weight 0 | +0.49 | [+0.11, +0.89] | 0.99 | 4/5 |
| core + precip + hottest, no learners, no combo | +0.28 | [−0.16, +0.75] | 0.85 | 3/5 |

*Reading.* (a) The core, precipitation and hottest-day changes each add reward in the order planned, and the combined candidate beats v0.5 at P = 1.00. (b) **One fold dominates.** Without 1962, the main candidate's mean over the other four origins is −0.00 points against average30 (v0.5: −0.02). The combo-free variant's is +0.35, and it wins 4/5 against v0.5. So the gain in reward and the robustness across origins favour different variants. The keep rule (reward and lower bound) favours keeping combo. The verdict's origin rule favours dropping it. Both are tested (v2.5). (c) Every candidate loses at 1976. The combo-free variants lose about half as much. (d) The chronological gate chose phi(I) = 1 at gate origins 1934 and 1941, and mostly 0 elsewhere. The outer phi was 0 at every outer origin, except 0.25 at 1948 in the combo-free variants. When combo is kept, lambda was 0 for temperature and CDD at every origin. Without combo it was 0-0.75 for temperature and 0-0.25 for CDD. For the hottest day it was 0-0.75. (e) Runtime of the complete nested procedure: 21-52 s per outer origin in one process (v0.5: 19-25 s; core-only: 7-9 s).

---

## 1. Strategy and model-selection logic

### 1.1 Choice

`forecast = assemble(base, combo, W, shift, lambda, combo_weight)` (§3.2), with:

1. **A structured probabilistic core:** the windowed local-level filter on `c[i, ·]` over `[t−29, t]`. For temperature and CDD it carries a known drift `phi · d_i`, where `d_i` is the floored regional OLS-30 slope. Its variance ratio comes from a bounded, pooled MLE. The correction is `W_m = kappa_m · (L_t − cbar_i + phi · d_i · h)`.
2. **A precipitation base:** `B_p = 0.5 · ref30 + 0.5 · (all-history mean)`, entered as `W_p = B_p − ref30`.
3. **The gated city-departure learner** (v0.5's Huber + GBM), fitted on `y − ref30 − W` with the cross-city median removed, then scaled by lambda.

### 1.2 Why this suits the data

- **Regional signal, heavy noise (E1, E2).** The predictable part at leads 4-7 is a regional level plus slow city departures. The filter estimates how fast the regional level moves from the same history, by likelihood, so no window length is tuned. The slope variance is 0 because the likelihood puts it there (E3).
- **The 1990 overshoot (lesson 2), corrected in r2.** With drift, the filtered level behaves like a weighted line with slope phi·d. Its lever arm is about `phi · d · (effective lag + h)`. The effective lag is about `(1 − K)/K` years at steady-state gain K, and it approaches the window centre (14.5 years) as K → 0. So **phi = 1 with a small gain reproduces v0.3's extrapolation.** Damping comes from two sources: phi is chosen chronologically (phi = 0 removes the slope term), and with q/r around 0.03 (K about 0.16) the lag is about 5 years instead of 14.5. In the analogue era this cut the temperature loss from −0.050 to +0.005 (E3). 1990 itself can no longer validate any damping (§4.5).
- **MAE scoring.** The filter estimates a conditional mean. Temperature is nearly symmetric (skew +0.06). Median-type corrections for the skewed metrics did not help (E5). The learners keep MAE-type losses (Huber, `absolute_error`).
- **Small data and reused years (lesson 5).** The core has no learned weights: 1 bounded variance ratio per metric and anchor, fitted by likelihood, which never uses lead-4-7 outcomes. The only outcome-selected quantities are phi (5-point grid) and lambda, both strictly chronological (§2.2). `kappa` and the precipitation weight were fixed from data ending in 1983 (E3, E5).

### 1.3 Alternatives considered

| Candidate | Verdict | Reason |
|---|---|---|
| Global neural forecaster as the main model (MLP, temporal convolution, N-BEATS-style, attention) | Rejected as the core | E8: the learners alone give +0.18 points, the core gives about +1.2 more. The predictable signal is a regional level, which a filter estimates with 1 parameter. A large network trained on 1930-1980 learns that era's cooling (lesson 1). |
| MLP deep ensemble as a departure member | **Deferred version v2.6** (was an ablation in r1) | E6 r2: with purged chronological stopping it beats Huber on temperature and CDD departures (+0.0476 against +0.0425; +0.0474 against +0.0431). In points that is small: departure MAE is about 60% of total MAE, and lambda is often 0. |
| Local linear trend with stochastic slope | Rejected | MLE slope variance 0 (E3). Worse than the windowed level (+0.014 against +0.031). |
| Bayesian hierarchical state-space model (PyMC/Stan) | Rejected for time | The pooled 1-parameter fit already captures the noise structure. MCMC would be needed at every inner origin, and neither package is installed. |
| Gaussian process in time | Rejected | A state-space model with a chosen kernel. Its length scale would be one more quantity tuned on reused years. |
| Distribution-aware centres | Rejected | E5: no precipitation gain. The hottest-day median gains coincide with an era drift. |
| `enhance-v2` ocean indices | Not used | Not gate-eligible, so a gain could not become champion. ENSO has no known skill at leads 4-7. If an exploratory run is ever made, it must load indices through `code/enhance_ocean.load_ocean_for_origin(release, origin)`, as the updated brief requires, and its gain cannot be carried to an official version unless the indices come from a gate-eligible release. |
| PyTorch | Not added | The core is numpy and scipy. The deferred MLP is sklearn on CPU. No lock change. |

---

## 2. Targets, inputs and time boundaries

### 2.1 What each metric predicts

| Metric | `W_m` (cities × leads 1-7) | learner target | combo half gets W? |
|---|---|---|---|
| annual_avg_temp | `1.0 · (L − cbar + phi·d·h)` | `y − ref30 − W`, cross-city median removed per (t, h) | no (combo keeps its own trend/mean-10) |
| cdd | `0.5 · (L − cbar + phi·d·h)` | same | no |
| hottest_day_temp | `0.5 · (L − cbar)` (phi does not apply) | same | yes (combo's hottest day is ref30) |
| total_precip_in | `0.5 · (allmean − ref30)` if ≥ 40 valid years ≤ t, else 0 | same | yes (combo's precipitation is ref30) |

`base` = `reference(..., method="average")`: ref30, with the harness fallbacks for cities without a 30-year mean (those cities are unscored). CDD and precipitation are clipped at 0 after blending.

### 2.2 Time boundary for every quantity

| Quantity | Data used | Boundary |
|---|---|---|
| `base[j]`, `c`, `L`, `cbar`, `d`, `ref30`, all-history mean at pseudo-origin t | columns ≤ t of the table already cut at the fit origin | ≤ t |
| `rho` used at t | MLE at anchor `a = 5·floor(t/5)` on years `[a−99, a]` (falls back to a−5, a−10) | ≤ a ≤ t |
| `phi(F)` for a fit origin F (outer O or gate origin I) | inner origins F−21..F−7, targets ≤ F | ≤ F |
| learner rows for fit origin F | pseudo-origins t ∈ [F−45, F−1], targets t+h ≤ F, W(t) with phi(F) | ≤ F |
| `lambda` at outer O | gate origins I = O−21, O−14, O−7, each refitting with its own `phi(I)` and its own rows ≤ I, scored on I+1..I+7 ≤ O | ≤ O |
| neighbours | static coordinates | static |

Caches (rho by (metric, anchor), W by (t, phi), folds by origin) are keyed by quantities whose inputs are a fixed cut of the history. A cached value is identical whichever fit asks for it, so caching never crosses a boundary. A unit test perturbs every value after t and checks that `W(t)`, `rho(a ≤ t)` and the phi(I) table do not change.

---

## 3. Training procedure

### 3.1 Core with explicit safeguards (numpy, vectorised over the 210 cities)

Year ranges are inclusive. Every array is cut at t before this code sees it.

```python
# Fixed in advance
NB_K = 15; BASE_YEARS = 60; BASE_MIN = 20; REG_MIN = 4; WINDOW = 30; MIN_WINDOW_OBS = 10
MLE_YEARS = 100; MIN_SERIES_YEARS = 30; MIN_SERIES = 20; ANCHOR_STEP = 5; ANCHOR_BACKOFF = (0, 5, 10)
RHO_BOUNDS = (1e-5, 0.25)          # steady-state gain K = (-rho + sqrt(rho^2 + 4 rho)) / 2 in [0.0032, 0.39]
KAPPA = {"annual_avg_temp": 1.0, "cdd": 0.5, "hottest_day_temp": 0.5}
DRIFT_METRICS = ("annual_avg_temp", "cdd"); PHI_GRID = (0, .25, .5, .75, 1); PHI_TIE = 0.002
PRCP_WEIGHT = 0.5; PRCP_MIN_YEARS = 40

def regional_series(M, t, start):                        # (cities, start..t)
    base = nanmean(M[:, t-59:t], min=BASE_MIN)
    return nanmean((M[:, start:t] - base[:, None])[self_and_15_nn], axis=members, min=REG_MIN)

def level_filter(c, rho, drift):
    """Local level with known drift, r = 1 (scaled), exact diffuse start: the first valid obs sets L = y, P = 1
    and is excluded from the likelihood. Gaps: predict only (L += drift, P += rho), no update.
    Returns the filtered level at the last column and the concentrated log-likelihood."""
    for each column k:
        L = where(started, L + drift, L); P = where(started, P + rho, P)
        upd = observed & started; F = P + 1; v = y - L; K = P / F
        accumulate sum(v^2/F), sum(log F), N over upd
        L, P = where(upd, (L + K v, P (1 - K)), (L, P)); first obs: L = y, P = 1
    r_hat = sum(v^2/F) / N
    loglik = -0.5 * (N log(2 pi r_hat) + sum(log F) + N)    # -inf if N == 0 or r_hat <= 0
    return L, loglik

def fit_rho(M, a):                                        # cached by (metric, a)
    c = regional_series(M, a, max(first_year, a-99)); keep = rows with >= MIN_SERIES_YEARS valid values
    if keep.sum() < MIN_SERIES or nanvar(c[keep]) <= 1e-12: return None          # too few series / constant
    res = minimize_scalar(lambda lr: -level_filter(c[keep], exp(lr), 0)[1],
                          bounds=log(RHO_BOUNDS), method="bounded", options={"xatol": 1e-3})
    if not res.success or not isfinite(res.fun): return None                   # no convergence
    record(at_bound = |res.x - log bound| < 1e-3)
    return exp(res.x)

def rho_for(M, t): first non-None of fit_rho(M, 5*(t//5) - b) for b in ANCHOR_BACKOFF, else None

def W(M, t, phi, metric):                                 # cities x 7
    if metric == "total_precip_in":
        n_all, allmean = count/mean of valid years <= t
        return where(isfinite(ref30) & (n_all >= PRCP_MIN_YEARS), PRCP_WEIGHT * (allmean - ref30), 0)
    rho = rho_for(M, t)
    if rho is None: return zeros                          # FALLBACK F1, metric-wide at this t
    win = regional_series(M, t, t-59)[:, -WINDOW:]
    p = phi if metric in DRIFT_METRICS else 0
    d = max(0, mean over self+15 of the city OLS-30 slopes) if p > 0 else 0
    L, _ = level_filter(win, rho, p * d)
    both = valid(M[:, t-29:t]) & valid(win)
    cbar = mean(win over both) if both.sum >= MIN_WINDOW_OBS else nan
    out = KAPPA[metric] * (L - cbar + p * d * h)          # for h = 1..7
    return where(valid(win).sum >= MIN_WINDOW_OBS & isfinite(out) & isfinite(ref30), out, 0)   # F2, per city
```

**When the core correction is 0, exactly.** F1 applies to a whole metric at one pseudo-origin: no anchor among a, a−5, a−10 has at least 20 eligible series, or the regional series is constant, or the optimiser fails or returns a non-finite value. F2 applies to one city: its window has fewer than 10 valid regional values, or fewer than 10 years where both the city and its region are valid, or ref30 is missing, or the result is non-finite. Then W = 0, and that city's forecast is `assemble()` with W = 0. That forecast equals the 30-year mean only if lambda = 0 and combo weight = 0. The adapter never falls back silently in any other case: any other exception fails the run. The artifact counts F1 and F2 cells per metric and origin.

**What the limits really are (corrects r1).** For a complete window and `rho → 0`:
- drift 0 and identical masks: `W = 0`;
- drift `phi·d`: `W = kappa · phi·d · ((t − mean window year) + h) = kappa · phi·d · (14.5 + h)`. That is v0.3's extrapolation. Sol's check (30 constant values, q = 0, r = 1, drift 0.03, h = 5) gives 0.03 × 19.5 = 0.585;
- drift 0 but differing masks: `W = kappa · (mean of c over its valid years − mean of c over years where city and region are both valid)`. This is not 0. It corrects the regional part of ref30 for the city's missing years, and is intended.

The bound `rho ≤ 0.25` caps the gain at 0.39. The level at t is then an average with an effective memory of at least about 1.6 years, and a last-year forecast (gain → 1) is impossible. All pre-1983 fits fell in 0-0.066 (temperature 0-0.032, precipitation 0.012-0.015, hottest day 0-0.038, CDD 0.012-0.066), so the bound binds only on outliers. Hits on a bound are logged.

### 3.2 The one assembly function

```python
COMBO_GETS_W = {"total_precip_in", "hottest_day_temp"}

def assemble(base, combo, W, shift, lam, combo_weight):
    """The only path from components to a forecast. Called by select_phi (shift=None, lam=0),
    by the lambda gate (every lambda in the grid) and by forecast() for the outer origin."""
    out = []
    for m in METRICS:
        own = base[m] + W[m] + lam[m] * (shift[m] if shift is not None else 0)
        if combo_weight > 0:
            comb = combo[m] + (W[m] if m in COMBO_GETS_W else 0)
            pred = (1 - combo_weight) * own + combo_weight * comb
        else:
            pred = own
        if m in {"total_precip_in", "cdd"}: pred = max(pred, 0)
        out.append(pred)            # W and shift are 0 for cities without ref30 (unscored); NaN is an error
    return validate_predictions(concat(out), targets)
```

`blend_with_combo()` is deleted. A test checks that `forecast()` returns exactly `assemble(...)` with the selected lambdas. A second test replaces `assemble` with a stub and checks that both the gate and deployment call it.

### 3.3 Selection order inside one fit origin F

```python
def select_phi(history_le_F, F):                    # targets <= F
    for I in F-21 .. F-7 (skip if build_fold raises: too few scored cities):
        fold = build_fold(history_le_F, cities, I); base, combo = reference(avg), reference(combo)
        for phi in PHI_GRID:
            s[phi][I] = mean final skill over DRIFT_METRICS of assemble(base, combo, W(..., I, phi), None, 0, CW)
    phi = smallest p with mean_I s[p] >= max - PHI_TIE   (0 if no inner origin)

def forecast(train, targets, O):
    phi_O = select_phi(train, O)
    for I in (O-21, O-14, O-7):                      # the lambda gate
        tr_I = train[year <= I]; phi_I = select_phi(tr_I, I)          # its own phi, never phi_O
        shift_I, W_I = fit_learners(tr_I, I, phi_I)
        for lam in LAMBDA_GRID: score(assemble(base_I, combo_I, W_I, shift_I, lam, CW), truth_I)  # truth_I <= O
    lambdas = v0.5 rule per metric (smallest within 0.005 of best; 0 if best <= 0)
    shift_O, W_O = fit_learners(train, O, phi_O)
    return assemble(base_O, combo_O, W_O, shift_O, lambdas, CW)
```

Phi is chosen before lambda, by design. The phi objective is the exact deployed forecast at lambda = 0, so it needs no learner fits. Lambda is then chosen given phi(I).

### 3.4 Learners, seeds, determinism, runtime

- Learners as in v0.5: `HuberRegressor(epsilon=1.35, alpha=1.0)` and `HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=500, l2_regularization=5.0, early_stopping=False)`, on rows from pseudo-origins [F−45, F−1] with targets ≤ F.
- Deferred MLP (v2.6), exactly as in E6 r2: `MLPRegressor((32, 16), alpha=1e-2, learning_rate_init=1e-3)`, five seeds `seed + k`. Trained with `partial_fit` epochs. Purged chronological early stopping inside each fit origin F: fit rows have target ≤ F−12, validation rows have origins F−11..F−7 and targets ≤ F, patience 10. The best epoch count is refit on all rows ≤ F. Robust-scaled target clipped to ±3 (squared loss). Everything runs under `threadpool_limits(1)`.
- Determinism: bounded Brent optimiser (deterministic), elementwise numpy over sorted `city_id`, stable neighbour ties, seed 42. A test runs one origin twice and requires identical outputs.
- Runtime: E8 measured 21-52 s per outer origin for the complete nested procedure (one process, history to ≤ 1976). Development origins have 7-21 more years of history, so the estimate is ≤ 90 s per origin, against the 600 s `evaluate.py` default and the 900 s brief limit. The MLP adds about 20 fits × 5 seeds, an estimated 150-250 s.

---

## 4. Selection without peeking

### 4.1 Fixed in advance

All constants in §3.1 and §3.4, the v0.5 learner and gate settings (`LAMBDA_GRID` 0..1 by 0.25, margin 0.005, offsets 21/14/7, `TRAIN_YEARS` 45), `COMBO_GETS_W`, and each version's combo weight.

### 4.2 Chosen on inner origins

`phi(F)` per fit origin and `lambda_m` per outer origin, as in §3.3. `rho` is estimated, not selected.

### 4.3 Keep rule (frozen; brief §5.4, `evaluation/REWARD.md`)

1. **Promote** if `reward.py` returns `promote` against the current champion (now `control:average30`): P(better on final) ≥ 0.95, a win on ≥ 2 of 3 origins, live not clearly worse, no metric below −2 points. The version becomes champion and the kept version.
2. **Otherwise keep** over the current kept version (now `Placeholder_v0.5`: reward **+0.834**, 90% lower bound **−0.603**) only if its 90% lower bound is higher AND its reward is higher, both measured against the same champion. If the champion changes, rerun `reward.py score <kept run>` against the new one first.
3. **Otherwise** it is not kept.
4. **Ablations are diagnostic only.** They are never kept, never become the parent, and never trigger a version. The only conditional candidate, v2.3, is triggered solely by v2.2's keep decision, and is scored and judged by rules 1-3 like any version.
5. The run must be labelled `official` by `line.py`. The manifest binds only `data/city_info.csv`. Although `line.lock.json` now names `enhance-v2`, no v2.x adapter reads it. A run that comes out `exploratory` does not count.
6. `within_noise` is reported as "not progress" even when the reward is higher. Origin-1997 skill is reported only as a diagnostic.

### 4.4 Promotion prospects (from E8 paired errors; not a forecast of the development result)

In the retrospective folds, v0.5 and the main candidate would both promote (P = 1.00, 4/5 origins), at a city-bootstrap sd of 0.32-0.35. On the development origins, v0.5's sd is 0.89 (E7), with 3 origins. The scale relation is unknown. One illustration: v0.5's development reward is 0.57 × its retrospective reward. Applied to the main candidate's +2.40, that gives about +1.4 development points. At sd 0.89, P ≥ 0.95 needs about 1.645 × 0.89 ≈ +1.5. So the main candidate is near the line: **promotion is plausible, not expected.** The concentration in 1962 (E8b) is the main reason for caution. The equivalent development fold is 1997 (+4.58 for v0.5). The combo-free variant has lower noise (sd 0.21) and wins more origins, which the verdict needs. That is why v2.5 is scored even though the reward-based keep rule predicts it will not be kept.

### 4.5 Honesty notes

The development years have been seen many times, including through the brief ("cool 1992-1997"). 1990 can show whether the overshoot shrinks, but it cannot validate the change. E8 used the same years that set kappa and the precipitation weight, so it shows the procedure works end to end without leakage. It is not an out-of-sample estimate.

---

## 5. Implementation and versions

### 5.1 Implementation order and stop rule (under one day)

1. **v2.0 code (about 2 h):** copy `placeholder_v0_5.py` to `placeholder_v2_0.py`. Replace the gate and deployment paths with `assemble()` (§3.2), add the tests of §5.2, and leave the model otherwise unchanged (v0.3's W stays).
2. **Core and precipitation (about 3 h):** `fit_rho`, `level_filter`, `W`, `select_phi` and the chronological gate in a version file. Unit tests in §5.2.
3. **Retrospective replication and benchmark (about 1 h, before any `line.py score`):** a driver script `notebooks/Meet/retro_v2.py` (committed with the adapter) calls the adapter's `run_forecast` on outer origins 1948-1976 with the ≤ 1983 table and prints the E8 rows. **Stop rule:** if the adapter's main-candidate result differs from E8 by more than 0.10 points on any origin, it is a bug: fix it before scoring. If the corrected adapter's paired Δ against v0.5 has a 90% lower bound ≤ 0, stop the core line, score only v2.0/v2.1 and report. Also time origin 1976 (must be < 120 s), and run it twice (identical output required).
4. Score versions in the order of §5.3. Implement v2.6 (MLP) only after the kept candidate passes step 3, reruns identically, and stays under 300 s per development origin in its `run.json`.

### 5.2 Tests (in `test_placeholder_v2_<n>.py`)

- `level_filter`, rho = 0, drift 0, complete window: W == 0 (1e-12).
- rho = 0, drift d, complete window: `W = kappa·d·(14.5 + h)` (Sol's case: 0.585).
- rho = 0, drift 0, city mask missing 5 years where the region is valid: W equals the closed form in §3.1.
- Gaps: a series with NaNs equals a brute-force scalar Kalman written as a plain loop (1e-10).
- Constant series and fewer than 20 eligible series: `fit_rho` returns None, so W = 0 (F1), recorded in the artifact.
- Optimiser at a bound: flagged. Non-finite input row: F2 for that city only.
- Causality: perturb all values after t; `W(t)`, `rho(a ≤ t)` and the `phi(I)` table are unchanged.
- Assembly identity and call-path tests (§3.2). Determinism: two runs give identical frames.

### 5.3 Versions and dependency tree

Each version is one adapter `notebooks/Meet/placeholder_v2_<n>.py` with `MODEL_NAME = "Placeholder_v2.<n>"` and assets `notebooks/Meet/placeholder-v0.1-assets.json`. Score command: `uv run python scripts/line.py score --name Placeholder_v2.<n> --adapter notebooks/Meet/placeholder_v2_<n>.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3`. Ablations use `--name Placeholder_v2.<n>-<ablation> --params <json>`. "Parent" is the code a version is built from. "Incumbent" is the kept version it is judged against (§4.3).

```
v0.5 (kept; incumbent)
 └─ v2.0  assembly fix                                parent v0.5          [infrastructure: becomes the parent regardless; judged and reported]
     └─ v2.1  precipitation base                      parent v2.0
         └─ P1 := v2.1 if kept else v2.0
             └─ v2.2  regional core, temperature + CDD   parent P1
                  ├─ kept      → P2 := v2.2
                  └─ not kept  → v2.3 core-only replacement   parent P1
                                   ├─ kept     → P2 := v2.3
                                   └─ not kept → P2 := P1, CORE_DROPPED
                 └─ v2.4  hottest-day level              parent P2  (uses level_filter; valid with or without the temp/CDD core)
                     └─ P3 := v2.4 if kept else P2
                         └─ v2.5  combo weight 0          parent P3  (skipped if P3 has no combo half)
                             └─ P4 := v2.5 if kept else P3
                                 └─ v2.6  departure filter + MLP member (deferred; §5.1 step 4)   parent P4
```

**v2.0: one assembly function.** Change: §3.2 replaces `assemble` + `blend_with_combo`. The gate now scores the blended forecast it deploys. Why: Sol item 2. The gate's selected lambdas may change, and that is the whole effect. It becomes the parent whatever its score, because a gate that scores a different forecast than it deploys is a defect, not a modelling choice. It is still scored and judged by §4.3, so if it is not kept, the incumbent stays v0.5 for comparisons. Ablations: none.

**v2.1: precipitation base.** Change: `W_p` (§2.1) in both halves. Why: lesson 3. E5 gives +0.0037 at 68% of origins, positive in both eras; E8 gives +0.08 points, winning all 5 origins, noise sd 0.05. The gain is small but spread across cities, so it raises the lower bound almost one-for-one. Expected: precipitation +0.3, headline +0.08. Ablations (diagnostic): weight 0.75 on ref30; long mean over the last 60 years.

**v2.2: regional core for temperature and CDD.** Change: v0.3's `warming_term` is replaced by `W` (§3.1) with phi per §3.3. The learners refit on `y − ref30 − W`. CDD uses its own regional series and slope. Why: lessons 1-2 (E3, E8). Expected (E8, with combo): reward +0.36 over the comparable v0.5 configuration, better on only 2 of 5 retrospective origins, so a `within_noise` result is likely. Ablations (diagnostic): `phi=0` fixed, `phi=1` fixed, `combo_weight=0`.

**v2.3: core-only replacement (conditional, decided right after v2.2).** Trigger: v2.2 not kept. Nothing else. Change from P1: the temperature and CDD forecasts become `ref30 + W` (same phi rule), and their combo half and learner contribution are dropped (lambda fixed at 0 for those two metrics). The hottest day and precipitation stay as in P1. Why: E8's no-learner, no-combo candidate has the lowest noise (sd 0.20) and wins 3/5 against v0.5. If the core fails inside v0.5's blend, this tests whether it works on its own. Same keep rule. Ablations: none. If it is not kept, the core is dropped (`CORE_DROPPED`), and later versions must not assume a core for temperature or CDD.

**v2.4: hottest-day regional level.** Change: `W_hot = 0.5·(L − cbar)` (phi does not apply), in both halves, with the learners refitted. Why: lesson 3. E2 (lag-3 national +0.47), E5 (+0.0143), E8 (+0.49 points on top of core and precipitation, which raises origin wins from 2/5 to 4/5). It does not depend on v2.2: under `CORE_DROPPED` it uses the same filter code for this metric alone. Expected: hottest day +1.0 to +2.0 points. Ablations: `kappa_hot=1.0`; W in the own half only.

**v2.5: drop the combo half.** Change: `combo_weight = 0` in `assemble()`. Exact effect per metric, with everything else equal:
- annual_avg_temp, cdd: the core W goes from weight 0.5 to 1. Combo's `0.5·trend30 + 0.5·mean10` term (weight 0.5) is removed. The learner term `lambda·shift` goes from weight 0.5 to 1.
- hottest_day_temp, total_precip_in: `base + W` is unchanged (both halves already carry it). The learner term `lambda·shift` doubles from 0.5 to 1 whenever lambda > 0. For the hottest day lambda was 0.25-0.75 in E8; for precipitation it has been 0 everywhere.
- Because the gate scores the full assembly, the lambdas are re-selected under the new weight.
- Clipping is unchanged.

Why: robustness. E8's combo-free stack wins 4/5 origins against both average30 and v0.5, with noise sd 0.21 against 0.35, and its 1976 loss is −1.23 against −2.88. Expected: by the reward-based keep rule probably **not kept** (E8: +1.95 against +2.40). It can still be the only version that meets the verdict's origin rule, so it is scored. Skipped if P3 has no combo half. Ablations: `combo_weight=0.25`; `kappa_cdd=1.0`.

**v2.6: departure filter and MLP member (deferred).** Change: the learner target becomes the city's departure from its region, with the 15 recent departures and their 5- and 15-year means as extra features. The MLP ensemble (§3.4) is added as a third member. Why: E2 and E6 r2. Expected: +0.1 to +0.3 points, mostly temperature and CDD, and only where lambda > 0. Ablations: MLP replaces GBM; no MLP (target change only).

Maximum: 7 versions, 11 ablations. With E8's runtimes, each run takes about 2-5 minutes.

---

## 6. How each lesson is addressed

| Lesson | Response |
|---|---|
| 1. Warming learned as cooling | The regional signal comes from a filter with no learned weights. The learners see median-removed targets. |
| 2. 1990 overshoot | The lever shrinks from about 14.5 + h years to about `phi·((1−K)/K + h)`, with phi chosen chronologically. The analogue era went from −0.050 to +0.005 (E3). 1990 is reported per metric. phi = 1 with a small gain would reproduce v0.3 (§1.2), and the artifact logs both phi and the gain. |
| 3. Zero precipitation skill, near-zero hottest day | v2.1 (+0.08 points, 5/5 origins in E8) and v2.4 (+0.49 points, origin wins 2/5 → 4/5 in E8). |
| 4. Noise dominates | Few parameters. Gains are checked with the paired bootstrap, not point estimates (E8, §4.4). |
| 5. Reused years | Constants come from data ending in 1983. Selection is strictly chronological. Ablations stay diagnostic. Retrospective evidence is labelled as such. |
| 6. Nothing beats average30 yet | The target is `promote`. §4.4 states openly that the main candidate is near the line, and v2.5 targets the origin rule. |

---

## 7. Risks, fallbacks, compute, dependencies

| Risk | Sign | Response |
|---|---|---|
| Adapter disagrees with E8 | > 0.10 points on any origin in step 3 | Stop and fix before scoring (§5.1) |
| The retrospective gain is one fold (1962) | development gain concentrated in 1997 only | Report it. v2.5 targets breadth. No retuning |
| phi = 1 with a small gain at a development origin | artifact shows phi = 1 and K < 0.05 | Expected behaviour (v0.3-like). Reported, not changed |
| rho at a bound | flag in the artifact | Lower bound: W is near 0 except for drift and mask terms. Upper bound: gain 0.39. Both logged |
| Core rejected | v2.2 and v2.3 not kept | `CORE_DROPPED`. v2.4 and v2.5 still run on P1 |
| Runtime | > 300 s per origin | `ANCHOR_STEP = 10`. Defer v2.6 |
| MLP not deterministic | rerun differs | `threadpool_limits(1)`. If it still differs, drop v2.6 and record why |

Compute: CPU only, an estimated 30-90 s per development origin (E8: 21-52 s). Dependencies: none new (numpy, scipy `minimize_scalar`, scikit-learn, threadpoolctl). No new data release, no `features-v1`, no `enhance-v2`.

---

## 8. Tracking

Every scored run, ablations included, goes through `line.py score` and then `uv run python scripts/line.py sync`. Tags: `series=Placeholder`, `version=v2.<n>`, `plan=B`, `parent=<version>`, `incumbent=<version>`, `ablation=<name or none>`, `core_dropped=<bool>`.

Metrics to watch, in decision order: verdict and P(better on final) against the champion; the 90% lower bound (v0.5: −0.603); reward (v0.5: +0.834) and live points (v0.5: +2.899); final points by origin (v0.5: +0.27 / −2.35 / +4.58) and by metric (v0.5: +2.41 / 0.00 / +0.37 / +0.56), with the −2 floor; `runtime_s`.

Artifacts in `placeholder_selection.json`: `phi(F)` with its 5 × 15 inner table for the outer origin and each gate origin; `rho` and the gain per metric and anchor, with bound and convergence flags; F1/F2 counts per metric; mean W per metric; lambdas; origin-1997 final skill as a diagnostic. The retrospective driver's output (step 3) is committed next to the adapter and logged as an artifact of the first scored v2.x run.

Each result goes to `docs/models/placeholder.md` with run id, commit, parent, incumbent, verdict, P(better), lower bound, reward, and the keep decision under §4.3.
