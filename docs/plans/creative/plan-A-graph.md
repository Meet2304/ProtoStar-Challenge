# Plan A: graph-based spatio-temporal learning for the Placeholder series (v1.x)

This plan treats the 210 cities as nodes of a fixed geographic graph. It forecasts the part of each city's recent anomaly that
its neighbours do not share: the city minus a Gaussian-kernel (600 km) average of its neighbours. That relative series has
27-44% less noise than the raw anomaly. Before 1983 its recent level persisted, with fitted coefficients between 0.45 and 0.80
across fit origins 1940-1975.

The promotion route, **v1.1 = 30-year average + a centred graph-local term**, has no warming term, no learners and no gate. Its
gain does not depend on the shared warming exposure that costs every W-based version at 1990.

The chronological check has the training-target cutoff enforced in code (`docs/plans/creative/analysis/planA_walkforward.py`).
Coefficients are fitted only on targets <= F and scored on F+1..F+7.
- v1.1 scored +0.66, +2.37 and +1.10 final points over three held-out pre-1983 blocks, positive on 70%, 100% and 85% of origins.
- On 22 held-out origin triplets, the paired city bootstrap gave a median reward of +1.85 points and a median 90% lower bound
  of +1.17. Every triplet would have promoted.
- Resampling 5-degree or 10-degree spatial blocks instead of cities lowers the median lower bound to +1.07 / +1.10. All 22
  triplets still promote.
- The v0.5-like warming base lost -9.22 points in the 1940-49 block, the pre-1983 analogue of the 1990 overshoot.

v1.2 is one frozen joint model (graph-local term plus a long-window relative offset), run whatever happens to v1.1. The v0.5 +
graph version is deferred out of this round. Everything is numpy, the runs are gate-eligible (`official`), and one origin takes
about 7 s from CSV to forecast, deterministically. The evidence is pre-1983 and was produced on the same years as the design. It
supports trying the route; it does not forecast the development verdict.

Status: draft r4

Author: strategy planner A (Claude Opus 5.5), 2026-11-08. Base code: `notebooks/Meet/placeholder_v0_5.py` (coordinates and
helpers). Brief: `docs/plans/creative/brief.md`. Reward rules: `evaluation/REWARD.md`, `code/evaluation/reward.py`. Evidence
script: `docs/plans/creative/analysis/planA_walkforward.py`.

---

## Changes in r4

Sol's re-review of r3 found a leak in the evidence. The r3 walk-forward script filtered training rows by `t + h <= 1982`
instead of `t + h <= F`, so the last three pseudo-origins (t = F-6..F-4, h up to 7) reached targets F+1..F+3. The adapter
specification (section 3.1) already required `t + h <= O`; the leak was in the r3 analysis only. Every number from it is
regenerated here.

| Item | What changed |
|---|---|
| 1. Regenerate the chronological evidence with the cutoff enforced (MUST) | The new script saves to `docs/plans/creative/analysis/planA_walkforward.py`. Its `coef_rows` keeps only `t + h <= F`, and `fit_coefs` asserts `max(training target year) <= F` before every fit. A separate end-to-end check rebuilds three origins from a table with every year > F set to NaN and reproduces the walk-forward coefficients and forecasts exactly. All affected numbers are replaced: block scores, live, per-metric points, positive shares, coefficients, coefficient medians for the fixed-coefficient ablations, bootstrap summaries and rationale. The results match Sol's reproduction: v1.1 +0.6582 / +2.3694 / +1.0970, first block 70% positive, median reward +1.8532, worst floor -0.1058, 22/22, and 1940 CDD in v1.2 (0.70, 0.55). Sol's median lower bound of +1.1679 differs from mine (+1.1742), probably from how bootstrap draws are generated across triplets (this script draws fresh weights per triplet from one seeded generator). |
| 2. Freeze one v1.2 definition (MUST) | v1.2 is the joint model `average30 + centre(beta L_raw + gamma L_long_raw)`, built on `control:average30` and run unconditionally. The long-only branch is removed. |
| 3. v1.3 (MUST) | Deferred out of this round, as recommended: the deadline is 2026-11-10, and the complete v1.3 procedure (residual learners with the 0.5 blend in their targets, inner lambda selection on the deployed forecast) has not been validated. Section 5 states what a later validation would need. The proxy-based v1.3 evidence and its ablations are removed. |
| 4. Zero-median wording (SHOULD) | Corrected everywhere. The cross-city median of the adjustment G is 0, but the median forecast level can still move. Sol's example: baseline [0, 10, 20] plus [20, 0, -1] moves the median from 10 to 19. |
| 5. Runtime and determinism (SHOULD) | Measured on pre-1983 data: one complete origin (CSV load, pivot, graph terms, v1.1 and v1.2 fits, forecast) takes 6.0-6.9 s, and two runs per origin give identical SHA-256 digests (section 3.3). The adapter itself is not written yet. The first implementation check is stated: `forecast()` called directly on a pre-1983 cut, twice, then the harness `runtime_s`. |
| 6. Dependence-sensitive diagnostic (SHOULD) | Added spatial-block bootstraps (42 blocks of 5 x 5 degrees, 14 blocks of 10 x 10 degrees) next to the official city bootstrap (section 3.4). The official city bootstrap remains the verdict. |

## Changes in r3 (for the record; its walk-forward numbers are superseded by r4)

r3 made these changes after Sol's review of r2:
- Removed post-origin station information from the rationale.
- Replaced the keep rules with the champion verdict plus the lower-bound-and-reward rule against v0.5.
- Added the `average30 + G` promotion route.
- Moved to one deploy path.
- Recomputed the supporting tables with the declared centred estimator: centred precipitation +0.00670 / +0.00685 / +0.00219.
- Added the walk-forward (leaky in r3, fixed in r4) and the design-overlap table.
- Replaced guarantees with measured checks and corrected the 1.5-point and "errors cancel" statements.
- Stated that the plan does not use the ocean loader.

---

## 1. Strategy and model-selection logic

### 1.1 What the data say (years <= 1982 only; method in appendix A)

Inputs are `data/annual_valid.csv` and `data/city_info.csv`, cut at 1982 right after reading. No year in 1983-2018, no
`private/` file and no run output was read for design.

**Spatial correlation is strong for temperature and weak for precipitation.** Median pairwise correlation of annual anomalies,
1941-1982:

| Distance (km) | 0-250 | 250-500 | 500-750 | 750-1000 | 1000-1500 | 1500-2500 | >2500 | e-folding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| annual_avg_temp | 0.77 | 0.71 | 0.64 | 0.54 | 0.38 | 0.10 | -0.19 | ~1180 km |
| cdd | 0.76 | 0.67 | 0.57 | 0.45 | 0.28 | 0.01 | -0.15 | ~1030 km |
| hottest_day_temp | 0.67 | 0.52 | 0.40 | 0.30 | 0.18 | 0.00 | -0.13 | ~860 km |
| total_precip_in | 0.60 | 0.42 | 0.27 | 0.18 | 0.10 | 0.04 | 0.00 | ~750 km |

Median distance to the 1st, 5th, 15th and 30th nearest city: 104, 234, 423 and 634 km.

**Subtracting the neighbour mean removes much of the weather noise.** `R = city anomaly - mean of its 15 nearest cities'
anomalies`, 1941-1982:

| Metric | sd(anomaly) | sd(R) | lag-1 anomaly | lag-1 R | lag-5 R |
|---|---:|---:|---:|---:|---:|
| annual_avg_temp | 1.26 F | 0.70 F | 0.30 | 0.57 | 0.15 |
| cdd | 184 | 114 | 0.25 | 0.46 | 0.11 |
| hottest_day_temp | 2.99 F | 2.08 F | 0.14 | 0.14 | 0.00 |
| total_precip_in | 7.26 in | 5.30 in | 0.05 | 0.06 | 0.01 |

**The local part is the era-stable one.** Pooled OLS of the lead-4-7 departure on three parts of the 10-year gap: national
(cross-city median), regional (15-NN mean minus national) and local (own minus regional). Pseudo-origins 1950-1975, national
part removed from the target:

| Metric | regional 1950-62 | regional 1963-75 | local 1950-62 | local 1963-75 |
|---|---:|---:|---:|---:|
| annual_avg_temp | 0.11 | 0.70 | 0.66 | 0.54 |
| cdd | 0.58 | 0.82 | 0.69 | 0.51 |
| hottest_day_temp | 0.25 | 1.42 | 0.53 | 0.44 |
| total_precip_in | -0.37 | -0.53 | 0.39 | 0.29 |

The national coefficient for temperature over 1950-1975 is 1.32: recent national cold persisted in the cooling era, which is
lesson 1's effect. Regional coefficients change several-fold between eras. Local ones stay within about 0.2 (largest gap: CDD,
0.18).

**Declared estimator, in-era description (same-era best coefficient; descriptive only).** These tables fit and score on the
same years, so they carry no cutoff and are not out-of-sample evidence; that is section 3.4.

The estimator is the centred L of section 2. The table gives skill of `ref30 + b * centre(L)` against `ref30`, leads 4-7. The
last column adds the best `b * centre(L)` to the v0.5-like proxy `0.5 (ref30 + W) + 0.5 combo` (no learners):

| Metric | 1915-35: b, skill | 1950-62: b, skill | 1963-75: b, skill | gain over v0.5-like proxy (3 eras) |
|---|---|---|---|---|
| annual_avg_temp | 0.65, +0.00865 | 1.00, +0.05054 | 0.65, +0.01433 | 0.00000 / +0.00600 / 0.00000 |
| total_precip_in | 0.70, +0.00670 | 0.75, +0.00685 | 0.40, +0.00219 | same as skill (proxy = ref30) |
| hottest_day_temp | 0.65, +0.01348 | 0.65, +0.01483 | 0.55, +0.01045 | same as skill (proxy = ref30) |
| cdd | 0.60, +0.00937 | 0.90, +0.03534 | 0.70, +0.01901 | +0.00011 / +0.00372 / +0.00022 |

Against the 30-year average, the local term helps all four metrics in all three eras. Against a combo-based base it helps only
precipitation and hottest day, because combo's last-10 mean already carries the temperature and CDD local signal. That is why
the promotion route is built on the 30-year average.

v1.2 in-era (0.1 grid, combined centring), skill of `ref30 + centre(b L + g L_long)`:

| Metric | 1915-35: b, g, skill | 1950-62 | 1963-75 |
|---|---|---|---|
| annual_avg_temp | 0.7, 0.4, +0.00930 | 1.0, 0.4, +0.05352 | 0.7, 0.3, +0.01599 |
| total_precip_in | 0.8, 0.4, +0.00797 | 0.9, 0.5, +0.01012 | 0.5, 0.2, +0.00307 |
| hottest_day_temp | 0.6, 0.0, +0.01343 | 0.6, 0.5, +0.01723 | 0.6, 0.2, +0.01157 |
| cdd | 0.6, 0.5, +0.00997 | 1.0, 0.5, +0.03815 | 0.8, 0.3, +0.02092 |

### 1.2 Chosen model

```
v1.1:  forecast(c, m, O+h) = clip_m( ref30(c, m, O) + G(c, m, O) ),   G = beta_m * centre(L_raw)
v1.2:  same, with                                                     G = centre(beta_m * L_raw + gamma_m * L_long_raw)
clip_m = max(0, .) for total_precip_in and cdd, identity otherwise. G is the same for all seven leads.
```

Why the 30-year average base:

- **The champion is `control:average30`.** The paired comparison then differs from the champion only by G, a city-relative
  adjustment with zero cross-city median. The median adjustment is 0, but the median forecast level can still move. Shared
  weather enters each city's absolute error in both runs, so the paired difference is less noisy but not noise-free.
- **The warming term is the source of the 1990-type loss.** In the walk-forward (section 3.4), the v0.5-like proxy scored
  -9.22 final points in the 1940-49 held-out block. In that block 84% of median-city warming rates were positive just before
  warming stalled. The proxy scored +3.75 and +2.66 in the cooling-era blocks, where rates were mostly floored at 0. Adding even
  0.5 W to the graph base turned 1940-49 from +0.66 to -2.82 points. Promotion needs wins on 2 of 3 origins and no metric
  below -2 points; W exposure risks both.
- **The cost:** no warming gain at 1983/1997. Carrying W with the graph term is v1.3, deferred (section 5).

Fit to the rest of the data:
- **Size:** one fixed linear operator plus one or two scalars per metric, each fitted on roughly 25,000-30,000 city-years.
- **Noise:** sd(R) is 27-44% below sd(anomaly).
- **Scoring:** coefficients minimise MAE of the exact deployed forecast.
- **Cost:** about 7 s per origin, measured.

### 1.3 Alternatives considered and rejected

Kernel comparison with the declared centred estimator (mean over three eras of in-era best skill; descriptive):

| Graph | temp | precip | hottest | cdd | mean |
|---|---:|---:|---:|---:|---:|
| 5-NN equal | +0.0217 | +0.0037 | +0.0115 | +0.0152 | +0.0130 |
| 15-NN equal (v0.3's graph) | +0.0240 | +0.0052 | +0.0112 | +0.0182 | +0.0147 |
| 25-NN equal | **+0.0274** | **+0.0055** | +0.0122 | +0.0204 | **+0.0164** |
| Gaussian 300 km | +0.0239 | +0.0049 | +0.0116 | +0.0170 | +0.0144 |
| **Gaussian 600 km (declared in r1, kept)** | +0.0245 | +0.0052 | **+0.0129** | +0.0203 | +0.0157 |
| ordinary kriging (exp. cov., e-fold per metric, nugget 0.25, 30 nearest) | +0.0222 | +0.0046 | +0.0107 | +0.0155 | +0.0133 |

25-NN leads by 0.0007 mean skill (0.07 points). I keep the 600 km kernel declared in r1: the difference is below the spread
between eras, and switching now would be selection on the same years. 25-NN is ablation 2. The 600 km kernel has a median of 45
effective neighbours (`1 / sum w^2`; 10th percentile 19), and no city has kernel mass below 1.

| Candidate | Why not |
|---|---|
| Graph neural network / learned message passing (`uv add torch`) | The signal is linear: one high-pass operator and a scalar. Learned messages over 210 nodes and about 40 overlapping snapshots would fit the era-unstable regional part. It also needs a lock change and risks GPU nondeterminism. |
| Graph-Laplacian per-city coefficients | Needs heterogeneity. The closest cheap test, per-city empirical-Bayes reliability weights, lowered skill on every metric (r2 test, uncentred L; e.g. hottest day 1915-35 +0.0119 to +0.0093). Not recomputed: rejected by a large margin. |
| Kriging weights | Table above: below the Gaussian kernel on every metric. |
| Correlation-learned graph | r2 (uncentred): better for temperature only. Not recomputed. |
| Break-aware local level | r2, centred L: no hottest-day gain; temperature helps in one era only. |
| Regional precipitation mean reversion (coef -0.37 / -0.53) | Both eras sit in one drought-to-wet cycle; likely era-specific. |
| Spatial smoothing or damping of W | W's error is shared across nodes, and any damping would be tuned. The no-W base is fixed now. |

---

## 2. Targets and inputs

**Predicted:** the level `predicted_value`. Internally it is a shift from `ref30(c, m, O)`, the official 30-year mean (>= 10
valid years), as produced by `evaluation.adapters.reference(..., method="average")`. Implementation check: with G = 0, v1.1
equals `control:average30`.

**Inputs:** suite `train` per fold and `data/city_info.csv` (lat/lon, `static_context`), with manifest
`notebooks/Meet/placeholder-v0.1-assets.json`. The runs are `official` and gate-eligible. The plan uses no `features-v1`,
`clean-v1`, `enhance-v1/v2`, daily data or ocean loader. SST indices (`code/enhance_ocean.load_ocean_for_origin`, `enhance-v2`)
are basin-scale low-pass signals, which G is built to exclude, and they would make runs exploratory. If a later version used
them, it would load them through that function and score with `--assets evaluation/enhance-v2-assets.json` (entries copied into
its own manifest, binding only `data/processed/releases/enhance-v2/` paths), outside the champion route.

**Graph terms at a (pseudo-)origin t, metric m, from years <= t only:**

- `K_ij = exp(-(d_ij / 600 km)^2)`, `K_ii = 0`, haversine distance (radius 6371 km), built once.
- `ref = ref30(t)`. `X` = values for `t-59..t` (NaN invalid). `A = X - ref[:, None]`.
- `N_c(y) = sum_j K_cj A_j(y) / sum_j K_cj 1[A_j(y) valid]`, NaN where the valid kernel mass is < 1.0. `R = A - N` where X is
  valid.
- `L_raw = EWMA_hl(R, valid years in t-29..t) - mean(R, valid years in t-29..t)`. Half-life `hl` = 3 years (temperature,
  hottest day, CDD) or 6 (precipitation). It needs >= 10 valid R years.
- `L_long_raw = mean(R, valid years in t-59..t) - mean(R, valid years in t-29..t)`. It needs >= 15 valid R years in t-59..t-30
  and >= 10 in t-29..t.
- `centre(v)`: subtract the median over cities where v is defined, then set undefined cities to 0. The cross-city median of
  the adjustment G is therefore 0. The median forecast level can still move (baseline [0, 10, 20] + [20, 0, -1] moves the
  median from 10 to 19), and G can correlate with any spatial pattern, including W's.
  - v1.1: `G = beta * centre(L_raw)`, which equals `centre(beta L_raw)` since beta >= 0. It is defined where L_raw is defined.
  - v1.2: `G = centre(beta nz(L_raw) + gamma nz(L_long_raw))`, defined where either part is defined, with nz(NaN) = 0.

Every array comes from the fold table cut at the fit origin (`assert_no_post_cutoff`). Neighbours' values come from the same
years, and coordinates are static.

---

## 3. Training procedure

### 3.1 Coefficient fit

At the outer origin O:

1. For each pseudo-origin t in [O-45, O-4], compute `ref30(t)`, `L_raw(t)` and `L_long_raw(t)`.
2. Rows: every (t, h) with h in {4, 5, 6, 7} **and t + h <= O**, with `y(t+h)` and `ref30(t)` finite.
   `assert max(target year) <= O` before the fit, as in the evidence script.
3. `beta_m` (v1.2: `(beta_m, gamma_m)`) is the argmin over GRID (x GRID) of `mean |y - deploy(ref30, G)|`, using the exact
   deployed formula with the same centring and clipping.
   - GRID = 0.00, 0.05, ..., 1.00, scanned in ascending order (gamma outer, beta inner). Ties keep the first, i.e. the smallest
     gamma, then the smallest beta.
   - With fewer than 2,000 rows the coefficients are 0.
4. Forecast: `deploy(ref30(O), G(O))` for all seven leads.

Leads 4-7 match the reward. Live (leads 1-3) gets the same G; it is measured, not assumed (section 3.4).

### 3.2 One assembly path

```python
def centre(v):                                      # v: (cities,), NaN = undefined
    return np.where(np.isfinite(v), v - np.nanmedian(v), 0.0)

def graph_shift(beta, gamma, L_raw, L_long_raw, use_long):
    defined = np.isfinite(L_raw) | (np.isfinite(L_long_raw) if use_long else False)
    v = beta * np.nan_to_num(L_raw) + (gamma * np.nan_to_num(L_long_raw) if use_long else 0.0)
    return centre(np.where(defined, v, np.nan))

def deploy(metric, ref30, G):                       # the only forecast formula: coefficient fit and outer forecast
    f = ref30 + G
    return np.maximum(f, 0.0) if metric in NON_NEGATIVE else f

def fit_coefs(rows, metric, use_long):              # rows: y, ref30, L_raw, L_long_raw, target_year over (t, h)
    assert max(rows.target_year) <= fit_origin
    ...grid scan of mean |y - deploy(metric, ref30, graph_shift(b, g, L_raw, L_long_raw, use_long))|...
```

The evidence script implements exactly these functions (`graph_shift`, `deploy`, `coef_rows`, `fit_coefs`). The adapter should
copy them, add the v0.5 helpers `load_coords` / `_check_coords`, and use `reference(..., "average")` for ref30. There is no gate
and no inner origin, so no second scoring path exists.

### 3.3 Seeds, determinism, runtime (measured)

- No randomness. Grid scans run in fixed order. Kernel sums use broadcasting (`(K[:, :, None] * A[None]).sum(1)`) instead of
  BLAS `@`, so the result does not depend on thread count.
- **Measured** with the evidence script on this machine. The script ran one complete origin from a table cut at F (CSV load,
  pivot, 46 pseudo-origins of graph terms and v0.5-style proxies, v1.1 and v1.2 fits for four metrics, forecast), twice each:

| Origin F | run 1 | run 2 | SHA-256 of forecasts (first 16 hex) | identical | equals walk-forward |
|---|---:|---:|---|---|---|
| 1940 | 6.7 s | 6.9 s | d2654cef2645705a | yes | yes |
| 1960 | 6.1 s | 6.5 s | f4b94357cdafa19f | yes | yes |
| 1975 | 6.0 s | 6.6 s | 75f65a152199aef2 | yes | yes |

- The v1.2 21 x 21 grid is most of that time. The adapter skips the W/combo proxies, so it should be faster. The harness limit
  is 900 s (brief) and 600 s (`evaluate.py run` default).
- **Not yet measured:** the adapter itself, which is not written. The first implementation step, before any scoring:
  1. Call `forecast(train, targets, origin=1975)` directly, with `train` = `data/annual_valid.csv` cut at 1975 and a
     1976-1982 target grid, twice. Compare SHA-256 and time.
  2. Check equality with this script's 1975 forecast to 1e-9.
  3. Read `runtime_s` from the first scorecard.

### 3.4 Chronological evidence (cutoff enforced)

**Procedure** (`planA_walkforward.py`). For each fit origin F = 1940..1975:
- Fit the coefficients exactly as in section 3.1, with training targets <= F (asserted).
- Forecast at F and score F+1..F+7 against the 30-year average with the reward formula: pooled over cities,
  `skill = 1 - MAE / MAE(ref30)`, points = 100 x mean over metrics.

All targets are <= 1982. Three origins (1940, 1960, 1975) were recomputed from a table with every year > F removed, and they
reproduce the walk-forward coefficients and forecasts exactly.

**Held-out points against average30, by block of fit origins:**

| Candidate | 1940-49 final | 1950-62 final | 1963-75 final | origins > 0 | live by block |
|---|---:|---:|---:|---|---|
| **v1.1** avg30 + G(L) | **+0.6582** | **+2.3694** | **+1.0970** | 70% / 100% / 85% | +2.32 / +3.00 / +2.56 |
| **v1.2** avg30 + G(L, L_long) | **+1.0576** | **+2.4880** | **+1.0703** | 80% / 100% / 85% | +2.71 / +2.97 / +2.46 |
| diagnostic: avg30 + G(L) + 0.5 W | -2.8193 | +2.1311 | +1.3292 | 40% / 100% / 92% | -0.66 / +2.91 / +2.62 |
| reference: v0.5-like proxy, no G | -9.2177 | +3.7541 | +2.6594 | 0% / 100% / 100% | -7.20 / +3.08 / +3.83 |

Per metric, final points by block (temp / precip / hottest / cdd):
- v1.1: 1940-49 +1.38 / -0.47 / +1.30 / +0.42; 1950-62 +4.40 / +0.65 / +1.49 / +2.94; 1963-75 +1.37 / +0.23 / +1.03 / +1.76.
  The worst single origin is -1.31 (1940-49).
- v1.2: 1940-49 +1.69 / +0.03 / +1.76 / +0.75; 1950-62 +4.49 / +0.81 / +1.62 / +3.05; 1963-75 +1.52 / +0.16 / +0.84 / +1.76.
  The worst single origin is -0.83.

The 1940-49 block follows early-century warming: 84% of median-city warming rates were positive, against 15% and 9% in the
later blocks. It is the closest pre-1983 analogue of the 1990 origin.

**Coefficients (beta, gamma) by fit origin** (data <= F only):

| Metric | v1.1 beta, 1940 / 1950 / 1960 / 1970 / 1975 | v1.1 median of 36 | v1.2 (beta, gamma), 1940 / 1950 / 1960 / 1970 / 1975 | v1.2 median of 36 |
|---|---|---:|---|---:|
| annual_avg_temp | 0.80 / 0.65 / 0.70 / 0.80 / 0.80 | 0.725 | (0.80,0.45) (0.65,0.35) (0.75,0.40) (0.90,0.35) (0.85,0.40) | (0.75, 0.40) |
| total_precip_in | 0.80 / 0.60 / 0.55 / 0.50 / 0.45 | 0.55 | (0.80,0.40) (0.60,0.75) (0.60,0.60) (0.55,0.55) (0.60,0.50) | (0.60, 0.65) |
| hottest_day_temp | 0.60 / 0.60 / 0.65 / 0.65 / 0.60 | 0.65 | (0.60,0.05) (0.60,0.65) (0.65,0.70) (0.65,0.50) (0.65,0.50) | (0.60, 0.65) |
| cdd | 0.70 / 0.45 / 0.55 / 0.65 / 0.70 | 0.55 | (0.70,0.55) (0.50,0.40) (0.60,0.45) (0.65,0.50) (0.75,0.25) | (0.60, 0.45) |

The coefficients for temperature, hottest day and CDD are stable. Precipitation's v1.1 beta falls from 0.80 to 0.45 over
1940-1975, and its 1940-49 block is negative (-0.47 points) with L alone and +0.03 with L_long. It is the weakest component.
This is the origin-available persistence evidence: every number uses data <= its fit origin.

**Promotion analogue.** Held-out triplets (F, F+7, F+14), F = 1940..1961, spaced like 1983/1990/1997: 22 overlapping triplets,
compared against average30 with 2,000 draws (seed 20261108). All four `reward.py` guards apply:
- P(better) >= 0.95 with a positive point estimate;
- wins on >= 2 origins;
- live P(worse) < 0.95;
- every metric's final points >= -2.

The official city resampling is shown next to two dependence-sensitive schemes that resample whole spatial blocks: 42 cells of
5 x 5 degrees and 14 of 10 x 10 degrees. Every city in a drawn block gets that block's count.

| Candidate | resampling | median reward | median P | min P | median 90% LB | min 90% LB | worst floor | promote |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v1.1** | cities (official) | +1.8532 | 1.000 | 0.958 | +1.1742 | +0.0322 | -0.1058 | 22 / 22 |
| v1.1 | 5-degree blocks | +1.8532 | 1.000 | 0.951 | +1.0662 | +0.0095 | -0.1058 | 22 / 22 |
| v1.1 | 10-degree blocks | +1.8532 | 1.000 | 0.956 | +1.0954 | +0.0202 | -0.1058 | 22 / 22 |
| **v1.2** | cities (official) | +1.8073 | 1.000 | 0.974 | +1.1230 | +0.1186 | -0.0344 | 22 / 22 |
| v1.2 | 5-degree blocks | +1.8073 | 1.000 | 0.971 | +1.0823 | +0.1119 | -0.0344 | 22 / 22 |
| v1.2 | 10-degree blocks | +1.8073 | 1.000 | 0.945 | +1.1401 | -0.0219 | -0.0344 | 21 / 22 |
| diag. + 0.5 W | cities | +1.5124 | 1.000 | 0.000 | +0.8550 | -2.4930 | -5.6441 | 17 / 22 |
| ref. v0.5-like proxy | cities | +3.0465 | 1.000 | 0.000 | +2.2895 | -7.8548 | -16.0311 | 16 / 22 |
| ref. v0.5-like proxy | 10-degree blocks | +3.0465 | 0.999 | 0.000 | +1.5743 | -8.5056 | -16.0311 | 16 / 22 |

Reading:
- Block resampling lowers v1.1's median lower bound by 0.08-0.11 points. It lowers the W-based proxy's by up to 0.72: the
  warming term's gains are spatially clustered, the graph term's are not.
- The weakest triplet is close to the 0.95 threshold under every scheme (minimum P 0.95-0.96 for v1.1), so a promotion in the
  development run is plausible but not assured.
- What the bootstrap cannot show: the triplets overlap, the kernel and half-lives were chosen on these years (section 4.2),
  and 1984-2004 may behave differently.

---

## 4. Selection without peeking

### 4.1 Fixed in advance vs fitted in-fold

**Fixed now:**
- Gaussian 600 km kernel, mass floor 1.0.
- Half-lives {temp 3, hottest 3, cdd 3, precip 6}.
- Windows of 30 and 60 years, with minimum valid years 10 / 15.
- Combined centring.
- Grid 0-1, step 0.05, ascending ties.
- Fit leads 4-7, `t + h <= O`, `MIN_COEF_ROWS` 2000.
- G on all leads.
- Base = 30-year average, with no warming term.

**Fitted in each fold:** beta (v1.1), or beta and gamma (v1.2), at the outer origin from targets <= O. There are no inner
origins and no gate. No value is chosen after looking at a development score.

### 4.2 Overlap between design years and the development folds

Design used pseudo-origins 1895-1975 with targets <= 1982. The descriptive eras (1915-35, 1950-62, 1963-75) and the kernel and
half-life comparisons used the same years, and the walk-forward's held-out targets are inside them.

| Outer origin | coefficient-fit pseudo-origins (t in [O-45, O-4], t + h <= O) | overlap with design pseudo-origins | targets in fit |
|---|---|---|---|
| 1983 | 1938-1979 | 1938-1975 (38 of 42) | 1942-1983 |
| 1990 | 1945-1986 | 1945-1975 (31 of 42) | 1949-1990 |
| 1997 | 1952-1993 | 1952-1975 (24 of 42) | 1956-1997 |

The fixed hyperparameters were chosen on years that also train every development fold. That is allowed, because all of them
are <= every origin. But the in-fold fit and the walk-forward are not independent evidence for the kernel or half-lives. The
development targets (1984-2004) were seen by no design choice in this plan; they were reused by earlier versions of the series.
The 25-NN ablation measures kernel sensitivity on development years.

### 4.3 Keep rule (one rule, mechanical, fixed now)

The rule uses `uv run python code/evaluation/reward.py score <run_id> --json` against the current champion: `control:average30`
until something promotes (`reward.py ladder`).

1. **Primary: `promote`.** All four `reward.py` guards must hold: final P(better) >= 0.95 with a positive delta, better on >= 2
   of 3 origins, live P(worse) < 0.95, and every metric's final points >= -2. A promoted version becomes the champion.
2. **Otherwise keep over `Placeholder_v0.5` only if both** its 90% lower bound and its reward are strictly higher than v0.5's.
   Both are computed against the same champion with the same `reward.py`, at the same time. v0.5's values come from
   `reward.py score 20261108T141546Z-d0ac8f8b36 --json`, read before the first v1.x run and written into
   `docs/models/placeholder.md`. Its reward is +0.834 (brief).
3. A `worse` verdict is never kept. Ablations are never kept.
4. **Secondary evidence only:** `reward.py score <run_id> --against 20261108T141546Z-d0ac8f8b36 --json`. It is reported and never
   decides. Shared errors do not cancel exactly: each city's paired difference is `|e + d| - |e|`, which depends on the shared
   error e.
5. **v1.1 and v1.2 are siblings, both built on `control:average30`.** Each is judged by rules 1-3 on its own. If v1.1 promotes
   first, v1.2 is compared with the new champion v1.1 by the same `reward.py` rules. If both qualify under rule 2, the
   candidate follows REWARD.md's order: verdict, then the 90% lower bound, then the reward.

There is no fixed point threshold: promotion depends on the bootstrap distribution and the four guards.

**Pre-registered expectation:**
- v1.1 final points of +0.5 to +2.0 against average30.
- Positive on at least 2 origins, including 1990.
- Hottest day and temperature carry most of the gain; precipitation within ±0.5 points.
- If v1.1 is negative at 2 or more origins, the pre-1983 persistence does not hold in 1984-2004.

---

## 5. Versions

Each version is one single-file adapter in `notebooks/Meet/`, committed, then scored from a clean commit:

```powershell
uv run python scripts/line.py status
uv run python scripts/line.py score --name Placeholder_v1.1 --adapter notebooks/Meet/placeholder_v1_1.py:forecast `
  --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3
# ablation: same command with --name Placeholder_v1.1-<ablation> --params notebooks/Meet/placeholder-v1.1-<ablation>.json
uv run python code/evaluation/reward.py score <run_id> --json
uv run python scripts/line.py sync
```

### Placeholder_v1.1: 30-year average + graph-local term (promotion route)

- **Built on:** `control:average30`.
- **Change:** `G = beta_m * centre(L_raw)` for all four metrics (section 3.1), with clipping for precipitation and CDD.
- **Ablations (max 4, diagnostic only, never kept):**
  1. `coef=fixed`: beta = v1.1 medians of 36 fit origins, rounded down to the grid (temp 0.70, precip 0.55, hottest 0.65,
     cdd 0.55).
  2. `graph=knn25`: 25 nearest neighbours, equal weights.
  3. `warming=half`: plus 0.5 W for temperature and CDD (v0.5's W). Shows the W exposure on development years (pre-1983:
     -2.82 points in 1940-49).
  4. `precip=off`: G = 0 for precipitation, the weakest component.
- **Record:** beta per metric, mean `|G|`, share of cities with G != 0, the median removed by centring, and the median forecast
  shift (which need not be 0).

### Placeholder_v1.2: 30-year average + joint graph term (frozen definition)

- **Built on:** `control:average30`. It is run whatever v1.1's result.
- **Model:** `G = centre(beta_m L_raw + gamma_m L_long_raw)`, with (beta_m, gamma_m) fitted jointly on the 21 x 21 grid by the
  exact transform. There is no other branch.
- **Why:** the 30-year mean carries about `sd(R)/sqrt(30)` of local sampling noise (0.97 in for precipitation, 0.38 F for hottest
  day). Against v1.1 in the walk-forward, v1.2 scored +0.40 / +0.12 / -0.03 points by block. Precipitation's 1940-49 block
  moved from -0.47 to +0.03, and the worst metric floor across triplets from -0.11 to -0.03.
- **Ablations (max 3):**
  - `coef=fixed`: (beta, gamma) = v1.2 medians of 36 fit origins: temp (0.75, 0.40), precip (0.60, 0.65), hottest (0.60, 0.65),
    cdd (0.60, 0.45).
  - `long_years=45`, with `MIN_LONG_YEARS` 10.
  - `long_metrics=hot_precip`: gamma 0 for temperature and CDD.

### Deferred: Placeholder_v1.3 (v0.5 + graph term)

v1.3 is not run in this round. A credible v1.3 needs its complete procedure validated chronologically before any development
scoring:
- in-fold beta;
- residual learners whose targets account for the 0.5 blend weight, i.e. learners fit `2 (y - 0.5 combo - G) - ref30 - W` or an
  equivalent derived from the deployed formula;
- inner lambda selection scored on the deployed blend;
- the deployment itself.

That means refitting HistGBM/Huber at every walk-forward fit origin and inner origin, more than a day of work before the
2026-11-10 deadline. The pre-1983 analogue also shows that W-based bases lose -9.2 points when warming stalls. If v1.1 or v1.2
promotes, v1.3 would have to beat it under the ladder.

**Order:** v1.1, then v1.2, each with its ablations. No other versions, and no version tuned on a development score.

**Implementation time:** v1.1 2-3 h (small adapter, by copying the script's functions, plus checks). v1.2 is about 30 min on top.

**Checks:**
- G has zero cross-city median at every origin.
- With beta = 0, v1.1 equals `control:average30`.
- No year > fit origin is read (cut the table at t and compare).
- The adapter's 1975 forecast on a pre-1983 cut equals the evidence script's.
- Two runs are identical.
- No NaN.

---

## 6. How each lesson is addressed

1. **Shared warming (lesson 1).** v1.1 and v1.2 omit W, fixed now from the 1940-49 analogue. They forgo the warming gain and
   take no warming risk. G's median adjustment is 0, so it adds no era-wide shift to the median city's change. It does not
   claim to leave the median forecast level or W's spatial pattern untouched.
2. **The 1990 overshoot (lesson 2).** The promotion route avoids it: there is no W. In the walk-forward, the block after
   early-century warming cost the W-based proxy -9.22 points and gave v1.1 +0.66 (70% of origins positive). Nothing is tuned on
   1990. The half-W variant is a diagnostic ablation.
3. **Zero precipitation and hottest-day skill (lesson 3).** Hottest day gained +1.03 to +1.49 points in every held-out block
   (v1.1). Precipitation is the weak point, at -0.47 / +0.65 / +0.23 (v1.1) and +0.03 / +0.81 / +0.16 (v1.2), with a falling
   beta. `precip=off` measures its contribution.
4. **Noise (lesson 4).** At most two scalars per metric, fitted by MAE. The spatial-block diagnostic shows the city bootstrap
   is mildly optimistic for G (median lower bound 0.08-0.11 points lower). Verdict first, then the lower bound, then the
   reward.
5. **Reused years (lesson 5).** Design used years <= 1982 only; the overlap is in section 4.2. Two versions and at most 7
   ablations are planned. Only the 2004/2011 assessment is clean.

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Measured so far (pre-1983, cutoff enforced) | Sign in development results | Response |
|---|---|---|---|
| Local persistence weaker in 1984-2004 | beta stable for temp, hottest day and CDD; precipitation beta 0.80 to 0.45 | v1.1 negative on >= 2 origins | Report a negative result; ablations show which metric failed; no retuning |
| Precipitation below -2 points | worst block -0.47 (v1.1), +0.03 (v1.2); worst triplet floor -0.11 / -0.03 | `reward.py` floor | Checked, not assumed |
| Live clearly worse | live +2.32 to +3.00 points by block (v1.1) | live P(worse) | Measured per run |
| Promotion only marginal | weakest triplet P 0.95-0.96 under every resampling scheme | P(better) just under 0.95 | Rule 4.3.2 decides; no extra runs to push P up |
| City bootstrap optimistic | spatial blocks lower v1.1's median LB by 0.08-0.11 points | | Official verdict unchanged; report the block diagnostic alongside |
| Hyperparameters chosen on evidence years | not measurable | promotes on development, fails assessment | Integrator's one-shot assessment; 25-NN ablation for kernel sensitivity |
| Edge cities | 10th-percentile effective neighbours 19; mass >= 1 everywhere | large `|G|` | Logged if `|G|` > 2 sd(R) |
| Adapter differs from script | | 1975 equality check fails | Fix before scoring |

Fallbacks are not run. Half-weight or half-W variants are diagnostics only.

**Dependencies:** none new (numpy, pandas). No PyTorch, GPU or new data release. The manifest is unchanged.

**Compute:** about 7 s per origin measured for the full v1.1 + v1.2 analogue. At most 2 versions + 7 ablations = 9 runs, a few
minutes in total.

---

## 8. Tracking

Every scored run is mirrored with `uv run python scripts/line.py sync`.

- **Names:**
  - `Placeholder_v1.1` and `Placeholder_v1.1-{coef_fixed, graph_knn25, warming_half, precip_off}`.
  - `Placeholder_v1.2` and `Placeholder_v1.2-{coef_fixed, long_years_45, long_metrics_hot_precip}`.
- **Tags:** `series=Placeholder`, `version`, `plan=A-graph`, `built_on=control:average30`, `ablation`, `label=official`.
- **Reward block per run (`reward.py score --json`):**
  - verdict;
  - final delta, 90% interval, P(better), noise sd;
  - points by origin and by metric;
  - live delta and live P(worse);
  - the four guard booleans.
  Plus the same block against v0.5 (secondary), and `reward.py ladder` after each version.
- **Scorecard metrics:** `skill.origin.{1983,1990,1997}`, `skill.final.*`, `skill.live.*`, `bias.*`, `win_rate`,
  `worst_decile_city_skill`, `p90_error_ratio.*`, `stability.range`, `runtime_s`.
- **Artifacts:** `placeholder_selection.json` with beta (gamma) per metric, mean `|G|`, nonzero share, centring median and median
  forecast shift. Results go in `docs/models/placeholder.md`.

---

## Appendix A. How the numbers were computed

- **Section 3.4 and every coefficient, block, live, bootstrap, runtime and determinism number:**
  `docs/plans/creative/analysis/planA_walkforward.py`, run as `uv run python docs/plans/creative/analysis/planA_walkforward.py`
  from the repository root (about 5 minutes).
  - It reads only `data/annual_valid.csv` and `data/city_info.csv`, drops years > 1982 with an assertion, and asserts
    `max(training target year) <= F` before every fit.
  - It rebuilds three origins from a table cut at F and checks exact equality with the walk-forward.
  - The bootstraps use 2,000 draws with seed 20261108: multinomial over 210 cities, or over 42 / 14 spatial blocks (cells of
    `floor(lat / 5), floor(lon / 5)` and `floor(lat / 10), floor(lon / 10)`).
- **Section 1 tables:** throwaway scripts (`$env:TEMP\planA_analysis*.py`, `planA_r3.py`).
  - They use the same data and cut, and fit and score on the same era. That is descriptive by design, with no forward
    cutoff involved.
  - The correlation, relative-noise and decomposition tables use anomalies against each city's 1941-1982 mean and OLS without
    intercept.
  - The in-era, kernel and v1.2 tables use the declared centred estimator.
  - The empirical-Bayes and correlation-graph results in section 1.3 are r2 (uncentred) numbers for rejected alternatives.
- **W and the v0.5-like proxy** (used only for the diagnostic and reference rows) follow v0.5's definitions.
  - W: the 30-year OLS slope averaged over self + 15 nearest, floored at 0. CDD goes through the regional OLS response to
    temperature (>= 15 joint years), floored at 0. W is extrapolated from the mean valid year.
  - Combo: `0.5 (30-year OLS line, >= 10 years, else average) + 0.5 (mean of the last 10, else average)` for temperature and
    CDD; the 30-year average otherwise.
