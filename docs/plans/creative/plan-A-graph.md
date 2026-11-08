# Plan A: graph-based spatio-temporal learning for the Placeholder series (v1.x)

This plan treats the 210 cities as nodes of a fixed geographic graph. It forecasts the part of each city's recent anomaly that
its neighbours do not share: the city minus a Gaussian-kernel (600 km) average of its neighbours. That relative series has
27-44% less noise than the raw anomaly. Before 1983 its recent level persisted with a coefficient that stayed between 0.45 and
0.80 across fit origins 1940-1975. The main version, **v1.1 = 30-year average + a centred graph-local term**, has no warming
term, no learners and no gate. It is the route most likely to win the champion verdict, because its gain does not depend on the
shared warming exposure that costs every W-based version at 1990. In a chronological check on pre-1983 data, coefficients were
fitted only on earlier pseudo-origins and scored on later held-out years. v1.1 then scored +0.71, +2.39 and +1.10 final points
over three held-out blocks, positive on 80-100% of origins. A champion-style paired city bootstrap on 22 held-out origin
triplets gave a median P(better) of 1.00 and a median 90% lower bound of +1.17 points, and every triplet would have promoted.
The same check gave the v0.5-like warming base -9.2 points in the 1940-49 block, the pre-1983 analogue of the 1990 overshoot.
v1.2 adds a long-window relative offset. v1.3 carries the graph term into v0.5, with one assembly path shared by inner scoring
and the outer forecast. Everything is numpy, the inputs are gate-eligible (`official`), and one origin takes about 10 s. The
pre-1983 bootstrap is optimistic: nearby cities are correlated, the triplets overlap, and the hyperparameters were chosen on the
same years. These numbers are evidence for trying the route, not a forecast of the development verdict.

Status: draft r3

Author: strategy planner A (Claude Opus 5.5), 2026-11-08. Base code: `notebooks/Meet/placeholder_v0_5.py`. Brief:
`docs/plans/creative/brief.md` (with the reward and ocean-loader notes). Reward rules: `evaluation/REWARD.md` and
`code/evaluation/reward.py`.

---

## Changes in r3

Review by Sol (r2 scored 0/100); every item below is handled in this revision.

| Item | What changed |
|---|---|
| 1. Post-origin station information (MUST) | Deleted the r2 argument based on station-change counts from years after the forecast origins: the risk row, its fallback text and the section 7 sentence. Persistence is now argued only from data each origin can see: the in-fold coefficient fit, and how the walk-forward coefficients move over fit origins 1940-1975 (section 3.4). The plan reads no station metadata. |
| 2. Conflicting keep rules (MUST) | Section 4.3 now has one rule. Primary: `promote` against the champion, with all four `reward.py` guards. Otherwise, keep a version only if both its 90% lower bound and its reward beat `Placeholder_v0.5`'s, computed against the same champion. Paired significance against v0.5 is secondary evidence only. The old "headline and 1997 strictly higher" rule is removed. |
| 3. Credible promotion route (MUST) | v1.1 is now `average30 + G`, with coefficients fitted against that exact base. Warming for temperature/CDD in this base is fixed now at none. A 0.5 W variant was evaluated chronologically: -2.80 points in 1940-49 and a metric floor of -5.57. It stays only as a diagnostic ablation. The r2 v0.5-based design becomes v1.3. |
| 4. One assembly path (MUST) | v1.1/v1.2 have no gate: the fit loss and the outer forecast use the same `deploy()` function, with the same centring and clipping. v1.3 uses a single `deploy()` for v0.5 + G that `choose_lambdas` also calls at every inner origin, so blending, G placement and clipping are identical (section 3.2). |
| 5. Centring (MUST) | All supporting tables are recomputed with the declared, centred estimator. Centred precipitation is now +0.00670 / +0.00685 / +0.00219, matching Sol's reproduction. For v1.2, G = centre(beta L + gamma L_long) and the coefficients are fitted on exactly that transform. The orthogonality-to-W claim is withdrawn: zero median means only that G does not move the median city. |
| 6. Chronological evidence (MUST) | New section 3.4: a walk-forward over 36 fit origins (1940-1975). At each fit origin the complete coefficient procedure uses pseudo-origins <= F-4 only, then scores the held-out years F+1..F+7, plus the promotion analogue on held-out triplets. The overlap between design years and the development folds' training and inner years is tabulated in section 4.2. |
| 7. Measured checks (SHOULD) | Guarantee language is replaced by measured values: metric floor, live, and runtime measured on pre-1983 data. Half-weight variants are diagnostic ablations only and are never a fallback. |
| Factual fixes | There is no fixed 1.5-point threshold: promotion depends on the bootstrap distribution and the four guards in `reward.py`. Shared errors do not cancel exactly in a paired MAE comparison (section 4.3). |
| Ocean loader (integrator note) | Not used: G is a high-pass term and basin-scale SST indices are low-pass. Section 2 says how they would be loaded if a later version needed them. |

---

## 1. Strategy and model-selection logic

### 1.1 What the data say (years <= 1982 only; method in appendix A)

Inputs are `data/annual_valid.csv` and `data/city_info.csv`, cut at 1982 right after reading. No year in 1983-2018, no
`private/` file and no run output was read for design.

**Spatial correlation is strong for temperature and weak for precipitation.** Median pairwise correlation of annual anomalies,
1941-1982, by distance:

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
the effect behind lesson 1. Regional coefficients change several-fold between eras, while local ones stay within about 0.2 (largest gap: CDD, 0.18).

**Declared estimator, in-era description (same-era best coefficient; descriptive only, not evidence of out-of-sample skill).**
The estimator is the centred L of section 2: Gaussian 600 km kernel, kernel-mass floor 1.0, EWMA half-life 3 years (6 for
precipitation), >= 10 valid relative years. The table gives skill of `ref30 + b * centre(L)` against `ref30`, leads 4-7. The
last column is the gain of adding the best `b * centre(L)` to the v0.5-like proxy `0.5 (ref30 + W) + 0.5 combo` (no learners):

| Metric | 1915-35: b, skill | 1950-62: b, skill | 1963-75: b, skill | gain over v0.5-like proxy (3 eras) |
|---|---|---|---|---|
| annual_avg_temp | 0.65, +0.00865 | 1.00, +0.05054 | 0.65, +0.01433 | 0.00000 / +0.00600 / 0.00000 |
| total_precip_in | 0.70, +0.00670 | 0.75, +0.00685 | 0.40, +0.00219 | same as skill (proxy = ref30) |
| hottest_day_temp | 0.65, +0.01348 | 0.65, +0.01483 | 0.55, +0.01045 | same as skill (proxy = ref30) |
| cdd | 0.60, +0.00937 | 0.90, +0.03534 | 0.70, +0.01901 | +0.00011 / +0.00372 / +0.00022 |

Two readings follow:
- Against the 30-year average, the local term helps all four metrics in all three eras.
- Against a combo-based base it helps only precipitation and hottest day, because combo's last-10 mean already carries the
  temperature and CDD local signal.

That is why the promotion route is built on the 30-year average (section 1.2).

**v1.2 long-window offset, in-era description:** `G = centre(b L + g L_long)` on a 0.1 grid, against the 30-year average:

| Metric | 1915-35: b, g, skill | 1950-62 | 1963-75 | L only (above) |
|---|---|---|---|---|
| annual_avg_temp | 0.7, 0.4, +0.00930 | 1.0, 0.4, +0.05352 | 0.7, 0.3, +0.01599 | +0.00865 / +0.05054 / +0.01433 |
| total_precip_in | 0.8, 0.4, +0.00797 | 0.9, 0.5, +0.01012 | 0.5, 0.2, +0.00307 | +0.00670 / +0.00685 / +0.00219 |
| hottest_day_temp | 0.6, 0.0, +0.01343 | 0.6, 0.5, +0.01723 | 0.6, 0.2, +0.01157 | +0.01348 / +0.01483 / +0.01045 |
| cdd | 0.6, 0.5, +0.00997 | 1.0, 0.5, +0.03815 | 0.8, 0.3, +0.02092 | +0.00937 / +0.03534 / +0.01901 |

These in-era numbers fit and score on the same years. The out-of-sample evidence is the walk-forward in section 3.4.

### 1.2 Chosen model

```
v1.1:  forecast(c, m, O+h) = clip_m( ref30(c, m, O) + G(c, m, O) )               G = beta_m * centre(L)
v1.2:  same, with                                                                 G = centre(beta_m * L + gamma_m * L_long)
v1.3:  forecast = clip_m( 0.5 * clip_m(ref30 + W + lambda_m * rhat) + 0.5 * combo + G )      (v0.5 + G, one deploy path)
clip_m = max(0, .) for total_precip_in and cdd; identity otherwise.  G is the same for all seven leads.
```

Why the 30-year average base for the promotion route:

- **The champion is `control:average30`.** With `average30 + G`, the paired comparison differs from the champion only by G.
  G is a city-relative term with zero cross-city median, so the paired draws vary only through G's city-level errors. They do
  not vary through a shared warming shift whose sign depends on the period. Shared weather still enters each city's absolute
  error, so the difference is not noise-free.
- **The warming term is the source of the 1990-type loss.** In the walk-forward (section 3.4), the v0.5-like proxy scored -9.22
  final points in the 1940-49 held-out block, where 84% of median-city warming rates were positive just before warming stalled.
  It scored +3.75 and +2.66 in the two cooling-era blocks, where the rate was mostly floored at 0. Adding even 0.5 W to the
  graph base turned the 1940-49 block from +0.71 to -2.80. A promotion needs a win on 2 of 3 origins and no metric below -2
  points, and the W exposure risks both.
- **The cost:** v1.1 gives up the warming gain that v0.5 earns at 1983 and 1997. v1.3 keeps the series route for carrying W.
  Under the ladder, v1.3 must beat whatever is champion at the time.

Why it suits the rest of the data:
- **210 nodes, about 40 effective pseudo-origins:** the graph is one fixed linear operator, and each metric has one or two
  scalar coefficients fitted on roughly 25,000-30,000 city-years.
- **Heavy noise:** R has 27-44% less sd. The EWMA and the 60-year mean reduce it further.
- **MAE scoring:** coefficients minimise mean absolute error of the exact deployed forecast, by grid search.
- **Compute:** measured 7.6 s per origin (section 3.3).

### 1.3 Alternatives considered and rejected

Kernel comparison, recomputed with the declared centred estimator (mean over three eras of in-era best skill):

| Graph | temp | precip | hottest | cdd | mean |
|---|---:|---:|---:|---:|---:|
| 5-NN equal | +0.0217 | +0.0037 | +0.0115 | +0.0152 | +0.0130 |
| 15-NN equal (v0.3's graph) | +0.0240 | +0.0052 | +0.0112 | +0.0182 | +0.0147 |
| 25-NN equal | **+0.0274** | **+0.0055** | +0.0122 | +0.0204 | **+0.0164** |
| Gaussian 300 km | +0.0239 | +0.0049 | +0.0116 | +0.0170 | +0.0144 |
| **Gaussian 600 km (declared in r1, kept)** | +0.0245 | +0.0052 | **+0.0129** | +0.0203 | +0.0157 |
| ordinary kriging (exp. cov., e-fold per metric, nugget 0.25, 30 nearest) | +0.0222 | +0.0046 | +0.0107 | +0.0155 | +0.0133 |

On the centred estimator, 25-NN is slightly ahead on the mean (+0.0007 skill, 0.07 points). I keep the Gaussian 600 km kernel
declared in r1:
- The difference is smaller than the spread between eras.
- The walk-forward in section 3.4 was run with the 600 km kernel.
- Switching after a recomputation would be selection on the same years.

The 25-NN graph is v1.1's ablation 2, so the development run measures the sensitivity. The 600 km kernel has a median of 45
effective neighbours (`1 / sum w^2`; 10th percentile 19), and no city has kernel mass below 1.

| Candidate | Why not |
|---|---|
| Graph neural network / learned message passing (`uv add torch`) | The signal is linear: one high-pass operator and a scalar. A one-layer GNN with fixed weights is the kernel mean. Learned messages over 210 nodes and about 40 overlapping snapshots would fit the regional part, the part shown to be era-unstable. It also adds a lock change and GPU nondeterminism. |
| Graph-Laplacian per-city coefficients | Needs real heterogeneity. The closest cheap test, per-city empirical-Bayes reliability weights, lowered skill on every metric (r2 test, uncentred L; e.g. hottest day 1915-35 +0.0119 to +0.0093). Not recomputed, because it was rejected on a large margin. |
| Kriging weights | Table above: below the Gaussian kernel on every metric. |
| Correlation-learned graph | r2 (uncentred) result: better for temperature only, worse for hottest day and precipitation. Not recomputed. |
| Break-aware local level | r2 result, centred L: no gain on hottest day; on temperature it helps one era only. |
| Regional precipitation mean reversion (coef -0.37 / -0.53) | Both eras sit in one drought-to-wet cycle. Too likely to be era-specific. |
| Spatial smoothing or damping of W | W's error is shared across nodes. Damping is not tuned (section 6). The no-W base is v1.1's choice, fixed now. |

---

## 2. Targets and inputs

**Predicted:** the level `predicted_value`. Internally every term is a shift from `ref30(c, m, O)`, the official 30-year mean
(>= 10 valid years), as produced by `evaluation.adapters.reference(..., method="average")`. With G = 0, v1.1 reproduces
`control:average30` exactly; this is an implementation check.

**Inputs:** suite `train` per fold, and `data/city_info.csv` (lat/lon, `static_context`). The manifest is the existing
`notebooks/Meet/placeholder-v0.1-assets.json`. Runs are `official` and gate-eligible. No `features-v1`, `clean-v1`,
`enhance-v1/v2` or daily data is used.

The ocean loader (`code/enhance_ocean.load_ocean_for_origin`, `enhance-v2`) is not used. SST indices are basin-scale, low-pass
information, and G is built to exclude low-pass signal. Using them would make runs exploratory, so they could not become
champion. If a later version wanted them, it would load through `load_ocean_for_origin(release, origin)`, score with
`--assets evaluation/enhance-v2-assets.json` (its entries copied into the model's manifest, binding only
`data/processed/releases/enhance-v2/` paths), and stay outside the champion route. The only gate-eligible extension I see is
`enhance-v1` static facts (elevation, coast distance) as edge weights. That is untested, so not planned.

**Graph terms at a (pseudo-)origin t, metric m, from years <= t only:**

- `K_ij = exp(-(d_ij / 600 km)^2)`, with `K_ii = 0` and haversine distance d (radius 6371 km), built once from coordinates.
- `ref = ref30(t)`. Window `X` = values for `t-59..t` (NaN invalid). Anomaly `A = X - ref[:, None]`.
- Neighbour mean `N_c(y) = sum_j K_cj A_j(y) / sum_j K_cj 1[A_j(y) valid]`. It is NaN where the valid kernel mass is below 1.0.
- `R = A - N` where `X` is valid.
- `L_raw = EWMA_hl(R over valid years in t-29..t) - mean(R over valid years in t-29..t)`. Half-life `hl` = 3 years for
  annual_avg_temp, hottest_day_temp and cdd, and 6 for total_precip_in. L_raw is NaN unless there are >= 10 valid R years.
- `L_long_raw = mean(R over valid years in t-59..t) - mean(R over valid years in t-29..t)`. It is NaN unless there are >= 15
  valid R years in t-59..t-30 and >= 10 in t-29..t.
- `centre(v)`: subtract the median over cities where v is defined, then set undefined cities to 0. The cross-city median of G is
  0 at every origin. That says nothing about G's correlation with W's spatial pattern, which is not claimed.
  - v1.1: `G = beta * centre(L_raw)`, which equals `centre(beta L_raw)` since beta >= 0.
  - v1.2: `G = centre(beta * nz(L_raw) + gamma * nz(L_long_raw))`, defined where either part is defined, with nz(NaN) = 0.

Every array comes from the fold table cut at the fit origin (`tables_at`, `assert_no_post_cutoff`). Neighbours' values come
from the same years, and coordinates are static.

---

## 3. Training procedure

### 3.1 Coefficient fit (v1.1, v1.2)

At the outer origin O (no inner origins are needed, since there is no gate):

1. For every pseudo-origin t in [O-45, O-4], compute `ref30(t)`, `L_raw(t)` (and `L_long_raw(t)`).
2. Rows: every (t, h) with h in {4, 5, 6, 7}, t + h <= O, `y(t+h)` and `ref30(t)` finite.
3. `beta_m` (v1.2: `(beta_m, gamma_m)`) is the argmin over GRID (x GRID) of `mean |y - deploy(ref30, G_b,g)|`. Here `deploy` is
   the exact outer formula: same centring, same clip. GRID = 0.00, 0.05, ..., 1.00, scanned in ascending order (gamma outer,
   beta inner). Ties keep the first, i.e. the smallest gamma then beta. If there are fewer than 2,000 rows, the coefficients
   are 0.
4. Forecast at O: `deploy(ref30(O), G(O))` for all seven leads.

The fit uses leads 4-7 because the reward scores those leads. Live (leads 1-3) gets the same G. It is measured, not assumed
(section 3.4).

### 3.2 One assembly path (all versions)

```python
def centre(v):                                   # v: (cities,) with NaN = undefined
    out = v - np.nanmedian(v)
    return np.where(np.isfinite(v), out, 0.0)

def graph_shift(L_raw, L_long_raw, beta, gamma):  # v1.1: gamma = 0 and L_long_raw unused
    defined = np.isfinite(L_raw) | (np.isfinite(L_long_raw) if L_long_raw is not None else False)
    v = beta * np.nan_to_num(L_raw) + (gamma * np.nan_to_num(L_long_raw) if L_long_raw is not None else 0.0)
    return centre(np.where(defined, v, np.nan))

def clip_m(x, metric):
    return np.maximum(x, 0.0) if metric in NON_NEGATIVE else x

def deploy(metric, ref30, G, W=None, rhat=None, lam=None, combo=None):
    """The only forecast formula. Used by coefficient fitting, by inner lambda scoring (v1.3) and by the outer forecast."""
    if combo is None:                                       # v1.1 / v1.2
        return clip_m(ref30 + G, metric)
    member = clip_m(ref30 + W + lam * rhat, metric)         # v1.3: v0.5's member, unchanged
    return clip_m(0.5 * member + 0.5 * combo + G, metric)   # G at net weight 1, outside the blend
```

**v1.3 details (v0.5 + G):**
- beta is fitted with `deploy(..., rhat=0)`, i.e. against `clip(0.5 clip(ref30 + W) + 0.5 combo + G)` at pseudo-origins.
  combo comes from `combo_model.combo_forecast(table, metric, t, [t+h])`, cut at t and cached per (metric, t).
- The learners fit `y - ref30 - W - G` with the cross-city median removed per (t, h).
- `choose_lambdas` scores every lambda candidate at each inner origin with the same `deploy`. That includes the inner origin's
  own combo, W, G (with the inner fit's beta) and clipping. The outer forecast calls `deploy` with the selected lambdas.
- v0.5's gate scored only the member (`ref30 + W + lambda rhat`). v1.3 changes the gate objective to the deployed forecast as
  part of its change group.
- Implementation check: with `beta = 0` and `gate=none`, v1.3 equals v0.5 with `gate=none` to the byte. With the gate on it
  need not equal v0.5, because the gate's objective differs.

### 3.3 Seeds, determinism, runtime (measured)

- v1.1/v1.2 are closed-form, with grid scans in fixed order and no randomness. v1.3 keeps v0.5's GBM `random_state=42`.
- Kernel sums use broadcasting (`(K[:, :, None] * A[None]).sum(1)`), not BLAS `@`, so the order does not depend on threads.
- Check: two runs give byte-identical `origin-*.csv`.
- **Runtime, measured** on this machine on pre-1983 data (analysis script, numpy only, excluding CSV load):
  - 0.092 s per pseudo-origin for all four metrics (60-year relative series, EWMA, W and combo proxy).
  - One complete origin from scratch (46 pseudo-origins plus the v1.1 and v1.2 fits for four metrics): **7.6 s**.
  - The v1.2 21 x 21 grid is the costly part: the 36-origin x 5-candidate walk-forward took 160 s.
  - Expected harness time: v1.1/v1.2 about 10-15 s per origin. v1.3 is v0.5 (under about 35 s per origin, inferred from run-id
    timestamps) plus 4 fit origins x about 8 s, so about 70 s.
  - All far below the 900 s brief limit and the 600 s `evaluate.py run` default.

### 3.4 Chronological evidence before development scoring

**Procedure.** For every fit origin F = 1940, 1941, ..., 1975:
- Fit the coefficients exactly as in section 3.1 (pseudo-origins t in [F-45, F-4], targets <= F).
- Forecast at F and score F+1..F+7 against the 30-year average with the reward formula: pooled over cities,
  `skill = 1 - MAE / MAE(ref30)`, points = 100 x mean over metrics.

All targets are <= 1982 and never enter the fit that scores them. Five candidates were evaluated:

| Candidate | 1940-49 final | 1950-62 final | 1963-75 final | live (3 blocks) | origins > 0 | worst block metric |
|---|---:|---:|---:|---|---|---|
| **v1.1** avg30 + G(L) | **+0.71** | **+2.39** | **+1.10** | +2.29 / +3.04 / +2.57 | 80% / 100% / 85% | precip -0.42 (1940-49) |
| **v1.2** avg30 + G(L, L_long) | **+1.08** | **+2.53** | **+1.06** | +2.70 / +3.01 / +2.55 | 90% / 100% / 85% | precip +0.08 (1940-49) |
| avg30 + G(L) + 0.5 W (temp/CDD) | -2.80 | +2.16 | +1.33 | -0.69 / +2.95 / +2.64 | 40% / 100% / 92% | temp -8.45 (1940-49) |
| v0.5-like proxy, no G | -9.22 | +3.75 | +2.66 | -7.20 / +3.08 / +3.83 | 0% / 100% / 100% | temp -21.44 (1940-49) |
| v1.3-like: proxy + G(L) | -9.00 | +4.33 | +2.93 | -6.65 / +3.73 / +4.62 | 20% / 100% / 100% | temp -21.44 (1940-49) |

Per metric, v1.1 final points by block (temp / precip / hottest / cdd):
- 1940-49: +1.41 / -0.42 / +1.30 / +0.54
- 1950-62: +4.44 / +0.65 / +1.50 / +2.99
- 1963-75: +1.37 / +0.22 / +1.03 / +1.79

G added to the proxy gains +0.22 / +0.58 / +0.27 points by block. The proxy has no learners, so it is not v0.5. The 1940-49
block follows early-century warming: 84% of median-city 30-year warming rates were positive (9-15% in the later blocks). It is
the closest pre-1983 analogue of the 1990 origin.

**Promotion analogue.** I took held-out triplets (F, F+7, F+14) for F = 1940..1961, spaced like 1983/1990/1997, 22 overlapping
triplets. Each ran the paired city bootstrap against average30 as in `reward.py`: 2,000 multinomial draws, seed 20261108, the
same draw for every origin, split and metric. All four guards were applied:

| Candidate | median points | median P(better) | median 90% LB | would promote | worst metric floor |
|---|---:|---:|---:|---:|---:|
| **v1.1** | +1.87 | 1.00 | +1.17 | 22 / 22 | -0.08 |
| **v1.2** | +1.82 | 1.00 | +1.15 | 22 / 22 | -0.00 |
| + 0.5 W | +1.52 | 1.00 | +0.86 | 17 / 22 | -5.57 |
| v0.5-like proxy | +3.05 | 1.00 | +2.27 | 16 / 22 | -16.03 |
| proxy + G(L) | +3.39 | 1.00 | +2.55 | 17 / 22 | -15.96 |

**Coefficients over fit origins** (beta at F = 1940 / 1950 / 1960 / 1970 / 1975; data <= F only):

| Metric | v1.1 beta | v1.2 beta | v1.2 gamma | v1.3-like beta |
|---|---|---|---|---|
| annual_avg_temp | 0.75 / 0.65 / 0.75 / 0.80 / 0.80 | 0.80 / 0.65 / 0.75 / 0.85 / 0.85 | 0.45 / 0.40 / 0.40 / 0.40 / 0.40 | 0.00 / 0.00 / 0.10 / 0.15 / 0.15 |
| total_precip_in | 0.80 / 0.55 / 0.55 / 0.50 / 0.50 | 0.80 / 0.60 / 0.60 / 0.60 / 0.60 | 0.40 / 0.75 / 0.60 / 0.50 / 0.50 | 0.80 / 0.55 / 0.55 / 0.50 / 0.50 |
| hottest_day_temp | 0.60 / 0.60 / 0.65 / 0.65 / 0.60 | 0.60 / 0.60 / 0.60 / 0.65 / 0.65 | 0.05 / 0.65 / 0.65 / 0.50 / 0.50 | 0.60 / 0.60 / 0.65 / 0.65 / 0.60 |
| cdd | 0.60 / 0.45 / 0.55 / 0.65 / 0.70 | 0.65 / 0.50 / 0.60 / 0.70 / 0.75 | 0.50 / 0.40 / 0.45 / 0.55 / 0.25 | 0.00 / 0.00 / 0.05 / 0.10 / 0.15 |

What this shows, and its limits:
- Coefficients fitted only on earlier years are stable for temperature, hottest day and CDD.
- Precipitation's beta falls from 0.80 to 0.50 over 1940-1975, and its held-out 1940-49 block is negative (-0.42 points). It is
  the weakest component, and the in-fold refit tracks its decline with a lag of up to 45 years. This is the origin-available
  persistence argument that replaces r2's station-change argument: every number here uses data <= its fit origin.
- The bootstrap is optimistic:
  - Neighbouring cities are correlated, so city draws understate the uncertainty, as REWARD.md says.
  - The triplets overlap.
  - The kernel and half-lives were chosen on these same pre-1983 years (section 4.2).
- What the walk-forward does show is that the coefficient-fitting procedure transfers forward in time on pre-1983 data. It does
  not show that the 1984-2004 development years behave the same way.

---

## 4. Selection without peeking

### 4.1 Fixed in advance vs fitted in-fold

**Fixed now:**
- Gaussian kernel, 600 km, kernel-mass floor 1.0.
- EWMA half-lives {temp 3, hottest 3, cdd 3, precip 6}.
- Relative windows of 30 and 60 years; minimum valid years 10 (recent) and 15 (early long window).
- Centring of the combined G.
- Coefficient grid 0-1, step 0.05, with ascending tie order. Fit leads 4-7, `MIN_COEF_ROWS = 2000`. G applies to all leads.
- v1.1 base = 30-year average with **no warming term**.
- v1.3 keeps every v0.5 setting: `TRAIN_YEARS 45`, `N_NEIGHBOURS 15` for W, Huber/GBM, `INNER_OFFSETS (21, 14, 7)`, lambda grid
  and margin, `COMBO_WEIGHT 0.5`.

**Fitted in each fold:**
- beta (and gamma) at the outer origin, from pseudo-origins t <= O-4 with targets <= O.
- For v1.3, also at each inner origin, plus lambda from inner origins O-21, O-14, O-7, scored by `deploy`.

No value is chosen after looking at a development score.

### 4.2 Overlap between design years and the development folds

Design used pseudo-origins 1895-1975 with targets <= 1982. Descriptive eras: 1915-35, 1950-62, 1963-75. Kernel and half-life
comparisons used the same eras.

| Outer origin | coefficient-fit pseudo-origins (t + h <= O, h 4-7) | overlap with design pseudo-origins | v1.3 inner origins | inner origins whose scored targets lie in design years (<= 1982) |
|---|---|---|---|---|
| 1983 | 1938-1979 | 1938-1975 (38 of 42) | 1962, 1969, 1976 | 1962, 1969 fully; 1976 except target year 1983 |
| 1990 | 1945-1986 | 1945-1975 (31 of 42) | 1969, 1976, 1983 | 1969 fully; 1976 except 1983 |
| 1997 | 1952-1993 | 1952-1975 (24 of 42) | 1976, 1983, 1990 | 1976 except 1983 |

The inner fits themselves (pseudo-origins <= inner origin - 4) lie almost entirely inside the design years for every outer origin.

Consequences:
- The fixed hyperparameters were chosen on years that also train every development fold. That is allowed (all <= every
  origin), but the inner folds and in-fold fits are not independent evidence for those choices.
- The development targets 1984-2004 were seen by no design choice in this plan. They have been reused by earlier versions of
  the series, so they are still reused evidence.
- The walk-forward of section 3.4 also shares years with the design, so it is not clean evidence for the kernel or half-life
  either. The 25-NN ablation measures that sensitivity on development years.

### 4.3 Keep rule (one rule, mechanical, fixed now)

Computed by `uv run python code/evaluation/reward.py score <run_id> --json` against the current champion. That is
`control:average30` until something promotes; `reward.py ladder` shows it.

1. **Primary: `promote`.** The verdict is `promote` only when all four `reward.py` guards hold:
   - P(better on final) >= 0.95 with a positive final delta (`PROMOTE_P`).
   - Better on >= 2 of 3 origins (`MIN_ORIGIN_WINS`).
   - Live not clearly worse: live P(worse) < 0.95.
   - Every metric's final points >= -2 (`METRIC_FLOOR` -0.02).

   A promoted version becomes the champion and the base for the next version.
2. **Otherwise keep over `Placeholder_v0.5` only if both** its 90% interval lower bound and its reward are strictly higher than
   v0.5's. Both are computed against the same champion with the same `reward.py`, at the same time. v0.5's values come from
   `reward.py score 20261108T141546Z-d0ac8f8b36 --json`, read before the first v1.x run and written into
   `docs/models/placeholder.md`. Its reward is +0.834 (brief); its lower bound is not yet recorded in the plan.
3. A `worse` verdict is never kept. Ablations are never kept. Exploratory runs (none planned) cannot be kept.
4. **Secondary evidence only:** the paired comparison against v0.5,
   `reward.py score <run_id> --against 20261108T141546Z-d0ac8f8b36 --json`. It is reported, but it never decides. In a paired
   MAE comparison the shared weather does not cancel exactly. Each city's error is `|e + d| - |e|`, which depends on the
   shared error e, so the paired draws are less noisy but not noise-free.

**On thresholds:** there is no fixed point threshold for promotion. Whether a gain promotes depends on the bootstrap
distribution of the paired deltas and the four guards. The eval station's remark that gains under about 1.5 points rarely
register describes past runs, not a rule.

**Pre-registered expectation for development scoring**, so a result can falsify it:
- v1.1 final points of +0.5 to +2.0 against average30.
- Positive on at least 2 origins, including 1990, because v1.1 carries no warming exposure.
- Hottest day and temperature carry most of the gain; precipitation is within ±0.5 points.
- If v1.1 is negative at 2 or more origins, the persistence seen before 1983 does not hold in 1984-2004.

---

## 5. Versions

Each version is one single-file adapter in `notebooks/Meet/`, committed, then scored from a clean commit:

```powershell
uv run python scripts/line.py status
uv run python scripts/line.py score --name Placeholder_v1.1 --adapter notebooks/Meet/placeholder_v1_1.py:forecast `
  --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3
# ablation: same, --name Placeholder_v1.1-<ablation> --params notebooks/Meet/placeholder-v1.1-<ablation>.json
uv run python code/evaluation/reward.py score <run_id> --json
uv run python scripts/line.py sync
```

### Placeholder_v1.1: 30-year average + graph-local term (promotion route)

- **Built on:** `control:average30`.
- **Change:** add `G = beta_m * centre(L_raw)` for all four metrics, fitted per section 3.1; clip for precipitation and CDD.
- **Code:** a new small adapter, `placeholder_v1_1.py`, with no learners, gate or combo:
  - `load_coords` and `_check_coords` reused from v0.5.
  - `graph_kernel`, `relative_series`, `centre`, `graph_shift`, `deploy`, `fit_coefs` as in section 3.2.
  - `reference(..., "average")` for ref30.
- **Ablations (max 4, diagnostic only):**
  1. `coef=fixed`: beta fixed at the medians of the five reported v1.1 fit origins in section 3.4 (temp 0.75, precip 0.55,
     hottest 0.60, cdd 0.60). Does the in-fold fit help?
  2. `graph=knn25`: 25 nearest equal weights, the best kernel on the recomputed mean. Measures kernel sensitivity.
  3. `warming=half`: plus 0.5 W for temperature/CDD (v0.5's W). It shows the W exposure on development years; pre-1983 it gave
     -2.80 points in 1940-49. It cannot replace v1.1.
  4. `precip=off`: G = 0 for precipitation, the weakest component in section 3.4.
- **Record:** beta per metric, `|G|` mean, share of cities with G != 0, the median removed by centring, runtime.

### Placeholder_v1.2: long-window relative offset

- **Built on:** the version kept at that time.
  - If v1.1 is kept, the change is to make `G = centre(beta L_raw + gamma L_long_raw)`, with (beta, gamma) fitted jointly by the
    exact transform.
  - If v1.1 is not kept, v1.2 is `average30 + centre(gamma L_long_raw)`, so the change stays single.
- **Why:** the 30-year mean carries about `sd(R)/sqrt(30)` of local sampling noise (0.97 in for precipitation, 0.38 F for
  hottest day). The walk-forward gained +0.37 / +0.14 / -0.04 points over v1.1 by block, and precipitation's 1940-49 block
  moved from -0.42 to +0.08.
- **Ablations (max 3):** `long_years=45` (with `MIN_LONG_YEARS` 10); `coef=fixed` (beta: temp 0.80, precip 0.60, hottest 0.60, cdd 0.65; gamma: temp 0.40,
  precip 0.50, hottest 0.50, cdd 0.45; medians of the five reported v1.2 fit origins in section 3.4); `long_metrics=hot_precip` (gamma 0 for temperature and CDD).

### Placeholder_v1.3: v0.5 + graph-local term, one deploy path

- **Built on:** `Placeholder_v0.5`, as the series route that keeps W.
- **Change group:** add G with net weight 1 outside the blend. The learners fit the residual, beta is fitted against the
  deployed proxy, and the lambda gate scores the deployed forecast through the same `deploy`.
- **Why:** walk-forward gain of G on the proxy is +0.22 / +0.58 / +0.27 points, mostly hottest day and precipitation; beta is
  about 0 for temperature and CDD. The 1940-49 analogue shows that this route carries W's 1990-type risk, so it is run third,
  and it must meet the same rule as any version.
- **Ablations (max 2):** `gate=none` (lambda 1, separates the gate from G); `gate_target=member` (v0.5's old objective, which
  shows how much the unified objective moves lambda; diagnostic only).

**Order and stop rules:**
- Run v1.1 first.
- Run v1.2 whatever v1.1's result, since its mechanism differs.
- Run v1.3 last. If v1.1 or v1.2 has promoted, v1.3 must promote against that new champion; otherwise it is judged by rule 4.3.2.
- No v1.4. No version is tuned on a development score.

**Implementation time:** v1.1 2-3 h (new small adapter plus checks), v1.2 1 h, v1.3 3-4 h (deploy path through `fit_at`,
`choose_lambdas` and `run_forecast`). Checks, following the `test_placeholder_v0_*.py` pattern:
- G has zero cross-city median at every origin.
- v1.1 with beta 0 equals `control:average30`.
- v1.3 with beta 0 and `gate=none` equals v0.5 with `gate=none`.
- No year > fit origin is read: cut the table at t and compare L.
- Two runs give identical output.
- No NaN.

---

## 6. How each lesson is addressed

1. **Shared warming (lesson 1).** v1.1/v1.2 omit W on purpose, fixed now from the 1940-49 analogue. They forgo the warming gain
   and take no warming risk. G is centred to zero cross-city median, so it does not move the median city. It is not claimed to
   be orthogonal to W. v1.3 keeps W unchanged.
2. **The 1990 overshoot (lesson 2).** The promotion route avoids it rather than fixing it: no W in v1.1/v1.2. In the
   walk-forward, the block after early-century warming cost the W-based proxy -9.22 points and v1.1 nothing (+0.71). No damping
   is chosen on 1990, and the half-W variant is a diagnostic ablation. For v1.3 the loss stays and is reported.
3. **Zero precipitation and hottest-day skill (lesson 3).** Hottest day gained +1.03 to +1.50 points in every held-out block.
   Precipitation is the weak point: -0.42 / +0.65 / +0.22 with L alone, and +0.08 / +0.82 / +0.14 with L_long (v1.2). Its
   in-fold beta is falling over time, and `precip=off` measures it directly.
4. **Noise (lesson 4).** Two scalars per metric at most, fitted by MAE on about 25,000-30,000 city-years. Expectations are stated
   in advance (section 4.3). Champion verdict first, then the lower bound, then the reward.
5. **Reused years (lesson 5).** Design used years <= 1982 only; the overlap with the folds is tabulated in section 4.2. Three
   versions and at most 9 ablations are planned, with no version tuned on development scores. Only the integrator's 2004/2011
   assessment is clean.

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Measured so far | Sign in development results | Response |
|---|---|---|---|
| Local persistence weaker in 1984-2004 than before 1983 | beta stable for temp/hottest/cdd over 1940-1975; precip beta 0.80 to 0.50 | v1.1 negative on >= 2 origins, or the in-fold beta for 1997 far below 1983's | Report a negative result. `coef=fixed` and `precip=off` show which component failed. No retuning |
| Precipitation below -2 points (metric floor) | worst held-out block -0.42; worst triplet floor -0.08 (v1.1), -0.00 (v1.2) | `reward.py` floor check | The guard is checked, not assumed; a version failing it is not promoted |
| Live clearly worse | live +2.29 to +3.04 points over held-out blocks (v1.1) | live P(worse) | Measured in every run; no lead-specific coefficient planned |
| Bootstrap optimistic (correlated cities, overlapping triplets, design overlap) | not measurable with city draws | promotes on development but fails assessment | Integrator's one-shot 2004/2011 assessment is the check; do not over-claim a promotion |
| Edge cities with one-sided neighbours | 10th-percentile effective neighbours 19; kernel mass >= 1 for all | large `|G|` in the selection record | Log cities with `|G|` > 2 sd(R); no clipping in this plan |
| Nondeterminism | not yet measured | two runs differ | Broadcast sums only; checked before scoring |
| Implementation over budget | | | v1.1 alone is the deliverable (2-3 h) |

Half-weight placements (`inside`, `warming=half`) are diagnostics. They can never replace a version, and no undeclared fallback
will be run.

**Dependencies:** none new (numpy, pandas). No PyTorch, GPU or new release. The manifest is `placeholder-v0.1-assets.json`,
unchanged. v1.3 imports `combo_model.combo_forecast` from `code/`, as v0.5 already reaches it through `evaluation.adapters`.

**Compute:** v1.1/v1.2 about 10-15 s per origin and v1.3 about 70 s, measured or estimated as in section 3.3. At most
3 versions + 9 ablations = 12 runs, under 30 min in total.

---

## 8. Tracking

Every scored run is mirrored with `uv run python scripts/line.py sync`.

- **Names:**
  - `Placeholder_v1.1` and `Placeholder_v1.1-{coef_fixed, graph_knn25, warming_half, precip_off}`.
  - `Placeholder_v1.2` and `-{long_years_45, coef_fixed, long_metrics_hot_precip}`.
  - `Placeholder_v1.3` and `-{gate_none, gate_target_member}`.
- **Tags:** `series=Placeholder`, `version`, `plan=A-graph`, `built_on`, `ablation`, `label=official`.
- **Reward block per run, from `reward.py score --json`:**
  - verdict;
  - final delta, 90% interval, P(better), noise sd;
  - points by origin and by metric;
  - live delta and live P(worse);
  - the four guard booleans.
  The same block against v0.5 (secondary). Then `reward.py ladder` after each version.
- **Scorecard metrics to watch:**
  - `skill.origin.{1983,1990,1997}` (1990 is the test of the no-W route);
  - `skill.final.{hottest_day_temp,total_precip_in}` (the hypothesis);
  - `skill.final.{annual_avg_temp,cdd}`;
  - `skill.live.*`, `bias.*`, `win_rate`, `worst_decile_city_skill`, `p90_error_ratio.*`, `stability.range`, `runtime_s`.
- **Artifacts:** `placeholder_selection.json` with the fitted coefficients per fit origin and metric, `|G|` mean, nonzero
  share, the centring median, and for v1.3 the inner lambda table.
- **Write-up:** results in `docs/models/placeholder.md` (one section per version, with the verdict and the rule-4.3 decision).

---

## Appendix A. How the numbers were computed

All design analyses read only `data/annual_valid.csv` (filtered to `year <= 1982` with an assertion) and `data/city_info.csv`.
The scripts were throwaway (`$env:TEMP\planA_analysis*.py`, `planA_r3.py`), because the task allowed editing only this file.
The recipes below reproduce them.

- **Distances:** haversine, radius 6371 km. Tables use NaN padding before each city's first record.
- **Correlation vs distance, relative noise, decomposition (section 1.1, first three tables):** as in r2.
  - Anomalies are taken against each city's 1941-1982 mean, and pairs need >= 25 shared valid years.
  - The e-folding length is a least-squares fit of `log corr = -d/L` through the origin (corr > 0.05, d < 2000 km).
  - The decomposition is OLS without intercept on pseudo-origins 1950-1975, leads 4-7.
- **Declared estimator (r3):** exactly section 2, including the kernel-mass floor, minimum counts and centring. Skill is pooled
  over cities and rows: `1 - sum|y - f| / sum|y - ref30|`, on rows where y and ref30 are finite. NON_NEGATIVE metrics are
  clipped as deployed.
  - In-era tables pick the best coefficient on the same era (descriptive).
  - The kernel table averages three in-era best skills. Kriging uses a kernel-mass floor of 0.5, since its weights sum to 1.
- **W and combo proxy:** v0.5's definitions.
  - W: the 30-year OLS slope averaged over self + 15 nearest (`nanmean`), NaN to 0, floored at 0. CDD uses the regional OLS
    response to temperature (>= 15 joint years), floored at 0. W is extrapolated from the mean valid year.
  - combo: `0.5 (30-year OLS line, >= 10 years, else average) + 0.5 (mean of the last 10, else average)` for temperature and
    CDD; the 30-year average otherwise.
- **Walk-forward:** fit origins F = 1940..1975.
  - Fit rows are pseudo-origins t in [F-45, F-4] and leads 4-7 with t+h <= F. Grid 0-1, step 0.05 (v1.2 joint 21 x 21), loss
    on the exact deployed forecast.
  - The forecast at F is scored on F+1..F+3 (live) and F+4..F+7 (final) against ref30(F).
  - Points = 100 x mean over metrics of the pooled skill. Block values are means over origins in the block.
- **Promotion analogue:** triplets (F, F+7, F+14), F = 1940..1961.
  - Per city and per (origin, metric, split): absolute-error sums for the candidate and ref30.
  - 2,000 multinomial city draws (seed 20261108) plus the point estimate. Each draw's skill is
    `1 - (w . E) / (w . R)`, averaged over metrics and origins.
  - Guards as in `reward.py`:
    - P(better) >= 0.95 with a positive point estimate;
    - point-estimate wins on >= 2 origins;
    - live P(worse) < 0.95;
    - every metric's mean final points >= -2.
- **Runtime:** `time.perf_counter` around the precompute (81 pseudo-origins, 7.5 s) and around one origin from scratch (7.6 s),
  on the workstation in the brief, numpy 2.5.
- **Not recomputed in r3 (rejected alternatives only, marked in section 1.3):** empirical-Bayes reliability weights and the
  correlation-learned graph (r2, uncentred L).
