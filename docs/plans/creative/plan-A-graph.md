# Plan A: graph-based spatio-temporal learning for the Placeholder series (v1.x)

This plan treats the 210 cities as nodes of a fixed geographic graph and splits each city's recent anomaly into two parts. The
**low-pass part** (national and regional, shared by neighbours) is noisy, era-dependent and already handled by `Placeholder_v0.5`
(warming term W, gated learners, combo blend). The **high-pass part** (the city minus its graph neighbours) is new. Data before
1983 show it is less noisy, persistent, and has a stable coefficient across three separate eras. The model adds a
graph-local departure term `L` (v1.1) and then a long-window relative offset `L_long` (v1.2) on top of v0.5. Their coefficients
are fitted inside each fold by least absolute deviation on earlier pseudo-origins. Neither term is gated, and both are
centred so they cannot add an era-wide shift. On pre-1983 pseudo-origins the local term alone gives hottest-day skill of
+0.010 to +0.015 and precipitation skill of +0.002 to +0.007, on the two metrics where the series currently has close to zero
skill. For temperature and CDD it adds almost nothing on top of the combo blend, and the in-fold coefficient is expected to
switch it off there. The plan does not claim to fix the 1990 warming overshoot: that error is shared by every node, and a graph
high-pass filter cannot see it (section 6). Everything is numpy, with no new packages and no new data releases. The plan
reuses v0.5's asset manifest and adds less than 30 s per origin. Under the eval station's reward (`evaluation/REWARD.md`,
`96c6257`), the expected gain over v0.5 is +0.1 to +0.4 final points, which leaves the series at about 0.9-1.2 points against
the 30-year average. That is still too little to beat the champion `control:average30` beyond noise. The plan's own success
test is therefore a paired bootstrap against v0.5, where the noise is much smaller (section 4.1).

Status: draft r2 (r1 plus the reward and champion-ladder rules from the eval station, 2026-11-08)

Author: strategy planner A (Claude Opus 5.5), 2026-11-08. Base: `notebooks/Meet/placeholder_v0_5.py`. Brief: `docs/plans/creative/brief.md`.

---

## 1. Strategy and model-selection logic

### 1.1 What the data say (years <= 1982 only; method in appendix A)

All design numbers below come from `data/annual_valid.csv` and `data/city_info.csv`, cut at 1982 before any computation. That is
stricter than the allowed 2004, so no target year of any development origin (1984-2004) informed a design choice.

**Spatial correlation is strong for temperature and weak for precipitation.** Median pairwise correlation of annual anomalies,
1941-1982:

| Distance (km) | 0-250 | 250-500 | 500-750 | 750-1000 | 1000-1500 | 1500-2500 | >2500 | e-folding length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| annual_avg_temp | 0.77 | 0.71 | 0.64 | 0.54 | 0.38 | 0.10 | -0.19 | ~1180 km |
| cdd | 0.76 | 0.67 | 0.57 | 0.45 | 0.28 | 0.01 | -0.15 | ~1030 km |
| hottest_day_temp | 0.67 | 0.52 | 0.40 | 0.30 | 0.18 | 0.00 | -0.13 | ~860 km |
| total_precip_in | 0.60 | 0.42 | 0.27 | 0.18 | 0.10 | 0.04 | 0.00 | ~750 km |

The graph is dense enough to use. Median distance to the 1st, 5th, 15th and 30th nearest city is 104, 234, 423 and 634 km.

**Subtracting the neighbour mean removes most of the weather noise, and what remains persists.** Neighbour-relative anomaly
`R = city anomaly - mean of its 15 nearest cities' anomalies`, 1941-1982:

| Metric | sd(anomaly) | sd(R) | lag-1 autocorr anomaly | lag-1 autocorr R | lag-5 autocorr R |
|---|---:|---:|---:|---:|---:|
| annual_avg_temp | 1.26 F | 0.70 F | 0.30 | 0.57 | 0.15 |
| cdd | 184 | 114 | 0.25 | 0.46 | 0.11 |
| hottest_day_temp | 2.99 F | 2.08 F | 0.14 | 0.14 | 0.00 |
| total_precip_in | 7.26 in | 5.30 in | 0.05 | 0.06 | 0.01 |

**Splitting the recent departure into national, regional and local parts shows which part is stable.** Pooled OLS of the
lead-4-7 departure `y(t+h) - ref30(t)` on three parts of the last-10-year gap. The parts are the national median gap, the
regional gap (15-NN mean) minus national, and the local gap (own minus regional). The fit uses pseudo-origins
t = 1950-1975 (about 20,000 city-years per metric), with the national part removed from the target:

| Metric | regional coef 1950-62 | regional coef 1963-75 | local coef 1950-62 | local coef 1963-75 |
|---|---:|---:|---:|---:|
| annual_avg_temp | 0.11 | 0.70 | 0.66 | 0.54 |
| cdd | 0.58 | 0.82 | 0.69 | 0.51 |
| hottest_day_temp | 0.25 | 1.42 | 0.53 | 0.44 |
| total_precip_in | -0.37 | -0.53 | 0.39 | 0.29 |

The national coefficient for temperature over 1950-1975 is 1.32: in the cooling era, recent national cold persisted. This is
the effect lesson 1 describes, and the reason a learner must not see it. Regional coefficients change several-fold between eras.
Local coefficients stay within about 0.1. **The local part is the era-stable signal, and it is the one the series does not model
explicitly.** In v0.3 the learners see `gap10` and `reg_gap10`, mixed with the unstable regional part, and the gate scales all of
it together. The gate chose lambda 0 at most origins (v0.1, v0.4).

**The local term alone is worth about +0.01 for hottest day and +0.005 for precipitation, in every era tested.** The table shows
skill of `ref30 + b * L` against `ref30` alone, at the best b, on leads 4-7. L uses a Gaussian 600 km kernel, centred across cities.
The half-life is 3 years for the temperature metrics and 6 for precipitation (definitions in section 2):

| Metric | 1915-35: b, skill | 1950-62: b, skill | 1963-75: b, skill | Gain on top of a combo-like blend (3 eras) |
|---|---|---|---|---|
| hottest_day_temp | 0.65, +0.0135 | 0.65, +0.0148 | 0.55, +0.0105 | same (combo = 30-year mean here) |
| total_precip_in | 0.65, +0.0062 | 0.75, +0.0073 | 0.35, +0.0018 | same (combo = 30-year mean here) |
| annual_avg_temp | 0.75, +0.0116 | 0.90, +0.0322 | 0.55, +0.0115 | +0.0001 / +0.0018 / 0.0000 |
| cdd | 0.60, +0.0093 | 0.75, +0.0254 | 0.60, +0.0145 | 0.0000 / +0.0016 / +0.0001 |

The combo-like blend is `0.5 * ref30 + 0.5 * combo`. combo is `0.5 * trend30 + 0.5 * last10` for temperature and CDD, and the
30-year mean for precipitation and hottest day. For temperature and CDD, combo's last-10 mean already carries the local signal.
For precipitation and hottest day nothing in v0.5 carries it except the gated learners.

**A long-window relative offset adds more for precipitation and hottest day.** Adding
`L_long = mean(R over 60 years) - mean(R over 30 years)`, with L at b = 0.55, gives these skills on pseudo-origins 1940-1975:

| Metric | L only | L + best gamma * L_long | best gamma | joint OLS coefs [L, L_long] |
|---|---:|---:|---:|---|
| hottest_day_temp | +0.0079 | +0.0105 | 0.4 | 0.49, 0.30 |
| total_precip_in | +0.0018 | +0.0045 | 0.5 | 0.52, 0.54 |
| annual_avg_temp | +0.0241 | +0.0251 | 0.2 | 0.84, 0.39 |
| cdd | +0.0206 | +0.0218 | 0.2 | 0.79, 0.36 |

Reading: a city's 30-year mean contains about `sd(R)/sqrt(30)` of local sampling noise, which is 0.97 in for precipitation and
0.38 F for hottest day. The city's longer record relative to its neighbours estimates its stable offset better. This is the
classic reference-series method for climate normals, applied as a forecast term.

### 1.2 Chosen model: graph high-pass terms on top of v0.5

```
forecast(c, m, t+h) = 0.5 * P(c, m, t+h) + 0.5 * combo(c, m, t+h) + 0.5 * G(c, m, t)        (clip >= 0 for precip, cdd)
P(c, m, t+h)        = ref30 + W(c, m, h) + G(c, m, t) + lambda_m * rhat(c, m, h)            (v0.3 member, now with G)
G(c, m, t)          = beta_m * L(c, m, t)                       (v1.1)
                    = beta_m * L(c, m, t) + gamma_m * L_long(c, m, t)   (v1.2)
```

G enters both members of the 50/50 blend, so its net weight is 1, and it is not gated. The learners fit what is left:
`y - ref30 - W - G`, with the cross-city median removed per (t, h) as in v0.3. beta and gamma are fitted inside each fold
(section 4). W, the combo blend, the learners and the lambda gate are v0.5's, unchanged.

Why this suits the data:

- **210 nodes and roughly 40 effective pseudo-origins** cannot support a per-node or deep model. The graph contributes a fixed
  linear operator, `I - P`, where P is the row-normalised kernel, plus one or two scalars per metric. Each scalar is fitted on
  roughly 25,000-30,000 city-years.
- **The shared non-stationary trend** sits in the low-pass part. `(I - P)` removes it, and the median centring removes any
  remainder, so G cannot learn mid-century cooling or add to the warming overshoot.
- **Heavy annual noise:** R has 27-44% smaller sd than the raw anomaly. EWMA and long-window averages of R reduce it further.
- **MAE scoring:** the coefficients minimise mean absolute error directly, by grid search.
- **Compute:** a 210 x 210 kernel and a few broadcasts per pseudo-origin. Under 30 s per origin on CPU, no GPU.

### 1.3 Alternatives considered and rejected

| Candidate | Why not |
|---|---|
| **Graph neural network / learned message passing** (needs `uv add torch`) | The signal in section 1.1 is linear: one high-pass operator and one coefficient. A one-layer GNN with fixed weights is exactly `P`. Learned message weights over 210 nodes and about 40 strongly overlapping snapshots would fit era-specific regional patterns, the part shown to be unstable. A GNN also adds a lock change and GPU nondeterminism risk for no expected gain. |
| **Graph-Laplacian multi-task regression** (per-city coefficients smoothed over the graph) | It assumes coefficients differ across cities. I tested the closest cheap version: empirical-Bayes per-city reliability weights on L (signal variance against white-noise variance per city). It lowered skill for every metric, e.g. hottest day +0.0119 to +0.0093 (1915-35) and precipitation to 0.0000, where the white-noise assumption sets every weight near 0. No evidence of usable heterogeneity, so the pooled coefficient stays. |
| **Spatial GP / ordinary kriging of the neighbour mean** (exponential covariance at each metric's e-folding length, nugget 0.25, 30 nearest) | Against the Gaussian kernel, mean over three eras: hottest +0.0121 vs +0.0112, precipitation +0.0045 vs +0.0051, temperature +0.0190 vs +0.0184, CDD +0.0153 vs +0.0164. No consistent gain, and it adds an in-fold length-scale fit. Not run. |
| **Kriging / graph smoothing of the warming trend W** | W's error at 1990 is common to all nodes (section 6), so smoothing over space cannot reduce it. A change to W cannot be validated before 1983, where the 30-year slope is mostly floored at 0. Not run. |
| **Correlation-learned graph** (top-10 neighbours by first-difference correlation, corr^2 weights) | Best for temperature (+0.0215 vs +0.0184 mean standalone), worse for hottest day (+0.0084 vs +0.0108) and precipitation. For temperature L is redundant with combo anyway. |
| **Break-aware local level** (level of R after the most significant single break, t > 4) | No gain for hottest day (+0.0128 / +0.0145 / +0.0101 vs EWMA +0.0135 / +0.0148 / +0.0105). For temperature it adds on top of the blend in one era only (1950-62: +0.0123 vs +0.0062). Kept as a possible later idea, not planned. |
| **Regional precipitation mean reversion** (regional coef -0.37 / -0.53) | Same sign in 1950-62 and 1963-75, but both eras sit in one drought-to-wet cycle. Not tested on an independent era. Too likely to be era-specific to apply ungated. |
| **Per-city EB shrinkage of noisy W slopes toward neighbours** | Already done by v0.3's 15-NN slope averaging. The remaining W error is shared. |

Kernel choice, from the standalone skill of L averaged over three eras (EWMA half-life 3, or 6 for precipitation):

| Graph | temp | cdd | hottest | precip |
|---|---:|---:|---:|---:|
| 5-NN equal | +0.0138 | +0.0119 | **+0.0115** | +0.0038 |
| 15-NN equal (v0.3's graph) | +0.0155 | +0.0144 | +0.0090 | +0.0053 |
| 25-NN equal | +0.0159 | +0.0148 | +0.0085 | **+0.0057** |
| Gaussian 300 km | +0.0175 | +0.0144 | +0.0106 | +0.0050 |
| **Gaussian 600 km (chosen)** | **+0.0184** | **+0.0163** | +0.0108 | +0.0052 |
| correlation top-10 | +0.0215 | +0.0132 | +0.0084 | +0.0048 |

The Gaussian 600 km kernel is best or within 0.0007 of the best on every metric except temperature, where L is switched off by
the blend anyway. It has a median of 45 effective neighbours (`1 / sum w^2`; 10th percentile 19), and no city has kernel mass
below 1.

---

## 2. Targets and inputs

**Predicted quantity:** unchanged from v0.5. Final output is the level `predicted_value`. Internally, every term is a shift from
`ref30(c, t)`, the official 30-year mean (>= 10 valid years).

**Inputs:** `train` (suite `train.csv` per fold, cut at the origin by the harness) and `data/city_info.csv` (lat/lon, the
declared `static_context` asset in `notebooks/Meet/placeholder-v0.1-assets.json`). Nothing else: no `features-v1`, no
`clean-v1`, no daily data, so runs stay `official`.

**Graph terms at a (pseudo-)origin t, per metric m, from years <= t only:**

- `K_ij = exp(-(d_ij / 600 km)^2)`, with `K_ii = 0` and d the haversine distance (earth radius 6371 km). Built once from coordinates.
- Window `X = values for t-29..t` (NaN where invalid). Anomaly `A = X - ref30(t)[:, None]`.
- Neighbour mean `N_c(y) = sum_j K_cj A_j(y) / sum_j K_cj 1[A_j(y) valid]`. It is NaN if the valid kernel mass is below 1.0.
- `R = A - N` where `X` is valid.
- `L = EWMA_hl(R over valid years) - mean(R over valid years)`, with half-life `hl = 3` years (annual_avg_temp,
  hottest_day_temp, cdd) or 6 (total_precip_in). L is required to have >= 10 valid R years, else 0. Then `L -= median_c(L)`.
- v1.2 only: extend the window to t-59..t with the same `ref30(t)`, and set
  `L_long = mean(R over t-59..t) - mean(R over t-29..t)`. It needs >= 15 valid R years in t-59..t-30 and >= 10 in t-29..t,
  else 0. Then
  `L_long -= median_c(L_long)`.
- L and L_long do not depend on h: one value per city and origin, applied to all seven leads.

Every array is built from `table.loc[:, :t]` for the fold table, which `tables_at` has already cut at the fit origin and checked
with `assert_no_post_cutoff`. Neighbours' values come from the same years. Coordinates are static.

---

## 3. Training procedure

### 3.1 Pseudo-origins and coefficient fit

At each fit origin F (the outer origin, and each inner origin O-21, O-14, O-7 inside `choose_lambdas`), the procedure is:

1. Build `states[t]` for t in [F-45, F-1] exactly as v0.5 (`build_states`), plus `L[t]` (and `L_long[t]`) per metric.
2. Coefficient rows: every (t, h) with h in {4, 5, 6, 7} and t + h <= F, where `y(t+h)`, `ref30(t)` are finite. Each row has a
   **proxy** for the forecast without G: `0.5 * (ref30 + W(t, h)) + 0.5 * combo(t, h)`. combo is
   `combo_model.combo_forecast(table, metric, t, [t+h])`, the same function the harness reference uses, cut at t.
3. `beta_m = argmin_{b in GRID} mean |y - proxy - b * L|` (v1.1). In v1.2, `(beta_m, gamma_m)` is the argmin over GRID x GRID of
   `mean |y - proxy - b L - g L_long|`. GRID = 0.00, 0.05, ..., 1.00. Ties go to the smallest gamma, then the smallest beta,
   by grid order. If there are fewer than 2,000 rows, the coefficient is 0.
4. Learner training rows as v0.5, with target `y - ref30 - W - G` and the cross-city median removed per (t, h).
5. Members (Huber + HistGBM, v0.5 settings) fitted once per metric.

The leads are restricted to 4-7 because the headline scores those leads. The lead-by-lead fit for precipitation falls from 0.75
(h4) to 0.45 (h7). A fit over leads 1-3 would overshoot the final window.

The proxy omits the learners (`lambda * rhat`). Including them would need learner fits at every pseudo-origin, which is too
expensive. Because the learners then fit the residual after G, the learners and G cannot both claim the same signal.

### 3.2 Pseudo-code (diff against `placeholder_v0_5.py`)

```python
MODEL_NAME = "Placeholder_v1.1"
KERNEL_KM = 600.0
MIN_KERNEL_MASS = 1.0
HALF_LIFE = {"annual_avg_temp": 3, "hottest_day_temp": 3, "cdd": 3, "total_precip_in": 6}
MIN_REL_YEARS = 10
COEF_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
COEF_LEADS = (4, 5, 6, 7)
MIN_COEF_ROWS = 2000
# v1.2 adds: LONG_YEARS = 60; MIN_LONG_YEARS = 15

def graph_kernel(coords):                      # coords sorted by city_id (as neighbour_index)
    d = haversine_km(coords)                   # same formula as neighbour_index, x 6371
    K = np.exp(-(d / KERNEL_KM) ** 2)
    np.fill_diagonal(K, 0.0)
    return K

def neighbour_mean(K, A):                      # A: cities x years, NaN = invalid
    ok = np.isfinite(A)
    num = (K[:, :, None] * np.where(ok, A, 0.0)[None, :, :]).sum(axis=1)   # broadcast-sum: deterministic order
    den = (K[:, :, None] * ok[None, :, :]).sum(axis=1)
    return np.where(den >= MIN_KERNEL_MASS, num / np.where(den > 0, den, 1.0), np.nan)

def relative_series(table, t, K, first_year):
    X = table.loc[:, :t].reindex(columns=range(first_year, t + 1)).to_numpy(float)
    ref = thirty_year_mean(table.loc[:, :t], t).to_numpy(float)
    A = X - ref[:, None]
    return np.where(np.isfinite(X), A - neighbour_mean(K, A), np.nan)

def centred(v):
    v = v - np.nanmedian(v)
    return np.nan_to_num(v, nan=0.0)

def local_term(table, t, K, metric):
    R = relative_series(table, t, K, t - 29)
    ok = np.isfinite(R); n = ok.sum(axis=1)
    w = 0.5 ** (np.arange(29, -1, -1) / HALF_LIFE[metric])
    ewma = (np.where(ok, R, 0) * w).sum(1) / np.maximum((ok * w).sum(1), 1e-12)
    mean30 = np.where(ok, R, 0).sum(1) / np.maximum(n, 1)
    return centred(np.where(n >= MIN_REL_YEARS, ewma - mean30, np.nan))

def long_offset(table, t, K):                  # v1.2
    R = relative_series(table, t, K, t - 59)
    ok = np.isfinite(R)
    early = ok[:, :30].sum(1)
    all_mean = np.where(ok, R, 0).sum(1) / np.maximum(ok.sum(1), 1)
    last_mean = np.where(ok[:, 30:], R[:, 30:], 0).sum(1) / np.maximum(ok[:, 30:].sum(1), 1)
    return centred(np.where((early >= MIN_LONG_YEARS) & (ok[:, 30:].sum(1) >= MIN_REL_YEARS), all_mean - last_mean, np.nan))

def fit_coef(y, proxy, L):                     # LAD by grid; first minimum = smallest coefficient
    if len(y) < MIN_COEF_ROWS:
        return 0.0
    loss = [np.mean(np.abs(y - proxy - b * L)) for b in COEF_GRID]
    return float(COEF_GRID[int(np.argmin(loss))])

# training_rows(tables, neighbours, K, fit_origin, ...):
#   states = build_states(...)                                   # unchanged
#   for metric: local[t] = local_term(tables[metric], t, K, metric) for t in t_values
#   coef rows: for h in COEF_LEADS, t in t_values with t+h <= F and t+h in table:
#       proxy = 0.5*(ref30 + W(t,h)) + 0.5*combo_forecast(table, metric, t, [t+h])[t+h]
#   beta[metric] = fit_coef(y, proxy, local)
#   learner target = y - ref30 - W - beta*local[t]; then subtract median (as v0.5)
#
# fit_at(...): also returns local_shift[metric] = DataFrame(beta*local[F]) for each target year (0 where no ref30)
# assemble(base, shifts, warming, local_shift, lambdas): pred = base + warming + local_shift + lambda*shift
#     (used unchanged by choose_lambdas, so the gate scores P including G)
# run_forecast: out = blend_with_combo(assemble(...), combo)
#               out.predicted_value += 0.5 * local_shift          # G at net weight 1
#               clip >= 0 for total_precip_in, cdd
#   record: beta per metric per fit origin, mean |G|, share of cities with G != 0, median-centring offsets
```

The combo values at pseudo-origins should be cached per (metric, t) and computed once per fit origin, not once per lead.

### 3.3 Seeds, determinism, runtime

- Only the GBM member is stochastic, and it keeps v0.5's `random_state=seed` (42). The graph terms are closed-form. Coefficient
  grids are scanned in fixed order. The kernel sums use broadcasting rather than BLAS `@`, so the summation order does not
  depend on thread count. Two consecutive runs must give byte-identical `origin-*.csv` (implementation check).
- Runtime per origin: v0.5's runs `...141546Z...` and `...141730Z...` started 104 s apart, so v0.5 takes under about 35 s per
  origin. This is inferred from run ids; read `runtime_s` in its scorecard to confirm. Additions per fit origin: about 45
  pseudo-origins x 4 metrics x one 210 x 210 x 30 (v1.2: x 60) broadcast, about 0.01 s each, so about 2-4 s. combo at 45
  pseudo-origins x 4 metrics is about 3 s (trend_forecast's per-city polyfit). With 4 fit origins per outer origin (1 outer +
  3 inner), that is about 25 s added. Expected total: about 60 s per origin, far below the 900 s brief limit and the 600 s
  default in `evaluate.py run`.

---

## 4. Selection without peeking

**Fixed in advance (from appendix A, years <= 1982, before the earliest outer origin):** kernel Gaussian 600 km;
`MIN_KERNEL_MASS = 1.0`; EWMA half-lives {3, 3, 3, 6}; 30-year relative window; `MIN_REL_YEARS = 10`; v1.2 long window 60 years,
`MIN_LONG_YEARS = 15`; median centring; coefficient grid 0-1 step 0.05; coefficient leads 4-7; `MIN_COEF_ROWS = 2000`;
G at net weight 1 (outside the blend's halving). All v0.5 settings are unchanged: `TRAIN_YEARS = 45`, `N_NEIGHBOURS = 15` for
W, Huber/GBM settings, `INNER_OFFSETS = (21, 14, 7)`, lambda grid and margin, `COMBO_WEIGHT = 0.5`.

**Chosen inside each fold:** `beta_m` (and `gamma_m` in v1.2), refitted at every fit origin from pseudo-origins t <= F-4 with
targets <= F (section 3.1). `lambda_m` comes from inner origins O-21, O-14, O-7 by v0.5's rule. The inner folds compute their
own beta from their own history, so nothing crosses an inner boundary. No value is set after looking at a development score.

**Pre-registered expectation** (so a result can falsify the idea): against v0.5, final skill on hottest_day_temp rises by
+0.004 to +0.012 and on total_precip_in by +0.001 to +0.005. annual_avg_temp and cdd move by less than 0.003, with fitted beta
below 0.3. Headline rises by +0.001 to +0.004. If hottest-day skill falls at 2 or more origins, the persistence found before
1983 does not hold in the development era. v1.2 is still run, since it uses a different mechanism (stable offsets, not recent
departures). Stop after v1.2.

**Keep rule (mechanical, fixed now).** A version replaces the current kept version only if both hold:
(a) its headline (equal-origin mean of unclipped final skill, leads 4-7) is strictly higher, and (b) its origin-1997 final skill
is strictly higher. Values are compared at full precision from the scorecard. A tie fails. The first comparison is against
`Placeholder_v0.5` (headline +0.0083, 1997 +0.0458, run `20261108T141546Z-d0ac8f8b36`). A later version is compared with
whichever version is kept at that time. Ablations can never be kept. To be a submission candidate a kept version must also pass
G1-G4 and be an `official` run. Scores at 1983 and 1990 are reported but do not decide.

### 4.1 Reward and champion ladder (eval station, `evaluation/REWARD.md` at `96c6257`)

The eval station now judges models by **reward = final points**: `100 x` the mean over origins 1983/1990/1997 of the mean
unclipped skill over the 4 metrics, on leads 4-7. That is the headline above times 100, so v0.5 is about +0.83 points. The
verdict against the champion (still `control:average30`) is `promote` only when all of these hold:

- P(better on final) >= 0.95 under a paired city bootstrap (2,000 draws).
- Better on >= 2 of 3 origins.
- Live points not clearly worse.
- No metric below -2 points.

v0.3-v0.9 are all `within_noise` (P 0.56-0.84). The eval station reports noise of roughly 0.3-0.9 points SD, so a gain needs to
be about 1.5 points or more against the 30-year average to register.

What this means for the plan:

1. **Honest expectation against the champion.** The pre-registered gain over v0.5 is +0.1 to +0.4 points: hottest day +0.4 to
   +1.2 points on its metric and precipitation +0.1 to +0.5, each divided by 4. That puts v1.1/v1.2 at about 0.9-1.2 points
   against the 30-year average, so the expected verdict is `within_noise`. This plan alone is unlikely to produce `promote`. A
   promotable model also needs a larger, more stable temperature/CDD gain, which only the shared warming term can deliver and
   which section 6 explains this plan leaves alone. The graph terms add to that gain and do not compete with it, because they
   are centred and orthogonal to the shared signal.
2. **The plan's own test is paired against v0.5:**
   `uv run python code/evaluation/reward.py score <v1.x run> --against 20261108T141546Z-d0ac8f8b36 --json`. v1.1 differs from
   v0.5 only by G (the learners' refit aside). The shared weather and warming errors are identical in both runs and cancel in
   the paired draws, so this comparison has far less noise than either run against the 30-year average.
   - **Success criterion for the graph idea, fixed now:** P(better than v0.5 on final) >= 0.95, better on >= 2 of 3 origins,
     and no metric worse than v0.5 by more than 0.5 points.
   - If that holds and the champion verdict is still `within_noise`, the term is real but small. Report it that way, not as
     progress on the ladder.
3. **Order of judgement, as REWARD.md asks:** verdict against the champion first, then the lower bound of the 90% interval,
   then the reward.
   - The series keep rule above (headline and 1997 strictly higher than the kept version) still decides what the next version
     builds on.
   - A version whose champion verdict is `worse` is never kept, whatever its headline.
   - Among versions that are all `within_noise`, the higher 90% lower bound is preferred when the team picks a candidate.
4. **Guards the plan already meets.**
   - *No metric below -2 points:* this is G3; coefficients are floored at 0 and fitted per fold.
   - *Live not clearly worse:* beta is fitted on leads 4-7 and applied unchanged to leads 1-3. Persistence is stronger at short
     leads (precipitation beta 0.75 at h4, falling with lead), so a lead-4-7 beta is conservative for the live window.
     `live` points are checked against v0.5 in the paired run.
5. **Why the graph terms suit a city bootstrap.** W's gain is one shared shift per origin, so its sign depends on the period,
   and that is not resampled. L and L_long are high-pass terms that change many cities by small, independent-looking amounts,
   which is the kind of gain a city bootstrap can confirm. The REWARD.md caveat applies: nearby cities are correlated, so the
   interval is if anything too narrow. L is less spatially correlated than raw anomalies by construction, which reduces that
   bias.
6. **Inputs stay gate-eligible.** The plan uses only `data/city_info.csv` and suite `train.csv`, so every run is `official` and
   can become champion. `enhance-v2` (SST indices) is not used: it would make runs exploratory, and it is low-pass climate
   information, outside this plan's high-pass mechanism. `enhance-v1` (static elevation, coast distance, land cover;
   gate-eligible) could add climatological-similarity edges to the kernel, for example by down-weighting pairs that differ by
   more than 500 m in elevation or across a coastline. I have not tested this (the release is not in this checkout), so it is
   not a planned version. It would be a v1.3 only if a pre-1983 check like appendix A shows a gain, and only with
   `enhance-v1` declared in a copied manifest.

---

## 5. Versions

Each version is one single-file adapter in `notebooks/Meet/`, committed, then scored from a clean commit:

```powershell
uv run python scripts/line.py status
uv run python scripts/line.py score --name Placeholder_v1.1 --adapter notebooks/Meet/placeholder_v1_1.py:forecast `
  --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3
# ablation: same command, --name Placeholder_v1.1-<ablation> --params notebooks/Meet/placeholder-v1.1-<ablation>.json
uv run python scripts/line.py sync
```

Ablations use adapter keyword parameters (as v0.5's `members` / `gate` / `features`), with the defaults equal to the version
itself.

### Placeholder_v1.1: graph-local departure (main bet)

- **Built on:** v0.5.
- **Change (one group):** add `G = beta_m * L` for all four metrics, as in sections 1.2, 2 and 3. That covers L, the in-fold
  beta, the learners fitting `y - W - G`, and G at net weight 1.
- **Why:** the only component in section 1.1 with a stable coefficient across three eras, and the only skill source found for
  hottest day and precipitation.
- **Ablations (max 4, in this order):**
  1. `coef=fixed`: beta fixed from appendix A instead of fitted. Values: hottest 0.60, precipitation 0.60, temperature 0, CDD 0
     (era-best values averaged and rounded to the grid; 0 for temperature and CDD, where L adds nothing on top of combo). Tests whether the in-fold
     fit is worth its noise.
  2. `graph=knn15`: equal-weight 15 nearest cities (v0.3's neighbour set) instead of the Gaussian kernel. Tests the kernel choice.
  3. `placement=inside`: G only inside P, so net weight 0.5. beta is fitted against the proxy `ref30 + W`. This is the
     conservative fallback if the full weight overshoots.
  4. `precip=off`: no G for total_precip_in. Isolates the lesson-3 claim.
- **Artifacts:** `placeholder_selection.json` gains `local_coef[fit_origin][metric]`, `local_abs_mean`, `local_nonzero_share`,
  `kernel_km`, `half_life`.

### Placeholder_v1.2: long-window relative offset

- **Built on:** the version kept at that time (v1.1 if kept, else v0.5).
- **Change (one group):** add `gamma_m * L_long` to G, with `(beta_m, gamma_m)` refitted jointly on the 21 x 21 grid. If built
  on v0.5, beta is fixed at 0 and only gamma is fitted, so the change stays single. The learners fit `y - W - G` with the new G.
- **Why:** section 1.1. The 30-year mean carries about `sd(R)/sqrt(30)` of local sampling noise. The 60-year relative record
  lowers it, which gave +0.0027 precipitation and +0.0026 hottest-day skill before 1983. This is the one part of the plan aimed
  at precipitation's own climatology rather than its persistence.
- **Ablations (max 3):**
  1. `long_years=45`: shorter long window, `MIN_LONG_YEARS = 10`.
  2. `long_metrics=hot_precip`: gamma fixed at 0 for temperature and CDD.
  3. `coef=fixed`: gamma fixed at hottest 0.4, precipitation 0.5, temperature 0.2, CDD 0.2 (appendix A), and beta as in v1.1's
     fixed ablation.

No v1.3 is planned. The remaining graph candidates (kriging weights, break-aware levels, correlation graphs) showed no
consistent gain before 1983 (section 1.3). Spending more runs on reused development years would add selection noise rather than
evidence. If both versions fail the keep rule, report that, and do not tune a v1.3 to the development scores.

### Implementation time

v1.1: 3-4 h (graph functions, coefficient fit, plumbing through `training_rows` / `fit_at` / `assemble` / `run_forecast`,
selection record, checks below), plus about 10 min of scoring with 4 ablations. v1.2: 1.5 h. Checks to write with the adapter,
following the existing `test_placeholder_v0_*.py` pattern:

- G has zero cross-city median at every origin.
- With all coefficients forced to 0, output equals v0.5 to the byte.
- The function reads no year > fit origin (cut the table at t and compare L).
- Two runs give identical output.
- No city gets NaN.

---

## 6. How each lesson is addressed

1. **Shared warming (lesson 1).** W is unchanged. G is a graph high-pass term, centred to a zero cross-city median at every
   origin, so it adds no era-wide shift and cannot learn mid-century cooling. The beta fit's proxy includes W, so beta only
   explains what W and combo leave. The national component, which has coefficient 1.32 in the cooling era, is never put into a
   learned term.
2. **The 1990 overshoot (lesson 2).** This plan does not try to fix it, for these reasons:
   - The 1990 temperature loss (-0.115 for v0.3, -0.073 for v0.5) comes from W with lambda 0 (v0.4), so it is the shared
     component.
   - Anomaly correlation is 0.38 at 1000-1500 km and still 0.10 at 1500-2500 km. A national-scale error moves every node
     together, and `(I - P)` removes exactly that part. No graph operator can tell a too-steep national slope from a real one.
   - The only fix would be damping W. Damping chosen now would be chosen after seeing 1990. Damping chosen before 1983 would
     select W = 0, because 30-year warming before 1983 was followed by mid-century cooling. Either way it would cost 1997.
   - What the graph does at 1990: G adds skill there through a mechanism that does not depend on W. Every pre-1983 era gained
     on hottest day and precipitation, including 1963-75, when the shared signal was moving against persistence. So the 1990
     loss should shrink slightly, not grow. The v0.5 blend, which halves W's error, stays.
3. **Zero precipitation and hottest-day skill (lesson 3).** This is the main target. Precipitation and hottest day have no
   persistence in raw anomalies (lag-1 0.05 and 0.14). They do have a small persistent local offset that the gated learners
   cannot separate from the unstable regional part. v1.1 applies that offset ungated, with a fitted beta, for +0.002 to +0.007
   (precipitation) and +0.010 to +0.015 (hottest day) before 1983. v1.2 denoises the climatology itself. The `precip=off`
   ablation measures the precipitation part alone.
4. **Noise dominates (lesson 4).** Expected gains are about +0.001 to +0.004 headline (+0.1 to +0.4 reward points), stated in
   advance. That is below the roughly 1.5 points needed to register against the champion, so the plan's own test is the paired
   bootstrap against v0.5 (section 4.1). Coefficients are
   fitted on roughly 25,000-30,000 city-years per metric, with a 0-1 grid and a floor at 0. The terms are closed-form, so variance comes
   only from beta and gamma.
5. **Reused development years (lesson 5).** Every design choice in this plan used years <= 1982. Two versions, with at most 7
   ablations in total, are planned, and no v1.3 tuned on development scores. Coefficients are refitted per fold, so the
   development scores test the procedure rather than one number. Development scores remain reused evidence. The integrator's
   2004/2011 assessment is the only clean test.

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Sign in the results | Fallback |
|---|---|---|
| Local persistence comes mostly from station changes, which are rarer after 1989 (32 listed in 1989-2018, none in 2019-2023). The pre-1983 beta would then be too large for later years. | hottest / precip skill below v0.5 at 1997; beta fitted at 1997 well below the 1983 value | `placement=inside` (half weight). v1.2's long offset works the other way (stable stations), so run it anyway. For the real forecast, breaks before 2018 persist and no new ones are listed, which favours L. |
| G and the learners double count hottest day | v1.1 hottest skill gain clearly below the standalone estimate; learner lambda for hottest falls to 0 | Learners already fit `y - W - G`; no further change |
| Edge cities (Florida, Pacific coast, isolated western cities) have one-sided neighbours | large `|G|` at edge cities in the selection record | The kernel-mass floor plus median centring; the 10th-percentile effective neighbour count is 19. If a city's `|G|` exceeds 2 sd(R), log it; do not clip in this version |
| G3 (no metric below -0.02) | a negative metric | Every coefficient is floored at 0 and its loss is checked on training rows, so a large loss means the era changed; report it |
| 1990 stays negative | expected | Not a fallback trigger; see section 6 |
| Nondeterminism | two runs differ | Broadcast sums only; no `@` in graph code |
| Implementation time | over 1 day | v1.1 alone is the deliverable; v1.2 is 1.5 h |
| Gain real but too small for the ladder | champion verdict `within_noise`, paired verdict against v0.5 meets the section 4.1 criterion | Report as a confirmed small component. Keep it in the series for whichever temperature change (for example plan B) targets promotion. Do not add versions to chase the ladder on reused folds |
| Gain not confirmed even against v0.5 | paired P(better than v0.5) < 0.95 | The graph terms are noise at this scale. Stop after v1.2 and record a negative result |

**Dependencies:** none new. No PyTorch, no GPU, no new data release, no `features-v1`, so runs stay `official`. The asset
manifest is `notebooks/Meet/placeholder-v0.1-assets.json`, unchanged. `combo_model.combo_forecast` is imported from `code/`,
which the adapter already reaches through `evaluation.adapters`.

**Compute:** about 60 s per origin, about 3 min per scored run, 9 runs at most (2 versions + 7 ablations), under 30 min in total.

---

## 8. Tracking

Every scored run (versions and ablations) is mirrored with `uv run python scripts/line.py sync`.

- Run names `Placeholder_v1.1`, `Placeholder_v1.1-coef_fixed`, `Placeholder_v1.1-graph_knn15`, `Placeholder_v1.1-placement_inside`,
  `Placeholder_v1.1-precip_off`, `Placeholder_v1.2`, `Placeholder_v1.2-long_years_45`, `Placeholder_v1.2-long_metrics_hot_precip`,
  `Placeholder_v1.2-coef_fixed`. Tags `series=Placeholder`, `version=v1.x`, `plan=A-graph`, `built_on=<kept version>`,
  `ablation=<name or none>`.
- Reward, after every run (`line.py score` prints it):
  - `uv run python code/evaluation/reward.py score <run_id>` (against the champion)
  - `uv run python code/evaluation/reward.py score <run_id> --against 20261108T141546Z-d0ac8f8b36 --json` (paired against v0.5,
    the plan's own test)
  - `uv run python code/evaluation/reward.py ladder` (current champion)
  - Record the verdict, the final delta, the 90% interval, P(better), the noise sd, the per-origin points and the live delta, for
    both comparisons, in `docs/models/placeholder.md`.
- Metrics that decide: the champion verdict and the paired verdict against v0.5 (section 4.1), then the series keep rule
  (headline and `skill.origin.1997`).
- Metrics to watch: `skill.origin.1983` and `skill.origin.1990`; `skill.final.hottest_day_temp` and `skill.final.total_precip_in`
  (the hypothesis); `skill.final.annual_avg_temp` and `skill.final.cdd` (must not move by more than 0.003);
  `bias.<metric>` (G must not shift bias); `skill.live.<metric>`; `win_rate`; `worst_decile_city_skill`;
  `p90_error_ratio.<metric>`; `stability.range`; `runtime_s`.
- Artifacts: `placeholder_selection.json` with the fitted beta and gamma per fit origin and metric, `local_abs_mean` and
  `local_nonzero_share`. A loss of 0 for any coefficient fit at every origin is itself a finding (signal absent), and should be
  reported in `docs/models/placeholder.md`.
- Results go into `docs/models/placeholder.md` (one section per version, with the keep decision) and section 6 of
  `docs/plans/placeholder-improvement-plan.md`, as for v0.3-v0.9.

---

## Appendix A. How the numbers were computed

Inputs: `data/annual_valid.csv` (filtered to `year <= 1982` right after reading, with an assertion) and `data/city_info.csv`.
Nothing under `private/`, nothing after 1982, and no run outputs were read. The scripts were throwaway (`$env:TEMP`), because
this task allowed editing only this file. The recipes below reproduce them.

- **Distances:** haversine, radius 6371 km.
- **Correlation vs distance:** per metric, a city x year table for 1941-1982, anomaly = value minus the city's mean over its valid
  years in that period. Pairwise Pearson correlation on shared valid years (>= 25), median per distance bin. e-folding length
  from a least-squares fit of `log corr = -d / L` through the origin, on pairs with corr > 0.05 and d < 2000 km.
- **Relative anomaly table (1.1, second table):** the same anomalies; R = anomaly minus the equal-weight mean of the 15 nearest
  cities' available anomalies. Lag-k autocorrelation is pooled over all cities and years.
- **Decomposition table:** pseudo-origins t = 1950-1975, h = 4-7. `ref30` = official 30-year mean (>= 10 valid years).
  `gap10` = mean of the valid years in t-9..t minus ref30. national = cross-city median of gap10; regional = 15-NN mean of gap10;
  local = own gap10 minus regional. OLS without intercept of `(y - ref30) - national` on [regional - national, local], per era.
- **Local-term skill:** L as in section 2, but for the analysis R uses anomalies against each city's own ref30(t). Skill =
  `1 - MAE(y - ref30 - b L) / MAE(y - ref30)`, best b on a 0-1.2 grid, step 0.05, per era. Eras: pseudo-origins 1915-35,
  1950-62, 1963-75; h = 4-7; every target <= 1982.
  - Centred L (hottest day, first table row) comes from the break-test script. The other rows are uncentred.
  - Kernel comparisons use one common sample (rows valid for every estimator), so their absolute values differ slightly from the
    main table.
- **Combo-like blend gain:** base = `0.5 ref30 + 0.5 combo`, where combo is `0.5 (30-year OLS line at t+h) + 0.5 (mean of last 10)`
  for temperature and CDD and ref30 otherwise. The table reports the gain in skill from adding the best `b * L`.
- **Long offset:** pseudo-origins 1940-1975, R over t-59..t against ref30(t), Gaussian 600 km kernel. `L_long` needs >= 15
  valid years in t-59..t-30. gamma grid -0.4 to 0.8, step 0.1, with b fixed at 0.55. Joint OLS without intercept.
- **EB reliability test:** per city, shrinkage `s = tau^2 / (tau^2 + noise)`. noise = `var(R) * sum(w^2) / (sum w)^2` for the EWMA
  weights, and `tau^2 = mean(L^2) - mean(noise)` per origin. It was compared with flat b on the same rows.
- **Break test:** for each city, the split of the 30-year R series (>= 3 years each side) with the largest two-sample t. If t > 4,
  the level is the mean after the split minus the 30-year mean; otherwise the EWMA level.

These are standalone analyses. The base is the 30-year mean or a combo-like blend, with no W and no learners, so the gain inside
v0.5 will be smaller than the standalone numbers, which is why the expectations in section 4 are lower.
