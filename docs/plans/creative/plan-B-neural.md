# Plan B: a regional state-space core with a gated city-departure learner

**Summary.** Plan B replaces the series' warming term with a regional Kalman local-level filter. One numpy filter per metric runs over each city's neighbourhood (the city plus its 15 nearest cities). It is fitted by maximum likelihood on years up to the origin and extrapolated only over the lead h, never over the roughly 20 years that v0.3's line extrapolation spans. The same filter, shrunk by half, gives the hottest day its first regional signal. Precipitation gets a longer-memory climatology, because the data show it is close to white noise. The existing Huber/boosting learners stay as a gated model of city departures, and a small neural deep ensemble is tested only as an ablation. I chose this from analysis of years up to 1983 only, which overlaps no development target, rather than from novelty. Two findings drove it. Extrapolating a 30-year slope is the worst trend method in both clean eras (skill −0.027 for temperature). The windowed local level is the best (+0.031). A neural learner did not beat a linear robust one on city departures (temperature +0.019 against +0.044). The plan targets a `promote` verdict against `control:average30`. That needs about +1.5 final points, roughly twice v0.5's +0.83, so the versions are ordered so that the gains stack.

Status: draft r1

Author: strategy planner B (Kiro, Claude Opus 5.5), 2026-11-08. Family: neural and probabilistic temporal models trained globally across cities.
Scope: plan only. No model was implemented or scored, and no development or assessment run was opened. The only sealed-run read was `reward.py score` on v0.5's existing run, to get the bound the keep rule needs.

---

## 0. Evidence used, and how it was obtained

All numbers below come from throwaway scripts that read only `data/annual_valid.csv` and `data/city_info.csv`, filtered to `year <= 1983`. Every pseudo-origin `t` was restricted so that its targets `t+4..t+7` are at most 1983. Nothing overlaps the development targets (1984-2004), and nothing under `private/`, origin 2004/2011, or after 1983 was read. Skill is the official arithmetic: `1 - pooled MAE / pooled MAE(30-year mean)`, over city-years with a valid target and a valid 30-year mean (at least 10 valid years). "% origins" is the share of pseudo-origins where the method beat the 30-year mean. The scripts were not committed: the task allowed only this file to be edited. Each computation is described exactly enough to redo it in about 40 lines.

Regional series used throughout (the same construction the adapter will use, section 3.1): `c[i, y]` is the mean over city `i` and its 15 nearest cities (haversine, ties by `city_id`) of `value[j, y] - base[j]`, with at least 4 valid members. `base[j]` is city j's mean over `[t-59, t]`, with at least 20 valid years.

**E1. Where the variance is (years 1951-1983, anomaly vs each city's 1951-1980 mean).** The 15 neighbours, excluding the city itself, explain far more variance than the national mean:

| Metric | sd | national mean explains | 15-NN regional mean explains |
|---|---:|---:|---:|
| annual_avg_temp | 1.23 F | 24% | 69% |
| total_precip_in | 7.29 in | 15% | 48% |
| hottest_day_temp | 3.00 F | 14% | 52% |
| cdd | 185 | 18% | 62% |

So the shared signal is regional, not national. A global model must condition on the neighbourhood.

**E2. Persistence (pooled lag correlations, 1931-1983; anomaly vs the trailing 30-year mean).** Lags 1, 3, 7:

| Metric | anomaly | city departure from region | national mean |
|---|---|---|---|
| annual_avg_temp | +0.36 +0.24 +0.22 | +0.64 +0.41 +0.20 | +0.37 +0.34 +0.40 |
| total_precip_in | +0.08 +0.05 −0.04 | +0.12 +0.09 +0.02 | +0.08 −0.05 −0.11 |
| hottest_day_temp | +0.24 +0.16 +0.07 | +0.23 +0.14 +0.09 | +0.37 +0.47 +0.20 |
| cdd | +0.34 +0.30 +0.15 | +0.53 +0.36 +0.18 | +0.41 +0.47 +0.19 |

Temperature and CDD carry persistent regional levels and persistent city departures (station and urban drift). The hottest day has a persistent regional level but weak city departures. Precipitation is close to white noise at every scale.

**E3. Trend method in two clean eras (annual_avg_temp; pseudo-origins 1915-1976, leads 4-7).** The 1915-1945 era (early-century warming, then the 1940s plateau) is a clean analogue of the 1990 overshoot: a steep slope, then a flat or cool spell.

| Method | 1915-1945 | 1946-1976 | all |
|---|---:|---:|---:|
| v0.3 W: regional OLS-30 slope, floored, from the window centre | −0.0495 | −0.0062 | −0.0272 |
| combo blend (half trend-30, half mean-10) | −0.0321 | +0.0377 | +0.0038 |
| mean of last 10 years | +0.0092 | +0.0448 | +0.0275 |
| Kalman local linear trend (MLE), slope floored at 0 | −0.0040 | +0.0313 | +0.0142 |
| **Kalman local level, windowed (section 3.1), no drift** | +0.0047 | **+0.0555** | **+0.0308** |
| local level plus the floored OLS-30 slope as drift (phi = 1) | −0.0465 | +0.0485 | +0.0024 |

CDD gives the same ranking: v0.3 W −0.0306, half-weight windowed level +0.0214 (65% of origins). The pooled maximum-likelihood fit of a local linear trend put the slope variance at 0 (median of 62 origins, for both temperature and CDD). These data do not support a time-varying slope. The bend has to come from the level, a random walk whose fitted signal-to-noise q/r rises from about 0 (1930) to 0.018 (1960) and 0.032 (1976) for temperature. That gives a Kalman gain of about 0.16, a memory of about 6 years. When q/r is 0 the filter returns the window mean, so the forecast falls back to the 30-year average. That fallback is the safe one.

**E4. Nested test of the drift selection rule (section 4).** Outer pseudo-origins 1950-1976. Inner origins O−21..O−7, phi on the grid {0, .25, .5, .75, 1}, shared between temperature and CDD, with ties within 0.002 going to the smaller phi. The rule chose phi = 0 at every outer origin, with skill temperature +0.0587 and CDD +0.0339 (temperature beat the 30-year mean at 85% of origins). Fixed phi = 1 gave +0.0571 / +0.0286. For 1963-1976, as warming began, phi = 1 was slightly better for temperature (+0.0500 against +0.0445). Before 1983, phi cannot be identified because the floored slope is mostly 0. The rule is safe: it does not turn the drift on without inner evidence. Whether it turns it on in a warming era can only be seen on the development origins.

**E5. Central values for precipitation and the hottest day (pseudo-origins 1915-1976 unless stated).**

| Estimator | precip | % origins | hottest day | % origins |
|---|---:|---:|---:|---:|
| median of 30 years (1940-1976) | −0.0097 | 24% | +0.0082 | 70% |
| median of last 10 years (1940-1976) | −0.0480 | | +0.0147 | |
| Hodges-Lehmann of 30 years | −0.0014 | 39% | +0.0026 | 71% |
| global shape shift: mean30 + kappa·sd30, kappa = pooled median z (−0.05 to −0.09) | −0.0007 | 47% | +0.0007 | 56% |
| **half 30-year mean + half all-history mean (>= 40 valid years)** | **+0.0037** | **68%** | +0.0087 | 48% |
| windowed regional level, full weight | −0.0095 | 26% | +0.0151 | 69% |
| **windowed regional level, half weight** | −0.0030 | 34% | **+0.0143** | **71%** |

Distribution-aware centres (medians, quantile shifts, Hodges-Lehmann) do not help precipitation. The skew is mild (+0.30), and a median from 30 points is noisier than a mean. For a near-white series, the gap to the ceiling comes from the noise in the 30-year mean itself. With iid errors a perfect mean gains at most `1 - 1/sqrt(1 + 1/30) ≈ 1.6%`. A longer memory recovers about a quarter of that, consistently in both eras (+0.0038 and +0.0036). The hottest day does respond to a regional level at half weight. With the windowed filter the adapter will use, it scores −0.0010 in 1915-1945 (48% of origins) and +0.0290 in 1946-1976 (94%). The 100-year filter variant scores +0.0037 and +0.0261.

**E6. City-departure learners (target = `y(t+h) - ref30 - mean over the 15 neighbours of the same quantity`; train on origins 1930-1962, test on 1966-1976; skill relative to predicting 0).** Features: the last 15 yearly departures, their 5- and 15-year means, the city's slope minus its neighbours' slope, the 30-year sd, and h.

| Metric | Huber | GBM (v0.5 settings) | MLP deep ensemble, 5 seeds (32-16, early stopping, clipped target) | half the 15-year mean departure |
|---|---:|---:|---:|---:|
| annual_avg_temp | **+0.0439** | +0.0376 | +0.0188 | +0.0170 |
| total_precip_in | +0.0041 | +0.0030 | +0.0013 | +0.0010 |
| hottest_day_temp | +0.0046 | +0.0057 | −0.0016 | **+0.0115** |
| cdd | **+0.0436** | +0.0378 | +0.0414 | +0.0141 |

City departures are predictable for temperature and CDD (E2), and a linear robust filter captures that best. With 24k strongly correlated rows per metric, the MLP adds variance and no signal.

**E7. Current reward of v0.5** (`reward.py score 20261108T141546Z-d0ac8f8b36`, read-only): +0.834 final points, 90% [−0.603, +2.308], P(better) 0.81, noise sd 0.890. By metric: temperature +2.41, precipitation 0.00, hottest day +0.37, CDD +0.56. By origin: 1983 +0.27, 1990 −2.35, 1997 +4.58. Live +2.90. To reach P ≥ 0.95 at this noise level, the final delta must be about `1.645 × 0.89 ≈ +1.5` points.

---

## 1. Strategy and model-selection logic

### 1.1 Choice

`forecast = base_m + W_m + lambda_m · departure_hat`. Every term comes from one of three parts:

1. **A structured probabilistic core (new): the regional state-space level.** For each metric and city, a local-level Kalman filter runs over the neighbourhood anomaly series `c[i, ·]` in the 30-year window. For temperature and CDD the filter carries a known drift `phi · d_i`, where d_i is the floored regional OLS-30 slope. Its variance ratio comes from a pooled maximum-likelihood fit on up to 100 years before the pseudo-origin. The shared signal is `W_m = kappa_m · (L_t − cbar_i + phi · d_i · h)`. Here `L_t` is the filtered level at the origin and `cbar_i` is the mean of `c` over the city's valid ref30 years, the part of the regional signal already inside ref30.
2. **A better base for precipitation (new):** half the 30-year mean plus half the all-history mean.
3. **The gated city-departure learner (kept from v0.3/v0.5, refined in v2.5):** the Huber and boosting members fit `y − base − W` with the cross-city median removed, scaled by the inner-origin lambda gate.

### 1.2 Why this suits the data

- **210 cities, about 45-130 years, heavy noise (E1, E2).** The predictable part at leads 4-7 is a regional level plus slow city departures. A filter whose signal-to-noise is estimated from that same history handles both without a window choice. Its trend can bend because the level is a random walk. The slope variance is fixed at 0 because the likelihood puts it there (E3), not by choice.
- **Shared non-stationary trend, and lesson 2 (the 1990 overshoot).** v0.3's term `rate × (t + h − mean valid year)` extrapolates the slope over 18.5-21.5 years. With the mean W of +0.64 F at 1990 (an input-side number from the model card), the implied regional rate is about 0.03 F/yr. The core starts from a level estimated at t and adds `phi · d · h`: at most 0.03 × 5.5 ≈ 0.17 F of pure extrapolation, against about 0.6 F. In the clean overshoot analogue (1915-1945) that cuts the temperature loss from −0.050 to +0.005 (E3). This is argued from inputs and clean eras only. 1990 is no longer a clean test of any damping (see 4.4).
- **MAE scoring.** The Kalman filter is Gaussian and estimates a conditional mean. For temperature (skew +0.06) the mean and median coincide. For the skewed metrics, every median-style correction was tested and none helped (E5), so no distribution-aware shift is added. The learners keep their MAE-type losses (Huber, `absolute_error`).
- **Lesson 5 (reused years) and small data.** The core has no learned weights. Per metric and fit origin it estimates 2 variances by likelihood, which never uses lead-4-7 outcomes. The only quantity selected on outcomes is phi (5-point grid, inner origins only), next to the existing lambda gate. The constants `kappa` and the precipitation weight were fixed from 1915-1976 data with targets ≤ 1983 (E3-E5), disjoint from every development target.

### 1.3 Alternatives considered and rejected

| Candidate | Verdict | Reason from the data |
|---|---|---|
| Global neural forecaster (MLP, temporal convolution, N-BEATS-style, attention) as the main model | Rejected as the core. A deep ensemble is kept as one ablation (v2.5) | E6: on clean pseudo-origins a 5-seed MLP ensemble scored below Huber for temperature (+0.019 against +0.044) and hottest day (−0.002 against +0.005), and about equal for CDD. The effective sample is about 40-90 independent years of a regional signal. A flexible learner trained on 1930-1980 learns that era's cooling (lesson 1). Attention and N-BEATS add capacity this signal-to-noise cannot use. |
| Kalman local *linear* trend with stochastic slope | Rejected in favour of local level plus gated drift | MLE slope variance = 0 (E3). With a fixed slope fitted over 100 years, the slope is the long-run rate and the level lags in steady warming, so it was worse than the windowed level (+0.014 against +0.031). |
| Full Bayesian hierarchical state-space (PyMC or Stan, city levels nested in regions) | Rejected for time | The data do not need it: the 2-parameter pooled fit already finds the noise structure. MCMC per inner origin would be slow, and neither package is installed. |
| Gaussian process in time per region | Rejected | Equivalent to a state-space model with a chosen kernel. The kernel length scale would be a new hyperparameter tuned on reused years. The Kalman form gives a causal filtered state at every t in one pass. |
| Distribution-aware precipitation and hottest-day centres (medians, quantiles, Hodges-Lehmann, global shape shift) | Rejected | E5: no gain for precipitation (−0.0007 to −0.0480). For the hottest day, the gains of median estimators coincide with the 1940s-1970s decline in summer maxima, so they are era drift, not shape. |
| Ocean indices from `enhance-v2` (Niño 3.4, North Pacific, North Atlantic SST) | Not in the main line | Not gate-eligible, so any gain could not become champion. ENSO has no known skill at leads 4-7. Decadal ocean memory shows up as persistent regional anomalies, which the regional level already uses. If tried, it is a separate exploratory run, and the only gate-eligible path would be an `enhance-v1`-style certified release of the index (section 7). |
| PyTorch | Not added | Nothing in the plan needs it. The core is about 80 lines of vectorised numpy. The neural ablation fits in seconds with `sklearn.neural_network.MLPRegressor` on CPU. A lock change for a 4 GB laptop GPU is not justified. |

---

## 2. Targets and inputs

**Predicted quantity.** Each metric's value is predicted as `base + W + lambda · departure_hat`:

| Metric | base | W (shared signal) | learner target |
|---|---|---|---|
| annual_avg_temp | ref30 | `1.0 · (L − cbar + phi·d·h)` | `y − ref30 − W`, cross-city median removed per (t, h) |
| cdd | ref30 | `0.5 · (L − cbar + phi·d·h)` | same |
| hottest_day_temp | ref30 | `0.5 · (L − cbar)` (from v2.2) | same |
| total_precip_in | `B_p = 0.5·ref30 + 0.5·mean(all valid years ≤ t)` if there are at least 40 such years, else ref30 (from v2.3) | 0 | `y − B_p`, median removed |

CDD and precipitation are clipped at 0, as now. Lambda scales only `departure_hat`. W is not gated, as in v0.3, but its drift part is selected through phi.

**Inputs and their time bounds.** Every quantity at pseudo-origin t uses columns ≤ t of the table already cut at the fit origin (`tables_at`, which calls `assert_no_post_cutoff`):

- `base[j]`: mean over `[t−59, t]` (at least 20 valid years). `c[i, y]` for `y ∈ [t−29, t]`.
- `d_i`: OLS slope of the metric over `[t−29, t]` (at least 10 valid years), averaged over city and neighbours, floored at 0.
- `q/r` for pseudo-origin t: the MLE at anchor year `a(t) = 5 · floor(t / 5)` (a ≤ t), fitted on series `[a−99, a]`. Anchors are cached per metric within one outer origin. Each anchor uses only data ≤ a, so the cache is valid across all inner fits of that outer origin. This keeps every training row strictly causal. Anchoring at the fit origin would have let each row see the noise ratio of later years.
- The all-history mean for `B_p` uses valid years ≤ t.
- The learner features are v0.5's `OWN_FEATURES + POOLED_FEATURES + h` (unchanged until v2.5). v2.5 adds the 15 most recent departures from the regional mean and their 5- and 15-year means, all ≤ t.
- City coordinates come from the declared static asset (`placeholder-v0.1-assets.json`). No new data asset, release or package is needed for v2.1-v2.5.

---

## 3. Training procedure

### 3.1 Core, pseudo-code (numpy, vectorised over the 210 cities)

Year ranges are inclusive (`t-59:t` means years t−59..t), and every array is cut at t before this code sees it.

```python
# Fixed in advance
NB_K = 15; BASE_YEARS = 60; BASE_MIN = 20; REG_MIN = 4; WINDOW = 30
MLE_YEARS = 100; MLE_MIN_SERIES_YEARS = 30; ANCHOR_STEP = 5
KAPPA = {"annual_avg_temp": 1.0, "cdd": 0.5, "hottest_day_temp": 0.5, "total_precip_in": 0.0}
DRIFT_METRICS = ("annual_avg_temp", "cdd")          # phi applies only here
PHI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0); PHI_TIE = 0.002
P0_LEVEL = 1e4                                         # diffuse start

def regional_series(M, t, start):                      # M: city x year array, columns <= t only
    base = nanmean(M[:, t-59:t], min=BASE_MIN)
    anom = M[:, start:t] - base[:, None]
    return nanmean(anom[self_and_15_nn], axis=members, min=REG_MIN)    # (city, years)

def local_level_filter(c, q, r, drift):                # c: (n, T) with NaN gaps; drift: (n,)
    L = first non-NaN of each row; P = P0_LEVEL; loglik = 0
    for each column k:
        if k > first: L += drift; P += q              # predict (random-walk level with known drift)
        ok = ~isnan(c[:, k]); v = c[:, k] - L; F = P + r; K = P / F
        loglik += sum over ok and (obs count > 2) of -0.5*(log(2*pi*F) + v*v/F)
        L = where(ok, L + K*v, L); P = where(ok, P - K*P, P)
    return L, loglik

def fit_variances(M, a):                               # pooled MLE, composite over all 210 series
    c = regional_series(M, a, start=max(first_year, a-99)); keep rows with >= 30 valid years
    s2 = nanvar(c); x0 = log([0.8*s2, 0.02*s2])
    res = scipy.optimize.minimize(lambda p: -local_level_filter(c, exp(p[1]), exp(p[0]), 0)[1],
                                  x0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    log_r, log_q = clip(res.x, [log(s2)-12, log(s2)-12], [log(s2)+2, log(s2)+2])
    return exp(log_q), exp(log_r)                      # cached by (metric, a)

def core_W(M, t, h, metric, phi, cache):
    q, r = cache[(metric, 5*(t//5))]
    c = regional_series(M, t, start=t-59)
    win = c[:, -WINDOW:]                               # years t-29..t
    d = maximum(0, mean_over_self_and_nn(ols_slope(M[:, t-29:t]))) if metric in DRIFT_METRICS else 0
    p = phi if metric in DRIFT_METRICS else 0.0
    L, _ = local_level_filter(win, q, r, drift=p*d)    # filtered level at t
    cbar = nanmean(where(valid(M[:, t-29:t]), win, nan))   # regional signal already inside ref30
    W = KAPPA[metric] * (L - cbar + p*d*h)
    return nan_to_num(W)                                # no state -> W = 0 -> the 30-year mean
```

### 3.2 Phi selection (per outer origin O, before any learner is fitted)

```python
inner = range(O-21, O-6)                               # 15 inner origins, targets <= O
for phi in PHI_GRID:
    for I in inner:
        fold = build_fold(history, cities, I)          # official validity and ref
        pred = ref30 + core_W(..., phi) for metrics in DRIFT_METRICS, leads 1..7
        skill[phi][I] = mean(score(pred, fold, "final").skill[m] for m in DRIFT_METRICS)
phi_star = min(p for p in PHI_GRID if mean_I skill[p] >= max_p mean_I skill[p] - PHI_TIE)
```

The pseudo-origins of the learners, the lambda gate and the outer fit all use `phi_star`. The lambda gate's inner fits at O−21, O−14 and O−7 therefore reuse a phi chosen on overlapping inner folds. All of these lie at or before O, but the gate's inner skill is slightly optimistic, and this is recorded in the artifact.

### 3.3 Learners, losses, seeds, determinism

- Learners: v0.5 unchanged (`HuberRegressor(epsilon=1.35, alpha=1.0)`; `HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, lr=0.05, 150 iterations, min_samples_leaf=500, l2=5.0)`). Training rows come from pseudo-origins `[F−45, F−h]`, as in v0.5, with targets now relative to `base + W_core`. Regularisation for small data comes from the median removal, the large leaf size, the Huber loss and the shrinkage gate.
- v2.5 neural ablation: `MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=200, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10)`. Five members with `random_state = seed + k`, k = 0..4, averaged (a deep ensemble). The target is divided by its robust scale and clipped to ±3, because sklearn's MLP has only squared loss and clipping makes it Huber-like. The fit runs inside `threadpoolctl.threadpool_limits(1)` so the BLAS reduction order is fixed.
- Determinism: the Nelder-Mead start is computed from the data and the iteration cap is fixed. Kalman sums are elementwise numpy over a fixed city order (sorted `city_id`). Neighbour ties break by `city_id` (stable argsort). Seed 42 is passed through. Rerunning the same commit must give identical `row_errors.csv`. The implementer checks this once, with one origin run twice.

### 3.4 Runtime per outer origin (estimate; budget 600 s, the `evaluate.py` default, below the brief's 900 s)

| Step | Cost |
|---|---:|
| v0.5 as now (sealed run: 85 s for 3 origins) | ≈ 30 s |
| Variance MLE: anchors O−66..O in steps of 5, about 14 per metric × 4 metrics, about 0.6 s each (measured: 62 fits took 22-53 s per metric, including other work) | ≈ 35 s |
| Phi selection: 15 inner origins × 5 phi × 2 metrics filters (≈ 5 ms each) + 75 official scorings | ≈ 15 s |
| W at about 45 pseudo-origins × 4 fits × 4 metrics (filters only) | ≈ 5 s |
| **v2.1-v2.4 total** | **≈ 85 s** |
| v2.5 MLP ablation: 5 members × 4 metrics × 4 fits, about 25k-60k rows, single thread | + 150-250 s |

---

## 4. Selection without peeking

### 4.1 Fixed in advance (from E1-E6 on targets ≤ 1983, or carried from v0.5)

`NB_K=15`, `BASE_YEARS=60` (at least 20 valid years), `REG_MIN=4`, `WINDOW=30`, `MLE_YEARS=100` (series with at least 30 valid years), `ANCHOR_STEP=5`, diffuse prior `1e4`, the Nelder-Mead settings above, `KAPPA` (temperature 1.0, CDD 0.5, hottest day 0.5), the `B_p` weight 0.5 with at least 40 valid years, `PHI_GRID` with tie margin 0.002 toward the smaller phi, the learner hyperparameters and gate of v0.5 (`LAMBDA_GRID`, margin 0.005, inner offsets 21/14/7, `TRAIN_YEARS=45`), and the MLP settings in 3.3.

### 4.2 Chosen on inner origins (always earlier than the outer origin)

- `phi_star` per outer origin (3.2), shared by temperature and CDD.
- `lambda_m` per metric, by v0.5's rule.
- `q, r` per metric and anchor year, by likelihood on years ≤ the anchor. This is estimation, not selection: no lead-4-7 outcome enters it.

Nothing is chosen on the development origins. Ablations are reported and never feed back into a scored version (see 4.4 for the one pre-declared conditional version).

### 4.3 Keep rule (fixed before any run; brief section 5 item 4 and `evaluation/REWARD.md`)

Each version is scored once from a clean commit with `line.py score`, which prints the reward block. Then:

1. **Promote:** if `reward.py` gives the verdict `promote` against the current champion (now `control:average30`): P(better on final) ≥ 0.95, a win on ≥ 2 of 3 origins, live not clearly worse, and no metric below −2 points. The version becomes the champion and the kept version.
2. **Otherwise keep** it over the current kept version (now `Placeholder_v0.5`: reward +0.834, 90% lower bound −0.603) only if **its 90% lower bound is higher AND its reward is higher**, both measured against the same champion. If the champion changes during the sequence, `reward.py score <kept run>` is rerun against the new champion before comparing.
3. Otherwise it is not kept, and the next version builds on the current kept version.
4. A `within_noise` result is reported as "not progress" even when its reward is higher. Origin-1997 final skill is reported as a diagnostic, because the earlier rule used it. It no longer decides anything.

All v2.x versions are `official` (no `features-v1`, no `enhance-v2`). An exploratory run would only be informative. To carry such a gain into a gate-eligible version, the same quantity must be computed inside the adapter from the harness `train` table or from a gate-eligible release (`clean-v1`, `enhance-v1`). Otherwise the gain is not carried.

### 4.4 Honesty notes

- Every development year has been seen many times, including by the authors of this plan through the brief (for example "cool 1992-1997"). 1990 can show whether the core reduces the overshoot, but it cannot validate the core. The constants were set only on data ending in 1983. The only safeguard left is the ladder's P ≥ 0.95 rule and the integrator's one-shot assessment.
- E4 shows that phi cannot be identified before 1983, so whether the drift turns on at 1990 and 1997 is a real open question, not something designed in.

---

## 5. Versions (single change each, built on the last kept version)

Each version is one adapter `notebooks/Meet/placeholder_v2_<n>.py`, starting from a copy of `placeholder_v0_5.py`, with `MODEL_NAME = "Placeholder_v2.<n>"`. Parameters for ablations are passed with `--params <json>`. Assets stay `notebooks/Meet/placeholder-v0.1-assets.json`. Command:
`uv run python scripts/line.py score --name Placeholder_v2.<n> --adapter notebooks/Meet/placeholder_v2_<n>.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3`. Ablations use `--name Placeholder_v2.<n>-<ablation>`.

Expected gains are rough priors from the clean eras (E3-E6), stated in final points vs `average30` (points = 100 × mean over the 4 metrics). The development era warmed more than 1946-1976, so the temperature and CDD numbers are the most uncertain.

### v2.1: regional state-space warming core (temperature and CDD), the main bet

- **Change:** in the v0.3 member of v0.5, replace `warming_term(rate, mean_valid_year, t, h)` with `core_W` (section 3.1), phi chosen per section 3.2, for annual_avg_temp and CDD. The CDD core uses CDD's own regional series and slope (as tested in E3), which replaces the temperature-rate × response path. The learners refit on `y − ref30 − W_core`. Combo half, gate, hottest day and precipitation are unchanged.
- **Why:** lessons 1 and 2. The core removes the 20-year slope extrapolation and lets the level bend.
- **Expected:** half weight, because combo still holds 50%. Temperature final points between about +2.4 (v0.5) and about +4. 1990 temperature loss reduced, 1983 roughly level. Headline about +1.0 to +1.4. Probably `within_noise` on its own.
- **Ablations (4):** `phi=0` (fixed), `phi=1` (fixed), `combo_weight=0` (core plus gated departures only), `core_only` (`ref30 + W_core`, no learners and no combo: the pure state-space model).

### v2.2: regional level for the hottest day

- **Change:** `W_hot = 0.5 · (L − cbar)` (phi = 0), added in both halves (v0.3 member base and combo's hottest-day forecast, which is ref30). Learners refit on `y − ref30 − W_hot`.
- **Why:** lesson 3. E2 shows a persistent national and regional hottest-day level (lag 3 national +0.47). E5: +0.0143 at 71% of origins. That gain comes from 1946-1976 (+0.0290); 1915-1945 is level (−0.0010), so expect less in an era without a hottest-day drift.
- **Expected:** hottest day from +0.37 to about +1.0 to +1.5 points. Headline +0.15 to +0.3.
- **Ablations (2):** `kappa_hot=1.0`; `W_hot` in the v0.3 member only.

### v2.3: pooled-history precipitation base

- **Change:** `B_p` replaces ref30 for precipitation everywhere: the base, the learners' target, and combo's precipitation.
- **Why:** lesson 3. Precipitation is near white noise (E2), so the only reachable error is the sampling noise of the 30-year mean (ceiling about 1.6%). E5: +0.0037, 68% of origins, the same in both eras. The gain is spread across cities, which helps the bootstrap's lower bound more than its size suggests.
- **Expected:** precipitation from 0.00 to about +0.3 points. Headline +0.05 to +0.1.
- **Ablations (2):** the long mean over the last 60 years instead of all history; weight 0.75 on ref30.

### v2.4: drop the combo half

- **Change:** `COMBO_WEIGHT = 0`, so `forecast = base + W + lambda · departure_hat`.
- **Why:** combo's temperature and CDD half is `0.5 · trend30 + 0.5 · mean10`. That trend30 is the same window-centre extrapolation that overshoots (E3: combo blend −0.032 in the 1915-1945 analogue), and combo was chosen on years that overlap development (lesson 5). Once the core carries the regional level, combo dilutes it. v2.2 and v2.3 already apply in both halves, so this changes only temperature and CDD.
- **Expected:** temperature and CDD cores at full weight. Using the clean-era priors (temperature about +3, CDD about +2), headline about +1.4 to +2.2 when stacked on v2.1-v2.3. This is the version most likely to reach `promote`.
- **Ablations (2):** `COMBO_WEIGHT=0.25`; `kappa_cdd=1.0`.

### v2.5: city-departure learner as a temporal filter (with the neural ablation)

- **Change:** the learner target becomes the city's departure from its region, `y − base − W − mean over the 15 neighbours of the same`. Feature list adds the last 15 regional departures and their 5- and 15-year means (E6 features). The Huber and GBM members are kept, and lambda is gated as before.
- **Why:** E2 and E6. City departures persist for temperature and CDD (lag 1 +0.64 / +0.53), and a robust linear filter on that history explains +4.4% of departure error on clean data. The new target stops the learner from relearning the regional level that the core already provides.
- **Expected:** +0.1 to +0.4 headline points, mostly temperature and CDD.
- **Ablations (3):** `members=mlp_ens5` (the 5-seed MLP deep ensemble replaces GBM); `members=ensemble+mlp` (MLP as a third member); `target=median_removed` (v0.5 target with the new features). The MLP is expected to lose (E6). If it is kept anyway, that is the only route by which a neural component enters the candidate.

### v2.6: conditional fallback, declared now

- **Only if v2.1 is not kept AND v2.1's `core_only` ablation has a reward and lower bound both above v0.5's:** v2.6 = v0.5 with its forecast for temperature and CDD replaced by `ref30 + W_core` (the pure state-space core, phi as in 3.2). v2.2-v2.5 then build on whichever version is kept. If the condition fails, v2.6 is not run.

Planned order: v2.1 → v2.2 → v2.3 → v2.4 → v2.5 (→ v2.6 only when its condition holds). That is at most 6 versions and 13 ablations, about 19 runs of about 5 minutes each (v2.5's MLP ablations about 12 minutes).

---

## 6. How each lesson is addressed

| Lesson | Response |
|---|---|
| 1. Shared warming learned as mid-century cooling | The shared signal comes from a filter with no learned weights. Learners still see median-removed targets, and from v2.5 targets with the regional mean removed, so they cannot learn an era-wide or regional shift. |
| 2. 1990 overshoot | Pure extrapolation shrinks from `slope × ~20 years` to `phi·d × h` (≤ 7 years). The level bends with the data. Phi is selected on inner origins, and for 1990 those include the 1980s. Clean analogue: −0.050 → +0.005 (E3). 1990 is reported per metric in every run. |
| 3. Zero precipitation skill, near-zero hottest day | Precipitation: a longer-memory climatology aimed at the analytic ceiling (v2.3), positive in both clean eras. Hottest day: a half-weight regional level (v2.2), positive overall but level in 1915-1945. |
| 4. Noise dominates; gains of 0.005 matter | Few parameters. Gains are stacked across metrics and versions toward the about +1.5 points the verdict needs. |
| 5. Reused development years | Constants fixed on data ending in 1983. One 5-point selection per origin. Ablations never feed back. The keep rule is the ladder's bootstrap, which only clear wins pass. |
| 6. Nothing beats average30 under the new reward | The target is `promote`. v2.4 is planned as the version most likely to promote, and the expected-gain section states that v2.1 alone probably will not. |

---

## 7. Risks, fallbacks, compute and dependencies

| Risk | Sign | Fallback |
|---|---|---|
| The core removes too much warming at 1997 (phi chosen as 0, or a lagging level) | 1997 temperature points fall below v0.5's | The `phi=1` ablation shows whether the drift was needed. A different phi rule would be a new, declared version, not a re-tune of v2.1 |
| The drift turns on at 1990 and still overshoots | 1990 temperature points ≤ v0.5's | Report it. `kappa` and the phi grid stay fixed: changing them now would be fitting to 1990 |
| MLE degenerates (q → 0 or q ≫ r) | `q/r` at a bound in the artifact | q → 0 gives W = 0 (the reference), so it is safe. q ≫ r makes the level equal the last year; the upper clip at `log(s2)+2` caps it. Log both |
| Station changes cause jumps in neighbour series | a city's W out of line with its neighbours | 15-neighbour averaging dilutes a single jump. The `_qc` series from `clean-v1` could be a later, separate version |
| Gains stay `within_noise` | P(better) < 0.95 | v2.4 is the planned stacking point. If even v2.4 is within noise, report the best lower bound and hand over without claiming progress |
| Runtime | > 300 s per origin | Raise `ANCHOR_STEP` to 10 (halves the MLE cost). The MLP ablations are optional |
| Neural ablation not deterministic | rerun differs | `threadpool_limits(1)`, fixed `random_state`. If it still differs, drop the ablation and record why |

**Compute:** CPU only, about 85 s per origin (section 3.4). The GPU is not used.
**Dependencies:** none new. numpy, scipy (`optimize.minimize`), scikit-learn (already pinned), and `threadpoolctl`, which comes with scikit-learn. No new data release, no `features-v1`, no `enhance-v2`. Assets are unchanged (`data/city_info.csv` as `static_context`).
**Implementation effort:** about 4-6 hours for v2.1 (core, cache, phi selection, artifact, a unit test that `core_W` at t reads no column > t, and a test that q → 0 gives W = 0). About 1 hour each for v2.2-v2.5.

---

## 8. Tracking

Every scored run (versions and ablations) goes through `line.py score`, then `uv run python scripts/line.py sync` (MLflow is unreachable in this checkout today, so sync once credentials are set). Tags: `series=Placeholder`, `version=v2.<n>`, `plan=B`, `ablation=<name or none>`, `built_on=<run id>`.

Metrics to watch, in decision order:

1. **Verdict** vs champion, and **P(better on final)**.
2. **90% interval lower bound** of the final delta (v0.5: −0.603).
3. **Reward** (final points; v0.5 +0.834), and live points (v0.5 +2.899, must not be clearly worse).
4. Final points **by origin** (1983 / 1990 / 1997; v0.5 +0.27 / −2.35 / +4.58) and **by metric** (v0.5 +2.41 / 0.00 / +0.37 / +0.56), with the −2-point floor.
5. Diagnostics written to `placeholder_selection.json` and logged as artifacts: `phi_star` per origin with the 5 × 15 inner skill table; `q/r` per metric and anchor; mean `W_core` per metric at the outer origin; the share of cities with W = 0; lambdas; `runtime_s`; origin-1997 final skill, as the former keep-rule diagnostic.

Each result is written to `docs/models/placeholder.md` with run id, commit, verdict, P(better), lower bound, reward, and the keep decision under rule 4.3.
