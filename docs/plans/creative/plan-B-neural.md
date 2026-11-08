# Plan B: a regional state-space core with a gated city-departure learner

**Summary.** Plan B adds a regional Kalman local-level filter to the Placeholder series. One numpy filter per metric runs over each city's neighbourhood (the city plus its 15 nearest cities). Its signal-to-noise ratio comes from a bounded maximum-likelihood fit on years up to each pseudo-origin. For forecasting it runs at a fixed (steady-state) gain of at most 0.39, so no single year can dominate. A drift term phi, chosen strictly chronologically, lets the filter carry the regional warming slope. Each version targets one insights-v1 finding, in this order: the regional core for temperature and CDD (the "bias flips by origin" finding), then city-departure learning (persistent city offsets), then a hottest-day level, then precipitation last. A retrospective, fully nested run of the whole tree on five outer origins 1948-1976 (all targets ≤ 1983) gives the end candidate +2.40 final points against the 30-year average and +0.94 against v0.5 (90% [+0.74, +1.15]). It wins 4 of 5 origins against the 30-year average and 3 of 5 against v0.5. That gain is regime-dependent: 98% of it over v0.5 comes from the 1962 fold. Without 1962 the candidate is at +0.00 against the 30-year average. The robust steps are the assembly fix (v2.0, +0.23) and the hottest-day level (v2.4, +0.50, 4/5 origins). The core is a bet on the warming regime of the development and forecast years, not a demonstrated gain. Every version is official and reads no feature release. Every table is reproducible from `docs/plans/creative/analysis/planB_*.py`.

Status: draft r3

Author: strategy planner B (Kiro, Claude Opus 5.5). r1 and r2 2026-11-08 (Sol: 71, then 85); r3 2026-11-08. Family: neural and probabilistic temporal models trained globally across cities.
Scope: plan and analysis scripts only. No adapter was implemented or scored, nothing was committed, and no development or assessment run was opened. The only sealed-run read was `reward.py score` on v0.5's run (E7).

---

## Changes in r3

| # | Item | How it is handled |
|---|---|---|
| 1 | Diagnostic exclusion must be executable | **No official development ablations.** `reward.py ladder` considers every complete official development run, and `line.py classify` marks a run exploratory only for an unclean commit, an unreleased evaluator, a different suite, an unmatched weather input, or a non-gate-eligible release. None of these is an honest way to tag a diagnostic. So every diagnostic runs in the retrospective driver `planB_e8_nested.py` (≤ 1983, no sealed run, not in the ladder). The only development runs are the versions in §5.3. Before each version, the champion is recorded with `reward.py ladder`, and the run is compared with it explicitly (§4.3). |
| 2 | Assembly contract for v2.2 and later | Per-metric settings `{W mode, combo weight, learn, departure target, gate rule}` (§3.2). Phi selection, lambda scoring and deployment all read the same settings. Each version derives its settings from its parent by a stated change (§5.3). E8 now evaluates the exact v2.2 configuration (core-only for temperature and CDD, everything else as its parent). |
| 3 | Kalman gain guarantee | The forecast filter now runs at a fixed gain `K = K_ss(rho) ≤ 0.390388`, starting from the drift-adjusted window mean (no diffuse start). A city also needs ≥ 3 valid regional values in the last 5 years, else fallback F2. Sol's cases: the second observation's gain is 0.390388 (the exact diffuse filter gives 0.555556). The window with nine early observations, a gap and one final observation triggers F2; with the rule disabled its final gain is 0.390388 (exact filter: 0.849410). The exact diffuse filter is used only inside the likelihood. Tests are in §5.2. |
| 4 | Reconcile E3/E5; deterministic E8 checks | E3/E5 in r1/r2 used an older estimator: an unbounded two-parameter Nelder-Mead fit per origin on [t−99, t] with a 1e4 diffuse start, and an exact windowed filter. Both tables are refreshed with the plan's estimator (`planB_core.py`). Temperature level without drift: +0.024661 (r2 +0.0308; Sol, r2 §3 estimator, +0.023740). Hottest-day half level: +0.012266, 50/62 origins (r2 +0.0143; Sol +0.011888, 48/62). Constants are **not** re-tuned. E8 writes `planB_e8_checks.json` (every rho with status and bound flag, every phi table, every lambda table, unrounded skills, seconds). Full replication stays mandatory before any score (§5.1). |
| 5 | Regime risk | Leave-one-origin-out and a 42-block spatial bootstrap are pre-declared as report-only diagnostics (§4.5). They are computed for every E8 configuration and never select anything. The plan states that against the champion analogue (average30) the main and combo-free candidates both win 4/5 origins; 3/5 against 4/5 is against v0.5. |
| 6 | MLP spec | §3.4 fixes everything: at most 200 epochs, patience 10, preprocessing fitted only on the early-stopping fit partition, exact departure features, target and reconstruction. The departure-target change (v2.3) is separate from adding the MLP (v2.6, deferred). |
| 7 | Anchor schedule | One parameterised schedule: `anchor(t) = S·floor(t/S)`, backoff `anchor − kS` for k = 0, 1, 2, with S ∈ {5, 10}. S may change once, before the first scored version, only if the step-3 benchmark exceeds 120 s per origin, and E8 must then be rerun with `--anchor-step 10` (§3.1). |
| — | Insights-v1 loop rule (brief §6) | Each version names one ranked finding and states its hypothesis first (§5.3). The core is tied to "bias flips by origin / regime sensitivity". Precipitation moves to last. Precipitation and the hottest day have no ranked finding, which is said openly. A version is *kept* only on `promote`; otherwise §5.4's lower-bound-and-reward rule decides what is *carried forward*. `insights.py` runs after every scored version (§8). |
| — | Feature releases | Every v2.x version reads **no feature release**, only `data/city_info.csv`. `features-v4` would be allowed but is not needed: everything is computed from the harness `train` table. Its `*_qc/_fill/_homog/_adj` columns are not causal at inner pseudo-origins and would break the chronology E8 verifies. `features-v5` (SST) appears only as an optional exploratory variant that can never be a parent (§5.3). |
| — | New evidence | E8 now covers the r3 tree. Two results change the plan. (a) The city-departure learner is a no-op in the retrospective folds: the v0.5 gate rule "0 if best ≤ 0" blocks it whenever the core's inner skill is negative. With a relative gate it changes the reward by −0.01. v2.3 therefore gets a pre-score skip rule. (b) Dropping combo (r2's v2.5) costs −0.53 against its parent, and it wins no more origins against the champion analogue, so it is removed from the tree. |

## Changes in r2 (summary)

Chronological gate with its own phi(I) at every gate origin. One assembly function (v2.0). The fully nested retrospective evidence E8. Purged chronology for E6. Bounded rho. Dependency tree. Ablations diagnostic only. Promotion prospects from paired errors. Priorities and the replication stop rule. Corrections: the q → 0 limit, the iid-Gaussian precipitation bound, clipped targets ≠ Huber.

---

## 0. Evidence and how to reproduce it

All scripts live in `docs/plans/creative/analysis/`. Run them from the repository root with `uv run python <script>`. They read only `data/annual_valid.csv` (cut at 1983, asserted) and `data/city_info.csv`. `planB_e8_nested.py` also imports `notebooks/Meet/placeholder_v0_5.py` and the harness functions `build_fold`, `reference` and `score`, unchanged. Each script asserts that no training or design target is later than its stated cutoff.

| Script | Produces | Cutoff asserts |
|---|---|---|
| `planB_core.py` | the core used by all scripts and specified for the adapter: `window` refuses reads after t | read ≤ t |
| `planB_design_tables.py` | E1, E2, E3, E5, rho by anchor (about 4 min) | data ≤ 1983; targets t+h ≤ 1983 |
| `planB_e6_departures.py` | E6 (about 3 min) | training targets ≤ 1965 (MLP fit partition ≤ 1953); test targets ≤ 1983 |
| `planB_e8_nested.py CONFIG` | E8 row errors and `CONFIG-checks.json` (15-50 s per origin per config) | fold targets ≤ the enclosing fit origin; learner targets ≤ F; outer targets ≤ 1983 |
| `planB_e8_boot.py CONFIGS --against X` | E8 bootstrap, spatial blocks, leave-one-origin-out | row years ≤ 1983 |
| `planB_e8_checks.json`, `planB_e8_summary.txt` | the recorded outputs of the two E8 scripts for every configuration | — |

**Evidence status.** E1-E6 informed the design on the same years E8 uses, so E8 is **retrospective**, not independent. Nothing overlaps the development targets (1984-2004).

**E1. Variance shares (1951-1983).** The 15 nearest cities (city excluded) explain 69 / 48 / 52 / 62% of the anomaly variance for temperature / precipitation / hottest day / CDD. The national mean explains 24 / 15 / 14 / 18%.

**E2. Persistence at lags 1, 3, 7 (1931-1983).** City departure from its region: temperature +0.64 / +0.41 / +0.20, CDD +0.53 / +0.36 / +0.18, hottest day +0.23 / +0.14 / +0.09, precipitation +0.12 / +0.09 / +0.02. National hottest-day mean: +0.37 / +0.47 / +0.20.

**E3 (refreshed). Trend methods, pseudo-origins 1915-1976, leads 4-7, skill vs ref30 (origins won of 62).**

| Method | temperature | wins | CDD | wins |
|---|---:|---:|---:|---:|
| v0.3 W (OLS-30 floored, from the window centre) | −0.027219 | 33 | −0.030632 | 25 |
| combo blend | +0.003793 | 35 | −0.009391 | 35 |
| mean of last 10 years | +0.027485 | 39 | +0.012684 | 39 |
| r1 local linear trend, slope floored (old estimator, kept for the row) | +0.014153 | 39 | +0.001698 | 35 |
| core level, no drift, kappa 1 | +0.024661 | 36 | +0.007686 | 32 |
| core level, no drift, kappa 0.5 | +0.017148 | 39 | +0.014164 | 35 |
| core level + drift phi 1, kappa 1 | +0.003977 | 41 | −0.007748 | 39 |
| core level + drift phi 1, kappa 0.5 | +0.022313 | 45 | +0.021573 | 42 |

In the 1915-1945 overshoot analogue, v0.3 W scores −0.049453 for temperature and the no-drift level −0.004321. The fixed constants (kappa_temp 1.0, kappa_cdd 0.5) were set in r1 and are not re-tuned. The refreshed table still favours the no-drift level over v0.3 W in both eras. The best row (drift, kappa 0.5) is not adopted, because adopting it now would be tuning on this table.

**rho by anchor (plan estimator).** Temperature: 0.0000 (1920) → 0.0149 (1935) → 0.0127 (1960) → 0.0210 (1975). CDD: 0.0367-0.0606 from 1935. Hottest day: 0.0185-0.0255 from 1935. Precipitation: 0.0059-0.0321. All are far below the bound 0.25. Anchors before 1905 return "too few series" (F1).

**E5 (refreshed). Precipitation and hottest-day centres (1915-1976).**

| Estimator | precipitation | wins | hottest day | wins |
|---|---:|---:|---:|---:|
| median of 30 years | −0.009443 | 13 | −0.004542 | 28 |
| median of last 10 years | −0.040961 | 3 | +0.007766 | 35 |
| Hodges-Lehmann of 30 years | −0.001392 | 24 | +0.002585 | 44 |
| **half ref30 + half all-history mean (≥ 40 years)** | **+0.003654** | **42** | +0.008654 | 30 |
| core level, kappa 1 | −0.006917 | 17 | +0.015514 | 48 |
| **core level, kappa 0.5** | −0.002032 | 22 | **+0.012266** | **50** |

The precipitation row matches Sol's reproduction exactly (+0.003654, 42/62). For iid Gaussian years, a perfectly known mean would beat the 30-year mean's MAE by at most about 1.6%. That is a reference scale, not a general bound.

**E6. City-departure learners with a purged chronology** (`planB_e6_departures.py`). Training targets ≤ 1965; test origins 1966-1976; leads 4-7. The MLP uses the §3.4 settings and partitions for F = 1965: fit targets ≤ 1953, validation origins 1954-1958, preprocessing from the fit partition only, at most 200 epochs, patience 10. Seeds are 0-4, and only leads 4-7 rows are used. Skill vs predicting 0:

| Metric | Huber | GBM | MLP ensemble (5 seeds; best epochs) | half 15-year mean departure |
|---|---:|---:|---:|---:|
| temperature | +0.042504 | +0.039385 | **+0.047672** (3-5) | +0.017000 |
| precipitation | **+0.002951** | +0.001965 | −0.001432 (7-21) | +0.000983 |
| hottest day | +0.003726 | +0.001048 | +0.000177 (5-8) | **+0.011549** |
| CDD | +0.043053 | +0.035640 | **+0.047956** (5-13) | +0.014076 |

**E7. v0.5's development reward** (read-only): +0.834, 90% [−0.603, +2.308], P 0.81, noise sd 0.890. Origins +0.27 / −2.35 / +4.58; metrics +2.41 / 0.00 / +0.37 / +0.56.

**E8. Retrospective, fully nested run of the r3 tree** (outer origins 1948/1955/1962/1969/1976, 208 scored cities; `reward.py` arithmetic: 2,000 multinomial city draws, seed 20261108). Each configuration is the exact settings of a version in §5.3. All selection is nested and chronological: rho at anchors ≤ t; phi(F) per fit origin from F−21..F−7; lambda from gate origins O−21/−14/−7, each with its own phi(I); everything goes through `assemble()`.

| Config (version) | final pts vs average30 | 90% interval | sd | live | origins won | by origin 1948 / 1955 / 1962 / 1969 / 1976 | metric pts temp / prcp / hot / cdd |
|---|---:|---|---:|---:|---:|---|---|
| v0.5 as is | +1.4647 | [+0.93, +1.97] | 0.32 | +2.49 | 4/5 | +0.06 / +2.12 / +7.40 / +0.18 / −2.44 | +2.37 / 0.00 / +0.32 / +3.17 |
| v2.0 assembly fix | +1.6955 | [+1.19, +2.18] | 0.31 | +2.69 | 4/5 | +0.05 / +2.12 / +7.40 / +0.08 / −1.17 | +2.59 / 0.00 / +0.34 / +3.85 |
| v2.1 core (temp, CDD) | +1.8288 | [+1.29, +2.37] | 0.33 | +2.87 | 2/5 | −0.25 / +2.44 / +10.40 / −0.88 / −2.57 | +3.20 / 0.00 / +0.34 / +3.78 |
| v2.2 core-only (conditional) | +1.2166 | [+0.93, +1.51] | 0.18 | +1.58 | 2/5 | −0.45 / +0.61 / +6.86 / −0.07 / −0.86 | +2.87 / 0.00 / +0.34 / +1.66 |
| v2.3 departure target + relative gate | +1.8185 | [+1.27, +2.36] | 0.33 | +2.90 | 2/5 | −0.25 / +2.44 / +10.35 / −0.88 / −2.57 | +3.16 / 0.00 / +0.34 / +3.78 |
| v2.4 hottest-day level | +2.3224 | [+1.75, +2.88] | 0.35 | +3.21 | 3/5 | +0.04 / +2.59 / +11.95 / −0.07 / −2.89 | +3.16 / 0.00 / +2.35 / +3.78 |
| **v2.5 precipitation base (end of tree)** | **+2.4044** | **[+1.81, +2.97]** | 0.35 | +3.31 | **4/5** | +0.19 / +2.64 / +12.01 / +0.06 / −2.89 | +3.16 / +0.33 / +2.35 / +3.78 |

Script names of these configurations: `v05`, `v20`, `t21`, `t22`, `t23g`, `t24g`, `t25g`. The `v21`-`v25b` and `t23`-`t25` rows in the summary file are r2-order and non-relative-gate diagnostics.

Each step against its parent (paired):

| Step | Δ points | 90% interval | P(better) | origins won |
|---|---:|---|---:|---:|
| v2.0 vs v0.5 | +0.2308 | [+0.16, +0.31] | 1.00 | 1/5 (+1.27 at 1976) |
| v2.1 vs v2.0 | +0.1333 | [−0.01, +0.27] | 0.94 | 2/5 |
| v2.2 vs v2.0 | −0.4789 | [−0.88, −0.07] | 0.03 | 1/5 |
| v2.3 vs v2.1 | −0.0103 | [−0.03, +0.01] | 0.16 | 0/5 |
| v2.4 vs v2.3 | +0.5039 | [+0.41, +0.60] | 1.00 | 4/5 |
| v2.5 vs v2.4 | +0.0819 | [−0.01, +0.16] | 0.94 | 5/5 |
| end (v2.5) vs v0.5 | +0.9397 | [+0.74, +1.15] | 1.00 | 3/5 |
| r2's "drop combo" vs the r2 main candidate | −0.5291 | [−0.86, −0.18] | 0.01 | 3/5 |

*Deterministic checks* (`planB_e8_checks.json`). Outer phi = 0 at every outer origin for v2.1, v2.3 and later, and phi(I) = 1 at gate origins 1934 and 1941. Lambda for CDD = 0 at every outer origin in v2.1-v2.5. Lambda for temperature = 0 everywhere except 0.25 at 1962 from v2.3 on. Hottest-day lambda 0-0.75. rho is at its lower bound only at the anchors 1905-1920. Unrounded totals: v2.0 +1.695493, v2.1 +1.828809, v2.3 +1.8185, v2.4 +2.3224, v2.5 +2.4044. Temperature metric points for v2.1: +3.20045; CDD: +3.77765. These differ from Sol's r2 reproduction (+3.19497 / +3.77573) because r3 changed the forecast filter to a fixed gain.

*Why v2.3 does nothing here.* With the v0.5 rule "lambda = 0 if the best mean inner skill ≤ 0", the learner is blocked whenever the base forecast scores below the 30-year mean at the inner origins. At gate origin 1969 the temperature skill rose from −0.1070 (lambda 0) to −0.1026 (lambda 0.75) but stayed negative, so lambda stayed 0. With the gate measured relative to lambda = 0 (part of v2.3), the learner opens only at 1962, and the reward moves by −0.01.

---

## 1. Strategy and model-selection logic

### 1.1 Choice

`forecast = assemble(settings, base, combo, W, shift, lambda)` (§3.2), with:

1. **A structured probabilistic core:** fixed-gain local-level filtering of the regional anomaly series `c[i, ·]` over `[t−29, t]`. For temperature and CDD it carries a known drift `phi·d_i`, with `d_i` the floored regional OLS-30 slope. The correction is `W_m = kappa_m · (L_t − cbar_i + phi·d_i·h)`.
2. **Gated city-departure learners** (v0.5's Huber + GBM; from v2.3 a departure-from-region target for temperature; from v2.6, optionally, an MLP member).
3. **Simple bases** for the metrics with weak signal: a half-weight regional level for the hottest day, and a longer-memory climatology for precipitation.

### 1.2 Why it suits the data, and what the evidence does and does not show

- **Regional signal and heavy noise (E1, E2).** The predictable part at leads 4-7 is a regional level plus slow city departures. The filter estimates how fast the level moves by likelihood, so there is no window to tune.
- **Regime sensitivity (insights finding 2; lesson 2).** The fixed-gain level reacts to the recent regional level, and the drift enters only when the chronological phi selection finds it helped. In the 1915-1945 analogue the core removes v0.3's overshoot (E3). But E8 shows the core also *widens* the swing between origins: v2.1 wins 2/5 origins against the champion analogue, where v2.0 wins 4/5. Its entire retrospective gain comes from 1962. So the core is a bet that the development and forecast eras reward tracking the regional level. It is kept in the tree because insights and the brief name this finding as the main open lever, and because E8's eras (cooling, then flat) are not the warming era the forecast faces.
- **MAE scoring.** The filter estimates a conditional mean. Median-type alternatives did not help (E5). The learners use MAE-type losses.
- **Small data and reused years (lesson 5).** No learned weights in the core. Outcome-selected quantities: phi (5 values) and lambda only, both chronological. Constants were fixed in r1 and not re-tuned on refreshed tables.

### 1.3 Alternatives

| Candidate | Verdict | Reason |
|---|---|---|
| Neural forecaster as the main model | Rejected | E8: learners add little to the core. The regional level needs 1 parameter. A large network trained on 1930-1980 learns that era's cooling (lesson 1). |
| MLP deep ensemble as a departure member | Deferred (v2.6) | E6: +0.0477 against Huber's +0.0425 on temperature departures. In the pipeline the gate keeps the temperature learner at lambda 0 (E8), so its expected value is small. |
| Local linear trend, stochastic slope | Rejected | MLE slope variance 0. Worse than the level (E3). |
| Bayesian hierarchical model or Gaussian process | Rejected for time and tuning risk | — |
| Distribution-aware centres | Rejected | E5. |
| Dropping combo (r2's v2.5) | Removed | E8: −0.53 against its parent; no more origins won against the champion analogue. |
| Ocean SSTs (`features-v5`/`enhance-v2`) | Optional exploratory variant only | Not gate-eligible, so no gain could become champion. ENSO has no known skill at leads 4-7. Insights list common-year error as the only class fixable by large-scale drivers. |
| PyTorch | Not added | numpy, scipy and sklearn suffice. |

---

## 2. Targets, inputs and time boundaries

### 2.1 Corrections per metric (the W modes)

| Mode | Formula (cities × leads 1-7) | Used by |
|---|---|---|
| `v03` | v0.3's `warming_term` | temperature and CDD in v2.0 |
| `core` | `kappa_m · (L − cbar + phi·d·h)`, kappa temp 1.0, CDD 0.5 | temperature and CDD from v2.1 |
| `hot` | `0.5 · (L − cbar)` (no drift) | hottest day from v2.4 |
| `prcp` | `0.5 · (all-history mean − ref30)` if ≥ 40 valid years ≤ t, else 0 | precipitation from v2.5 |
| `none` | 0 | otherwise |

`base` = `reference(..., "average")` (ref30, with harness fallbacks for unscored cities).

### 2.2 Time boundary for every quantity

Every quantity at pseudo-origin t uses columns ≤ t (`planB_core.window` asserts it): base, c, L, cbar, d, ref30, all-history mean, departure features. rho at t comes from the anchor schedule (§3.1), each anchor fitted on years ≤ the anchor. phi(F) uses inner origins F−21..F−7 (targets ≤ F). Learner rows for fit origin F have targets ≤ F. Lambda at outer O uses gate origins I with their own phi(I) and rows ≤ I, scored on I+1..I+7 ≤ O. Caches are keyed by quantities whose inputs are a fixed cut of the history.

### 2.3 Data releases read

| Version | Release read | Label |
|---|---|---|
| v2.0-v2.6 | none (only `data/city_info.csv`, `static_context`) | official |
| optional SST variant (§5.3) | `features-v5` ocean columns via its `assets-template.json`, or `enhance-v2` via `load_ocean_for_origin` | exploratory, never a parent |

`features-v4` is permitted but not used. Its adjusted columns are not causal at inner pseudo-origins. Its raw columns duplicate what the adapter already computes from `train`.

---

## 3. Training procedure

### 3.1 Core (`planB_core.py` is the reference implementation)

```python
# Fixed in advance
NB_K=15; BASE_YEARS=60; BASE_MIN=20; REG_MIN=4; WINDOW=30; MIN_WINDOW_OBS=10; RECENT_YEARS=5; MIN_RECENT_OBS=3
MLE_YEARS=100; MIN_SERIES_YEARS=30; MIN_SERIES=20; RHO_BOUNDS=(1e-5, 0.25)
ANCHOR_STEP=S in {5, 10} (default 5); anchors(t) = [S*floor(t/S) - k*S for k in (0, 1, 2)]
KAPPA = {temp: 1.0, cdd: 0.5, hottest: 0.5}; PHI_GRID = (0, .25, .5, .75, 1); PHI_TIE = 0.002

rho(metric, t) = first non-None fit_rho(metric, a) over anchors(t)
fit_rho(a):  c = regional series over [a-99, a]; eligible rows: >= 30 valid
             None if < 20 eligible series, or the series is constant, or no convergence / non-finite (F1)
             bounded Brent search on log rho of the exact diffuse concentrated log-likelihood; at_bound flagged
K_ss(rho)  = (-rho + sqrt(rho^2 + 4 rho)) / 2          # <= 0.390388 at rho = 0.25
level(win, K, drift):   L_0 = mean over valid k of (c_k - drift*k)       # drift-adjusted window mean
                        for k: L += drift (k > 0); if valid: L += K (c_k - L)   # every obs weight <= K
                        NaN (F2) if < 10 valid values or < 3 valid in the last 5 years
W = kappa * (L - cbar + phi*d*h), cbar = mean of c over years where city and region are both valid (>= 10, else F2)
```

**Gain guarantee.** The exact diffuse filter (likelihood only) has gains above K_ss early in a series and after gaps: Sol's 0.555556 and 0.849410. The forecast level uses the steady-state gain, so the weight of any one observation is at most 0.390388, wherever it sits in the window. The recent-observation rule turns stale windows into F2 instead of extrapolating from old data. The trade-off is accepted on purpose: the forecast is a steady-state Kalman filter, slightly less efficient during the first few observations. Those observations are at the start of a 30-year window and carry little weight at t.

**Fallbacks.** F1 (whole metric at one t): W = 0. F2 (one city): W = 0. That city's forecast is then `assemble()` with W = 0. It equals the 30-year mean only if lambda = 0 and the combo weight is 0. Any other exception fails the run. Counts go into the artifact.

**Limits (closed forms, K = 0).** Complete window and drift 0: W = 0. Drift phi·d: `W = kappa·phi·d·(14.5 + h)` (Sol's 0.585). Differing masks: `W = kappa·(mean c over its valid years − mean c over both-valid years)`. This is intended: it corrects ref30's regional component for the city's missing years.

**Anchor-step change.** S = 10 is allowed once, before the first scored version, only if the §5.1 benchmark exceeds 120 s per origin. Then E8 must be rerun with `--anchor-step 10`, and the replication target updated. S never changes after a scored version.

### 3.2 The assembly contract

```python
Settings[m] = {"W": mode, "cw": combo weight, "learn": bool, "dep": bool, "gate_rel": bool}
def assemble(settings, base, combo, W, shift, lam):        # the only path to a forecast
    for m in METRICS:
        own  = base[m] + W[m] + lam[m] * shift[m]            # shift = 0 if not settings[m].learn
        pred = own if cw == 0 else (1 - cw) * own + cw * (combo[m] + (W[m] if m in {precip, hottest} else 0))
        clip at 0 for precip and cdd; NaN is an error
```

Callers: `select_phi` (lambda = 0, shift = None), the lambda gate (every lambda in the grid, only for metrics with learn = True; others fixed at 0) and `forecast()`. Phi is scored on the metrics whose W mode is `core`; if there are none, phi = 0. With `gate_rel`, lambda is chosen on the gain over lambda = 0 (smallest lambda within 0.005 of the best gain; 0 if the best gain ≤ 0). Otherwise v0.5's absolute rule applies. Learners are fitted only for learn = True. With `dep`, the target and features of §3.4 replace v0.5's for that metric.

**Inheritance.** Each version's settings are `derive(parent, changes)`, with the changes listed in §5.3, so a child changes exactly the stated fields and nothing else. Example: under the v2.2 branch, v2.4 changes only the hottest day's W mode. The temperature and CDD core-only settings stay. E8's configurations are built the same way (`planB_e8_nested.py`, lines defining `T21`-`T25G`).

### 3.3 Selection order inside a fit origin F

phi(F) on inner origins F−21..F−7, then learners with phi(F), then (outer only) lambda from three gate origins, each refitting with its own phi(I). This is as in r2 and as run in E8.

### 3.4 Learners, the departure target and the MLP

- v0.5 members unchanged: `HuberRegressor(epsilon=1.35, alpha=1.0)`; `HistGradientBoostingRegressor(absolute_error, depth 3, lr 0.05, 150 iterations, min_samples_leaf 500, l2 5.0, early_stopping=False)`. Rows: pseudo-origins F−45..F−1, leads 1-7, targets ≤ F.
- **Departure target (v2.3, temperature).** For pseudo-origin t and lead h: `r_i = y_i(t+h) − ref30_i(t) − W_i(t,h)`; target = `r_i − mean over the 15 nearest j of r_j` (≥ 4 valid), with no median removal. Rows need ≤ 5 missing departures.
- **Departure features**, added to v0.5's features:
  - 15 yearly departures `dep_k = (y_i(t−k) − ref30_i(t)) − mean over the 15 nearest j of (y_j(t−k) − ref30_j(t))` for k = 14..0, with NaN → 0;
  - their 5- and 15-year means;
  - OLS-30 slope minus the neighbours' mean slope;
  - 30-year sd.
- **Reconstruction:** `forecast = base + W + lambda·dep_hat`. W supplies the regional shift, `dep_hat` the city's departure from it.
- **MLP member (v2.6).** `MLPRegressor((32, 16), alpha=1e-2, learning_rate_init=1e-3)`, seeds `42 + k`, k = 0..4, trained with `partial_fit` epochs under `threadpool_limits(1)`.
  - Partitions inside fit origin F: fit rows have targets ≤ F−12; validation rows have origins F−11..F−7 and targets ≤ F.
  - `StandardScaler` and the robust target scale are fitted on the fit partition only. The target is `clip(target / scale, ±3)` with squared loss.
  - At most 200 epochs, patience 10 on validation MAE in original units. The best epoch count n* is then refit from scratch on all rows ≤ F, with the same scaler and scale, for n* epochs.
  - The prediction is the 5-seed mean times the scale. It joins Huber and GBM as an equal-weight member.
- Determinism: bounded Brent search, stable neighbour ties, sorted `city_id`, seed 42. Same-commit reruns must give identical frames.
- Runtime: E8 measured 15-65 s per outer origin per configuration, with up to four concurrent processes. The estimate for development origins is ≤ 120 s. The MLP adds about 150-250 s.

---

## 4. Selection without peeking

### 4.1 Fixed in advance

All constants in §3.1 and §3.4, the v0.5 gate settings, and each version's settings change.

### 4.2 Chosen on inner origins

phi(F) and lambda_m, chronologically (§3.3). rho is estimated, not selected.

### 4.3 Keep rule and champion bookkeeping (brief §5.4 and §6)

Before scoring each version:
1. Record the champion: `uv run python code/evaluation/reward.py ladder`. Write its run id into the version's entry in `docs/models/placeholder.md`.
2. Run the pre-score checks of §5.1.
3. Score the version with `line.py score` (it prints the reward against the current champion), then run `reward.py score <run> --against <champion run>` and `reward.py score <run> --against <carried-forward run>`.

Decision:
1. **Kept** only if the verdict against the recorded champion is `promote`. It then becomes champion and the parent.
2. **Otherwise carried forward** (it becomes the parent of the next version, not the champion) only if its 90% lower bound and its reward, both against the recorded champion, exceed those of the current carried-forward version. The current carried-forward version is `Placeholder_v0.5`: reward +0.834, lower bound −0.603 against `control:average30`. If the champion changed, both are recomputed against the new champion. If the carried-forward version *is* the champion, the rule reduces to "lower bound > 0 and Δ > 0".
3. Otherwise neither. The next version's parent is the current carried-forward version.
4. **No official development ablations** (r3 item 1). Diagnostics run only in the retrospective driver, which creates no sealed run. A run is counted only if `line.py` labels it `official`.
5. `within_noise` is reported as "not progress".

### 4.4 Promotion prospects (E8 against E7)

In the retrospective folds the end candidate would promote against the champion analogue: P 1.00, 4/5 origins, lowest metric +0.33. So would v2.0 alone. The retrospective city noise (sd 0.31-0.35) is about a third of the development noise (0.89). One illustration: v0.5 earns 0.57 of its retrospective points on development (0.834 / 1.465). On that scale the end candidate maps to about +1.37, near the +1.5 that P ≥ 0.95 needs at sd 0.89. **Promotion is possible, not expected.** The 1962 concentration (§4.5) is the main reason for caution.

### 4.5 Report-only sensitivity diagnostics (pre-declared; never used to select)

For every E8 configuration and, after scoring, for every development run:
- **Leave one origin out.** Final points with each origin removed.
- **Spatial-block bootstrap.** 42 blocks of 5° × 5°, 2,000 draws, seed 20261108.
- (Development only) the insights zone table.

E8 values (vs average30; vs v0.5 in brackets):

| Config | drop 1962 | lowest drop elsewhere | spatial-block 90% vs avg30 | spatial-block 90% vs v0.5 |
|---|---:|---:|---|---|
| v0.5 | −0.02 | +1.30 (drop 1955) | [+0.69, +2.16] | — |
| v2.0 | +0.27 (+0.29) | +1.59 | [+0.93, +2.36] | [+0.15, +0.32] |
| v2.1 | −0.31 (−0.29) | +1.68 | [+1.05, +2.55] | [+0.17, +0.55] |
| v2.5 end | +0.00 (+0.02) | +2.34 | [+1.53, +3.18] | [+0.68, +1.17] |

98% of the end candidate's gain over v0.5 is the 1962 fold (Sol computed 98.5% on r2's numbers). Against the champion analogue, both the end candidate and r2's combo-free candidate win 4/5 origins. The 3/5 against 4/5 difference is against v0.5. The core (v2.1) is the step that adds the regime dependence: without 1962 it is −0.58 against v2.0.

---

## 5. Implementation and versions

### 5.1 Order and stop rule (under one day)

1. **v2.0 code (about 2 h):** `placeholder_v2_0.py` with `assemble(settings, ...)` and per-metric settings. Tests in §5.2.
2. **Core (about 3 h):** port `planB_core.py` into the adapter file (single-file rule), plus `select_phi` and the chronological gate.
3. **Replication and benchmark, before any `line.py score`:** a driver `notebooks/Meet/retro_v2.py` calls the adapter's `run_forecast` on the E8 origins with the ≤ 1983 table.
   - **Stop rule:** phi and lambda must equal `planB_e8_checks.json` exactly, rho must match within 1e-4 relative, and final points per origin within 0.01 for v2.0 and v2.1. Otherwise fix the bug before scoring.
   - Time origin 1976 (< 120 s) and run it twice (identical output).
4. Versions in §5.3 order, each with its pre-score check. v2.6 only after the carried-forward candidate is reproducible and under 300 s per development origin.

### 5.2 Tests

- Fixed gain: second observation weight = K_ss (0.390388 at rho = 0.25, not 0.555556).
- Nine early observations + gap + one final observation: F2 by the recent-observation rule; with the rule disabled, the final weight is 0.390388, not 0.849410.
- Long gap (10 missing years, then 3 observations): each weight K_ss; level carried with drift through the gap.
- Initialisation: with K = 0 the level equals the drift-adjusted window mean projected to t (closed form).
- K = 0 closed forms of §3.1 (zero, drift `kappa·phi·d·(14.5 + h)`, differing masks).
- `fit_rho` returns None for a constant series and for fewer than 20 eligible series. Bound hits are flagged.
- Causality: perturb every value after t; W(t), rho(anchor ≤ t) and the phi tables do not change.
- Assembly: `forecast()` returns exactly `assemble()`. The gate and phi selection call the same function, checked with a stub. `derive()` changes only the listed fields.
- Determinism: two runs give identical frames.

### 5.3 Versions and dependency tree

Assets for every version: `notebooks/Meet/placeholder-v0.1-assets.json` (city_info only). Release read: none. Command: `uv run python scripts/line.py score --name Placeholder_v2.<n> --adapter notebooks/Meet/placeholder_v2_<n>.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3`. "Parent" is the carried-forward code base; the champion is recorded separately (§4.3).

```
v0.5 (carried forward; champion control:average30)
 └─ v2.0 assembly fix                             parent v0.5 (infrastructure: becomes the parent regardless; scored and judged)
     └─ v2.1 regional core, temp + CDD            parent v2.0
          ├─ carried/kept → C := v2.1
          └─ not          → v2.2 core-only (immediately)   parent v2.0
                              ├─ carried/kept → C := v2.2
                              └─ not          → C := v2.0, CORE_DROPPED
         └─ v2.3 temperature departure learner    parent C (pre-score skip rule)
             └─ D := v2.3 if carried/kept else C
                 └─ v2.4 hottest-day level        parent D (independent of the core)
                     └─ E := v2.4 if carried/kept else D
                         └─ v2.5 precipitation base   parent E
                             └─ F := v2.5 if carried/kept else E
                                 └─ v2.6 MLP member (deferred, §5.1 step 4)   parent F; skipped if F has no departure learner
optional, exploratory, never a parent: v2.Xs = F + features-v5 SST anomalies as an MLP input (only after v2.6)
```

Each version's hypothesis is written before its run. The finding names are those of `insights.py` on v0.9 (brief §6).

| Version | Settings change from parent | Insight finding targeted | Hypothesis | E8 step (vs parent) |
|---|---|---|---|---|
| v2.0 | the gate scores `assemble()` with the combo blend; per-metric settings | none (correctness) | The gate should judge the forecast it deploys | +0.23 [+0.16, +0.31] |
| v2.1 | temp, CDD: `W = core` | `bias_flips` / `origin_instability` (bias −0.98 / +0.52 / −0.20 F by origin; 1990 −2.11) | A level that follows the region, with chronologically gated drift, removes the sign-flipping level bias that a fixed slope extrapolation creates | +0.13 [−0.01, +0.27], 2/5 |
| v2.2 | temp, CDD: `W = core`, `cw = 0`, `learn = False` (others as v2.0) | same | If the core fails inside the blend, it may work alone | −0.48 [−0.88, −0.07] |
| v2.3 | temperature: `dep = True`, `gate_rel = True` | `city_offsets` (temperature, 32% of squared error, honest +0.98) | City departures persist (E2 lag 1 +0.64), and a departure-from-region learner can recover them once the gate admits gains over lambda 0 | −0.01 [−0.03, +0.01] |
| v2.4 | hottest day: `W = hot` | none ranked for v0.9 (hottest day +0.37); checked against the hottest-day `headroom` and `calibration` sections | A persistent regional hottest-day level (E2 national lag 3 +0.47) is predictable at half weight | +0.50 [+0.41, +0.60], 4/5 |
| v2.5 | precipitation: `W = prcp` | `no_signal` (low priority, run last) | A longer memory reduces the noise in the 30-year mean | +0.08 [−0.01, +0.16], 5/5 |
| v2.6 | temperature: MLP joins the members | `city_offsets` (second attempt) | A nonlinear member captures more of the departure (E6 +0.048 against +0.043) | not run |

**Pre-score skip rule (v2.3, v2.6).** Before scoring, run the adapter's selection step on each development origin's training table only. This uses the inner gate, targets ≤ the origin, and reads no outer targets. If temperature's lambda is 0 at all three origins, the forecast is identical to the parent: record "not scored (gate closed)" and skip, which avoids a redundant ladder entry. E8 predicts lambda 0 at 4 of 5 retrospective origins.

**v2.2 trigger.** It runs only if v2.1 is neither kept nor carried forward. Nothing else triggers it. Under `CORE_DROPPED`, later versions keep v2.0's `v03` W for temperature and CDD.

**Retrospective forecast of the decisions** (only an expectation: development noise differs). v2.0 promotes against the analogue. v2.1 would fail rule 2 against a v2.0 champion (lower bound −0.01). v2.2 fails. v2.3 is likely skipped. v2.4 passes. v2.5's lower bound is −0.01. On development, with the champion still average30 and sd ≈ 0.9, rule 2 compares bounds against average30, so the outcomes can differ.

---

## 6. Lessons and findings addressed

| Lesson / finding | Response |
|---|---|
| 1. Warming learned as cooling | Core with no learned weights. Learners see median-removed or region-removed targets. |
| 2. 1990 overshoot / `bias_flips` | v2.1's fixed-gain level with chronologically gated drift. E3 shows the analogue fixed. E8 shows the core adds regime dependence (honestly reported, §4.5). |
| 3. Zero precipitation skill, near-zero hottest day | v2.4 (+0.50, 4/5 in E8) and v2.5 (+0.08, 5/5), with precipitation last (`no_signal`). |
| `city_offsets` (honest +0.98) | v2.3 and v2.6. E8 shows the gate blocks the learner in the retrospective eras. The skip rule prevents empty runs. |
| `amplitude` CDD (best scaling 0.72) and `control_beats_model` (mean10 on CDD) | Not a separate version. kappa_cdd = 0.5 (fixed in r1) halves the CDD correction. v2.1's CDD points are checked against these two findings after scoring. |
| Northern zones (−0.5 each) | Reported through insights' zone table and the spatial-block diagnostic; not targeted. |
| 5. Reused years | No development ablations, chronological selection, fixed constants, retrospective label. |

---

## 7. Risks, compute, dependencies

| Risk | Sign | Response |
|---|---|---|
| Adapter disagrees with E8 | §5.1 stop rule fails | Fix before scoring |
| Regime dependence (1962) | development gain from one origin | Reported via §4.5; no retuning |
| Core widens origin swings | v2.1 wins < 2/3 development origins | v2.2 then, or `CORE_DROPPED` |
| Gate keeps departure learners closed | skip rule fires | Recorded as "not scored"; the tree continues |
| rho at a bound | flag | Logged; the gain is still ≤ 0.39 |
| Runtime | > 120 s per origin in the benchmark | Anchor step 10 (once, §3.1); v2.6 deferred |

Compute: CPU only. Dependencies: none new (numpy, scipy, scikit-learn, threadpoolctl). No feature release, no new data.

---

## 8. Tracking

Every scored version goes through `line.py score`, then `uv run python scripts/line.py sync`. Tags: `series=Placeholder`, `version=v2.<n>`, `plan=B`, `parent`, `champion_before=<run id>`, `finding=<kind:metric>`, `core_dropped`. After each score, run `uv run python code/evaluation/insights.py <run_id> --json`, and record the targeted finding's honest headroom before and after the version, plus the new top 3 findings.

Metrics, in decision order: verdict and P(better) against the recorded champion; 90% lower bound; reward and live points; points by origin and metric (−2 floor); the §4.5 diagnostics; `runtime_s`. Artifacts in `placeholder_selection.json`: phi tables (outer and every gate origin), rho by anchor with flags, F1/F2 counts, lambdas and their tables, mean W per metric, and settings. Results go to `docs/models/placeholder.md` with run id, commit, parent, champion before, verdict and decision.
