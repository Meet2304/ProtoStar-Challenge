# Plan B: a regional state-space core with a gated city-departure learner

**Summary.** Plan B extends the Placeholder series in the order the chronological evidence supports. The first step fixes the lambda gate so it judges the forecast it deploys (v2.0). Next come a half-weight regional level for the hottest day (v2.1) and a longer-memory precipitation base (v2.2). Only then does the plan test its experimental parts: a regional Kalman local-level core for temperature and CDD (v2.3, with a core-only fallback v2.4), a city-departure learner (v2.5), and a deferred MLP member (v2.6). The core runs one numpy filter per metric over each city's neighbourhood (the city plus its 15 nearest cities). Its signal-to-noise ratio comes from a bounded maximum-likelihood fit on years up to each pseudo-origin, and it runs at a fixed gain. A drift term phi is used only when inner origins before the outer origin show that it helped. I replayed the actual version policy on five retrospective outer origins (1948-1976, all targets ≤ 1983), including rejected versions, the conditional branch and the skip rule. It keeps v2.0 and v2.1 (both promote against the champion analogue), rejects precipitation by a hair (lower bound −0.005), rejects the core and core-only, and carries the departure learner forward without promoting it. The champion endpoint (v2.1) scores +2.20 final points against the 30-year average (90% [+1.67, +2.71], 4/5 origins) and +0.73 against v0.5 (90% [+0.62, +0.86], better at all 5 origins). Without the 1962 fold it keeps +0.52 over v0.5 (lower bound +0.39). Every version is official and reads no feature release. Every number is reproducible from `docs/plans/creative/analysis/planB_*.py`.

Status: draft r4

Author: strategy planner B (Kiro, Claude Opus 5.5). r1-r3 2026-11-08 (Sol: 71, 85, 88); r4 2026-11-08. Family: neural and probabilistic temporal models trained globally across cities.
Scope: plan and analysis scripts only. No adapter was implemented or scored, nothing was committed, and no development or assessment run was opened. The only sealed-run read was `reward.py score` on v0.5's run (E7).

---

## Changes in r4

| # | Item | How it is handled |
|---|---|---|
| 1 | Remove forecast-period knowledge | Every statement about the conditions of the forecast or assessment years is gone. The core's hypothesis now rests only on what each origin's history shows (§1.2): the inner-origin phi table, the persistence of regional levels (E2), and the retrospective analogue (E3). Nothing in the plan refers to realised 2019-2025 or assessment-period conditions. |
| 2 | Supported path first; replay the policy | New order: v2.0 assembly fix → v2.1 hottest-day level → v2.2 precipitation → v2.3 core → v2.4 core-only (conditional) → v2.5 departure learner → v2.6 MLP (deferred). `planB_e8_replay.py` applies §4.3 and the §5.3 tree to the retrospective folds: champion bookkeeping, rejected parents, the conditional v2.4 and the skip check. The endpoints reported are what the policy produces (E9). **v2.0 reconciled:** it is judged by §4.3 like any version. If it is neither kept nor carried forward, plan B stops and reports, because every later version depends on the corrected gate. The phrase "parent regardless" is removed. |
| 3 | Skip rule | A version is skipped only if its prediction frame equals its parent's on the full target grid at all three development origins. Both frames come from the adapter run on each origin's training table only, with no outer target read. Child lambda 0 alone is not enough: the replay shows exactly that case. v2.5's temperature lambda is 0 at 1969 while its parent's is 0.75, so the frames differ, the version is scored, and that difference is where its retrospective gain comes from (E9). |
| 4 | Gain guarantee | The initialisation is unchanged. The bound is restated (§3.1): the innovation gain is K ≤ K_ss(0.25) = 0.390388203, but an observation's total weight is `K(1−K)^a + (1−K)^n/n`. The maximum is the last observation's `K + (1−K)^n/n` ≤ **0.391097019** for n ≥ 10, Sol's value. `planB_gain_check.py` verifies the formula and the bound on 3,000 random windows, and that weights sum to 1. The algorithm did not change, so E3/E5/E8 need no regeneration. The r3 test "second weight = K_ss" is replaced by tests of the total weight. |
| 5 | `--diagnostic` | Pre-declared official development diagnostics, 1-2 per version (§5.4), run with `line.py score --diagnostic`. Rule: a diagnostic that would have been kept or carried forward where its version was not is re-scored once without the flag as candidate `v2.<n>r` (same parent, judged by §4.3) before the tree moves on. |
| 6 | MLP rehearsal | Before v2.6, run a pre-declared E8 configuration that matches deployment: seeds 42-46, the full v0.5 + departure feature set, the W-adjusted departure target, and its parent from the replay. Measure its incremental gain over the parent. v2.6 runs only if that gain's 90% lower bound is > 0 (§5.3). E6 now uses seeds 42-46: MLP temperature +0.049030, CDD +0.047206. |
| 7 | CDD comparator | E3 now has two rows. The **actual** v0.3/v0.5 CDD warming (floored regional temperature rate × floored regional CDD-on-temperature response, from the CDD window's mean valid year) scores +0.004495 (33/62 origins). The old row is relabelled as a **direct regional CDD-slope proxy** (−0.030632). This weakens r1-r3's CDD argument: the actual v0.3 CDD term is better than the core in the 1915-1945 analogue (+0.010117 against −0.001406), though worse over all origins (+0.004495 against +0.014164). Stated in §1.2. |

## Changes in r3 and r2 (summary)

r3: no official development ablations; per-metric assembly contract; fixed-gain forecast filter; anchored, bounded rho; refreshed E3/E5; deterministic E8 checks; leave-one-origin-out and spatial-block diagnostics; full MLP spec; one anchor schedule; insights loop rule; feature-release table.
r2: chronological gate with phi(I) per gate origin; one assembly function; nested retrospective E8; purged E6; dependency tree; replication stop rule.

---

## 0. Evidence and how to reproduce it

Scripts live in `docs/plans/creative/analysis/`. Run each with `uv run python <script>` from the repository root. They read only `data/annual_valid.csv` (cut at 1983, asserted) and `data/city_info.csv`, and each asserts that no training or design target is later than its stated cutoff. E8 also imports `notebooks/Meet/placeholder_v0_5.py` and the harness `build_fold`, `reference` and `score`, unchanged.

| Script | Produces |
|---|---|
| `planB_core.py` | the core (reference implementation for the adapter); `window` refuses reads after t |
| `planB_gain_check.py` | the observation-weight formula and bound (no data read) |
| `planB_design_tables.py` → `planB_design_tables.txt` | E1, E2, E3, E5, rho by anchor |
| `planB_e6_departures.py` | E6 |
| `planB_e8_nested.py CONFIG` | E8 row errors, full prediction frames, `CONFIG-checks.json` (15-60 s per origin) |
| `planB_e8_boot.py CONFIGS --against X` | bootstrap, spatial blocks, leave-one-origin-out |
| `planB_e8_replay.py` → `planB_e8_replay.txt` | E9, the policy replay |
| `planB_e8_checks.json`, `planB_e8_summary.txt` | recorded E8 outputs for all 25 configurations |

**Evidence status.** E1-E6 informed the design on the same years E8 uses, so E8/E9 are **retrospective**, not independent. Nothing overlaps the development targets (1984-2004). A rerun of v2.0 gave byte-identical row files at the three origins compared (1962, 1969, 1976).

**E1. Variance shares (1951-1983).** The 15 nearest cities (city excluded) explain 69 / 48 / 52 / 62% of the anomaly variance for temperature / precipitation / hottest day / CDD. The national mean explains 24 / 15 / 14 / 18%.

**E2. Persistence at lags 1, 3, 7 (1931-1983).**
- City departure from its region: temperature +0.64 / +0.41 / +0.20, CDD +0.53 / +0.36 / +0.18, hottest day +0.23 / +0.14 / +0.09, precipitation +0.12 / +0.09 / +0.02.
- National mean: temperature +0.37 / +0.34 / +0.40, hottest day +0.37 / +0.47 / +0.20.

**E3. Trend methods, pseudo-origins 1915-1976, leads 4-7, skill vs ref30 (origins won of 62).**

| Method | temperature | wins | CDD | wins |
|---|---:|---:|---:|---:|
| v0.3 W, temperature (OLS-30 floored, from the window centre) | −0.027219 | 33 | | |
| **v0.3 W, CDD actual** (temp rate × CDD response) | | | +0.004495 | 33 |
| direct regional CDD-slope proxy (r1-r3's "v0.3 W" row) | | | −0.030632 | 25 |
| combo blend | +0.003793 | 35 | −0.009391 | 35 |
| mean of last 10 years | +0.027485 | 39 | +0.012684 | 39 |
| r1 local linear trend (old estimator) | +0.014153 | 39 | +0.001698 | 35 |
| core level, no drift, kappa 1 | +0.024661 | 36 | +0.007686 | 32 |
| core level, no drift, kappa 0.5 | +0.017148 | 39 | +0.014164 | 35 |
| core level + drift phi 1, kappa 1 | +0.003977 | 41 | −0.007748 | 39 |
| core level + drift phi 1, kappa 0.5 | +0.022313 | 45 | +0.021573 | 42 |

In 1915-1945: temperature v0.3 W −0.049453 against the no-drift level −0.004321; CDD actual v0.3 +0.010117 against the half level −0.001406. The constants (kappa temp 1.0, CDD 0.5) were fixed in r1 and are not re-tuned.

**rho by anchor.** Temperature 0.0000 (1920) → 0.0149 (1935) → 0.0210 (1975). CDD 0.037-0.061, hottest day 0.019-0.026 (from 1935). Precipitation 0.006-0.032. Anchors before 1905 return "too few series" (fallback F1).

**E5. Precipitation and hottest-day centres (1915-1976).** Half ref30 + half all-history mean: precipitation **+0.003654 (42/62)**, hottest day +0.008654 (30/62). Core level at kappa 0.5: hottest day **+0.012266 (50/62)**, precipitation −0.002032 (22/62). Medians and Hodges-Lehmann are worse for precipitation (−0.009443, −0.001392). For iid Gaussian years, a known mean beats the 30-year mean's MAE by at most about 1.6%. That is a reference scale, not a general bound.

**E6. City-departure learners, purged chronology** (training targets ≤ 1965; test origins 1966-1976; leads 4-7). MLP: seeds 42-46, fit targets ≤ 1953, validation origins 1954-1958, preprocessing from the fit partition only, at most 200 epochs, patience 10. Skill vs predicting 0:

| Metric | Huber | GBM | MLP ensemble (best epochs) | half 15-year mean departure |
|---|---:|---:|---:|---:|
| temperature | +0.042504 | +0.039385 | **+0.049030** (2-7) | +0.017000 |
| precipitation | **+0.002951** | +0.001965 | +0.001320 (8-15) | +0.000983 |
| hottest day | +0.003726 | +0.001048 | −0.002107 (6-15) | **+0.011549** |
| CDD | +0.043053 | +0.035640 | **+0.047206** (4-17) | +0.014076 |

**E7. v0.5's development reward** (read-only): +0.834, 90% [−0.603, +2.308], P 0.81, sd 0.890. Origins +0.27 / −2.35 / +4.58.

**E8. Configurations of the r4 tree** (fully nested, chronological; 208 cities; `reward.py` arithmetic, 2,000 city draws, seed 20261108). Script names are in brackets.

| Version (config) | vs average30 | 90% | sd | live | wins | by origin 1948 / 1955 / 1962 / 1969 / 1976 | metric pts temp / prcp / hot / cdd | vs v0.5 (90%), wins |
|---|---:|---|---:|---:|---:|---|---|---|
| v0.5 (`v05`) | +1.4647 | [+0.93, +1.97] | 0.32 | +2.49 | 4/5 | +0.06 / +2.12 / +7.40 / +0.18 / −2.44 | +2.37 / 0 / +0.32 / +3.17 | — |
| v2.0 (`v20`) | +1.6955 | [+1.19, +2.18] | 0.31 | +2.69 | 4/5 | +0.05 / +2.12 / +7.40 / +0.08 / −1.17 | +2.59 / 0 / +0.34 / +3.85 | +0.23 [+0.16, +0.31], 1/5 |
| v2.1 (`r21`) | +2.1994 | [+1.67, +2.71] | 0.32 | +3.00 | 4/5 | +0.33 / +2.27 / +9.00 / +0.89 / −1.50 | +2.59 / 0 / +2.35 / +3.85 | +0.73 [+0.62, +0.86], 5/5 |
| v2.2 (`r22`) | +2.2814 | [+1.73, +2.81] | 0.33 | +3.10 | 4/5 | +0.49 / +2.32 / +9.06 / +1.02 / −1.49 | +2.59 / +0.33 / +2.35 / +3.85 | +0.82 [+0.68, +0.96], 5/5 |
| v2.3 on v2.1 (`r23`) | +2.3327 | [+1.76, +2.89] | 0.35 | +3.18 | 3/5 | +0.04 / +2.59 / +12.00 / −0.07 / −2.89 | +3.20 / 0 / +2.35 / +3.78 | +0.87 [+0.67, +1.06], 2/5 |
| v2.4 on v2.1 (`r24`) | +1.7205 | [+1.39, +2.05] | 0.20 | +1.89 | 3/5 | −0.17 / +0.76 / +8.46 / +0.74 / −1.19 | +2.87 / 0 / +2.35 / +1.66 | +0.26 [−0.18, +0.70], 3/5 |
| v2.5 on v2.1 (`d03`) | +2.2444 | [+1.72, +2.75] | 0.32 | +3.06 | 4/5 | +0.33 / +2.27 / +8.95 / +1.16 / −1.50 | +2.77 / 0 / +2.35 / +3.85 | +0.78 [+0.65, +0.91], 5/5 |

Branch configurations for parents the policy did not reach (`v21`, `t21`, `t22`, `v22`, `v23`, `r23p`, `r24p`) are recorded in `planB_e8_summary.txt`. `r23p` and `r24p` are the core and core-only on parent v2.2.

**E9. Policy replay** (`planB_e8_replay.py`; champion analogue starts at average30, carried-forward version at v0.5; promote needs P ≥ 0.95, ≥ 4/5 origin wins, live not clearly worse, every metric ≥ −2):

| Step | Compared with (champion) | Δ | 90% lower | P | wins | Decision |
|---|---|---:|---:|---:|---:|---|
| v2.0 assembly fix | average30 | +1.6955 | +1.189 | 1.000 | 4/5 | **kept** (promote) |
| v2.1 hottest-day level | v2.0 | +0.5039 | +0.410 | 1.000 | 4/5 | **kept** (promote) |
| v2.2 precipitation base | v2.1 | +0.0819 | −0.005 | 0.938 | 5/5 | not carried (needs lower bound > 0) |
| v2.3 regional core | v2.1 | +0.1333 | −0.006 | 0.942 | 2/5 | not carried |
| v2.4 core-only (conditional, ran) | v2.1 | −0.4789 | −0.882 | 0.027 | 1/5 | not carried; `CORE_DROPPED` |
| v2.5 departure learner (skip check: frames differ) | v2.1 | +0.0450 | +0.002 | 0.956 | 1/5 | **carried forward** (not promoted: 1/5 origins) |
| v2.6 MLP | — | — | — | — | — | not replayed (deferred; §5.3 rehearsal first) |

**Endpoints.** Champion: v2.1 = `r21`. Against average30: +2.1994, 90% [+1.67, +2.71], 4/5. Against v0.5: +0.7347, 90% [+0.62, +0.86], 5/5. Carried-forward parent: v2.5 = `d03`. Against average30: +2.2444, 90% lower +1.720, 4/5. Against v0.5: +0.7797, 90% lower +0.647, 5/5. v2.5's gain over v2.1 (+0.27 at 1969, −0.05 at 1962) comes from its gate closing the parent's temperature learner at 1969 (lambda 0.75 → 0), not from learning city offsets. The replay therefore treats it as a parent, not as progress.

**Robustness of the champion endpoint (report-only, §4.5).** Without 1962: +0.50 against average30 (lower −0.02) and +0.52 against v0.5 (lower +0.39). The lowest other leave-one-out is +2.18 (without 1955). Spatial-block bootstrap (42 blocks of 5° × 5°): against average30 [+1.35, +2.92], against v0.5 [+0.53, +0.91]. r3's fixed endpoint, by contrast, kept only +0.02 over v0.5 without 1962.

**Deterministic checks** (`planB_e8_checks.json`). v2.1's lambdas by outer origin: hottest 0.25 / 0.5 / 0 / 0.25 / 0, temperature 0 except 0.75 at 1969, CDD and precipitation 0. Outer phi is 0 wherever the core is present (0.25 at 1948 for the core-only configurations). rho is at its lower bound only at anchors 1905-1920. Sol reproduced r3's numbers exactly; r4 adds configurations and does not change any r3 configuration.

---

## 1. Strategy and model-selection logic

### 1.1 Choice

`forecast = assemble(settings, base, combo, W, shift, lambda)` (§3.2). The pieces, in order of evidence:
1. **A corrected gate** (v2.0).
2. **Simple regional bases** for the weak-signal metrics: a half-weight fixed-gain regional level for the hottest day, and a longer-memory climatology for precipitation.
3. **Experimental:**
   - the regional local-level core with chronologically gated drift for temperature and CDD;
   - a city-departure learner;
   - an MLP member.

### 1.2 Why it suits the data (justified only from history available at each origin)

- **Regional signal, heavy noise (E1, E2).** The predictable part at leads 4-7 is a regional level plus slow city departures. Regional hottest-day levels persist (national lag 3 +0.47). That is why the hottest-day level comes first. In E8 it is the most consistent step (+0.50, 4/5 origins against its parent).
- **Precipitation (E5).** It is close to white noise. A longer memory reduces the noise in the 30-year mean: +0.0037, positive in both eras. That gain is too small to promote.
- **Core hypothesis (H-core).** At an origin O, where inner origins O−21..O−7 show that following the regional level (with or without drift) beats extrapolating a 30-year slope from the window centre, the core reduces the sign-flipping level bias that insights report. The evidence available at each origin is the inner-origin phi table and the gate's inner skills. The retrospective evidence (E3) is the 1915-1945 analogue for temperature. There, extrapolation from the window centre lost −0.049 and the level lost −0.004.

  The evidence against is just as specific:
  - for CDD, the actual v0.3 term did better in that analogue (+0.010 against −0.001);
  - in E8, adding the core widens the swings between origins (v2.3 wins 2/5 against its parent).

  The core is therefore an experiment placed after the supported steps. Whether its drift switches on at a given origin is decided only by that origin's inner folds.
- **MAE scoring.** The filter estimates a conditional mean. Median-type centres did not help (E5). The learners use MAE-type losses.
- **Small data, reused years (lesson 5).** No learned weights in the core. The only outcome-selected quantities are phi (5 values) and lambda, both chronological. Constants were fixed in r1.

### 1.3 Alternatives

| Candidate | Verdict | Reason |
|---|---|---|
| Neural forecaster as the main model | Rejected | The regional level needs 1 parameter. Learners add little in E8. |
| MLP member | Deferred (v2.6) with a rehearsal gate | E6: +0.049 against Huber's +0.043 on temperature departures. Its pipeline value is unknown until the rehearsal runs. |
| Local linear trend, stochastic slope | Rejected | MLE slope variance 0. Worse than the level (E3). |
| Bayesian hierarchical model or Gaussian process | Rejected for time and tuning risk | — |
| Distribution-aware centres | Rejected | E5 |
| Dropping combo | Removed (r3) | −0.53 against its parent in E8 |
| Ocean SSTs (`features-v5`) | Exploratory only, never a parent | Not gate-eligible |
| PyTorch | Not added | numpy, scipy and sklearn suffice |

---

## 2. Targets, inputs, time boundaries, releases

**W modes** (cities × leads 1-7):

| Mode | Formula |
|---|---|
| `v03` | v0.3's `warming_term` (temperature; CDD via temp rate × response) |
| `core` | `kappa·(L − cbar + phi·d·h)`, kappa temperature 1.0, CDD 0.5 |
| `hot` | `0.5·(L − cbar)` |
| `prcp` | `0.5·(all-history mean − ref30)` if ≥ 40 valid years ≤ t, else 0 |
| `none` | 0 |

`base` = `reference(..., "average")`.

**Time boundaries.** Every quantity at pseudo-origin t uses columns ≤ t (asserted in `planB_core.window`). rho comes from anchors ≤ t. phi(F) comes from inner origins F−21..F−7 (targets ≤ F). Learner rows for F have targets ≤ F. Lambda at O comes from gate origins I, each with its own phi(I), scored on I+1..I+7 ≤ O.

**Releases.** Every v2.x version reads no feature release, only `data/city_info.csv` (`static_context`), and is official. `features-v4` is allowed but unused; its adjusted columns are not causal at inner pseudo-origins. `features-v5` is only for the optional exploratory SST variant, and it can never be a parent.

---

## 3. Training procedure

### 3.1 Core (reference implementation: `planB_core.py`)

```python
NB_K=15; BASE_YEARS=60; BASE_MIN=20; REG_MIN=4; WINDOW=30; MIN_WINDOW_OBS=10; RECENT_YEARS=5; MIN_RECENT_OBS=3
MLE_YEARS=100; MIN_SERIES_YEARS=30; MIN_SERIES=20; RHO_BOUNDS=(1e-5, 0.25)
ANCHOR_STEP S in {5, 10} (default 5); anchors(t) = [S*floor(t/S) - k*S for k in (0, 1, 2)]
rho(metric, t) = first non-None bounded-MLE fit over anchors(t)   # exact diffuse concentrated likelihood; F1 if none
K = K_ss(rho) = (-rho + sqrt(rho^2 + 4 rho)) / 2                  # <= 0.390388203
level: L_0 = drift-adjusted mean of the window's n valid values; each column L += drift (after the first);
       each valid observation L += K (c - L); F2 if n < 10 or < 3 valid in the last 5 years
W = kappa * (L - cbar + phi*d*h), cbar over years where city and region are both valid (>= 10, else F2)
```

**Gain and sensitivity (corrected).** The innovation gain is K at every update. Because the initial level averages all n window observations, observation k's total weight in L_t is `K(1−K)^{a_k} + (1−K)^n/n`, where a_k is the number of valid observations after it. The weights sum to 1. The largest weight belongs to the last observation: `K + (1−K)^n/n`. This increases with K (its derivative is `1 − (1−K)^{n−1} ≥ 0`) and decreases with n. Under the bounds, **no observation's weight exceeds 0.391097019** (K = 0.390388203, n = 10), Sol's value. The exact diffuse filter's larger gains (0.555556 for the second observation; 0.849410 in Sol's gap window) arise only inside the likelihood. Sol's gap window is F2 under the recent-observation rule. The all-window initialisation is kept because it gives the correct limit when rho → 0 (the drift-adjusted window mean, so W = 0 for a complete window with no drift). An initialisation from the first few observations would have over-weighted the early window years.

**Fallbacks.** F1 (metric-wide at t) and F2 (one city): W = 0. Any other exception fails the run.

**Closed forms (K = 0).** Complete window, drift 0: W = 0. With drift: `kappa·phi·d·(14.5 + h)`. Differing masks: `kappa·(mean c over its valid years − mean c over both-valid years)`.

**Anchor step.** S = 10 is allowed once, before the first scored version, only if the §5.1 benchmark exceeds 120 s per origin. In that case E8/E9 are rerun with `--anchor-step 10`.

### 3.2 Assembly contract

```python
Settings[m] = {"W": mode, "cw": combo weight, "learn": bool, "dep": bool, "gate_rel": bool}
assemble(settings, base, combo, W, shift, lam):    # the only path to a forecast
    own  = base + W + lam * shift                   # shift = 0 if not learn
    pred = own if cw == 0 else (1 - cw) * own + cw * (combo + (W if m in {precip, hottest} else 0))
    clip at 0 for precip and cdd; NaN is an error
```

`select_phi` (lambda = 0), the lambda gate (learn metrics only; `gate_rel` chooses on the gain over lambda = 0) and `forecast()` all call it. Each version's settings are `derive(parent, changes)` (§5.3).

### 3.3 Selection order

phi(F) on inner origins F−21..F−7 → learners with phi(F) → (outer only) lambda from gate origins O−21, O−14, O−7, each with its own phi(I).

### 3.4 Learners, the departure target and the MLP

- v0.5 members unchanged: Huber `epsilon=1.35, alpha=1.0`; GBM `absolute_error`, depth 3, lr 0.05, 150 iterations, min leaf 500, l2 5.0. Rows: pseudo-origins F−45..F−1, leads 1-7, targets ≤ F.
- **Departure target (v2.5, temperature).** `r_i = y_i(t+h) − ref30_i(t) − W_i(t,h)`; target = `r_i − mean over the 15 nearest j of r_j` (≥ 4 valid), with no median removal. Rows need ≤ 5 missing departures.
  - Extra features: 15 yearly departures from the regional mean (NaN → 0), their 5- and 15-year means, OLS-30 slope minus the neighbours' mean slope, and the 30-year sd.
  - Reconstruction: `base + W + lambda·dep_hat`.
- **MLP member (v2.6).** `MLPRegressor((32, 16), alpha=1e-2, lr 1e-3)`, seeds 42-46.
  - Inside fit origin F: fit rows have targets ≤ F−12; validation origins F−11..F−7 with targets ≤ F.
  - Scaler and robust target scale come from the fit partition only. Target clipped to ±3, squared loss.
  - At most 200 epochs, patience 10. The best epoch count is refit on all rows ≤ F. The prediction is the 5-seed mean, an equal-weight member. Runs under `threadpool_limits(1)`.
- Determinism and runtime: E8 measured 15-60 s per outer origin (v2.1: 15-26 s). The development estimate is ≤ 120 s, plus 150-250 s for the MLP.

---

## 4. Selection without peeking

### 4.1 Fixed in advance

All constants above, the v0.5 gate settings, each version's settings change, and the diagnostics of §5.4.

### 4.2 Chosen on inner origins

phi(F) and lambda_m, chronologically. rho is estimated, not selected.

### 4.3 Keep rule and champion bookkeeping (brief §5.4, §6, §7)

Before each version: record the champion with `reward.py ladder` and run the pre-score checks (§5.1, plus the skip check for v2.5/v2.6). After scoring: run `reward.py score <run> --against <champion run>` and `--against <carried-forward run>`.

1. **Kept** only on `promote` against the recorded champion. It becomes the champion and the carried-forward version.
2. **Otherwise carried forward** if its 90% lower bound and reward against the recorded champion exceed those of the current carried-forward version, which starts as v0.5 (reward +0.834, lower bound −0.603 against average30). If the carried-forward version is the champion, this means a lower bound > 0 and Δ > 0.
3. **Otherwise** neither. The next version builds on the current carried-forward version.
4. **v2.0:** judged by rules 1-3. If it is neither kept nor carried forward, plan B stops and reports. Every later version depends on the corrected gate, so continuing on v0.5's mismatched gate is not an option.
5. **Diagnostics** run with `--diagnostic` (never promoted; §5.4). A diagnostic that would have been kept or carried forward where its version was not is re-scored once, without the flag, as candidate `v2.<n>r` (same parent, rules 1-3) before the tree moves on.
6. Only `official` runs count. `within_noise` is "not progress".

### 4.4 Promotion prospects

In E9 the champion endpoint (v2.1) promotes against the analogue, at city sd 0.32 against development sd 0.89 for v0.5. One scale illustration: v0.5 earns 0.57 of its retrospective points on development, which would map v2.1 to about +1.25. P ≥ 0.95 at sd 0.89 needs about +1.5. **Promotion on development is possible, not expected.** The step most likely to promote against its parent is v2.1 (paired lower bound +0.41 in E9).

### 4.5 Report-only diagnostics (pre-declared, never select)

Leave-one-origin-out and the 42-block spatial bootstrap, for every E8 configuration and every development run, plus the insights zone table. E9 values for the champion endpoint are given above.

---

## 5. Implementation and versions

### 5.1 Order and stop rule (under one day)

1. v2.0 code with per-metric settings (about 2 h).
2. Hottest-day level and precipitation base (about 1 h; they use `planB_core`).
3. **Replication before any `line.py score`:** `notebooks/Meet/retro_v2.py` runs the adapter on the E8 origins with the ≤ 1983 table.
   - Stop rule: phi and lambda equal to `planB_e8_checks.json`, rho within 1e-4 relative, final points per origin within 0.01, for `v20`, `r21` and `r22`. Otherwise fix the bug before scoring.
   - Time origin 1976 (< 120 s) and run it twice (identical output).
4. Core (about 3 h) after v2.2 is scored. Replicate `r23` and `r24` the same way before scoring v2.3.
5. Departure learner and MLP last (§5.3).

### 5.2 Tests

- Observation weights: the last weight equals `K + (1−K)^n/n` (0.391097019 for n = 10 at K_ss(0.25)); every weight ≤ 0.391097019; weights sum to 1 (as `planB_gain_check.py` does).
- Sol's gap window: F2. Long gap followed by 3 observations: level carried with drift; weights follow the formula.
- K = 0 closed forms (zero; drift `kappa·phi·d·(14.5 + h)`; differing masks).
- `fit_rho` returns None for a constant series and for fewer than 20 eligible series. Bound hits are flagged.
- Causality: perturbing values after t leaves W(t), rho(anchors ≤ t) and the phi tables unchanged.
- Assembly identity: `forecast()` == `assemble()`. The gate, phi selection and deployment call the same function. `derive()` changes only the listed fields.
- Skip check: parent and child frames compared element-wise on the full target grid. A test case in which the child's lambda is 0 and the parent's lambda·shift is non-zero must report "different".
- Determinism: two runs give identical frames.

### 5.3 Versions and dependency tree

Assets: `notebooks/Meet/placeholder-v0.1-assets.json`. Release read: none. Score with `uv run python scripts/line.py score --name Placeholder_v2.<n> --adapter notebooks/Meet/placeholder_v2_<n>.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3`.

```
v0.5 (carried; champion control:average30)
 └─ v2.0 assembly fix                      parent v0.5      → if neither kept nor carried: STOP
     └─ v2.1 hottest-day level             parent v2.0      → P1 := v2.1 if kept/carried else v2.0
         └─ v2.2 precipitation base        parent P1        → P2 := v2.2 if kept/carried else P1
             └─ v2.3 regional core         parent P2        → P3 := v2.3 if kept/carried
                 └─ else v2.4 core-only    parent P2 (runs immediately) → P3 := v2.4 if kept/carried else P2 (CORE_DROPPED)
                     └─ v2.5 departure learner   parent P3 (skip check)  → P4 := v2.5 if kept/carried else P3
                         └─ MLP rehearsal (E8 driver, not scored) → v2.6 MLP member   parent P4, only if the rehearsal's lower bound > 0
optional, exploratory, never a parent: v2.Xs = P4 + features-v5 SST anomalies as an MLP input
```

| Version | Settings change from parent | Insight finding (v0.9) | Hypothesis, stated before the run | E9 decision |
|---|---|---|---|---|
| v2.0 | the gate scores the full `assemble()` | none (correctness) | The gate should judge the forecast it deploys | kept (promote) |
| v2.1 | hottest: `W = hot` | none ranked; checked against the hottest-day headroom and calibration | A persistent regional hottest-day level is predictable at half weight | kept (promote) |
| v2.2 | precipitation: `W = prcp` | `no_signal` (low priority) | A longer memory reduces the noise in the 30-year mean | not carried (lower bound −0.005) |
| v2.3 | temp, CDD: `W = core` | `bias_flips` / `origin_instability` | H-core (§1.2) | not carried |
| v2.4 | temp, CDD: `W = core`, `cw = 0`, `learn = False` | same | The core alone, without the blend | not carried; `CORE_DROPPED` |
| v2.5 | temperature: `dep`, `gate_rel` | `city_offsets` (honest +0.98) | City departures persist (E2 lag 1 +0.64); a departure-from-region learner with a gate on the gain over lambda 0 recovers them | carried forward (not promoted; the gain comes from closing the parent's learner) |
| v2.6 | temperature: MLP joins the members | `city_offsets` | A nonlinear member adds to v2.5 (E6 +0.049 against +0.043) | rehearsal first |

**Skip rule (v2.5, v2.6).** For each development origin, build the parent's and the child's prediction frames from that origin's training table only. Compare them element-wise on the full target grid. Skip, and record "not scored (identical frames)", only if they are equal at all three origins.

**MLP rehearsal (before v2.6).** A new E8 configuration: `derive(P4, annual_avg_temp={"mlp": True})` with seeds 42-46, the full v0.5 + departure feature set and the W-adjusted departure target, nested exactly like E8. Report its paired Δ against P4. v2.6 is scored only if that Δ's 90% lower bound is > 0. It is declared now and has not been run.

### 5.4 Pre-declared development diagnostics (`line.py score --diagnostic`)

| Version | Diagnostics (name: change) |
|---|---|
| v2.1 | `hot-k1`: kappa_hot = 1.0 |
| v2.2 | `prcp-w75`: weight 0.75 on ref30 |
| v2.3 | `core-phi0`: phi fixed 0; `core-phi1`: phi fixed 1 |
| v2.5 | `dep-absgate`: departure target with v0.5's absolute gate |

Each diagnostic has the same parent and settings as its version except the named change. Rule §4.3.5 applies.

---

## 6. Lessons and findings addressed

| Lesson / finding | Response |
|---|---|
| 1. Warming learned as cooling | Core with no learned weights; median- or region-removed targets |
| 2. 1990 overshoot / `bias_flips` | v2.3/v2.4 test H-core under the chronological phi rule. Rejected in E9, and that rejection is reported |
| 3. Zero precipitation skill, near-zero hottest day | v2.1 (+0.50, promotes in E9) and v2.2 (+0.08, not carried) |
| `city_offsets` | v2.5, with a frame-level skip check; v2.6 behind a rehearsal |
| CDD `amplitude` / `control_beats_model` | kappa_cdd 0.5 in the core. E3's corrected row shows the actual v0.3 CDD term is competitive; reported after scoring |
| Northern zones | Insights zone table and spatial blocks (report-only) |
| 5. Reused years | Retrospective label, chronological selection, fixed constants, diagnostics flagged and pre-declared |

---

## 7. Risks, compute, dependencies

| Risk | Response |
|---|---|
| Adapter disagrees with E8 | Stop rule §5.1 |
| The hottest-day gain does not transfer | v2.1 remains judged by §4.3; plan B then continues from v2.0 |
| Small steps fail the strict carried-forward rule (as precipitation does in E9) | Accepted and mechanical; reported, not retried |
| The core widens origin swings | Placed after the supported steps; `CORE_DROPPED` branch defined |
| Departure learner adds only gate noise | E9 shows this; the carried-forward status is reported as such, and v2.6 needs a positive rehearsal |
| Runtime | Anchor step 10 (once); MLP deferred |

Compute: CPU only. Dependencies: none new. No feature release.

---

## 8. Tracking

Every version and diagnostic goes through `line.py score` (diagnostics with `--diagnostic`), then `line.py sync`. Tags: `series=Placeholder`, `version`, `plan=B`, `parent`, `champion_before`, `finding`, `core_dropped`, `line.role` (diagnostic). After each score, run `code/evaluation/insights.py <run_id> --json` and record the targeted finding's honest headroom before and after.

Metrics, in decision order: verdict and P(better) against the recorded champion; 90% lower bound; reward and live points; points by origin and metric (−2 floor); the §4.5 diagnostics; `runtime_s`. Artifacts: phi tables, rho by anchor with flags, F1/F2 counts, lambda tables, mean W, settings, and the skip-check result. Results go to `docs/models/placeholder.md`.
