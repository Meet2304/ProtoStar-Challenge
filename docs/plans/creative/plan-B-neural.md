# Plan B: a regional statistical / state-space hybrid (the neural part is unproven and deferred)

**Summary.** Plan B's surviving family is a regional statistical / state-space hybrid anchored on the 30-year average with shrunk corrections. It overlaps plan A's direction. Its neural part (an MLP member) is unproven and deferred, so this is not established neural diversification.

The ladder champion is now plan A's `Placeholder_v1.2` (brief §5b). Plan B's route to promotion before the deadline is therefore one single-change version on top of v1.2:

**`Placeholder_v2.7` = v1.2 + plan B's fixed-gain Kalman regional hottest-day level W_hot**, added after v1.2's `centre()` and frozen at kappa 0.5, with kappa 1 as a declared `--diagnostic` run. v1.2's hottest-day points on development are only +0.14, and W_hot is the signal that v1.2 does not carry.

A pre-1983 walk-forward against a v1.2 base built from plan A's own functions (36 fit origins, 22 triplets, plan A's reward.py-matched bootstrap) gives these results for v2.7 against v1.2:
- **Promotes in 18 of 22 triplets**, with median Δ +0.645 points (mean +0.550), and better at 83% of fit origins.
- It changes only the hottest day, adding +1.73 metric points.
- The weakest triplet, (1940, 1947, 1954), loses −0.40. All four non-promoting triplets start at 1940-1943, right after the extreme-heat 1930s.

v2.7 is promoted only against v1.2; if it does not promote, plan B stops. A precipitation version on top of v2.7 is not declared, because the walk-forward does not support it (median −0.018, promotes 3/22).

The earlier v0.5-based tree (assembly fix, core, core-only, departure learner, MLP) and its retrospective replay (+2.553192 under the revised rescue rule, +2.2600 under r4's) are kept as deferred evidence. Every v2.x configuration is *designed to be official-eligible*: no feature release, only `data/city_info.csv`. All numbers are reproducible from `docs/plans/creative/analysis/planB_*.py`.

Status: draft r6

Author: strategy planner B (Kiro, Claude Opus 5.5). r1-r5 2026-11-08 (Sol: 71, 85, 88, 89, 90; Grok r5: 79); r6 2026-11-08.
Scope: plan and analysis scripts only. No adapter was implemented or scored and nothing was committed. Development numbers read: v0.5's reward (E7) and, from brief §5b, v1.2's published development results.

**Disclosure.** v1.2's development numbers shaped this revision. Its per-metric development points, especially the hottest day's +0.14, motivated choosing the hottest-day term as the single change. Grok, having seen them, suggested kappa 0.5 as primary. Neither W_hot's construction nor its constants were re-tuned: W_hot is r1-r5's estimator unchanged. The precipitation decision came from the pre-1983 walk-forward only.

---

## Changes in r6

| # | Item | How it is handled |
|---|---|---|
| 1 | A version whose parent is v1.2 | `Placeholder_v2.7` = `deploy(ref30 + G + W_hot)` for the hottest day, where G is v1.2's centred graph term with v1.2's fitted coefficients unchanged. W_hot is added after `centre()`, so the median centring does not remove it. kappa 0.5 is frozen as primary; `hot-k1` (kappa 1) is a declared `--diagnostic` run. The new walk-forward `planB_v12_walkforward.py` imports `planA_walkforward.py` (unedited) for v1.2 and the bootstrap, with the same cutoff asserts. Results are in E10. A refit check (v1.2's hottest-day coefficients refitted on base ref30 + W_hot) changes v2.7 by only +0.014, so the no-refit form stays. A second version (precipitation history blend on v2.7) was evaluated and **not declared**: against v2.7 it scores median −0.0175, promotes 3/22. The rule that later versions depend on v2.0's gate repair is deleted. The v0.5-based tree is deferred unless v2.7 promotes and time remains, and even then any version must be re-declared on the champion of that moment with its own walk-forward. |
| 2 | §4.3, §4.4, §5.3 for the current champion | Rewritten. The bar is v1.2: +1.268, 90% [+0.59, +1.94] against average30, 3/3 origins. v2.7 must promote against the champion recorded just before it, otherwise stop; the carried-forward rule is disabled in the v1.2 line. Transfer figures sit beside the retrospective endpoints (§4.4). The disclosure is above. |
| 3 | §4.3.5 frozen, r4 rule selectable | §4.3.5 is now one sentence, frozen before any development score. The r4 rule (rescues only for versions that are neither kept nor carried) can be chosen once, before v2.7 is scored, and the choice is recorded. `hot-k1` against v2.1 in E9 failed only the retrospective 4/5 origin bar: 3/5 wins, with P 1.000, lower bound +0.167, live and floor passing. |
| 4 | v2.6 predicate | One predicate: v2.6 runs only when the carried parent contains the departure target. `gate-only` does not qualify; `dep-absgate` does if §4.3.5 selects it. The `gate_rel` requirement is deleted. Deferred anyway. |
| 5 | Steps paired with their parents | E9 reports each step against its own parent. Precipitation against its parent v2.1r is +0.081938 (90% lower −0.005217, P 0.938, 5/5): it would not promote on its own. Its +0.353772 against the champion v2.1 includes the unpromoted kappa-1 rescue. |
| 6 | `prcp-w75` figures committed | `planB_e8_prcpw75_vs_v21.txt` (written by the new `planB_e8_pair.py`): `prcp-w75` against v2.1, +0.060626, 90% [+0.017973, +0.100684], P 0.987, 5/5. Also `hot-k1` against v2.1 and precipitation against v2.1r. |
| 7 | Title | "a regional statistical / state-space hybrid (the neural part is unproven and deferred)" |
| 8 | Wording | "Designed to be official-eligible" for retrospective configurations. The departure learner shows **no demonstrated benefit** over gate-only: −0.0304, 90% [−0.08, +0.02]. That is not established inferiority. |

Earlier rounds, in brief: **r5** replayed the diagnostic rescues, restored the temperature E3 rows and added the gate-only comparator. **r4** replayed the policy with the supported path first, the frame-level skip rule, the true gain bound 0.391097019 and the actual v0.3 CDD row. **r3** added the per-metric assembly, fixed-gain filter, bounded rho, deterministic checks and analysis scripts. **r2** added the chronological gate, one assembly function, nested E8 and the purged E6.

---

## 0. Evidence and how to reproduce it

All scripts are in `docs/plans/creative/analysis/`. Run them with `uv run python <script>` from the repository root. They read only `data/annual_valid.csv` and `data/city_info.csv`, cut at 1982 (E10, plan A's loader) or 1983 (E1-E9), and assert their cutoffs.

| Script | Produces |
|---|---|
| `planB_v12_walkforward.py` → `planB_v12_walkforward.txt` | **E10** (v2.7 and its diagnostics against v1.2; about 3 min) |
| `planB_core.py`, `planB_gain_check.py` | the core (reference for the adapter's W_hot); the observation-weight bound 0.391097019 |
| `planB_design_tables.py` → `.txt` | E1-E5 |
| `planB_e6_departures.py` | E6 |
| `planB_e8_nested.py`, `planB_e8_boot.py`, `planB_e8_pair.py`, `planB_e8_replay.py` → `planB_e8_replay.txt`, `planB_e8_prcpw75_vs_v21.txt`, `planB_e8_summary.txt`, `planB_e8_checks.json` | E8-E9 (v0.5-based tree, deferred) |

**Evidence status.** All evidence is retrospective (pre-1983 or pre-1984 targets) and was used for design. Nothing overlaps the development targets.

### E10. The v1.2-based version (primary)

The construction follows plan A's walk-forward exactly (`planA_walkforward.py`, imported, not edited):
- `load()` keeps years ≤ 1982 (asserted).
- v1.2 at each fit origin F = 1940..1975: `precompute` (gauss kernel), `fit_coefs(..., "avg30", use_long=True)`, which asserts that every coefficient-fit target ≤ F, then `forecast_at(..., "v1.2")`.
- **v2.7:** v1.2's deployed forecast + `planB_core.core_W(hottest, F, rho, phi=0, kappa=0.5)`. This reads columns ≤ F only (asserted by `window`), with rho from anchors ≤ F.
- Scoring: plan A's `city_sums` and `compare()` (reward.py-matched: seed 20261108 reset per triplet, 2,000 draws, scored cities resampled; promote = P ≥ 0.95, Δ > 0, ≥ 2/3 origin wins, live not clearly worse, every metric ≥ −2).
- Triplets: (F, F+7, F+14), F = 1940..1961, targets ≤ 1982 (asserted).
- No hottest-day city hit fallback F2 at any fit origin. Hottest-day rho ranged 0.0198-0.0255.

Final points against average30, mean over the 36 fit origins:

| Config | total | temp | precip | hottest | CDD |
|---|---:|---:|---:|---:|---:|
| v1.2 | +1.5787 | +2.6408 | +0.3582 | +1.3747 | +1.9412 |
| **v2.7 (kappa 0.5)** | **+2.0102** | +2.6408 | +0.3582 | **+3.1005** | +1.9412 |
| diagnostic `hot-k1` (kappa 1) | +2.1326 | +2.6408 | +0.3582 | +3.5903 | +1.9412 |
| check: refit v1.2's hottest coefficients on ref30 + W_hot | +2.0244 | | | +3.1573 | |
| v2.8 candidate (v2.7 + precipitation blend 0.5), not declared | +2.0346 | | +0.4560 | | |

Promotion analogue over 22 triplets:

| Pair | median Δ | mean Δ | median P | min P (triplet) | median LB90 | min LB90 | ≥ 2 wins | **promote** | worst floor |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| v1.2 vs average30 | +1.8073 | +1.8251 | 1.000 | 0.9765 (1961) | +1.1497 | +0.1104 | 22/22 | 22/22 | −0.034 |
| **v2.7 vs v1.2** | **+0.6450** | **+0.5501** | 1.000 | 0.0000 (1940) | +0.5028 | −0.5477 | 21/22 | **18/22** | −0.579 |
| `hot-k1` vs v1.2 | +0.9656 | +0.7796 | 1.000 | 0.0000 (1940) | +0.7190 | −1.4386 | 21/22 | 16/22 | −3.513 |
| `hot-k1` vs v2.7 | +0.3362 | +0.2296 | 1.000 | 0.0000 (1940) | +0.2085 | −0.9020 | 18/22 | 10/22 | −3.513 |
| check refit vs v2.7 | +0.0096 | +0.0136 | 0.804 | 0.1565 (1947) | −0.0088 | −0.0842 | 12/22 | 2/22 | −0.563 |
| v2.8 vs v2.7 | −0.0175 | −0.0233 | 0.382 | 0.0000 (1950) | −0.1100 | −0.2586 | 6/22 | 3/22 | −0.579 |
| v2.8 with weight 0.25 vs v2.7 | +0.0081 | +0.0079 | 0.619 | 0.0060 (1950) | −0.0373 | −0.1055 | 10/22 | 5/22 | −0.579 |

**v2.7's non-promoting triplets** (Δ; P; origin wins; by origin):

| Triplet | Δ | P | wins | by origin |
|---|---:|---:|---:|---|
| (1940, 1947, 1954) | −0.3988 | 0.000 | 1/3 | −1.12 / +0.18 / −0.25 |
| (1941, 1948, 1955) | −0.2023 | 0.005 | 2/3 | −1.05 / +0.35 / +0.09 |
| (1942, 1949, 1956) | −0.1072 | 0.048 | 2/3 | −0.71 / +0.09 / +0.30 |
| (1943, 1950, 1957) | +0.0295 | 0.680 | 2/3 | −0.67 / +0.19 / +0.56 |

Every failure has a first origin just after the 1930s extreme-heat decade, where a carried regional hottest-day level overstates what follows. `hot-k1` fails the same four triplets more deeply (floor −3.513), plus two more, so kappa 0.5 stays primary.

**Reading.** v2.7 adds a signal v1.2 does not have: v1.2's own walk-forward hottest-day points are +1.37 and v2.7's are +3.10. It is the only change, and it is robust except after an extreme-heat regime. The precipitation blend does not add to v1.2, whose 60-year relative offset already carries long memory for precipitation.

### E1-E9 (v0.5-based tree, deferred; unchanged numbers, condensed)

- **E1, E2.** The regional (15-NN) mean explains 48-69% of anomaly variance. The regional hottest-day level persists (national lag 3: +0.47). City temperature departures persist (lag 1: +0.64).
- **E3** (1915-1976, targets ≤ 1983; every row regenerated by `planB_design_tables.py`). Temperature: v0.3 W −0.027219, combo +0.003793, mean10 +0.027485, core level kappa 1 +0.024661. CDD: actual v0.3 +0.004495, direct-slope proxy −0.030632, core half level +0.014164.
- **E5.** Hottest-day level kappa 0.5: +0.012266 (50/62 origins); kappa 1: +0.015514 (48/62). Precipitation half all-history blend: +0.003654 (42/62).
- **E6** (purged; seeds 42-46). Temperature departures: MLP +0.049030, Huber +0.042504.
- **E7.** v0.5 on development: +0.834, 90% [−0.603, +2.308].
- **E9, the full policy replay on 1948-1976.** Each step is paired with its own parent (`planB_e8_replay.txt`, `planB_e8_prcpw75_vs_v21.txt`):

| Step | Parent | Δ vs parent | 90% lower | P | wins | Decision in the replay |
|---|---|---:|---:|---:|---:|---|
| v2.0 assembly fix | v0.5 | +0.2308 | +0.16 | 1.000 | 1/5 | promote vs average30 (+1.695493) |
| v2.1 hottest level kappa 0.5 | v2.0 | +0.503926 | +0.410 | 1.000 | 4/5 | promote |
| `hot-k1` rescue (v2.1r) | v2.0 (vs v2.1: +0.271834) | +0.7758 | vs v2.1 +0.167 | 1.000 | vs v2.1 3/5 | carried; failed only the 4/5 bar |
| v2.2 precipitation | v2.1r | **+0.081938** | **−0.005** | 0.938 | 5/5 | promoted vs champion v2.1 only because that +0.353772 includes the kappa-1 rescue |
| `prcp-w75` (r4 rule path) | v2.1 | +0.060626 | +0.017973 | 0.987 | 5/5 | would promote under the r4 rule (endpoint +2.2600) |
| v2.3 core | v2.2 | +0.133316 | −0.006 | 0.942 | 2/5 | not carried |
| v2.4 core-only | v2.2 | −0.478913 | −0.882 | 0.027 | 1/5 | not carried; `CORE_DROPPED` |
| v2.5 departure learner | v2.2 | +0.044980 | +0.002 | 0.956 | 1/5 | carried, then displaced by `gate-only` (+0.075393) |
| departure vs gate-only | gate-only | −0.0304 | [−0.08, +0.02] | 0.16 | 1/5 | no demonstrated benefit |

**Retrospective endpoints, with transfer figures.** Revised-policy endpoint `v20+hotk1+prcp`: **+2.553192** against average30 (90% [+1.96, +3.11], 4/5); ×0.57 (v0.5's development/retrospective ratio) gives **about +1.45**; without 1962 it is **+0.67**. The r4-rule endpoint `r21+prcpw25`: **+2.2600**. These are against average30 under v0.5's tree. They are not comparable to v1.2 and are not a promotion route.

---

## 1. Strategy and model-selection logic

1. **Primary (tonight): v2.7 on v1.2.** v1.2 already captures city-minus-neighbour levels and the 60-year relative offset. What it lacks is a regional hottest-day level: its development hottest-day points are +0.14 (disclosed), and its walk-forward points are +1.37 against v2.7's +3.10. Plan B's fixed-gain Kalman level is regional, so it is not removed by v1.2's city-minus-neighbours construction. It is added after `centre()`, so it is not removed by median centring either. One change, one metric.
2. **Not declared:** precipitation on v2.7 (E10: no gain).
3. **Deferred:** the v0.5-based tree (E1-E9). The core and core-only were rejected retrospectively. The departure learner showed no demonstrated benefit over gate-only. The MLP is unproven. None of them has a v1.2 parent or walk-forward evidence.

Alternatives rejected earlier (a neural main model, a local linear trend, Bayesian or GP models, distribution-aware centres, ocean SSTs, PyTorch) stay rejected for the reasons in r5's §1.3.

---

## 2. Inputs and releases

v2.7 reads `data/city_info.csv` (`static_context`) through `notebooks/Meet/placeholder-v0.1-assets.json`, the same manifest as v1.2. It reads no feature release, so it is designed to be official. W_hot at the outer origin O uses columns ≤ O of the harness `train` table: regional series, window, rho from anchors ≤ O (`planB_core`'s estimator, unchanged: fixed gain ≤ 0.390388, total observation weight ≤ 0.391097019; fallbacks F1/F2 give W_hot = 0).

---

## 3. v2.7 specification

```python
# notebooks/Meet/placeholder_v2_7.py = placeholder_v1_2.py (MODEL_NAME "Placeholder_v2.7") + the planB_core functions,
# inlined (single-file rule): regional, window, exact_loglik, fit_rho, RhoCache, steady_gain, fixed_gain_level, core_W.
KAPPA_HOT = 0.5            # frozen; diagnostic hot-k1: KAPPA_HOT = 1.0 via --params
W_hot = core_W(hottest_panel_cut_at_O, O, rho_hot(O), phi=0, kappa=KAPPA_HOT)   # cities x 7 (constant over h)
# in v1.2's assemble(), hottest-day rows only:
out = where(has_ref, deploy(m, value + I + W_hot[city], G[city]), value)       # I = 0 in v1.2; G and its coefficients as v1.2
```

Everything else (other metrics, coefficients, fallbacks, record) is v1.2 byte-for-byte. Runtime: v1.2 plus a few seconds (about 14 bounded rho fits and one filter; measure it in §5.1 step 2).

---

## 4. Selection without peeking

### 4.1 Fixed in advance

kappa_hot = 0.5 (primary), the diagnostic kappa 1.0, every `planB_core` constant, and v1.2 unchanged.

### 4.2 Chosen on data

v1.2's coefficients, by v1.2's own rule. rho by bounded MLE at anchors ≤ O. Nothing is chosen on development outcomes.

### 4.3 Keep rule and bookkeeping (current champion `Placeholder_v1.2`)

Before scoring:
1. Run `uv run python code/evaluation/reward.py ladder` and record the champion run id (expected `20261108T180515Z-c2ea877eff`).
2. Pass the §5.1 checks.
3. Record the §4.3.5 rule choice.

Then score with `uv run python scripts/line.py score --name Placeholder_v2.7 --adapter notebooks/Meet/placeholder_v2_7.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --suite outputs/evaluation/suite-v3`, followed by `reward.py score <run> --against <champion run>`.

Decision rules:
1. **Kept** only on `promote` against the recorded champion. The bar is v1.2's own standing: +1.268, 90% [+0.59, +1.94] against average30, 3/3 origins. v2.7 must be clearly better than that paired.
2. **The carried-forward rule is disabled in the v1.2 line.** If v2.7 (or its rescue, rule 5) does not promote, plan B stops and reports.
3. Only `official` runs count. `within_noise` is "not progress".
4. The deferred v0.5-based tree is not scored. If v2.7 promotes and time remains, any further version must be re-declared on the then-champion with its own walk-forward before scoring.
5. **§4.3.5 (frozen; not to be edited after any development score):** each declared diagnostic is judged by rule 1 against the champion as it stands after its version's decision, and if any would promote, the first such in declaration order is re-scored once without `--diagnostic` as `v2.<n>r`. The r4 alternative judges diagnostics only for versions that did not promote; it may be chosen once before v2.7 is scored.

   For v2.7 the only diagnostic is `hot-k1`:
   - If v2.7 promotes, under the default rule `hot-k1` is rescued only if it also promotes against v2.7. Under the r4 rule it is not considered.
   - If v2.7 does not promote, under either rule `hot-k1` is rescued only if it promotes against v1.2.

   E10 favours kappa 0.5. `hot-k1` against v2.7 promotes in only 10/22 triplets, with a floor of −3.513.

### 4.4 Prospects and transfer

| Quantity | Retrospective | Transfer illustration |
|---|---|---|
| v2.7 vs v1.2 (E10, 22 triplets) | median +0.645, mean +0.550; promotes 18/22 | × 0.70 (v1.2's development +1.268 / walk-forward median +1.807) gives **about +0.39 to +0.45**; × 0.57 (v0.5's ratio) gives about +0.31 to +0.37 |
| v2.7 hottest-day metric | +1.73 over v1.2 (36 fit origins) | about +0.43 headline points before shrink |
| Revised-policy v0.5-tree endpoint (E9) | +2.553192 vs average30 | about +1.45 (×0.57); +0.67 without 1962; r4-rule endpoint +2.2600 |

The development bootstrap for v2.7 against v1.2 pairs two forecasts that differ only in the hottest day, so its noise should be well below v1.2's own (90% width 1.35 against average30). **Promotion is plausible but not assured.** The retrospective failure mode is a carried hottest-day level just after an extreme-heat regime: 4 of 22 triplets.

### 4.5 Report-only diagnostics

After scoring: insights (hottest-day headroom and calibration), leave-one-origin-out, and the 42-block spatial bootstrap. These are never used to select.

---

## 5. Implementation tonight

### 5.1 Steps (about 2-3 hours)

1. **Build** `placeholder_v2_7.py` from `placeholder_v1_2.py`, with the `planB_core` functions inlined and W_hot added in `assemble()` for the hottest day only (§3).
2. **Parity before scoring.** Run the adapter on tables cut at F = 1940, 1960 and 1975. Its hottest-day forecasts must equal `planB_v12_walkforward.py`'s v2.7 forecasts within 1e-9, its other metrics must equal v1.2's exactly, and two runs must be identical. Stop if any check fails.
3. **Tests:**
   - W_hot reads no column after O (perturbation test).
   - F1/F2 give W_hot = 0.
   - Observation weights ≤ 0.391097019.
   - With kappa 0 the frame equals v1.2's.
4. **Score** v2.7, then `hot-k1` with `--diagnostic`. Apply §4.3.
5. **Sync** with `line.py sync`, run `insights.py <run>`, and record everything in `docs/models/placeholder.md`.

### 5.2 Versions

| Version | Parent | Change | Status |
|---|---|---|---|
| **v2.7** | `Placeholder_v1.2` | hottest day: + W_hot (kappa 0.5) after `centre()` | **declared; score tonight** |
| v2.7 `hot-k1` | `Placeholder_v1.2` | same with kappa 1.0 | declared `--diagnostic` |
| v2.8 | v2.7 | precipitation history blend | **not declared** (E10: median −0.0175, promotes 3/22) |
| v2.0-v2.6 | v0.5 | assembly fix, hottest/precipitation bases, core, core-only, departure learner, MLP | **deferred** (E1-E9); the v2.0-dependency rule is deleted |

**v2.6 predicate (deferred).** v2.6 runs only when the carried parent contains the departure target. `gate-only` does not qualify; `dep-absgate` does if §4.3.5 selects it.

---

## 6. Lessons and findings addressed

| Lesson / finding | Response |
|---|---|
| Hottest day near zero (lesson 3; v1.2 development +0.14) | v2.7: retrospective +1.73 hottest-day points over v1.2 |
| `bias_flips` / regime sensitivity | v2.7's retrospective failures are a regime case (just after the 1930s heat). Reported, not tuned |
| Reused years (lesson 5) | Disclosure above; kappa frozen; a single change; no development-tuned constant |
| Precipitation | Not changed: v1.2 already carries its long memory (E10) |
| `city_offsets` | Deferred (no demonstrated benefit of the departure learner over gate-only) |

---

## 7. Risks

| Risk | Response |
|---|---|
| A carried hottest-day level after an extreme regime (E10's 4/22 failures) | Accepted; reported; no fallback tuned now |
| Adapter disagrees with the walk-forward | §5.1 step 2 stops the run |
| `hot-k1` deeper failure mode | Diagnostic only; rescued only on promote (§4.3.5) |
| v2.7 does not promote | Plan B stops and reports (§4.3 rule 2) |

## 8. Tracking

`line.py score`, then `line.py sync`. Tags: `series=Placeholder`, `version=v2.7`, `plan=B`, `parent=Placeholder_v1.2`, `champion_before=<run id>`, `finding=hottest_day:no_signal`, `rescue_rule=<r5|r4>`, `line.role` (diagnostic). Watch, in order: verdict and P(better) against v1.2; 90% lower bound; reward; hottest-day metric points and the −2 floor; origin wins; live points.
