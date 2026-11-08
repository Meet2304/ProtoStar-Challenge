# Plan B: a regional state-space core with a gated city-departure learner

**Summary.** **What survives is not a neural model.** Plan B's surviving path is a regional statistical / state-space hybrid anchored on the 30-year average, with shrunk corrections. That overlaps plan A's direction. Plan B's neural part (the MLP member) is unproven and does not run on the replayed path, so this plan is not established neural diversification. The version order follows the chronological evidence:
1. v2.0 fixes the lambda gate so it judges the forecast it deploys.
2. v2.1 adds a fixed-gain Kalman regional level for the hottest day.
3. v2.2 adds a longer-memory precipitation base.
4. Then the experiments: a regional local-level core with gated drift for temperature and CDD (v2.3, with a core-only fallback v2.4), a city-departure learner (v2.5) and an MLP member (v2.6).

Each version has 1-2 pre-declared diagnostics. I replayed the whole policy on five retrospective outer origins (1948-1976, all targets ≤ 1983): rejected versions, the conditional branch, the frame-level skip rule, and the diagnostic rescues with their effect on later parents (E9).
- **Kept:** v2.0 and v2.1 promote against the champion analogue.
- **Rescued:** the hottest day's `hot-k1` diagnostic (kappa 1) is carried forward, and v2.2 then promotes on top of it.
- **Rejected:** the core and core-only (`CORE_DROPPED`).
- **v2.5:** its `gate-only` diagnostic beats the departure learner, so the departure target is not carried and v2.6 does not run.

The **champion endpoint** is v2.2 on the rescued parent: +2.5532 final points against the 30-year average (90% [+1.96, +3.11], 4/5 origins) and +1.0885 against v0.5 (90% [+0.87, +1.32], better at all 5 origins). Without the 1962 fold it keeps +0.69 over v0.5 (lower bound +0.47). Every version is official, reads no feature release, and is reproducible from `docs/plans/creative/analysis/planB_*.py`.

Status: draft r5

Author: strategy planner B (Kiro, Claude Opus 5.5). r1-r4 2026-11-08 (Sol: 71, 85, 88, 89); r5 2026-11-08. Family: neural and probabilistic temporal models trained globally across cities.
Scope: plan and analysis scripts only. No adapter was implemented or scored, nothing was committed, and no development or assessment run was opened. The only sealed-run read was `reward.py score` on v0.5's run (E7).

---

## Changes in r5

| # | Item | How it is handled |
|---|---|---|
| 1 | Replay the diagnostic rescues | `planB_e8_replay.py` now runs every declared diagnostic and applies §4.3.5. The parents that follow, the configurations built on them (computed on demand) and the rehearsal ancestry all come from the replay. **Rule (made precise in r5):** after each version's decision, its diagnostics are judged against the state *after* that decision, by rules 1-2. If several qualify, the order is: one that would promote first, then the higher 90% lower bound against the champion, then the higher reward, then declaration order. At most one rescue per version. The rescue is re-scored without the flag as `v2.<n>r`. **Sol's counterexample is reproduced exactly:** `prcp-w75` on v2.1 gives +0.060626, lower bound +0.017973, P 0.987, 5/5, which under r4's wording would promote to a +2.2600 endpoint. Under the r5 rule the path changes earlier: `hot-k1` beats the promoted v2.1 (+0.271834 against v2.1, lower bound +0.166987, 3/5) and is carried forward. v2.2 on that parent then promotes, and `prcp-w75` no longer qualifies. **Corrected endpoint:** champion v2.2 at +2.553192 against average30 and +1.088468 against v0.5. *Refinement note:* r4's wording ignored diagnostics of versions that promoted. I changed that after this retrospective replay showed `hot-k1` beating a promoted v2.1, to match brief §7 ("if an ablation turns out to be the better model, re-score it"). It is a design choice made on data ≤ 1983. The r4-wording endpoint (+2.2600) is also reported. |
| 2 | Temperature E3 rows | Fixed in `planB_design_tables.py`: the combo, local-linear-trend and core rows had been placed under the CDD branch in r4. The regenerated `planB_design_tables.txt` prints every temperature row the plan quotes, with values identical to the plan's table. |
| 3 | Gate-only comparator for v2.5 | New diagnostic `gate-only`: v2.5's relative gate without the departure target. On the replayed parent: gate-only +0.075393 (lower bound +0.030224, 2/5); the departure learner +0.044980 (lower bound +0.002302, 1/5); `dep-absgate` +0.054366. Departure learning against gate-only: −0.0304 (90% [−0.08, +0.02]). **The v2.5 "gain" is the gate change, not departure learning.** The replay rescues gate-only as v2.5r. |
| 4 | v2.6 when v2.5 is not carried | v2.6 runs only if the carried-forward P4 contains the departure learner, i.e. v2.5 itself was kept or carried. If v2.5 is rejected, skipped, or replaced by a rescue without the departure target, v2.6 is **not run**: it would bundle two unproven changes. In E9, P4 = v2.5r (gate-only), so v2.6 is not run. When it does run, its rehearsal configuration is `derive(P4, annual_avg_temp={"mlp": True})`, and P4 then already holds `dep` and `gate_rel`. |
| — | Summary | States that the surviving path is a regional statistical / state-space hybrid anchored on average30 with shrinkage, that it overlaps plan A, and that the neural part is unproven. |

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
| `planB_e8_checks.json`, `planB_e8_summary.txt` | recorded E8 outputs for all 43 configurations |

Configurations are named `<parent>+<modifier>...` (`MODS` in `planB_e8_nested.py`): `hot`, `hotk1`, `prcp`, `prcpw25`, `core`, `coreonly`, `phi0`, `phi1`, `dep`, `depabs`, `gaterel`. `r21` = `v20+hot`, `r22` = `v20+hot+prcp`, and so on (aliases in `planB_e8_replay.py`).

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

Every row is regenerated by the committed script (`planB_design_tables.txt`; r5 fixed the temperature rows that r4's script had placed under the CDD branch). In 1915-1945: temperature v0.3 W −0.049453 against the no-drift level −0.004321; CDD actual v0.3 +0.010117 against the half level −0.001406. The constants (kappa temp 1.0, CDD 0.5) were fixed in r1 and are not re-tuned.

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

**E8. Configurations** (fully nested, chronological; 208 cities; `reward.py` arithmetic, 2,000 city draws, seed 20261108). Columns: vs average30 | 90% interval | live | origins won | by origin 1948 / 1955 / 1962 / 1969 / 1976 | metric points temp / prcp / hot / cdd | vs v0.5 (90%), origins won.

| Version (config) | vs avg30 | 90% | live | wins | by origin | metric pts | vs v0.5 |
|---|---:|---|---:|---:|---|---|---|
| v0.5 (`v05`) | +1.4647 | [+0.93, +1.97] | +2.49 | 4/5 | +0.06 / +2.12 / +7.40 / +0.18 / −2.44 | +2.37 / 0 / +0.32 / +3.17 | — |
| v2.0 (`v20`) | +1.6955 | [+1.19, +2.18] | +2.69 | 4/5 | +0.05 / +2.12 / +7.40 / +0.08 / −1.17 | +2.59 / 0 / +0.34 / +3.85 | +0.23 [+0.16, +0.31], 1/5 |
| v2.1 (`v20+hot` = `r21`) | +2.1994 | [+1.67, +2.71] | +3.00 | 4/5 | +0.33 / +2.27 / +9.00 / +0.89 / −1.50 | +2.59 / 0 / +2.35 / +3.85 | +0.73 [+0.62, +0.86], 5/5 |
| v2.1r (`v20+hotk1`) | +2.4713 | | | 4/5 vs v2.0 | | hot +3.44 | |
| **v2.2 on v2.1r (`v20+hotk1+prcp`)** | **+2.5532** | **[+1.96, +3.11]** | +3.00 | 4/5 | +0.68 / +2.30 / +10.10 / +1.70 / −2.02 | +2.59 / +0.33 / +3.44 / +3.85 | **+1.0885 [+0.87, +1.32], 5/5** |
| v2.5r gate-only (`v20+hotk1+prcp+gaterel`) | +2.6286 | [+2.03, +3.20] | +3.15 | 4/5 | +0.68 / +2.36 / +10.43 / +1.70 / −2.02 | +2.89 / +0.33 / +3.44 / +3.85 | +1.1639 [+0.94, +1.39], 5/5 |
| r4-wording endpoint (`r21+prcpw25`) | +2.2600 | [+1.72, +2.77] | +3.07 | 4/5 | +0.43 / +2.31 / +9.06 / +0.97 / −1.47 | +2.59 / +0.24 / +2.35 / +3.85 | +0.7953 [+0.67, +0.92], 5/5 |

(v2.1r's total is v2.0 + 0.7758, from the paired comparison `v20+hotk1` vs `v20`.) The core, core-only, phi and departure configurations on every parent the replay visited are in `planB_e8_replay.txt` and `planB_e8_summary.txt`. For temperature and CDD their deltas are identical across parents (for example the core +0.133316 and core-only −0.478913), because each metric's forecast depends only on its own settings.

**E9. Full policy replay** (`planB_e8_replay.py` → `planB_e8_replay.txt`). The champion analogue starts at average30 and the carried-forward version at v0.5. Promote needs P ≥ 0.95, ≥ 4/5 origin wins, live not clearly worse and every metric ≥ −2. Each diagnostic is judged after its version's decision (§4.3.5).

| Step [config] | vs champion | Δ | 90% lower | P | wins | Decision |
|---|---|---:|---:|---:|---:|---|
| v2.0 [`v20`] | average30 | +1.695493 | +1.189 | 1.000 | 4/5 | **kept** (promote) |
| v2.1 [`v20+hot`] | v2.0 | +0.503926 | +0.410 | 1.000 | 4/5 | **kept** (promote) |
| · diagnostic `hot-k1` [`v20+hotk1`] | v2.1 | +0.271834 | +0.167 | 1.000 | 3/5 | would carry → **v2.1r carried forward** |
| v2.2 [`v20+hotk1+prcp`] | v2.1 | +0.353772 | +0.220 | 1.000 | 4/5 | **kept** (promote) |
| · diagnostic `prcp-w75` [`v20+hotk1+prcpw25`] | v2.2 | −0.021312 | −0.064 | 0.201 | 1/5 | does not qualify |
| v2.3 core [`…+core`] | v2.2 | +0.133316 | −0.006 | 0.942 | 2/5 | not carried |
| · diagnostics `core-phi0` / `core-phi1` | v2.2 | +0.133316 / +0.062942 | −0.006 / −0.061 | 0.942 / 0.805 | 2/5 / 1/5 | do not qualify |
| v2.4 core-only [`…+coreonly`] | v2.2 | −0.478913 | −0.882 | 0.027 | 1/5 | not carried → `CORE_DROPPED` |
| v2.5 departure learner [`…+dep`] (frames differ) | v2.2 | +0.044980 | +0.002 | 0.956 | 1/5 | carried forward |
| · diagnostic `dep-absgate` | v2.2 | +0.054366 | +0.021 | 0.995 | 1/5 | would carry |
| · diagnostic `gate-only` [`…+gaterel`] | v2.2 | +0.075393 | +0.030 | 0.996 | 2/5 | would carry; higher lower bound → **v2.5r carried forward** |
| v2.6 MLP | — | — | — | — | — | not run (P4 has no departure learner; §5.3) |

**Endpoints.**
- **Champion:** v2.2 = `v20+hotk1+prcp` (v2.0 + full-weight hottest-day level + precipitation base). Against average30: **+2.553192**, 90% [+1.96, +3.11], 4/5. Against v0.5: **+1.088468**, 90% [+0.87, +1.32], 5/5.
- **Carried forward (not kept):** v2.5r = gate-only. Against average30: +2.628584, lower bound +2.032. Against v0.5: +1.163861. It is not a promotion and not departure learning. It changes temperature's lambda selection (0 / 0.5 / 0.5 / 0.75 / 0 by origin against the parent's 0 / 0 / 0 / 0.75 / 0). Departure learning against gate-only: −0.0304 (90% [−0.08, +0.02]).
- **Under r4's rescue wording** (diagnostics only for rejected versions), the endpoint would be `r21+prcpw25`, Sol's +2.2600.

**Robustness of the champion endpoint (report-only, §4.5).**
- Leave one origin out: without 1962, +0.67 against average30 (lower +0.09) and +0.69 against v0.5 (lower +0.47). The lowest other leave-one-out is +2.62 (without 1955).
- Spatial-block bootstrap (42 blocks of 5° × 5°): against average30 [+1.59, +3.38], against v0.5 [+0.72, +1.39].
- The 1976 fold still loses −2.02 against average30.

**Deterministic checks** (`planB_e8_checks.json`).
- The champion's lambdas by outer origin are all 0, except temperature 0.75 and hottest day 0.25 at 1969. Runtime 16-22 s per origin.
- Outer phi is 0 wherever the core is present (0.25 at 1948 for core-only).
- rho is at its lower bound only at anchors 1905-1920.
- r5 changed no configuration that r3/r4 reported: `planB_core` only gained optional `kappa`/`weight` arguments, with defaults unchanged. Sol's r4 reproduction still holds for those rows.

---

## 1. Strategy and model-selection logic

### 1.1 Choice

`forecast = assemble(settings, base, combo, W, shift, lambda)` (§3.2). The pieces, in order of evidence:
1. **A corrected gate** (v2.0).
2. **Simple regional bases** for the weak-signal metrics: a fixed-gain regional level for the hottest day (declared at kappa 0.5; the replay carries its kappa-1 diagnostic), and a longer-memory climatology for precipitation.
3. **Experimental:**
   - the regional local-level core with chronologically gated drift for temperature and CDD;
   - a city-departure learner;
   - an MLP member.

### 1.2 Why it suits the data (justified only from history available at each origin)

- **Regional signal, heavy noise (E1, E2).** The predictable part at leads 4-7 is a regional level plus slow city departures. Regional hottest-day levels persist (national lag 3 +0.47). That is why the hottest-day level comes first. In E8 it is the most consistent step (+0.50, 4/5 origins against its parent; kappa 1 adds +0.27).
- **Precipitation (E5).** It is close to white noise. A longer memory reduces the noise in the 30-year mean: +0.0037, positive in both eras. On its own the gain is small (+0.08 headline). It promotes in E9 only because it is compared against the champion v2.1 while sitting on the stronger rescued parent.
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
| `hot` | `kappa_hot·(L − cbar)`, kappa_hot 0.5 (v2.1) or 1.0 (diagnostic `hot-k1` / rescue v2.1r) |
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
5. **Diagnostics and rescues.** Diagnostics run with `--diagnostic` (never promoted; §5.4), right after their version is decided.
   - Each diagnostic is judged by rules 1-2 against the state **after** its version's decision: the champion and carried-forward version as they now stand, whether the version was kept, carried or rejected.
   - If any diagnostic would be kept or carried, exactly one is re-scored without the flag as candidate `v2.<n>r`. Order: one that would promote first, then the higher 90% lower bound against the champion, then the higher reward, then declaration order.
   - The rescue is judged by rules 1-3 and updates the champion and carried-forward version like any version. The next version's parent is the resulting carried-forward version. At most one rescue per version.
6. Only `official` runs count. `within_noise` is "not progress".

### 4.4 Promotion prospects

In E9 the champion endpoint (v2.2 on v2.1r) promotes against the analogue at city sd 0.35, against development sd 0.89 for v0.5. One scale illustration: v0.5 earns 0.57 of its retrospective points on development, which would map the endpoint to about +1.45. P ≥ 0.95 at sd 0.89 needs about +1.5. **Promotion on development is possible, not expected.** The steps with the largest paired lower bounds are v2.1 (+0.41) and v2.1r's hottest-day kappa 1 (+0.17 over v2.1).

### 4.5 Report-only diagnostics (pre-declared, never select)

Leave-one-origin-out and the 42-block spatial bootstrap, for every E8 configuration and every development run, plus the insights zone table. E9 values for the champion endpoint are given above.

---

## 5. Implementation and versions

### 5.1 Order and stop rule (under one day)

1. v2.0 code with per-metric settings (about 2 h).
2. Hottest-day level and precipitation base (about 1 h; they use `planB_core`).
3. **Replication before any `line.py score`:** `notebooks/Meet/retro_v2.py` runs the adapter on the E8 origins with the ≤ 1983 table.
   - Stop rule: phi and lambda equal to `planB_e8_checks.json`, rho within 1e-4 relative, final points per origin within 0.01, for `v20`, `r21`, `v20+hotk1` and `v20+hotk1+prcp`. Otherwise fix the bug before scoring.
   - Time origin 1976 (< 120 s) and run it twice (identical output).
4. Core (about 3 h) after v2.2 is scored. Before scoring v2.3, replicate `<C2>+core` and `<C2>+coreonly` the same way (E9: `v20+hotk1+prcp+core` / `+coreonly`).
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
 └─ v2.0 assembly fix                 parent v0.5   → if neither kept nor carried: STOP
     (each version Vn: decide Vn, then its diagnostics against the new state; at most one rescue Vnr; §4.3.5)
     └─ v2.1 hottest-day level        parent C0     → C1 := carried-forward version after v2.1 / v2.1r
         └─ v2.2 precipitation base   parent C1     → C2
             └─ v2.3 regional core    parent C2     → C3 := v2.3 / v2.3r if kept/carried
                 └─ else v2.4 core-only (immediately) parent C2 → C3 := v2.4 if kept/carried else C2 (CORE_DROPPED)
                     └─ v2.5 departure learner   parent C3 (skip check)   → C4
                         └─ v2.6 MLP, only if C4 contains the departure learner (v2.5 itself kept/carried):
                            rehearsal derive(C4, temp: mlp) first; score only if its paired 90% lower bound vs C4 > 0
optional, exploratory, never a parent: C4 + features-v5 SST anomalies as an MLP input
```

`Ck` is always the carried-forward version after step k, including any rescue. E9 path: C0 = v2.0, C1 = v2.1r (`hot-k1`), C2 = v2.2, C3 = C2 (`CORE_DROPPED`), C4 = v2.5r (`gate-only`), and v2.6 is not run.

| Version | Settings change from parent | Insight finding (v0.9) | Hypothesis, stated before the run | E9 decision |
|---|---|---|---|---|
| v2.0 | the gate scores the full `assemble()` | none (correctness) | The gate should judge the forecast it deploys | kept (promote) |
| v2.1 | hottest: `W = hot` (kappa 0.5) | none ranked; checked against the hottest-day headroom and calibration | A persistent regional hottest-day level is predictable | kept (promote); its diagnostic `hot-k1` rescued as v2.1r (carried) |
| v2.2 | precipitation: `W = prcp` | `no_signal` (low priority) | A longer memory reduces the noise in the 30-year mean | kept (promote, on parent v2.1r) |
| v2.3 | temp, CDD: `W = core` | `bias_flips` / `origin_instability` | H-core (§1.2) | not carried |
| v2.4 | temp, CDD: `W = core`, `cw = 0`, `learn = False` | same | The core alone, without the blend | not carried; `CORE_DROPPED` |
| v2.5 | temperature: `dep`, `gate_rel` | `city_offsets` (honest +0.98) | City departures persist (E2 lag 1 +0.64), and a departure-from-region learner can recover them | carried, then displaced by its `gate-only` diagnostic (v2.5r) |
| v2.6 | temperature: MLP member (requires `dep` in C4) | `city_offsets` | A nonlinear member adds to v2.5 (E6 +0.049 against +0.043) | not run (C4 has no departure learner) |

**Skip rule (v2.5, v2.6).** For each development origin, build the parent's and the child's prediction frames from that origin's training table only. Compare them element-wise on the full target grid. Skip, recording "not scored (identical frames)", only if they are equal at all three origins. A skipped version's diagnostics are still judged (§4.3.5).

**v2.6 and its rehearsal.** v2.6 runs only if C4 contains `dep` and `gate_rel` for temperature, i.e. v2.5 itself was kept or carried and no rescue replaced it. If v2.5 is rejected, skipped or displaced by a rescue without the departure target, v2.6 is not run: adding the MLP would then also add the unproven departure target. When it does run, the rehearsal configuration is `derive(C4, annual_avg_temp={"mlp": True})`: seeds 42-46, the full v0.5 + departure feature set, the W-adjusted departure target, nested exactly like E8. v2.6 is scored only if the rehearsal's paired 90% lower bound against C4 is > 0.

### 5.4 Pre-declared development diagnostics (`line.py score --diagnostic`)

| Version | Diagnostics (name: change), each on the version's parent |
|---|---|
| v2.1 | `hot-k1`: hottest-day kappa 1.0 |
| v2.2 | `prcp-w75`: weight 0.75 on ref30 (0.25 on the all-history mean) |
| v2.3 | `core-phi0`: phi fixed 0; `core-phi1`: phi fixed 1 |
| v2.5 | `dep-absgate`: departure target with v0.5's absolute gate; `gate-only`: v2.5's relative gate without the departure target |

Each diagnostic has the same parent and settings as its version except the named change. Rule §4.3.5 applies.

---

## 6. Lessons and findings addressed

| Lesson / finding | Response |
|---|---|
| 1. Warming learned as cooling | Core with no learned weights; median- or region-removed targets |
| 2. 1990 overshoot / `bias_flips` | v2.3/v2.4 test H-core under the chronological phi rule. Rejected in E9, and that rejection is reported |
| 3. Zero precipitation skill, near-zero hottest day | v2.1 (+0.50, promotes) and its rescue v2.1r (kappa 1, +0.27 more); v2.2 (+0.33 metric points, promotes on v2.1r) |
| `city_offsets` | v2.5 tested; its `gate-only` diagnostic shows the gain is the gate, not departure learning (−0.03), so this finding stays open; v2.6 not run |
| CDD `amplitude` / `control_beats_model` | kappa_cdd 0.5 in the core. E3's corrected row shows the actual v0.3 CDD term is competitive; reported after scoring |
| Northern zones | Insights zone table and spatial blocks (report-only) |
| 5. Reused years | Retrospective label, chronological selection, fixed constants, diagnostics flagged and pre-declared |

---

## 7. Risks, compute, dependencies

| Risk | Response |
|---|---|
| Adapter disagrees with E8 | Stop rule §5.1 |
| The hottest-day gain does not transfer | v2.1 remains judged by §4.3; plan B then continues from v2.0 |
| Small steps fail the strict carried-forward rule (precipitation did in r4's replay) | Accepted and mechanical; reported, not retried |
| The core widens origin swings | Placed after the supported steps; `CORE_DROPPED` branch defined |
| Departure learner adds only gate noise | E9's `gate-only` comparator shows this (departure learning −0.03 against gate-only); v2.6 runs only if v2.5 itself survives |
| The precipitation promotion is an artefact of comparing against an older champion | Reported; the paired step on its own parent is +0.08 |
| Runtime | Anchor step 10 (once); MLP deferred |

Compute: CPU only. Dependencies: none new. No feature release.

---

## 8. Tracking

Every version and diagnostic goes through `line.py score` (diagnostics with `--diagnostic`), then `line.py sync`. Tags: `series=Placeholder`, `version`, `plan=B`, `parent`, `champion_before`, `finding`, `core_dropped`, `line.role` (diagnostic). After each score, run `code/evaluation/insights.py <run_id> --json` and record the targeted finding's honest headroom before and after.

Metrics, in decision order: verdict and P(better) against the recorded champion; 90% lower bound; reward and live points; points by origin and metric (−2 floor); the §4.5 diagnostics; `runtime_s`. Artifacts: phi tables, rho by anchor with flags, F1/F2 counts, lambda tables, mean W, settings, and the skip-check result. Results go to `docs/models/placeholder.md`.
