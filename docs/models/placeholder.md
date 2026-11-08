# Placeholder model series

The Placeholder series is the output of the model-training station in the team's assembly line (cleaning, feature
engineering, enhancement, training, evaluation). Each version is a single reviewed change, evaluated on the same
development folds through `scripts/eval_model.py`. Plan: [`docs/plans/placeholder-training-plan.md`](../plans/placeholder-training-plan.md).

## Placeholder_v0.1 (2026-11-08)

**Status: first baseline. Development evidence only; not assessed, and not used for `submission.csv`.**

| | |
|---|---|
| Method | Direct, pooled, multi-horizon shift regression. `forecast = 30-year mean + lambda * shift_hat`, where `shift_hat` averages a Huber regression and a shallow gradient-boosted model, per metric |
| Data | Suite `train.csv` per origin, plus `data/city_info.csv` for choosing neighbouring cities (declared static asset). Nothing dated after 2018 |
| Files | `notebooks/Meet/placeholder_v0_1.py`, `placeholder-v0.1-assets.json`, `test_placeholder_v0_1.py`, `03-placeholder-v0.1-training.ipynb` (all in `notebooks/Meet/`) |
| Run | `20261108T111421Z-b7a95baa04`, suite-v3 `4029caa1...`, protocol `parent-audited-v6`, origins 1983/1990/1997. The identical earlier run under `audited-inputs-v5` (`20261108T104530Z-098e5a76c8`) is archived |
| MLflow | Run `Placeholder_v0.1`, tags `series=Placeholder`, `version=v0.1`, in experiment `protostar/4029caa1.../development/parent-audited-v6/official`, with its four ablations alongside it |

### Results (development origins, reused history)

Headline is the equal-origin mean of unclipped final-analogue skill (leads 4-7); 0 equals the 30-year mean.

| Model | Headline |
|---|---:|
| 30-year average (reference) | 0.0000 |
| **Placeholder_v0.1** | **-0.0008** |
| nested selection | -0.0065 |
| 10-year mean (best control) | -0.0087 |
| combo (method behind the current `submission.csv`) | -0.0134 |
| ridge_pooled | -0.0225 |
| blend | -0.0524 |
| 30-year trend | -0.1216 |
| starter random forest | -0.2648 |
| last year repeated | -0.3746 |

| Metric | Final skill | Live skill |
|---|---:|---:|
| annual_avg_temp | -0.0042 | +0.0010 |
| total_precip_in | 0.0000 | 0.0000 |
| hottest_day_temp | -0.0034 | +0.0020 |
| cdd | +0.0045 | +0.0070 |

By origin, final skill is -0.0073 (1983), +0.0050 (1990), 0.0000 (1997). Leaderboard formula per fold, averaged: final 0.17,
live 0.25.

**Gates, as printed by `scripts/eval_model.py`:** G1 beat the 30-year average **FAIL** (-0.0008); G2 beat the best control
**PASS** (`mean10` -0.0087); G3 no metric below -0.02 **PASS**; G4 beat combo **FAIL** (mean +0.0127, better on two origins,
but -0.0257 on 1997, the origin closest to a 2018 forecast).

Selected shrinkage:

| Origin | annual_avg_temp | total_precip_in | hottest_day_temp | cdd |
|---|---:|---:|---:|---:|
| 1983 | 0.50 | 0 | 0.50 | 0.50 |
| 1990 | 0 | 0 | 0.25 | 0.25 |
| 1997 | 0 | 0 | 0 | 0 |

Mean inner-fold final skill (nine folds, three inner origins repeated across outer origins; per-fold records with
`outer_origin` and `inner_origin` are in MLflow artifact `inner_fold_skill.csv`):

| Metric | lambda 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| annual_avg_temp | 0 | +0.0072 | -0.0071 | -0.0417 | -0.0946 |
| total_precip_in | 0 | -0.0035 | -0.0081 | -0.0140 | -0.0208 |
| hottest_day_temp | 0 | +0.0098 | +0.0112 | +0.0045 | -0.0097 |
| cdd | 0 | +0.0107 | +0.0096 | -0.0025 | -0.0242 |

### Reading the result

The model does what the design intends: it rarely does worse than the reference, and it ranks ahead of every other method
tried except the reference itself. It does not yet beat the 30-year mean. On these eras no method does, and the gate
correctly returns most forecasts close to that mean. The headline differences from the reference are within noise for three
folds. Expect a comparable or better signal in warmer-regime origins, but that is untested here: the training windows used
for these folds include mid-century cooling, which makes learned warming persistence less useful, and a later era may
behave differently.

Limits: three reused development folds; the lambdas are not independent across origins (see the plan, section 13); no
assessment run; and a pre-run look at inner-fold skill overlapped development outcomes for origins 1983 and 1990, so origin
1997 is the only clean development check of any change suggested by it. Origin 1997 selected lambda 0, so its score does
not test the learned shift.

### Ablations (protocol `parent-audited-v6`, pre-declared; they cannot replace v0.1)

Each changes one thing. Final skill by origin is unclipped.

| Variant | Change | Headline | 1983 | 1990 | 1997 | Gates |
|---|---|---:|---:|---:|---:|---|
| **Placeholder_v0.1** | - | **-0.0008** | -0.0073 | +0.0050 | 0.0000 | G2, G3 pass |
| `gbm` | boosted member only | +0.0047 | +0.0055 | +0.0085 | 0.0000 | G1, G2, G3 pass |
| `lin` | linear member only | -0.0041 | -0.0124 | 0.0000 | 0.0000 | G2, G3 pass |
| `no_pool` | no regional or cross-metric features | -0.0045 | -0.0158 | +0.0054 | -0.0031 | G2, G3 pass |
| `lambda1` | no shrinkage (lambda = 1 everywhere) | -0.0304 | -0.0356 | -0.0854 | +0.0297 | none pass |

What these show, and their limits:

- **Regional and cross-metric pooling looks helpful.** Removing it is worse on 1983 and 1997 and about level on 1990, which
  supports the design choice within the limits of three folds.
- **The boosted member looks stronger than the linear one.** `gbm` alone is the only variant with a positive headline. It
  is one look at three reused folds and was not selected by a pre-registered rule, so it is a hypothesis for a later
  version, not a result. It still fails G4, because 1997 is 0 while combo is positive there.
- **The gate is doing real work.** Without shrinkage, the headline falls to -0.0304, led by origin 1990.
- **Origin 1997 is the clean check, and it cuts the other way.** The unshrunk model scores +0.0297 there, against 0 for
  v0.1, whose gate chose lambda 0. So on the one origin with no prior look, the gate gave up real skill. One origin is not
  enough to change the rule; this is an open question for the next versions.

Finished v0.1 runs are reused when the notebook is re-executed; `PLACEHOLDER_RUN_ABLATIONS=1` runs any missing ablation.

## Placeholder_v0.2 (2026-11-08)

**Status: not kept. Placeholder_v0.1 (45-year window) stays the series baseline.**

| | |
|---|---|
| Change | One setting: the training window, `TRAIN_YEARS = 30` instead of 45. `notebooks/Meet/test_placeholder_v0_2.py` checks that nothing else differs |
| Files | `notebooks/Meet/placeholder_v0_2.py`, `placeholder-v0.1-assets.json` (the same asset manifest) |
| Run | `20261108T130154Z-799a27c056`, labelled `official`, commit `0028950`, suite-v3 `4029caa1...`, protocol `parent-audited-v6`, evaluator `eval-v1` |
| MLflow | Not mirrored yet: the team server returned 401 (no credentials in this environment). The sealed run is safe; run `line.py sync` once credentials are set |

| | Headline | 1983 | 1990 | 1997 |
|---|---:|---:|---:|---:|
| Placeholder_v0.1 (45 years) | -0.0008 | -0.0073 | +0.0050 | 0.0000 |
| Placeholder_v0.2 (30 years) | -0.0064 | -0.0103 | -0.0090 | 0.0000 |

Final skill by metric: annual_avg_temp -0.0182, total_precip_in 0.0000, hottest_day_temp -0.0008, cdd -0.0068. Gates: G1 fail,
G2 pass (`mean10` -0.0087), G3 pass, G4 fail.

**Decision, by the pre-declared rule:** keep the 30-year window only if origin 1997's final skill is higher than v0.1's.
It is not: both are 0.0000, because the gate picked lambda 0 for every metric at 1997 in both versions. So 1997 could not
test the window here. The shorter window is worse on 1983 and 1990, and the selected lambdas are identical to v0.1's at
every origin, so the difference comes from the fitted shifts, not the gate. This is one comparison on three reused folds
and does not show that 45 years is the right window, only that 30 years did not earn a change.

### Pending

Move to `features-v1` as soon as it is released, one change group at a time. `gbm`-only and the strictness of the gate
remain candidates, one change group each.

## Placeholder_v0.1_clean (2026-11-08)

**Status: not kept.** Training on cleaned history works end to end, but it does not raise the headline. The first version of
the series to train on cleaned history, from the per-origin release `clean-v1` (gate-eligible). Built on `Placeholder_v0.1`,
the kept version when this started (v0.3 has since replaced it; see below).

| | |
|---|---|
| Change | One change group: the tables behind every feature and shift target come from `origin-YYYY/annual_clean.parquet`, one series per metric through the `inputs` parameter. The anchor `ref30` and the inner-fold scoring stay on the official labels, because those are what is scored and lambda 0 must stay the reference. With every metric on `official` this is v0.1 (tested) |
| Files | `notebooks/Meet/placeholder_v0_1_clean.py`, `placeholder-v0.1-clean-assets.json`, `placeholder-v0.1-clean-{qc,homog,fill,pick}.json`, `test_placeholder_v0_1_clean.py` (all in `notebooks/Meet/`) |
| Data | `clean-v1/origin-YYYY/annual_clean.parquet`, one file per origin, each built only from data up to that origin. Declared as `weather_input` per origin |

**Caveat.** Inner folds use the cleaned series of their outer origin, so the cleaning behind an inner fold saw years up to that
outer origin, not just up to the inner one. Nothing passes the outer origin, so the run stays inside the data rule, but the gate
may slightly favour cleaned inputs.

**Selection rule, fixed before any run.** `clean-v0`'s picks came from backtests through 2018 and are not reused.

1. Run three recipes on suite-v3 development origins 1983, 1990 and 1997, and read no other origin:
   `qc` (all four metrics on QC values), `homog` (station-adjusted temperature metrics and CDD, QC precipitation) and `fill`
   (filled and station-adjusted temperature and CDD, filled precipitation, station-adjusted hottest day).
2. Per metric, take the series with the highest equal-origin mean final skill. Keep it only if it beats that metric's skill in
   the sealed v0.1 run `20261108T111421Z-b7a95baa04` by at least 0.002. Otherwise the metric stays on `official`.
3. If the per-metric picks are not one of the three recipes, run that combination once. That is the fourth and last run.
4. Apply the series keep rule (`docs/plans/placeholder-improvement-plan.md`, section 1) against v0.1: headline higher, and either
   a higher 1997 final skill if a shift applied at 1997, or a win on at least 2 of 3 origins if lambda is 0 there in both.

The development years have been seen several times, so a pick here is reused evidence, not an independent test.

**Runs.** suite-v3, development origins only, adapter and evaluation code identical in all four (no diff from `af917dd` to
`c217d1b`). None are on MLflow yet: the server refused the connection; run `line.py sync` once it is up.

| Recipe | Run | Commit | Headline | 1983 | 1990 | 1997 |
|---|---|---|---:|---:|---:|---:|
| v0.1 (reference) | `20261108T111421Z-b7a95baa04` | `feafddc` | 0.1678 | 0.0000 | 0.5035 | 0.0000 |
| `qc` | `20261108T140227Z-695b90931b` | `7532708` | 0.1678 | 0.0000 | 0.5035 | 0.0000 |
| `homog` | `20261108T140426Z-2b27228f7a` | `7532708` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `fill` | `20261108T140654Z-19687d1bb4` | `7532708`, exploratory | 0.0634 | 0.0000 | 0.1901 | 0.0000 |
| pick | `20261108T140925Z-7c06c3dfed` | `c217d1b` | 0.1678 | 0.0000 | 0.5035 | 0.0000 |

Headline and origin columns are the clipped final score. The `fill` run is labelled exploratory because another thread's
untracked v0.4 files were in the checkout while it ran; its code matched the other runs.

Mean final skill over the three origins (step 2):

| Metric | v0.1 | `qc` | `homog` | `fill` | Pick |
|---|---:|---:|---:|---:|---|
| annual_avg_temp | -0.00416 | **-0.00132** | -0.00504 | -0.00493 | `qc` (+0.00284) |
| total_precip_in | 0.00000 | 0.00000 | 0.00000 | 0.00000 | official (no gain) |
| hottest_day_temp | -0.00340 | -0.00340 | -0.00160 | -0.00210 | official (`homog` gains +0.00180, under 0.002) |
| cdd | 0.00449 | 0.00449 | -0.00253 | 0.00132 | official (nothing beats v0.1) |

The pick (QC average temperature, everything else official) is none of the three recipes, so it was run once
(`placeholder-v0.1-clean-pick.json`). Its scores match `qc` to the digit, as expected: under `qc` the other three metrics already
scored exactly as on official labels, so only average temperature differs from v0.1.

**Keep rule (step 4).** The headline is equal to v0.1 (0.1678), not higher, so the pick is **not kept**. The whole gain is at
1983, where QC average temperature moves final skill from -0.0125 to -0.0040; the clipped 1983 score is 0 either way. At 1990 and
1997 the gate chose lambda 0 for average temperature in both runs, so the forecast there is the 30-year average.

**What this means.** The cleaning removes some error at 1983, but the gate does not find a shift worth applying at the later
origins from cleaned inputs alone. Station-adjusted series (`homog`) made CDD worse at every origin. The cleaned release is ready
for any later version that builds features from it; v0.3, the current base, still reads official labels only.

## Placeholder_v0.3 (2026-11-08)

**Status: kept** (replaces v0.1 as the base of the series). Plan: `docs/plans/placeholder-improvement-plan.md`, revised after
the Grok review (44/100).

| | |
|---|---|
| Change | One change group on v0.1: an ungated regional warming term W for annual_avg_temp and CDD (30-year OLS temperature slope averaged over the city and its 15 nearest cities, floored at 0; CDD through its regional response to temperature, floored at 0), and learners fitted on `y - W` with the cross-city median removed per (t, h) |
| Files | `notebooks/Meet/placeholder_v0_3.py`, `test_placeholder_v0_3.py`; assets `placeholder-v0.1-assets.json` |
| Run | `20261108T140236Z-218e9f305e`, clean commit `7532708`, suite-v3, eval-v1. Not on MLflow yet: the server refused the connection; run `line.py sync` once it is up |

| Final skill | Headline | 1983 | 1990 | 1997 |
|---|---:|---:|---:|---:|
| v0.1 | -0.0008 | -0.0073 | +0.0050 | 0.0000 |
| v0.3 | **+0.0079** | +0.0251 | -0.0386 | +0.0374 |

Per metric, v0.3: annual_avg_temp +0.054 / -0.115 / +0.139 (1983 / 1990 / 1997), CDD +0.035 / -0.048 / +0.013, hottest day
+0.012 / +0.009 / -0.002, precipitation 0.000 everywhere. Mean W at the outer origin: temperature +0.09 F (1983), +0.64 F
(1990), +0.54 F (1997).

**Keep rule:** headline higher (+0.0079 > -0.0008) and 1997 strictly higher (+0.0374 > 0.0000), so kept.

**Reading.** The warming term fixes most of the cold bias at 1983 and 1997. At 1990 it overshoots: the 1961-1990 slope is
steep (1960s-70s cooling, then 1980s warming) and the final years 1994-1997 include the cold 1996, so temperature loses 0.115.
One origin out of three going the wrong way is a real risk for the submission, not noise to explain away. Adaptation (applies
to later versions only): if v0.4-v0.7 do not reduce the 1990 loss, a version that damps W (for example a slope from a longer
window blended with the 30-year one, or W gated with its own lambda) is the next candidate.

## Placeholder_v0.4 (2026-11-08)

**Status: not kept.** Built on v0.3. Change: lambda chosen on 15 inner origins (O-21 to O-7), one shared lambda for
annual_avg_temp, hottest_day_temp and CDD. Files `placeholder_v0_4.py`, `test_placeholder_v0_4.py`. Run
`20261108T140828Z-f06f18f6a3`, clean commit `f6431b5`.

| Final skill | Headline | 1983 | 1990 | 1997 |
|---|---:|---:|---:|---:|
| v0.3 (kept) | +0.0079 | +0.0251 | -0.0386 | +0.0374 |
| v0.4 | +0.0017 | +0.0108 | -0.0432 | +0.0376 |

Selected lambdas: 0 for every metric at 1983 and 1990, 0.5 for the temperature group at 1997. Headline lower, so not kept.
The 1990 loss stays with lambda 0, so it comes from the warming term, not from the learners: damping W is the next fix.
