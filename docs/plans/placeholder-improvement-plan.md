# Plan: improving the Placeholder series (v0.3 onward)

Status: **revised after Grok review (44/100); see section 5.** Author: Claude Code (Claude Opus 5.5), 2026-11-08. Owner: training station. Deadline: challenge
closes 2026-11-10 23:59 ET. Current kept version: `Placeholder_v0.1`. Results: `docs/models/placeholder.md`.

## 0. Where we are

Task: forecast four annual metrics for 210 U.S. cities for 2019-2025, as of 2018-12-31. Score per metric: `skill = 1 - MAE / MAE(30-year
city mean)`. Development runs score on suite-v3 origins 1983, 1990, 1997 through `line.py score` (evaluator `eval-v1`). Headline = equal-origin
mean of unclipped final-analogue skill (leads 4-7).

| Version | Headline | 1983 | 1990 | 1997 |
|---|---:|---:|---:|---:|
| v0.1 (kept) | -0.0008 | -0.0073 | +0.0050 | 0.0000 |
| v0.2 (30-year training, not kept) | -0.0064 | -0.0103 | -0.0090 | 0.0000 |
| combo | -0.0134 | | | |
| best control `mean10` | -0.0087 | | | |

Diagnosis (details: model card, plan section 13 of `placeholder-training-plan.md`):

1. Final-split bias, forecast minus actual over reference MAE: annual_avg_temp -0.575, cdd -0.307, precipitation -0.058, hottest day -0.044. Almost all fixable error is common warming in temperature and CDD.
2. On inner folds the ensemble's mean shift for annual_avg_temp was -0.35 F against a real +0.17 F; CDD -43 against +8. The learners absorbed mid-century cooling as a shared intercept. (That look overlapped development outcomes.)
3. The gate chose lambda 0 at 1997 for every metric, while unshrunk `lambda1` scored +0.0297 there.
4. v0.1 beats combo at 1983 (+0.030) and 1990 (+0.034) and loses at 1997 (-0.026), so the errors differ.
5. Combo's backtest score rises from 1.6 (1987-94) to 7.4 (2011-18): development years understate how much warming pays later.

Honesty note: every development year has now been seen several times (v0.1, v0.2, ablations, including 1997). All are reused evidence. The only clean test left is the integrator's one-shot assessment on 2004 and 2011.

## 1. Rules for this round

- One change group per version, built on the last kept version; each version is one single-file adapter, committed, scored from a clean commit with
  `line.py score --suite outputs/evaluation/suite-v3`.
- Each version logged to MLflow with `line.py sync`. At most 4 extra comparison runs per version; none are planned.
- **Keep rule (fixed before any run).** A version replaces the current kept version only if its headline is higher AND its origin-1997 final skill is strictly higher. A 1997 tie is not kept. The rule is applied mechanically.
- Do not read origin 2004/2011, any `private/` folder, or anything after 2018. Do not change `submission.csv`. Assessment and any origin-2018 forecast belong to the integrator.

## 2. The versions (planned order)

**v0.3: regional warming plus a gated city departure (main bet).**
`forecast = base + W + lambda * residual_hat`. W is not gated. `W = rate * (t + h - mean valid year of the 30-year mean)`.
For annual_avg_temp the rate is the 30-year OLS slope averaged over the city and its 15 nearest cities, floored at 0. For CDD the rate is that temperature rate times the regional OLS response of CDD to temperature in the same window (at least 15 valid years, floored at 0). W is 0 for precipitation and the hottest day.
Learners fit `y - W`, with the median across cities removed in each (t, h), so they rank cities and cannot learn an era-wide shift. Lambda, chosen on inner origins by the v0.1 rule, scales only that residual. One pre-declared ablation: no median removal.

**v0.4: steadier gate (built on the last kept version).** Inner origins every year from O-21 to O-7 (15 origins, overlapping) instead of three, and one shared lambda across the temperature-based metrics (annual_avg_temp, hottest_day_temp, cdd), with precipitation separate. Same selection rule (smallest lambda within 0.005 of the best; 0 if best <= 0).

**v0.5: fixed 50/50 average with combo.** `forecast = 0.5 * Placeholder + 0.5 * combo`, with no fitted weight. Caveat: combo's method was picked on years overlapping the development folds, so this flatters it.

**v0.6: physics-based CDD (uses `features-v1`, so runs are `exploratory`).** Replace the learned regional trend for CDD with the first-order warming response `days_ge_65_mean_30y * W_temp(h)`, because a uniform warming of x F adds about `days at or above 65 F` x x degree-days. Hottest-day percentiles are not in `features-v1`, so that part is dropped for now.

**v0.7: boosted member only** (`members = gbm`), one change.

**v0.8: station-change covariates** from `features-v1` (`years_since_station_change`, `station_changes_10y`, `station_changes_30y`), `exploratory`.

Not runnable yet: per-origin cleaned inputs (`clean-v0` is not gate-eligible and no `clean-v1` exists), geography and ocean indices (no `enhance-v1`).

Adaptation: if a result shows a version needs adjusting (for example the bias stays, or W is unstable), I change only the next version and record why. No version is re-tuned after seeing its own score.

## 3. Candidate and hand-off

The candidate for submission must be an `official` run (so not v0.6/v0.8) and the best kept version. Passing the gates and the keep rule is the bar to ask the integrator for a freeze and the one-shot assessment. Fitting origin 2018 and changing `submission.csv` come after that, by the team's decision.

## 4. What I cannot promise

Three reused periods cannot prove an improvement. The likely ceiling is low because annual weather variability dominates at leads 4-7 and precipitation will probably stay at skill 0.

## 5. Grok review (44/100) and response

Adopted: the regional slope uses 30 years, not 50 (the 50-year slope is cooling or flat on these origins), floored at 0; W is applied outside the gate; W covers only temperature and CDD, with CDD through a response estimated from the data; extrapolation from the mean valid year; averaging neighbours' slopes; the stricter keep rule.

Declined, with reasons: (a) dropping the median removal: Grok's concern depended on the wrong-sign 50-year W, and removing era-wide means is the direct fix for the diagnosed cooling bias; it is tested as a v0.3 ablation. (b) Dropping v0.4, v0.5 and v0.7: the project owner asked for every model to be trained; each still has to pass the keep rule. v0.5's combo method was chosen on backtests that include these development years, so a v0.5 gain is flattered, and this is reported with its result. (c) Removing adaptation between versions: the owner asked for it; a change only ever applies to the next version, never to a scored one, and is recorded here.
