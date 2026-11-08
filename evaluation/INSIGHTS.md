# Insights: where a model loses points and what to try next

The [reward](REWARD.md) answers *how good is this run, and is it really better?* Insights answer *where
does it lose points, why, how much could a fix win back, and what should the next version try?*
Insights never change a score, the evaluator (`eval-v1`) or the reward. They read only the sealed
development run (years up to 2004) and never read 2019–2025 observations.

```bash
uv run python scripts/line.py score ...                     # prints reward, then the top 6 insights
uv run python code/evaluation/insights.py <run_id>          # ranked findings (vs the reward champion)
uv run python code/evaluation/insights.py <run_id> --json   # everything, machine-readable
```

Each run writes `outputs/evaluation/insights/<run_id>.json` (for agents) and `.md` (for people).

## Units

Everything is in **headline points**, the reward's unit. A metric's own points move the headline by a quarter of their size.
City and zone *contributions* are additive: summed over all cities or zones, they equal the reward exactly.

## What is measured

| Section | Question it answers | Key numbers |
|---|---|---|
| `headroom` | How much could each metric still give, and which simple fix would help? | points now; **ceiling** (each city's future window known: unreachable, shows the remaining year-to-year noise); what-if gains for removing the mean bias, rescaling departures and removing city offsets |
| `calibration` | Is the model right about how far, and in which direction, the future departs from the 30-year average? | mean bias (and by origin); pattern correlation of departures (within origin); **best scaling** (<1: too bold, >1: too timid); share of the observed shift captured; bias change per lead year |
| `decomposition` | What kind of error is it? | share of squared error from mean bias, **common year** (all cities share it: that year's national weather), **persistent city** (a city's level is off), residual |
| `lead` | Does skill hold at leads 4–7, which the final score uses? | points and bias for each lead 1–7 |
| `segments` | Where is it lost? | origins; geographic zones; station-change cities; worst 10 cities with their worst metric; how concentrated the losses are |
| `versus_champion`, `best_controls` | Who does better, and on what? | metric and zone deltas vs the reward champion; the best control per metric and the gap to it |
| `insights` | What to do next, ranked | `{kind, metric, finding, evidence, suggestion, headroom_points, confidence}` |

## Honest versus fitted what-ifs

A correction fitted on the same rows it is scored on always looks like a win. Every what-if is therefore
computed twice:

- **fit**: learned on the scored rows. This is optimistic and shown only for reference.
- **honest**: learned elsewhere. The mean bias and the departure scaling come from the *other origins*; a
  city's offset comes from its *other years*.

Findings are ranked by the honest gain. If an in-sample win disappears when cross-fitted, it was noise.
On Placeholder_v0.9, removing the temperature bias gains +1.72 in-sample but −2.69 honestly, because the bias
flips sign between origins. Precipitation city offsets gain +4.20 in-sample but −2.95 honestly.

Even honest gains use the development target years, so they are **upper bounds**. Never copy a fitted number
into a model. Build the correction from training data inside each origin, then confirm it with a new run
and its reward verdict.

## Finding kinds

| Kind | Fires when | Typical fix |
|---|---|---|
| `level_bias` | a mean bias that also holds in the other origins (honest gain ≥ 0.25) | level or trend estimation, recent-year weighting, pooled trends |
| `bias_flips` | a large bias in-sample that changes sign between origins | damp the trend; more robust level windows; fall back when the trend is unstable |
| `amplitude` | best scaling < 0.8 or > 1.25, with an honest gain | shrink toward, or move away from, the 30-year average |
| `shift_captured` | a warming metric captures < 60% of the observed shift | weight recent years more, regional trend |
| `city_offsets` | persistent city share ≥ 30% and honest gain ≥ 0.25 | robust or break-aware city levels; station-change handling |
| `common_year` | common-year share ≥ 40% | not fixable from station history; only large-scale drivers (for example the enhance-v2 ocean indices) |
| `no_signal` | the metric equals climatology | add a signal only if a control does better there |
| `lead_decay` | leads 1–3 beat leads 4–7 by > 1.5 points | adjust the trend by lead; blend toward the 30-year average at long leads |
| `origin_instability` | one origin < −1 while another is > +1 | regime-robust method; fallback rule |
| `zone`, `station_changes`, `concentrated_losses` | a segment of ≥ 10 cities loses ≥ 0.25 points; the 10 worst cities hold ≥ 40% of all losses | regional fitting, break handling, per-city fallback |
| `control_beats_model` | a control beats the run on a metric by ≥ 0.25 headline points | borrow or blend that control's method for that metric |

## How training agents should use it

1. Score with `line.py score`, read the reward verdict, then the insights.
2. Pick **one** top finding. Prefer high honest headroom, `measured on the same folds`, or
   `cross-fitted` confidence. Write down the hypothesis before changing code.
3. Implement the fix so it learns only from each origin's training data, then score it as the next version.
4. Keep the change only if the reward verdict is `promote`. A finding is a hypothesis, not a result.

Diagnosing many runs on the same three folds overfits them like any other tuning. One hypothesis per version, and
the ladder, keep that in check.

## Example: Placeholder_v0.9 (reward +0.93, `within_noise`)

1. Temperature: 32% of squared error is a persistent city offset (honest +0.98).
2. Origin 1990 fails (−2.11). Temperature and CDD biases flip sign between origins (1983 −0.98 °F,
   1990 +0.52 °F, 1997 −0.20 °F), so the method is regime-sensitive.
3. The Central-North and East-North zones lose about 0.5 points each.
4. `control:mean10` beats it on CDD by 0.29. CDD departures are too large (best scaling 0.72, honest +0.26).
5. Precipitation equals climatology, and no control does better, so it is low priority.
