# Reward v1: what training agents optimise

One number per model, in **leaderboard points**, plus a verdict that says whether a gain is real.
It reads only sealed development runs (`row_errors.csv`, years up to 2004). It does not change the
evaluator (still `eval-v1`) or any existing score, and it never reads 2019–2025 observations.

## Commands

```bash
uv run python scripts/line.py score --name Placeholder_vX --adapter <file.py:forecast> --assets <manifest.json>
#   ...prints the reward block automatically at the end. Or, for any sealed run:
uv run python code/evaluation/reward.py score <run_id>              # vs the current champion
uv run python code/evaluation/reward.py score <run_id> --against <run_id> --json
uv run python code/evaluation/reward.py ladder                       # every official run, oldest first
```

## The number

The host's formula, applied to our historical analogues
(see [the leaderboard check](../tests/test_reward.py)):

```
skill_m(origin, split) = 1 - MAE_m / MAE_m(30-year average)        pooled over cities, never clipped
points(split)          = 100 * mean over origins 1983/1990/1997 of mean over the 4 metrics
REWARD                 = points(final)        # leads 4-7: the analogue of 2022-2025, the final score
```

`live` points (leads 1–3, the analogue of 2019–2021) are reported and guarded. Points are on the same
scale as the leaderboard's Score column. **They are not a forecast of our rank.** The development
periods (1984–2004) had a weaker warming trend than 2019–2025, so every model scores near 0 here.

Verified: our pinned scorer reproduces all 20 displayed leaderboard Scores from their displayed MAEs
(±0.055, display rounding), and the board confirms per-metric skill is not clipped. The leaderboard is used
only for that arithmetic check, never for model choice (team policy and the official FAQ).

## The verdict (champion ladder)

Each run is compared with the current **champion** by a paired city bootstrap (2,000 draws, fixed seed).
The 210 cities are resampled with replacement, and each draw is applied to both runs, all origins, splits and metrics.

| Verdict | Meaning |
|---|---|
| `promote` | P(better on final) ≥ 0.95, better on ≥ 2 of 3 origins, live not clearly worse, no metric below −2 points. The run becomes the champion. |
| `within_noise` | Neither clearly better nor clearly worse. **Not progress**, even when the reward is higher. |
| `worse` | P(worse on final) ≥ 0.95. |

The ladder replays every official run on the locked suite in time order, starting from
`control:average30` (reward 0). It is deterministic, so anyone gets the same champion from the same runs.
This is the Ladder mechanism (Blum & Hardt, 2015): many agents retrying the same three folds cannot
push the bar up on noise, because only clear wins move it.

**Optimise in this order:** the verdict first, then the 90% interval's lower bound, then the reward.
A higher point estimate that stays `within_noise` should not be reported as an improvement.

## Limits

- The bootstrap measures noise across cities, not across climate regimes. Three reused periods remain
  the main risk; the 2-of-3-origins rule only partly covers it. Nearby cities are correlated, so the
  interval is, if anything, too narrow.
- The assessment origins (2004, 2011) stay sealed. Only the integrator opens them, once, for the chosen model.

## State on 2026-11-08 (eval-v1, suite-v3)

The champion is still `control:average30`. Placeholder v0.3–v0.9 score +0.17 to +0.93 final points,
all `within_noise` (P(better) 0.56–0.84). v0.9's gain is all temperature, CDD and hottest day
(precipitation equals climatology), and it loses on the 1990 origin (−2.1). On the live analogue, v0.3–v0.9
are clearly better (≈ +2.2 to +3.1, P = 1.00 for v0.9).

## Enhanced inputs: `enhance-v2` is the default (from 2026-11-08)

At the enhancement station's request, newer models that use external context default to **`enhance-v2`**
(enhancement commit `4c4317e`). It contains static city facts (elevation, coast distance, WorldCover land cover) and the ERSSTv5 Niño 3.4,
North Pacific and North Atlantic SST indices, with per-origin anomalies, January 1950 to December 2018.

```bash
uv run python scripts/line.py pull enhance-v2      # canonical, hash-checked copy in data/processed/releases/enhance-v2/
uv run python scripts/line.py score --name Placeholder_vX --adapter <file.py:forecast> --assets evaluation/enhance-v2-assets.json
```

- [`enhance-v2-assets.json`](enhance-v2-assets.json) declares `static/city_static.csv` (static context) and
  `ocean/origin-O/{annual,monthly}.csv` as weather inputs with `information_end_year = O`, scoped to their
  own origin. It covers 1983, 1990 and 1997, plus 2004 and 2011 so a freeze can reuse it. Declaring them does
  not open the assessment. A model with its own weights or settings copies these entries into its manifest.
- Every run pins the release per file: the asset binding stores each file's SHA-256, and `line.py` matches the
  canonical `data/processed/releases/enhance-v2/` paths to the release to tag the run's lineage. Bind
  those paths, not the friendly `enhanced/` copy, which the line does not recognise as a release.
- **Runs that use `enhance-v2` are `exploratory`** (`enhance-v2 is not gate-eligible`). The ERSSTv5
  reconstruction vintage is not certified for every early origin. They get a reward and a verdict, which
  `reward.py score` labels as informative, but they **cannot become champion**. `enhance-v1` (static facts only)
  is gate-eligible if a model needs only static context and must stay official.
- Nothing about the targets, scoring arithmetic, sealed runs or assessment policy changes.
