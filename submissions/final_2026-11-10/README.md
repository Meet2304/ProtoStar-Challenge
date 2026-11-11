---
pretty_name: "Protostar Climate Challenge Submission: Team Placeholders"
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- climate
- forecasting
- protostar-climate-2026
configs:
- config_name: default
  data_files:
  - split: test
    path: submission.csv
---

<p align="center">
  <img src="./assets/protostar-header-v0.1.png" alt="ProtoStar Challenge" width="100%">
</p>

<h1 align="center">Team Placeholders: Protostar Climate Challenge Submission</h1>

<p align="center"><i>Climate Risk Intelligence | Team Placeholders</i></p>

**Negative result:** no

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made as of the end of 2018 (no data that reveals 2019–2025 weather), for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence).

<p align="justify">
The ProtoStar Challenge, hosted by Carnegie Mellon University, explores how machine learning can improve climate forecasting using historical weather records from 210 U.S. cities. The challenge focuses on predicting four key climate indicators: average annual temperature, total annual precipitation, maximum daily temperature, and cooling degree days, translating historical climate data into actionable insights for urban resilience and adaptation.

As Team Placeholders, we are developing an iterative, data-driven forecasting pipeline that combines statistical baselines with machine learning models to improve prediction accuracy. Our workflow is built around Python and uv for reproducible dependency management, Google Colab for collaborative experimentation, and MLflow for tracking model performance, parameters, and artifacts. Through continuous evaluation and refinement, we aim to develop reliable city-level forecasts and translate our findings into a practical climate adaptation recommendation.
</p>

## At a glance

- **Forecast (Placeholder_F4):**
  - **Temperature:** two warming-pace hypotheses, one following IPCC AR6 forcing and one the recent pace, plus a learned per-city correction.
  - **CDD:** recomputed from each city's daily temperatures shifted by the forecast warming.
  - **Precipitation and hottest day:** the 30-year mean, a neighbour term and a shift toward the median.
  - **Backtest:** on rolling origins 1990–2017 (data up to 2018 only), it scores 8.59, against 7.51 for our previous model (the 30-year average scores 0 by construction). Details: *Method*.
- **Challenge data:** we used all four subsets and cleaned them (impossible and stuck readings, outliers, station changes); the full documentation is in [`documentation/`](documentation/README.md).
- **Outside data:**
  - IPCC AR6 forcing, used as a forecast input.
  - El Niño and other ocean indices, taken from Internet Archive copies captured before 2019; tested, and left out of the forecast.
  - 2020 Census population, used for the policy.
- **Policy Idea:** a yearly "cooling outlook" so cities size heat plans to the forecast, not the old 30-year normal. Weighted by population, the forecast cooling need is 206 °F-days a year above 1989–2018 (216 million people); 84 million of them live in the ten metro areas with the largest total rise.
- **Reproduce:** `pip install -r code/requirements.txt`, then the four commands in *How to Reproduce*, which rebuild `submission.csv` byte for byte.
- **Reading guide:** *Method* starts with the submitted model; the earlier versions after it are kept for the record.
- **Provenance:** this dataset is [`Meet2304/protostar-placeholders`](https://huggingface.co/datasets/Meet2304/protostar-placeholders) at commit `099a52d` (Placeholder_F4, identical `submission.csv`) plus Kohav's outside-data files, F4's processed daily data, the team's cleaning documentation and pipeline from GitHub, and the documentation updates described under *AI Tools and Agents*.

## Team

| Name | Hugging Face username | University, Program |
|---|---|---|
| Meet Bhatt | Meet2304 | Carnegie Mellon University, MSAIE-ETIM |
| Kohav Dey | DragonDare | Carnegie Mellon University, MISM |
| Ananya Gahalaut | Vyaghra | Carnegie Mellon University, MSAIE-IS |
| Achintya Gahalaut | agahalau | Carnegie Mellon University, MSAIE-IS |

## Method

**Current version: Placeholder_F4** (`code/make_submission.py` → `code/f2/f4_model.py`; model card [`docs/models/placeholder-f4.md`](docs/models/placeholder-f4.md)). Forecasts are made as of 2018-12-31 from the challenge data and one physical driver series that ends in 2018.

| Metric | Forecast for 2019–2025 |
|---|---|
| Average temperature | Two hypotheses about the warming pace, weight 1/2 each. *Forced response:* three members; each fits the city's record to a forcing curve known in 2018 (IPCC AR6 anthropogenic forcing, or a 10-year lagged response of it or of total forcing), uses the national median slope, takes the mean of the last 15 years of the detrended record as the city's level and extends the curve as a straight line (0.48 to 0.58 °F per decade). *Recent pace:* the v3.8 / v9-S rule (0.80 °F per decade). Both add a fraction (0.24, estimated from earlier hindcasts) of 2018's departure to the 2019 forecast. A **learned city correction** is then added: a robust (Huber) regression, fitted on years up to 2018, of how far a city's later years ended up from its 15-year mean on how the city has moved against its ten nearest neighbours over the last 1, 3, 5, 10 and 15 years. Its national median is zero, so it only moves cities against each other. |
| Cooling degree days | Recomputed from the definition. Each recent year's observed daily mean temperatures (challenge `daily` subset, cleaned by `code/f2/f2_daily.py`) are shifted by the forecast change (pace hypothesis plus city correction) and `CDD = Σ max(0, T − 65)` is recomputed, then averaged over 10 to 15 base years. This keeps the convex response of CDD to warming. |
| Total precipitation | 30-year mean + a neighbour-graph term (as in v9-S) + a shift toward the median: the score is a mean absolute error, whose best point forecast is the median, and yearly precipitation is right-skewed. Shift = −pooled skewness × standard deviation / 6, centred nationally, with a scale fitted on earlier hindcasts. |
| Hottest day | Same construction as precipitation, with skewness pooled over 25 neighbours. |

Cities without a 30-year baseline (49 rows, not scored) keep the v9-S value. The script rebuilds `submission.csv` byte for byte (SHA-256 `8344e656…3bc37`).

**Compared with v9-S** (the previous submission): temperature is lower by 0.17 °F on average (per city −5.1 to +1.0 °F; the −5.1 is Belvidere, which has no 30-year baseline and is not scored), CDD by 69 °F-days (53 in 2019 to 85 in 2025), precipitation by 0.06 in and hottest day by 0.01 °F.

**Validation and why it was chosen (November 10):**
- **Bench:** rolling origins 1990–2017 (train on years up to the origin, forecast the next seven years, all seven years pooled as on the organisers' updated final board). F4 scores 8.59, v9-S 7.51 and the best earlier team model (v10b-E) 7.88; 44 models are ranked in `reports/f4/BENCH.md`.
- **Where the gain is:** +1.08 over v9-S (90 % interval [+0.85, +1.37] from resampling 5° regions). It is concentrated in origins 1990–2003 (+2.05); it is small on 2004–2011 (+0.10) and 2012–2017 (+0.15), and an interval over four seven-origin blocks, [−0.42, +2.59], includes zero.
- **Against the team's earlier promotion rule** (plan v8 §2, leads 4–7, late window with an interval excluding zero): F4 does not pass it; its recent-window gains have intervals that include zero. Meet chose F4 as the submission on the all-years bench above.
- **Limits:** all eras were visible during development, so there is no untouched holdout; the pace weight 0.5 is a stated bet, set after the bench had been seen; the hottest-day skewness multiplier sits at its upper bound. Details: `docs/models/placeholder-f4.md` §6, `reports/f4/EVALUATION.md`, independent audit `reports/f4/REVIEW_audit.md`.

**Placeholder_v9-S (previous submission, rebuilt by `code/make_submission_v9s.py`, SHA-256 `d1873055…7e5f`)** is Placeholder_v3.8 (below) with one change, in CDD: a better estimate of how many °F-days CDD gains per °F of warming.

- **v9-S's CDD response:** `r = 0.75·k + 0.25·d`. It replaces v3.8's response in both CDD terms below.
  - **d (physical):** the city's average number of days a year with a daily mean above 65 °F, over 1989–2018, valid CDD years only, from the challenge's daily data. A uniform +1 °F adds 1 °F-day on each such day.
  - **k (statistical):** v3.8's CDD-on-temperature regression, averaged over the city and the 15 cities nearest in both distance and d.
  - **How 0.25 was picked:** on origins ≤ 2004, leads 4–7.
  - **Fallback:** cities with fewer than 10 valid years keep v3.8's response.
  - **Origin of the idea:** it came from the planning council (plan v9, `docs/plans/plan-v9-lineup.md`), and two training stations built it independently.

Placeholder_v3.8, which v9-S keeps for everything except that response:

| Metric | Forecast for 2019–2025 |
|---|---|
| Average temperature | The city's mean of its last 15 years (2004–2018), carried forward at **1.75 ×** the warming rate shared by all cities (the median of the cities' 1969–2018 least-squares slopes). v3.6 and v4 used 1 ×. The 50-year median slope lags the recent warming rate; the multiplier was chosen on origins ≤ 2004 only |
| Cooling degree days | v3.6's CDD, plus the city's CDD response to annual temperature (fitted on the city and its 15 nearest neighbours) times the extra temperature warming above. No new tuned number. v3.6's CDD is v4's forecast (last 10 years, shared relative 30-year rate) plus **half of any shortfall against a warming trend**: `clip(v4 + 0.5 · max(trend − v4, 0), 0)`, where the trend is the city's 1989–2018 mean plus the same CDD response times the national 30-year warming rate |
| Total precipitation | The city's 1989–2018 average plus **Placeholder_v1.4's neighbour-graph term**: a centred shift that moves a city relative to its neighbours, never the national level. Its two coefficients are refitted on pseudo-origins up to 2014, leads 4–7, with data up to 2018 only |
| Hottest day | Same as precipitation: 1989–2018 average plus the graph term |

Precipitation and CDD are clipped at 0.

**Inputs:**
- the challenge `annual_indices` table up to 2018-12-31 (`data/annual_indices_train_to_2018.parquet`, official 10-day validity rule);
- city coordinates from `data/city_info.csv`;
- per city-year counts of days with both Tmax and Tmin and of days with a mean above 65 °F (`data/daily_warm_days.csv`), built by `code/prepare_warm_days.py` from the challenge `daily` subset up to 2018-12-31.

There is no external data, no pretrained model and no randomness (seed 42). The script rebuilds `submission.csv` byte for byte (SHA-256 `d1873055…7e5f`).

**Compared with v3.8:** only CDD changes, by +2.4 °F-days on average (+1.9 in 2019 to +2.9 in 2025; per city −90 to +41). v3.8 itself was higher than v3.6 by 0.30 °F (2019) to 0.51 °F (2025) for temperature and by 34 to 56 °F-days for CDD.

**How v9-S was chosen (November 9–10, plans v8 and v9):**
- **Promotion rule (plan v8 §2):** the final ranking scores 2022–2025 only, which is leads 4–7 from 2018. So a candidate must:
  - beat v3.8 at leads 4–7 on origins 2004–2011 and on origin 2011;
  - beat it on the late window (origins 2012–2014, leads 4–6) with a 90% city-bootstrap interval excluding 0;
  - lose at most 0.10 on 1990–2003;
  - have every tuned number chosen on origins ≤ 2004.
- **Plan v8 (rejected):** six candidates, all re-run unchanged in the submission session (`docs/plans/v8-analysis/v8_compare.txt`). One retuned the warming rate, one computed CDD from daily temperatures, and the others changed the level window, used regional slopes or used homogenised input. None passed except a homogenised input that moved the 2018 forecast by only 0.009 °F.
- **Plan v9:** an independent planning council of two separate reviewers (Codex `gpt-6.1-sol` and Grok `grok-4.7`) screened precipitation, hottest-day, new model-family and CDD ideas. Only the CDD response survived.
- **The two builds:** the K station (`Meet/train`) and the 04_ii station (`Meet/train-dl`) built it independently. Both were re-run here unchanged, and they give identical forecasts. Comparison in `docs/plans/v9-analysis/v9_compare.txt`.

| Leaderboard-style score (0–100) | 2004–2011, leads 4–7 | Late 2012–2014, leads 4–6 | Origin 2011, leads 4–7 | 1990–2003, leads 4–7 |
|---|---:|---:|---:|---:|
| Placeholder_v3.8 (previous submission) | 9.675 | 18.754 | 20.060 | 4.772 |
| **Placeholder_v9-S (submitted)** | **9.863** | **18.924** | **20.264** | 4.793 |
| Difference (90% city interval) | +0.189 (+0.104 to +0.295) | +0.170 (+0.094 to +0.254) | +0.204 | +0.021 |

**Robustness and limits:**
- v9-S wins at all 8 origins in 2004–2011.
- Its late-window interval stays above 0 under a 5° spatial-block bootstrap.
- Leads 1–3 also gain about +0.10.
- **Known limits:** the gain is small, about +0.2 points. The late window had been read in earlier plans. Both council members estimate only a 15–25% chance of a final score above 20, because the 2022–2025 weather matters far more than this change.

**Placeholder_v3.8 (previous submission, rebuilt by `code/make_submission_v3_8.py`, SHA-256 `7b799ac9…72e5`), as described in the table above.**

**How v3.8 was chosen (November 9, plan v7, `docs/plans/plan-v7-warming-rate.md`).** All earlier candidates shared v4's temperature forecast. Three training stations each built one change of plan v7: Placeholder_v3.7 (temperature rate × 1.75), Placeholder_v3.8 (v3.7 plus CDD following temperature), and two principled replacements for the multiplier (a recency-weighted rate and a quadratic trend). Every candidate was rerun unchanged through `code/bench_adapters.py`, on origins 1990–2011 and on a new **late window, origins 2012–2017** (every lead ≤ 2018). Paired city bootstrap, 1,000 draws; full output in `reports/bench_v38_selection.txt`.

| Leaderboard-style score (0–100) | Origins 2004–2011 | Origins 1990–2003 | Origin 2011 | Late 2012–2017 | Leads 4–7, 2004–2011 |
|---|---:|---:|---:|---:|---:|
| Placeholder_v3.6 (previous submission) | 7.41 | 4.00 | 10.68 | 12.92 | 8.50 |
| Placeholder_v3.7 (temperature rate × 1.75) | 7.54 | 4.09 | 11.88 | 14.22 | 8.95 |
| v3.7 with a quadratic trend | 7.56 | 4.10 | 12.20 | 14.21 | 9.06 |
| v3.8 with rate × 1.5 | 7.97 | 4.05 | 12.04 | 14.34 | 9.41 |
| **Placeholder_v3.8 (rate × 1.75, submitted)** | **8.08** | 4.04 | **12.54** | **14.82** | **9.68** |

- **Promotion rule (plan v7 §4):** better on 2004–2011, on origin 2011 and on the late window with a 90% interval excluding 0; at most 0.10 worse on 1990–2003; every tuned number chosen on origins ≤ 2004.
- **v3.8 against v3.7:** +0.54 on 2004–2011, +0.60 on 2012–2017 (90% interval +0.37 to +0.81), +0.66 on 2011, −0.05 on 1990–2003. It passes. Against v3.6 the late-window gain is +1.90 (+1.45 to +2.34).
- **The multiplier:** 1.75 maximises temperature skill on origins ≤ 2004 (v3.7's declared rule). The K station's rule (full points on origins ≤ 2004, ties within 0.01 to the smaller value) picked 1.5, which led 1.75 by 0.002. We used the plan's definition, v3.8 = v3.7 + CDD follow-through. The late window had already been seen when this was decided, so it is not a blind test of that choice.
- **Known weaknesses:** v3.8 is much worse than v3.6 at origin 2007 (0.43 against 2.66 points) and slightly worse at origins 2002–2003; its CDD skill in 1990–2003 drops from 1.80% to 1.66%. It is a bet that warming continues at about the 2005–2018 pace.

**Placeholder_v3.6 (previous submission, rebuilt by `code/make_submission_v3_6.py`, SHA-256 `b2a69a69…1494`):** v4 with the half-warmth CDD term above and the neighbour-graph term on precipitation and hottest day; temperature as v4.

**How v3.6 was chosen (November 9).** Three training stations independently built "v4 plus the graph term on precipitation and hottest day" (Placeholder_tf1, plan I's K1 and Placeholder_v3.5); all three give identical scores. Every candidate was rerun, unchanged, through `code/bench_adapters.py` (origins 1990–2011, data ≤ 2018, paired city bootstrap with 1,000 draws):

| Leaderboard-style score (0–100) | Origins 2004–2011 | Leads 1–3 | Leads 4–7 | Origins 1990–2003 | Origin 2011 |
|---|---:|---:|---:|---:|---:|
| Combined model v4 (previous submission) | 6.53 | 4.92 | 7.58 | 3.50 | 10.44 |
| tf1 = K1 = v3.5 (v4 + graph term) | 6.83 | 5.26 | 7.82 | 3.89 | 10.51 |
| tf3 (tf1 + regional warming rates) | 6.83 | 5.24 | 7.87 | 3.80 | 10.55 |
| tf10 (tf1 + boosted hottest-day correction) | 7.02 | 5.28 | 8.22 | 3.81 | 10.88 |
| K3 (CDD half v4, half Placeholder_v0.3)* | 7.23 | | | 4.25 | 9.75 |
| **Placeholder_v3.6 (submitted)** | **7.41** | **5.73** | **8.50** | **4.00** | 10.68 |

\* Plan I's own run.

- **v3.6 against tf1/K1/v3.5:** +0.58 on 2004–2011 (90% interval +0.45 to +0.72), +0.11 on 1990–2003, and +0.17 on 2011 (+0.06 to +0.30). It is the only candidate that passes the three-part rule against them.
- **v3.6 against tf10:** +0.39 on 2004–2011 (+0.25 to +0.53) and +0.19 on 1990–2003 (+0.07 to +0.33). It loses on 2011 by 0.21.
- **tf10 fails the rule against tf1** on 1990–2003. Its gain comes from a boosted learner tuned on this period.
- **Nested check:** picking on origins ≤ 2004 alone also gives v3.6.
- **Mean skill over origins 2004–2011:**

  | Metric | v4 | v3.6 |
  |---|---:|---:|
  | Temperature | 12.1% | 12.1% |
  | Precipitation | 0.0% | 0.5% |
  | Hottest day | 0.0% | 0.7% |
  | CDD | 14.0% | 16.4% |

- **Comparison files:** `reports/bench_v36_selection_table.csv`, `reports/bench_v36_selection_pairs.csv` and `reports/bench_v36_selection_per_origin.csv`.
- **Development history:** the DL station's screens P–U on `Meet/train-dl` (`docs/models/placeholder-dl.md`).

**Known weakness of v3.6:** the DL station rated its robustness "thin".
- Its 1990–2003 margin depends on origins 2000–01.
- The 0.5 weight matters: at weight 1, 1990–2003 drops to 3.62.
- All of 1990–2011 has now been used for development, so no untouched pre-2019 test period remains.

**Combined model v4 (previous submission, rebuilt by `code/make_submission_v4.py`):**
- **Temperature and CDD:** as above, without the CDD trend term.
- **Precipitation and hottest day:** the city's 1989–2018 average.

**Why v4 replaced v1.4, and why Placeholder_v1.4 was a mistake.** We chose Placeholder_v0.1–v2.7 on three historical forecasts made as of 1983, 1990 and 1997.
- **The test windows favoured the average.** Warming was weak in those windows, because the 30-year windows include mid-century cooling. "Stay close to the 30-year average" won there, so every Placeholder version stayed anchored to it (v1.4 adds at most +0.5 °F).
- **The real forecast is different.** It is made from 2018, in a period of strong warming.
- **The new test bench** (`code/tournament.py`) scores every model the leaderboard's way from **every origin 1990–2011**:
  - each origin forecasts the next 7 years: leads 1–3 resemble 2019–2021, leads 4–7 resemble 2022–2025;
  - every origin and test year is inside the training data;
  - the origins nearest 2018 (2004–2011, test years 2005–2018) get the most weight.

**Selection rule.** Everything uses data up to 2018. No 2019–2025 observation and no leaderboard score was used to choose.
- A metric keeps the choice from Kohav's combined model v3 unless a candidate beats it on all three checks: mean skill over origins 2004–2011, mean skill over origins 1990–2003, and skill on origin 2011 alone.
- **Only average temperature changed:** a 15-year level instead of 10 years, and a 50-year shared slope instead of 30 years.
- **Candidates tried for each metric:**
  - levels: the 30-year mean, the mean of the last 5, 10 or 15 years, a level built from the national anomaly plus half the city's own, and medians over 10–30 years (MAE is minimised by the median);
  - slopes: none, the city's own 30-year slope, shared 30-, 50- and 70-year slopes, and a shared relative slope;
  - each level and slope pair alone and mixed 50/50 with the 30-year mean.

| Leaderboard-style score (0–100, mean over origins) | Origins 2004–2011 | Leads 1–3 | Leads 4–7 | Origins 1990–2003 | Origin 2011 |
|---|---:|---:|---:|---:|---:|
| 30-year average | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30-year linear trend | 2.14 | 3.06 | 2.02 | 0.46 | 2.31 |
| 10-year mean (Ocarina of Ten) | 3.67 | 3.16 | 4.36 | 1.81 | 5.48 |
| Placeholder_v1.4 (previous submission) | 3.56 | 3.31 | 3.65 | 4.01 | 3.17 |
| Combined model v1 | 5.05 | 4.47 | 5.77 | 3.27 | 7.59 |
| Combined model v2 | 5.61 | 4.77 | 6.62 | 3.58 | 9.33 |
| Combined model v3 | 6.03 | 4.93 | 7.27 | 3.58 | 10.10 |
| **Combined model v4** | **6.53** | 4.92 | **7.58** | 3.50 | **10.44** |

**Mean MAE over origins 2004–2011:**
- **Average temperature:** v4 1.281 °F; v3 1.316; v1.4 1.382; 30-year average 1.463.
- **CDD:** v4 and v3 170.1; v1.4 183.1; 30-year average 197.9.
- **Precipitation and hottest day:** v4 equals the 30-year average (6.018 in and 2.685 °F).

The full tables are `reports/tournament_leaderboard.csv`, `reports/tournament_mae_recent.csv` and `reports/tournament_metric_summary.csv`. Every candidate, origin and metric is in `reports/tournament_scores.csv`.

**Limits.**
- **Overlapping origins.** Each origin tests 7 years, so origins 1990–2011 hold only about three independent periods.
- **Why the rule is conservative.** Picking each metric's best candidate on origins 2004–2011 alone would have scored 7.03 there. But the same procedure, run on origins up to 2004, scored worse than v3 on origin 2011 (7.3 vs 10.1). Maximising each metric over many candidates overfits.
- **Precipitation and hottest day** stayed at the 30-year average in v4 because no candidate in that family beat it consistently. v3.6 adds the graph term instead.
- **Not used:** station-change information, ENSO (Kohav tested it and it did not help), and external data of any kind.

**Earlier versions.**
- **Placeholder_v0.1–v2.7:** neighbour-graph and shrinkage models. See `code/placeholder_v1_4.py`, `code/make_submission_v1_4.py`, [`docs/models/placeholder.md`](docs/models/placeholder.md) and [`submissions/Placeholder_v1.4/`](submissions/Placeholder_v1.4/README.md).
- **Combined models v1–v3:** `code/combo_model.py`; v2 and v3 are on Kohav's `kohav` branch.
- **The challenge's starter random forest:** `code/starter_model.py`; worse than the 30-year average on three of the four metrics.

### Outside data we tested (Kohav)

None of this changes the forecasts above: `submission.csv` is Placeholder_F4 byte for byte (SHA-256 `8344e656…3bc37`).

- **El Niño adjustment, tested and left out.** We joined NOAA's Oceanic Niño Index (ONI, `data/ocean_indices.csv`) to the city records by year and shifted each city's first forecast year by its El Niño sensitivity, fitted on earlier years only and averaged over the city and its 8 nearest cities.
  - Over 38 one-year-ahead forecasts (1981–2018, `reports/enso_rolling_test.csv`), it cut first-year errors by about 1.2% for temperature and 1.5% for CDD.
  - But it was better in only 21 and 23 of the 38 years, and it lowered the combined model's 4-window backtest Score from 5.4 to 5.1 (big misses after strong La Niña years, e.g. the forecast for 2011). By our selection rule it stays out.
  - Code: `code/enso_test.py` and `code/combo_model.py` (`USE_ENSO = False`) on the `kohav` branch of [`Meet2304/protostar-challenge`](https://github.com/Meet2304/protostar-challenge/tree/kohav) (commit `4cbbf9e`).
- **Other ocean cycles.** The Pacific Decadal Oscillation (PDO) relates only weakly to the next year (median correlation about +0.18 for northern temperature, +0.13 for southern precipitation), and the Atlantic Multidecadal Oscillation (AMO) hardly at all (|median r| ≤ 0.11), so neither was tried as an input. For comparison, the year-end ONI relates to the next year's temperature in northern cities at about +0.27 (El Niño winters are mild in the northern U.S.) and to precipitation in southern cities at about +0.14 (1951–2015, each city's departure from its previous 10 years).
- **Fixed geographic traits** (vegetation, soil, distance from the sea, relief, latitude). Cities that warmed fastest in 1959–1988 were not the fastest in 1989–2018: the correlation of city warming rates is −0.08, and latitude against the 1989–2018 temperature trend is −0.03. So a trait that does not change cannot predict which cities warm faster, and we did not use them as forecast inputs. Tree canopy and cool surfaces appear in the Policy Idea instead.
- **Population (2020 Census)** is used only for the Policy Idea, to say how many people live where the forecast rise is largest.

## AI Tools and Agents

- **Tools and models:** Claude Code (Claude Opus 5.5 and Claude Fable 5.2, partly run through T3 Code); Claude in the Claude app (Claude Opus 5.5); Codex (GPT-6.1-sol); Grok (Grok 4.7); Cursor Composer 2.5 (plan review). They wrote code, ran tests and drafted text; every forecast value comes from the code in `code/`. **No pretrained forecasting model** is used by Placeholder_F4.
- **What they did:** read the challenge rules and submission template; set up the repository layout, `code/AGENTS.md` / `CLAUDE.md`, the data-rule guard, `prepare_data.py`, `starter_model.py` (from the challenge's starter code) and the backtest; drafted this README and the data card. Downloaded only the challenge training dataset (data up to 2018).
- **Combined model:** Claude (Anthropic, via the Claude app) wrote `code/combo_model.py`, ran the four-window backtest and drafted this Method section, working only from `data/annual_valid.csv` (data up to 2018).
- **Exploration (`notebooks/Meet/01-dataset-structure.ipynb`):** Claude Code (Claude Opus 5.5) wrote the dataset-structure and data-quality notebook, working only from the challenge training data (up to 2018). A person should review the cleaning rules in its section 10 before they are used for a model.
- **Cleaning pipeline (`notebooks/Meet/02-cleaning-daily-annual.ipynb`, `notebooks/Meet/meet_cleaning.py`):** Claude Code (Claude Opus 5.5) designed and wrote the cleaning of the `daily` and `annual_indices` subsets: deleting impossible and stuck readings, checking outliers against neighbouring cities, rebuilding the yearly table from cleaned days, filling missing days and empty years for inputs only, adjusting for station changes, and a four-origin backtest (all years up to 2018) to choose which inputs to use. It used only the challenge training data (up to 2018); `submission.csv` was not changed. A person should review the isolated outliers in `notebooks/Meet/qc/daily_outlier_decisions.csv`, the thresholds at the top of `meet_cleaning.py`, and the station-change adjustment before it feeds a model.
- **Cleaning of all four subsets (November 8):** Claude Code (Claude Opus 5.5, through T3 Code) planned the cleaning of `daily`, `annual_indices`, `city_info` and `coverage_and_stations`; had the plan reviewed over four rounds by Sol (GPT-6.1-sol) and Grok (Grok 4.7) until both rated it at least 90 (final 95 and 97; [`notebooks/Meet/plan/`](notebooks/Meet/plan/cleaning-plan.md)); and implemented it as `scripts/02_clean_data.py` (logic in `src/protostar_challenge/cleaning.py`, 17 tests in `tests/test_cleaning.py`), writing `data/processed/`. Documented step by step in [`documentation/`](documentation/README.md). Only the challenge training data (up to 2018) was used; every backtest fold was rebuilt from data up to its origin; official labels are unchanged (`= data/annual_valid.csv`); `submission.csv` was not changed. A person should review the isolated outliers in `data/qc/daily_outlier_decisions.csv`, the adopted station steps in `data/processed/qc/homog_steps.csv`, and the input choice (only Avg Temp ships a cleaned input; [`documentation/06-input-selection.md`](documentation/06-input-selection.md)) before a model uses it.
- **What a person reviewed:**
  - **Meet Bhatt** directed every agent task, set the data rules and stopping points, and chose the decision rules. He reviewed the head-to-head results that picked Placeholder_v1.4 (November 8), and reviewed F4's bench and evaluation results before choosing F4 as the submission (November 10).
  - **Kohav Dey** chose the outside sources to try and ran the outside-data, policy and rebuild scripts on his laptop. On November 10 he read and agreed with the Policy Idea, *Outside data we tested*, the outside-data parts of *Data* and his Feedback items, and confirmed the rebuild's SHA-256.
  - **Not reviewed line by line by a person:** the model code (`code/f2/`), which was instead checked by the station's look-ahead test, an independent Grok audit (`reports/f4/REVIEW_audit.md`) and the clean-room rebuild below.
- **External-data research (November 7):** Codex (GPT-6.1-sol, through T3 Code) reviewed the official challenge rules and rubric, the current model and stored backtests, and primary documentation for adjacent climate datasets; wrote [`reports/adjacent-datasets-research.md`](reports/adjacent-datasets-research.md) and a judging-weight visualization. No external weather observations were downloaded, no new forecasting experiment was run, and no submission values were changed. A person should review source eligibility, licenses, and the proposed experiments before implementation.
- **Internal evaluation platform (November 7):** Codex (GPT-6.1-sol, through T3 Code) built the historical evaluation CLI, model adapters, run registry, local HTML report, rubric self-review and contract tests. It used the already-downloaded pinned training data through 2018, verified the published scoring source and FAQ, and ran the reference methods and current combo model on five seven-year historical tasks. No post-2018 observations were downloaded, no model was selected from official-year scores, and `submission.csv` was not regenerated. A person should review model adapters for origin-specific leakage, the score-mask interpretation and the limits of reused historical holdouts. See [`evaluation/README.md`](evaluation/README.md).
- **Opus-directed evaluator fixes:** Sol (GPT-6.1-sol through T3 Code, including a Sol helper for the nested selector) implemented the Opus audit's origin-aware guards, ranking on unclipped skill, nested method selection, separate development/assessment roles with committed freezes and access logs, period-level comparisons, explicit parameter/timeout failures, external-prediction provenance and run integrity checks. The new default suite remains labeled as reused historical evidence. Tests and real development runs are recorded in [`reports/evaluation-opus-fixes.md`](reports/evaluation-opus-fixes.md). No model was promoted to the official submission and `submission.csv` was not rewritten. A person should review the selected procedure, commit intended source before freezing assessment, and retain the exact artifacts with any performance claim.
- **Control baselines, scorecard and MLflow tracking (November 8):** Claude Code (Claude Opus 5.5) wrote a plan, had it reviewed over five rounds by Sol (GPT-6.1-sol) and Grok (Grok 4.7) until both rated it above 90 ([`docs/plans/baselines-and-tracking-plan.md`](docs/plans/baselines-and-tracking-plan.md)), and implemented it: seven fixed control baselines (`code/evaluation/controls.py`), a per-run characteristic scorecard (`scorecard.py`), a verified, reusable control panel (`panel.py`), an MLflow mirror of verified runs (`tracking.py`), the one-command loop `scripts/eval_model.py`, and 26 tests. The evaluator's owner (Sol, through T3 Code) integrated and committed it and built the first panel. Only the challenge training data (up to 2018) and development origins were used; no assessment scores were opened and `submission.csv` was not changed. A person should review the control definitions, the gates and the first results in [`docs/evaluation-baselines.md`](docs/evaluation-baselines.md).
- **Placeholder_v0.1 model training (November 8):** Claude Code (Claude Opus 5.5, through T3 Code) planned and wrote the first Placeholder model (`notebooks/Meet/placeholder_v0_1.py`, its tests, and `notebooks/Meet/03-placeholder-v0.1-training.ipynb`) and the model card [`docs/models/placeholder.md`](docs/models/placeholder.md). Grok 4.7 and Cursor Composer 2.5 reviewed the plan ([`docs/plans/placeholder-training-plan.md`](docs/plans/placeholder-training-plan.md)) over four rounds, ending at 93/100 each. It used only the suite's history up to each development origin (1983, 1990, 1997) and city coordinates; no post-2018 data was used, and `submission.csv` was not changed. A person should review the shrinkage rule, the pre-run inner-fold look described in the plan, and whether any later version earns a freeze and assessment.
- **Placeholder_v0.2–v2.7 and the submitted forecast (November 8):** Claude Code (Claude Opus 5.5, through T3 Code) wrote every later Placeholder version (plans A and B, reviewed by Sol (GPT-6.1-sol) and Grok (Grok 4.7): 92/94 and 93/93) and scored each on development origins 1983/1990/1997 only; v1.4 became the champion and v2.7 was worse. A separate Claude Code session (same model) re-scored v1.4 and v2.7 on suite-v3-r5 as labelled diagnostics (same ranking: the two suites hold the same data), moved v1.4 into `code/placeholder_v1_4.py` (byte-identical to the scored adapter), wrote `code/make_submission.py`, generated `submission.csv` from it (trained on the challenge data up to 2018-12-31; no 2019+ data), ran both validators, and drafted this Method section, the Policy Idea and the Feedback items. No value was written by hand, no model was chosen on 2019+ data or the leaderboard, and the one-shot assessment origins (2004, 2011) were not opened. A person should review the Policy Idea (agent draft), `code/make_submission.py` against `evaluation.core.build_fold`, and the evidence in [`submissions/Placeholder_v1.4/DISCLOSURE.md`](submissions/Placeholder_v1.4/DISCLOSURE.md).
- **Combined model v4, the new test bench and the v4 submission (November 9):** Placeholder_v1.4's live score fell below the linear-trend baseline. Meet then asked Claude Code (Claude Opus 5.5, through T3 Code) to find the best model and submit it overnight.
  - **Use of the leaderboard:** it only prompted a re-examination of the test design. It was never used to choose between models.
  - **What Claude did:** wrote `code/tournament.py` (rolling origins 1990–2011, all inside the training data); scored the baselines, Kohav's combined models v1–v3, Placeholder_v1.4 and a family of simple candidates; fixed a conservative selection rule; wrote `code/make_submission.py` (now `code/make_submission_v4.py`) for the winner (combined model v4); generated and validated `submission.csv`; rewrote this Method section; uploaded the dataset; and pressed Submit in the Space, as Meet had authorised.
  - **Data:** no 2019+ data was read, and no value was written by hand.
  - **For a person to review:** the selection rule, the tournament's candidate family, the decision to change only the temperature forecast, and whether the team agrees with replacing v1.4 (v4 is built on Kohav's v3).
- **Placeholder_v3.6 and this submission (November 9):** the training stations, each running Claude Code (Claude Opus 5.5, through T3 Code), built candidates on the new bench. Sol (GPT-6.1-sol) and Grok (Grok 4.7) critiqued them, and each screen was declared before it was run.
  - **The stations' candidates:** the TF series (`Meet/train-static-v1`), plan I's K series (`Meet/train`), and Placeholder_v3.5/v3.6 (`Meet/train-dl`; v3.6's CDD rule was proposed by Grok).
  - **What the submission session did:** reran every candidate unchanged through `code/bench_adapters.py` and compared them with a paired city bootstrap (`reports/bench_v36_selection_*.csv`). It found that tf1, K1 and v3.5 are the same model, and that v3.6 is the only candidate passing the three-part rule against them. It then copied the adapter into `code/placeholder_v3_6.py`, wrote `code/make_submission.py`, and verified `submission.csv` against the bench harness's own 2018 export and the DL station's file (SHA-256 `b2a69a69…`, identical). Finally it updated this README, uploaded the dataset and pressed Submit in the Space, at Meet's request.
  - **Who chose:** Meet chose v3.6 over tf10 after seeing the comparison.
  - **Data and selection:** no 2019+ data was read, no value was written by hand, and the leaderboard was not used to choose.
  - **For a person to review:**
    - v3.6's "thin" robustness label: the 0.5 weight and origins 2000–01;
    - that all of 1990–2011 is now reused development data;
    - the CDD trend term in `code/placeholder_v3_6.py`.
- **Placeholder_F4 and this submission (November 10):**
  - **The build:** the F station (Claude Code, Claude Fable 5.2, branch `Meet/train-f2`) built the F series: data cleaning, forced-response members, CDD from shifted daily temperatures, the learned city correction, a single bench of all team models, the evaluation suite and the model card. A Codex `gpt-6.1-sol` delegate built the precipitation and hottest-day model (`code/f2/p_*.py`, `reports/f4/P_REPORT.md`) and ran challenger screens (`code/f2/h_*.py`). Codex delegates also downloaded the leading teams' public forecast files and summarised their methods. Grok 4.7 delegates audited F3 and F4 for look-ahead, correctness and overstated claims (`reports/f3/REVIEW_independent.md`, `reports/f4/REVIEW_audit.md`).
  - **The submission session** (Claude Code, Claude Opus 5.5, through T3 Code), at Meet's request:
    - merged `Meet/train-f2` into the submission branch;
    - made `code/make_submission.py` the F4 build (the station's `code/make_submission_f4.py`, writing `submission.csv`) and kept v9-S as `code/make_submission_v9s.py`;
    - rebuilt the forecast here and got the station's hash (`8344e656…`), ran `code/check_submission.py` and the station's look-ahead test (`code/f2/test_no_lookahead_f4.py`, pass);
    - updated this README and the data card, uploaded the dataset and pressed Submit.
  - **Data and selection:** no 2019+ weather data was read and no value was written by hand. No live score was used to choose a model, member or weight. The leaders' public forecast files were compared with ours forecast to forecast (`reports/f4/LEADERBOARD_COMPARISON.md`), with their live scores copied for context.
  - **For a person to review:**
    - `docs/models/placeholder-f4.md` §6: the pace weight is a stated bet, and there is no untouched holdout;
    - that the forecast-to-forecast comparison with the leaders (first written for F3) existed before F4's pace weight was raised from 0.25 to 0.5;
    - that F4 does not pass plan v8 §2, the rule that chose v9-S;
    - `reports/f4/P_REPORT.md` limits, and findings 1 to 3 of the F4 audit.
- **Placeholder_v9-S (November 9–10, plans v8 and v9):** at Meet's request, the submission session (Claude Code, Claude Opus 5.5, through T3 Code) did the following:
  - wrote plan v8 (`docs/plans/plan-v8-final-leads.md`), which judges candidates on leads 4–7 because the final score covers only 2022–2025;
  - re-ran the stations' plan v8 candidates unchanged and found that none qualified;
  - started a planning council with no access to the repository: Codex `gpt-6.1-sol` and Grok `grok-4.7`, briefed separately, read-only, data ≤ 2018. Their records are in `docs/plans/v9-council/`.
  - **What the council found:** Sol proposed the CDD response. Grok tested new model families (analogues, a direct ridge, regime-chosen rates) and showed that they under-warm.
  - **The builds:** the K station and the 04_ii station (Claude Code, Claude Opus 5.5) built v9-S independently, each committing its selection rule before running.
  - **What the submission session then did:**
    - re-ran both builds, which gave identical forecasts;
    - copied 04_ii's adapter into `code/placeholder_v9s.py`, changing only the input path;
    - added `code/prepare_warm_days.py` and `data/daily_warm_days.csv`;
    - verified that `submission.csv` equals the bench harness's 2018 export;
    - updated this README, uploaded the dataset and pressed Submit, at Meet's request.
  - **Data and selection:** no 2019+ data was read, no value was written by hand, and the leaderboard was not used to choose.
  - **For a person to review:**
    - the warm-day count as a CDD response;
    - the fallback for cities with fewer than 10 valid years;
    - that the late window had been read in earlier plans.
- **Placeholder_v3.8 (November 9, plan v7):** after v3.6's live score, Meet asked the submission session (Claude Code, Claude Opus 5.5, through T3 Code) to find out why and how to improve. It ran checks on data ≤ 2018 only, found that every candidate shared v4's temperature forecast, whose warming rate lags, wrote plan v7 (`docs/plans/plan-v7-warming-rate.md`, with a new late test window, origins 2012–2017) and sent one task to each training station.
  - **The stations' candidates:** Placeholder_v3.7 (`Meet/train-dl`), Placeholder_v3.8 (`Meet/train`), and the principled-slope variants (`Meet/train-static-v1`), each built by Claude Code (Claude Opus 5.5) with its multiplier or weight declared and chosen on origins ≤ 2004.
  - **What the submission session did:** reran all of them unchanged (`docs/plans/v7-analysis/v7_compare.py`, `reports/bench_v38_selection.txt`); found that the two v3.7 builds are identical and that v3.8 at 1.75 × passes the rule against v3.7; copied the K station's adapter into `code/placeholder_v3_8.py` with the multiplier set to 1.75; pointed `code/make_submission.py` at it; and verified `submission.csv` against the bench harness's own 2018 export of the station's adapter (maximum difference 0). It then updated this README, uploaded the dataset and pressed Submit in the Space, at Meet's request (2026-11-09 20:06 UTC).
  - **Who chose:** Meet asked for v3.8 to be submitted after seeing the comparison.
  - **Data and selection:** no 2019+ data was read, no value was written by hand, and the leaderboard was not used to choose.
  - **For a person to review:** the choice of 1.75 over the K station's 1.5 (explained under Method); that the late window 2012–2017 was used to screen ideas before the candidates were built; the CDD follow-through in `code/placeholder_v3_8.py`.
- **Outside data and policy numbers (Kohav, November 8–10):** Claude (Claude Opus 5.5, in the Claude app with a cloud workspace linked to Kohav's laptop) wrote `code/prepare_ocean_indices.py`, `code/enso_test.py`, `code/prepare_population.py` and `code/policy_cdd.py`, ran the El Niño test, and drafted *Outside data we tested*, the population part of the Policy Idea, the outside-data rows of the *Data* section and data card, and the matching Feedback items. Kohav chose the sources to try, ran the scripts on his laptop, and uploaded and submitted this copy.
  - **Data:** ocean indices only from Internet Archive copies captured on or before 2018-12-31; population is not weather data. No 2019+ weather data was read.
  - **Forecasts:** unchanged. This copy adds files to Placeholder_F4 as submitted at `Meet2304/protostar-placeholders` commit `099a52d`, and the policy numbers are computed from its `submission.csv`.
  - **For a person to review:** the use of archive captures as "data as of 2018"; the metro area (or county) as the unit of "people served by a station"; and that the station-change flags in `policy_cdd.py` were skipped here (they need `data/raw/`).
- **Agent instructions file:** [`code/AGENTS.md`](code/AGENTS.md) (the single agent instructions file; in the team's GitHub repository Claude Code loads it through a root `CLAUDE.md`, which is not part of this dataset). Some of its notes still name earlier models as current; the submitted model is Placeholder_F4.
- **Final review (November 10):** Meet's Claude Code (Claude Opus 5.5) ran a read-only review of this dataset: file-by-file diff against `Meet2304/protostar-placeholders@099a52d`, the data rule on every added file, a search for secrets, the submission requirements, and a clean-room rebuild of `submission.csv` (same SHA-256). Claude (Claude Opus 5.5, Kohav's session) then made the documentation fixes it suggested (this AI section, the review line, the Files table, the data card and the order of *How to Reproduce*); no code, data or forecast changed.

## Policy Idea

*Drafted with AI agents (Claude Code; population analysis with Claude) from the model's findings and `code/policy_cdd.py`; read and agreed by Kohav Dey on 2026-11-10.*

1. **What risk did we notice?** Cities plan cooling (cooling centres, heat-alert staffing, school and transit air conditioning, utility peak capacity) against 30-year "normals". Our backtests show that a 30-year average lags a warming climate, and our forecast (Placeholder_F4) puts 2019–2025 above the 1989–2018 normal in all 208 scored cities for average temperature (median +1.1 °F) and in 207 of 208 for cooling degree days (median +162 °F-days a year, +14%). Budgets sized to the old normal will be short in exactly the hot years they exist for, and the shortfall is largest where neighbouring cities show the same drift.
   - **Who is exposed** (`code/policy_cdd.py`, with the 2020 Census population of each station's metro area, or county outside metro areas, from `code/prepare_population.py`): across 198 areas holding 216 million people, the population-weighted forecast rise is **+206 °F-days a year**.
   - **Where the most people are affected:** the ten areas with the largest total extra cooling need (population × rise) hold 84 million people (39%): New York, Houston, Dallas–Fort Worth, Los Angeles, Washington, Chicago, Atlanta, Riverside–San Bernardino, Miami and Tampa.
   - **Where the change is largest for the place:** among cities with at least 500 °F-days a year in 1989–2018, the largest relative rises are in cooler places, where cooling has been a smaller part of planning: San Diego (+35%), Colorado Springs (+32%), Reno (+30%), then Muskegon, Burlington (VT), Beckley, Lansing, Pueblo, Albany and Asheville (+25 to +28%).
   - Tables: `reports/policy_cdd_by_area.csv`, `reports/policy_cdd_by_city.csv`.
2. **What would we do about it, and who would do it?** Each city's sustainability or emergency-management office would publish, every January, a one-page "cooling outlook": its CDD and average-temperature forecast for the next seven years (from public station records, as in this challenge), the gap to its 30-year normal, and the same for its neighbours within about 600 km. Cooling-centre budgets, heat-plan staffing and utility demand planning would be sized to the outlook instead of the normal. A regional council of governments or the state climatologist would run one shared script for all its cities, so small cities get it without staff time. States and federal heat-preparedness funds would rank cities two ways: by people affected (the large metro areas above) and by relative change (cooler cities whose buildings and plans assume little cooling).
3. **Why is it sustainable?** It needs only free, already-published daily station records and a small open-source script that runs in seconds; no new sensors or paid data. It updates itself each year as a new year of records arrives, and it is checked against reality every year: last January's outlook is compared with the observed year, so cities can see whether it beats the old normal before relying on it more.

## Feedback to the Ecosystem

- **What got in the way:** the challenge dataset is served from a CDN host (`us.aws.cdn.hf.co`) separate from `huggingface.co`, so sandboxed agent environments with a host allowlist failed to read the parquet files until that host was added. scikit-learn's `RandomForestRegressor` with `n_jobs=-1` gives slightly different predictions between runs (summing trees in thread order), which breaks byte-for-byte reproducibility unless prediction runs on a single thread.
- **What would fix it:** list every host the dataset is served from on the dataset card, for teams running agents behind an allowlist; mention the parallel-prediction issue next to the starter code.
- **Reused history is the main evaluation risk:** with only 1851–2018 to learn from, every team tests on the same few historical periods many times, so backtest gains are optimistic. Publishing a few fixed, pre-2019 held-out periods (with the answers hidden and a small daily scoring budget) would let teams check a model without touching 2019–2025.
- **Getting other agencies' data as of 2018:** the climate indices we needed (NOAA CPC's ONI, JISAO's PDO, NOAA PSL's AMO) are published only as files that run to the present, with no way to request data up to a date, so a team that follows the data rule strictly cannot call them. Internet Archive copies captured before 2019 worked, but coverage is uneven: NOAA PSL's own ONI and PDO files were last captured in early 2017, so we had to find other publishers (CPC, JISAO) with December 2018 captures. Each index also has its own layout (PSL year-by-12-month tables, CPC 3-month seasons, JISAO with `**` markers and partial years), so each needs its own parser.
- **Joining people to stations:** the stations come only with a GHCN ID, name and coordinates, with no state or Census code, and 55 of 210 sit outside the city they are named for (mostly airports), so a city-name join fails and we had to geocode the coordinates. The Census Bureau's data API now needs a key, while its Geocoder (no key) returns 2020 populations directly.
- **What would fix both:** a snapshot of common climate indices (ONI, PDO, AMO, NAO) cut at 2018-12-31 next to the challenge dataset, in one tidy layout (`index, year, month, value`); and each city's state and Census place/metro code in `city_info`.
- **Hard-to-beat metrics:** on total precipitation and hottest day, almost nothing beats the 30-year average, so those two metrics mostly add noise to the score; saying so in the rules (as the hosts partly do) helps teams spend time where skill is possible.

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/` | Scripts that rebuild `submission.csv` (see [`code/README.md`](code/README.md)) and the agent instructions |
| `data/` | The curated data our code uses; described in its data card, [`data/README.md`](data/README.md) |
| `reports/` | Backtest results written by the code: `f4/` (bench, evaluation and audits of the submitted model, Placeholder_F4), `f3/` (its predecessor), `bench_v9s_selection.txt` (the comparison that chose v9-S), `bench_v38_selection.txt` and `bench_v36_selection_*.csv` (earlier ones), `tournament_*.csv` (v4's tournament), `placeholder_v1_4_selection.json` and older backtests |
| [`submissions/Placeholder_v1.4/`](submissions/Placeholder_v1.4/README.md) | Evidence for the previous model (Placeholder_v1.4): run ids, per-origin and per-metric scores, comparisons, and the full disclosure |
| Outside data (Kohav) | `code/prepare_ocean_indices.py` → `data/ocean_indices.csv` and `data/ocean_indices_sources.csv`; `code/prepare_population.py` → `data/city_population.csv`; `code/policy_cdd.py` → `reports/policy_cdd_by_area.csv` and `reports/policy_cdd_by_city.csv`; `reports/enso_rolling_test.csv` (El Niño test). Analysis only: none feeds `submission.csv` |
| `scripts/`, `src/`, `tests/` | The team's data pipeline (`scripts/01_download_data.py`, `scripts/02_clean_data.py`, logic in `src/protostar_challenge/`, tests in `tests/`), copied from the team's GitHub repository ([`Meet2304/protostar-challenge`](https://github.com/Meet2304/protostar-challenge), branch `main`). Not needed to rebuild `submission.csv`: `code/download_raw.py` does the one download F4 needs |
| `data/processed/f2/` | F4's processed daily data, as built for the submission: `daily_features.parquet` (cleaned daily temperatures summarised per city and year), `cdd_grid.npz` (CDD response tables) and `daily_qc_report.json` (what the cleaning changed). Rebuilt by `code/f2/f2_daily.py` and `code/f2/f2_cddgrid.py`; described in [`data/README.md`](data/README.md#processed-daily-data-for-placeholder_f4) |
| Not included | `data/raw/` (the challenge's raw subsets, downloaded at the pinned revision by `code/download_raw.py`) is not copied. The team's cleaned dataset (the rest of `data/processed/`, including `data/processed/qc/homog_steps.csv`) is rebuilt by `scripts/02_clean_data.py` and not copied; F4 does not read it. `code/enso_test.py` (the El Niño test) is on the GitHub `kohav` branch. Notes linked from `docs/evaluation-baselines.md`, `evaluation/README.md` and `reports/eval-v1-official-baselines.md` (`ASSETS.md`, `team-mlflow-guide.md`, `ASSESSMENT_HANDOFF.md`) were internal and are not published. The working notes of plans v7–v9 (`docs/plans/plan-v7-*`, `plan-v8-*`, `plan-v9-*`, `docs/plans/v*-analysis/`, `docs/plans/v9-council/`) and `docs/models/placeholder-dl.md` stayed on team members' machines and are not published; their conclusions are in *Method*. The station-inventory snapshots of `evaluation/independent/` are listed in its manifests but not copied |
| `notebooks/` | Each team member's exploration notebooks (not needed to reproduce `submission.csv`) |
| [`documentation/`](documentation/README.md) | **Data cleaning documentation**: index of every flaw found in the four training subsets and the strategy applied to it (`daily`, `annual_indices`, `city_info`, `coverage_and_stations`, station adjustment, input-selection backtest, outputs, review history) |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

1. Get the files with `hf download DragonDare/protostar-placeholders-f4 --repo-type dataset --local-dir protostar-f4` (with `git clone`, also run `git lfs pull`, or the parquet and npz files stay pointers). Install Python 3.13 and the packages: `pip install -r code/requirements.txt` (`scipy`, used by `code/f2/p_model.py`, comes with scikit-learn).
2. Build the submitted model (Placeholder_F4):
   - `python code/download_raw.py` (raw challenge data to `data/raw/`, pinned revision, ends 2018-12-31, about 1 minute; or point the `F2_RAW` environment variable at an existing copy);
   - `python code/f2/f2_daily.py` and `python code/f2/f2_cddgrid.py` (cleaned daily features and CDD response tables in `data/processed/f2/`, about 1.5 minutes);
   - `python code/make_submission.py` writes `submission.csv` byte for byte (SHA-256 `8344e656…3bc37`, about 1 minute). It reads `data/annual_indices_train_to_2018.parquet`, `data/city_info.csv`, `data/forcing_ar6_to_2018.csv`, the cleaned daily features and, for the 49 unscored fallback rows, Placeholder_v9-S (`code/make_submission_v9s.py`, which also reads `data/daily_warm_days.csv`);
   - `python code/f2/test_no_lookahead_f4.py` checks that no input after the origin changes the forecast (about 4 minutes).
   The full F-series bench and evaluation: `docs/models/placeholder-f4.md` §7. `python code/bench_adapters.py run --name X --adapter FILE:FUNC` and `summary` rerun the bench; `python code/tournament.py --with-v14` reruns v4's tournament.
3. `python code/check_submission.py`: checks `submission.csv` like the challenge Space.
   Checked on 2026-11-10 from this dataset as uploaded, twice: on Kohav's laptop, and in a clean-room rebuild (fresh download, new virtual environment) by Meet's independent review. Both gave SHA-256 `8344e656…3bc37`. The rebuilt `data/processed/f2/` files match the shipped ones (`cdd_grid.npz` differs only in compression; its arrays are identical).
   The bench reruns (not the submission build) still refer to folders on the F station's machine in `code/f2/f2lib.py`, `code/f2/a_repro.py`, `code/f2/a_extra.py` and `code/f2/f4_zoo.py`; point them at your copy.
4. Optional, analysis only: `python code/prepare_ocean_indices.py` and `python code/prepare_population.py` rebuild the outside data (both need internet: the Internet Archive and the Census Geocoder); `python code/policy_cdd.py` recomputes the Policy Idea numbers from `submission.csv`.

Earlier models: `python code/make_submission_v9s.py` (Placeholder_v9-S, the previous submission; `python code/prepare_warm_days.py` rebuilds its `data/daily_warm_days.csv`); `python code/make_submission_v3_8.py` (Placeholder_v3.8); `python code/make_submission_v3_6.py` (Placeholder_v3.6); `python code/make_submission_v4.py` (combined model v4); `python code/make_submission_v1_4.py` (Placeholder_v1.4); `python code/prepare_data.py` rebuilds `data/annual_valid.csv` and `data/city_info.csv`; `python code/combo_model.py` (previous submission) and `python code/starter_model.py` write their backtests to `reports/` and overwrite `submission.csv`.

**Evaluation platform (for testing new models, not needed for the submission):** start with the [evaluation platform guide](evaluation/README.md).
`uv run python code/evaluate.py prepare` builds version-2 historical tasks with declared phases;
`uv run python code/evaluate.py run --name my-model` evaluates nested method selection on development origins;
`uv run python code/evaluate.py report` writes the local experiment dashboard.
Teammates can instead score their own CSV predictions or plug in a Python adapter.
Imported predictions require a provenance manifest. Assessment runs require a
committed model freeze; all current historical periods remain reused evidence.
The official FAQ prohibits choosing models by scoring variants against public
2019–2025 observations, so this platform's internal labels end by 2018.

## Data

**Analysis only:** a preregistered pre-2019 station benchmark supplements the reused city backtests; see [selection, provenance and limits](evaluation/independent/README.md). It does not feed the submitted forecasts.

- **External data (forecast input):** `data/forcing_ar6_to_2018.csv`, IPCC AR6 WGI Chapter 7 historical effective radiative forcing, 1750–2018 only (the team's `enhance-v3` release; source, commit and license in [`data/README.md`](data/README.md#global-radiative-forcing-enhance-v3)). It is a physical driver, not a weather observation. It was published in 2021, so it is a later estimate of the past; the F station measured that replacing it by a fit with no external data changes the forecast by 0.11 °F and 13 °F-days on average.
- **Forecast input:** `data/annual_indices_train_to_2018.parquet`, the challenge's `annual_indices` subset, byte-identical to the pinned revision below (SHA-256 `0b5f5fa9…e411`); the challenge `daily` subset up to 2018-12-31, cleaned by `code/f2/f2_daily.py` (9 days with Tmax < Tmin, 462 climatological outliers and 32 spikes set to missing; the processed result F4 reads is in `data/processed/f2/`); `data/daily_warm_days.csv` (v9-S fallback rows only). `data/city_info.csv` gives the list of 210 cities.
- **Analysis only:** the leading teams' public forecast files (forecasts, not observations), compared with ours in `reports/f4/LEADERBOARD_COMPARISON.md`.
- **External data, analysis only (Kohav):** details in [`data/README.md`](data/README.md#outside-data-analysis-and-tested-inputs).
  - **Oceanic Niño Index (ONI)**, NOAA Climate Prediction Center; **Pacific Decadal Oscillation (PDO)**, University of Washington JISAO (Mantua et al., 1997); **Atlantic Multidecadal Oscillation (AMO, unsmoothed)**, NOAA Physical Sciences Laboratory (Enfield et al., 2001). ONI was tested as a forecast input and left out; PDO and AMO were analysis only.
  - **How we kept them inside the data rule:** the live files run to the present and cannot be cut at 2018, so `code/prepare_ocean_indices.py` never calls them. It downloads the Internet Archive's newest copy of each file captured on or before 2018-12-31, and checks every row with `assert_no_post_cutoff()`. This also gives the values as published at the time. Provenance: `data/ocean_indices_sources.csv`.
  - **How they join the challenge data:** by year. Each index is one monthly series for a whole ocean, so we take its value at the end of a year (ONI: mean of its Aug–Oct and Sep–Nov seasons) and relate it to each city's next year.
  - **2020 Census population** around each station (county, metro area and city), from the U.S. Census Bureau Geocoder, joined by each station's latitude/longitude. All 210 stations matched: 165 in a metro area, 45 county only; 155 lie inside city limits. Used only for the Policy Idea.
- **Curated data:** see the data card in [`data/README.md`](data/README.md) (each file, its source, license and preparation).

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1

<p align="center"><i>Made on Earth, by Humans.</i></p>
