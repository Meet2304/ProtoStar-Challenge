# 05 — Station adjustment (homogenisation)

Each city's record joins several stations over time. When the station moves (often from a city office to an
airport), the record can jump although the climate did not — and a trend fitted across the jump mixes two stations.
The adjustment puts old readings on the level of the station in use at the end year. **Inputs only:** the official
labels keep the jumps, as the 2019–2025 answers will. Plan reference: §4 H1–H4.

## H1 Difference series with a fixed reference

For a city, a variable (annual QC `Avg Tmax`, `Avg Tmin`) and a change cluster: the reference is the
inverse-distance-weighted mean of the neighbours' departures from their own mean, using a **fixed neighbour set**
(neighbours with ≥ 5 valid years on both sides; ≥ 2 required) and only the window years in which the city and *every*
neighbour of the set have a value, with fixed weights. `diff = city − reference`. Shared weather cancels; a level shift
in `diff` is the station. Because membership and weights are fixed, the reference cannot shift when a neighbour
enters or leaves the record.

## H2 Clusters, test and offsets

- **Windows:** up to 10 valid years before the cluster's first change and after its last, not crossing other clusters
  or the end year, overlap years (Abilene) excluded; ≥ 5 valid years each side, else "unestimable".
- **Step:** the coefficient s in `diff = a + b·(year − mean) + s·after` — trend-adjusted, so a relative trend between a
  city and its neighbours is not written into the offset.
- **Test:** t = s / SE, with the OLS standard error inflated for year-to-year autocorrelation,
  `SE·√((1 + r)/(1 − r))`, r = lag-1 autocorrelation of the city's within-segment residuals (consecutive years inside
  one station segment only), capped at 0.8. Median r: 0.64 (Tmax), 0.72 (Tmin) — the inflation roughly doubles the
  standard error, which is why far fewer steps pass than in v0.
- **Critical value from placebos:** a placebo change on 1 July of every year ≥ 10 years from any real change, with the
  same window rule; per variable and window class (short: < 10 years on a side; full) t_crit = max(2, 95th percentile
  of placebo \|t\|). In the 2018 run every t_crit stayed at 2.0 and placebos passed it 1.7–2.2 % of the time (about
  18,000 placebos per variable and pass).
- **Adopt** iff \|t\| ≥ t_crit and \|s\| ≥ 0.25 °F; two passes (the second with adjusted neighbours); then the
  return-pair rule ([page 03](03-city-info.md), C5).
- **Offsets per day:** the sum of the adopted steps of all later clusters; inside a cluster, days from P also get that
  cluster's step, days from Q do not, and days from a third station are removed (only if the step is adopted). Every
  day gets exactly one offset. All annual inputs (`Avg Temp`, `Max Tmax`, `cdd`) are rebuilt from the adjusted days, so
  non-linear metrics are adjusted correctly.

## Result (2018 run)

| | Tmax | Tmin |
|---|---:|---:|
| clusters tested | 229 | 229 |
| estimated | 197 | 196 |
| unestimable (too few years / neighbours) | 30 | 31 |
| same station before and after | 2 | 2 |
| **adopted** | **42** | **61** |
| median adopted step | −1.65 °F | −3.01 °F |

Most adopted steps are city-office → airport moves, where nights become cooler outside the city's heat island: e.g.
Marquette 1979 Tmin −6.1 °F, Houston 1969 Tmin −5.4, Chattanooga 1940 Tmin −5.1 (t = −10.2), Baltimore 1950 Tmin −4.0,
Toledo 1939 Tmin −4.4. Largest Tmax step: Redding 1944 +7.2 °F. Adopted in 1989–2018: Kennewick → Pasco 1998 (Tmax
+1.6, Tmin −3.4), Crescent City 2000 (Tmax −3.7), Omaha 1994 (Tmin −1.4), Hastings 2002 (Tmin +1.3). Every test is in
`data/processed/qc/homog_steps.csv`; the calibration in `qc/homog_placebo_calibration.csv`.

## H3 Order when filling and adjusting are combined

QC days → steps from QC annual values → offsets added to observed days (transition days removed) → normals and
neighbour anomalies computed from the adjusted days and missing days filled on the reference level → the plain filled
series is the same estimates moved back by each day's offset.

## H4 Late changes

At every backtest origin the adjustment runs with changes ≤ the origin and the same rules, so changes with only 5–9
years after them are exercised in the backtest as they are in production. Riverside 2009-01-01 and Natchitoches
2010-08-01 can only be estimated in the 2018 run.

## Limitations

- A change shared by all neighbours at once (a network-wide instrument change) cancels in `diff` and cannot be seen by
  any pairwise method. Undocumented changes are not detected in this version.
- Station locations before the current station are unknown; neighbour distances use the current site.
- Precipitation is not adjusted: annual totals vary by ~20 % year to year, so ratio steps under ~10 % cannot be
  detected in 10-year windows.
