# Supplementary geographic validation

The user confirmed on November 8, 2026 that the team had used only the
challenge dataset. This permits a new geographic holdout from other stations;
it does not make the challenge-city historical periods independent again.

All observed weather requests use the NOAA NCEI Daily Summaries service with
explicit dates **1950-01-01 through 2018-12-31**. Neither full-history station
files nor observations from 2019 onward are acquired. Station/inventory catalogs
are structural metadata, not weather observations. The API documentation is at
<https://www.ncei.noaa.gov/access/search/documentation/data-service/>.

## Frozen design and chronology

1. `preregistration-v1.json`, builder and model source pins were committed in
   `41d5966` before acquisition. Selection of 24 stations was committed in
   `0c9811e` before any external daily observations were fetched.
2. V1 failed its predeclared quality gate: **5 eligible stations versus 12
   required**. Its quality record is preserved. **No V1 model scores were
   computed.** Quality thresholds were not relaxed.
3. V2 is an explicit amendment based on that missingness failure, without
   model-performance information. `preregistration-v2.json`, a separate builder
   and the analysis runner were committed in `e38ba3a`. Selection was committed
   in `3433c6a`, before V2 observations were fetched.
4. V2 selects 48 USW/WBAN stations, eight per latitude/longitude stratum, by
   a fixed hash of station ID. It excludes challenge IDs, listed station names,
   locations within 100 km of challenge locations, and **every V1 station and
   its 100 km buffer**. USW stations are often, but not universally, airports.
   The inventory must cover all three required daily variables from 1950 to
   2018. All exclusions and quality failures are reported, without replacements.
5. Both versions require at least 20 valid baseline years in 1982–2011, six
   valid target years in 2012–2018, two live years and three final years for
   every metric. At least 12 stations across four strata must qualify. These
   completeness requirements are supplemental design choices; the scorer
   retains the official ten-day validity and ten-year baseline rules.
6. The primary method is `evaluation.nested:forecast`. Seven fixed controls
   and combo are secondary descriptions. Methods, parameters, aggregation,
   selection and source pins precede acquisition. All nine execution freezes
   precede any score. Every method is reported regardless of outcome.

## Reproduce and inspect

The builders refuse changed preregistrations, source pins, station selections,
cached responses and aggregation inputs. Outputs are exclusive and immutable.
The compressed structural snapshots and manifest preserve the original metadata
and challenge exclusion bytes. Run from the source checkpoint named in the
preregistration; later source edits intentionally fail verification.

```bash
uv run python scripts/build_independent_holdout_v2.py fetch --selection outputs/independent-v2/selection.json
uv run python scripts/build_independent_holdout_v2.py build --selection outputs/independent-v2/selection.json
uv run python scripts/evaluate_independent_holdout.py prepare
uv run python scripts/evaluate_independent_holdout.py run
```

The final command has an exclusive opening receipt and cannot silently repeat
the analysis. Existing copies should be inspected rather than rerun. Preparation
creates a 1990 development origin solely for the framework contract; **no
development model scores are opened**. The only scored origin is 2011 at the
new stations. Original challenge-city assessment remains separate.

## Interpretation and limits

This is one new geographic cohort over one historical forecast period. Regional
and temporal dependence remain; stations are not independent random trials.
Completeness filtering changes the population, WBAN stations need not represent
all 210 cities, and nearby regional weather can correlate with previously used
challenge observations. Listed station names and a distance buffer cannot prove
the absence of every undocumented historical component station.

Current NOAA archive revisions and QC can differ from the organizers' archive.
Single-station annual temperature aggregation is an explicit approximation to
the challenge's city composites, and physical station moves are unknown. Missing
values and QC flags are handled before annual aggregation; unknown station
history is recorded as unknown, never zero changes.

The framework conservatively keeps `history_status=reused_historical`; the
separate provenance describes the user-attested new station labels. This
benchmark supplements geographic evidence, **not private-key certification or
leaderboard prediction**. Once opened, it is no longer untouched. It must not
become another repeatedly consulted tuning set or promote a submission variant.
