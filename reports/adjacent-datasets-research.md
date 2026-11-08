# External datasets and a practical strategy for ProtoStar

Research date: November 7, 2026. This report examines the organizers' current rules and source code, the team's existing methods and backtest reports, and primary scientific and dataset documentation. It proposes experiments; it does not establish that any external dataset improves the team's forecasts. No external weather observations were downloaded for this research, no model was fitted, and no submission values were changed.

**My recommendation is to improve the way the model learns regional warming and cooling demand, then test a small number of additional datasets.** Start with station history, geography, and land cover. Test ocean temperatures before undertaking a large ocean-current project. Treat solar-cycle features as an optional, low-priority experiment. These priorities reflect the forecast horizon and the deadline, rather than a claim that the lower-priority processes are physically unimportant.

## 1. What this challenge actually rewards

The task is to predict four annual quantities for 210 U.S. cities for 2019–2025, as though standing at December 31, 2018. They are average temperature, total precipitation, the highest daily maximum temperature, and cooling degree days. Cooling degree days, abbreviated CDD, add up the amount by which each day's average temperature exceeds 65°F. A day averaging 80°F contributes 15°F-days. This measures temperature-related cooling need, rather than actual electricity use. [Official challenge description](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/main/content/about.md).

The live forecast score covers 2019–2021. The final forecast score covers 2022–2025, which are **four to seven years after the forecast origin**. A forecast origin is the date at which all input information is frozen. This is principally a problem of estimating annual climate conditions several years ahead. Forecasting a specific storm or the exact date of a future heat wave is a different problem.

Weather and weather-responsive inputs must end at 2018-12-31. Post-2018 ocean temperatures, soil moisture, snow, river flows, vegetation responses, air quality, and electricity demand cannot supply indirect answers. The rules expressly permit structural information such as census, land cover, canopy, elevation, and building stock from any year. Still distinguish a land-cover classification from a vegetation-health or greenness series: the latter can respond directly to drought. Post-2018 training data or an unknown training cutoff disqualify pretrained forecasting models as forecast inputs. Brainstorming and writing code with an LLM are allowed. [Official data rules and FAQ](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/main/content/data.md).

The authoritative deadline is **November 10, 2026, at 11:59 p.m. Eastern Time**. November 13 is the winners' announcement and CMU presentation date. Some broader launch articles describe the dates or prediction years differently; use the detailed challenge rules.

Winning also depends heavily on the research artifact. The current rubric assigns:

| Criterion | Weight |
|---|---:|
| Demonstrated reuse and extensibility | 25% |
| Interoperability and integration | 20% |
| Reproducibility and openness | 20% |
| Scientific, research, or societal value | 15% |
| Feedback to the ecosystem | 15% |
| Technical creativity | 5% |

Interoperability means making different data sources work together consistently. Reproducibility means another team can regenerate the results using the supplied code, data, and instructions. The forecast score and policy idea inform the value and creativity criteria; the score is not added as an independent points bonus. Reuse, integration, reproducibility, and feedback together account for 80%. A useful integration with a careful negative finding can therefore be valuable, although failed experiments alone do not guarantee a strong submission. [Current rubric in the organizers' configuration](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/main/config.py).

## 2. Why weather changes

These processes work together. Their physical importance does not automatically make their future states predictable.

| Process | Plain-language explanation | Implication for this task |
|---|---|---|
| Sunlight, latitude, and season | Earth's tilt and latitude change the amount and timing of sunlight. | Calendar and geographic information are known ahead of time. Annual averages hide seasonal differences. |
| Greenhouse gases | Gases such as carbon dioxide absorb and emit outgoing heat radiation, changing Earth's energy balance. | A slowly changing warming background can be more useful at long horizons than last year's weather anomaly. |
| Atmospheric circulation | Moving air transports heat and water. The jet stream is a band of strong winds high in the atmosphere that helps guide storms. | Circulation strongly affects individual seasons, but its exact state years ahead is difficult to forecast. |
| Ocean temperatures and currents | Oceans store heat; currents move it. Ocean conditions also alter where tropical rain and rising air occur. | Slow ocean changes can contain useful information, but the model must forecast their future evolution. |
| Soil moisture and vegetation | Evaporating water uses energy that would otherwise heat the surface. Plants also move water into the air. | Dry land can amplify hot conditions. Late-2018 land conditions cannot simply be assumed to persist through 2025. |
| Snow and sea ice | Bright snow and ice reflect sunlight. Melting them consumes energy and changes water availability. | They can matter for regional conditions, but their effects are seasonal and overlap other climate signals. |
| Mountains, coasts, and large lakes | Elevation changes temperature; mountains redirect moist air; water moderates nearby temperatures. | These are durable geographic differences that may help cities share information sensibly. |
| Buildings, pavement, and urban vegetation | Materials absorb and release heat differently, while shade and evaporation change local temperatures. | Relevant to station surroundings and urban adaptation; neighborhood conditions may differ from an airport record. |
| Aerosols and volcanic eruptions | Small airborne particles can reflect or absorb sunlight and alter clouds. Large eruptions can affect climate for years. | Past eruptions help explain historical variation, but the timing of future eruptions is not known at the forecast origin. |
| Solar activity | The Sun's energy output varies slightly over an approximately 11-year activity cycle. | This is a real influence, but it provides no established shortcut to accurate city forecasts several years ahead. |

For the physical mechanisms, see NASA on the [greenhouse effect](https://science.nasa.gov/climate-change/faq/what-is-the-greenhouse-effect/), NOAA on [ENSO's atmospheric connections](https://www.climate.gov/news-features/blogs/enso/how-enso-leads-cascade-global-impacts), NASA on [soil moisture and weather](https://science.nasa.gov/missions/smap/nasas-smap-may-clarify-link-from-wet-soil-to-weather/), the RAPID project's [ocean-circulation explanation](https://rapid.ac.uk/methodology), EPA on [vegetation and cooling](https://www.epa.gov/heatislands/benefits-trees-and-vegetation), and NASA on [aerosols](https://science.nasa.gov/science-research/earth-science/climate-science/aerosols-small-particles-with-big-climate-effects/).

Two distinctions matter throughout the research. First, an **anomaly** is the difference from a reference average, such as an ocean region being warmer than its historical average for that month. Second, a **teleconnection** is a relationship between conditions in distant places, transmitted through the atmosphere or ocean. A teleconnection changes the odds of an outcome; it does not guarantee that a particular city will be wet or hot.

## 3. Which datasets to investigate

All weather-related observation ranges below end at 2018 for this project, even when the provider continues the record today. These are candidate ranges, not records downloaded during this research. Availability does not imply that the data were equally reliable throughout their history. Confirm the exact edition, coverage, access requirements, and redistribution terms before incorporating a source.

### A. Station histories and regional observations: first priority

The challenge already includes `coverage_and_stations` and the station periods in `city_info`. Use these before downloading another dataset. The records join different stations over time. A station relocation or instrument change can produce an apparent temperature change that a trend model mistakes for climate change.

NOAA's [Historical Observing Metadata Repository and station histories](https://www.ncei.noaa.gov/products/land-based-station/station-histories) can provide further context. NOAA's [Global Historical Climatology Network–Daily, or GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), provides nearby station observations with quality checks. Request only dates through 2018. The challenge already derives from this observing network, so downloading the same stations again does not create independent evidence.

Useful derived inputs include a station-change flag, years since the last change, nearby-station temperature anomalies, and a regional warming estimate. A derived input is a variable calculated from raw data for the model. Compare nearby stations after accounting for elevation and local climate. Correct a discontinuity only when supported by metadata and a stable reference; an automatic adjustment can also remove real climate variation.

**Proposed test:** compare the present trend blend with a model that pulls unreliable city trends toward a regional trend. This is called partial pooling: each city retains its own information while borrowing strength from comparable cities. It should be tested even before external data are added.

### B. Elevation, coast distance, and land cover: first external integration

USGS [3D Elevation Program, or 3DEP](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services), provides terrain information. A digital elevation model is a grid whose cells store ground height. The approximately 10-meter product covers the continental states, Alaska, Hawaii, and U.S. territories. Coast and lake distance can be calculated from a documented shoreline dataset rather than inferred from a city name.

The USGS [National Land Cover Database, or NLCD](https://www.usgs.gov/centers/eros/science/about-annual-nlcd), describes land surfaces. Legacy products include historical map years such as 2001, 2006, 2011, and 2016. They use 30-meter cells. Useful variables include the proportion of developed land, tree cover where available, and impervious surface. Impervious surfaces, such as pavement and roofs, prevent water from soaking into the soil. Check each product's geographic coverage; continental-U.S. land-cover products need a documented fallback for other locations.

Calculate these around the actual observing station, not just the municipal center. A city record can represent an airport several miles from residential neighborhoods. Start with one or two specified buffer sizes, such as 1 and 5 kilometers. A buffer is the surrounding area over which a map is summarized. Match historical station locations where possible; current station coordinates may not describe earlier station segments.

Static geography is not a year-to-year weather forecast. If the model already gives every city its own average, elevation may add little to that average. Its better use may be choosing comparable cities or allowing warming and CDD responses to differ by geography. Historical urban development can add information beyond a fixed city average, but only if represented with appropriate map years. The challenge permits later structural maps; using a 2016 map to describe a city in 1980 nevertheless weakens a claim of historically realistic evaluation.

EPA distinguishes [surface temperatures from air temperatures](https://www.epa.gov/heatislands/measuring-heat-islands). A satellite's hot pavement measurement is not interchangeable with the station air temperature scored here.

### C. Ocean surface temperatures and compact climate indices: the best ocean starting point

NOAA's [Extended Reconstructed Sea Surface Temperature version 5, or ERSSTv5](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00927), supplies monthly ocean temperatures on a 2-degree latitude/longitude grid beginning in 1854. A 2-degree cell spans roughly 220 kilometers north to south. Later periods have better observing coverage. For an initial experiment, the period 1950–2018 is more practical than treating sparse nineteenth-century ocean estimates as equally reliable.

Start with a few regional averages instead of millions of ocean cells. NOAA's [climate-index catalog](https://psl.noaa.gov/data/climateindices/) documents useful summaries:

- **ENSO**, the El Niño–Southern Oscillation, is a coupled ocean-and-atmosphere pattern in the tropical Pacific. El Niño is its warm phase and La Niña its cool phase. **Niño 3.4** measures surface-temperature conditions in a specified part of the equatorial Pacific. ENSO influences storm tracks and seasonal U.S. temperature and rainfall probabilities.
- **PDO**, the Pacific Decadal Oscillation, summarizes a broad North Pacific surface-temperature pattern. Its name does not make it a clock that switches predictably every decade.
- **AMV**, Atlantic multidecadal variability, describes broad, slow North Atlantic temperature variation. Related published indices are often called AMO. Index definitions and treatment of the global warming trend differ; do not treat all versions as interchangeable or as proof of an independent natural cycle.

Candidate inputs are the last available seasonal Niño 3.4 anomaly, its trailing change, and a small number of Atlantic and North Pacific averages. Fit regional effects because the same ocean condition need not produce the same response everywhere.

The limitation is the horizon. NOAA describes a [spring predictability barrier](https://www.climate.gov/news-features/blogs/enso/spring-predictability-barrier-we%E2%80%99d-rather-be-spring-break): ENSO forecasts can lose accuracy when predicting across spring. A seasonal influence is therefore not evidence of reliable seven-year predictability. Test a model that allows the initial ocean anomaly's effect to diminish with lead time; estimate the rate from historical backtests instead of choosing a convenient disappearance date.

**Never train with the actual ocean index in the target year and quietly supply that value at prediction time.** For every simulated forecast origin, either predict future ocean conditions from earlier information, average over plausible future paths, or estimate the target directly from origin-time inputs. The last option is the simplest hackathon experiment.

### D. Ocean heat below the surface: interesting second-stage research

The Met Office's [EN4 dataset](https://www.metoffice.gov.uk/hadobs/en4/) contains subsurface temperature and salinity profiles, monthly analyses, and uncertainty estimates from 1900 onward. A profile records how conditions vary with depth. Salinity is the concentration of dissolved salt. Together with temperature it influences seawater density and circulation.

Heat stored below the surface can persist longer than an atmospheric disturbance. This makes subsurface ocean information scientifically relevant to longer forecasts. A candidate feature is an upper-ocean temperature or heat-content anomaly for a few predefined ocean regions, using only months through 2018. Proper heat content requires depth and volume weighting; averaging whichever profiles happen to be available is not equivalent.

The tradeoffs are substantial: early observations are sparse, uncertainty varies, and extracting physically meaningful summaries takes work. I would investigate this after the compact surface-temperature experiment, not before it.

### E. Ocean currents: physically important, lower hackathon priority

NASA's [Ocean Surface Current Analyses Real-time final product, OSCAR v2](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0), estimates global surface currents on a quarter-degree grid, with daily records starting in 1993. Its eastward and northward velocities could be summarized over ocean regions. The product combines observations and a diagnostic calculation; it is not a complete measurement of ocean circulation at every depth. Earthdata authentication is required for protected downloads.

Restricting it to 1993–2018 leaves about 26 calendar years. That provides little independent information for fitting relationships and evaluating several seven-year forecasts. A large grid does not solve the shortage of independent years.

The [RAPID array](https://rapid.ac.uk/about-us/history) measures Atlantic overturning circulation from 2004. The Atlantic Meridional Overturning Circulation, or AMOC, is the broad north–south movement of warmer upper-ocean water and colder deep water. It is not identical to the Gulf Stream or to OSCAR's surface currents. Only about 15 years remain through 2018, making an independent city-level forecasting experiment particularly difficult.

**Recommendation:** use surface-temperature summaries first. Add current measurements only if they improve forecasts beyond those summaries on identical test years. Otherwise the extra processing may duplicate information already captured by ocean temperatures.

### F. Soil moisture and snow: useful near-term candidates

NASA's [North American Land Data Assimilation System phase 2, NLDAS-2](https://ldas.gsfc.nasa.gov/nldas/nldas-2-model-data), provides modeled land conditions. Its record begins in 1979 at approximately one-eighth-degree spacing, with hourly and monthly outputs. See the [NASA overview of its coverage](https://ldas.gsfc.nasa.gov/sites/default/files/ldas/nldas/presentations/Mocko_NCR-Water-Symposium_2017-04-07.pdf). Soil moisture is estimated using weather inputs and land models; it is not simply a direct soil sensor reading everywhere.

[ERA5 monthly reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) offers a global alternative from 1940, with a quarter-degree atmospheric grid and variables including soil conditions, snow, pressure, wind, and humidity. A reanalysis combines observations and a physical model to reconstruct past conditions consistently. It is historical information, not a future forecast.

Use trailing, seasonally appropriate soil-moisture and snow summaries through each forecast origin. Dry ground can amplify heat because less incoming energy is spent evaporating water. Snow influences reflection and meltwater. However, a wet or dry condition at the end of 2018 does not establish the state of summer soil several years later. Test their effect by horizon and prioritize early leads. Avoid target-year soil moisture and snow, which respond to the weather and leak information.

### G. Pressure, winds, and humidity: strong mechanisms, limited long-range availability

NOAA's index catalog includes the North Atlantic Oscillation and Arctic Oscillation, which summarize large-scale pressure patterns, and the Pacific–North American pattern, which describes a broad atmospheric configuration. ERA5 supplies pressure and wind fields directly. Humidity is the water vapor in the air; it affects clouds, rainfall, nighttime cooling, and the heat people experience.

These are important for understanding weather. They are lower-priority inputs for a forecast made four to seven years in advance, because their future states are not supplied. Use historical seasonal averages, forecast-origin states, or explicitly modeled future distributions. Humidity is especially useful for a policy discussion about heat stress, but this challenge scores dry-bulb air temperature and CDD, rather than a combined heat-and-humidity index.

### H. Snow and sea-ice extent: exploratory

The [National Snow and Ice Data Center's Sea Ice Index](https://nsidc.org/data/seaice_index/data-and-image-archive) includes daily and monthly information beginning in late 1978. Sea-ice extent is the ocean area meeting the product's ice-coverage threshold. A historical seasonal Arctic-ice summary could be tested, but it can overlap the global warming signal and ocean conditions already in the model. Do not assume a simple one-directional relationship between a single Arctic index and every U.S. city's winter.

Use a product version and source granules that can be requested entirely through 2018. Do not download a present-day all-years CSV and then rely on deleting forbidden rows afterward.

### I. Greenhouse gases and global temperature: useful background, avoid duplication

NOAA's [Mauna Loa carbon-dioxide observing program](https://gml.noaa.gov/obop/mlo/programs/coop/scripps/co2/co2.html) begins in 1958. Carbon dioxide is an important physical driver of long-term warming. A historic concentration record through 2018, followed by a predeclared forecast or scenario, is different from supplying actual future concentrations. Year-to-year carbon uptake also responds to climate, so do not assume observed later concentrations are exempt from the data rule.

A historical global-temperature series can represent common warming. It must also be forecast beyond the origin; observed 2019–2025 global temperatures are prohibited inputs. NOAA's [current global-temperature product page](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp) documents newer editions that use neural-network reconstruction. The newest product is therefore not automatically the best choice under the pretrained-model rule: inspect reconstruction training and version provenance.

An archived [HadCRUT4](https://www.metoffice.gov.uk/hadobs/hadcrut4/) edition is one possible historical comparison: it was documented in 2012 and provides monthly records from 1850. The provider has superseded it and recommends newer data for general research. Its relevance here would be a clearly labeled historical-vintage experiment, rather than a recommendation to use an obsolete product for current climate assessment.

For the first implementation, a regional temperature signal derived from the supplied cities may capture much of this background without another download. Test whether global temperature or carbon dioxide adds information beyond that signal and a time trend. Strong correlations among steadily increasing series do not establish extra predictive value.

### J. Solar cycles: a legitimate but low-priority experiment

The Royal Observatory of Belgium's [SILSO monthly sunspot record](https://www.sidc.be/SILSO/infosnmtot) begins in 1749. Sunspots are dark regions on the Sun associated with magnetic activity. Their number is an indicator of solar activity, rather than a direct measurement of the energy reaching a city. Solar irradiance is the Sun's radiant power per unit area.

NASA describes approximately [0.1% variation in solar output over an 11-year cycle](https://science.nasa.gov/science-research/planetary-science/08jan_sunclimate/). Solar variability has physical effects, but NASA's [climate explanation](https://science.nasa.gov/earth/climate-change/what-is-the-suns-role-in-climate-change/) identifies its role in recent climate change as small compared with human-produced greenhouse gases.

Use historical sunspot data only through 2018 and estimate any subsequent cycle path without later observations. A simple repeating sine wave assumes a stable cycle length and amplitude that the real Sun does not exactly follow. A sunspot correlation with city temperature can be accidental or redundant with another time-varying process. Keep solar features only if their addition improves repeated historical forecasts, especially at leads four through seven.

The [13-month smoothed sunspot series](https://www.sidc.be/SILSO/newdataset) is distinct from the raw monthly series. Do not use a centered smooth at a forecast origin when its calculation needs future months. A trailing smooth uses only earlier months and avoids that information issue.

### K. Volcanic aerosols: explain uncertainty rather than predict eruptions

NASA explains how [large eruptions can change climate for several years](https://science.nasa.gov/earth/earth-observatory/volcanos-and-climate-change/). A historical aerosol series can help prevent a trend model from treating a temporary volcanic disturbance as a permanent change. Aerosol optical depth is a measure of how much airborne particles diminish light passing through the atmosphere.

For this deadline, treat eruptions mainly as a reason to communicate forecast uncertainty. A future eruption's date and magnitude are not known at the origin. Retrospectively inserting them into a historical forecast would make the test too easy. Effects of an eruption already observed before an origin may legitimately be carried forward with an estimated decay.

### L. Climate-model projections: an advanced, conditional extension

NASA's original [NEX-GDDP CMIP5 archive](https://www.nccs.nasa.gov/nex-gddp/) contains downscaled temperature and precipitation simulations from 21 climate models under two concentration pathways. Downscaling translates coarse global-model output to a finer grid. The [technical description](https://esgf.nccs.nasa.gov/esgdoc/NEX-GDDP_Tech_Note_v0.pdf) describes calibration against observations from 1950–2005. A concentration pathway is an assumed future evolution of greenhouse-gas concentrations.

A simulation of 2025 made without observing 2025 is conceptually different from 2025 observations. Nevertheless, eligibility depends on the precise archive, calibration inputs, and the organizers' treatment of that product. Verify these before downloading or proposing it as a compliant input. A recent file release alone does not prove leakage, and an old-looking filename alone does not prove eligibility.

Use such projections to investigate a forced warming response, meaning the climate response to persistent changes in energy balance. These simulations generally do not synchronize their internal weather fluctuations with the actual city weather of each calendar year. Average appropriately across models and time rather than matching an arbitrary simulated hot year to an observed year. Archive-wide bias correction can also use years later than an early backtest origin, so an existing downscaled product is not automatically valid for every historical fold.

This is a promising research direction, but it is more work to validate than the initial station, geography, and ocean-index experiments.

## 4. How to show that an added dataset actually helps

### Step 1: state a small, testable claim

For example: "Origin-time Pacific and Atlantic temperature summaries improve precipitation forecasts beyond the current model at leads four through seven." Define the target, city groups, inputs, forecast horizons, and primary comparison before running many alternatives. A feature is an input variable; a feature group is a related set, such as the ocean summaries.

Write down why the new variable should help, what it adds beyond current inputs, and how its value is obtained for a future prediction. If the future feature itself is unknown, its uncertainty is part of the forecasting problem.

### Step 2: keep the team's current model as the main comparator

The existing `code/combo_model.py` uses a half-trend, half-recent-average forecast for temperature and CDD, and a 30-year average for precipitation and hottest day. This is the model a new addition must improve. Also retain the official 30-year average, linear trend, and last-valid-year reference forecasts.

The existing `reports/backtest_combo.csv` and README show that simple methods are already competitive in the team's historical tests. These are stored results inspected during this research, not new experiments. They do not establish that precipitation or hottest-day forecasting is impossible to improve.

### Step 3: simulate the actual seven-year task

Use rolling-origin backtesting. This means choosing an earlier origin, fitting with information available by then, predicting the next seven years without updates, and repeating at other origins. Randomly distributing city-year rows between training and testing would let the model see future conditions and produce an unrealistic result. [Forecasting textbook discussion](https://otexts.com/fpptr/tscv.html).

One possible non-overlapping set is:

| Freeze inputs at the end of | Predict | Evaluate leads 4–7 separately on |
|---|---|---|
| 1983 | 1984–1990 | 1987–1990 |
| 1990 | 1991–1997 | 1994–1997 |
| 1997 | 1998–2004 | 2001–2004 |
| 2004 | 2005–2011 | 2008–2011 |
| 2011 | 2012–2018 | 2015–2018 |

These are proposed folds, not completed evaluations. Each external dataset must have enough preceding history. NLDAS, NLCD, OSCAR, and RAPID cannot all support every early fold meaningfully. For each source, report its usable origins and train-history length; compare both models on the same available years and cities. Do not interpret a different test period as an effect of the added source. Older and newer periods may represent different climate conditions, so show period-specific results as well as a pooled summary.

The team's current code forecasts eight years in its backtests. Retain those as historical diagnostics, but add seven-year tests and explicitly emphasize leads four through seven to match final scoring. Do not update an ocean index, soil state, or target history with actual values from inside a held-out forecast period.

### Step 4: fit all transformations using training information

At every origin, recalculate reference averages, trends, missing-value replacements, feature scaling, and any learned geographic groupings using training data alone. Scaling is converting inputs to comparable numerical ranges. Ocean-pattern extraction must also use training-only data.

Watch for reference periods such as 1991–2020 in a supposedly pre-2019 climate index. A reference average computed with future years can contaminate the feature. Reconstruct anomalies from allowed observations or use a documented compatible historical edition. Avoid centered running averages that cross the cutoff. Historical observations can also be revised; record the exact edition and determine whether reconstruction or pattern learning used forbidden later information. A dataset published recently is not automatically forbidden, but the dates of the underlying observations and any learned reconstruction matter.

The same principle applies to the supplied derived indices. Some warm-day percentages use a reference period extending to 1987, so they are not origin-time features for a 1983 backtest unless rebuilt from earlier daily records. Using the official target itself for evaluation is different from using a future-calibrated derived index as a predictor.

### Step 5: compare matched experiments

An ablation is a controlled comparison in which one component is added or removed.

| Run | Purpose |
|---|---|
| A: current combination model | Establish the actual starting point. |
| B: regional pooling, same existing data | Measure the benefit of changing the learning method. |
| C: B plus geography and land cover | Measure the incremental contribution of those data. |
| D: B plus ocean summaries | Measure the incremental ocean contribution. |
| E: B plus both groups | Check whether they complement or duplicate each other. |
| F: E with each group removed in turn | Check whether each still matters in the combined model. |

Use identical target rows, training windows, forecast horizons, random seeds where relevant, and comparable tuning effort. If the new dataset is missing for some cities, compare A and B on that same subset as well as reporting the full-coverage fallback. Otherwise an easier subset can masquerade as an improvement.

### Step 6: use the actual scoring rule

Mean absolute error, or MAE, is the average absolute difference between prediction and observation. For each metric:

`skill = 1 - model_MAE / thirty_year_average_MAE`

`Score = 100 * max(0, mean_of_the_four_metric_skills)`

Keep the four MAEs in their own units. Do not average Fahrenheit, inches, and degree-days directly. Compute each metric's MAE over its scored city-year rows before dividing by the reference MAE; averaging separately normalized city scores is a different evaluation. Negative skill on one metric reduces the average even when another metric improves. Retain the unclipped mean skill during development because the displayed zero can hide deterioration. [Official scoring implementation](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/main/scoring.py).

For illustration only, if precipitation MAE falls from 6.0 inches to 5.7 inches on an otherwise identical test, that is a 5% error reduction. If its reference MAE is also 6.0 inches, precipitation skill rises by 0.05. With the other skills unchanged, the unclipped overall score rises by 1.25 points. These numbers are invented to explain the arithmetic; they are not challenge results or a promised gain.

Use the organizers' missing-day rules and minimum baseline history. Predict every submission row, but exclude the prescribed unscored rows from evaluation. For CDD, inspect the number of days with both temperatures as well as the separate missing-temperature counts. The dataset documentation flags that these can differ. Keep the official scoring mask distinct from any stricter training-quality choices; do not silently redefine which rows are scored.

### Step 7: assess repeatability and practical value

Report gains by metric, forecast year, historical origin, region, and station-change status. In addition to the official pooled score, report how many cities improve, how much the worst affected groups lose, and whether a gain disappears outside one unusual year. These are diagnostics, not replacements for the official formula.

For each scored row calculate `absolute_error_old - absolute_error_new`. A positive value favors the new model. To assess uncertainty, resample whole years or short blocks of consecutive years while keeping cities from the same year together. This is a block bootstrap: repeated recomputation on temporally grouped samples gives an approximate range for the gain. Preserve pairing between models. Do not treat thousands of city-year rows as thousands of independent climate experiments; neighboring cities share weather, and a global index supplies only one value per time period.

With a few forecast windows, uncertainty will remain substantial. More overlapping origins provide diagnostics, not equally many independent confirmations. A narrow interval from an unsuitable resampling method is not persuasive evidence.

Choose a practical adoption rule in advance. For example, require positive average improvement on leads four through seven, consistency across a majority of usable origins, and no unacceptable regional losses. Decide what counts as an unacceptable loss in the metric's actual units. If the uncertainty range includes both improvement and deterioration, describe the evidence as inconclusive. Extra download cost, missing coverage, and irreproducible processing can outweigh a tiny uncertain gain.

### Step 8: keep model selection separate from reporting

Hyperparameters are modeling settings, such as how strongly city trends are pulled toward a regional trend. Select them using earlier training-era validation, then evaluate the selected procedure on later held-out years. This is nested evaluation: tuning happens inside the training portion of an outer test.

The existing four windows have already influenced the team's model choice, so they are not untouched final evidence. New folds overlapping those years also reuse historical information. Be transparent about this limitation, restrict the number of experiments, and freeze decisions before final reporting. A known limitation honestly described is better than calling reused history an independent test.

Do not tune against 2019–2025 observations or use live leaderboard differences as the feature-selection mechanism. The repository's agent instructions explicitly prohibit leaderboard tuning. All model-generated future numbers must continue to come from code.

### Step 9: separate predictive help from physical causation

An improved forecast shows incremental predictive value under the evaluation design. It does not prove that the new variable caused the outcome. Ocean indices can overlap each other or the warming trend. A model's feature-importance ranking or a visually impressive correlation is not a substitute for an ablation.

A useful diagnostic is to ask whether a supposedly important input adds value beyond a simple time trend and regional history. If many unrelated trending inputs all look helpful during training but fail historical forecasting, the model is probably exploiting shared trends or chance. Prefer a few physically motivated feature groups over a large search through arbitrary correlations.

## 5. An implementation order suited to the deadline

1. **Repair the comparison before expanding inputs.** Add seven-year forecasts and lead-four-to-seven reporting to the existing backtest design. Review station changes, missing days, and the CDD completeness issue. These changes can affect the reliability of every later experiment.
2. **Try regional pooling with the supplied data.** Compare it with the present city-by-city trend blend. This establishes whether additional modeling complexity helps before attributing a gain to an external source.
3. **Integrate one structural source.** Elevation is simple and durable. Land cover adds a stronger policy connection but more geographic and historical alignment work. Explain how station surroundings affect interpretation and use the data to help share information between comparable cities.
4. **Test a compact ocean-temperature group.** Start with a few monthly or seasonal regional means through 2018, rather than raw current maps. Publish the matched ablation and the result by forecast lead. Keep a negative result if the experiment is fair and informative.
5. **Improve CDD modeling with daily or seasonal information already supplied.** Warming a cold winter does not add CDD, while warming days near or above 65°F can. Consider separate seasonal warming patterns and a calibrated distribution of daily temperatures rather than converting the annual average to CDD with one formula. A distribution describes the range of plausible values and their probabilities. If simulating daily weather, preserve realistic sequences of hot days and take the annual totals from each simulated path. Evaluate a median annual forecast under MAE, and compare with the existing blend before adopting it. This is a proposal, not a tested improvement.
6. **Spend the remaining effort on a reproducible artifact and one specific policy idea.** Favor a complete, reviewable experiment over starting a large new ocean-current, solar, or deep-learning pipeline close to the deadline.

Temperature and CDD deserve the first modeling effort because the current project and the organizers' guidance identify them as promising targets. Precipitation and the hottest day remain worthwhile research targets, but do not force noisy year-specific adjustments simply to make forecasts look sophisticated.

## 6. Turn the work into a submission judges can use

Publish a small source register containing the provider, URL or DOI, edition, observation date range, retrieval date, license, preparation steps, units, geographic coverage, and whether the file is a forecast input or policy-only evidence. For gridded sources, record the station-to-grid matching method and any averaging region. Pin versions and save checksums, which are file fingerprints used to detect later changes. Include a small schema describing each column and its meaning.

Do not assume a government-hosted file has no citation or redistribution conditions. If redistribution is restricted, the official challenge permits an exact version and download instructions instead of copying the source. Keep credentials out of the repository. Make date-limited requests at the provider, then apply the existing `assert_no_post_cutoff()` guard to every admitted weather table. The current guard enforces the final 2018 limit; a backtest additionally needs its own earlier origin check.

Give reviewers one concise comparison table showing what each dataset added, what it cost to process, where it was missing, and whether the improvement survived the final-horizon tests. A failed download or conflicting metadata becomes useful ecosystem feedback when accompanied by the precise problem and a proposed fix. Do not claim an integration succeeded if only its catalog page was inspected.

A policy direction that fits the prediction task is **efficient, equitable cooling preparation**. Use the model's CDD outlook as a city-scale planning signal, then use building stock, population vulnerability, and land cover to identify where low-emission cooling upgrades or shade deserve investigation. The policy should identify an owner, financing or maintenance approach, and measurable outcomes. Temperature-based CDD is not household energy use, health risk, or a neighborhood heat map. Those conclusions require additional evidence. Tree shade, efficient building upgrades, and cleaner electricity have different mechanisms and local constraints; choose one focused intervention rather than promising benefits from all of them.

The strongest defensible story would be: **we combined public datasets under a clear information cutoff, tested whether each addition improved seven-year forecasts, documented both successes and limits, and linked the resulting evidence to a specific adaptation decision.**

## Research and AI-use record

Tool and model: Codex, GPT-6.1-sol, through T3 Code. Tasks: inspected repository instructions, the current model and stored historical backtests; verified live official challenge rules, rubric, and scoring source; researched primary dataset and mechanism documentation; prepared this report and a rubric visualization. External observation datasets were not downloaded or incorporated. No new forecasting experiment was run and no submission was generated or edited. A person should review source eligibility and licenses before downloading, verify the proposed experiment design, and review scientific and policy claims before adopting them in the submission.
