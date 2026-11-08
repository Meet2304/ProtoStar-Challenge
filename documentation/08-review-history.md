# 08 — How the cleaning strategy was reviewed

The strategy was written as a plan, scored by two independent reviewer models, revised, and implemented only after
both scored it at least 90/100. The final plan is
[`notebooks/Meet/plan/cleaning-plan.md`](../notebooks/Meet/plan/cleaning-plan.md); every review is in
[`notebooks/Meet/plan/reviews/`](../notebooks/Meet/plan/reviews/).

- **Author:** Claude Code (Claude Opus 5.5), from Meet's request and the v0 cleaning of `daily` and `annual_indices`
  (`notebooks/Meet/02-cleaning-daily-annual.ipynb`).
- **Reviewers:** Sol (OpenAI `gpt-6.1-sol`, through Codex in T3 Code) and Grok (xAI `grok-4.7`), each with read-only
  access to the repository and the raw data (up to 2018), asked to verify claims against the data. Rubric (100):
  detection methods 25, actions and evidence 20, leakage and target integrity 15, completeness across the four
  subsets 15, validation design 15, reproducibility 10.

| Round | Plan | Grok | Sol | Main changes that followed |
|---|---|---:|---:|---|
| 1 | v1 | 68 | — (usage limit) | labels kept official; inverted days swapped instead of deleted; fragment rule fixed (v0 had deleted 64 years of Huntsville); a selection rule that can reject a variant; ids/constants/row contract; AR(1)-adjusted step test; undocumented-break detection removed until specified |
| 2 | v2 | 89 | 82 | labels = published values (the `Avg Temp` rounding chain); nested chronological evaluation; `official` as a candidate; spatial block bootstrap; trend-adjusted step coefficient; complete cluster rules (same endpoints, third stations, one offset per day) |
| 3 | v3 | 93 | 89 | an audit that can pass (two-stage tolerance); forecaster chosen inside the nest; full official mask; one fill objective; hashes of the forecasting code; override evidence year; fixed realised reference; placebos with the real window rule; fragments per variable group |
| 4 | v4 | **97** | **95** | approved, no must-fix items; nice-to-haves taken before the final run (plan §10): absolute rain error in the bake-off, conditional nested check with `METHOD` fixed, last-year reference, filled-pair clipping, no label completion over transition years, acceptance tests |

## Findings the reviews added (all verified against the data)

- The published `Avg Temp` cannot be rebuilt exactly from days (rounding chain): the labels are taken as published and
  QC values use a delta form.
- v0's fragment rule deleted long real records (Huntsville 1895–1958, Akron 1888–1952, Wheeling 1879–1953).
- Deleting inverted days changed labels (Montgomery 1917 CDD, Portland 1999 Avg Temp); swapping does not.
- Repeated small rain amounts are not multi-day totals, as v0 assumed; they are kept on evidence instead.
- An iid t-test with 10 years a side adopted 283 of 382 steps; the autocorrelation-adjusted, trend-adjusted,
  placebo-calibrated test adopts 103 (42 Tmax, 61 Tmin).
- The team's forecaster (`METHOD`) was chosen on the same windows; the ship rule now needs a chronological check in
  which the forecaster is chosen from earlier data only — which is why CDD ships `official`.

## Acceptance tests

`tests/test_cleaning.py` (17 tests, synthetic data, run with `uv run python -m unittest tests.test_cleaning`): swap
and limits; stuck runs with varying, steady and sparse neighbours; missing is not dry for repeated rain; zero MAD;
leap day; fill keeps observed values and clips the filled member; empty-period removal and same-key merge; clusters
with same endpoints and overlap years; one offset per day on a multi-cluster city and transitions only when adopted;
injected steps recovered with 5+5, 5+10, 10+10 years; no bias on trend-only and uneven-neighbour series; low false
adoption; exact delta form; fragment rule; no infill in dropped years; the override evidence-year filter.
