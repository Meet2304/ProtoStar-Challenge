# Internal evaluation validation — November 7, 2026

The implementation is documented in [the team guide](../evaluation/README.md).
It reproduces the published forecast-scoring function on historical tasks;
it does not reproduce the private 2019–2025 answer key or predict official rank.

## Verified scope

- **19 offline contract tests passed** with Python 3.13.1, pandas 3.1.1 and
  NumPy 2.5.4. The tests include hand-calculated scores, full metric weighting,
  negative skill and aggregate clipping, official-function AST identity,
  upstream-file fingerprints, randomized numerical parity, missing/duplicate
  predictions, invalid years/values, baseline eligibility at nine versus ten
  valid years, date guards, suite tampering, real subprocess adapters, external
  CSV evaluation, failure records, human rubric normalization, paired bootstrap
  identity and report injection escaping.
- The five configured seven-year tasks were built for all 210 cities from the
  already-cached pinned training `annual_indices`. Every fold contains a
  complete 5,880-row target grid. All observations and labels end by 2018.
- The last-valid-year, thirty-year-average and linear-trend reference methods
  were evaluated across all five tasks. The existing combo model was evaluated
  through the same adapter interface. These are validation runs, not a new
  model-selection decision.
- Repeating the combo adapter in fresh subprocesses produced **byte-identical
  prediction CSVs for all five folds**. Exporting those predictions and scoring
  them through the external-CSV interface produced the same complete summary.
- A paired whole-year bootstrap completed with 2,000 draws. Its assumptions and
  interpretation limits are documented in the guide and output JSON.
- Suite verification passed for fingerprint
  `6080a20846b98ae099bb23034271bc8ab198e510e560e68f943326ec595c4608`.
- The generated report contains six complete experiment records. Its JavaScript
  passed `node --check`; expected control/data IDs were checked. The HTML was
  generated and inspected as an artifact; interactive browser rendering has
  not been exercised.
- The offline submission checker validated all 5,880 forecast rows.
  `submission.csv` still matches Git HEAD after normalizing Git line endings;
  it was not rewritten. Existing forecasting scripts and curated data are
  unchanged.
- A Windows/Linux GitHub Actions workflow was added, but no hosted CI execution
  was initiated in this session.

The local report is `outputs/evaluation/report/index.html`; experiment metadata,
predictions, row errors, per-city/year/lead diagnostics, environment snapshots
and hashes are under `outputs/evaluation/runs/`. These outputs are git-ignored
and should be regenerated or archived with the exact suite when shared.

## Source and interpretation limits

Official Space source revision:
`44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6`.
Training dataset revision:
`bbbdebdab2a18e00f97ef87bd3f35a71e222180b`.
The Space's MIT license, source snapshots, exact URLs and hashes are retained
in `evaluation/upstream/`.

The official pure scoring function is preserved. Historical baseline values
and eligibility are reconstructed from the documented public rules. The
private answer key, private scoring mask, official reference MAEs and baseline
generation code cannot be independently verified here. Previously used
historical windows overlap these tasks, so they are not untouched evidence.
Worker subprocesses provide a clean execution context but do not sandbox a
trusted adapter's filesystem or network access.

No 2019–2025 weather observations were downloaded. No submission was uploaded,
no official submission quota was used, and no external service was deployed.

## AI-use record

Codex, GPT-6.1-sol through T3 Code, researched the official public rules,
scoring and judging source and NOAA dataset documentation; implemented and
tested the platform; ran historical validation; and wrote the guide and this
record. The frontend-design skill informed the portable report layout. Human
review should focus on any new adapter's origin-specific information use,
the documented eligibility interpretation, dependencies outside code snapshots,
and the limitations of historical model comparisons and rubric self-ratings.
