# Opus-directed evaluation fixes and validation

Sol (GPT-6.1-sol through T3 Code) implemented these changes from the
[Opus review](evaluation-review-claude-opus.md). A Sol helper implemented the
nested selector and its focused behavioral tests. The other review was not
used as the implementation brief. No new reviewer score is claimed.

## Required fixes

| Opus item | Implementation | Acceptance evidence |
|---|---|---|
| M1: enforce historical origin | `common.effective_cutoff_year()` tightens the existing guard with `PROTOSTAR_FOLD_CUTOFF`. Workers set it before adapter import and receive temporary train/target/parameter copies outside the evaluator's label directory. | Import-time and ordinary-loader canaries fail. Opus's original `outputs/review-opus/leaky_adapter.py` now fails with `DataRuleViolation` at the 1983 origin and is recorded failed. Compliant adapters still complete. |
| M2: correct internal ranking | Rank on equal-fold mean unclipped final skill, with ten-decimal numerical ties. Show a secondary summary that clips only after the mean; preserve official per-fold scoring unchanged. | Regression checks cover a negative-raw-skill model with positive clipped periods, numerical ties and both candidates clipping to zero. On the real development suite the baseline correctly outranks the worse-raw-skill linear trend. |
| M3: replace narrow comparison interval | Report paired raw-skill differences for each period, metric and lead, sign counts and leave-one-period-out sensitivity. Optional whole-period resampling keeps every lead/city together. No four-year bootstrap interval remains. | Known nonzero differences remain visible when both candidates' official Scores clip to zero. A real three-period comparison completes with the new output schema. Fewer than three periods or overlapping periods produce no bootstrap range. |
| M4: separate development/assessment and evaluate selection | Version-2 suites declare development/assessment roles and reused-history status. Assessment requires an immutable freeze tied to committed source, parameters, seed, suite and selected model. Access is logged. The default nested selector chooses methods from earlier-only inner folds for each outer origin. | Development artifacts contain no assessment-origin results. Real working-tree assessment freezes fail before evaluation until source is committed. A complete prepare/freeze/assessment/access-log CLI flow succeeds in a separate committed synthetic fixture. Nested tests verify a known winner and every inner information boundary. |
| M5: fix unsupported combo parameters | The fixed combo explicitly rejects nonempty unsupported parameters. The nested adapter declares and validates its own parameters. | Unknown combo parameters fail clearly before a run is created; empty parameter objects work. Unknown nested parameters and invalid candidate/fold settings are tested. |

The official `score` function and `ScoringError` extraction remain unchanged.
These changes improve the evaluator, not the submitted forecasting model.

## Operational and provenance fixes from Opus

- Timeouts now produce a concise nonzero CLI failure with origin and log path;
  failed runs remain excluded. Invalid callback return types, NaN and booleans
  also fail and retain failure evidence.
- Imported CSVs require author, model repository, full commit SHA, parameters,
  dependencies and per-origin training-cutoff declarations. Original bytes and
  hashes are preserved. External training claims remain author declarations.
- Identical pinned official parquet bytes receive the same source identity
  whether supplied through `--annual` or the default pinned source.
- Runs snapshot Python from `code/` and `src/`, environment/project metadata,
  adapter source, parameters and source pins. A source change during execution
  invalidates the run.
- Reports and comparisons verify saved artifact hashes; version-2 metadata
  also has a content fingerprint. Accidental score/metadata corruption is
  tested and rejected.
- The report groups runs by suite, phase and protocol, defaults to the current
  development protocol, exposes negative raw skill and labels legacy runs.
  Official per-fold Scores remain available as evidence.
- Documentation distinguishes the team's stricter live-leaderboard policy
  from the FAQ's ban on tuning against public observed answers. Human rubric
  wording no longer suggests a fixed numerical forecast-score bonus.

CSV numeric strings may parse to numbers. Integral years and finite values are
the relevant contract; booleans are rejected before an automatic adapter can
serialize them ambiguously. This is deliberate rather than rejecting ordinary
text-encoded CSV numbers.

## Validation completed

**45 tests pass** on the local Windows Python 3.13 environment. Coverage includes
the prior scorer contract tests, six nested-selection behavioral tests, and
regressions for Opus's leakage, ranking, uncertainty, freeze/provenance and CLI
findings. Real-data tests reproduce the published nine city-metrics / 63 rows
and practice-run CDD errors from the already-present pinned cache. The original
combo adapter remains consistent with `submission.csv` within its three-decimal
rounding.

The real version-2 suite was built for all 210 cities with fingerprint
`d77f1e4fffcd2d471e3f264ba7228f0f85d5f79e2e3990518e5c50ff2b075199`.
Only its development origins, 1983/1990/1997, were evaluated in the real-data
validation. Reference methods and the new default nested selector completed.
Repeated nested runs produced byte-identical `predictions.csv` AND
`nested_selection.json` for each development origin. Every recorded inner fit
ended by its own origin; every inner target ended by the outer origin.

The assessment workflow was exercised through the real CLI in an isolated,
committed fixture containing synthetic observations only:

```text
outputs/evaluation/opus-fix-validation/committed-fixture-970964af/
```

It successfully built a suite, created a committed freeze, evaluated only the
declared assessment origin, copied the freeze into the run and appended an
assessment access record. The team's working tree was not committed. A freeze
attempt against the actual uncommitted implementation was correctly rejected;
no real assessment run was created and no real assessment score was exposed.

The refreshed local report is `outputs/evaluation/report/index.html`.
Its JavaScript passes syntax and execution checks for defaulting to the current
protocol, separating legacy groups, searching and displaying empty results.
These checks used a minimal DOM stub rather than a browser renderer.
`outputs/evaluation/comparison-v2.json` records a real-data comparison using the
new period-level schema. The original Opus leak failure is preserved under
`outputs/evaluation/opus-fix-validation/runs/`.

`submission.csv` remains unchanged; its working-file SHA256 is
`2c9b61dae823ce2e576cdbef685e5e6c54b0ad2deee11c15ee654ce885ce4091`.
No post-2018 observations were downloaded, no official submissions were made,
and no existing suite/run records were rewritten. Existing unrelated notebook
and research changes were preserved.

## Boundaries that fixes cannot remove

- Current historical assessment years were previously used in research. A
  freeze and new fold roles cannot make them untouched retroactively. Nested
  selection repairs the procedure's direct look-ahead; choices made after
  seeing historical results still limit independent performance claims.
- Guarded loaders and temporary inputs protect against accidental leakage.
  Workers are trusted code, not an OS sandbox. Deliberate absolute file reads,
  unguarded third-party loaders or environment changes require stronger
  isolation/procedural controls.
- Whole-period bootstrap ranges are descriptive sensitivity, not formal
  confidence intervals. Few periods, dependence and climate/station changes
  remain limitations. Two assessment periods deliberately get no such range.
- The private answer key, exact private mask and official denominators remain
  unverified. Station-change filtering and Days Both sensitivity are optional
  future diagnostics, not changes to the official-style eligibility mask.
- Hosted Windows/Linux CI has not been executed from this session. Local tests
  pass; the existing workflow will run after the team pushes the intended
  changes. Interactive browser rendering was not exercised.

See [the updated team guide](../evaluation/README.md) for normal development,
external-CSV provenance, nested parameters and the committed-freeze workflow.
