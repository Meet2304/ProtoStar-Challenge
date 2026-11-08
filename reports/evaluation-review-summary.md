# Independent evaluation reviews: consolidated assessment

Two independent reviewers assessed the implementation without changing it:

- [Claude Opus 5.5](evaluation-review-claude-opus.md): **70/100**.
- [Gemini 3.8 Flash](evaluation-review-gemini-flash.md): **85/100**.

These are subjective engineering assessments using the same requested rubric,
not measured forecast performance, calibrated confidence probabilities, or
official challenge scores. Averaging them would obscure their substantive
disagreement about readiness.

| Dimension | Maximum | Opus | Gemini |
|---|---:|---:|---:|
| Challenge-rule/scoring fidelity | 30 | 26 | 29 |
| Statistical validation/test-set design | 30 | 15 | 22 |
| Implementation robustness | 20 | 14 | 17 |
| Reproducibility/provenance | 10 | 8 | 9 |
| Team usability | 10 | 7 | 8 |
| Total | 100 | 70 | 85 |

## Decision for the team

**Use the current platform for exploratory historical diagnostics. Do not yet
rely on its default internal ranking to select the final model or treat its
scores as independent final evidence.** Neither reviewer supports predicting an
official leaderboard rank from these historical results.

The scoring arithmetic and basic data preparation received strong support.
Both reviewers verified the pinned upstream source and deterministic suite
rebuild. The important weaknesses concern how comparisons are interpreted,
whether models can accidentally consume target-period observations, and how
the team separates development from assessment.

Gemini calls comparative selection conditionally ready when using recent folds
and raw skill. Opus calls it not ready. The more conservative assessment is
warranted because Opus demonstrated that an adapter calling the team's normal
data loader can see the target years and be recorded as a successful perfect
forecast. Gemini's statement that the 2018 guard has no bypass does not address
this distinct issue: for a 2011 origin, 2012–2018 observations pass that guard
but are still future information. Subprocess execution is not a data-access
sandbox.

## Consolidated findings and fixes

| Priority | Finding | Evidence and practical consequence | Proposed fix |
|---|---|---|---|
| 1 — before model selection | Forecast-origin leakage through ordinary loaders | Opus's `outputs/review-opus/leaky_adapter.py` calls `load_valid_annual()`, which permits all observations through 2018. The evaluator records the resulting answer-based predictions as complete. | Supply an explicit fold-origin context to every supported weather loader; fail closed when a model loads later observations. Separate model input files from evaluator labels. Add a regression adapter that attempts this access and must fail. Document that arbitrary filesystem/network reads still require an actual sandbox or procedural separation. |
| 1 — before model selection | Mean clipped fold scores distort internal rankings | Both reviewers found that averaging nonnegative fold Scores hides negative skill. The current register can rank linear trend above the thirty-year mean despite worse average raw skill. | Rank by a predeclared average of unclipped metric skill; preserve official per-fold Scores for display. Show raw fold results and baseline comparisons, including negative values. Use a numerical tolerance for ties. |
| 1 — before independent performance claims | Assessment years already influenced combo selection | Both reviewers found overlap with the existing selection backtests. Opus quantified that every final-analogue year in suite-v1 was already used. | Label suite-v1 as reused development data. Evaluate the entire selection procedure inside each historical origin using nested, earlier-only tuning. Where genuinely unused eligible years exist, register them before opening scores and record a committed model/parameter freeze. A new folder or different fold boundaries do not make reused years untouched. |
| 1 — before uncertainty claims | Current bootstrap answers a narrow conditional question | Both reviewers flagged only four years per final fold. Opus highlighted missing between-period variation; Gemini highlighted serial dependence and changing lead composition. | Report paired raw-skill differences by fold and metric first. If intervals are retained, state the estimand and limitations, account for period-level variation, and preserve relevant dependence/horizon structure. Five periods are also too few for a strong generalization claim; replacing one interval with a fold-level t interval is not a guarantee of valid coverage. |
| 2 — before general team rollout | Default combo receives unsupported parameters poorly | Opus reproduced an unexpected-keyword failure with nonempty unsupported `--params`. An empty parameter object is not the issue. | Declare supported parameters and reject unknown keys clearly before launching workers, or implement meaningful tunable parameters. Add a CLI regression test. |
| 2 — before general team rollout | Cross-drive report crash on Windows | Gemini identified `os.path.relpath` failing when runs and report output are on different drives. | Generate a valid file URI or copy/link metadata into the report package when relative paths cannot be formed. A raw `D:/...` string is not necessarily a portable HTML link. Test cross-drive behavior. |
| 2 — before general team rollout | Adapter import-path assumptions | Gemini found the worker adds `code/` but not the repository root. Some root-relative imports fail. | Define the supported module contract, prefer imports from the installed `protostar_challenge` package, and support repository-root modules deliberately. Add a real subprocess import test. |
| 2 — before general team rollout | Timeout reporting and incomplete failure tests | Opus found timeout exceptions escape the CLI's friendly error handler, although run metadata is correctly marked failed. Both reviewers found coverage gaps. | Convert subprocess timeout errors to a clear evaluator error including origin and log path; test actual CLI exit codes, timeouts, invalid returns and missing prediction files. |
| 2 — before publishing reproducibility claims | External predictions and non-code dependencies lack provenance | Opus identified external-CSV provenance gaps. Gemini identified omitted `src/` code. | Require a source manifest for imported predictions: author, model commit, parameters, training cutoff and dependency version; hash original input files. Snapshot declared source/assets, including `src/`, without copying secrets or unrelated files. Existing environment lockfiles are already captured. |
| 3 — diagnostic improvement | Historical periods differ from the official target regime | Both flagged climate drift; Opus also examined station changes and coverage. | Show recent-period and station-stability diagnostics alongside every fold. Predeclare any weighting/selection rule. Choosing only the best-looking recent fold after seeing results does not repair contamination or establish generalization. Keep diagnostic filtering separate from the official eligibility mask. |

Minor input-validation questions include whether booleans should be rejected as
predictions. Numeric year strings in CSV are not inherently a defect; parsing
them consistently is reasonable. The meaningful contract is integer-valued
years, exact target keys and finite numerical predictions. Equivalent source
bytes supplied through different provenance modes receive different suite
identities; this is conservative separation rather than corrupted scoring, but
clearer canonical-source recognition would improve usability.

## Claims confirmed and boundaries retained

- The pure scoring function matches the official pinned source. The historical
  validity and baseline rules reproduce the published baseline-eligibility
  counts; Opus also reproduced the published practice-run CDD errors.
- Public 2019–2025 observations cannot be used to compare/tune variants under
  the FAQ. This is distinct from the official live leaderboard, which the
  public About text describes as a way to check progress. The repository's
  stricter no-leaderboard-tuning rule is team policy. Preserve it unless the
  team deliberately changes its instructions.
- The official answer key, its exact scored-row mask and reference denominators
  remain private. Reproducing public arithmetic does not certify those inputs.
- Forecast Score informs two human judging criteria whose combined weight is
  20%. It does not independently contribute a fixed 20% numerical bonus, and
  the human prize ranking is separate from the forecast leaderboard.
- The current 19 tests pass, but the reviewers found important behavior they do
  not cover. Hosted CI has not been run from this workspace.

## Review status

Both requested reviews are complete. Their detailed reports and scratch
reproductions are retained separately. This consolidation adds no fixes and
changes no models, evaluation data, configuration, submissions or experiment
records. The existing platform guide and dashboard describe the pre-review
implementation; consult this report before relying on their readiness claims.
