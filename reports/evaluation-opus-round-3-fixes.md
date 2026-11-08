# Sol implementation after Opus round 3

Protocol: `audited-runtime-v4`; default immutable suite: schema 3.
The independent review remains 81/100 until a fresh Opus audit changes it.

| Opus finding | Implemented behavior | Evidence |
|---|---|---|
| I1 legacy gate bypass | Schema-1/2 suites are readable archives; new runs/freezes fail before model execution or scoring | protocol-v4 legacy gate regression |
| I2 undeclared helper drift | Imported Python source outside interpreter/environment must match existing pins or declared source assets; no discovered pin is automatically trusted | runtime-audit sibling/helper/cache/namespace regressions |
| I3 labels declared static | Pinned annual hash, known raw-data paths and recognizable dated weather/outcome schemas require weather declarations | seven classification regressions, census exception |
| I4 docs-only commit | Frozen commit must be an ancestor; unchanged source/assets and installed environment still required | real synthetic Git descendant/nonancestor/source-change tests and CLI workflow |
| I5 line endings | LF-normalized logical source/input identity; LF derived CSVs; separate exact physical manifest seal | six portability regressions; Windows/Linux CI identity comparison |
| I6 assessment baselines | Frozen method inferred by `baselines --phase assessment --freeze PATH` | committed synthetic CLI assessment |
| I7 repeated assessment use | Report counts known development trials, assessment accesses, unique freezes and repeat accesses; unavailable ledgers show lower bounds/unknown totals | eight report/exposure tests |
| I8 installed packages | Freeze pins Python implementation/version and all installed distribution names/versions | environment-drift regression |
| I9 missing diagnostics | Separate development bundles show stable-station, strict Days Both CDD baseline AND target masks, and hypothetical temperature precision sensitivities; predictions/default scores unchanged | ten numeric diagnostic tests; three bundle tests; real development bundle |
| I10 reproducibility | Targeted committed evaluator and cross-platform offline workflow, including logical-ID equality | CI result recorded separately after execution |
| L4 common raw read | Pre-import Python open tripwire rejects repository data, suite and sibling evaluation artifacts, including swallowed errors | ten worker audit subprocess tests |
| Workflow complexity | `freeze --from-run RUN` copies verified model settings; assessment automatically reuses them | full custom-asset CLI development/freeze/assessment regression |

Validation: the initial full suite passed 120/120; the three newly added bundle
checks passed separately, giving 123 checks. Real 210-city development runs,
baselines, diagnostics, portable report and submission format check succeeded.
The unchanged submission SHA256 is
`2c9b61dae823ce2e576cdbef685e5e6c54b0ad2deee11c15ee654ce885ce4091`.
No real assessment score was opened; assessment tests use synthetic observations.
No observed weather after 2018 was obtained.

Remaining constraints are disclosed: all historical labels are reused; there
are few, nonstationary periods; private answer masks, precision and denominators
cannot be certified; author-declared weight history cannot be proved. Python
tripwires are practical accident checks, not an operating-system sandbox or
protection against deliberate native/subprocess/interpreter bypass. Installed
versions do not attest package binary integrity. New freezes cannot undo exposure.
