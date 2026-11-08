# Sol fixes after independent Opus round 4

Protocol `audited-inputs-v5`; original rubric retained. Last accepted score is
89/100 for the earlier pinned core. This implementation awaits its own audit.

| Finding | Change | Evidence |
|---|---|---|
| N1 ordinary native reads and networking | Checked ordinary Arrow Parquet/CSV/Feather/dataset/IPC/JSON/OSFile/memory-map/local filesystem APIs; native remote filesystem constructors refused; socket connect/name resolution and urllib requests denied and logged before model import | native input regressions, including declared/supplied input positives |
| N2 custom-suite role bypass | Fixed source-bound policy protects target years 2005-2018 for the challenge's city IDs, even with converted input formats; preparation, verification and phase selection enforce it; report warns about known cross-suite exposure | synthetic pinned-source and reserved-ID cases; shifted-window and converted-CSV tests; report filter test |
| N3 undeclared JSON/code reads | Supported reads inside project and evaluator roots default-deny unless pinned source/cache, supplied input, declared snapshot, interpreter environment or own output; undeclared exec/spec reads fail earlier at open | native/project regressions; existing helper/import tests |
| N4 classifier shapes | TSV, gzip CSV/TSV, Feather/Arrow file schema checks; four-digit wide-year and long year/metric/value outcomes require weather handling; explicit census population columns remain allowed | nine classification tests |
| N5 missing final audits | Both verified worker audits required before scoring; verification also refuses v5 adapter runs missing them | zero-exit early termination and forged-complete regression |
| N6 evidence/source mismatch | Joint committed checkpoint followed by fresh real development/baseline/diagnostic evidence; record actual commit/source pins and unrelated workspace edits | final evidence manifest and CI recorded separately |
| Integrated controls scope | Review includes approved controls, panel, scorecards, lazy MLflow mirror, wrapper scripts and built-in allowlist addition | owned files approved by controls thread; synthetic panel suite; next independent audit |

The one-command team loop builds/reuses controls, runs a candidate, writes paired
raw-skill comparisons and characteristic scorecards, optionally mirrors verified
records to MLflow, and refreshes the report. Screening gates remain team policy.

Limits remain explicit. These are practical supported-API tripwires, not an OS
sandbox: hostile extension internals, subprocesses, existing descriptors or
restored functions can bypass them. Reads outside project/evaluator roots are
recorded but not comprehensively restricted. All historical labels remain
reused; few nonstationary periods and the unknown private answer key still limit
inference. No process improvement can manufacture untouched evidence or an
official leaderboard-rank estimate. No 2019+ observed weather was obtained,
no real assessment score was opened, and no submission was made or modified.
