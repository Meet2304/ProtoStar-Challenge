# Asset-bound evaluator validation

Sol implemented the independently reproduced gap where a tracked non-Python
model asset could change after a Python-only freeze. The reproduction is in
`outputs/sol-review-evidence/asset-freeze-b0929c67/evidence.json`.

The current execution protocol is `asset-bound-v3`; the immutable historical
input suite remains `suite-v2`. Official scoring arithmetic and historical
data were not changed.

## Changes

- Custom Python adapters must explicitly declare dependencies with `--assets`.
  `evaluation/no-assets.json` is an explicit empty declaration for callbacks
  using only provided train/targets and code/libraries. Built-in reference and
  nested adapters have an internal provided-history-only declaration.
- Asset bindings include full manifest bytes, metadata, scope, content hashes
  and sizes. Model weights require an origin-safe declared training cutoff.
  Weather CSV/parquet times are also inspected, rather than trusting metadata.
- Workers receive only files applicable to their own origin. Snapshots are
  verified before adapter import and after prediction. Original inputs are
  checked again during/after the run. Inactive later-origin files are not read.
- Committed assessment freezes verify declarations/assets against normal Git
  blobs or local materialized LFS oid/size, without remote fetching. Controlled
  Git CRLF text equivalence is supported, but raw runtime hashes remain exact.
- Report metadata links are now copied into the report package, avoiding
  cross-drive relative-path failures and links to another machine's folders.
- CLI and short/full guides document explicit dependencies, snapshots and the
  distinction between author declarations and verified facts.

## Acceptance evidence

**72 automated tests pass** on local Windows/Python 3.13. The suite includes
24 asset-component tests and a full committed synthetic CLI workflow. The
workflow proves development execution, committed asset freeze, assessment,
access logging, rejection of changed settings, and rejection of weights whose
declared training cutoff exceeds the origin. Other regressions verify explicit
declaration requirements and self-contained report metadata links.

The real-data default selector completed on development origins only as run
`20261108T082628Z-b29b0edef9` under `outputs/evaluation/runs/`. No real assessment
score was exposed. Source-era baseline counts and practice-run CDD errors still
pass, as do the original scorer's AST and numeric-parity checks.

An additional synthetic acceptance fixture is retained at
`outputs/sol-review-evidence/asset-cli-fixture-14565df4/`. No observed 2019+
weather was fetched. `submission.csv`, existing suites/runs and unrelated
notebooks were preserved.

## Remaining scope limits

The tool cannot certify the true training history inside an opaque weight
file, a source's claimed weather independence, or deliberately undeclared
reads by trusted code. Those remain author declarations and code-review duties.
Asset binding protects against accidental drift; it is not an OS sandbox.

Historical labels remain reused. Nested selection, phase roles and freezes do
not restore untouched evidence. Period sensitivity is descriptive, and the
official private answer key/reference errors remain inaccessible.

The snapshot review previously requested as round 2 was cancelled before
these production changes; no score from that obsolete snapshot is accepted.
A fresh independent Opus review of the full patch is required. The last
accepted Opus score is still the original 70/100 until a new report arrives.
