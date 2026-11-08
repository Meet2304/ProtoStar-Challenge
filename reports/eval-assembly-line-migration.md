# Eval station migration

Station: `eval`  
Branch: `Meet/eval`  
Worktree: `wt/eval`  
Committed evaluator history merged with `git merge evaluation/opus-hardening-v4` (`3dd2653`).

## Files copied

- `reports/evaluation-review-gemini-flash.md`
- `reports/evaluation-review-summary.md`
- `reports/protostar-judging-weights.html`
- `README.md`
- `data/README.md`

Four files are byte-identical to the old shared copies. The Gemini report is
identical after excluding its repository-header line and normalizing line
endings. That header now refers to the repository root instead of the old
absolute checkout path, as required by the migration guide. No review findings
or scores were changed.

## Existing storage and ownership

Raw caches, sealed evaluation runs, suites and release storage remain in the
old checkout as instructed. Existing tracked preregistrations and archives came
through the explicitly requested evaluator-history merge; no new data was
acquired or added for this migration. Other stations' uncommitted files and
the proposed deletion of `notebooks/01_explore.ipynb` were not copied.

No assessment, forecast fit, holdout scoring, Opus review or original review-loop
work was started. The original review loop remains paused. `submission.csv`,
dependency locks, assembly-line lock and credentials were not changed. No data
release is used by this migration's synthetic tests.

## Migration checker limitation

`line.py migration` matches only exact current file content on each branch.
The required path repair therefore makes the Gemini report appear as
`NOT MIGRATED`, even though the whole report is committed on `Meet/eval`.
This is a path-only normalization for the integrator to review, not a missing
file. The source and destination SHA256 values below preserve the distinction.

- `reports/evaluation-review-gemini-flash.md`:
  source `83e521dffa64e5e9d04c4fd7afb9f62b9794212470ffdd58c0950f5b4c3b263b`;
  destination `b2f74b6e09e904c188f15b8fe007cde498afc5cd2684e0fc5650dfe6000b67da`.
- `reports/evaluation-review-summary.md`:
  source `8fabf87a7e062fe3413216f4c22a8ccac17d23abde515488eaf6eac7ee16023a`;
  destination `8fabf87a7e062fe3413216f4c22a8ccac17d23abde515488eaf6eac7ee16023a`.
- `reports/protostar-judging-weights.html`:
  source `e0a451d9aa738e5ab2a5cdbc3dbf20d7eba612d706a6a978dc18e0d668f6c0aa`;
  destination `e0a451d9aa738e5ab2a5cdbc3dbf20d7eba612d706a6a978dc18e0d668f6c0aa`.
- `README.md`:
  source `516c7f4a78a504eff23268a3cc742817809dc41c3f3f186a22831eab38e142ee`;
  destination `516c7f4a78a504eff23268a3cc742817809dc41c3f3f186a22831eab38e142ee`.
- `data/README.md`:
  source `226b6e7d0493a057c5d34f70b8c96490007a7e02336530c582cd58ad8a42ed8b`;
  destination `226b6e7d0493a057c5d34f70b8c96490007a7e02336530c582cd58ad8a42ed8b`.
