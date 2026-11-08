---
name: assembly-line
description: Check this agent's work against the ProtoStar assembly line and fix any drift. Use at the start of every task, before scoring a model, before publishing data, and whenever told the line has changed.
---

# Assembly line check

Any agent can follow this file, whichever tool it runs in. The data rule in `code/AGENTS.md` always comes first.
Background: `docs/plans/assembly-line-plan.md`.

1. Run `uv run python scripts/line.py status`.
2. Finish your current step first. Never interrupt a harness run in progress.
3. Fix each finding below, in order, then run status again.
4. End by reporting: station, branch, commit, data releases in use, and any finding left open and why.

If a command fails or a finding is not in this table, stop and ask the integrator.

| Id | Finding | Fix |
|----|---------|-----|
| D0 | No lock on main yet | Wait for the integrator |
| D1 | Working in the old shared checkout | Follow `docs/assembly-line-migration.md`, then work only from `wt/<station>` on branch `Meet/<station>` |
| D2 | Uncommitted work | Commit it to your branch |
| D3 | Behind main | `git merge main`, then run your tests. Teammates: `git fetch` and merge `origin/main` |
| D4 | Evaluator is not the released one | Same as D3. Only the eval station edits scoring files |
| D5 | Newer upstream data exists | Finish your current version. Then `line.py pull` and adopt it as one declared change group |
| D6 | Reading another station's working files | `line.py pull`, then read `data/processed/releases/<id>/` |
| D7 | Output with no release | `line.py release <station> <folder> --note "why"` after committing |
| D9 | Runs missing from MLflow, or MLflow unreachable | `line.py sync`. If unreachable, carry on: sealed runs are safe and sync later |

## How to work

| To do this | Run |
|---|---|
| Score a model | `uv run python scripts/line.py score --name Placeholder_v0.2 --adapter <file.py:forecast> --assets <manifest.json>` |
| Publish data | Commit, then `line.py release <station> <folder> --note "why"`. Add `--gate-eligible` only if built per origin with nothing after that origin |
| Get the line's data | `line.py pull`. Declare the files from `data/processed/releases/<id>/` in your asset manifest |

Your run is `official` only if the code is committed, the evaluator and test set match the lock, and all its data releases are gate-eligible. Anything else is still logged, as `exploratory`. If a run comes out exploratory, read the `line.reasons` tag in MLflow.

## Only touch your own station

| Station | Owns |
|---|---|
| clean | Cleaning code and notebooks |
| enhance | External data and its entries in `data/README.md` |
| features | Feature code |
| train | Model adapters, model cards |
| eval | `code/evaluate.py`, `code/evaluation/`, `evaluation/`, `tests/` for the harness |

Ask the integrator for changes to `pyproject.toml`, `uv.lock`, `requirements.txt`, `line.lock.json`, `README.md` or `submission.csv`.

## Never

- Edit a release or a sealed run.
- Edit `submission.csv` by hand.
- Open assessment origins 2004 or 2011. Only the integrator runs assessment.
- Run `evaluate.py prepare` on `suite-v3`. Teammates use `line.py get-suite`.
- Hard-code a path to the main checkout in model code.
- Write credentials into any file.
- Delete another station's files.
