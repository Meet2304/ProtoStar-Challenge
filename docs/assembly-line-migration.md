# Moving your existing work onto the assembly line

For agents that worked in the old shared checkout. Do this once, then stop and wait for the go-ahead.
The data rule in `code/AGENTS.md` comes first.

The old checkout is `D:\Projects_Ad Astra\Project_27_ProtoStar Challenge\ProtoStar-Challenge`. From now on it is storage
only: it keeps `data/raw`, `outputs/evaluation` and the releases. **Nobody edits there again.**

## Your station

| Thread | Station | Branch | Worktree |
|---|---|---|---|
| 01_Dataset Cleaning | clean | `Meet/clean` | `D:\Projects_Ad Astra\Project_27_ProtoStar Challenge\wt\clean` |
| 02_Feature Engineering | features | `Meet/features` | `...\wt\features` |
| 03_Dataset Enhancement | enhance | `Meet/enhance` | `...\wt\enhance` |
| 04_Model Training | train | `Meet/train` | `...\wt\train` |
| 05_Model Evaluation | eval | `Meet/eval` | `...\wt\eval` |

05_ii_Baseline Creation shares the eval station. Its work is already committed, so it has nothing to move.

## Your files

| Station | Copy these from the old checkout |
|---|---|
| clean | `notebooks/Meet/01-dataset-structure.ipynb`, `notebooks/Meet/02-cleaning-daily-annual.ipynb`, `notebooks/Meet/meet_cleaning.py`, `notebooks/Meet/plan/`, `notebooks/Meet/qc/`, `notebooks/Meet/README.md` |
| train | `notebooks/Meet/03-placeholder-v0.1-training.ipynb`, `notebooks/Meet/placeholder_v0_1.py`, `notebooks/Meet/placeholder-v0.1-assets.json`, `notebooks/Meet/test_placeholder_v0_1.py`, `docs/plans/placeholder-training-plan.md`, `docs/models/placeholder.md` |
| enhance | `reports/adjacent-datasets-research.md` |
| eval | `reports/evaluation-review-gemini-flash.md`, `reports/evaluation-review-summary.md`, `reports/protostar-judging-weights.html`, `README.md`, `data/README.md` |
| features | Nothing yet. Only run steps 1, 8 and 9. |

If you created a file that is not listed, take it too. If you are not sure a file is yours, leave it and mention it in your
report. Do not copy the deletion of `notebooks/01_explore.ipynb`; a person will decide on that.

## Steps

Finish your current step first. Never interrupt a harness run in progress.

1. Work from your worktree from now on, using its absolute path. Confirm with `git branch --show-current` that you are on
   your `Meet/<station>` branch.
2. **eval only:** bring in the committed evaluator work with `git merge evaluation/opus-hardening-v4`.
3. **Copy** each of your files from the old checkout to the same relative path in your worktree. Never move or delete the
   originals.
4. Fix paths inside your files: nothing may point at the old checkout by absolute path.
5. Run the tests your files already have, using `uv run` from your worktree.
6. Commit only your own files, naming each path in `git add`, then publish your branch:
   ```bash
   git add <your paths>
   git commit -m "Move <station> work onto the assembly line"
   git push -u origin Meet/<station>
   ```
7. **clean only:** register the existing cleaned data as version 0. It is marked not gate-eligible, because its QC used the
   full record through 2018:
   ```bash
   uv run python scripts/line.py adopt clean "<a folder holding only your final cleaned output, without older runs>" --note "Cleaning output before the assembly line"
   ```
   A warning that it is not on MLflow yet is fine.
8. Run `uv run python scripts/line.py status --offline` and `uv run python scripts/line.py migration`. Every file you moved
   must show your branch name.
9. Report: station, branch, commit, files moved, test results, anything you left behind and why. **Then stop.** Do not start
   new work until told to.

## Do not

- Run `git add`, `commit`, `stash`, `switch`, `checkout` or `reset` in the old checkout. It is shared with every other agent.
- Delete or move anything in the old checkout.
- Push to `main` or to another station's branch. Only the integrator merges to `main`.
- Commit data, credentials, `.env` files, or `submission.csv` changes.
- Rewrite history or force-push.
