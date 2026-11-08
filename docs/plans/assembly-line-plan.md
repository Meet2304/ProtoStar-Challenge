# Plan: the assembly line (worktrees, releases, MLflow lineage, sync skill)

Status: **r2, being implemented.** Author: Claude Code (Claude Opus 5.5), 2026-11-08. Tooling is built and rehearsed; agents
are paused until it is live, then rapid iteration begins. The challenge closes **2026-11-10 23:59 ET**. This adds one script
and one skill, and changes nothing in the harness. Teammates: start with [the team guide](../team-mlflow-guide.md).

## 1. The whole idea on one page

**Stations never share a working folder. They hand each other numbered, read-only releases, and every score says
exactly which releases it used.**

```
 [1 Cleaning] ──────► clean-vN ────┐
                                   ├──► [2 Features] ──► features-vN ──► [4 Training] ──► model run
 [3 Enhancement] ───► enhance-vN ──┘                                          │
                                                                              ▼
 [5 Evaluation] ──► eval-vN + suite-v3 (the fixed test set) ───────► sealed run ──► MLflow
```

Evaluation is a service the line calls, not a step at the end. Features sits after Enhancement in the flow because
it consumes both data sources, even though the threads are numbered 02 and 03.

Five rules. This is the entire procedure.

| # | Rule | Why |
|---|---|---|
| 1 | One station, one worktree, one branch. Nobody edits in the main checkout. | No overwrites, and no run killed by someone else's edit |
| 2 | Hand-offs happen only through releases (`clean-v3`). Never read another station's working files. | A release cannot change after someone starts using it |
| 3 | `line.lock.json` on `main` says what the current line is. Only the integrator changes it. | One answer to "what is current?" |
| 4 | A score is **official** only if it came from a clean commit, the released evaluator and the locked test set. | Official scores are always comparable |
| 5 | Every run goes to MLflow, labelled `official`, `exploratory` or `legacy`. Nothing is refused or deleted. | Agents keep moving, and old work stays visible |

The **integrator** is Meet or one thread Meet names. It is the only role that merges to `main`, cuts `eval-vN`,
edits `line.lock.json` or runs the one-shot assessment.

## 2. Why this is needed (measured on 2026-11-08)

- **Every agent shares one checkout**, on branch `evaluation/opus-hardening-v4`, with 26 changed or untracked entries.
  `code/evaluation/tracking.py` appeared while this plan was being researched.
- **Git cannot separate the existing runs.** All 17 sealed runs record commit `16e0f84` with a dirty tree.
- **Most existing runs are not comparable.** They span 3 suites and 4 protocol versions; only 4 are on the current pair
  (`audited-runtime-v4`, suite `4029caa1…`).
- **The shared tree breaks runs by design.** `source_pins` hashes all of `code/` and `src/`, and a run fails if any of
  it changes mid-run.
- **No dataset identity exists.** `data/processed/meet_annual_clean.parquet` and `data/processed/meet/annual_clean.parquet`
  sit side by side with no version or lineage.

Much of the lineage is already sealed in each `run.json`: `suite_id`, `created_utc`, `git`, `source_pins`, and a SHA-256
for every declared data file. What is missing is names for those hashes and a view of them.

## 3. Layout

```
Project_27_ProtoStar Challenge\
  ProtoStar-Challenge\      THE DEPOT, on main. No agent edits here. Owns the shared state:
    data\raw\                  pinned download
    outputs\evaluation\        suite-v3 and all sealed runs (nothing moves, so old paths stay valid)
    outputs\releases\          <station>\<id>\  immutable releases
    outputs\mlflow\            mlflow.db and artifacts
  wt\clean  wt\features  wt\enhance  wt\train  wt\eval     one worktree each, branch station/<name>
```

Inside each worktree:

| Path | How it gets there | Why |
|---|---|---|
| `code/`, `src/`, `notebooks/`, `docs/` | git | Isolated per station |
| `.venv` | `uv sync` (about 22 s) | Must be per worktree: the editable install ties `ROOT` to one checkout |
| `outputs/evaluation`, `data/raw` | NTFS junction to the depot | One suite and one run history; run folders have unique names |
| `data/processed/releases/<id>/` | real **copy**, by `line.py pull` | The harness rejects assets reached through a junction |

## 4. Versions: what gets a number

| Thing | Name | Identity | Made by |
|---|---|---|---|
| Cleaned data | `clean-v3` | SHA-256 of each file | Cleaning |
| External data | `enhance-v2` | SHA-256 of each file | Enhancement |
| Feature tables | `features-v4` | SHA-256 of each file, plus upstream ids | Features |
| Model | `Placeholder_v0.N` | commit + adapter hash + params | Training |
| Evaluator | `eval-v1` (git tag) | hash of the scoring files | Evaluation, cut by integrator |
| Test set | `suite-v3` | `suite_id` `4029caa1…`; never re-prepared | Fixed |

**Evaluator identity.** Hash only these entries from the sealed `source_pins`: `code/evaluate.py`, `code/common.py`, and
`core`, `protocol`, `worker`, `runtime_audit`, `assets`, `suite_identity`, `official_score` under `code/evaluation/`.
Adding a model elsewhere in `code/` then leaves the evaluator's identity unchanged.

**Evaluator changes.** At most one release per half-day. A change that alters scores also bumps `PROTOCOL_VERSION`, which
puts it in a new MLflow experiment automatically. `line.py status` lists the runs that need re-scoring.

**A release** is a folder plus a `release.json`:

```json
{"id": "features-v4", "station": "features", "created_utc": "20261109T140000Z",
 "git_commit": "<40-char sha>", "upstream": ["clean-v3", "enhance-v2"],
 "gate_eligible": true, "lineage": "built", "note": "adds regional pooling",
 "files": {"origin-1983.parquet": {"sha256": "...", "information_end_year": 1983}}}
```

`line.py release` refuses a dirty tree, writes to a temp folder, then renames it in, so two agents cannot collide.

**`gate_eligible` protects the data rule.** A dataset can only count for gates if it was built per origin, using nothing
after that origin. Today's cleaning output fits its QC on the full record through 2018 (baselines plan section 3), so it
enters the line as `clean-v0`, `gate_eligible: false`. Runs that use it are `exploratory` until a per-origin build exists.

**`line.lock.json`** (on `main`):

```json
{"evaluator": "eval-v1", "suite_id": "4029caa1...", "evaluator_hash": "...",
 "releases": {"clean": "clean-v1", "enhance": "enhance-v1", "features": "features-v1"}}
```

## 5. MLflow: one row answers your question

One server for the whole team, with a personal login for each person, started from the depot with `line.py serve`. Stations
and teammates all log to it; the address comes from `line.lock.json` or `MLFLOW_TRACKING_URI`.

**Team access.** The server stays on `127.0.0.1` and is reached through Tailscale, so it never touches the public internet.
MLflow's login is on top of that: requests without a valid password get 401, and unknown host names get 403. Everyone can
create and edit runs, and only the admin can manage users. Credentials live in environment variables, never in a file.

**Teammates need no access to the depot.** MLflow also distributes what they would otherwise need from this disk:
`line.py get-suite` downloads the exact `suite-v3`, and `line.py pull` downloads data releases, both hash-checked.
Teammates must not run `prepare`: a rebuilt suite has byte-identical files but a different `suite_id` (the ID also hashes
`core.py`), and the harness refuses to compare runs across suite IDs. Only the depot machine publishes data releases, so
version numbers cannot collide. Each run carries `line.author`.

Each run shows:

| You want to know | MLflow field | Source (already sealed in `run.json`) |
|---|---|---|
| Which model | run name, `line.model` | `model`, `adapter_sha256` |
| Trained on which data | **Datasets** column, context `train` | release ids matched from the asset binding's paths and hashes |
| Tested on which set | **Datasets** column, context `test` | `suite_id`, phase, origins |
| Which evaluator | `line.evaluator` | evaluator hash, matched to `eval-vN` |
| When | start time | `created_utc` |
| Which code | `git.commit`, `git.dirty` | `git` |
| Can I trust the comparison | `line.status` | section 6 |

Filter the UI with `` tags.`line.status` = 'official' `` for the leaderboard. Each release also gets one MLflow run in
`protostar/releases`, so data versions have their own timeline. `line.py sync` adds the dataset inputs and `line.*` tags on
top of the existing `tracking.py`, which is unchanged.

## 6. The three labels (backward compatibility)

| Label | Meaning | Counts for gates G1-G4? |
|---|---|---|
| `official` | Complete and verified, clean commit, evaluator and suite match the lock, all data releases `gate_eligible` | Yes |
| `exploratory` | Anything else, including runs from a dirty tree or against an old evaluator | No, but visible |
| `legacy` | Created before the line started (the 17 existing runs) | No, and never re-run for the label's sake |

Labels are computed from sealed files at sync time, so nobody has to remember them and an agent cannot claim one.
Existing commands, paths and plans keep working. `evaluate.py` is unchanged.

## 7. The skill: `assembly-line`

Your idea, with one design constraint from what is actually running. A skill can tell an agent what to do, but it cannot
see the other agents or stop them. So the work is split:

- **`scripts/line.py status`** finds the facts. It is deterministic and reports the same answer to every agent.
- **The skill** says what to do about each finding.
- **The labels in section 6** are the real enforcement: an agent that skips the skill still works, but cannot produce an
  `official` score.

**Where it lives:** `.claude/skills/assembly-line/SKILL.md`, committed, plus one pointer line in `code/AGENTS.md`. The live
threads use five provider setups (two Claude accounts, Codex, Grok, Cursor), and only Claude Code auto-loads skills, so the
file must also work when any agent is simply told to read it. It stays out of the hashed source, so adding it affects no run.

### Draft skill

```markdown
---
name: assembly-line
description: Check this agent's work against the ProtoStar assembly line and fix any drift. Use at the start of
  every task, before scoring a model, before publishing data, and whenever told the line has changed.
---
# Assembly line check

Plan: docs/plans/assembly-line-plan.md. The data rule in code/AGENTS.md always comes first.

1. Run `uv run python scripts/line.py status`. If you cannot run it, you are not in a worktree: go to D1.
2. Fix each finding below, in order. Run status again after each fix.
3. Finish your current step before fixing anything. Never interrupt a harness run in progress.
4. End by reporting: station, branch, commit, pinned releases, and any finding left open and why.

| Id | Finding | Fix |
|----|---------|-----|
| D1 | Working in the depot | Commit only your own files to station/<name>, then move to wt/<name> |
| D2 | Uncommitted work | Commit it to your station branch |
| D3 | Behind main | `git merge main`, then run your tests |
| D4 | Evaluator is not the locked release | Same as D3. Re-score only the runs status lists |
| D5 | Upstream release is out of date | Finish your current version first. Adopt the new release as one declared change group |
| D6 | Reading another station's working files | `line.py pull`, and read data/processed/releases/<id>/ |
| D7 | Output handed over without a release | `line.py release <station> <folder> --note "..."` |
| D8 | Existing output with no version | `line.py adopt <station> <folder> --note "..."` (lineage: reconstructed) |
| D9 | Run missing from MLflow | `uv run python scripts/mlflow_sync.py` |

## Only touch your own station
clean: cleaning code and notebooks | enhance: external data | features: feature code | train: adapters, model cards |
eval: code/evaluate.py, code/evaluation/.
pyproject.toml, uv.lock, line.lock.json, README.md and submission.csv: ask the integrator.

## Never
Edit a release or a sealed run. Edit submission.csv by hand. Open assessment origins (2004, 2011); only the integrator
runs that. Hard-code a depot path in model code. Delete another station's files. Re-prepare suite-v3.
```

**What each finding checks.** D1, D2, D3: git. D4: evaluator hash against the lock. D5: pins against the lock. D6: a scan of
the station's code for other stations' paths, best effort. D7, D8: unversioned files in known output folders. D9: sealed
runs against MLflow fingerprints.

### Message to paste into each running thread

> The project now runs as an assembly line. Finish your current step and do not start another harness run yet. Then
> read `docs/plans/assembly-line-plan.md` sections 1 and 7, and `.claude/skills/assembly-line/SKILL.md`, and follow the
> skill. Your station is `<name>` and your worktree is `D:\Projects_Ad Astra\Project_27_ProtoStar Challenge\wt\<name>`.
> Use that absolute path from now on. Report back with the skill's final summary.

## 8. Getting there without breaking running work

New code: `scripts/line.py` (`setup`, `status`, `pull`, `release`, `adopt`), the skill, `tracking.py` additions.
`scripts/` is not hashed, so `line.py` cannot invalidate a run.

| Step | Action | Who |
|---|---|---|
| 0 | Build and test `line.py` and the skill in a separate worktree | Integrator |
| 1 | Let in-flight work reach a checkpoint (table below); no new harness runs | All |
| 2 | Back up the working tree, including untracked files | Integrator |
| 3 | Sort the 26 entries by owner and commit each onto its station branch. A person confirms the deletion of `notebooks/01_explore.ipynb` | Integrator |
| 4 | Merge evaluator work to `main`, tag `eval-v1`, write `line.lock.json` | Integrator |
| 5 | Put the depot on `main`; run `line.py setup` for the 5 stations | Integrator |
| 6 | Start MLflow; backfill the 17 runs as `legacy`; run the controls under `eval-v1` | Integrator |
| 7 | `adopt` existing outputs as `v0`; paste the message into each thread | Integrator |

| Thread | State now | Next |
|---|---|---|
| 01_Dataset Cleaning | Running | Checkpoint, then wt/clean. Output adopted as `clean-v0`. Needs a per-origin build for `clean-v1` |
| 02_Feature Engineering | Idle, advice only | Restart in wt/features. Publish `features-v1` from the Placeholder feature code |
| 03_Dataset Enhancement | Idle, research report only | Restart in wt/enhance. First release is static context (weather-independent) |
| 04_Model Training | Running; plan under review | Checkpoint, then wt/train. Run v0.1 after `eval-v1` |
| 05_ii_Baseline Creation | Running, Grok review r3 in flight | Its `controls`, `scorecard`, `tracking` work goes into `eval-v1` |
| 05_Model Evaluation | Cancelled; Opus audit round 4 still running | Wait for the audit, then fold its fixes into `eval-v1` |

## 9. Sequence

Infrastructure goes in first while agents are paused, as fast as possible. Rapid iteration starts as soon as Step 7 is done,
with no fixed milestones before the challenge deadline.

Two requirements for the final build, both existing rules:

- **Commit the chosen data.** Assessment rejects git-ignored assets (confirmed below), so the winning release files must
  be committed at a tracked path before `freeze`. Keep them to compact per-origin tables.
- **Port the model into `code/`.** `submission.csv` must rebuild byte for byte without an agent.

## 10. What was tested

Run on a throwaway clone of commit `daa67ab` in a scratch folder, since deleted. The real repo was not touched.

| Check | Result |
|---|---|
| Two worktrees with junctions to shared state | `git status` clean; `ROOT` correct per worktree; suite-v3 verifies |
| Two stations scoring at once into one runs folder, while the evaluator was edited in another checkout | All 4 runs completed and sealed, `dirty False`, and passed `verify_run` |
| Evaluator-hash approach | Same hash across both worktrees (the scoped hash in section 4 is a narrower version of this) |
| Data release as a real copy | Accepted |
| Data release through a junction | **Rejected**, hence `line.py pull` copies |
| Git-ignored data at assessment | **Rejected**, hence section 9 |
| One MLflow server, 30 parallel writers | 30 of 30 recorded, no errors |
| Train and test dataset inputs | Stored and returned correctly; tag filters work |
| Existing `tracking.py` through the server | 4 logged, then 4 skipped on the second sync |

**Full rehearsal of `line.py`** on a throwaway copy with spaces in its path, on the real evaluator (`57fd221`):

| Check | Result |
|---|---|
| No login / wrong password / right password / unknown host | 401 / 401 / 200 / 403 |
| `setup` creates station worktrees | Clean, including paths with spaces |
| Release with `--gate-eligible`, then the same flag on data that is not per origin | Accepted, then refused |
| `status` on an out-of-date station; in the depot | Reports D3; reports D1 |
| Committed model on a released dataset | `official`, with `clean-v1` as train and `suite-v3:development` as test |
| Same model with an uncommitted edit | `exploratory`, reason "code was not a clean commit" |
| Run from before the line started | `legacy` |
| Teammate: separate clone, own login, no access to the depot's disk | Fetched suite and release from MLflow; logged an `official` run with the right author |
| Sync run twice | No duplicates |
| Server config file | Contains no password |

Not yet tested: reaching the server through Tailscale (not installed), and a Colab session talking to it directly.

**Current bar, from the same lab** (equal-fold mean final raw skill, development origins): 30-year average 0.000,
combo **-0.013**, linear trend -0.122, last year repeated -0.375. Combo produces today's `submission.csv` and is slightly
below the 30-year average on these folds, so G1 is a real hurdle.

## 11. Risks

- **More agents means more looks at the same three folds.** Parallel work overfits them faster. Each model version keeps
  the training plan's limit of at most 3 pre-declared ablations, and `official` run counts are shown per station.
- **The evaluator moving late.** It resets comparability, so it follows the half-day release limit and stops after the data
  freeze except for bugs.
- **Agents ignoring the process.** The labels make that harmless: their work appears as `exploratory` and cannot win.
- **Junctions and the runtime audit.** From reading the code, normal repo-relative paths into `data/raw` are still denied,
  but a hard-coded absolute depot path would not be. The audit is a tripwire, not a sandbox; the skill forbids this.
- **The server runs on one PC.** If it sleeps, teammates cannot log, but nothing is lost: runs stay sealed on their disks,
  commands fail fast, and `line.py sync` catches up without duplicates.
- **`suite-v3` holds historical answer keys.** It reaches only logged-in teammates, who could rebuild those keys anyway,
  and covers years through 2018 only.

## 12. Decided

- Infrastructure first, with agents paused; no fixed timetable.
- One MLflow instance shared with the whole team.
