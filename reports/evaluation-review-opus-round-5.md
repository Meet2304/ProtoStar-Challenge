# Independent evaluator audit, round 5 (Claude Opus 5.5)

Date: 2026-11-08. Reviewer: Claude Opus 5.5 (`claude-opus-5-5`), working alone and read-only on product code.
I wrote only this report plus scratch files under `outputs/review-opus-round-5/`. No source was edited, staged
or committed, no second checkout was made, and no task was delegated.

## 0. Disclosure: one deviation from the review's safety brief

The brief says real checks must stay on development years 1983/1990/1997. I ran the full `unittest discover`
sweep **before reading what the data-guarded tests do**. That sweep ran two pre-existing tests that touch real data:

| Test | What it did with real data | What reached me |
|---|---|---|
| `tests/test_opus_fixes.py:308` `test_published_baseline_counts_and_practice_cdd` | Loaded the whole pinned annual table through 2018 and computed CDD errors over **2011-2018**. It asserts them against the organizers' published 211.69 and 174.60. This covers target years of the protected 2004 and 2011 folds. | Pass/fail dots only. No value was printed. |
| `tests/test_controls_panel.py:336` `RealSuiteTests` | `verify_suite(suite-v3)` byte-hashed every suite file, including assessment answer keys. Parsing and scoring were development origins only. | Pass/fail only. |

I should have excluded these tests. No score or label value reached me, no assessment ledger was written, and my review
used nothing from them. I am reporting it because the brief forbids it, and because it exposes a real gap (R5).
Everything else I ran was synthetic, or real data with a `Year<=2004` pushdown filter.

## 1. Score: 90.25 / 100 (round 1: 70; round 3: 81; round 4: 89)

The rubric is unchanged and no points were moved between dimensions. The audited scope is now **larger**: it includes
controls, panel, scorecards, MLflow mirroring, scripts and the built-in allowlist entry.

| Dimension | Weight | R1 | R3 | R4 | **R5** | Main reason |
|---|---:|---:|---:|---:|---:|---|
| Rule / scoring fidelity | 30 | 26 | 28 | 29 | **29** | Scorer unchanged. Space pin and five rule-file hashes match live. New controls reproduced independently. Only the private key stays uncertifiable. |
| Statistical validation / test design | 30 | 15 | 21 | 23.5 | **23.5** | Custom-suite development bypass closed (+0.5). Two smaller gaps offset it: assessment counters reset per suite (R3), and the new one-command gates carry no trial count (R4). L1 and L2 stay permanent. |
| Implementation robustness | 20 | 14 | 16 | 18 | **18.75** | N3, N4, plain N5 closed. Panel and controls are solid. Common Arrow and stdlib paths still evade the tripwire (R1), and audits are self-reported by the worker (R2). |
| Reproducibility / provenance | 10 | 8 | 8 | 9.5 | **9.75** | All nine runs bind to the committed source. One manifest claim is broader than its evidence (R5). |
| Team usability | 10 | 7 | 8 | 9 | **9.25** | One-command loop works end to end. README overstates Arrow coverage (R1). Freeze ceremony remains. |
| **Total** | 100 | 70 | 81 | 89 | **90.25** | |

| Dimension | Exact deductions |
|---|---|
| Rules −1 | L3 private key/mask/precision −1 (**permanent**). |
| Statistics −6.5 | L1 reused labels −4 and L2 few non-stationary periods −2 (**permanent**); R3 −0.25; R4 −0.25 (repairable). |
| Robustness −1.25 | No OS sandbox −0.5 (**permanent in practice**); R1 −0.5; R2 −0.25 (repairable). |
| Provenance −0.25 | R5 −0.25 (repairable). |
| Usability −0.75 | Freeze/asset ceremony −0.5 (**inherent**); R1 documentation overstatement −0.25 (repairable). |

**Confidence: moderate-high.** Likely range 88.5 to 91.5. The main uncertainty is how much weight the R1 holes deserve.
They need an adapter author to pick one of those APIs, but `ParquetDataset` and `shutil.copy2` are ordinary code.

| Use | Readiness |
|---|---|
| Exploration | **Ready.** |
| Comparative selection | **Ready with conditions.** Development phase only. Read per-period raw deltas, not gate PASS alone. Human-check each adapter for the APIs in R1, and count candidates tried. |
| Independent final evidence | **No, and not repairable.** Assessment years are reused. A freeze is procedural discipline, not independence. |
| Leaderboard or rank prediction | **No, by design.** |

## 2. Audited scope and evidence

**Authority.** Commit `d63347e787445ed7c90d0c0fee5623a7e96798a9` from Git objects; HEAD `f50484f` changes only two report files.
Normalized `source_pins` canonical hash, 31 files: `5a59a1a39d73fe89b19ba1f117707825d5f956e86fecacc3232b1612a2213879`.
It **matches before and after**, at both commits (`fp-before.txt`, `fp-after.txt`). The workspace had zero pinned-file
differences, so executing workspace code was equivalent to the commit. `submission.csv` SHA256 was unchanged at `2c9b61da…4091`.

| Evidence type | What it covers | Result |
|---|---|---|
| **Hosted CI (confirmed with `gh`)** | Run 42471119448, head `d63347e`; Ubuntu, Windows, `portable-suite-identity` | All success. 164 tests, 3 skipped on each OS. Same logical suite ID `6fed3d07…333e`. Synthetic only. |
| **Local full sweep** | `unittest discover`, workspace = commit | 164 tests, OK, no skips. See §0. |
| **Live rule source** | HF Space API and pinned rule files (metadata and rules only) | Space SHA `44b8ec9f…`. `scoring.py`, `config.py`, `utils.py`, `content/data.md`, `content/about.md` match the manifest. |
| **Real development read** | `evidence_check.py`, `evidence.out.txt`: `verify_run` on 10 runs; raw cache at `Year<=2004` | See §3. |
| **Synthetic repros** | `repros.py`, `loop_check.py`, outputs beside them | See §4. Worker runs used a synthetic project root; loop runs used temp roots. |
| **Code inspection only** | `.venv` exemption, statistical design, doc claims | Labelled below. |

No observed 2019+ weather was obtained. No private key was touched. No submission was made.

## 3. Verified claims from the fix report

| Claim | Verdict | Evidence |
|---|---|---|
| N6: fresh evidence matches the committed source | **Pass** | All 9 panel runs and the guide run: `verify_run` OK, `source_pins` equal the committed pins, commit `d63347e`, phase development, origins 1983/1990/1997. Workspace noise was recorded honestly (19 lines). |
| Panel entries correct | **Pass** | Fingerprints match the manifest. Best control is `mean10` (−0.0087 final raw skill); combo −0.0134; nested −0.0065. |
| Controls are honest implementations | **Pass, independently reproduced** | My own code from the raw cache, `Year<=2004`: `mean10` skill matches to 2.2e-16; `mean10` predictions exact; `ridge_pooled` matches a from-scratch closed-form solve at origin 1990, lead 5, to 9.2e-12. |
| Built-in control exemption is justified | **Pass** | In 27 origin-runs across the panel, `io_audit` shows only supplied inputs, pinned source, own output, and 3 CPU-limit probes per origin-run outside the project. No project data read. |
| 9 MLflow mirrors | **Pass** | Read-only SQLite open: 9 fingerprints, complete and active. Idempotent re-sync confirmed on a synthetic store. |
| Freeze without assessment | **Pass** | Freeze is present, `git_commit` and pins match. No `assessment_access` folder in `suite-v3` or `suite-v3-r5`. Zero assessment-phase runs. See R5 for wording. |
| Space and scorer | **Pass** | §2. |

## 4. Round-4 items

| Item | Verdict | Evidence |
|---|---|---|
| N1 ordinary native reads, network | **Mostly closed, gap remains** | `read_parquet`, `read_table`, `ds.dataset`, `file:///`, socket and DNS are all caught (c1 to c5). The wrapped list is incomplete: see R1. |
| N2 custom-suite bypass | **Pass, with residual** | Protected city IDs at dev origin 2004 are refused (C1). Fixed list equals the real 210 cities. Relabelled IDs still pass (C2, deliberate). Assessment counts stay per suite (R3). |
| N3 project default-deny | **Pass** | Regression tests pass and my controls confirm it. It inherits the R1 holes. |
| N4 classifier | **Pass** | d2 to d6 rejected, including upper-case `.CSV.GZ`; census exception intact. Residual formats listed in R6. |
| N5 missing audits | **Pass for accidents** | B2: plain `os._exit(0)` now fails. A forged audit still passes (R2). |

New integrated scope checked with no defect: `controls.py` boundaries and determinism, panel reuse key and re-derivation,
scorecard no-mix-units aggregation, gate arithmetic, MLflow lazy import, lock and backdating. Scripts ran end to end on a
synthetic suite (first build, cache reuse, candidate, report, mirror).

## 5. Findings, most serious first

All are repairable. Labels refer to the outputs in `outputs/review-opus-round-5/`.

### R1. Common Arrow and stdlib calls still evade the tripwire (high within robustness; −0.5 robustness, −0.25 usability)

Evidence: synthetic, `repros.out.txt`. Each call targets a synthetic protected file in `data/` and ends with worker exit 0 and audit `verified`.

| Repro | Call | Result |
|---|---|---|
| p01 | `pq.ParquetDataset(path).read()` | 138 rows read, no event |
| p03 | `pyarrow.orc.ORCFile(path).read()` | 50 rows, no event |
| p06 | `fs.SubTreeFileSystem(dir, LocalFileSystem()).open_input_file(...)` | opened, no event |
| p07 | `ds.ParquetFileFormat().make_fragment(path, fs)` | 138 rows, no event |
| p08 | **`shutil.copy2(protected, outside_path)`** (Windows, native `CopyFile2`), then read the copy | 138 rows; only the outside read logged |
| p10 | `sqlite3.connect(<project db>)` | query OK, no event |
| p18 | `from pyarrow.parquet.core import read_table` | 138 rows, no event |
| w2, w3 | `pa.ipc.new_file`, `csv.CSVWriter` into `data/` | files planted in protected folder |
| w4, w5 | `os.remove`, `os.replace` inside `data/` | file removed, file renamed |

`read_table`'s module-level wrap covers the guarded name only, so the implementation function and class paths are unguarded.
`ParquetDataset` and `copy2` are ordinary code, and `copy2` is the repo's Windows platform. `evaluation/README.md:446`
says "Ordinary Parquet … entry points are checked too", which is too broad.

*Fix.* Wrap `pq.ParquetDataset`, `orc.ORCFile`, `ParquetFileFormat.make_fragment`, `ipc.new_file/new_stream`, `csv.CSVWriter`, and `SubTreeFileSystem`
base paths. Patch `shutil.copy2`/`_winapi.CopyFile2` and `pyarrow.parquet.core.read_table`. Add `sqlite3.connect`, `os.remove`, `os.rename`,
`os.replace`, `os.link`, `os.symlink` to the audit hook. Add these repros as tests, then correct the README sentence.

### R2. Audits are self-reported in a folder the model can write (medium; −0.25 robustness)

Evidence: synthetic run_suite. B1: an adapter hit a denied read, swallowed it, wrote `io_audit.json` and `imported_source.json`
as `verified`, then called `os._exit(0)`. The run ended `complete` and `verify_run` accepted it. It needs deliberate action, but it also defeats the
documented "an attempted blocked read fails even if caught".

Related visibility gaps. External reads are recorded and left at status `verified`, and `runtime_audits` summarizes only status and denied counts, so
`p13` (a label copy outside the project) and `p14` (subprocess read, disclosed) are invisible unless someone opens the JSON. Code inspection: paths under
`sys.prefix`, including project `.venv`, return with **no** record.

*Fix.* Return audits to the parent over a pipe inherited by the worker, not as files in `own_output`. Summarize external reads and subprocess
events in `run.json` and the report; the 27 clean origin-runs show only CPU-limit probes, so an allowlist is cheap. Refuse a freeze with non-allowlisted
external reads. Record non-package files read from the environment.

### R3. Assessment access is counted per suite only (medium; −0.25 statistics)

Evidence: synthetic. C3: two suites assessing 2004/2011 each report 1 access, 0 repeats, no warning. The new cross-suite warning (`report.py:56`) covers
development only. `suite_id` includes preparation source pins, so any change to `core.py`, `phase_policy.py`, `common.py`, the lock file and similar
files gives a new suite and a new counter. This has already happened once: `suite-v3` and `suite-v3-r5` are separate suites with identical
development files (21 files, identical hashes). No assessment has been opened yet, so the fix is cheap now.

*Fix.* Key exposure on target city-years, not suite ID. Warn when a run's assessment years were accessed in any known suite, and report a combined count.

### R4. The one-command gate has no trial count (medium; −0.25 statistics)

The new gates are honestly labelled as heuristics. But the loop makes iterating on three reused periods very cheap and prints a PASS/FAIL with no number of
candidates tried. Simulation (independent period deltas, zero true difference; `loop.out.txt`): G4 passes **32%** of the time for one candidate, 86% of the
time at least once across 5, and 98% across 10. The report's `known_development_trials` mixes panel runs and candidates (10 in my synthetic loop: 7
panel runs plus 3 candidate runs).

*Fix.* Print "candidate N tried in this suite" in `eval_model.py`, counting candidates separately from controls, and show it beside G4. State that a passing
gate after several tries is weak evidence.

### R5. Default test discovery reads protected real labels (low-medium; −0.25 provenance)

Evidence: §0, source inspection. On any machine with the pinned cache, plain `unittest discover` computes real CDD errors over 2011-2018. Hosted
CI skips this because the cache is absent. The authors' local 163/164 sweep very likely executed it too (I infer this because all 164 ran). So
`real_assessment_access_count: 0` in the round-5 manifest is accurate for the evaluator's assessment phase, and **not** a statement that no protected-year label
was ever read. These reads appear in no ledger.

*Fix.* Gate real-data tests behind an explicit environment variable, keep `verify_suite` on real suites out of default tests, and reword the manifest field to
"assessment-phase accesses".

### R6. Smaller items (no score effect)

- `panel.json` `role` is trusted on load: flipping one role changed `best_control` (synthetic). Re-derive it from `entries(profile)`.
- Default `--suite` is `suite-v3` (`4029caa…`), but the audited panel and freeze are on `suite-v3-r5` (`f2a07d…`), and only the latter has a panel. A default run will build a second panel in a second comparison group. Make one the single default.
- `eval_model.py` prints a raw traceback on adapter failure.
- Classifier still accepts `.json`, `.pkl`, `.npz`, `.orc`, `y2011` wide columns and `yr/var/val` long tables. This is documented as conservative.
- Relabelled city IDs pass the phase policy (C2). This is a deliberate act; the docstring's "cannot relabel" is about CSV conversion only and could say so.

## 6. Permanent limits versus actionable defects, and whether more than 92 is possible

| Limit | Status | Points kept off |
|---|---|---:|
| L1 reused labels | Permanent. Procedure cannot restore independence. | −4 |
| L2 few, non-stationary periods | Permanent with this data. | −2 |
| L3 private key, mask, precision | Permanent. | −1 |
| Tripwire is not an OS sandbox | Permanent on team machines in practice. | −0.5 |
| Freeze and asset ceremony | Inherent to the protocol; can only shrink. | −0.5 |
| **Total** | | **−8 → cap 92** |
| Actionable R1 to R5 | Repairable. | −1.75 |

Fixing every repairable item gives **92.0**, not more. Going above 92 would need a real OS sandbox on every team platform and a costless freeze workflow, which
together reach an arithmetic 93. I do not expect that before the 2026-11-10 deadline and would not award it for tests or documentation. L1 can move only with
genuinely unused, documented pre-2019 labels. Even those share years and climate with existing folds, so I would credit at most about one point after a person
confirms they were unused. **More than 92 is not an honest target for the remaining time.**

## 7. Prioritized further fixes

| # | Change | Acceptance check | Points |
|---|---|---|---:|
| 1 | Close R1 read and write paths; correct README | p01, p03, p06, p07, p08, p10, p18, w2 to w5 fail with audit events | +0.5 robustness, +0.25 usability |
| 2 | Move audits off the model-writable folder and surface external reads (R2) | B1 fails; p13 and p14 appear in `run.json` and the report | +0.25 robustness |
| 3 | Count assessment by city-year across suites (R3) | C3 reports a repeat and a warning | +0.25 statistics |
| 4 | Print candidate count at the gates (R4) | `eval_model.py` shows it; controls not counted | +0.25 statistics |
| 5 | Guard real-data tests and reword the manifest field (R5) | `discover` with cache present and no flag skips them | +0.25 provenance |
| 6 | R6 nits, especially the single default suite | Default command reuses the audited panel | none |

## 8. Commands and files

```text
.venv/Scripts/python.exe outputs/review-opus-round-5/fp.py             # before and after
gh run view 42471119448 --repo Meet2304/ProtoStar-Challenge             # jobs, logs, suite identity
curl Space API and 5 pinned rule files                                  # metadata and rules only
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"   # 164 OK; see §0
.venv/Scripts/python.exe outputs/review-opus-round-5/evidence_check.py  # development runs, Year<=2004
.venv/Scripts/python.exe outputs/review-opus-round-5/repros.py          # synthetic
.venv/Scripts/python.exe outputs/review-opus-round-5/loop_check.py      # synthetic loop, gate simulation
```

All Python ran with `PYTHONDONTWRITEBYTECODE=1` and Git calls with `GIT_OPTIONAL_LOCKS=0`. Sealed runs and ledgers were not written. No isolated worktree was needed
or used; Git objects were the authority for pins, and the workspace matched them. Repro p16 attempted a `datasets` load and its DNS lookup was blocked by the tripwire, so nothing left the machine.

**AI-use record.** Tool and model: Claude Code, Claude Opus 5.5, one agent. Tasks: source audit, test execution, synthetic repros, independent recomputation, this report.
A person should review §0 first, then R1 and R5, and should decide how to treat the protected-year test.
