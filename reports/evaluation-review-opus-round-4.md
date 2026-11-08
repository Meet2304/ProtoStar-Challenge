# Independent evaluator audit, round 4 (Claude Opus 5.5)

Date: 2026-11-08. Reviewer: Claude Opus 5.5 (`claude-opus-5-5`), working alone and read-only.
I wrote only this report plus scratch files under `outputs/review-opus-round-4/`. I did not read or
score any real assessment label or legacy protected 2004/2011 label, did not query observed 2019+
weather, did not touch the private key, and made no submission.

## 1. Score: 89 / 100 (round 1: 70; round 3: 81)

| Dimension | Weight | R1 | R3 | **R4** | Main reason |
|---|---:|---:|---:|---:|---|
| Rule / scoring fidelity | 30 | 26 | 28 | **29** | Scorer unchanged and live-identical. `Days Both` and precision sensitivities now exist and reproduce numerically. Only the private key stays uncertifiable. |
| Statistical validation / test design | 30 | 15 | 21 | **23.5** | Legacy gate closed and exposure counted. Reused labels and few periods are permanent. A documented custom-suite route still bypasses the gate (N2). |
| Implementation robustness | 20 | 14 | 16 | **18** | I2, I3, I4 and I8 meet acceptance. The tripwire misses parquet reads and all network use, and project files outside `data/` are unobserved (N1, N3). |
| Reproducibility / provenance | 10 | 8 | 8 | **9.5** | Committed, cross-OS CI, equal logical suite ID. The real dev evidence predates the pinned commit (N6). |
| Team usability | 10 | 7 | 8 | **9** | Inferred freezes and assessment baselines work. One documented guarantee is wrong (N1); the freeze ceremony remains. |
| **Total** | 100 | 70 | 81 | **89** | |

| Dimension | Deductions |
|---|---|
| Rules −1 | L3 −1 (permanent). |
| Statistics −6.5 | L1 −4 and L2 −2 (permanent); N2 −0.5. |
| Robustness −2 | N1 −1 (of which −0.5 is the permanent no-sandbox limit); N3 −0.5; N4 −0.25; N5 −0.25. |
| Provenance −0.5 | N6 −0.5. |
| Usability −1 | N1 documentation error −0.5; workflow ceremony −0.5. |

**Confidence: moderate-high.** The likely range is 87 to 91 on this rubric. The rubric is unchanged, and
no points were moved between dimensions.

| Use | Readiness |
|---|---|
| Exploration | **Ready.** |
| Comparative selection | **Ready with conditions:** development phase on the default suite, per-period raw deltas, and a human check of each adapter for parquet, `hf://`, network and custom-suite use until N1 and N2 are fixed. |
| Independent final evidence | **No, and not repairable.** Assessment years are reused. A freeze is procedural discipline, not independence. |
| Leaderboard or rank prediction | **No, by design.** |

## 2. Audited scope and evidence

**Authority.** Commit `daa67ab80b18dd58d4d8a140239d8ec2baf73e08`, read via `git show`. The normalized
`source_pins` canonical hash, computed from Git objects (25 files), is
`4831e22636fdc94fbf6e159eef57d157625972806566c16ab3de26e7e6e1a18a`, which **matches the expected value
before and after this review**. `submission.csv` SHA256 was unchanged at `2c9b61da…4091` before and after.

**Excluded from scope.** The uncommitted `controls.py`, `scorecard.py`, `tracking.py`, `panel.py`,
`scripts/*`, `tests/test_controls_panel.py`, docs and notebooks. The one workspace difference in
pinned files is the `("evaluation.controls","forecast")` allowlist entry in `protocol.py`. **That entry
makes an unreviewed module exempt from asset declaration. This score does not cover it.** A re-check
is needed if the final integrated source differs from the pinned commit.

**Rule source.** Space API `sha` = pinned `44b8ec9f…`. Live `scoring.py`, `config.py`, `utils.py`,
`content/data.md` and `content/about.md` match the manifest hash prefixes. `official_score.py` has the same hash
as in round 3 (`a610f9bf…`).

| Evidence type | What it covers |
|---|---|
| **Hosted CI (independently confirmed via `gh`)** | Run 42468348316, headSha = `daa67ab`. Ubuntu, Windows and `portable-suite-identity` all succeeded. 124 tests, 1 skipped, on both OS. Same logical suite ID on both: `98aafceb…5845`. **Best evidence for the pinned core.** Synthetic fixtures only. |
| **Local tests, shared workspace** | Selected modules: protocol-v4, runtime audit, exposure, classification, portability, bundle, quality diagnostics (52 tests, OK), and `test_asset_workflow` (1 test, OK). These execute workspace source, which differs from the pin only by the allowlist line. The workflow test copies workspace source into a temp fixture, as it always does. |
| **My synthetic repros** | `outputs/review-opus-round-4/repros.py`, log `repros.out.txt`. Workspace evaluator, synthetic data only, no source copy or checkout. Worker runs use a synthetic project root. Run-suite cases are development-phase on synthetic suites. |
| **Real development read** | Run `…2883ad9e10` and bundle `47ddf741…` read-only. Raw caches loaded with a pushdown `Year<=2004` filter (`devcheck.out.txt`), so no assessment target year (2005+) entered memory. |
| **Code inspection only** | Exposure-counter edge cases, statistical design, report text. |

**Concurrency.** One repro attempt failed with `model/evaluator source changed during execution`,
because in-scope uncommitted `code/**.py` files were being edited. **That is the source-stability guard
working correctly, not a core defect.** A retry succeeded. A separate failure was my own fixture cleanup
(read-only Git objects on Windows). A first `unittest` error was my invocation (missing `tests/` on
the path).

## 3. Acceptance of round-3 items

| Item | Verdict | Evidence |
|---|---|---|
| I1 legacy gate | **Pass** | Hand-built synthetic schema-1 suite verifies, then both phases are refused with no ledger write (B2). See N2 for a separate route. |
| I2 helper drift | **Pass, with a gap** | Undeclared sibling import fails (a13); tests cover drift. Non-registered loads are not seen (N3). |
| I3 mislabelled labels | **Pass, with a gap** | Label-schema table as `static_context` is rejected (d1, d7), as are pinned hash and raw paths. Other shapes pass (N4). |
| I4 descendant docs commit | **Pass** | CLI workflow: docs-only descendant assesses. Tests: source change and non-ancestor fail. |
| I5 line endings | **Pass** | Normalized identity plus physical seal; CI proves equal IDs across OS. |
| I6 assessment baselines | **Pass** | `baselines --phase assessment --freeze` runs and logs in the CLI workflow. |
| I7 exposure counters | **Pass** | Report shows trials, accesses, unique freezes and repeats; ledger gaps become "unknown". Scope caveat in N2. |
| I8 installed environment | **Pass** | Hash is frozen and enforced. Names and versions only, as disclosed. |
| I9 scientific diagnostics | **Pass, independently reproduced** | My separate pandas calculation matches the real bundle across 3 origins × 4 variants × 4 metrics on final-split row counts and skill (worst difference 3.3e-16). Stable cities 168, 170, 165 of 210. Diagnostic-to-default final skill deltas are small (largest 0.0076), and the bundle is development-gated. |
| I10 committed plus CI | **Pass** | See §2. |
| L4 accidental raw read | **Round-3 canary passes; guard incomplete** | `DATA_DIR` CSV, swallowed error and `os.open` are all caught (a01, a02, a07). The sibling parquet is not (N1). |

## 4. New defects, most serious first

All are repairable. Repro labels refer to `repros.out.txt`.

### N1. The tripwire misses parquet reads and all network use (high within robustness)

`runtime_audit.py:81` acts only on Python's `open` audit event, and pyarrow's C++ reader never raises it.

| Repro | Adapter call on a synthetic `data/` file | Result |
|---|---|---|
| a03 | `pd.read_parquet(data/raw/annual_indices/train-00000-of-00001.parquet)` | rc=0, `io_audit` **verified**, no event |
| a04 | `pyarrow.parquet.read_table(...)` | rc=0, verified, no event |
| a05 | `pyarrow.csv.read_csv(data/annual_valid.csv)` | rc=0, verified, no event |
| a15 | `socket.connect` to loopback closed port | rc=0, verified, no event |

This matters more than the "deliberate native I/O" limit implies. The file is the repository's actual
raw label cache, the file `pinned_annual()` itself uses, and `pd.read_parquet` is ordinary one-line
pandas. `evaluation/README.md:443-444` states raw repository data is "blocked", which is false for
parquet, and the disclaimer at `:448` mentions only *deliberate* native I/O. `tests/test_runtime_audit.py:75`
checks CSV forms only.

Network is unobserved as well. `code/prepare_data.py:21` already uses the `hf://` idiom, and nothing
would flag a model fetching outcomes, or weather after the cutoff, at run time. I tested only loopback.

*Fix.* In the worker, before adapter import:
1. Wrap `pyarrow.fs.LocalFileSystem` opens, `pa.OSFile`, `pa.memory_map` and the `parquet`, `csv`, `feather` and `dataset` entry points, applying the same deny rule.
2. Deny the `socket.connect`, `socket.getaddrinfo` and `urllib.Request` audit events unless a declared flag allows them, and record them.
3. Add a03, a04, a05 and a15 as tests.
4. Correct README:443-448.

### N2. A documented custom suite bypasses the assessment gate (medium)

`validate_config` allows `assessment_origins: []` (`core.py:94`), and `prepare --config` is documented
(`README:376`). B1: a schema-3 suite declaring origins 2004 and 2011 as *development* completed with status
`complete`, no freeze and no ledger. Exposure counters are per suite, so the default suite would still
report zero assessment accesses. `diagnose` would also accept such a run. This reopens the purpose of
I1 by a different path, although the literal I1 check passes.

*Fix.* In `prepare`, if the source is the pinned annual file and any `development_origins` is in the default
`assessment_origins`, refuse unless an explicit flag writes a ledger event. Add year-overlap warnings across
suites to the report.

### N3. Project reads outside protected folders are unobserved, so freezes can drift (medium)

`runtime_audit.py:108` returns with no record for any path outside `data/`, the suite and evaluation outputs.

- a08: reading a label copy under `notebooks/` gives rc=0, verified, no event.
- a09 and C1: an adapter reading undeclared `code/cfg.json`. After a committed change to it, `verify_freeze`
  **passes**, because pins cover only `*.py` and lock files (`protocol.py:44`). This is the non-Python twin of I2.
- a10 and a11: `exec(read_text())` and `spec_from_file_location` without registering in `sys.modules` run
  undeclared code, and the import audit misses them (`runtime_audit.py:136`).

*Fix.* Default-deny inside the project root: allow pins, declared assets, supplied inputs, own output and the
interpreter environment; record and fail anything else. Module loading goes through the same audited open, so
this closes a08 to a11 together.

### N4. Label classifier misses common shapes (low-medium)

`assets.py:88,106`. Accepted as `static_context`: wide city-by-year matrix (d2), long `city_id,year,metric,value`
(d3), `.tsv` (d4), `.csv.gz` (d5), `.feather` (d6). The check is honestly documented as conservative.

*Fix.* Inspect tsv, gz and feather; treat four-digit-year column names as temporal; add `metric`+`value` as an outcome pattern.

### N5. Missing audits do not fail the run (low)

B3: an adapter that writes `adapter_output.csv` itself and calls `os._exit(0)` ends `complete`, with
`runtime_audits` showing `not_recorded`, and `verify_run` accepts it (`core.py:510`). It needs deliberate action.

*Fix.* Before scoring, require both audit artifacts to exist with status `verified`.

### N6. Real evidence predates the pinned commit (low, process)

Real dev run `…2883ad9e10` has pin hash `183071ff…`, against `4831e226…`, with recorded commit `16e0f845` and 28
dirty entries. Only `runtime_audit.py` differs in the two manifests, with no plausible numeric effect, and
the run records this honestly. Also, the manifest at `daa67ab` still names `183071ff…`; the corrected
version exists only uncommitted.

*Fix.* After final integration, re-run development, baselines and diagnostics from a clean commit, and commit a manifest
that matches.

### Checked, no defect found

Symlink, short-alias and casing paths; snapshot immutability; descendant ancestry; failed-attempt versus access counting;
`--from-run` rejection cases; ranking on raw skill; nested selection boundaries; bundle sealing; phase gate
on `diagnose`. Nits only: a fresh suite with no ledger folder reports "unknown" instead of zero; a 3-period bootstrap
is close to the observed range and is suitably labelled "sensitivity"; four frozen comparisons count as four accesses.

## 5. Permanent limits versus defects, and whether more than 92 is possible

| Limit | Status | Points kept off |
|---|---|---:|
| L1 reused labels | Permanent. Procedure cannot restore independence. | −4 |
| L2 few, non-stationary periods | Permanent with this data. | −2 |
| L3 private key, mask and precision | Permanent. | −1 |
| Python tripwire is not an OS sandbox | Permanent on team machines in practice. | −0.5 |
| Freeze and asset ceremony | Inherent to the protocol; can only shrink. | −0.5 |
| L5 author-declared weight cutoffs | Disclosed; no separate deduction. | 0 |

Fixing every repairable item in §4 yields **about 92**. On this rubric the arithmetic cap is **93**, and only if
the worker runs in a real OS sandbox on every team platform and the ceremony costs nothing. I do not expect that
before the 2026-11-10 deadline, and I would not award it for tests or documentation alone.

**More than 92 is therefore not a realistic target by consistent scoring, and it should not drive the next round.**
The limits cannot be fixed by more process, and independent evidence cannot be manufactured from the same reused years.
The only legitimate way to move L1 is genuinely new, documented, unused labels, for example pre-2019 records for
cities truly not seen by the team, with end dates capped at 2018-12-31. Even that shares years and climate with existing
folds, so I would credit about one point at most, after a person confirms they were unused.

## 6. Prioritized further improvements

| # | Change | Acceptance check | Points |
|---|---|---|---:|
| 1 | Close parquet, pyarrow and network gaps; correct the README (N1) | a03, a04, a05, a15 fail or are flagged | +0.5 robustness, +0.5 usability |
| 2 | Guard default-assessment years in custom suites; report overlap (N2) | B1 refused unless a flag writes a ledger event | +0.5 stats |
| 3 | Project default-deny for reads (N3) | a08 to a11 fail; C1 no longer undetected | +0.5 robustness |
| 4 | Fresh real dev evidence plus matching committed manifest (N6) | Run pin hash equals final source | +0.5 provenance |
| 5 | Require audit artifacts before scoring (N5) | B3 fails | +0.25 robustness |
| 6 | Broaden label classifier (N4) | d2 to d6 rejected, census exception intact | +0.25 robustness |
| 7 | Review `controls.py` and its allowlist entry before accepting it | Fresh audit of the integrated source | Scope condition |

Optional statistical work, not scored as a defect because round 3 did not deduct it: region-clustered city sensitivity
within each period, and explicit reporting of how many candidates were compared when a winner is named.

## 7. Commands

```text
.venv/Scripts/python.exe outputs/review-opus-round-4/fp.py        # before and after
gh run view 42468348316 --repo Meet2304/ProtoStar-Challenge       # jobs, log, suite identity
curl Space API and 5 pinned rule files                            # metadata and hashes only
.venv/Scripts/python.exe -m unittest <8 selected modules>         # 52 + 1 tests OK
.venv/Scripts/python.exe outputs/review-opus-round-4/repros.py    # A, B, C, D, synthetic
.venv/Scripts/python.exe outputs/review-opus-round-4/devcheck.py  # dev only, Year<=2004
```

All Python ran with `PYTHONDONTWRITEBYTECODE=1`, and Git-touching runs with `GIT_OPTIONAL_LOCKS=0`, to reduce interference.

**AI-use record.** Tool and model: Claude Code, Claude Opus 5.5, one agent. Tasks: source audit, selective test
execution, synthetic repros, this report. A person should review N1 and N2 first, and should decide whether the
`evaluation.controls` allowlist entry is acceptable.
