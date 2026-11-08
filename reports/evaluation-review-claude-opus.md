# Independent review of the internal evaluation platform (Claude Opus 5.5)

Date: 2026-11-07. Reviewer: Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code, working alone (no
subagents). The question: is the construction and implementation of the test set thorough enough for
the team to rely on? No implementation, data, configuration, submission or experiment record was
changed. Scratch artifacts are in `outputs/review-opus/` (git-ignored).

**Data rule compliance:** no weather observation dated 2019 or later was downloaded, queried or used.
I made no attempt to access the private answer key. All experiments used the cached pinned training
data (ends 2018-12-31) or synthetic fixtures. The only web requests fetched the Space's rules and
source files, plus GitHub tag metadata for the CI action versions.

---

## 1. Verdict

**Score: 70 / 100. Confidence: medium-high on the individual findings, medium on the overall number.**

| Dimension | Weight | Score | One-line reason |
|---|---:|---:|---|
| Challenge-rule / scoring fidelity | 30 | **26** | Scorer is byte- and AST-identical to the live Space. The validity mask and baseline reproduce the organizers' published 9 / 63 and 212 / 175. Minor wording problems remain, and some things cannot be certified against the private key. |
| Statistical validation / test-set design | 30 | **15** | All 20 final-analogue years were already used to select the current model. Ranking by the mean of clipped fold scores puts models in the wrong order (shown on real runs). The bootstrap is conditional on the fixed periods and overconfident. Early folds are unrepresentative. |
| Implementation robustness | 20 | **14** | Strong validation, immutability and tamper detection. But the origin guard can be bypassed by the team's own loader (shown: Score 100, run recorded `complete`). The default adapter crashes when given any `--params`. Timeouts escape the CLI's error handler. |
| Reproducibility / provenance | 10 | **8** | Rebuilding the suite gives the same fingerprint. Weak points: external-CSV runs carry no model provenance, `outputs/` is git-ignored, and every recorded run comes from a dirty tree. |
| Team usability | 10 | **7** | Clear CSV contract and a portable report. Two documented paths fail (`--params` on the default adapter, a timeout traceback), and there is no enforced dev/holdout workflow. |
| **Total** | 100 | **70** | |

| Use | Ready? |
|---|---|
| Exploratory iteration | **Yes.** Fix the origin-guard bypass (M1) soon, because accidental look-ahead is easy. |
| Comparative model selection | **Not yet.** Fix M1–M4 first. Until then, compare raw per-metric skill fold by fold, not the ranked "final mean". |
| Final evidence (claims in the README or to judges) | **No.** suite-v1 has no untouched final-analogue years for the current model. |
| Predicting leaderboard rank | **No, by design.** Historical Score is not official Score, and official Score is not the prize ranking. Judges use Score only as evidence under two criteria worth 20% of the rubric. |

---

## 2. Sources and revisions checked

| Source | Revision / hash | Result |
|---|---|---|
| Live Space API `https://huggingface.co/api/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence` | `sha = 44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6`, lastModified 2026-11-07T18:26:14Z | Live `main` **is** the pinned revision |
| `scoring.py` (live `main`) | `066de39a…f396d` | Matches `evaluation/upstream/manifest.json` |
| `config.py` (live) | `0e7cea32…8374` | Matches |
| `utils.py` (live) | `55ceb09a…57ae` | Matches |
| `content/data.md` (live) | `369e0ff3…2782` | Matches |
| `content/about.md` (live) | `70c20591…f5a9` | Matches |
| `tabs/judging.py`, `leaderboard.py`, `scheduler.py` (pinned revision) | read only | Used to check judging and rank semantics |
| Training data `CMU-OSPO/protostar-climate-train-to-2018` | `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`, cached `annual_indices` sha256 `0b5f5fa9…e411` | Matches the suite manifest |
| `actions/checkout`, `actions/setup-python` tags (GitHub API) | `v7` exists for both | The CI workflow's action pins are valid; the workflow has never been executed |

---

## 3. Commands and experiments run

```bash
uv run python -m unittest discover -s tests -v          # 19 tests, OK (7.1 s), Python 3.13.1
uv run python code/evaluate.py verify                   # Verified suite 6080a208…4608
uv run python outputs/review-opus/e1_fidelity.py        # official-number reproduction, mask diff
uv run python outputs/review-opus/e2_robustness.py      # adapter/CLI/CSV minimal repros (temp synthetic suite)
uv run python outputs/review-opus/e3_design.py          # fold coverage, reuse overlap, clipping, bootstrap
uv run python outputs/review-opus/e4_parity.py          # combo adapter vs submission.csv; value precision
uv run python code/evaluate.py prepare --suite outputs/review-opus/suite-rebuild
uv run python code/evaluate.py prepare --annual data/raw/annual_indices/train-00000-of-00001.parquet --suite outputs/review-opus/suite-custom
uv run python code/evaluate.py run --name leaky-port --adapter outputs/review-opus/leaky_adapter.py:forecast --runs outputs/review-opus/runs
```

Key outputs (all reproducible from the scripts):

| Check | Result |
|---|---|
| City-metrics without a 1989–2018 baseline | **9 city-metrics → 63 rows** (official: 9 / 63) |
| Practice run: CDD trained ≤2010, scored 2011–2018 | 30-year average MAE **211.69**, trend **174.60** (official: 212 / 175) |
| Platform mask vs the team's `data/annual_valid.csv` | Identical for all four metrics (max diff 9e-13 on CDD). 1,184 `-inf` Max Tmax sentinels, all masked |
| `combo` adapter at origin 2018 vs `submission.csv` | Max abs diff 0.0005 (the 3-dp rounding in `combo_model.py`). The adapter wraps the real model faithfully |
| Suite rebuild from the pinned cache | Same fingerprint `6080a208…4608`: deterministic |
| Same official parquet passed via `--annual` | Different fingerprint `d36ee528…`, labeled `custom_local` |

---

## 4. Audit area (1): rule claims

| Claim (location) | Verdict | Evidence |
|---|---|---|
| Selecting or tuning variants by their scores on public 2019–2025 observations is prohibited (`evaluation/README.md:8-13`, `code/AGENTS.md`) | **Correct** | `content/data.md:124-130` (FAQ table: "Scoring model variants on 2019–2025 observations to choose or tune one ❌"). `about.md:29` (Nov 6 update) |
| Scoring a frozen, committed forecast afterwards is allowed if declared with its commit hash | **Correct** | `data.md:130` |
| Live = 2019–2021, final = 2022–2025; internal live = leads 1–3, final = leads 4–7 | **Correct** | `about.md:87-90`; `config.py:47-48` |
| "Do not use those years to choose models" (README:8) and "The live leaderboard is not a tuning signal" (AGENTS.md) | **Stricter than the official rule.** A defensible team policy, but it is not an official prohibition | `about.md:98` says the Live Score is there to "**check how your model is doing as you improve it**". `about.md:92` assumes some teams will tune on 2019–2021. The ban covers *public observations*, not the official live board. The README should say this is a self-imposed choice (a sound one: 3 noisy years), so teammates don't mistake it for a disqualification risk. |
| Hottest day uses both Tmax and Tmin missing counts | **Correct** | `about.md:94`; `common.py:28` |
| The 10-day rule applies to the baseline years; at least 10 valid years in 1989–2018 | **Correct, and reproduced numerically** | `about.md:27,94`; experiment E1 |
| 10 submissions per team per UTC day; the latest counts | **Correct** | `config.py:84`; `data.md:48` |
| "Forecast Score informs value and creativity rather than being a separate ranking" (README:312-315) | **Mostly correct; wording imprecise** | `judging.py:16`: "it is not a ranking by itself". But the Space *does* publish a ranked leaderboard (`leaderboard.py:84-93`), ranked on Score **rounded to 1 decimal** with ties broken by submission time. The prize ranking comes from judges. The report footer's "Forecast scores inform 20% of the judges' rubric" (`report.py:52`) reads as if Score were worth 20%. `about.md:140` says the two criteria *together* carry 20%, and Score is "evidence the judges read, not points added". |

---

## 5. Audit area (2): masks, baselines, precision, arithmetic

**Verified (strong):**
- `official_score.py` has the same AST as the upstream `score`/`ScoringError` (test), the same file hashes, and the live hashes match. The pure arithmetic is the official arithmetic.
- Validity mask (`core.py:101-111`): uses `<= 10`, unknown counts make a value invalid, `-inf` is masked only when the year is invalid, and a finite-valued `inf` is rejected. It matches the team's own `prepare_data.py` on real data exactly. The pinned data has **no** NaN missing counts, so the "unknown counts" path is untested by real data.
- Baseline eligibility is fixed at the origin for all seven years (`core.py:143-149`). This matches the official "1989–2018 for every year 2019–2025".
- Reference MAE uses the same eligible rows as the candidate, per split (`core.py:153-162`). The official 30-year average gives skill 0 exactly, up to float noise.

**Cannot be certified against private truth (uncertain, not defects):**
1. **CDD with `Days Both`.** The official key may or may not exclude CDD years where more than 10 days lack *either* temperature. Historical folds contain 7 / 12 / 4 / 0 / 0 such *scored* CDD rows (origins 1983…2011). These read low and inflate CDD MAE slightly. The impact is small and limited to early folds.
2. **Precision of the 2019–2025 labels.** Historical `Max Tmax` is 0% non-integer through 2017 and **41% non-integer in 2018**. CDD jumps from about 50% to 93% non-integer in 2018. The private key likely carries decimals, while historical hottest-day labels are integers. This barely affects MAE ordering, but historical hottest-day MAE is not exactly comparable to official MAE.
3. **Official reference-forecast code** (last year, trend) is not public. The trend matches the practice-run CDD figure (174.60 vs 175), which is good evidence for CDD, but it is not proof for the other metrics.
4. **The official baseline MAE denominators and scored-row mask** are private. Only the published counts (9 / 63) could be checked, and they match.

---

## 6. Audit area (3): fold design and statistics

### P1. Selection reuse: every final-analogue year is already contaminated for the current model (proven, high)
`combo_model.py:40-43,98-124` chose the per-metric methods on test windows 1987–1994, 1995–2002, 2003–2010 and 2011–2018. Against suite-v1 (E3):

| Origin | Target years reused in combo selection | Final-split years reused |
|---|---:|---:|
| 1983 | 4 / 7 | **4 / 4** |
| 1990–2011 | 7 / 7 | **4 / 4** |

The only untouched years are 1984–1986 (live analogue of one fold). The README discloses the overlap (`README:280-286`), but "useful comparative diagnostics" understates it. Any comparison of a new model *against combo* on suite-v1 is biased toward combo: combo's method choice was fitted on these exact labels. So `paired-comparison.json` (combo +2.21 over trend, CI [1.33, 3.81]) is **not** evidence that combo generalizes better.
*Fix:* label suite-v1 "development: reused". For the combo-vs-challenger question, either (a) also build a pre-1984 suite (for example origins 1948, 1955, 1962, 1969, 1976; see P5 for caveats), or (b) re-run combo's *selection procedure* inside each fold (nested: select the method on windows ending at or before the origin), rather than its frozen output.

### P2. Ranking on the mean of clipped fold Scores puts models in the wrong order (proven, high)
`core.py:367-375` summarizes each split as the mean of per-fold `100·max(0, mean skill)`, and `core.py:399` ranks on it. On the recorded real runs:

| Run | Raw mean final skill | Mean clipped fold Score | Internal rank |
|---|---:|---:|---:|
| Linear trend | **−0.0685** | 0.998 | **4** |
| 30-year average | 0.0000 | 1.7e-15 | 5 |
| Last year repeated | −0.3487 | 0.0 | 6 |

Linear trend is ranked *above* the 30-year average although its raw skill is far worse. It gets credit for one positive fold (4.99), while its four negative folds are floored at 0. Combo's mean clipped Score (3.21) is about 70% higher than its clipped mean raw skill (1.89). Float noise (1.7e-15) also breaks the tie between the average and last-year runs. The README admits that "clipping can make average fold Score look more favorable" (`README:130`) but still ranks on it.
*Fix:* rank by mean **unclipped** skill (or per-metric skills), and show `100·max(0, mean raw skill)` as the "official-style" summary. Keep fold Scores as display only. Round or tolerance-compare before ranking.

### P3. The bootstrap interval is conditional and overconfident (proven by comparison, medium-high)
`compare_runs` (`core.py:405-457`) resamples only the four years *within* each final fold, keeps the five periods fixed, and differences the *clipped* Scores.
- Each fold has only 35 distinct year multisets. With n=4 clusters, the percentile bootstrap understates variance (by roughly a factor of (n−1)/n before discreteness).
- It ignores **between-period variance**, which is the main uncertainty for a *new* period (2022–2025). The per-fold Score deltas (combo − trend) are [0, 0, 2.57, 2.56, 5.92]: mean 2.21, SD 2.44. A fold-level paired t interval (4 df) is **[−0.82, 5.24]**, which includes 0. The reported bootstrap interval [1.33, 3.81] does not.
- Clipping hides signal: in folds 1983 and 1990 both models clip to 0, so the delta is 0, yet their raw-skill deltas are +10.4 and +13.5 points. In raw skill, combo beats trend in **5 / 5** folds. That is far stronger evidence than the clipped interval suggests, but it is contaminated (P1).
*Fix:* report the per-fold paired raw-skill deltas, their sign count, and a fold-level t or permutation interval. If you keep the bootstrap, use a two-stage design (resample folds, then years) on raw skill. Label it "conditional on these periods".

### P4. Tuning leakage by design: no development/holdout separation (design flaw, medium-high)
The README advises "use earlier origins for development and later origins to assess a frozen choice" (`README:280-282`). But `run_suite` scores **every** origin on **every** run (`core.py:342-365`), and the report shows all folds. Each iteration therefore reveals the "assessment" folds. Nothing records a freeze.
*Fix:* add `development_origins` / `holdout_origins` to the config. Withhold holdout scores unless the run references a committed freeze record (git commit + params hash), and log every holdout unseal.

### P5. Early folds do not represent the 2019–2025 task (design limitation, medium)
| Origin | Scored rows / 5,880 | City-metrics without baseline | Station changes in target window |
|---|---:|---:|---:|
| 1983 | 5,421 | 24 | 6 |
| 1990 | 5,358 | 26 | 10 |
| 1997 | 5,509 | 26 | **17** |
| 2004 | 5,562 | 32 | 4 |
| 2011 | 5,706 | 15 | 0 |
| Official 2019–2025 | ≈5,817 | 9 | none listed 2019–2023 |

The early folds also have baseline windows (1954–1983, 1961–1990) that include the mid-century cooling period. Trend-type methods are penalized there (trend avg-temp skill −0.18 / −0.16 in folds 1983 and 1990, versus +0.25 in fold 2011). Equal fold weighting therefore mixes regimes and station discontinuities that the official test does not contain.
*Fix:* report per fold, flag the 2004 and 2011 folds as the most representative, and add an option to exclude city-years with a station change in the target window. A pre-1984 suite (P1) would be even less representative, so treat it as a secondary check only.

### Correct in the fold design
The windows don't overlap. The 7-year horizon and 1–3 / 4–7 leads match the official task. Every fold requires the full 5,880-row grid. Eligibility is fixed at the origin. The nested training sets are expected for rolling-origin evaluation.

---

## 7. Audit area (4): adapters, validation, provenance, usability

### R1. The origin guard can be bypassed by the team's standard loader (proven, high)
The worker checks only the `train` frame it passes in (`worker.py:41-43`), and `reference()` does the same (`adapters.py:14-16`). `assert_no_post_cutoff` enforces only the fixed `CUTOFF_YEAR = 2018` (`common.py:21,41-49`). An adapter that calls the team's usual `common.load_valid_annual()` (`common.py:52-59`), or reads `data/raw/…`, gets every target year up to 2018. My repro `outputs/review-opus/leaky_adapter.py` did exactly this: **final analogue mean 100.0, run status `complete`**. A *partial* leak (for example, recomputing a 10-year mean from the full file) would give plausible, inflated scores with no signal that anything is wrong. This is the most likely way a well-meaning teammate porting `combo_model.py`-style code would corrupt the register. Reading `private/answer_key.csv` deliberately also scores 100 (documented at `README:204-207`, but undetected).
*Fix:* in `worker.py`, set an environment variable such as `PROTOSTAR_FOLD_CUTOFF=<origin>`. Make `assert_no_post_cutoff` and `load_valid_annual` use `min(2018, env)`. Run the worker from a temp cwd with only `train.csv` and `targets.csv` copied in, so the private key is not a sibling file. Add a canary test that a `load_valid_annual()` adapter fails.

### R2. The documented default adapter crashes with any `--params` (proven, medium)
`adapters.py:44-45`: `combo(..., **params)` forwards to `reference(train, targets, origin, seed, method="combo", **params)`, and `reference` accepts no extra keyword arguments. Repro: `TypeError: reference() got an unexpected keyword argument 'x'`. The README documents `--params` for adapters (`README:190-193`).
*Fix:* `def combo(train, targets, origin, seed=42, **params): if params: raise ValueError("combo takes no params")`. Or accept and record them.

### R3. A timeout produces a traceback, not the CLI's error message (proven, low)
`subprocess.TimeoutExpired` (`core.py:360-361`) is not a `RuntimeError`, so `evaluate.py:96` (the `except (ValueError, FileNotFoundError, FileExistsError, RuntimeError)`) does not catch it. The run record is still correctly marked `failed`.
*Fix:* catch `subprocess.SubprocessError` in `run_suite` and re-raise it as `RuntimeError` with the log path.

### R4. Lenient prediction typing (proven, low)
`validate_predictions` (`core.py:236-240`) accepts `predicted_value=True` (becomes 1.0) and years such as `"2012.0"`. Neither is harmful to scoring, but booleans point to a broken adapter.
*Fix:* require numeric dtypes and reject `bool`.

### R5. Provenance gaps (proven, low-medium)
- External-CSV runs (`score`) record only a **relative, OS-specific** `prediction_dir` (`outputs\evaluation\external-smoke`). They keep no hash of the original files (only the normalized copy) and no model code or author. Nothing ties those scores to a model.
  *Fix:* require `--provenance model.json` (repo URL + commit, author, training cutoff, params) and hash the input files.
- Passing the identical official parquet via `--annual` produces a different suite ID and a `custom_local` label (`core.py:199-201`). Provenance is decided by the flag used, not by the content hash.
  *Fix:* if the sha256 equals the pinned file's, label it official.
- All six recorded runs ran from a dirty tree (9–12 changed paths). The code snapshot covers only `code/**/*.py`.
- `leaderboard()` and `compare_runs()` trust `run.json` without checking `artifact_sha256`, and don't re-run `verify_suite`.
- `outputs/` is git-ignored, so the shared register exists only on one laptop. The rebuild is deterministic, which mitigates this for the suite, but not for runs.

### Strengths (verified)
- Immutable suite creation (atomic rename, refuses to overwrite).
- Full-file hash fingerprint; tampering is detected.
- Strict grid contract including unscored rows.
- No `astype(bool)` on CSV strings (`core.py:257-259`).
- Fresh subprocess with seeds and thread limits; byte-identical repeat runs.
- Portable, escaped HTML report.
- Clear CSV path for teammates on other frameworks.
- The `combo` adapter reproduces `submission.csv` to 3 dp.

---

## 8. Audit area (5): test gaps

The 19 tests are good contract tests on synthetic data. Missing:
1. **Real-data fidelity tests** (skip if the cache is absent): 9 city-metrics / 63 rows at 2018, and practice-run CDD 211.7 / 174.6. These are the strongest fidelity anchors available and are currently unasserted.
2. **Origin-leak canary**: an adapter calling `load_valid_annual()` must fail (R1).
3. **`combo` with params** and **adapter–submission parity** at origin 2018 (R2).
4. **Leaderboard ordering**: a model with negative raw skill but one positive fold must not outrank the 30-year average (P2). Float-noise ties.
5. **Bootstrap with a known nonzero difference**: today only the zero-difference identity is tested (`test_evaluation.py:215-216`). Also missing: behavior when both models clip to 0.
6. **Timeout path** and CLI error handling (R3).
7. **Dtype strictness** (bool, string years) (R4).
8. **Independent check of the `last_year` and `trend` references** (only indirectly covered).
9. **Report with multiple suites**, and with failed and running runs mixed in.
10. **The CI workflow has never run.** Run it once, because Linux path handling and `metadata_url` quoting are untested.

---

## 9. Prioritized fixes

**Must fix before the team relies on it for model selection** (each is small: one function or one config field):

| # | Item | Effort |
|---|---|---|
| M1 | Enforce the fold origin in all guarded loaders inside the worker. Run the adapter from a temp cwd without access to the private key. Add a canary test. (R1) | ~1 h |
| M2 | Rank and summarize on mean **unclipped** skill. Show clipped Score as secondary; compare with a tolerance. (P2) | ~30 min |
| M3 | Replace the headline interval with fold-level paired raw-skill deltas, a sign count and a t/permutation interval. Label the bootstrap "conditional". (P3) | ~1 h |
| M4 | Mark suite-v1 as reused for combo. Add a dev/holdout split with a freeze record. For combo-vs-challenger questions, use nested selection inside each fold. (P1, P4) | ~2 h |
| M5 | Fix `combo(**params)`. (R2) | 5 min |

**Later improvements:**
- Catch `TimeoutExpired`; strict dtypes. (R3, R4)
- Provenance JSON and input hashes for external CSVs. Content-based official labeling. Verify `artifact_sha256` in the report and compare. (R5)
- Station-change exclusion option. Per-fold emphasis on 2004 and 2011. Optional pre-1984 secondary suite. (P5)
- `Days Both` sensitivity report for CDD. A note on 2018+ decimal precision. (§5)
- README wording: the live-leaderboard ban is a team policy; the official rank uses 1-dp rounding; the report footer's "20%" wording. (§4)
- Add the real-data and edge-case tests in §8. Run CI once.

---

## 10. What this platform can and cannot tell the team

- **Can:** reproduce the official scoring arithmetic and the published validity and baseline conventions on historical data. It gives a sound, reproducible harness for *relative* comparisons between models that were **not** tuned on 1984–2018, once M1–M3 are in place.
- **Cannot:** certify scores against the private key (`Days Both` treatment, precision, private denominators). It cannot give untouched evidence for the current combo model. It cannot predict official Score, official leaderboard rank (rounded to 1 dp, time tiebreak), or the judges' ranking. Score is evidence under *Scientific Value* and *Technical Creativity*, not a ranking input.

---

## AI-use record (for the README's *AI Tools and Agents* section)

- **Tool / model:** Claude Code with Claude Opus 5.5 (`claude-opus-5-5`). One agent; no subagents.
- **Tasks:** read the evaluation code, tests, docs, suite and run artifacts. Compared the live and pinned Space rules and source by hash. Ran the test suite, suite verification and rebuild, and read-only experiments on pre-2019 data, plus synthetic minimal repros. Wrote this report.
- **Not done:** no implementation, data, config, `submission.csv` or existing experiment record was changed. No 2019+ weather data or private answer key was accessed. Nothing was uploaded or submitted.
- **What a person should review:** the R1 leak repro (`outputs/review-opus/leaky_adapter.py`), the clipping and ranking analysis (P2), the fold-level interval (P3), and whether to treat the live leaderboard as a team-policy restriction (§4).
