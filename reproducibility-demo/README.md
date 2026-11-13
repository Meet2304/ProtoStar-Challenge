# Reproducibility demo: rebuild our submission live

One command downloads our published code and the raw challenge data, runs the whole Placeholder_F4 pipeline in
an empty folder, and shows on screen that the rebuilt `submission.csv` has the same SHA-256 hash as the file we
published and submitted.

```bash
uv run reproducibility-demo/verify_submission.py
```

A SHA-256 hash is a 64-character fingerprint of a file's bytes. If a single byte of a file changes, the hash
changes completely, so two files with the same SHA-256 are, for all practical purposes, identical.

## What it proves

| Claim | How the script checks it |
|---|---|
| The forecasts we submitted are the ones our code produces | Rebuilds `submission.csv` from raw data and compares its SHA-256 with the published file |
| The hash we state in our README and slides is the real one | Reads the sentence from the published README and compares it with the hash of the published file |
| Nothing precomputed is reused | The build folder starts with code and input tables only: no `submission.csv` and no `data/processed/` |
| The downloaded code and data are what is on Hugging Face | Every downloaded file is checked against Hugging Face's own checksums (SHA-256 for large files, git blob hash for the rest) |
| The output is valid | Runs the challenge format check (`code/check_submission.py`) |
| The model does not peek at the future (optional) | `--with-leak-test` runs `code/f2/test_no_lookahead_f4.py`: it replaces every year after a forecast origin with garbage and checks that the forecast does not move |

Sources the script uses (all public, no token needed):

| What | Where | Pinned at |
|---|---|---|
| Final submission (code, inputs, `submission.csv`) | [`DragonDare/protostar-placeholders-f4`](https://huggingface.co/datasets/DragonDare/protostar-placeholders-f4) | `4321975` |
| Model release (second published copy of `submission.csv`) | [`Meet2304/protostar-placeholders`](https://huggingface.co/datasets/Meet2304/protostar-placeholders) | `099a52d` |
| Raw challenge data (ends 2018-12-31) | [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018) | `bbbdebd` (the revision pinned in the published `code/common.py`) |
| Stated hash | `8344e656c942e95fd522d9e7f10f2d4c5a4847b82650f4e6de1d5c9d4633bc37` | README of the final submission, `submissions/final_2026-11-10/NOTE.md` on the `kohav` branch, and the slides |

The script runs the published pipeline unchanged, as separate processes, in this order:
`code/f2/f2_daily.py` → `code/f2/f2_cddgrid.py` → `code/make_submission.py` → `code/check_submission.py`.
It only sets `F2_RAW` (an option the published code already reads) to the folder holding the verified raw data.

## Presenter guide

### A. One-time setup (the day before, on the presentation laptop)

1. **Install uv** (the Python package and environment manager the repository uses). It also installs the right
   Python (3.13) by itself.
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Close and reopen the terminal, then check: `uv --version`.
2. **Get the repository and this branch:**
   ```bash
   git clone https://github.com/Meet2304/ProtoStar-Challenge.git
   cd ProtoStar-Challenge
   git checkout Meet/reproducibility-demo
   ```
   (Once the branch is merged, `main` works too.)
3. **Pre-download and verify the inputs** (about 80 MB; also installs the pinned packages into uv's cache):
   ```bash
   uv run reproducibility-demo/verify_submission.py --prefetch
   ```
   It ends with "Inputs downloaded and verified ... Ready for the live demo". The downloads are kept in
   `reproducibility-demo/.cache/`. They are inputs only; on the day every computed file is rebuilt from scratch, and
   every cached file is checked against Hugging Face's checksums again before use.
4. **Do one full rehearsal** and note how long it takes on this laptop:
   ```bash
   uv run reproducibility-demo/verify_submission.py
   ```
   It should end with a large green **MATCH** panel. On the development laptop the full run took about 4 minutes
   (see *Timing* below). Each run writes `reproducibility-demo/runs/<UTC time>/receipt.json` (all hashes, timings,
   package versions) and the full logs of every step.
5. **Set up the terminal** so the hashes fit on one line: Windows Terminal (not the old console), maximised,
   at least 120 columns wide, font size 16 to 20 so the back row can read it, dark theme.

### B. Just before you present (2 minutes)

1. Open the terminal in the repository folder: `cd ProtoStar-Challenge`.
2. Clear the screen: `cls` (Windows) or `clear` (macOS / Linux).
3. Check the network, if you have time: `uv run reproducibility-demo/verify_submission.py --prefetch --fast`
   (a few seconds when the cache is warm). If there is no network, use the offline command in section D.
4. Optional: have the published dataset page open in a browser tab:
   <https://huggingface.co/datasets/DragonDare/protostar-placeholders-f4/blob/main/submission.csv>.

### C. Live, step by step

Type the one command:

```bash
uv run reproducibility-demo/verify_submission.py
```

Add `--step` if you want the script to wait for Enter before each step, so you can talk at your own pace.

What the audience sees, and what to say:

| Step | On screen | What to say |
|---|---|---|
| Title panel | The claim, the test, the pinned sources and this machine | "We claim anyone can rebuild our forecasts from our public code. Let's do it now, from scratch." |
| 1. Read the published claim | The sentence from our published README with `SHA-256 8344e656…3bc37`, and the full hash of the published `submission.csv` | "This is the fingerprint we published. The file on Hugging Face has exactly this fingerprint." |
| 2. Code and inputs into an empty folder | File counts verified against Hugging Face checksums; "The build folder has **no submission.csv** and **no processed data**" | "We start with only code and input tables. No answers are sitting in the folder." |
| 3. Raw challenge data | The pinned revision from our code and the checksums of the four raw tables (10.2 million daily rows, ending 2018-12-31) | "This is the organisers' raw data, cut at the end of 2018, exactly the version our code pins." |
| 4. Clean the daily weather records | Live output of `f2_daily.py` (~1 min) | "Our cleaning removes impossible readings, outliers and spikes from 10 million daily records." |
| 5. CDD response tables | Live output of `f2_cddgrid.py` (~40 s) | "We recompute cooling degree days from daily temperatures, so warming feeds CDD the way physics says it should." |
| 6. Run the model | Live output of `make_submission.py` (~1 min): rows per metric, "Wrote 5,880 Placeholder_F4 forecasts" | "This is the submitted model: 210 cities, 7 years, 4 metrics." |
| 7. Format check | "OK: submission.csv has all 5880 forecasts and can be scored." | "Same check the challenge Space runs." |
| 8. Compare | Three published hashes, then the rebuilt hash revealed character by character in green; a table of every comparison; Pittsburgh's forecasts side by side; the **MATCH** banner | "Every character matches. Byte for byte, the file we just built is the file we submitted." |

### D. If something goes wrong

| Problem | What to do |
|---|---|
| No internet in the room | `uv run reproducibility-demo/verify_submission.py --offline`. Uses the cached downloads (still checked against the cached checksums). The format check is skipped because it reads the city list online; the hash comparison still runs. Needs step A3 done on this laptop. |
| Short on time | Show the rehearsal result instead: open the latest `reproducibility-demo/runs/<time>/receipt.json`, or run with `--fast` (no pauses or animation). |
| A step fails | The script stops, prints the last 25 lines and the log folder. The full output of each step is in `reproducibility-demo/runs/<time>/logs/`. |
| Text wraps or looks broken | Widen the window to 120+ columns; use Windows Terminal, not the old `cmd` console. |
| `uv` not found | Reopen the terminal after installing uv, or run `python -m pip install uv`. |

### E. Options

| Option | Effect |
|---|---|
| `--prefetch` | Download and verify the inputs only (steps 1 to 3); no build |
| `--step` | Wait for Enter before each step |
| `--fast` | No pauses or animations |
| `--offline` | No network; use cached downloads and cached checksum lists |
| `--fresh-download` | Delete `.cache/` and download everything again (strongest demo, needs good Wi-Fi) |
| `--with-leak-test` | Also run the no-look-ahead test (about 4 more minutes) |

Exit code: 0 if the rebuilt file matches, 1 if it does not, 2 if a download or a pipeline step failed.

## Timing

Measured on the development laptop (Windows 11, Python 3.13.1), cache warm:

| Step | Time |
|---|---:|
| 1-3. Read the claim, download and verify code, inputs and raw data (cache warm) | 27 s |
| 4. `f2_daily.py` | 60 s |
| 5. `f2_cddgrid.py` | 44 s |
| 6. `make_submission.py` | 82 s |
| 7. `check_submission.py` | 5 s |
| 8. Comparison | a few seconds, plus about 15 s of on-screen pauses and animation |
| **Total** | **3 to 4 minutes** (2.7 and 3.7 minutes in two runs on the development laptop) |

The very first run also installs the pinned packages (about 10 s) and downloads about 80 MB.

## Limits

- The script trusts Hugging Face's checksums for the downloaded files and runs the code it downloads; it proves
  that the published code turns the published inputs into the published file, not that the model is good.
- `huggingface_hub`, `numpy`, `pandas`, `pyarrow`, `scikit-learn` and `scipy` are pinned to the versions in
  `code/requirements.txt` and `uv.lock` of the submission; other versions may change the last digits of some
  forecasts and therefore the hash.
- The no-look-ahead test checks that forecasts do not use data after their origin; it is not part of the default
  run because it takes about 4 minutes.

## Clean-up

Delete `reproducibility-demo/runs/` (one build folder per run, about 30 MB each) and, when the demo is over,
`reproducibility-demo/.cache/` (about 80 MB). Both are git-ignored.
