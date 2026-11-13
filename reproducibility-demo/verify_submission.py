# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "huggingface_hub==2.4.0",
#   "joblib==1.6.0",
#   "numpy==2.5.4",
#   "pandas==3.1.1",
#   "pyarrow==26.0.1",
#   "rich==14.2.0",
#   "scikit-learn==1.9.1",
#   "scipy==1.18.1",
#   "threadpoolctl==3.8.0",
# ]
# ///
"""Rebuild Team Placeholders' published submission from scratch and show that it matches, byte for byte.

    uv run reproducibility-demo/verify_submission.py

What it does, in order (see README.md in this folder for the presenter guide):

1. Reads the published claim: the SHA-256 of submission.csv stated in the final submission's README on Hugging Face.
2. Downloads the published code and input data (final submission dataset, pinned commit) into a new, empty build
   folder. Every file is checked against Hugging Face's own checksums. Published outputs (submission.csv and the
   processed daily data) are NOT copied into the build folder; they are kept apart, only for the final comparison.
3. Downloads the raw challenge data (pinned revision, data up to 2018-12-31) and checks it the same way.
4-6. Runs the published pipeline unchanged: code/f2/f2_daily.py, code/f2/f2_cddgrid.py, code/make_submission.py.
7. Runs the challenge format check (code/check_submission.py).
8. Optional (--with-leak-test): the no-look-ahead test (code/f2/test_no_lookahead_f4.py, about 4 minutes).
9. Compares the rebuilt submission.csv with the claim and with both published copies (SHA-256, then value by
   value), compares the rebuilt intermediate files with the published ones, and writes a receipt.

Exit code 0: rebuilt file matches. 1: mismatch. 2: the pipeline or a download failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from importlib import metadata
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hides the "unauthenticated requests" notice; no token is needed

from rich import box  # noqa: E402
from rich.align import Align  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.spinner import Spinner  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

# --------------------------------------------------------------------------------------------------------------
# What is being verified. These pins are the public record of the final submission (2026-11-10).
# --------------------------------------------------------------------------------------------------------------
FINAL = {
    "repo": "DragonDare/protostar-placeholders-f4",
    "revision": "432197522f2e019e757a8493c55db36c8d6b1b42",
    "label": "Final submission",
}
RELEASE = {
    "repo": "Meet2304/protostar-placeholders",
    "revision": "099a52d5970d131fc52df100209a51edb369cbe0",
    "label": "Model release (Placeholder_F4)",
}
CLAIMED_SHA256 = "8344e656c942e95fd522d9e7f10f2d4c5a4847b82650f4e6de1d5c9d4633bc37"
TRAIN_REPO = "CMU-OSPO/protostar-climate-train-to-2018"
TRAIN_REVISION = "bbbdebdab2a18e00f97ef87bd3f35a71e222180b"
SAMPLE_CITY = ("USW00094823", "Pittsburgh")

# Published outputs: downloaded for comparison only, never placed in the build folder.
REFERENCE_FILES = [
    "README.md",
    "submission.csv",
    "submissions/Placeholder_F4/submission.csv",
    "data/processed/f2/daily_features.parquet",
    "data/processed/f2/cdd_grid.npz",
    "data/processed/f2/daily_qc_report.json",
]
PIPELINE = [
    ("Clean the daily weather records", ["code/f2/f2_daily.py"], "data/processed/f2/daily_features.parquet"),
    ("Build the cooling-degree-day response tables", ["code/f2/f2_cddgrid.py"], "data/processed/f2/cdd_grid.npz"),
    ("Run the model: forecasts for 210 cities, 2019-2025", ["code/make_submission.py"], "submission.csv"),
]

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
RUNS = HERE / "runs"

console = Console(highlight=False)
PACE = 1.0


def pause(seconds: float) -> None:
    if PACE > 0:
        time.sleep(seconds * PACE)


def is_build_input(path: str) -> bool:
    """Code and the top-level input tables. Excludes data/processed/ (published intermediate outputs)."""
    if path.startswith("code/") and (path.endswith(".py") or path == "code/requirements.txt"):
        return True
    parts = path.split("/")
    return len(parts) == 2 and parts[0] == "data" and parts[1].endswith((".csv", ".parquet"))


# --------------------------------------------------------------------------------------------------------------
# Hashing and verified downloads
# --------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote_manifest(repo: str, revision: str, offline: bool) -> dict[str, dict]:
    """path -> {size, sha256 | git_sha1} as Hugging Face reports it; cached for offline reruns."""
    slug = f"{repo.replace('/', '__')}@{revision[:12]}.json"
    cached = CACHE / "manifests" / slug
    if not offline:
        try:
            from huggingface_hub import HfApi

            out = {}
            for item in HfApi().list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True):
                if not hasattr(item, "blob_id"):
                    continue  # folder
                lfs = getattr(item, "lfs", None)
                sha = getattr(lfs, "sha256", None) if lfs is not None else None
                if sha is None and isinstance(lfs, dict):
                    sha = lfs.get("sha256")
                out[item.path] = {"size": item.size, "sha256": sha} if sha else {"size": item.size, "git_sha1": item.blob_id}
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(out, indent=1))
            return out
        except Exception as exc:  # network down: fall back to the cached manifest
            if not cached.exists():
                raise
            console.print(f"  [yellow]Could not reach Hugging Face ({type(exc).__name__}); using the cached file list.[/]")
    if not cached.exists():
        raise SystemExit(f"Offline and no cached file list for {repo}. Run once online first (--prefetch).")
    return json.loads(cached.read_text())


def verify_file(path: Path, expect: dict) -> bool:
    if not path.is_file() or path.stat().st_size != expect["size"]:
        return False
    if "sha256" in expect:
        return sha256_file(path) == expect["sha256"]
    return git_blob_sha1(path) == expect["git_sha1"]


def fetch_verified(repo: str, revision: str, paths: list[str], dest: Path, manifest: dict, offline: bool, what: str) -> int:
    """Download paths (skipping ones already present and intact) and verify each against the manifest."""
    missing = [p for p in paths if p not in manifest]
    if missing:
        raise SystemExit(f"{repo}@{revision[:7]} has no {missing[0]}")
    todo = [p for p in paths if not verify_file(dest / p, manifest[p])]
    if todo:
        if offline:
            raise SystemExit(f"Offline and {len(todo)} {what} files are not cached. Run once online first (--prefetch).")
        from huggingface_hub import snapshot_download

        with console.status(f"  Downloading {len(todo)} {what} file(s) from huggingface.co/{repo} ...", spinner="dots"):
            snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=dest, allow_patterns=todo)
    bad = []
    with Progress(
        TextColumn("  Checking {task.description}"), BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as prog:
        task = prog.add_task(f"{what} against Hugging Face checksums", total=len(paths))
        for p in paths:
            if not verify_file(dest / p, manifest[p]):
                bad.append(p)
            prog.advance(task)
    if bad:
        raise SystemExit(f"Checksum mismatch for {len(bad)} {what} file(s), e.g. {bad[0]}. Delete {dest} and rerun.")
    return len(paths) - len(todo)


# --------------------------------------------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------------------------------------------
FONT = {
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #"],
    "A": [" ### ", "#   #", "#####", "#   #", "#   #"],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
    "C": [" ####", "#    ", "#    ", "#    ", " ####"],
    "H": ["#   #", "#   #", "#####", "#   #", "#   #"],
    "I": ["###", " # ", " # ", " # ", "###"],
    "S": [" ####", "#    ", " ### ", "    #", "#### "],
    " ": ["  ", "  ", "  ", "  ", "  "],
}


def banner(word: str, style: str) -> Text:
    rows = ["  ".join(FONT[c][r] for c in word).replace("#", "██").replace(" ", "  ") for r in range(5)]
    return Text("\n".join(rows), style=style)


def hash_text(actual: str, expected: str | None, shown: int = 64) -> Text:
    t = Text()
    for i in range(64):
        if i >= shown:
            t.append("·", style="grey35")
        elif expected is None:
            t.append(actual[i], style="bold white")
        else:
            t.append(actual[i], style="bold green" if actual[i] == expected[i] else "bold white on red")
    return t


def step_header(n: int, total: int, title: str, interactive: bool) -> None:
    console.print()
    console.print(Rule(Text(f" Step {n}/{total}  {title} ", style="bold cyan"), style="cyan", align="left"))
    if interactive:
        console.input("[grey50]  press Enter to run this step [/]")


def ok(msg: str) -> None:
    console.print(f"  [bold green]✔[/] {msg}")


def info(msg: str) -> None:
    console.print(f"  [grey62]{msg}[/]")


def run_logged(title: str, cmd: list[str], cwd: Path, env: dict, log_path: Path) -> tuple[int, float, list[str]]:
    """Run a command, showing a spinner, the elapsed time and the last lines of its output; log everything."""
    lines: list[str] = []
    tail: deque[str] = deque(maxlen=7)
    lock = threading.Lock()
    spinner = Spinner("dots", style="cyan")
    start = time.perf_counter()
    shown_cmd = " ".join(["python", *cmd[1:]])

    def render():
        spinner.update(text=Text.assemble((f" {shown_cmd}", "bold"), (f"   {time.perf_counter() - start:5.1f} s", "cyan")))
        with lock:
            body = Text("\n".join(tail) if tail else "(starting ...)", style="grey62", overflow="ellipsis", no_wrap=True)
        return Panel(Group(spinner, Text(), body), title=title, title_align="left", border_style="cyan", box=box.ROUNDED)

    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    with Live(get_renderable=render, console=console, refresh_per_second=12, transient=True):
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            lines.append(line)
            if line.strip():
                with lock:
                    tail.append(line)
        rc = proc.wait()
    seconds = time.perf_counter() - start
    log_path.write_text(f"$ {' '.join(cmd)}\n(cwd {cwd})\n\n" + "\n".join(lines) + f"\n\nexit code {rc}, {seconds:.1f} s\n",
                        encoding="utf-8")
    return rc, seconds, lines


# --------------------------------------------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------------------------------------------
def compare_npz(a: Path, b: Path) -> tuple[bool, str]:
    import numpy as np

    with np.load(a) as x, np.load(b) as y:
        if sorted(x.files) != sorted(y.files):
            return False, f"different arrays: {sorted(x.files)} vs {sorted(y.files)}"
        for k in x.files:
            u, v = x[k], y[k]
            try:
                same = u.shape == v.shape and np.array_equal(u, v, equal_nan=True)
            except TypeError:
                same = u.shape == v.shape and np.array_equal(u, v)
            if not same:
                return False, f"array '{k}' differs"
        return True, f"all {len(x.files)} arrays identical"


def compare_parquet(a: Path, b: Path) -> tuple[bool, str]:
    import pandas as pd

    x, y = pd.read_parquet(a), pd.read_parquet(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False, f"shape {x.shape} vs {y.shape}"
    return (True, f"{x.shape[0]:,} rows x {x.shape[1]} columns, identical values") if x.equals(y) else (False, "values differ")


def compare_intermediate(rel: str, rebuilt: Path, published: Path) -> tuple[bool, str]:
    if sha256_file(rebuilt) == sha256_file(published):
        return True, "identical bytes"
    if rel.endswith(".json"):
        a, b = rebuilt.read_bytes(), published.read_bytes()
        if a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
            return True, "identical content (only line endings differ: Windows CRLF)"
        return (True, "identical content") if json.loads(a) == json.loads(b) else (False, "content differs")
    if rel.endswith(".npz"):
        same, why = compare_npz(rebuilt, published)
    elif rel.endswith(".parquet"):
        same, why = compare_parquet(rebuilt, published)
    else:
        return False, "bytes differ"
    return same, (why + " (file bytes differ only in compression)") if same else why


def compare_values(rebuilt: Path, published: Path) -> dict:
    import numpy as np
    import pandas as pd

    key = ["city_id", "year", "metric"]
    a = pd.read_csv(rebuilt, dtype={"city_id": str, "metric": str})
    b = pd.read_csv(published, dtype={"city_id": str, "metric": str})
    m = a.merge(b, on=key, how="outer", suffixes=("_rebuilt", "_published"), indicator=True)
    both = m["_merge"] == "both"
    diff = (m.predicted_value_rebuilt - m.predicted_value_published).abs()
    return {
        "rows_rebuilt": len(a), "rows_published": len(b), "rows_matched": int(both.sum()),
        "values_equal": int((both & (diff == 0)).sum()), "max_abs_diff": float(np.nanmax(diff)) if both.any() else float("nan"),
        "cities": int(a.city_id.nunique()), "years": f"{a.year.min()}-{a.year.max()}", "metrics": int(a.metric.nunique()),
        "frame": m,
    }


# --------------------------------------------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------------------------------------------
def main() -> int:
    global PACE
    ap = argparse.ArgumentParser(description="Rebuild the published submission from scratch and compare SHA-256 hashes.")
    ap.add_argument("--prefetch", action="store_true", help="only download and verify inputs (run before the demo)")
    ap.add_argument("--fresh-download", action="store_true", help="delete the download cache first and download everything again")
    ap.add_argument("--offline", action="store_true", help="use only cached downloads (no network)")
    ap.add_argument("--with-leak-test", action="store_true", help="also run the no-look-ahead test (about 4 extra minutes)")
    ap.add_argument("--step", action="store_true", help="wait for Enter before each step (for narrating live)")
    ap.add_argument("--fast", action="store_true", help="no presentation pauses or animations")
    args = ap.parse_args()
    PACE = 0.0 if args.fast else 1.0
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.fresh_download and CACHE.exists():
        shutil.rmtree(CACHE)

    total = 3 if args.prefetch else (9 if args.with_leak_test else 8)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / stamp
    build = run_dir / "build"
    logs = run_dir / "logs"
    pub_cache = CACHE / "published" / f"{FINAL['repo'].replace('/', '__')}@{FINAL['revision'][:12]}"
    rel_cache = CACHE / "published" / f"{RELEASE['repo'].replace('/', '__')}@{RELEASE['revision'][:12]}"
    raw_cache = CACHE / "raw" / TRAIN_REVISION[:12]
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    # ---- Title ------------------------------------------------------------------------------------------------
    console.print()
    intro = Table.grid(padding=(0, 2))
    intro.add_column(style="grey62", justify="right")
    intro.add_column()
    intro.add_row("Claim", "Our published submission.csv is produced by our published code, and anyone can rebuild it.")
    intro.add_row("Test", "Download the code, run the whole pipeline from raw data in an empty folder, compare SHA-256.")
    intro.add_row("Published", f"huggingface.co/datasets/{FINAL['repo']}  @ {FINAL['revision'][:7]}")
    intro.add_row("Raw data", f"huggingface.co/datasets/{TRAIN_REPO}  @ {TRAIN_REVISION[:7]}  (ends 2018-12-31)")
    intro.add_row("Machine", f"{platform.system()} {platform.release()}, Python {platform.python_version()}")
    console.print(Panel(intro, title="[bold]Team Placeholders · Reproducibility check · Placeholder_F4[/]",
                        border_style="bright_blue", box=box.DOUBLE, padding=(1, 2)))
    pause(1.0)

    # ---- Step 1: the published claim ---------------------------------------------------------------------------
    step_header(1, total, "Read the published claim", args.step)
    t0 = time.perf_counter()
    final_manifest = remote_manifest(FINAL["repo"], FINAL["revision"], args.offline)
    fetch_verified(FINAL["repo"], FINAL["revision"], REFERENCE_FILES, pub_cache, final_manifest, args.offline, "published")
    readme = (pub_cache / "README.md").read_text(encoding="utf-8")
    pattern = re.compile(r"SHA-256 `([0-9a-f]{8})…([0-9a-f]{5})`")
    claims = [(line, m) for line in readme.splitlines() for m in pattern.finditer(line)]
    if not claims or not all(CLAIMED_SHA256.startswith(m.group(1)) and CLAIMED_SHA256.endswith(m.group(2)) for _, m in claims):
        raise SystemExit("The published README does not state the expected SHA-256.")
    line, m = claims[0]
    sentence = next(s for s in re.split(r"(?<=[.)])\s+(?=[A-Z])", line) if m.group(0) in s)
    info(f"README.md of {FINAL['repo']} @ {FINAL['revision'][:7]} states the hash {len(claims)} times, e.g.:")
    console.print(Panel(Text(f"“{sentence.strip()}”", style="italic"), border_style="grey50", box=box.SIMPLE,
                        padding=(0, 4)))
    published_sha = sha256_file(pub_cache / "submission.csv")
    console.print(Text.assemble("  SHA-256 of the published submission.csv  ", hash_text(published_sha, None)))
    ok("Published file carries the hash stated in its README" if published_sha == CLAIMED_SHA256
       else "[red]Published file does NOT carry the stated hash[/]")
    timings["1 claim"] = time.perf_counter() - t0
    pause(1.0)

    # ---- Step 2: published code and inputs into an empty folder ------------------------------------------------
    step_header(2, total, "Download the published code and inputs into an empty folder", args.step)
    t0 = time.perf_counter()
    inputs = sorted(p for p in final_manifest if is_build_input(p))
    cached = fetch_verified(FINAL["repo"], FINAL["revision"], inputs, pub_cache, final_manifest, args.offline, "code and input")
    n_code = sum(p.startswith("code/") for p in inputs)
    ok(f"{len(inputs)} files ({n_code} code files, {len(inputs) - n_code} input tables) match Hugging Face's checksums"
       + (f" [grey50]({cached} already cached)[/]" if cached else ""))
    if not args.prefetch:
        build.mkdir(parents=True)
        logs.mkdir(parents=True)
        for p in inputs:
            (build / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pub_cache / p, build / p)
        assert not (build / "submission.csv").exists() and not (build / "data" / "processed").exists()
        ok(f"Copied into a new build folder: [bold]{build.relative_to(HERE.parent)}[/]")
        ok("The build folder has [bold]no submission.csv[/] and [bold]no processed data[/]: everything is rebuilt")
        for p in ["data/annual_indices_train_to_2018.parquet", "data/forcing_ar6_to_2018.csv", "code/make_submission.py"]:
            info(f"{p:45s} sha256 {sha256_file(build / p)[:16]}…")
    # second published copy: only its submission.csv, for the comparison
    rel_manifest = remote_manifest(RELEASE["repo"], RELEASE["revision"], args.offline)
    fetch_verified(RELEASE["repo"], RELEASE["revision"], ["submission.csv"], rel_cache, rel_manifest, args.offline, "release")
    timings["2 code"] = time.perf_counter() - t0
    pause(0.8)

    # ---- Step 3: raw challenge data -----------------------------------------------------------------------------
    step_header(3, total, "Download the raw challenge data (pinned revision, ends 2018-12-31)", args.step)
    t0 = time.perf_counter()
    pinned = re.search(r'TRAIN_REVISION = "([0-9a-f]{40})"', (pub_cache / "code/common.py").read_text(encoding="utf-8"))
    if not pinned or pinned.group(1) != TRAIN_REVISION:
        raise SystemExit("The published code pins a different raw-data revision than this script expects.")
    info(f"The published code (code/common.py) pins {TRAIN_REPO} @ {TRAIN_REVISION[:7]}")
    raw_manifest = remote_manifest(TRAIN_REPO, TRAIN_REVISION, args.offline)
    raw_files = sorted(raw_manifest)
    cached = fetch_verified(TRAIN_REPO, TRAIN_REVISION, raw_files, raw_cache, raw_manifest, args.offline, "raw data")
    size_mb = sum(v["size"] for v in raw_manifest.values()) / 1e6
    ok(f"{len(raw_files)} raw files ({size_mb:.0f} MB) match Hugging Face's checksums"
       + (f" [grey50]({cached} already cached)[/]" if cached else ""))
    for p in raw_files:
        if p.endswith(".parquet"):
            info(f"{p:52s} sha256 {raw_manifest[p].get('sha256', '')[:16]}…")
    timings["3 raw"] = time.perf_counter() - t0

    if args.prefetch:
        console.print()
        console.print(Panel(f"Inputs downloaded and verified in {time.perf_counter() - t_all:.0f} s. Ready for the live demo:\n"
                            "[bold]uv run reproducibility-demo/verify_submission.py[/]", border_style="green"))
        return 0

    # ---- Steps 4-6: the published pipeline, unchanged ------------------------------------------------------------
    env = os.environ.copy()
    env.update({"F2_RAW": str(raw_cache), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    for i, (title, script, product) in enumerate(PIPELINE, start=4):
        step_header(i, total, title, args.step)
        rc, secs, lines = run_logged(title, [sys.executable, *script], build, env, logs / f"{i}_{Path(script[0]).stem}.log")
        timings[f"{i} {Path(script[0]).stem}"] = secs
        if rc != 0:
            console.print("\n".join(lines[-25:]), style="red", markup=False)
            console.print(f"[bold red]  ✘ python {' '.join(script)} failed (exit {rc}); full log: {logs}[/]")
            return 2
        ok(f"python {' '.join(script)}  [grey50]({secs:.1f} s)[/]")
        for line in [s for s in lines if s.strip()][-4:]:
            info(line.strip()[:140])
        ok(f"Wrote {product}  ({(build / product).stat().st_size / 1e6:.2f} MB)")
        pause(0.6)

    # ---- Step 7: format check ------------------------------------------------------------------------------------
    step_header(7, total, "Check the file the way the challenge Space does", args.step)
    rc, secs, lines = run_logged("Format check", [sys.executable, "code/check_submission.py", "submission.csv"], build, env,
                                 logs / "7_check_submission.log")
    timings["7 check"] = secs
    check_line = next((s for s in reversed(lines) if s.strip()), "")
    if rc == 0:
        ok(check_line)
        check_status = check_line
    elif args.offline:
        console.print("  [yellow]Skipped: the format check reads the city list from Hugging Face (offline mode).[/]")
        check_status = "skipped (offline)"
    else:
        console.print(f"  [bold red]✘ {check_line}[/]")
        check_status = f"FAILED: {check_line}"

    # ---- Step 8 (optional): no-look-ahead test --------------------------------------------------------------------
    leak_status = "not run (add --with-leak-test)"
    if args.with_leak_test:
        step_header(8, total, "No-look-ahead test: corrupt every year after the origin, forecasts must not move", args.step)
        leak_env = dict(env, PYTHONPATH=str(build / "code" / "f2"))
        rc, secs, lines = run_logged("No-look-ahead test", [sys.executable, "code/f2/test_no_lookahead_f4.py"], build, leak_env,
                                     logs / "8_test_no_lookahead_f4.log")
        timings["8 leak test"] = secs
        leak_status = next((s for s in reversed(lines) if s.strip()), f"exit {rc}")
        (ok if rc == 0 else console.print)(leak_status)

    # ---- Final step: compare ---------------------------------------------------------------------------------------
    step_header(total, total, "Compare the rebuilt file with what we published", args.step)
    rebuilt_sha = sha256_file(build / "submission.csv")
    release_sha = sha256_file(rel_cache / "submission.csv")
    rows = [
        ("Stated in our README and slides", CLAIMED_SHA256),
        (f"Published: {FINAL['repo']} @ {FINAL['revision'][:7]}", published_sha),
        (f"Published: {RELEASE['repo']} @ {RELEASE['revision'][:7]}", release_sha),
    ]
    width = max(len(r[0]) for r in rows + [("Rebuilt just now on this machine", "")])
    for label, h in rows:
        console.print(Text.assemble(f"  {label:<{width}}  ", hash_text(h, None)))
        pause(0.4)
    label = "Rebuilt just now on this machine"
    with Live(console=console, refresh_per_second=40, transient=False) as live:
        for shown in range(0, 65, 2 if PACE else 64):
            live.update(Text.assemble((f"  {label:<{width}}  ", "bold"), hash_text(rebuilt_sha, CLAIMED_SHA256, shown)))
            pause(0.045)
    pause(0.5)

    values = compare_values(build / "submission.csv", pub_cache / "submission.csv")
    inter = {rel: compare_intermediate(rel, build / rel, pub_cache / rel) for rel in REFERENCE_FILES if rel.startswith("data/processed")}
    match = rebuilt_sha == CLAIMED_SHA256 == published_sha == release_sha

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold", pad_edge=False)
    table.add_column("What was compared")
    table.add_column("Result")
    table.add_column("", justify="center")

    def mark(good: bool) -> Text:
        return Text("✔", style="bold green") if good else Text("✘", style="bold red")

    table.add_row("submission.csv SHA-256: rebuilt vs. stated", "identical" if rebuilt_sha == CLAIMED_SHA256 else "DIFFERENT",
                  mark(rebuilt_sha == CLAIMED_SHA256))
    table.add_row(f"submission.csv SHA-256: rebuilt vs. {FINAL['label'].lower()}", "identical" if rebuilt_sha == published_sha
                  else "DIFFERENT", mark(rebuilt_sha == published_sha))
    table.add_row("submission.csv SHA-256: rebuilt vs. model release", "identical" if rebuilt_sha == release_sha
                  else "DIFFERENT", mark(rebuilt_sha == release_sha))
    vals_ok = values["values_equal"] == values["rows_published"] == values["rows_rebuilt"]
    table.add_row("Forecast values, one by one",
                  f"{values['values_equal']:,} of {values['rows_published']:,} equal "
                  f"({values['cities']} cities x {values['metrics']} metrics x years {values['years']})", mark(vals_ok))
    for rel, (same, why) in inter.items():
        table.add_row(f"Intermediate: {Path(rel).name}", why, mark(same))
    table.add_row("Format check (code/check_submission.py)", check_status[:70],
                  Text("-", style="bold yellow") if check_status.startswith("skipped") else mark(check_status.startswith("OK")))
    if args.with_leak_test:
        table.add_row("No-look-ahead test", leak_status[:70], mark(leak_status.startswith("PASS")))
    console.print()
    console.print(table)

    # sample forecasts for one city, so the audience sees actual numbers
    frame = values["frame"]
    city, city_name = SAMPLE_CITY
    sample = frame[(frame.city_id == city) & frame.year.isin([2019, 2025])].sort_values(["metric", "year"])
    if len(sample):
        st = Table(title=f"Example: {city_name} ({city})", box=box.MINIMAL, title_style="bold", header_style="grey62")
        for c in ("Metric", "Year", "Published", "Rebuilt"):
            st.add_column(c, justify="right" if c != "Metric" else "left")
        for r in sample.itertuples():
            same = r.predicted_value_rebuilt == r.predicted_value_published
            st.add_row(r.metric, str(r.year), f"{r.predicted_value_published:.3f}",
                       Text(f"{r.predicted_value_rebuilt:.3f}", style="green" if same else "bold red"))
        console.print(st)
    pause(0.8)

    elapsed = time.perf_counter() - t_all
    if match and vals_ok:
        verdict = Group(
            Align.center(banner("MATCH", "bold green")), Text(),
            Align.center(Text("Rebuilt from raw data on this machine, submission.csv is", style="bold")),
            Align.center(Text("byte-for-byte identical to the file we published and submitted.", style="bold")), Text(),
            Align.center(Text(f"SHA-256 {rebuilt_sha}", style="green")),
            Align.center(Text(f"full rebuild in {elapsed / 60:.1f} minutes", style="grey62")),
        )
        console.print(Panel(verdict, border_style="bold green", box=box.HEAVY, padding=(1, 2)))
    else:
        verdict = Group(
            Align.center(banner("MISMATCH", "bold red")), Text(),
            Align.center(Text("The rebuilt file differs from the published one. See the table above and the logs.", style="bold")),
            Align.center(Text(f"rebuilt  {rebuilt_sha}\nstated   {CLAIMED_SHA256}", style="red")),
        )
        console.print(Panel(verdict, border_style="bold red", box=box.HEAVY, padding=(1, 2)))

    # ---- Receipt ----------------------------------------------------------------------------------------------------
    pkgs = {p: metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "huggingface_hub")}
    receipt = {
        "verdict": "MATCH" if match and vals_ok else "MISMATCH",
        "utc": stamp,
        "sha256": {"stated": CLAIMED_SHA256, "published_final": published_sha, "published_release": release_sha,
                   "rebuilt": rebuilt_sha},
        "sources": {"final": FINAL, "release": RELEASE, "raw": {"repo": TRAIN_REPO, "revision": TRAIN_REVISION}},
        "values": {k: v for k, v in values.items() if k != "frame"},
        "intermediates": {k: {"identical": s, "detail": w} for k, (s, w) in inter.items()},
        "format_check": check_status,
        "no_lookahead_test": leak_status,
        "timings_s": {k: round(v, 1) for k, v in timings.items()} | {"total": round(elapsed, 1)},
        "machine": {"os": platform.platform(), "python": platform.python_version(), "packages": pkgs},
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    info(f"Receipt: {(run_dir / 'receipt.json').relative_to(HERE.parent)}   Logs: {logs.relative_to(HERE.parent)}")
    return 0 if match and vals_ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
        sys.exit(130)
