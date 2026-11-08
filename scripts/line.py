"""Assembly-line tool: worktrees, immutable data releases, lineage labels and MLflow sync.

Plan: docs/plans/assembly-line-plan.md. Agents: follow .claude/skills/assembly-line/SKILL.md.

    uv run python scripts/line.py status                 what is out of sync here, and how to fix it
    uv run python scripts/line.py score --name M --adapter code/m.py:forecast --assets evaluation/no-assets.json
    uv run python scripts/line.py release clean <folder> --note "why"
    uv run python scripts/line.py pull                   copy the locked data releases into this checkout
    uv run python scripts/line.py sync                   mirror sealed runs to MLflow, with labels
    uv run python scripts/line.py get-suite              fetch the shared test set (teammates)

Integrator only: setup, lock, adopt, publish-suite, serve, add-user.

This file lives outside code/ and src/ so it cannot change a run's source pins. It never edits the
harness, a sealed run or submission.csv. Credentials come from the environment and are never stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import getpass

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")  # fail fast if the server's machine is asleep

LOCK_PATH = ROOT / "line.lock.json"
STATIONS = ("clean", "enhance", "features", "train", "eval")
BRANCH_PREFIX = "Meet/"
# Only the MLflow server needs Flask-WTF. It stays out of the project so uv.lock, which runs and freezes pin, is unchanged.
SERVER_COMMAND = "uv run --with flask-wtf python scripts/line.py"
UPSTREAM = {"features": ("clean", "enhance"), "train": ("features",)}
EVALUATOR_FILES = ("code/evaluate.py", "code/common.py", *(f"code/evaluation/{name}.py" for name in (
    "core", "protocol", "worker", "runtime_audit", "native_io", "phase_policy", "assets", "suite_identity",
    "official_score")))
RELEASE_EXPERIMENT = "protostar/releases"
SUITE_EXPERIMENT = "protostar/suites"
SUITE_NAME = "suite-v3"
FINGERPRINT_TAG = "harness.run_fingerprint"
PULLED = Path("data/processed/releases")
RELEASE_PATH = re.compile(r"^data/(?:processed/)?releases/([a-z]+-v\d+)/")
ORIGIN_FILE = re.compile(r"origin-\d{4}")
UNVERSIONED_PATH = re.compile(r"data[/\\]+processed[/\\]+(?!releases)|notebooks[/\\]+\w+[/\\]+\S+\.(?:parquet|csv)")


class LineError(RuntimeError):
    """A problem the user can act on; printed without a traceback."""


# ---------------------------------------------------------------- git and paths
def git(*args, cwd=ROOT, check=False):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise LineError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip() if result.returncode == 0 else ""


def depot_root():
    """The main checkout: the first entry of `git worktree list`."""
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1]).resolve()
    return ROOT


def releases_root():
    return depot_root() / "outputs/releases"


def branch():
    return git("branch", "--show-current")


def main_ref():
    """Stations share the local main; teammates' clones follow the team's published main."""
    entries = [l for l in git("worktree", "list", "--porcelain").splitlines() if l.startswith("worktree ")]
    return "main" if len(entries) > 1 or not git("rev-parse", "--verify", "origin/main") else "origin/main"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- the lock
def empty_lock():
    return {"schema_version": 1, "line_started_utc": None, "mlflow_uri": None, "suite_id": None,
            "evaluators": {}, "current_evaluator": None, "releases": {}}


def read_lock(path=None):
    path = Path(path or LOCK_PATH)
    if not path.is_file():
        raise LineError("line.lock.json is missing. The integrator creates it with `line.py lock`.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_lock(lock, path=None):
    Path(path or LOCK_PATH).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def locked_on_main():
    """The lock as published on main, which is what every station must align with."""
    text = git("show", f"{main_ref()}:line.lock.json")
    return json.loads(text) if text else None


# ---------------------------------------------------------------- identity and labels
def evaluator_hash(pins):
    """Hash only the scoring files, so adding a model elsewhere in code/ leaves it unchanged."""
    return canonical({name: pins.get(name) for name in EVALUATOR_FILES})


def evaluator_name(pins, lock):
    digest = evaluator_hash(pins or {})
    for name, known in lock.get("evaluators", {}).items():
        if known == digest:
            return name
    return f"unreleased-{digest[:8]}"


def run_assets(meta):
    binding = ((meta.get("model_spec") or {}).get("assets") or {}).get("binding") or {}
    return binding.get("files") or {}


def find_release(release_id, roots=None):
    for root in roots or (ROOT / PULLED, *sorted(releases_root().glob("*"))):
        path = Path(root) / release_id / "release.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def used_releases(meta, lookup=find_release):
    """Match a run's sealed asset paths and hashes to releases. Returns (found, problems)."""
    found, problems = {}, []
    for name, record in run_assets(meta).items():
        match = RELEASE_PATH.match(name)
        if not match:
            if record.get("kind") == "weather_input":
                problems.append(f"weather input is not a release: {name}")
            continue
        release = lookup(match.group(1))
        if release is None:
            problems.append(f"unknown release: {match.group(1)}")
            continue
        entry = release["files"].get(name[match.end():])
        if entry is None or entry["sha256"] != record.get("sha256"):
            problems.append(f"file does not match {release['id']}: {name}")
            continue
        found[release["id"]] = release
    return found, problems


def classify(meta, lock, lookup=find_release):
    """Label a sealed run from its own record. Returns (official|exploratory|legacy, reasons)."""
    started = lock.get("line_started_utc")
    if not started or meta["created_utc"] < started:
        return "legacy", ["created before the assembly line started"]
    reasons = []
    status = (meta.get("git") or {}).get("status")
    if status is None or status:
        reasons.append("code was not a clean commit")
    if evaluator_name(meta.get("source_pins"), lock).startswith("unreleased-"):
        reasons.append("evaluator is not a released version")
    if meta.get("suite_id") != lock.get("suite_id"):
        reasons.append("test set is not the locked suite")
    found, problems = used_releases(meta, lookup)
    reasons.extend(problems)
    reasons.extend(f"{name} is not gate-eligible" for name, release in sorted(found.items())
                   if not release.get("gate_eligible"))
    return ("exploratory", reasons) if reasons else ("official", [])


# ---------------------------------------------------------------- MLflow
def mlflow_uri(lock=None):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri and lock is None:
        try:
            lock = read_lock()
        except LineError:
            lock = {}
    uri = uri or (lock or {}).get("mlflow_uri")
    if not uri:
        raise LineError("No MLflow server is configured: set MLFLOW_TRACKING_URI or mlflow_uri in line.lock.json.")
    os.environ["MLFLOW_TRACKING_URI"] = uri
    return uri


def mlflow_client(lock=None):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = mlflow_uri(lock)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def ensure_experiment(client, name):
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    try:
        return client.create_experiment(name)
    except Exception:  # another writer created it first
        found = client.get_experiment_by_name(name)
        if found is None:
            raise
        return found.experiment_id


def find_tagged(client, experiment, tag, value):
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return None
    hits = client.search_runs([found.experiment_id], filter_string=f"tags.`{tag}` = '{value}'", max_results=1)
    return hits[0] if hits else None


def dataset_input(name, digest, context):
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.entities import DatasetInput, InputTag

    meta = MetaDataset(source=HTTPDatasetSource(url=f"file:///protostar/{name}"), name=name, digest=digest[:36])
    return DatasetInput(dataset=meta._to_mlflow_entity(), tags=[InputTag("mlflow.data.context", context)])


def suite_label(suite_id, lock):
    return SUITE_NAME if suite_id == lock.get("suite_id") else f"suite-{str(suite_id)[:8]}"


def lineage_tags(meta, lock, lookup=find_release):
    status, reasons = classify(meta, lock, lookup)
    found, _ = used_releases(meta, lookup)
    author = "" if status == "legacy" else (os.environ.get("LINE_AUTHOR") or git("config", "user.name"))
    return {"line.status": status, "line.reasons": "; ".join(reasons)[:4900],
            "line.model": meta["model"], "line.evaluator": evaluator_name(meta.get("source_pins"), lock),
            "line.test_set": f"{suite_label(meta['suite_id'], lock)}:{meta.get('phase', 'development')}",
            "line.train_data": ",".join(sorted(found)) or "provided-history",
            "line.author": author}


def apply_lineage(client, run, meta, lock):
    """Idempotent: labels can be recomputed, dataset inputs are only added once."""
    rid = run.info.run_id
    tags = lineage_tags(meta, lock)
    if run.data.tags.get("line.author"):  # a later sync on another machine must not change the author
        tags["line.author"] = run.data.tags["line.author"]
    for key, value in tags.items():
        if run.data.tags.get(key) != value:
            client.set_tag(rid, key, value)
    if not run.inputs.dataset_inputs:
        found, _ = used_releases(meta)
        phase = meta.get("phase", "development")
        inputs = [dataset_input(name, canonical(release["files"]), "train") for name, release in sorted(found.items())]
        if not found:
            inputs.append(dataset_input("provided-history", str(meta["suite_id"]), "train"))
        inputs.append(dataset_input(f"{suite_label(meta['suite_id'], lock)}:{phase}", str(meta["suite_id"]), "test"))
        client.log_inputs(rid, datasets=inputs)
    return tags["line.status"]


def sync_runs(runs, only=None, echo=print):
    from evaluation.core import read_json
    from evaluation.tracking import experiment_name, log_run

    lock = read_lock()
    client = mlflow_client(lock)
    counts = {"official": 0, "exploratory": 0, "legacy": 0, "failed": 0, "incomplete": 0, "pre_fingerprint": 0}
    paths = [Path(only) / "run.json"] if only else sorted(Path(runs).glob("*/run.json"))
    for path in paths:
        try:
            meta = read_json(path)
            if meta.get("status") != "complete":
                counts["incomplete"] += 1
                continue
            if "run_fingerprint" not in meta:  # first harness format (schema 1): the mirror cannot identify it
                counts["pre_fingerprint"] += 1
                continue
            log_run(path.parent, client)
            run = find_tagged(client, experiment_name(meta), FINGERPRINT_TAG, meta["run_fingerprint"])
            if run is None:
                raise LineError("run was not found after mirroring")
            counts[apply_lineage(client, run, meta, lock)] += 1
        except Exception as exc:  # one bad run never stops the backfill
            counts["failed"] += 1
            echo(f"  could not mirror {path.parent.name}: {type(exc).__name__}: {exc}")
    return counts


# ---------------------------------------------------------------- releases
def station_versions(station, root):
    pattern = re.compile(rf"^{station}-v(\d+)$")
    folder = Path(root) / station
    if not folder.is_dir():
        return []
    return sorted(int(m.group(1)) for m in (pattern.match(p.name) for p in folder.iterdir()) if m)


def make_release(station, source, note, upstream=(), gate_eligible=False, lineage="built", first_version=1,
                 root=None, commit=None):
    """Copy a folder into an immutable, numbered release. Two writers can never share a number."""
    if station not in STATIONS:
        raise LineError(f"station must be one of {', '.join(STATIONS)}")
    source, root = Path(source).resolve(), Path(root or releases_root())
    if not source.is_dir():
        raise LineError(f"not a folder: {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files:
        raise LineError(f"nothing to release in {source}")
    names = [p.relative_to(source).as_posix() for p in files]
    if gate_eligible:
        loose = [n for n in names if n.endswith((".csv", ".parquet")) and not ORIGIN_FILE.search(n)
                 and not n.startswith("static")]
        if loose:
            raise LineError("gate-eligible data must be per origin (origin-YYYY in the name) or under static*: "
                            + ", ".join(loose[:5]))
        for name in upstream:
            found = find_release(name)
            if found is None or not found.get("gate_eligible"):
                raise LineError(f"upstream {name} is missing or not gate-eligible")
    (root / station).mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".incoming-", dir=root / station))
    try:
        record = {}
        for path, name in zip(files, names):
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            year = ORIGIN_FILE.search(name)
            record[name] = {"sha256": sha256_file(target), "size": target.stat().st_size,
                            "information_end_year": int(year.group(0)[-4:]) if year else None}
        manifest = {"schema_version": 1, "station": station, "created_utc": utc_now(),
                    "git_commit": commit or git("rev-parse", "HEAD"), "upstream": sorted(upstream),
                    "gate_eligible": bool(gate_eligible), "lineage": lineage, "note": note, "files": record}
        while True:
            versions = station_versions(station, root)
            number = first_version if first_version not in versions else versions[-1] + 1
            manifest["id"] = f"{station}-v{number}"
            (temp / "release.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                temp.rename(root / station / manifest["id"])  # atomic; fails if the number was just taken
                break
            except OSError:
                if not (root / station / manifest["id"]).exists():
                    raise
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    final = root / station / manifest["id"]
    for path in final.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    return manifest, final


def publish_release(manifest, folder, echo=print):
    """Best effort: a release is valid without MLflow, and `line.py sync` retries."""
    try:
        client = mlflow_client()
        if find_tagged(client, RELEASE_EXPERIMENT, "line.release_id", manifest["id"]):
            return True
        experiment = ensure_experiment(client, RELEASE_EXPERIMENT)
        run = client.create_run(experiment, run_name=manifest["id"], tags={
            "line.release_id": manifest["id"], "line.station": manifest["station"],
            "line.gate_eligible": str(manifest["gate_eligible"]), "line.lineage": manifest["lineage"],
            "line.upstream": ",".join(manifest["upstream"]), "git.commit": manifest["git_commit"]})
        client.log_param(run.info.run_id, "note", manifest["note"][:6000])
        client.log_artifacts(run.info.run_id, str(folder), artifact_path="release")
        client.set_terminated(run.info.run_id)
        return True
    except Exception as exc:
        echo(f"Release is saved but not on MLflow yet ({type(exc).__name__}: {exc}). `line.py sync` retries.")
        return False


def sync_releases(echo=print):
    published = 0
    for path in sorted(releases_root().glob("*/*/release.json")):
        published += publish_release(json.loads(path.read_text(encoding="utf-8")), path.parent, echo)
    return published


def verify_release_folder(folder):
    manifest = json.loads((Path(folder) / "release.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        if sha256_file(Path(folder) / name) != entry["sha256"]:
            raise LineError(f"{manifest['id']}: {name} does not match its release record")
    return manifest


def pull_release(release_id, destination=None, echo=print):
    """A real copy, because the harness refuses data assets reached through a junction."""
    destination = Path(destination or ROOT / PULLED) / release_id
    if (destination / "release.json").is_file():
        return verify_release_folder(destination)
    station = release_id.rsplit("-v", 1)[0]
    source = releases_root() / station / release_id
    if (source / "release.json").is_file():
        verify_release_folder(source)
        cleanup = None
    else:
        echo(f"Downloading {release_id} from MLflow")
        client = mlflow_client()
        run = find_tagged(client, RELEASE_EXPERIMENT, "line.release_id", release_id)
        if run is None:
            raise LineError(f"release not found locally or on MLflow: {release_id}")
        cleanup = tempfile.mkdtemp(prefix="line-pull-")
        source = Path(client.download_artifacts(run.info.run_id, "release", cleanup))
        verify_release_folder(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".incoming-", dir=destination.parent))
    try:
        for path in source.rglob("*"):
            if path.is_file():
                target = temp / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        manifest = verify_release_folder(temp)
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
    return manifest


# ---------------------------------------------------------------- status
def station_of(name=None):
    name = branch() if name is None else name
    return name.split("/", 1)[1] if name.startswith(BRANCH_PREFIX) else None


def findings(lock=None, check_mlflow=True):
    """Return [(id, message, fix)]. Pure facts: the skill says what to do about them."""
    out = []
    here, depot, current = ROOT.resolve(), depot_root(), branch()
    linked = len([l for l in git("worktree", "list", "--porcelain").splitlines() if l.startswith("worktree ")]) > 1
    if here == depot and linked:
        out.append(("D1", "You are working in the depot (the main checkout).",
                    f"Follow docs/assembly-line-migration.md to move your files to {BRANCH_PREFIX}<station> in your worktree."))
    if git("status", "--porcelain"):
        out.append(("D2", "This checkout has uncommitted work.", "Commit it to your branch."))
    published = lock if lock is not None else locked_on_main()
    if published is None:
        out.append(("D0", f"{main_ref()} has no line.lock.json yet.", "Wait for the integrator to start the line."))
        return out
    behind = git("rev-list", "--count", f"HEAD..{main_ref()}") or "0"
    if int(behind):
        out.append(("D3", f"{behind} commit(s) behind {main_ref()}.", f"git merge {main_ref()}, then run your tests."))
    from evaluation.protocol import source_pins
    pins = source_pins(ROOT)
    name = evaluator_name(pins, published)
    if name != published.get("current_evaluator") and station_of(current) != "eval":
        out.append(("D4", f"Evaluator here is {name}, the line uses {published.get('current_evaluator')}.",
                    f"git merge {main_ref()}. Do not edit scoring files outside the eval station."))
    station = station_of(current)
    for upstream in UPSTREAM.get(station, ()):  # teammates choose their own data
        wanted = published.get("releases", {}).get(upstream)
        if wanted and not (ROOT / PULLED / wanted / "release.json").is_file():
            out.append(("D5", f"The line's {upstream} release {wanted} is not in this checkout.",
                        "Finish your current version, then `line.py pull` and adopt it as one declared change."))
    changed = git("diff", "--name-only", f"{main_ref()}...HEAD").splitlines()
    readers = [n for n in changed if n.endswith((".py", ".ipynb")) and (ROOT / n).is_file()
               and UNVERSIONED_PATH.search((ROOT / n).read_text(encoding="utf-8", errors="ignore"))]
    if station in ("features", "train") and readers:
        out.append(("D6", "May read another station's working files: " + ", ".join(readers[:5]),
                    "`line.py pull` and read data/processed/releases/<id>/ instead."))
    if station in ("clean", "enhance", "features"):
        loose = [p.name for p in (ROOT / "data/processed").glob("*") if p.name != "releases"]
        if loose:
            out.append(("D7", "Output with no release: data/processed/" + ", ".join(loose[:5]),
                        f"`line.py release {station} <folder> --note \"...\"`"))
    runs = ROOT / "outputs/evaluation/runs"
    if check_mlflow and runs.is_dir():
        try:
            client = mlflow_client(published)
            known = set()
            for exp in client.search_experiments():
                if exp.name.startswith("protostar/"):
                    known.update(r.data.tags.get(FINGERPRINT_TAG) for r in
                                 client.search_runs([exp.experiment_id], max_results=5000))
            missing = 0
            for path in runs.glob("*/run.json"):
                meta = json.loads(path.read_text(encoding="utf-8"))
                fingerprint = meta.get("run_fingerprint")
                missing += meta.get("status") == "complete" and fingerprint is not None and fingerprint not in known
            if missing:
                out.append(("D9", f"{missing} sealed run(s) are missing from MLflow.", "`line.py sync`"))
        except Exception as exc:
            out.append(("D9", f"MLflow is not reachable ({type(exc).__name__}).",
                        "Check MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD; your work is safe, sync later."))
    return out


def migration_report():
    """For each uncommitted file in the old shared checkout, find a branch holding identical content."""
    depot = depot_root()
    branches = ["main", *git("for-each-ref", "--format=%(refname:short)", f"refs/heads/{BRANCH_PREFIX}").split()]
    listing = subprocess.run(["git", "ls-files", "-z", "-m", "-o", "--exclude-standard"], cwd=depot,
                             capture_output=True, text=True, check=True).stdout
    rows = []
    for name in sorted({item for item in listing.split("\0") if item}):
        if not (depot / name).is_file():
            rows.append((name, "deleted in the old checkout"))
            continue
        blob = git("hash-object", "--", name, cwd=depot)
        holder = next((b for b in branches if git("rev-parse", "--verify", "-q", f"{b}:{name}") == blob), None)
        rows.append((name, holder or "NOT MIGRATED"))
    return rows


# ---------------------------------------------------------------- integrator: worktrees and server
def link_junction(link, target):
    link, target = Path(link), Path(target)
    if link.exists() or link.is_symlink():
        return
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def setup_station(station):
    depot = depot_root()
    path = depot.parent / "wt" / station
    target_branch = f"{BRANCH_PREFIX}{station}"
    if not path.exists():
        exists = git("rev-parse", "--verify", target_branch, cwd=depot)
        arguments = [str(path), target_branch] if exists else ["-b", target_branch, str(path), "main"]
        git("worktree", "add", "--quiet", *arguments, cwd=depot, check=True)
    (path / "outputs").mkdir(exist_ok=True)
    for name in ("outputs/evaluation", "data/raw"):
        (depot / name).mkdir(parents=True, exist_ok=True)
        link_junction(path / name, depot / name)
    subprocess.run(["uv", "sync", "--quiet"], cwd=path, check=True)
    return path


AUTH_TEMPLATE = """[mlflow]
default_permission = EDIT
database_uri = sqlite:///{database}
admin_username = admin
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
"""


def serve(host, port, allowed_hosts):
    try:
        import flask_wtf  # noqa: F401
    except ImportError as exc:
        raise LineError(f"The login server needs one extra package. Start it with: {SERVER_COMMAND} serve") from exc
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1" if host == "0.0.0.0" else host, port)) == 0:
            raise LineError(f"Port {port} is already in use, so the login server cannot start. Stop the other server first.")
    store = depot_root() / "outputs/mlflow"
    store.mkdir(parents=True, exist_ok=True)
    config = store / "basic_auth.ini"  # holds no secret: the admin password stays in the environment
    config.write_text(AUTH_TEMPLATE.format(database=(store / "basic_auth.db").as_posix()), encoding="utf-8")
    environment = {**os.environ, "MLFLOW_AUTH_CONFIG_PATH": str(config)}
    if not (store / "basic_auth.db").exists() and not environment.get("MLFLOW_AUTH_ADMIN_PASSWORD"):
        password = getpass.getpass("First start. Choose an admin password (12+ characters): ")
        if len(password) < 12:
            raise LineError("The admin password must be at least 12 characters.")
        environment["MLFLOW_AUTH_ADMIN_PASSWORD"] = password  # used once by MLflow, never written by this tool
    environment.setdefault("MLFLOW_FLASK_SERVER_SECRET_KEY", secrets.token_hex(32))  # per start, never stored
    command = [sys.executable, "-m", "mlflow", "server", "--app-name", "basic-auth",
               "--backend-store-uri", f"sqlite:///{(store / 'mlflow.db').as_posix()}",
               "--artifacts-destination", (store / "artifacts").as_posix(),
               "--host", host, "--port", str(port), "--workers", "1"]
    if allowed_hosts:
        command += ["--allowed-hosts", allowed_hosts]
    print(f"MLflow on {host}:{port}, login required. Stop with Ctrl+C.")
    return subprocess.call(command, env=environment)


def zip_suite(suite, output):
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(suite).rglob("*")):
            relative = path.relative_to(suite)
            if path.is_file() and relative.parts[0] != "assessment_access":
                archive.write(path, relative.as_posix())


def fetch_suite(destination, expected, echo=print):
    from evaluation.core import read_json, verify_suite

    destination = Path(destination)
    if (destination / "manifest.json").is_file():
        found = read_json(destination / "manifest.json")["suite_id"]
        if found != expected:
            raise LineError(f"{destination} holds a different suite ({found[:8]}). Move it aside, then retry.")
        verify_suite(destination)
        return "already present"
    client = mlflow_client()
    run = find_tagged(client, SUITE_EXPERIMENT, "line.suite_id", expected)
    if run is None:
        raise LineError("The shared test set is not on MLflow yet. Ask the integrator to run publish-suite.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = client.download_artifacts(run.info.run_id, f"{SUITE_NAME}.zip", temp)
        unpacked = Path(tempfile.mkdtemp(prefix=".incoming-", dir=destination.parent))
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(unpacked)
    try:
        if verify_suite(unpacked)["suite_id"] != expected:
            raise LineError("downloaded suite does not match the lock")
        unpacked.rename(destination)
    except BaseException:
        shutil.rmtree(unpacked, ignore_errors=True)
        raise
    return "downloaded"


# ---------------------------------------------------------------- commands
def cmd_status(args):
    items = findings(check_mlflow=not args.offline)
    summary = {"checkout": str(ROOT), "branch": branch(), "commit": git("rev-parse", "--short", "HEAD"),
               "findings": [{"id": a, "finding": b, "fix": c} for a, b, c in items]}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['branch']} @ {summary['commit']}  ({summary['checkout']})")
        for code, finding, fix in items:
            print(f"  {code}  {finding}\n      fix: {fix}")
        print("  In sync." if not items else f"  {len(items)} finding(s).")
    return 1 if args.strict and items else 0


def cmd_migration(args):
    rows = migration_report()
    for name, where in rows:
        print(f"  {where:<30} {name}")
    missing = sum(where == "NOT MIGRATED" for _, where in rows)
    print(f"{len(rows) - missing} of {len(rows)} accounted for, {missing} not migrated.")
    return 1 if missing else 0


def cmd_release(args):
    if git("status", "--porcelain"):
        raise LineError("Commit your work first: a release must trace to a clean commit.")
    manifest, folder = make_release(args.station, args.folder, args.note, args.upstream, args.gate_eligible)
    publish_release(manifest, folder)
    print(f"Released {manifest['id']} ({len(manifest['files'])} files, gate_eligible={manifest['gate_eligible']})")


def cmd_adopt(args):
    manifest, folder = make_release(args.station, args.folder, args.note, lineage="reconstructed", first_version=0)
    publish_release(manifest, folder)
    print(f"Adopted {manifest['id']} (reconstructed lineage, not gate-eligible)")


def cmd_pull(args):
    published = locked_on_main() or read_lock()
    wanted = args.release or sorted(published.get("releases", {}).values())
    for release in wanted:
        print(f"{pull_release(release)['id']} ready in {PULLED.as_posix()}/{release}/")
    if not wanted:
        print("The lock names no data releases yet.")


def cmd_sync(args):
    print(sync_runs(args.runs))
    print(f"Releases on MLflow: {sync_releases()}")


def cmd_score(args, extra):
    if git("status", "--porcelain"):
        print("Note: uncommitted changes, so this run will be labelled exploratory.")
    command = [sys.executable, str(ROOT / "code/evaluate.py"), "run", "--name", args.name, *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        return result.returncode
    run_dir = re.findall(r"^Run: (.+)$", result.stdout, flags=re.MULTILINE)[-1].strip()
    try:
        print(sync_runs(None, only=run_dir))
    except Exception as exc:
        print(f"Run is sealed but not on MLflow ({type(exc).__name__}: {exc}). Retry with `line.py sync`.")
    if "--phase" not in extra:  # development runs only; the reward never reads assessment origins
        print("\n" + "-" * 30 + " reward (evaluation/REWARD.md) " + "-" * 30)
        subprocess.run([sys.executable, str(ROOT / "code/evaluation/reward.py"), "score", run_dir], cwd=ROOT, check=False)
        print("\n" + "-" * 28 + " insights (evaluation/INSIGHTS.md) " + "-" * 28)
        subprocess.run([sys.executable, str(ROOT / "code/evaluation/insights.py"), run_dir, "--top", "6"],
                       cwd=ROOT, check=False)
    return 0


def cmd_lock(args):
    from evaluation.core import read_json
    from evaluation.protocol import source_pins

    lock = read_lock() if LOCK_PATH.is_file() else empty_lock()
    lock["line_started_utc"] = utc_now() if args.restart_clock else lock["line_started_utc"] or utc_now()
    if args.evaluator:
        lock["evaluators"][args.evaluator] = evaluator_hash(source_pins(ROOT))
        lock["current_evaluator"] = args.evaluator
    if args.suite:
        lock["suite_id"] = read_json(Path(args.suite) / "manifest.json")["suite_id"]
    if args.mlflow_uri:
        lock["mlflow_uri"] = args.mlflow_uri
    for release in args.release or []:
        if find_release(release) is None:
            raise LineError(f"unknown release: {release}")
        lock["releases"][release.rsplit("-v", 1)[0]] = release
    write_lock(lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    print("Commit line.lock.json on main for this to take effect.")


def cmd_publish_suite(args):
    from evaluation.core import verify_suite

    manifest = verify_suite(args.suite)
    client = mlflow_client()
    if find_tagged(client, SUITE_EXPERIMENT, "line.suite_id", manifest["suite_id"]):
        print("Already published.")
        return
    experiment = ensure_experiment(client, SUITE_EXPERIMENT)
    run = client.create_run(experiment, run_name=SUITE_NAME, tags={"line.suite_id": manifest["suite_id"]})
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / f"{SUITE_NAME}.zip"
        zip_suite(args.suite, archive)
        client.log_artifact(run.info.run_id, str(archive))
    client.set_terminated(run.info.run_id)
    print(f"Published {SUITE_NAME} ({manifest['suite_id'][:8]}).")


def cmd_get_suite(args):
    published = locked_on_main() or read_lock()
    print(f"{SUITE_NAME}: {fetch_suite(args.suite, published['suite_id'])}")


def cmd_add_user(args):
    from mlflow.server import get_app_client

    password = secrets.token_urlsafe(18)
    try:
        get_app_client("basic-auth", tracking_uri=mlflow_uri()).create_user(username=args.username, password=password)
    except Exception as exc:
        if isinstance(exc, ImportError):
            raise LineError(f"Adding a user needs one extra package. Run: {SERVER_COMMAND} add-user {args.username}") from exc
        if "404" in str(exc):
            raise LineError("The server at this address has no login system. Start it with `line.py serve`.") from exc
        raise
    print(f"Created {args.username}. Password, shown once: {password}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    default_suite = ROOT / "outputs/evaluation" / SUITE_NAME
    default_runs = ROOT / "outputs/evaluation/runs"

    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.add_argument("--strict", action="store_true", help="exit 1 if anything is out of sync")
    status.add_argument("--offline", action="store_true", help="skip the MLflow check")
    commands.add_parser("migration", help="is every uncommitted file in the old checkout safely on a branch?")
    release = commands.add_parser("release")
    release.add_argument("station", choices=STATIONS)
    release.add_argument("folder", type=Path)
    release.add_argument("--note", required=True)
    release.add_argument("--upstream", nargs="*", default=[])
    release.add_argument("--gate-eligible", action="store_true", help="built per origin with nothing after that origin")
    adopt = commands.add_parser("adopt")
    adopt.add_argument("station", choices=STATIONS)
    adopt.add_argument("folder", type=Path)
    adopt.add_argument("--note", required=True)
    pull = commands.add_parser("pull")
    pull.add_argument("release", nargs="*")
    sync = commands.add_parser("sync")
    sync.add_argument("--runs", type=Path, default=default_runs)
    score = commands.add_parser("score", help="evaluate.py run, then mirror to MLflow; other flags pass through")
    score.add_argument("--name", required=True)
    setup = commands.add_parser("setup")
    setup.add_argument("station", nargs="+", choices=[*STATIONS, "infra"])
    lock = commands.add_parser("lock")
    lock.add_argument("--evaluator", help="name this checkout's scoring code, e.g. eval-v1")
    lock.add_argument("--suite", type=Path, help="suite folder to lock")
    lock.add_argument("--mlflow-uri")
    lock.add_argument("--release", nargs="*")
    lock.add_argument("--restart-clock", action="store_true", help="runs before now become legacy")
    publish = commands.add_parser("publish-suite")
    publish.add_argument("--suite", type=Path, default=default_suite)
    get_suite = commands.add_parser("get-suite")
    get_suite.add_argument("--suite", type=Path, default=default_suite)
    server = commands.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=5000)
    server.add_argument("--allowed-hosts", default="")
    add_user = commands.add_parser("add-user")
    add_user.add_argument("username")

    args, extra = parser.parse_known_args(argv)
    if args.command == "score":
        return cmd_score(args, extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "setup":
        for name in args.station:
            print(f"{name}: {setup_station(name)}")
        return 0
    if args.command == "serve":
        return serve(args.host, args.port, args.allowed_hosts)
    return {"status": cmd_status, "migration": cmd_migration, "release": cmd_release, "adopt": cmd_adopt, "pull": cmd_pull, "sync": cmd_sync,
            "lock": cmd_lock, "publish-suite": cmd_publish_suite, "get-suite": cmd_get_suite,
            "add-user": cmd_add_user}[args.command](args) or 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LineError as exc:
        print(f"line: {exc}", file=sys.stderr)
        sys.exit(1)
