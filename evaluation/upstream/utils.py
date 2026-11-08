"""Storage and challenge logic shared by the tabs.

Every registration, team, submission and judge score is one JSON file in the
private STORAGE_REPO (or in STORAGE_DIR for local tests). Writing one file per
entry means two people submitting at the same time cannot overwrite each other.
"""

import io
import json
import re
import secrets
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from config import (DEADLINE, MAX_SUBMISSION_FILE_MB, MAX_SUBMISSIONS_PER_DAY,
                    MAX_TEAM_SIZE, RUBRIC, SCORE_RANGE, STORAGE_DIR, STORAGE_REPO,
                    SUBMISSION_FILES, TOKEN)

api = HfApi(token=TOKEN)
# without a token: team datasets must be readable by anyone (public)
public_api = HfApi(token=False)
# Entries are never edited: leaving a team and deleting an empty team are
# entries too ("leaves", "team_deletions"), and the current state is derived.
KINDS = ("teams", "participants", "leaves", "team_deletions", "submissions", "judgements")


def now() -> datetime:
    return datetime.now(timezone.utc)


# ── Storage ───────────────────────────────────────────────────────────────────

class Store:
    """Append-only records, one JSON file per record under <kind>/."""

    def __init__(self):
        self._cache: dict[str, dict] = {}  # path -> record (files never change)
        self._lock = threading.Lock()

    def _paths(self, kind):
        if STORAGE_DIR:
            root = Path(STORAGE_DIR)
            return [str(p.relative_to(root)) for p in sorted((root / kind).glob("*.json"))]
        try:
            files = api.list_repo_files(STORAGE_REPO, repo_type="dataset")
        except Exception:
            return []
        return sorted(f for f in files if f.startswith(f"{kind}/") and f.endswith(".json"))

    def _read(self, path):
        if STORAGE_DIR:
            return json.loads((Path(STORAGE_DIR) / path).read_text())
        local = hf_hub_download(STORAGE_REPO, path, repo_type="dataset", token=TOKEN)
        return json.loads(Path(local).read_text())

    def records(self, kind) -> list[dict]:
        with self._lock:
            for p in self._paths(kind):
                if p not in self._cache:
                    r = self._read(p)
                    # entries written before team IDs existed (Oct 26, 2026)
                    if "team" in r and "team_id" not in r:
                        r["team_id"] = f"legacy:{norm(r['team'])}"
                    self._cache[p] = r
            return [r for p, r in self._cache.items() if p.startswith(f"{kind}/")]

    def add(self, kind, record, files: dict[str, bytes | str] | None = None):
        """Store a record, plus optional extra files in one commit.

        `files` maps a repo path to bytes or to a local file path (for large
        files, which are then not read into memory).
        """
        # microseconds keep entries in order (e.g. join then leave) after a restart
        stamp = now().strftime("%Y%m%dT%H%M%S%f")
        path = f"{kind}/{stamp}_{uuid.uuid4().hex[:8]}.json"
        payload = {path: json.dumps(record, indent=1).encode(), **(files or {})}
        if STORAGE_DIR:
            for p, data in payload.items():
                target = Path(STORAGE_DIR) / p
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data if isinstance(data, bytes) else Path(data).read_bytes())
        else:
            api.create_commit(STORAGE_REPO, repo_type="dataset",
                              operations=[CommitOperationAdd(
                                  p, io.BytesIO(d) if isinstance(d, bytes) else d)
                                  for p, d in payload.items()],
                              commit_message=f"Add {kind[:-1]}")
        with self._lock:
            self._cache[path] = record

    def write_files(self, files: dict[str, bytes], message: str):
        """Overwrite fixed paths (e.g. the score files) in one commit.

        Records are append-only (add); this is only for derived files that are
        rebuilt from the records, so replacing them loses nothing.
        """
        if STORAGE_DIR:
            for p, data in files.items():
                target = Path(STORAGE_DIR) / p
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        else:
            api.create_commit(STORAGE_REPO, repo_type="dataset", commit_message=message,
                              operations=[CommitOperationAdd(p, io.BytesIO(d)) for p, d in files.items()])

    def local_path(self, path) -> str:
        if STORAGE_DIR:
            return str(Path(STORAGE_DIR) / path)
        return hf_hub_download(STORAGE_REPO, path, repo_type="dataset", token=TOKEN)

    def read_file(self, path) -> bytes:
        return Path(self.local_path(path)).read_bytes()


store = Store()


# ── Teams and registration ────────────────────────────────────────────────────

def norm(name: str) -> str:
    return " ".join(name.lower().split())


def _in_order(kind):
    return sorted(store.records(kind), key=lambda r: r["timestamp"])


def teams() -> dict[str, dict]:
    """Team key (normalised name) -> record of the existing (not deleted) team."""
    deleted = {d["team_id"] for d in store.records("team_deletions")}
    return {norm(t["team"]): t for t in _in_order("teams") if t["team_id"] not in deleted}


def _active_participants() -> dict[str, dict]:
    """Username -> current membership record (joins minus leaves, in time order)."""
    events = ([("join", r) for r in store.records("participants")]
              + [("leave", r) for r in store.records("leaves")])
    active = {}
    for kind, r in sorted(events, key=lambda e: e[1]["timestamp"]):
        if kind == "join":
            active[r["username"]] = r
        elif active.get(r["username"], {}).get("team_id") == r["team_id"]:
            del active[r["username"]]
    return active


def participant(username):
    return _active_participants().get(username)


def members(team_key):
    team = teams().get(team_key)
    if team is None:
        return []
    return [p for p in _active_participants().values() if p["team_id"] == team["team_id"]]


CMU_DOMAIN = "andrew.cmu.edu"  # CMU students register with their Andrew e-mail (About › Teams)


def is_cmu_email(email: str) -> bool:
    """A CMU student account: the e-mail domain is exactly andrew.cmu.edu."""
    return (email or "").rsplit("@", 1)[-1].strip().lower() == CMU_DOMAIN


_CMU_AFFILIATION = re.compile(r"carneg\w*\s+mell|\bcmu\b", re.IGNORECASE)  # tolerates "Carnegire Mellon"


def is_cmu_affiliation(affiliation: str) -> bool:
    """The self-declared affiliation names CMU: "Carnegie Mellon …" or the word "CMU"."""
    return bool(_CMU_AFFILIATION.search(affiliation or ""))


def is_cmu_member(member) -> bool:
    """A CMU member: an Andrew e-mail OR a CMU affiliation. `member` is a participant
    record (dict with `email`, `affiliation`) or a bare e-mail string."""
    if isinstance(member, dict):
        return is_cmu_email(member.get("email", "")) or is_cmu_affiliation(member.get("affiliation", ""))
    return is_cmu_email(member)


def is_cmu_team(members: list) -> bool:
    """CMU prize rule: more than half of the current members are CMU members (is_cmu_member)."""
    return len(members) > 0 and 2 * sum(map(is_cmu_member, members)) > len(members)


def cmu_teams() -> set[str]:
    """Team keys of the CMU teams, from the current membership (people who left do not count)."""
    by_team = {}
    for p in _active_participants().values():
        by_team.setdefault(p["team_id"], []).append(p)
    return {key for key, t in teams().items() if is_cmu_team(by_team.get(t["team_id"], []))}


def register(username, full_name, email, affiliation, team_name, join_code, create):
    """Create or join a team. Returns (ok, message)."""
    full_name, email, affiliation, team_name = (
        full_name.strip(), email.strip(), affiliation.strip(), team_name.strip())
    if not all([full_name, email, team_name]):
        return False, "Full name, e-mail and team name are required."
    if "@" not in email:
        return False, "Please enter a valid e-mail address."
    if participant(username):
        return False, (f"You are already in team **{participant(username)['team']}**. "
                       "Leave it first to join another team.")
    if any(p["email"].lower() == email.lower() for p in _active_participants().values()):
        return False, "This e-mail address is already registered."
    if now() > DEADLINE:
        return False, "The challenge is over; registration is closed."

    existing = teams().get(norm(team_name))
    if create:
        if existing:
            return False, "This team name is taken. Choose another one, or join the team."
        code = secrets.token_hex(3).upper()
        team_id = uuid.uuid4().hex
        store.add("teams", {"team": team_name, "team_id": team_id, "join_code": code,
                            "created_by": username, "timestamp": now().isoformat()})
        team = team_name
        msg = (f"Team **{team_name}** created. Share this join code with your "
               f"teammates: **{code}**")
    else:
        if not existing:
            return False, "No team with this name. Check the spelling, or create it."
        if join_code.strip().upper() != existing["join_code"]:
            return False, "Wrong join code. Ask a member of the team for it."
        if MAX_TEAM_SIZE and len(members(norm(team_name))) >= MAX_TEAM_SIZE:
            return False, f"This team already has {MAX_TEAM_SIZE} members."
        team, team_id = existing["team"], existing["team_id"]
        msg = f"You joined team **{team}**."
    store.add("participants", {"username": username, "full_name": full_name,
                               "email": email, "affiliation": affiliation,
                               "team": team, "team_id": team_id,
                               "timestamp": now().isoformat()})
    return True, msg


def leave_team(username):
    """Leave your team; the team is deleted when its last member leaves."""
    p = participant(username)
    if not p:
        return False, "You are not in a team."
    if now() > DEADLINE:
        return False, "The challenge is over; teams can no longer change."
    key = norm(p["team"])
    store.add("leaves", {"username": username, "team": p["team"], "team_id": p["team_id"],
                         "timestamp": now().isoformat()})
    if not members(key):
        store.add("team_deletions", {"team": p["team"], "team_id": p["team_id"],
                                     "deleted_by": username, "timestamp": now().isoformat()})
        return True, (f"You left team **{p['team']}**. It had no other members, so it "
                      "was deleted, with its submissions.")
    return True, f"You left team **{p['team']}**."


def my_team(username) -> dict | None:
    """Team name, join code and members of the user's team, or None."""
    p = participant(username)
    if not p:
        return None
    key = norm(p["team"])
    return {"team": p["team"], "join_code": teams()[key]["join_code"],
            "members": [(m["full_name"], m["username"]) for m in members(key)]}


def team_info(username) -> str:
    t = my_team(username)
    if t is None:
        return "You are not in a team yet: create one or join one below."
    names = "\n".join(f"- {name} (@{user})" for name, user in t["members"])
    return (f"You are in team **{t['team']}**.\n\nMembers:\n{names}\n\n"
            f"Join code for new teammates: **{t['join_code']}**")


# ── Submissions ───────────────────────────────────────────────────────────────

# https://huggingface.co/datasets/<owner>/<name>, huggingface.co/datasets/<owner>/<name>
# or <owner>/<name>; model and Space links do not match (a huggingface.co link
# must continue with datasets/, so huggingface.co/gpt2 is not read as owner/name)
DATASET_LINK = re.compile(
    r"^(?:https?://)?(?:huggingface\.co/datasets/|(?!huggingface\.co/))"
    r"([\w-][\w.-]*/[\w-][\w.-]*?)/?$")


def submissions(team_key=None) -> list[dict]:
    """Submissions of existing teams in time order (of one team if team_key)."""
    ids = {t["team_id"]: key for key, t in teams().items()}
    subs = [s for s in _in_order("submissions") if s["team_id"] in ids]
    if team_key is not None:
        subs = [s for s in subs if ids[s["team_id"]] == team_key]
    return subs


def parse_dataset_link(link) -> str | None:
    """'owner/name' from a Hugging Face dataset link or ID, or None."""
    m = DATASET_LINK.match((link or "").strip())
    return m.group(1) if m else None


def fetch_dataset(repo_id) -> tuple[str | None, dict[str, bytes], str | None]:
    """(commit, {file: bytes}) of a public dataset at its current commit, or an error."""
    try:
        info = public_api.dataset_info(repo_id, files_metadata=True)
    except Exception:
        return None, {}, (f"Dataset **{repo_id}** was not found, or it is not public. "
                          "Check the link and make the dataset public.")
    sizes = {s.rfilename: s.size for s in info.siblings}
    missing = [f for f in SUBMISSION_FILES if f not in sizes]
    if missing:
        return None, {}, f"Your dataset is missing: {', '.join(f'`{f}`' for f in missing)}."
    too_big = [f for f in SUBMISSION_FILES if (sizes[f] or 0) > MAX_SUBMISSION_FILE_MB * 1_000_000]
    if too_big:
        return None, {}, f"Larger than {MAX_SUBMISSION_FILE_MB} MB: {', '.join(too_big)}."
    files = {f: Path(hf_hub_download(repo_id, f, repo_type="dataset", revision=info.sha,
                                     token=False)).read_bytes()
             for f in SUBMISSION_FILES}
    return info.sha, files, None


CITY_IDS = set((Path(__file__).parent / "data" / "city_ids.txt").read_text().split())
YEARS = set(range(2019, 2026))
METRICS = {"annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"}
COLUMNS = ["city_id", "year", "metric", "predicted_value"]


def check_forecasts(csv_bytes) -> str | None:
    """None if submission.csv can be scored, otherwise what is wrong (first problems only)."""
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype={"city_id": str, "metric": str})
    except Exception:
        return "`submission.csv` could not be read as a CSV file."
    if list(df.columns) != COLUMNS:
        return f"`submission.csv` must have exactly these columns: `{','.join(COLUMNS)}`."
    problems = []

    def bad(mask, what, column):
        if mask.any():
            line = mask.idxmax() + 2  # +1 for the header, +1 for 1-based lines
            problems.append(f"{mask.sum()} rows {what} (first at line {line}: `{df.loc[mask, column].iloc[0]}`)")

    bad(~df["city_id"].isin(CITY_IDS), "have an unknown `city_id`", "city_id")
    year = pd.to_numeric(df["year"], errors="coerce")
    bad(~year.isin(YEARS), "have a `year` outside 2019–2025", "year")
    bad(~df["metric"].isin(METRICS), "have an unknown `metric`", "metric")
    value = pd.to_numeric(df["predicted_value"], errors="coerce")
    bad(value.isna() | value.abs().eq(float("inf")),
        "have a missing or non-numeric `predicted_value`", "predicted_value")
    bad(df.duplicated(["city_id", "year", "metric"], keep="first"), "are duplicates", "city_id")
    if not problems and len(df) != len(CITY_IDS) * len(YEARS) * len(METRICS):
        problems.append(f"it has {len(df)} rows; it needs {len(CITY_IDS) * len(YEARS) * len(METRICS):,} "
                        "(210 cities × 7 years × 4 metrics)")
    if problems:
        return "`submission.csv` cannot be scored yet: " + "; ".join(problems) + "."
    return None


def submit(username, dataset_link):
    """Validate and store a submission. Returns (ok, message).

    A submission is a link to the team's public Hugging Face dataset. Its
    submission.csv and README.md (the write-up) are copied at the current commit,
    so later changes to the dataset only count if the team submits again.
    """
    p = participant(username)
    if not p:
        return False, "Register your team first (Registration tab)."
    if now() > DEADLINE:
        return False, "The deadline has passed; submissions are closed."
    repo_id = parse_dataset_link(dataset_link)
    if not repo_id:
        return False, ("Paste the link to your Hugging Face **dataset**, like "
                       "https://huggingface.co/datasets/your-name/your-dataset "
                       "(model and Space links are not accepted).")
    commit, contents, error = fetch_dataset(repo_id)
    if error:
        return False, error
    error = check_forecasts(contents["submission.csv"])
    if error:
        return False, error
    team_key = norm(p["team"])
    today = now().date().isoformat()
    used = sum(s["timestamp"][:10] == today for s in submissions(team_key))
    if MAX_SUBMISSIONS_PER_DAY and used >= MAX_SUBMISSIONS_PER_DAY:
        return False, (f"Your team has used its {MAX_SUBMISSIONS_PER_DAY} submissions "
                       "for today. The count resets at 7 PM ET (midnight UTC).")

    sid = f"{now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    writeup_file = f"files/{sid}_README.md"
    submission_file = f"files/{sid}_submission.csv"
    # who submitted what: the submitter's Hugging Face username, team and time,
    # and the dataset version (commit) that was copied
    store.add("submissions", {
        "id": sid, "team": p["team"], "team_id": p["team_id"], "username": username,
        "timestamp": now().isoformat(), "dataset": repo_id, "commit": commit,
        "writeup_file": writeup_file, "submission_file": submission_file,
    }, files={writeup_file: contents["README.md"],
              submission_file: contents["submission.csv"]})

    left = (f" Submissions left today: {MAX_SUBMISSIONS_PER_DAY - used - 1}."
            if MAX_SUBMISSIONS_PER_DAY else "")
    return True, (f"Submission saved for **{p['team']}**: [{repo_id}]({dataset_url(repo_id)}) "
                  f"at commit `{commit[:7]}`.{left}\n\nChanged your dataset since? "
                  "Submit again: only the version recorded at your latest submission counts.\n\n"
                  + "Wait a few hours for your Live Score to appear on the leaderboard.")


def dataset_url(repo_id, commit=None) -> str:
    url = f"https://huggingface.co/datasets/{repo_id}"
    return f"{url}/tree/{commit}" if commit else url


def writeup_text(sub_record) -> str:
    """The README (write-up) of a stored submission ('' if none)."""
    if not sub_record.get("writeup_file"):
        return ""
    return store.read_file(sub_record["writeup_file"]).decode("utf-8", errors="replace")


def team_history(username) -> pd.DataFrame:
    """The submissions of the user's team, newest first, the counting one marked."""
    cols = ["Counts", "Time (UTC)", "Submitted by", "Dataset", "Commit"]
    p = participant(username) if username else None
    subs = submissions(norm(p["team"])) if p else []
    if not subs:
        return pd.DataFrame(columns=cols)
    rows = [{"Counts": "◀ counts" if i == len(subs) - 1 else "",
             "Time (UTC)": s["timestamp"][:16].replace("T", " "),
             "Submitted by": f"@{s['username']}",
             "Dataset": s.get("dataset", "–"), "Commit": (s.get("commit") or "–")[:7]}
            for i, s in enumerate(subs)]
    return pd.DataFrame(rows[::-1], columns=cols)


def latest_submissions() -> dict[str, dict]:
    """Team key -> its latest valid submission (the one that counts)."""
    ids = {t["team_id"]: key for key, t in teams().items()}
    latest = {}
    for s in submissions():
        latest[ids[s["team_id"]]] = s
    return latest


# ── Judging ───────────────────────────────────────────────────────────────────

def weighted_score(scores: dict) -> float:
    """One judge's rubric scores as 0-100 (all lowest = 0, all highest = 100).

    Shared by the ranking (judging_table) and the organizers' preview, so both
    always give the same number.
    """
    lo, hi = SCORE_RANGE
    return sum(w * (scores[k] - lo) / (hi - lo) for k, _, w, _ in RUBRIC)


def judge_scores(judge, team, scores: dict, comment):
    team_id = teams()[norm(team)]["team_id"]
    store.add("judgements", {"judge": judge, "team": team, "team_id": team_id,
                             "scores": scores, "comment": comment.strip(),
                             "timestamp": now().isoformat()})


def judging_table() -> pd.DataFrame:
    """Per existing team: weighted judge score (0-100, each judge's latest scores count)."""
    existing = {t["team_id"] for t in teams().values()}
    latest = {}
    keys = {k for k, _, _, _ in RUBRIC}
    for j in _in_order("judgements"):
        # scores saved under an older rubric (other criteria) do not count
        if j["team_id"] in existing and keys <= set(j["scores"]):
            latest[(j["judge"], norm(j["team"]))] = j
    rows = []
    for (judge, key), j in latest.items():
        rows.append({"team_key": key, "Team": j["team"], "judge": judge,
                     "weighted": weighted_score(j["scores"]), **{label: j["scores"][k]
                                              for k, label, _, _ in RUBRIC}})
    if not rows:
        return pd.DataFrame(columns=["Rank", "Team", "Judges", "Judge score (0-100)"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["team_key", "Team"]).agg(
        **{"Judges": ("judge", "nunique"), "Judge score (0-100)": ("weighted", "mean")},
        **{label: (label, "mean") for _, label, _, _ in RUBRIC}).reset_index()
    agg = agg.drop(columns="team_key").round(2).sort_values(
        "Judge score (0-100)", ascending=False)
    agg.insert(0, "Rank", range(1, len(agg) + 1))
    return agg
