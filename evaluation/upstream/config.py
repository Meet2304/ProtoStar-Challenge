import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("HF_TOKEN")

# ── Challenge Configuration ──────────────────────────────────────────────────

CHALLENGE_NAME = "Protostar Challenge: Climate Risk Intelligence"
ORG = "CMU-OSPO"

# Private dataset where teams, registrations, submissions and judge scores are
# stored, one JSON file per entry (concurrent writes cannot overwrite each other)
STORAGE_REPO = f"{ORG}/protostar-track4-submissions"
# For local tests: store entries in this folder instead of on the Hub
STORAGE_DIR = os.environ.get("STORAGE_DIR")

# Timeline (event announcement): submissions close Tuesday, November 10, 11:59 PM ET;
# finalists Nov 11–12; in-person final presentations Friday, November 13.
TZ = ZoneInfo("America/New_York")
CHALLENGE_START = datetime(2026, 11, 2, 0, 0, tzinfo=TZ)
DEADLINE = datetime(2026, 11, 10, 23, 59, tzinfo=TZ)

# A submission is a link to a public Hugging Face dataset holding these files;
# they are copied to the submissions repo at the commit current when submitting
SUBMISSION_FILES = ("submission.csv", "README.md")
MAX_SUBMISSION_FILE_MB = 5
LEADERBOARD_OPEN = True    # True: Leaderboard tab shown; About drops the "Coming soon" badge
# The Leaderboard tab appears when the leaderboard opens; PREVIEW_LEADERBOARD=1 shows it earlier (local tests)
SHOW_LEADERBOARD_TAB = LEADERBOARD_OPEN or os.environ.get("PREVIEW_LEADERBOARD") == "1"

# ── Scoring ──────────────────────────────────────────────────────────────────

# Private answer key (2019–2025) and baselines, read with TOKEN
ANSWER_KEY_REPO = f"{ORG}/protostar-climate-ground-truth"
ANSWER_KEY_FILE = "answer_key.csv"
BASELINE_SCORES_FILE = "baselines/baseline_scores.csv"
REFERENCE_MAE_FILE = "baselines/reference_mae_30_year_average.csv"
# For local tests: read the files above from this folder instead of the Hub
ANSWER_KEY_DIR = os.environ.get("ANSWER_KEY_DIR")

# Years per split (the answer key's `split` column); only `scored` rows count
LIVE_YEARS = range(2019, 2022)    # 2019–2021, shown during the challenge
FINAL_YEARS = range(2022, 2026)   # 2022–2025, shown after the deadline
METRICS = ("annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd")

# Every cycle rescores all latest submissions from scratch (idempotent), well
# inside the 24-hour delay announced to teams
SCORING_INTERVAL_HOURS = 6
# One extra cycle this long after the deadline, so the final tables are complete
# before finalists are chosen (submissions made at 11:59 PM are committed by then)
DEADLINE_GRACE_MINUTES = 5
# Space variable SCORING_ENABLED=0 turns the background scoring off (e.g. local UI work)
SCORING_ENABLED = os.environ.get("SCORING_ENABLED", "1") != "0"
# Public texts never state how or how often the board is updated (manual now, automatic
# later): they only say a new submission takes "a few hours" to appear.
SCORE_META_FILE = "scores/meta.json"          # last update, status, next update

# Score files written to STORAGE_REPO (or STORAGE_DIR)
LEADERBOARD_FILE = "scores/leaderboard.csv"   # live split: teams + baselines
FINAL_SCORES_FILE = "scores/final_scores.csv"  # final split: read only after the deadline

# Test teams (e.g. the dummy participant) are scored but not shown on the leaderboard
TEST_TEAMS = {t.strip() for t in os.environ.get("TEST_TEAMS", "").split(",") if t.strip()}

# Reference teams (e.g. "Ocarina of Ten", the example notebook's 10-year mean) are
# shown and ranked like teams, with a "reference" tag, so teams can compare themselves
REFERENCE_TEAMS = {t.strip() for t in os.environ.get("REFERENCE_TEAMS", "").split(",") if t.strip()}


def show_final() -> bool:
    """Final scores (teams and baselines) become visible after the deadline.

    A function, not a constant: the Space runs for days, so it is checked at
    every request instead of once at startup.
    """
    return datetime.now(TZ) > DEADLINE

# Not decided by the organisers yet: None = no limit
MAX_SUBMISSIONS_PER_DAY = 10   # per team, UTC day (resets 7 PM ET); valid submissions only
MAX_TEAM_SIZE = 5

# Hugging Face usernames allowed to use the Judging tab (Space variable JUDGES,
# comma-separated)
JUDGES = {u.strip() for u in os.environ.get("JUDGES", "").split(",") if u.strip()}
# Hugging Face usernames that see the judges' panel in preview mode (Space variable
# ORGANIZERS, comma-separated): same view as a judge, scores are never saved.
# A username in both lists is a judge.
ORGANIZERS = {u.strip() for u in os.environ.get("ORGANIZERS", "").split(",") if u.strip()}
# Manual switch for saving judge scores (Space variable JUDGING_OPEN): "1" = judges can
# save; "0" or unset = closed, judges get the organizers' preview mode. Unset means
# closed, so a deploy never opens scoring by accident. Changing it restarts the Space.
JUDGING_OPEN = os.environ.get("JUDGING_OPEN", "0") == "1"

# Final judging rubric: (key, label, weight in %, description shown to judges)
RUBRIC = [  # (key, criterion, weight %, what judges consider)
    ("reuse", "Demonstrated Reuse & Extensibility", 25,
     "Meaningful use or rigorous attempted reuse of ProtoStar artifacts; useful extension, diagnosis, or "
     "recommendation for future researchers. Evidence about discovery and reuse friction is scored under "
     "Feedback to the Ecosystem, not here."),
    ("interop", "Interoperability & Integration", 20,
     "Successful combination or portability across repositories, institutions, agencies, environments, or "
     "domains, or rigorous evidence showing why integration fails; value and limits of common standards and "
     "machine-actionable formats."),
    ("reproducibility", "Reproducibility & Openness", 20,
     "Ability for others to reproduce the work using provided materials; for submissions declared in advance "
     "as negative results, a well-documented attempt that isolates why reproduction is not possible with the "
     "available artifact, its metadata, or its documented environment. Recommendations arising from that "
     "finding are scored under Feedback to the Ecosystem, not here."),
    ("value", "Scientific, Research, or Societal Value", 15,
     "Useful knowledge, new research capability, improved decision-making, or meaningful public benefit."),
    ("feedback", "Feedback to the Ecosystem", 15,
     "Specific, actionable evidence about barriers, failed reuse or reproduction, metadata issues, licensing, "
     "documentation, infrastructure, policy, or other opportunities for improvement."),
    ("creativity", "Technical Creativity", 5,
     "Novel methods, AI approaches, technical solutions, or combinations of resources that advance the goals "
     "of the challenge."),
]
SCORE_RANGE = (1, 5)  # judges' scale per criterion: not decided yet

# Training data (up to 2018-12-31); the tab texts are in content/about.md and content/data.md
DATASETS = [
    (f"{ORG}/protostar-climate-train-to-2018",
     "Protostar Climate Challenge: training data up to 2018",
     ["daily", "annual_indices", "city_info"]),
]
