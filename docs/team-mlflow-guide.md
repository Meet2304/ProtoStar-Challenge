# Team guide: logging your models to the shared MLflow

Everyone logs to one MLflow server. Each run records the model, the data it trained on, the test set, the evaluator
version, the code commit and the time. Background: [the assembly-line plan](plans/assembly-line-plan.md).

## Set up once

You need the server address and your own username and password from Meet.

```bash
git clone https://github.com/Meet2304/ProtoStar-Challenge
cd ProtoStar-Challenge
git switch -c team/<your-name> origin/main
uv sync
```

Set these three variables in your shell, or in Colab Secrets. **Never put them in a file in this repo.**

```bash
export MLFLOW_TRACKING_URI="<address from Meet>"
export MLFLOW_TRACKING_USERNAME="<your-name>"
export MLFLOW_TRACKING_PASSWORD="<your-password>"
```

On Windows PowerShell, use `$env:MLFLOW_TRACKING_URI = "..."`, and likewise for the other two.

Then fetch the shared test set and any data releases:

```bash
uv run python scripts/line.py get-suite
uv run python scripts/line.py pull
uv run python scripts/line.py status
```

**Do not run `evaluate.py prepare`.** A freshly built suite has the same data but a different ID, so your runs could not be
compared with anyone else's.

## Each time you try a model

1. Write `forecast(train, targets, origin, seed, **params)` in `notebooks/<your-name>/my_model.py`. It returns
   `city_id,year,metric,predicted_value`.
2. **Commit it.** Uncommitted code is still logged, but only as `exploratory`.
3. Score it:

```bash
uv run python scripts/line.py score --name kohav-gbm-v1 \
  --adapter notebooks/Kohav/my_model.py:forecast --assets evaluation/no-assets.json
```

If your model reads extra files, replace `no-assets.json` with your own manifest; see `evaluation/ASSETS.md`.

**Training on Colab?** Save predictions for each development origin, then score them from a laptop on the server's network
with `evaluate.py score` (see `evaluation/TEAM_GUIDE.md`) and run `line.py sync`.

Before you begin each session, run `git fetch`, then `uv run python scripts/line.py status`, and do what it says.

## Reading the results

Open the server address in a browser and go to the experiment starting `protostar/4029caa1`. Filter with:

```
tags.`line.status` = 'official'
```

| Label | Meaning |
|---|---|
| `official` | Clean commit, released evaluator, shared test set, eligible data. Comparable with every other official run. |
| `exploratory` | Logged and visible, but not comparable. The `line.reasons` tag says why. |
| `legacy` | Made before the line started. |

If the server is unreachable, your work is safe: the sealed run stays on your disk. Run `line.py sync` later.

## Rules that still apply

The data rule in `code/AGENTS.md` comes first: no weather data after 2018-12-31. Choose and tune models only on the
development origins. Never open assessment origins 2004 or 2011; the integrator runs that once, for the chosen model.

## For Meet: running the server

Run in your own terminal so it outlives any agent. The first start asks you to choose an admin password.

```powershell
cd "D:\Projects_Ad Astra\Project_27_ProtoStar Challenge\wt\infra"
uv run --with flask-wtf python scripts/line.py serve --allowed-hosts "127.0.0.1:5000,localhost:5000,meetpc.taild2c878.ts.net,meetpc.taild2c878.ts.net:443"
```

Without `--allowed-hosts`, the server works locally but refuses teammates coming in through Tailscale (403).

Add a teammate (the password is shown once, so send it privately):

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_TRACKING_USERNAME = "admin"
$env:MLFLOW_TRACKING_PASSWORD = "<admin password>"
uv run --with flask-wtf python scripts/line.py add-user kohav
```

Publish the test set once with `uv run python scripts/line.py publish-suite`, then backfill with
`uv run python scripts/line.py sync`.

**Reaching it from other machines:** the server stays on `127.0.0.1`, and Tailscale carries teammates to it, so it is
never on the public internet. Set up on 2026-11-08: this PC is `meetpc` on the tailnet, and `tailscale serve --bg 5000`
(set once; it survives restarts) forwards `https://meetpc.taild2c878.ts.net` to the server. To let a teammate in, share
`meetpc` with them at https://login.tailscale.com/admin/machines (the `...` menu, **Share**); they install Tailscale and
accept. Teammates use `https://meetpc.taild2c878.ts.net` as `MLFLOW_TRACKING_URI`. Agents on this PC keep using
`http://127.0.0.1:5000`. Your PC must stay awake for teammates to log.
