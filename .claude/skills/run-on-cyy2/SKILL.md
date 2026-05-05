---
name: run-on-cyy2
description: Run commands, sync code/data, and manage jobs on the cyy2 GPU server. Use this skill whenever the user wants to run anything on cyy2, check GPU server status, sync code or data to the remote server, manage screen sessions, or execute training/sweep scripts remotely. Supersedes the run_on_cyy custom command.
---

# run-on-cyy2

Skill for interacting with the cyy2 GPU server: running commands, syncing code and data, and managing long-running jobs.

## Server Details

- **Host**: `cyy2` (SSH alias configured in `~/.ssh/config`)
- **Project root**: `/root/stgym`
- **Virtual environment**: `/root/stgym/.venv` — activate with `source .venv/bin/activate`
- **Screen session**: named `run` (reuse existing; create only if absent)

## Running Commands on cyy2

### One-off commands (short-lived)

```bash
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && <command>"
```

### Long-running jobs (training, sweeps)

Always use the existing `run` screen session to avoid losing jobs on disconnect:

```bash
# Attach to existing screen and send command
ssh cyy2 "screen -S run -X stuff '<command>\n'"

# Or, if screen doesn't exist yet, create it first:
ssh cyy2 "screen -dmS run bash -c 'cd /root/stgym && source .venv/bin/activate && <command>'"
```

**Check if screen exists first:**
```bash
ssh cyy2 "screen -ls | grep run"
```

**View screen output (tail last N lines of scrollback):**
```bash
ssh cyy2 "screen -S run -X hardcopy /tmp/screen_dump && cat /tmp/screen_dump" | tail -50
```

## Pre-flight Checklist (required before running experiments or sweeps)

Run these steps before every code-execution operation (single experiment or sweep). Skip for status checks, artifact fetches, or screen inspection.

### Step 1: Check and reconcile branches

```bash
# Get local branch
LOCAL_BRANCH=$(git branch --show-current)

# Get cyy2 branch
REMOTE_BRANCH=$(ssh cyy2 "cd /root/stgym && git branch --show-current")

echo "Local: $LOCAL_BRANCH  |  cyy2: $REMOTE_BRANCH"
```

If they differ, **inform the user** and ask which branch to use. Default recommendation: check out the local branch on cyy2. Claude will perform the checkout:

```bash
# Check out local branch on cyy2 (default resolution)
ssh cyy2 "cd /root/stgym && git fetch origin && git checkout <LOCAL_BRANCH>"
```

Do not proceed until both sides are on the same branch.

### Step 2: Push local changes

```bash
git push
```

If the push is rejected (remote has diverged), **stop and inform the user**. Offer two options:
1. **Resolve via Claude Code** — Claude will run the appropriate git commands (pull + rebase, or merge)
2. **Resolve manually** — user handles the divergence themselves

Do not proceed until the push succeeds.

### Step 3: Pull on cyy2

```bash
ssh cyy2 "cd /root/stgym && git pull"
```

If `git pull` fails, **stop and surface the full error** to the user. Do not proceed.

## Data Sync

Raw datasets live on the server. To sync them to the local machine:

```bash
# Script on the server (run from project root)
ssh cyy2 "cd /root/stgym && bash scripts/rsync-data.sh"
```

Or copy specific files locally:
```bash
scp cyy2:/root/stgym/data/<path> ./local/destination/
```

## Checking Server Status

```bash
# Ray cluster / running jobs (primary way to check GPU activity)
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && ray status"

# Running Python processes
ssh cyy2 "pgrep -a python"

# Disk usage
ssh cyy2 "df -h /root"
```

## Typical Workflow 1: Run a Single Experiment

```bash
# 1. Run the Pre-flight Checklist (branch sync + push + pull)

# 2. Launch in screen
ssh cyy2 "cd /root/stgym && \
  source .venv/bin/activate && \
  screen -S run -X stuff 'python run_experiment_by_yaml.py <config_path> [--no-tracking] [--mlflow-uri URI]\n'"
```

## Typical Workflow 2: Run a Sweep

```bash
# 1. Run the Pre-flight Checklist (branch sync + push + pull)

# 2. Launch sweep in screen
ssh cyy2 "cd /root/stgym && \
  source .venv/bin/activate && \
  screen -S run -X stuff 'python run_rct.py +exp=<exp> design_space=<space> resource=gpu-6 sample_size=<n>\n'"
```

Only run `rsync-data.sh` if the user explicitly asks to sync data.

## Retrieving MLflow Artifacts from cyy2

MLflow uses the default local store at `/root/stgym/mlruns`. To fetch an artifact (e.g. `training_error.txt`) from a run:

```bash
# Find the artifact path on the server
ssh cyy2 "find /root/stgym/mlruns -name 'training_error.txt' -path '*<run_id>*'"

# Copy it locally to /tmp
scp cyy2:<path-from-above> /tmp/<prefix>/training_error.txt
```

## Installing Dependencies

The server does not have `uv` installed. Use `pip` for package installation:

```bash
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && pip install <package>"
```

## Notes

- Always reuse the `run` screen; never create a second one unless the user explicitly asks.
- If a new screen must be created, activate the venv before launching any Python command.
- MLflow tracking server runs at `http://127.0.0.1:5001` on the server (port-forwarded locally).
- Sweep configs live in `conf/exp/*.yaml`; design spaces in `conf/design_space/`.
- Use `pip install` for dependencies — `uv` is not available on cyy2.
