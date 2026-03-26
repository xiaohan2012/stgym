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

## Code Sync

The server pulls from git — push locally, then pull on the server. The branch must match on both sides.

```bash
# 1. Check current branch locally
git branch --show-current

# 2. Push local changes
git push

# 3. Pull on the server (confirm it's on the same branch)
ssh cyy2 "cd /root/stgym && git branch --show-current && git pull"
```

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

## Typical Workflow: Run a Single Experiment

```bash
# 1. Push code
git push

# 2. Pull on server + launch in screen
ssh cyy2 "cd /root/stgym && \
  git pull && \
  source .venv/bin/activate && \
  screen -S run -X stuff 'python run_experiment_by_yaml.py <config_path> [--no-tracking] [--mlflow-uri URI]\n'"
```

## Typical Workflow: Run a Sweep

```bash
# 1. Push code
git push

# 2. Pull on server + launch sweep in screen
ssh cyy2 "cd /root/stgym && \
  git pull && \
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

## Notes

- Always reuse the `run` screen; never create a second one unless the user explicitly asks.
- If a new screen must be created, activate the venv before launching any Python command.
- MLflow tracking server runs at `http://127.0.0.1:5001` on the server (port-forwarded locally).
- Sweep configs live in `conf/exp/*.yaml`; design spaces in `conf/design_space/`.
