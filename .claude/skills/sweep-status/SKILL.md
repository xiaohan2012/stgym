---
name: sweep-status
description: Check sweep progress on cyy2 GPU server. Runs sweep_status.py in a dedicated screen session and saves output to persistent log files. Use when checking experiment sweep status, monitoring running sweeps, or diagnosing stale runs.
---

# sweep-status

Check sweep progress on the cyy2 GPU server with persistent logging.

## Parameters

- **Experiment ID** (`-i <id>`) — MLflow experiment ID (required, unless using `-n`)
- **Experiment Name** (`-n <name>`) — MLflow experiment name (alternative to `-i`)
- **Sample Size** (`--sample-size <n>`) — Number of runs to sample per dimension (default: 100)
- **Stale Threshold** (`--stale-threshold <min>`) — Minutes after which a RUNNING run is considered stale/OOM-killed (default: 20)

## Workflow

### 1. Ensure screen session exists

```bash
# Check if sweep-status screen exists
ssh cyy2 "screen -ls | grep sweep-status"

# If not found, create it
ssh cyy2 "screen -dmS sweep-status bash -c 'cd /root/stgym && source .venv/bin/activate && exec bash'"
```

### 2. Run sweep_status.py

```bash
# Send command to the sweep-status screen
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i <experiment-id> [options]\n'"
```

### 3. Capture output to local log

```bash
# Create log directory if needed
mkdir -p logs/sweep-status

# Wait briefly for command to execute, then capture screen output
sleep 3
ssh cyy2 "screen -S sweep-status -X hardcopy /tmp/sweep_status_dump && cat /tmp/sweep_status_dump" > "logs/sweep-status/<experiment-id>-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
```

### 4. Display output

```bash
cat "logs/sweep-status/<experiment-id>-<datetime>.log"
```

## Examples

### Check status by experiment ID

```bash
# 1. Ensure screen exists
ssh cyy2 "screen -ls | grep sweep-status" || ssh cyy2 "screen -dmS sweep-status bash -c 'cd /root/stgym && source .venv/bin/activate && exec bash'"

# 2. Run status check
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i 42\n'"

# 3. Capture and display
mkdir -p logs/sweep-status
sleep 3
ssh cyy2 "screen -S sweep-status -X hardcopy /tmp/sweep_status_dump && cat /tmp/sweep_status_dump" > "logs/sweep-status/42-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
cat logs/sweep-status/42-*.log | tail -100
```

### Check status by experiment name

```bash
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -n sweep-all-20260323\n'"
```

### With custom options

```bash
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i 42 --sample-size 50 --stale-threshold 30\n'"
```

## Output Interpretation

The script outputs a table per design dimension showing:
- **Dimension** — the experiment config being varied (e.g., `layer_type`, `global_pooling`)
- **Task** — `graph-clf`, `node-clf`, or `both`
- **Done/Total** — completed runs vs expected
- **Running** — currently active runs
- **Stale** — runs exceeding stale threshold (likely OOM-killed)
- **Failed** — runs that errored

## Notes

- Uses dedicated `sweep-status` screen to avoid interfering with the main `run` session
- Logs persist in `logs/sweep-status/` for later analysis
- The MLflow tracking server on cyy2 runs at `http://127.0.0.1:5001`
- A RUNNING run is considered stale when its age exceeds the stale threshold (default 20 minutes)
