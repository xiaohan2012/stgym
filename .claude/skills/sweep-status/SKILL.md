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

### Step 1: Submit command (do this first)

The script may take a minute or more to run. Just submit the command and let it run.

```bash
# Ensure screen exists, then send command
ssh cyy2 "screen -ls | grep sweep-status" || ssh cyy2 "screen -dmS sweep-status bash -c 'cd /root/stgym && source .venv/bin/activate && exec bash'"

# Submit the command (returns immediately)
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i <experiment-id> [options]\n'"
```

### Step 2: Check output (do this later)

After the script finishes (typically 30-60 seconds), capture and save the output.

```bash
# Capture screen output to local log
mkdir -p logs/sweep-status
ssh cyy2 "screen -S sweep-status -X hardcopy /tmp/sweep_status_dump && cat /tmp/sweep_status_dump" > "logs/sweep-status/<experiment-id>-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"

# Display the output
cat logs/sweep-status/<experiment-id>-*.log | tail -100
```

## Examples

### Submit status check by experiment ID

```bash
ssh cyy2 "screen -ls | grep sweep-status" || ssh cyy2 "screen -dmS sweep-status bash -c 'cd /root/stgym && source .venv/bin/activate && exec bash'"
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i 42\n'"
```

### Submit status check by experiment name

```bash
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -n sweep-all-20260323\n'"
```

### Submit with custom options

```bash
ssh cyy2 "screen -S sweep-status -X stuff 'python scripts/sweep_status.py -i 42 --sample-size 50 --stale-threshold 30\n'"
```

### Capture output later

```bash
mkdir -p logs/sweep-status
ssh cyy2 "screen -S sweep-status -X hardcopy /tmp/sweep_status_dump && cat /tmp/sweep_status_dump" > "logs/sweep-status/42-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
cat logs/sweep-status/42-*.log | tail -100
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
