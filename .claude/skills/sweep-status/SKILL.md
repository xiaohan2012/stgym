---
name: sweep-status
description: Check sweep progress on cyy2 GPU server. Runs sweep_status.py remotely and saves output to local log files. Use when checking experiment sweep status, monitoring running sweeps, or diagnosing stale runs.
---

# sweep-status

Check sweep progress on the cyy2 GPU server with persistent logging.

## Parameters

- **Experiment ID** (`-i <id>`) — MLflow experiment ID (required, unless using `-n`)
- **Experiment Name** (`-n <name>`) — MLflow experiment name (alternative to `-i`)
- **Sample Size** (`--sample-size <n>`) — Number of runs to sample per dimension (default: 100)
- **Stale Threshold** (`--stale-threshold <min>`) — Minutes after which a RUNNING run is considered stale/OOM-killed (default: 20)

## Workflow

Run the script and capture output in a single command:

```bash
mkdir -p logs/sweep-status
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && python scripts/sweep_status.py -i <experiment-id> [options]" | tee "logs/sweep-status/<experiment-id>-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
```

## Examples

### Check by experiment ID

```bash
mkdir -p logs/sweep-status
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && python scripts/sweep_status.py -i 42" | tee "logs/sweep-status/42-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
```

### Check by experiment name

```bash
mkdir -p logs/sweep-status
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && python scripts/sweep_status.py -n sweep-all-20260323" | tee "logs/sweep-status/sweep-all-20260323-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
```

### Check with custom options

```bash
mkdir -p logs/sweep-status
ssh cyy2 "cd /root/stgym && source .venv/bin/activate && python scripts/sweep_status.py -i 42 --sample-size 50 --stale-threshold 30" | tee "logs/sweep-status/42-$(date -u +'%Y-%m-%dT%H:%M:%SZ').log"
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

- Logs are saved locally to `logs/sweep-status/` for later analysis
- The MLflow tracking server on cyy2 runs at `http://127.0.0.1:5000`
- A RUNNING run is considered stale when its age exceeds the stale threshold (default 20 minutes)
