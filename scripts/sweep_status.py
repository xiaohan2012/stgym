#!/usr/bin/env python
"""
Summarize sweep progress by design dimension from an MLflow experiment.

Usage:
    python scripts/sweep_status.py -i <id>
    python scripts/sweep_status.py -n sweep-all-20260323
    python scripts/sweep_status.py -i <id> --sample-size 50
    python scripts/sweep_status.py -i <id> --stale-threshold 30
    python scripts/sweep_status.py -i <id> --conf-dir conf/exp
    python scripts/sweep_status.py -i <id> --max-results 100

A RUNNING run is considered stale (OOM-killed worker that never closed its MLflow
run) when its age exceeds --stale-threshold minutes.
"""

import argparse
import datetime
import glob
import os
import sys
import time
from collections import defaultdict

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

_DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
_DEFAULT_SAMPLE_SIZE = 100
_DEFAULT_STALE_THRESHOLD_MIN = 20
_DEFAULT_CONF_DIR = "conf/exp"


def _get_choices(runs: list) -> str:
    for r in runs:
        c = r.data.tags.get("design_chocies")
        if c:
            return c
    return "?"


def _fmt_task_types(task_types: set) -> str:
    has_graph = any("graph" in t for t in task_types)
    has_node = any("node" in t for t in task_types)
    if has_graph and has_node:
        return "both"
    if has_graph:
        return "graph-clf"
    if has_node:
        return "node-clf"
    return "?"


def load_expected_dims(conf_dir: str) -> dict[str, str]:
    """Return {design_dimension: choices_str} from all conf/exp/*.yaml files."""
    result = {}
    for path in glob.glob(os.path.join(conf_dir, "*.yaml")):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        dim = cfg.get("design_dimension")
        choices = cfg.get("design_choices", [])
        if dim:
            result[dim] = "|".join(str(c) for c in choices)
    return result


def compute_exp_stats(runs: list, now_ms: float) -> dict:
    """Compute experiment-level timing and throughput statistics."""
    terminal = [r for r in runs if r.info.status in ("FINISHED", "FAILED")]
    finished = [r for r in runs if r.info.status == "FINISHED"]
    failed = [r for r in runs if r.info.status == "FAILED"]

    if not runs:
        return {}

    exp_start_ms = min(r.info.start_time for r in runs)
    exp_duration_h = (now_ms - exp_start_ms) / 3_600_000

    def _completed_in_last(hours: float) -> int:
        cutoff_ms = now_ms - hours * 3_600_000
        return len(
            [r for r in terminal if r.info.end_time and r.info.end_time >= cutoff_ms]
        )

    return {
        "exp_start_ms": exp_start_ms,
        "exp_duration_h": exp_duration_h,
        "total_finished": len(finished),
        "total_failed": len(failed),
        "throughput_per_h": len(finished) / exp_duration_h if exp_duration_h > 0 else 0,
        "completed_2h": _completed_in_last(2),
        "completed_12h": _completed_in_last(12),
        "completed_24h": _completed_in_last(24),
    }


def classify_dims(runs: list, stale_threshold_min: float, now_ms: float) -> dict:
    """Group runs by design_dimension and compute per-dimension stats."""
    buckets = defaultdict(
        lambda: {
            "finished": 0,
            "failed": 0,
            "active": 0,
            "stale": 0,
            "durations": [],
            "runs": [],
            "task_types": set(),
            "min_start_ms": float("inf"),
            "max_end_ms": 0,
        }
    )

    for r in runs:
        dim = r.data.tags.get("design_dimension")
        if dim is None:
            continue
        age_min = (now_ms - r.info.start_time) / 60_000
        b = buckets[dim]
        b["runs"].append(r)
        b["min_start_ms"] = min(b["min_start_ms"], r.info.start_time)
        end_ms = r.info.end_time if r.info.end_time else now_ms
        b["max_end_ms"] = max(b["max_end_ms"], end_ms)
        task_type = r.data.tags.get("task_type", "")
        if task_type:
            b["task_types"].add(task_type)
        if r.info.status == "FINISHED":
            b["finished"] += 1
            if r.info.end_time:
                b["durations"].append((r.info.end_time - r.info.start_time) / 60_000)
        elif r.info.status == "FAILED":
            b["failed"] += 1
        elif r.info.status == "RUNNING":
            if age_min > stale_threshold_min:
                b["stale"] += 1
            else:
                b["active"] += 1

    result = {}
    for dim, b in buckets.items():
        choices = _get_choices(b["runs"])
        num_choices = len(choices.split("|")) if choices != "?" else None
        avg_dur = sum(b["durations"]) / len(b["durations"]) if b["durations"] else None
        time_span_min = (b["max_end_ms"] - b["min_start_ms"]) / 60_000
        total_runs = b["finished"] + b["failed"] + b["active"] + b["stale"]
        result[dim] = {
            "state": "STARTED",
            "task_types": b["task_types"],
            "choices": choices,
            "num_choices": num_choices,
            "finished": b["finished"],
            "failed": b["failed"],
            "active": b["active"],
            "stale": b["stale"],
            "total_runs": total_runs,
            "avg_dur_min": avg_dur,
            "cumul_min": sum(b["durations"]),
            "time_span_min": time_span_min,
        }
    return result


def build_sweep_status_df(
    dim_stats: dict, sample_size: int, expected_dims: dict
) -> pd.DataFrame:
    """Build a DataFrame summarising sweep progress per design dimension.

    Rows cover STARTED dims (from MLflow) plus PENDING dims (from expected_dims
    but not yet seen in MLflow).  Columns include run counts by status, timing,
    and the expected number of logical trials per dimension.
    """
    rows = []

    for dim, s in dim_stats.items():
        trials = sample_size * s["num_choices"] if s["num_choices"] else None
        rows.append(
            {
                "dimension": dim,
                "state": s["state"],
                "task": _fmt_task_types(s["task_types"]),
                "choices": s["choices"],
                "trials": trials,
                "total_runs": s["total_runs"],
                "finished": s["finished"],
                "failed": s["failed"],
                "active": s["active"],
                "stale": s["stale"],
                "avg_min": round(s["avg_dur_min"], 1) if s["avg_dur_min"] else None,
                "cumul_min": round(s["cumul_min"], 1),
                "time_span_min": round(s["time_span_min"], 1),
            }
        )

    # Add PENDING rows for dims in conf/exp/ but not yet in MLflow
    seen_dims = set(dim_stats.keys())
    for dim, choices_str in expected_dims.items():
        if dim not in seen_dims:
            num_choices = len(choices_str.split("|")) if choices_str else None
            rows.append(
                {
                    "dimension": dim,
                    "state": "PENDING",
                    "task": "?",
                    "choices": choices_str,
                    "trials": sample_size * num_choices if num_choices else None,
                    "total_runs": 0,
                    "finished": 0,
                    "failed": 0,
                    "active": 0,
                    "stale": 0,
                    "avg_min": None,
                    "cumul_min": None,
                    "time_span_min": None,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["state", "dimension"]).reset_index(drop=True)
    return df


def print_summary(
    exp_name: str,
    exp_id: str,
    df: pd.DataFrame,
    sample_size: int,
    exp_stats: dict,
):
    n_started = (df["state"] == "STARTED").sum()
    n_pending = (df["state"] == "PENDING").sum()
    sep = "=" * 80

    print(f"\nExperiment : {exp_name}")
    print(f"ID         : {exp_id}")
    print(f"Sample size: {sample_size} groups/dimension")
    print(f"Dimensions : {n_started} started, {n_pending} pending")

    if exp_stats:
        start_dt = datetime.datetime.fromtimestamp(
            exp_stats["exp_start_ms"] / 1000, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        print(
            f"\nExp duration: {exp_stats['exp_duration_h']:.1f} h  (started {start_dt})"
        )
        print(
            f"Finished    : {exp_stats['total_finished']} ok"
            f"  /  {exp_stats['total_failed']} failed"
        )
        print(f"Throughput  : ~{exp_stats['throughput_per_h']:.1f} runs/h")
        print(f"\nCompleted last  2 h : {exp_stats['completed_2h']:>5}")
        print(f"Completed last 12 h : {exp_stats['completed_12h']:>5}")
        print(f"Completed last 24 h : {exp_stats['completed_24h']:>5}")

    display_cols = {
        "STARTED": [
            "dimension",
            "task",
            "choices",
            "trials",
            "total_runs",
            "finished",
            "failed",
            "active",
            "stale",
            "avg_min",
            "cumul_min",
            "time_span_min",
        ],
        "PENDING": ["dimension", "choices", "trials"],
    }

    for state in ["STARTED", "PENDING"]:
        sub = df[df["state"] == state]
        if sub.empty:
            continue
        print(f"\n{sep}")
        print(state)
        print(sep)
        print(sub[display_cols[state]].to_string(index=False))

    print(
        "\nNote: trials = sample_size × num_choices (one k-fold CV job = 1 trial).\n"
        "      cumul_min = sum of individual run durations; "
        "time_span_min = wall-clock span (first start to last end)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize sweep progress by design dimension from MLflow"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--experiment-id", type=str)
    group.add_argument("-n", "--experiment-name", type=str)
    parser.add_argument("--tracking-uri", default=_DEFAULT_TRACKING_URI)
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--stale-threshold",
        type=float,
        default=_DEFAULT_STALE_THRESHOLD_MIN,
        metavar="MINUTES",
        help=f"Age in minutes before a RUNNING run is considered stale (default: {_DEFAULT_STALE_THRESHOLD_MIN})",
    )
    parser.add_argument(
        "--conf-dir",
        default=_DEFAULT_CONF_DIR,
        help=f"Directory with exp YAML configs for PENDING inference (default: {_DEFAULT_CONF_DIR})",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5000,
        help="Max MLflow runs to fetch (default: 5000; set to 0 for unlimited)",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    exp = (
        client.get_experiment(args.experiment_id)
        if args.experiment_id
        else client.get_experiment_by_name(args.experiment_name)
    )
    if exp is None:
        print("Experiment not found.")
        return 1

    max_results = args.max_results if args.max_results > 0 else None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id], max_results=max_results
    )
    print(f"Fetched {len(runs)} runs from '{exp.name}'.")
    if args.max_results > 0 and len(runs) == args.max_results:
        print(
            f"WARNING: results may be truncated at {args.max_results} runs. "
            "Increase --max-results to fetch more."
        )

    # Derive server-side "now" from the latest known run timestamp to avoid
    # clock skew between the local machine and the remote MLflow server.
    _ts = [r.info.start_time for r in runs]
    _ts += [r.info.end_time for r in runs if r.info.end_time]
    now_ms = max(_ts) if _ts else time.time() * 1000

    exp_stats = compute_exp_stats(runs, now_ms)

    dim_stats = classify_dims(runs, args.stale_threshold, now_ms)
    if not dim_stats:
        print("No runs with 'design_dimension' tag found.")
        return 1

    expected_dims = {}
    if os.path.isdir(args.conf_dir):
        expected_dims = load_expected_dims(args.conf_dir)
    else:
        print(
            f"Note: --conf-dir '{args.conf_dir}' not found; skipping PENDING inference."
        )

    df = build_sweep_status_df(dim_stats, args.sample_size, expected_dims)
    print_summary(exp.name, exp.experiment_id, df, args.sample_size, exp_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
