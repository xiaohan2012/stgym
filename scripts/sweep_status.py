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


def classify_dims(runs: list, stale_threshold_min: float) -> dict:
    """Group runs by design_dimension and compute per-dimension stats."""
    now_ms = time.time() * 1000
    buckets = defaultdict(
        lambda: {
            "finished": 0,
            "failed": 0,
            "active": 0,
            "stale": 0,
            "durations": [],
            "runs": [],
            "task_types": set(),
        }
    )

    for r in runs:
        dim = r.data.tags.get("design_dimension")
        if dim is None:
            continue
        age_min = (now_ms - r.info.start_time) / 60_000
        b = buckets[dim]
        b["runs"].append(r)
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
        result[dim] = {
            "state": "IN PROGRESS" if b["active"] > 0 else "DONE",
            "task_types": b["task_types"],
            "choices": choices,
            "num_choices": num_choices,
            "finished": b["finished"],
            "failed": b["failed"],
            "active": b["active"],
            "stale": b["stale"],
            "done": b["finished"] + b["failed"],
            "avg_dur_min": avg_dur,
            "total_dur_min": sum(b["durations"]),
        }
    return result


def build_sweep_status_df(
    dim_stats: dict, sample_size: int, expected_dims: dict
) -> pd.DataFrame:
    """Build a DataFrame summarising sweep progress per design dimension.

    Rows cover DONE and IN PROGRESS dims (from MLflow) plus PENDING dims
    (from expected_dims but not yet seen in MLflow).  Columns include run
    counts, timing, and the expected total number of runs per dimension.
    """
    rows = []

    for dim, s in dim_stats.items():
        total_expected = sample_size * s["num_choices"] if s["num_choices"] else None
        rows.append(
            {
                "dimension": dim,
                "state": s["state"],
                "task": _fmt_task_types(s["task_types"]),
                "choices": s["choices"],
                "finished": s["finished"],
                "failed": s["failed"],
                "active": s["active"],
                "stale": s["stale"],
                "done": s["done"],
                "total": total_expected,
                "avg_min": round(s["avg_dur_min"], 1) if s["avg_dur_min"] else None,
                "total_min": round(s["total_dur_min"], 1),
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
                    "finished": 0,
                    "failed": 0,
                    "active": 0,
                    "stale": 0,
                    "done": 0,
                    "total": sample_size * num_choices if num_choices else None,
                    "avg_min": None,
                    "total_min": None,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["state", "dimension"]).reset_index(drop=True)
    return df


def print_summary(exp_name: str, exp_id: str, df: pd.DataFrame, sample_size: int):
    n_done = (df["state"] == "DONE").sum()
    n_in_progress = (df["state"] == "IN PROGRESS").sum()
    n_pending = (df["state"] == "PENDING").sum()
    sep = "=" * 80

    print(f"\nExperiment : {exp_name}")
    print(f"ID         : {exp_id}")
    print(f"Sample size: {sample_size} groups/dimension")
    print(
        f"Dimensions : {n_done} done, {n_in_progress} in progress, {n_pending} pending"
    )

    display_cols = {
        "DONE": [
            "dimension",
            "task",
            "choices",
            "finished",
            "failed",
            "stale",
            "avg_min",
            "total_min",
        ],
        "IN PROGRESS": [
            "dimension",
            "task",
            "choices",
            "done",
            "total",
            "active",
            "failed",
            "stale",
        ],
        "PENDING": ["dimension", "choices", "total"],
    }

    for state in ["DONE", "IN PROGRESS", "PENDING"]:
        sub = df[df["state"] == state]
        if sub.empty:
            continue
        print(f"\n{sep}")
        print(state)
        print(sep)
        cols = display_cols[state]
        print(sub[cols].to_string(index=False))

    print(
        "\nNote: total_min = total wall time for all finished runs in that dimension."
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

    dim_stats = classify_dims(runs, args.stale_threshold)
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
    print_summary(exp.name, exp.experiment_id, df, args.sample_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
