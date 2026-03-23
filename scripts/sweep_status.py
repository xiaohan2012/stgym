#!/usr/bin/env python
"""
Summarize sweep progress by design dimension from an MLflow experiment.

Usage:
    python scripts/sweep_status.py --experiment-id <id>
    python scripts/sweep_status.py --experiment-name sweep-all-20260323
    python scripts/sweep_status.py --experiment-id <id> --sample-size 50
    python scripts/sweep_status.py --experiment-id <id> --stale-threshold 30

A RUNNING run is considered stale (OOM-killed worker that never closed its MLflow
run) when its age exceeds --stale-threshold minutes.
"""

import argparse
import sys
import time
from collections import defaultdict

import mlflow
from mlflow.tracking import MlflowClient

_DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
_DEFAULT_SAMPLE_SIZE = 100
_DEFAULT_STALE_THRESHOLD_MIN = 20


def _get_choices(runs: list) -> str:
    for r in runs:
        c = r.data.tags.get("design_chocies")
        if c:
            return c
    return "?"


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
        }
    )

    for r in runs:
        dim = r.data.tags.get("design_dimension")
        if dim is None:
            continue
        age_min = (now_ms - r.info.start_time) / 60_000
        b = buckets[dim]
        b["runs"].append(r)
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
        avg_dur = sum(b["durations"]) / len(b["durations"]) if b["durations"] else None
        result[dim] = {
            "state": "IN PROGRESS" if b["active"] > 0 else "DONE",
            "choices": _get_choices(b["runs"]),
            "finished": b["finished"],
            "failed": b["failed"],
            "active": b["active"],
            "stale": b["stale"],
            "done": b["finished"] + b["failed"],
            "avg_dur_min": avg_dur,
        }
    return result


def print_summary(exp_name: str, exp_id: str, dim_stats: dict, sample_size: int):
    done = {d: s for d, s in dim_stats.items() if s["state"] == "DONE"}
    in_progress = {d: s for d, s in dim_stats.items() if s["state"] == "IN PROGRESS"}
    sep = "=" * 70

    print(f"\nExperiment : {exp_name}")
    print(f"ID         : {exp_id}")
    print(f"Sample size: {sample_size}")
    print(f"Dimensions : {len(done)} done, {len(in_progress)} in progress")

    if done:
        print(f"\n{sep}")
        print("DONE")
        print(sep)
        for dim, s in sorted(done.items()):
            dur = f"avg {s['avg_dur_min']:.1f}m/trial" if s["avg_dur_min"] else ""
            stale = f"  ({s['stale']} stale)" if s["stale"] else ""
            fail = f"  {s['failed']} failed" if s["failed"] else ""
            print(
                f"  {dim:<38} [{s['choices']}]  {s['finished']} ok{fail}  {dur}{stale}"
            )

    if in_progress:
        print(f"\n{sep}")
        print("IN PROGRESS")
        print(sep)
        for dim, s in sorted(in_progress.items()):
            stale = f"  ({s['stale']} stale)" if s["stale"] else ""
            fail = f"  {s['failed']} failed" if s["failed"] else ""
            print(
                f"  {dim:<38} [{s['choices']}]"
                f"  {s['done']}/{sample_size} done  {s['active']} active{fail}{stale}"
            )

    print(
        "\nNote: PENDING dimensions (not yet started) have no runs in MLflow and are not shown here."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize sweep progress by design dimension from MLflow"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-id", type=str)
    group.add_argument("--experiment-name", type=str)
    parser.add_argument("--tracking-uri", default=_DEFAULT_TRACKING_URI)
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--stale-threshold",
        type=float,
        default=_DEFAULT_STALE_THRESHOLD_MIN,
        metavar="MINUTES",
        help=f"Age in minutes before a RUNNING run is considered stale (default: {_DEFAULT_STALE_THRESHOLD_MIN})",
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

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5000)
    print(f"Fetched {len(runs)} runs from '{exp.name}'.")

    dim_stats = classify_dims(runs, args.stale_threshold)
    if not dim_stats:
        print("No runs with 'design_dimension' tag found.")
        return 1

    print_summary(exp.name, exp.experiment_id, dim_stats, args.sample_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
