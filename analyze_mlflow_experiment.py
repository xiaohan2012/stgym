#!/usr/bin/env python
"""
Script to analyze MLFlow experiment results.
Calculates timing statistics for all runs in a given experiment.
"""

import argparse
import sys
from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient


def analyze_experiment(
    experiment_name: str, mlflow_uri: str = "http://127.0.0.1:5001"
) -> Optional[dict]:
    """
    Analyze timing statistics for all runs in an MLFlow experiment.

    Args:
        experiment_name: Name of the experiment to analyze
        mlflow_uri: MLFlow tracking server URI

    Returns:
        Dictionary with analysis results or None if experiment not found
    """
    # Connect to MLFlow server
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found")
            return None

        # Get all runs in the experiment
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        if not runs:
            print(f"❌ No runs found for experiment '{experiment_name}'")
            return None

        print(f"📊 Found {len(runs)} runs for experiment: {experiment_name}")

        # Extract timing data
        start_times = []
        end_times = []
        durations = []
        completed_runs = 0

        for run in runs:
            start_time = run.info.start_time / 1000  # Convert from ms to seconds
            end_time = run.info.end_time / 1000 if run.info.end_time else None

            start_times.append(start_time)
            if end_time:
                end_times.append(end_time)
                durations.append(end_time - start_time)
                completed_runs += 1

        if not end_times:
            print("❌ No completed runs found (all runs may still be running)")
            return None

        # Calculate metrics
        min_start = min(start_times)
        max_end = max(end_times)
        total_time = max_end - min_start
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        # Convert timestamps to readable format
        min_start_dt = datetime.fromtimestamp(min_start)
        max_end_dt = datetime.fromtimestamp(max_end)

        results = {
            "experiment_name": experiment_name,
            "total_runs": len(runs),
            "completed_runs": completed_runs,
            "min_start_time": min_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "max_end_time": max_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "total_running_time_seconds": total_time,
            "total_running_time_minutes": total_time / 60,
            "total_running_time_hours": total_time / 3600,
            "mean_duration_seconds": mean_duration,
            "mean_duration_minutes": mean_duration / 60,
            "std_duration_seconds": std_duration,
            "std_duration_minutes": std_duration / 60,
            "durations": durations,  # Individual run durations
        }

        if completed_runs < len(runs):
            print(
                f"⚠️  Warning: {len(runs) - completed_runs} runs are still running or failed"
            )

        return results

    except Exception as e:
        print(f"❌ Error querying experiment '{experiment_name}': {e}")
        return None


def print_results(results: dict):
    """Print analysis results in a formatted way."""
    print("\n" + "=" * 80)
    print("🔍 EXPERIMENT ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\n📋 Experiment: {results['experiment_name']}")
    print(f"   Total runs: {results['total_runs']}")
    print(f"   Completed runs: {results['completed_runs']}")

    print(f"\n⏰ Timing Summary:")
    print(f"   Start time (earliest): {results['min_start_time']}")
    print(f"   End time (latest): {results['max_end_time']}")
    print(f"   Total running time: {results['total_running_time_hours']:.2f} hours")
    print(
        f"                       ({results['total_running_time_minutes']:.2f} minutes)"
    )
    print(
        f"                       ({results['total_running_time_seconds']:.1f} seconds)"
    )

    print(f"\n📈 Run Duration Statistics:")
    print(
        f"   Mean duration: {results['mean_duration_minutes']:.2f} ± {results['std_duration_minutes']:.2f} minutes"
    )
    print(
        f"   Mean duration: {results['mean_duration_seconds']:.1f} ± {results['std_duration_seconds']:.1f} seconds"
    )

    # Show individual run durations
    durations_min = [d / 60 for d in results["durations"]]
    print(
        f"\n📊 Individual run durations (minutes): {[f'{d:.2f}' for d in durations_min]}"
    )
    print(f"   Min duration: {min(durations_min):.2f} minutes")
    print(f"   Max duration: {max(durations_min):.2f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MLFlow experiment timing statistics"
    )
    parser.add_argument(
        "experiment_name", type=str, help="Name of the experiment to analyze"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://127.0.0.1:5001",
        help="MLFlow tracking URI (default: http://127.0.0.1:5001)",
    )

    args = parser.parse_args()

    print(f"🔍 Analyzing experiment: {args.experiment_name}")
    print(f"📡 MLFlow server: {args.mlflow_uri}")

    results = analyze_experiment(args.experiment_name, args.mlflow_uri)

    if results:
        print_results(results)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
