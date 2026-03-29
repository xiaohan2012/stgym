#!/usr/bin/env python
"""
Analyze job concurrency for an MLflow experiment and produce a Gantt chart.

Usage:
    python scripts/analyze_concurrency.py --experiment-id 2
    python scripts/analyze_concurrency.py --experiment-id 2 --mlflow-uri http://127.0.0.1:5001
    python scripts/analyze_concurrency.py --experiment-id 2 --output /tmp/concurrency.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mlflow import MlflowClient


def fetch_run_intervals(
    client: MlflowClient, experiment_id: str
) -> list[tuple[int, int]]:
    """Fetch (start_time, end_time) for all finished runs."""
    runs = client.search_runs(
        experiment_id, order_by=["start_time ASC"], max_results=5000
    )
    intervals = []
    for r in runs:
        if r.info.end_time and r.info.start_time:
            intervals.append((r.info.start_time, r.info.end_time))
    return intervals


def compute_concurrency(
    intervals: list[tuple[int, int]],
) -> tuple[list[int], list[int], dict[int, int]]:
    """Sweep-line algorithm to compute concurrency over time.

    Returns (times, concurrency_values, concurrency_histogram).
    """
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()

    times, concurrency = [], []
    hist: dict[int, int] = {}
    current = 0
    prev_time = events[0][0]

    for t, delta in events:
        # Step function for plotting
        times.append(t)
        concurrency.append(current)
        # Update histogram
        if t > prev_time:
            duration = t - prev_time
            hist[current] = hist.get(current, 0) + duration
        current += delta
        times.append(t)
        concurrency.append(current)
        prev_time = t

    return times, concurrency, hist


def print_stats(
    intervals: list[tuple[int, int]], hist: dict[int, int], max_concurrent: int
):
    """Print concurrency statistics."""
    total_wall = intervals[-1][1] - intervals[0][0]
    weighted_sum = sum(level * ms for level, ms in hist.items())
    avg = weighted_sum / total_wall

    print(f"Total runs: {len(intervals)}")
    print(f"Total wall time: {total_wall / 1000:.1f}s")
    print(f"Max concurrency: {max_concurrent}")
    print(f"Average concurrency: {avg:.1f}")
    print()
    print("Concurrency distribution:")
    for level in sorted(hist.keys()):
        ms = hist[level]
        pct = ms / total_wall * 100
        print(f"  {level:2d} concurrent: {ms / 1000:7.1f}s ({pct:5.1f}%)")


def plot_gantt(
    intervals: list[tuple[int, int]],
    times: list[int],
    concurrency: list[int],
    max_gpus: int,
    output: str,
):
    """Generate Gantt chart and concurrency plot."""
    # Normalize to seconds from first start
    t0 = intervals[0][0]
    intervals_s = [((s - t0) / 1000, (e - t0) / 1000) for s, e in intervals]
    times_s = [(t - t0) / 1000 for t in times]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Greedy lane assignment for Gantt chart
    lane_end: dict[int, float] = {}
    assignments = []
    for s, e in intervals_s:
        assigned = False
        for lid in sorted(lane_end.keys()):
            if lane_end[lid] <= s:
                lane_end[lid] = e
                assignments.append((lid, s, e))
                assigned = True
                break
        if not assigned:
            lid = len(lane_end)
            lane_end[lid] = e
            assignments.append((lid, s, e))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for lid, s, e in assignments:
        ax1.barh(
            lid,
            e - s,
            left=s,
            height=0.7,
            color=colors[lid % 10],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )

    ax1.set_ylabel("Worker Slot (virtual)")
    ax1.set_title(f"Job Execution Gantt Chart ({len(intervals)} runs)")
    ax1.set_yticks(range(max(a[0] for a in assignments) + 1))
    ax1.invert_yaxis()

    # Concurrency over time
    ax2.fill_between(times_s, concurrency, alpha=0.5, color="steelblue")
    ax2.plot(times_s, concurrency, color="steelblue", linewidth=0.8)
    ax2.axhline(
        y=max_gpus,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Max GPUs ({max_gpus})",
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Concurrent Jobs")
    ax2.set_ylim(0, max_gpus + 2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze job concurrency for an MLflow experiment"
    )
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment ID")
    parser.add_argument(
        "--mlflow-uri", default="http://127.0.0.1:5001", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--max-gpus", type=int, default=6, help="Max GPU count for reference line"
    )
    parser.add_argument(
        "--output", default="/tmp/concurrency_gantt.png", help="Output image path"
    )
    args = parser.parse_args()

    client = MlflowClient(args.mlflow_uri)
    intervals = fetch_run_intervals(client, args.experiment_id)

    if not intervals:
        print("No runs found.")
        return

    times, concurrency, hist = compute_concurrency(intervals)
    max_concurrent = max(concurrency)

    print_stats(intervals, hist, max_concurrent)
    plot_gantt(intervals, times, concurrency, args.max_gpus, args.output)


if __name__ == "__main__":
    main()
