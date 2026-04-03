#!/usr/bin/env python
"""
Level 2 stress test: verify that a hard CPU concurrency limit prevents OOM kills
when loading the mouse-kidney dataset.

Usage:
    # Baseline — no constraint, all workers compete for RAM (expect kills)
    python scripts/stress_test_mouse_kidney.py --n-workers 6

    # With fix — hard CPU limit caps concurrency to floor(n_workers / cpus_per_worker)
    python scripts/stress_test_mouse_kidney.py --n-workers 6 --cpus-per-worker 3
"""

import argparse
import sys

import ray

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader import STDataModule


@ray.remote
def load_dataset(knn_k: int) -> bool:
    """Load mouse-kidney dataset only — no training."""

    task = TaskConfig(
        dataset_name="mouse-kidney",
        type="graph-classification",
        num_classes=2,
    )
    dl = DataLoaderConfig(graph_const="knn", knn_k=knn_k, batch_size=8)
    STDataModule(task, dl)
    return True


def main():
    parser = argparse.ArgumentParser(description="mouse-kidney concurrency stress test")
    parser.add_argument(
        "--n-workers", type=int, default=6, help="Number of concurrent Ray workers"
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=None,
        help="CPUs to reserve per worker (hard constraint); "
        "max concurrency = floor(n_workers / cpus_per_worker)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Ray max_retries per worker (0 = no retry, gives clean first-pass kill count)",
    )
    parser.add_argument(
        "--knn-k", type=int, default=20, help="KNN k for graph construction"
    )
    args = parser.parse_args()

    if args.cpus_per_worker is not None:
        max_concurrent = args.n_workers // args.cpus_per_worker
        mode = (
            f"cpus_per_worker={args.cpus_per_worker} (max {max_concurrent} concurrent)"
        )
    else:
        mode = "no constraint"

    print(f"\n=== mouse-kidney stress test ===")
    print(f"Workers     : {args.n_workers}")
    print(f"Mode        : {mode}")
    print(f"Max retries : {args.max_retries}")
    print(f"knn_k       : {args.knn_k}")
    print()

    ray.init(num_cpus=args.n_workers, num_gpus=0)

    opts = {}
    if args.cpus_per_worker is not None:
        opts["num_cpus"] = args.cpus_per_worker

    futures = [
        load_dataset.options(max_retries=args.max_retries, **opts).remote(
            knn_k=args.knn_k
        )
        for _ in range(args.n_workers)
    ]

    survived, killed = 0, 0
    errors = []
    for i, f in enumerate(futures):
        try:
            ray.get(f)
            survived += 1
            print(f"  worker {i + 1}: OK")
        except Exception as e:
            killed += 1
            errors.append(str(e)[:120])
            print(f"  worker {i + 1}: KILLED — {type(e).__name__}")

    ray.shutdown()

    print()
    print(
        f"Result  : {survived}/{args.n_workers} survived, {killed}/{args.n_workers} killed"
    )
    print()

    if killed > 0:
        print("Errors:")
        for err in errors:
            print(f"  {err}")

    sys.exit(0 if killed == 0 else 1)


if __name__ == "__main__":
    main()
