#!/usr/bin/env python
"""
Level 2 stress test for PR #136: verify that the Ray memory constraint
prevents concurrent OOM kills when loading the mouse-kidney dataset.

Usage:
    # Baseline — no constraint, workers compete for RAM (expect kills)
    python scripts/stress_test_mouse_kidney.py --n-workers 6

    # With fix — Ray limits concurrency to floor(RAM / memory_gb)
    python scripts/stress_test_mouse_kidney.py --n-workers 6 --memory-gb 40
"""

import argparse
import sys

import ray


@ray.remote
def load_dataset(knn_k: int) -> bool:
    """Load mouse-kidney dataset only — no training."""
    from stgym.config_schema import DataLoaderConfig, TaskConfig
    from stgym.data_loader import STDataModule

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
        "--memory-gb",
        type=float,
        default=None,
        help="Memory reservation per worker in GB (omit to run without constraint)",
    )
    parser.add_argument(
        "--knn-k", type=int, default=20, help="KNN k for graph construction"
    )
    args = parser.parse_args()

    mode = f"memory={args.memory_gb} GB/worker" if args.memory_gb else "no constraint"
    print(f"\n=== mouse-kidney stress test ===")
    print(f"Workers : {args.n_workers}")
    print(f"Mode    : {mode}")
    print(f"knn_k   : {args.knn_k}")
    print()

    ray.init(num_cpus=args.n_workers)

    opts = {}
    if args.memory_gb is not None:
        opts["memory"] = int(args.memory_gb * 1024**3)

    futures = [
        load_dataset.options(**opts).remote(knn_k=args.knn_k)
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
