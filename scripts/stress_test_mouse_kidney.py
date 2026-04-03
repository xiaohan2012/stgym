#!/usr/bin/env python
"""
Level 2 stress test: verify that DatasetLoadGate prevents OOM kills
when loading the mouse-kidney dataset concurrently.

Usage:
    # Baseline — no gate, all workers load simultaneously (expect OOM kills)
    python scripts/stress_test_mouse_kidney.py --n-workers 6

    # With gate — workers serialize through DatasetLoadGate (expect all survive)
    python scripts/stress_test_mouse_kidney.py --n-workers 6 --use-gate
"""

import argparse
import sys

import ray

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader import STDataModule, get_info
from stgym.data_loader.const import DatasetName
from stgym.utils import DatasetLoadGate, gated_load

DATASET_NAME = DatasetName.mouse_kidney


@ray.remote
def load_dataset(knn_k: int, gated_datasets: frozenset) -> bool:
    """Load mouse-kidney dataset only — no training."""
    info = get_info(DATASET_NAME)
    task = TaskConfig(
        dataset_name=DATASET_NAME,
        type=info["task_type"],
        num_classes=info["num_classes"],
    )
    dl = DataLoaderConfig(graph_const="knn", knn_k=knn_k, batch_size=8)
    with gated_load(DATASET_NAME, gated_datasets):
        STDataModule(task, dl)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="mouse-kidney DatasetLoadGate stress test"
    )
    parser.add_argument(
        "--n-workers", type=int, default=6, help="Number of concurrent Ray workers"
    )
    parser.add_argument(
        "--use-gate",
        action="store_true",
        help="Serialize loads via DatasetLoadGate (the fix being tested)",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=1,
        help="CPUs reserved per worker (controls Ray parallelism budget)",
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

    mode = "gate ON (serialized loads)" if args.use_gate else "gate OFF (baseline)"

    print(f"\n=== mouse-kidney stress test ===")
    print(f"Workers        : {args.n_workers}")
    print(f"Mode           : {mode}")
    print(f"CPUs per worker: {args.cpus_per_worker}")
    print(f"Max retries    : {args.max_retries}")
    print(f"knn_k          : {args.knn_k}")
    print()

    ray.init(num_cpus=args.n_workers * args.cpus_per_worker, num_gpus=0)

    gated_datasets = frozenset([DATASET_NAME]) if args.use_gate else frozenset()
    gate_actor = None
    if args.use_gate:
        gate_actor = DatasetLoadGate.options(name="dataset_load_gate").remote(
            max_concurrent=1
        )

    futures = [
        load_dataset.options(
            num_cpus=args.cpus_per_worker, max_retries=args.max_retries
        ).remote(knn_k=args.knn_k, gated_datasets=gated_datasets)
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
