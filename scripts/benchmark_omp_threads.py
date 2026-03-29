#!/usr/bin/env python
"""
Benchmark: effect of OMP_NUM_THREADS on DataLoader batch loading speed.

On multi-socket NUMA systems (e.g. 2-socket Xeon), the default thread count
(= all physical cores) can cause severe cross-socket contention for memory-bound
ops like torch.cat, making DataLoader collation 50-200x slower than optimal.

This script sweeps thread counts and reports per-epoch data loading time
using pre-transformed (cached) graph data.

Usage:
    python scripts/benchmark_omp_threads.py --dataset brca-test --batch-size 4
    python scripts/benchmark_omp_threads.py --dataset brca --batch-size 16
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from stgym.data_loader import build_graph_construction_tag, get_dataset_class

KNN_K = 10
GRAPH_CONST = "knn"


def build_transform():
    return T.Compose(
        [
            T.KNNGraph(k=KNN_K),
            T.ToSparseTensor(remove_edge_index=False, layout=torch.sparse_coo),
        ]
    )


def run_epoch(loader, device: str) -> float:
    """Run one epoch through the DataLoader. Returns total time in seconds."""
    t0 = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark effect of OMP_NUM_THREADS on DataLoader speed"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. brca-test, brca, inflammatory-skin)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--threads",
        type=str,
        default=None,
        help="Comma-separated thread counts to test (default: 1,2,4,8,16,32 up to max)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_name = args.dataset.lower()
    is_test = ds_name.endswith("-test")
    root = f"./tests/data/{ds_name}" if is_test else f"./data/{ds_name}"

    tag = build_graph_construction_tag(GRAPH_CONST, KNN_K, None)
    ds_cls = get_dataset_class(ds_name)
    pre_filter = (lambda g: g.num_nodes <= 500) if is_test else None

    max_threads = torch.get_num_threads()

    if args.threads:
        thread_counts = sorted(int(t) for t in args.threads.split(","))
    else:
        candidates = [1, 2, 4, 8, 16, 32]
        thread_counts = [t for t in candidates if t <= max_threads]

    print(
        f"Dataset: {ds_name} | {tag} | device={device} | batch_size={args.batch_size}"
    )
    print(f"Default num_threads: {max_threads}")
    print(f"Thread counts to test: {thread_counts}")
    print(f"Epochs per thread count: {args.epochs}")

    t0 = time.perf_counter()
    ds = ds_cls(
        root=root,
        pre_transform=build_transform(),
        pre_filter=pre_filter,
        graph_construction_tag=tag,
    )
    t1 = time.perf_counter()
    print(f"\n{len(ds)} graphs, loaded in {t1 - t0:.2f}s\n")

    col_w = 14
    print(
        f"{'threads':>8}  {'mean (s)':>{col_w}}  {'std (s)':>{col_w}}  {'speedup':>{col_w}}"
    )
    print(f"{'-' * 8}  {'-' * col_w}  {'-' * col_w}  {'-' * col_w}")

    results = {}

    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)

        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        times = []
        for epoch in range(args.epochs):
            t = run_epoch(loader, device)
            times.append(t)

        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        results[n_threads] = mean_t

        speedup_str = ""
        if max_threads in results and n_threads != max_threads:
            speedup = results[max_threads] / mean_t
            speedup_str = f"{speedup:.1f}x"

        print(
            f"{n_threads:>8}  {mean_t:>{col_w}.4f}  {std_t:>{col_w}.4f}  {speedup_str:>{col_w}}"
        )

    best_threads = min(results, key=results.get)
    worst_threads = max(results, key=results.get)
    print(f"\nFastest: {best_threads} threads ({results[best_threads]:.4f}s)")
    print(f"Slowest: {worst_threads} threads ({results[worst_threads]:.4f}s)")
    if results[worst_threads] > 0:
        ratio = results[worst_threads] / results[best_threads]
        print(f"Ratio (slowest/fastest): {ratio:.1f}x")

    torch.set_num_threads(max_threads)


if __name__ == "__main__":
    main()
