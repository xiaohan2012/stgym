#!/usr/bin/env python
"""
Benchmark: pre_transform (CPU-cached graph) vs transform (on-the-fly CPU graph).

Approach A (pre_transform):
  - Graph pre-built once, cached to disk (e.g. data_knn10.pt)
  - Each epoch: read pre-built graph → move to device

Approach B (transform):
  - Raw data cached to disk (data.pt, no edges)
  - Each epoch: read raw data → apply KNN/radius transform on CPU (per __getitem__) → move to device

Usage:
    python scripts/benchmark_transform_vs_pretransform.py \
        --dataset brca-test --graph-const knn --knn-k 10 \
        --batch-size 4 --epochs 3 --device cpu

    python scripts/benchmark_transform_vs_pretransform.py \
        --dataset inflammatory-skin --graph-const knn --knn-k 10 \
        --batch-size 16 --epochs 5 --device cuda
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
from stgym.data_loader.ds_info import get_info


def build_transform(graph_const: str, knn_k: int | None, radius: float | None):
    graph_t = T.KNNGraph(k=knn_k) if graph_const == "knn" else T.RadiusGraph(r=radius)
    return T.Compose(
        [graph_t, T.ToSparseTensor(remove_edge_index=False, layout=torch.sparse_coo)]
    )


def tensor_bytes(t: torch.Tensor) -> int:
    if t.is_sparse:
        return t.indices().nbytes + t.values().nbytes
    return t.nbytes


def batch_size_mb(batch) -> float:
    total = 0
    for attr in batch.keys():
        val = batch[attr]
        if isinstance(val, torch.Tensor):
            total += tensor_bytes(val)
    return total / (1024**2)


def run_epoch(
    loader,
    device: str,
    label: str = "",
    verbose: bool = False,
    measure_mb: bool = False,
) -> tuple[float, float | None]:
    """Run one epoch. Returns (total_time, mean_batch_mb or None)."""
    t0 = time.perf_counter()
    n_batches = len(loader)
    mb_samples = []
    for i, batch in enumerate(loader):
        if measure_mb:
            mb_samples.append(batch_size_mb(batch))
        batch = batch.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"    {label} batch {i + 1}/{n_batches}  elapsed={elapsed:.2f}s",
                flush=True,
            )
    total = time.perf_counter() - t0
    mean_mb = statistics.mean(mb_samples) if mb_samples else None
    return total, mean_mb


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pre_transform vs transform data loading"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. inflammatory-skin, brca-test)",
    )
    parser.add_argument("--graph-const", default="knn", choices=["knn", "radius"])
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--radius-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--warmup", type=int, default=0, help="Warmup epochs (excluded from timing)"
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_name = args.dataset.lower()
    is_test = ds_name.endswith("-test")
    root = f"./tests/data/{ds_name}" if is_test else f"./data/{ds_name}"

    # Compute radius if needed
    radius = None
    if args.graph_const == "radius":
        ds_info = get_info(ds_name)
        radius = args.radius_ratio * (ds_info["max_span"] - ds_info["min_span"])

    tag = build_graph_construction_tag(args.graph_const, args.knn_k, args.radius_ratio)
    full_transform = build_transform(args.graph_const, args.knn_k, radius)
    ds_cls = get_dataset_class(ds_name)

    pre_filter = (lambda g: g.num_nodes <= 500) if is_test else None

    print(
        f"Loading datasets for: {ds_name} | {tag} | device={device} | batch_size={args.batch_size}"
    )
    print()

    # Dataset B: raw data on disk (data.pt), transform applied per __getitem__
    # Load B first to test page cache effect on load times
    print("  B: transform     → data.pt        ...", flush=True)
    t0 = time.perf_counter()
    print("    [B] calling ds_cls()...", flush=True)
    ds_b = ds_cls(
        root=root,
        transform=full_transform,
        pre_filter=pre_filter,
        # no graph_construction_tag → data.pt
    )
    t1 = time.perf_counter()
    print(
        f"    [B] ds_cls() done in {t1 - t0:.2f}s, len={len(ds_b)} graphs", flush=True
    )

    # Dataset A: graph pre-built on disk (data_knn10.pt)
    print(f"  A: pre_transform → data_{tag}.pt  ...", flush=True)
    print("    [A] calling ds_cls()...", flush=True)
    t0 = time.perf_counter()
    ds_a = ds_cls(
        root=root,
        pre_transform=full_transform,
        pre_filter=pre_filter,
        graph_construction_tag=tag,
    )
    t1 = time.perf_counter()
    print(
        f"    [A] ds_cls() done in {t1 - t0:.2f}s, len={len(ds_a)} graphs", flush=True
    )

    loader_a = DataLoader(
        ds_a, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    loader_b = DataLoader(
        ds_b, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Warmup (also measures batch size on first warmup epoch)
    mb_a = mb_b = None
    if args.warmup > 0:
        print(f"\nWarming up ({args.warmup} epoch(s))...")
        for w in range(args.warmup):
            measure = w == 0
            _, mb_a_w = run_epoch(
                loader_a, device, label="A warmup", verbose=True, measure_mb=measure
            )
            _, mb_b_w = run_epoch(
                loader_b, device, label="B warmup", verbose=True, measure_mb=measure
            )
            if measure:
                mb_a, mb_b = mb_a_w, mb_b_w

    if mb_a is not None:
        print("\nMean batch size:")
        print(f"  A (pre_transform, data_{tag}.pt): {mb_a:.3f} MB")
        print(f"  B (transform,     data.pt):       {mb_b:.3f} MB")

    # Timed runs
    times_a = []
    times_b = []

    col_w = 20
    print(f"\nEpoch times (s), {args.epochs} epochs:")
    print(
        f"  {'Epoch':>5}  {'A (pre_transform)':>{col_w}}  {'B (transform, CPU)':>{col_w}}"
    )
    print(f"  {'-' * 5}  {'-' * col_w}  {'-' * col_w}")

    for epoch in range(1, args.epochs + 1):
        print(f"  Epoch {epoch}:")
        t_a, _ = run_epoch(loader_a, device, label="A", verbose=True)
        t_b, _ = run_epoch(loader_b, device, label="B", verbose=True)
        times_a.append(t_a)
        times_b.append(t_b)
        print(f"  {epoch:>5}  {t_a:>{col_w}.3f}  {t_b:>{col_w}.3f}")

    mean_a = statistics.mean(times_a)
    mean_b = statistics.mean(times_b)
    std_a = statistics.stdev(times_a) if len(times_a) > 1 else 0.0
    std_b = statistics.stdev(times_b) if len(times_b) > 1 else 0.0

    print(f"  {'-' * 5}  {'-' * col_w}  {'-' * col_w}")
    print(
        f"  {'Mean':>5}  {f'{mean_a:.3f} ± {std_a:.3f}':>{col_w}}  {f'{mean_b:.3f} ± {std_b:.3f}':>{col_w}}"
    )

    speedup = mean_b / mean_a if mean_a > 0 else float("inf")
    winner = "A (pre_transform)" if speedup > 1 else "B (transform)"
    print(f"\nSpeedup B/A: {speedup:.2f}x  →  {winner} is faster")


if __name__ == "__main__":
    main()
