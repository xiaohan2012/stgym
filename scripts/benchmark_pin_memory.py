#!/usr/bin/env python
"""
Benchmark: effect of pin_memory on pre_transform data loading speed.

Compares DataLoader with pin_memory=False vs pin_memory=True using
pre-transformed (cached) graph data. Reports per-batch load vs transfer
time breakdown.

Usage:
    python scripts/benchmark_pin_memory.py \
        --dataset brca-test --graph-const knn --knn-k 10 \
        --batch-size 4 --device cpu

    python scripts/benchmark_pin_memory.py \
        --dataset inflammatory-skin --graph-const knn --knn-k 10 \
        --batch-size 16 --device cuda
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

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


def run_epoch_detailed(loader, device: str, verbose: bool = True):
    """Run one epoch. Returns (t_load, t_transfer)."""
    t_load = 0.0
    t_transfer = 0.0
    it = iter(loader)
    n = len(loader)
    i = 0
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        batch = batch.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_load += t1 - t0
        t_transfer += t2 - t1
        i += 1
        if verbose:
            print(
                f"    batch {i}/{n}  load={t1 - t0:.4f}s  xfer={t2 - t1:.4f}s",
                flush=True,
            )
    return t_load, t_transfer


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pin_memory effect on pre_transform data loading"
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
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_name = args.dataset.lower()
    is_test = ds_name.endswith("-test")
    root = f"./tests/data/{ds_name}" if is_test else f"./data/{ds_name}"

    radius = None
    if args.graph_const == "radius":
        ds_info = get_info(ds_name)
        radius = args.radius_ratio * (ds_info["max_span"] - ds_info["min_span"])

    tag = build_graph_construction_tag(args.graph_const, args.knn_k, args.radius_ratio)
    full_transform = build_transform(args.graph_const, args.knn_k, radius)
    ds_cls = get_dataset_class(ds_name)

    pre_filter = (lambda g: g.num_nodes <= 500) if is_test else None

    print(
        f"Dataset: {ds_name} | {tag} | device={device} | batch_size={args.batch_size}"
    )

    t0 = time.perf_counter()
    ds = ds_cls(
        root=root,
        pre_transform=full_transform,
        pre_filter=pre_filter,
        graph_construction_tag=tag,
    )
    t1 = time.perf_counter()
    print(f"{len(ds)} graphs, loaded in {t1 - t0:.2f}s\n")

    loader_no_pin = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    loader_pin = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    all_loads = {"pin=False": [], "pin=True": []}
    all_xfers = {"pin=False": [], "pin=True": []}
    all_totals = {"pin=False": [], "pin=True": []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}:")

        for label, loader in [("pin=False", loader_no_pin), ("pin=True", loader_pin)]:
            print(f"\n  {label}:")
            t_load, t_xfer = run_epoch_detailed(loader, device)
            t_total = t_load + t_xfer
            all_loads[label].append(t_load)
            all_xfers[label].append(t_xfer)
            all_totals[label].append(t_total)
            print(
                f"    total: load={t_load:.4f}s  xfer={t_xfer:.4f}s  epoch={t_total:.4f}s"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary ({args.epochs} epoch(s)):\n")

    col = 12
    print(f"  {'':>15} {'pin=False':>{col}} {'pin=True':>{col}} {'Diff':>{col}}")
    print(f"  {'':>15} {'-' * col} {'-' * col} {'-' * col}")

    for metric, data in [
        ("load (s)", all_loads),
        ("transfer (s)", all_xfers),
        ("total (s)", all_totals),
    ]:
        mean_no = statistics.mean(data["pin=False"])
        mean_yes = statistics.mean(data["pin=True"])
        diff_pct = ((mean_yes - mean_no) / mean_no * 100) if mean_no > 0 else 0
        sign = "+" if diff_pct >= 0 else ""
        print(
            f"  {metric:>15} {mean_no:>{col}.4f} {mean_yes:>{col}.4f} {sign}{diff_pct:>{col - 1}.1f}%"
        )


if __name__ == "__main__":
    main()
