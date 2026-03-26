#!/usr/bin/env python
"""
Pre-generate processed dataset files for all (dataset, graph_const, knn_k/radius_ratio)
combinations defined in a design space YAML.

Run this before launching a sweep to avoid race conditions where parallel Ray workers
all call process() simultaneously on the same file.

Usage:
    python scripts/preprocess_datasets.py --design-space conf/design_space/graph_clf.yaml
    python scripts/preprocess_datasets.py --design-space conf/design_space/graph_clf.yaml --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch_geometric.transforms as T
import yaml

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stgym.data_loader import build_graph_construction_tag, get_dataset_class
from stgym.data_loader.ds_info import get_info


def load_merged_design_space(yaml_path: str) -> dict:
    """Load a design space YAML, resolving Hydra-style defaults from the same directory."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve `defaults` list (Hydra convention: load base files first, then override)
    defaults = cfg.pop("defaults", [])
    merged = {}
    for item in defaults:
        if item == "_self_":
            # _self_ means "apply this file's values here" — handled by the final merge
            continue
        if isinstance(item, str):
            base_name = item
        elif isinstance(item, dict):
            # e.g. {override: value} — skip non-file entries
            if len(item) == 1 and list(item.keys())[0] not in ("override", "override/"):
                base_name = list(item.values())[0]
            else:
                continue
        else:
            continue

        base_path = yaml_path.parent / f"{base_name}.yaml"
        if base_path.exists():
            with open(base_path) as f:
                base_cfg = yaml.safe_load(f) or {}
            _deep_merge(merged, base_cfg)

    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _as_list(val) -> list:
    """Wrap scalar in list if not already a list."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def extract_combos(cfg: dict) -> list[tuple[str, str, int | None, float | None]]:
    """Extract all unique (dataset_name, graph_const, knn_k, radius_ratio) combos."""
    dataset_names = _as_list(cfg.get("task", {}).get("dataset_name"))
    dl = cfg.get("data_loader", {})
    graph_consts = _as_list(dl.get("graph_const", "knn"))
    knn_ks = _as_list(dl.get("knn_k", 10))
    radius_ratios = _as_list(dl.get("radius_ratio", 0.1))

    combos = []
    for ds_name in dataset_names:
        for graph_const in graph_consts:
            if graph_const == "knn":
                for knn_k in knn_ks:
                    combos.append((ds_name, graph_const, knn_k, None))
            else:  # radius
                for radius_ratio in radius_ratios:
                    combos.append((ds_name, graph_const, None, radius_ratio))
    return combos


class DeviceWrappedGraphTransform:
    """Wrap a graph transform to move data to a device, apply, then return to CPU."""

    def __init__(self, graph_transform, device: str):
        self.graph_transform = graph_transform
        self.device = device

    def __call__(self, data):
        if self.device != "cpu":
            data = data.to(self.device)
        data = self.graph_transform(data)
        if self.device != "cpu":
            data = data.cpu()
        return data

    def __repr__(self):
        return f"DeviceWrapped({self.graph_transform}, device={self.device})"


def build_pre_transform(
    graph_const: str,
    knn_k: int | None,
    radius_ratio: float | None,
    radius: float | None,
    device: str,
):
    if graph_const == "knn":
        graph_transform = T.KNNGraph(k=knn_k)
    else:
        graph_transform = T.RadiusGraph(r=radius)

    if device != "cpu":
        graph_transform = DeviceWrappedGraphTransform(graph_transform, device)

    return T.Compose(
        [
            graph_transform,
            T.ToSparseTensor(remove_edge_index=False, layout=torch.sparse_coo),
        ]
    )


def process_combo(
    ds_name: str,
    graph_const: str,
    knn_k: int | None,
    radius_ratio: float | None,
    device: str,
) -> str:
    """Process one combo. Returns 'SKIP', 'OK', or raises on error."""
    tag = build_graph_construction_tag(graph_const, knn_k, radius_ratio)
    processed_file = Path(f"./data/{ds_name}/processed/data_{tag}.pt")

    if processed_file.exists():
        return "SKIP"

    # Compute radius if needed
    radius = None
    if graph_const == "radius":
        ds_info = get_info(ds_name)
        radius = radius_ratio * (ds_info["max_span"] - ds_info["min_span"])

    pre_transform = build_pre_transform(
        graph_const, knn_k, radius_ratio, radius, device
    )
    ds_cls = get_dataset_class(ds_name)
    # Instantiating the dataset triggers process() if the tagged .pt is missing
    ds_cls(
        root=f"./data/{ds_name}",
        pre_transform=pre_transform,
        graph_construction_tag=tag,
    )
    return "OK"


def main():
    parser = argparse.ArgumentParser(description="Pre-generate processed dataset files")
    parser.add_argument(
        "--design-space",
        required=True,
        help="Path to design space YAML (e.g. conf/design_space/graph_clf.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for graph construction (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = load_merged_design_space(args.design_space)
    combos = extract_combos(cfg)

    if not combos:
        print("No combos found — check your design space YAML.")
        return 1

    print(f"Found {len(combos)} combo(s) to process.\n")
    errors = []
    for ds_name, graph_const, knn_k, radius_ratio in combos:
        tag = build_graph_construction_tag(graph_const, knn_k, radius_ratio)
        label = f"{ds_name} / {tag}"
        try:
            result = process_combo(ds_name, graph_const, knn_k, radius_ratio, device)
            print(f"  [{result:4s}] {label}")
        except Exception as e:
            print(f"  [ERR ] {label}: {e}")
            errors.append((label, e))

    print()
    if errors:
        print(f"{len(errors)} error(s) occurred.")
        return 1
    else:
        print("All combos processed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
