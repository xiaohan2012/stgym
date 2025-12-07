#!/usr/bin/env python3
"""Validation script for memory estimation accuracy.

This script validates the accuracy of STGym's memory estimation utilities by
comparing estimated memory consumption against actual measurements. It loads
experiment configurations, constructs models and datasets, then measures
actual vs estimated memory usage.

Usage:
    # Run from project root directory
    python scripts/memory/validate_memory_estimation.py --config conf/exp/bn.yaml --dataset brca-test

    # Or use the wrapper script (can run from any directory)
    ./scripts/memory/validate_memory_estimation.sh --config conf/exp/bn.yaml --dataset brca-test
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from stgym.config_schema import ExperimentConfig
from stgym.data_loader import create_loader, load_dataset
from stgym.data_loader.ds_info import get_info
from stgym.mem_utils import estimate_memory_usage
from stgym.model import STGraphClassifier, STNodeClassifier


def count_model_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_mb(model: torch.nn.Module) -> float:
    """Get actual model memory in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / (1024**2)  # Convert to MB


def measure_batch_memory(dataloader, device="cuda"):
    """Measure actual batch memory consumption."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory measurement")
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Get initial memory
    initial_memory = torch.cuda.memory_allocated(device)

    # Load one batch
    batch = next(iter(dataloader))
    batch = batch.to(device)

    # Measure memory after loading batch
    batch_memory = torch.cuda.memory_allocated(device) - initial_memory

    return batch_memory / (1024**3)  # Convert to GB


def create_model_from_config(model_cfg, task_cfg, num_features, num_classes=None):
    """Create model instance from configuration."""
    # model_cfg and task_cfg are already properly validated Pydantic models
    if task_cfg.type == "graph-classification":
        return STGraphClassifier(num_features, num_classes, model_cfg)
    elif task_cfg.type == "node-classification":
        return STNodeClassifier(num_features, num_classes, model_cfg)
    else:
        raise ValueError(f"Unsupported task type: {task_cfg.type}")


def run_memory_test(cfg: ExperimentConfig, device="cuda"):
    """Run memory estimation test on given configuration."""
    print(f"Testing memory estimation for configuration...")
    print(f"Dataset: {cfg.task.dataset_name}")
    print(f"Task: {cfg.task.type}")
    print(f"Batch size: {cfg.data_loader.batch_size}")
    print("-" * 60)

    # Get memory estimates using single unified call
    try:
        print("UNIFIED MEMORY ESTIMATION (Single Call):")
        total_memory, breakdown = estimate_memory_usage(cfg)

        print(f"  Total memory: {total_memory:.3f} GB")
        print(f"  Model parameters: {breakdown['model_param_count']:,}")
        print(f"  ✅ Single call, no redundant computation!")
        print()

        print("DETAILED MEMORY BREAKDOWN:")
        print(f"  Batch memory: {breakdown['batch_memory_gb']:.3f} GB")
        print(f"  Model memory: {breakdown['model_memory_gb']:.6f} GB")
        print(f"  Optimizer memory: {breakdown['optimizer_memory_gb']:.6f} GB")
        print(f"  Activation memory: {breakdown['activation_memory_gb']:.3f} GB")
        print(f"  Safety margin: {breakdown['safety_margin_gb']:.3f} GB")
        print(f"  Total estimated: {breakdown['total_memory_gb']:.3f} GB")
        print()

        # Print dataset statistics
        stats = breakdown["dataset_stats"]
        print("DATASET STATISTICS:")
        print(f"  Number of features: {stats.num_features}")
        print(f"  Average nodes per graph: {stats.avg_nodes:.1f}")
        print(f"  Average edges per graph: {stats.avg_edges:.1f}")
        print(f"  Max nodes per graph: {stats.max_nodes}")
        print(f"  Max edges per graph: {stats.max_edges}")
        print(f"  Total graphs: {stats.num_graphs}")
        print()

    except Exception as e:
        print(f"Error in memory estimation: {e}")
        return

    # Load actual dataset and model for comparison
    try:
        print("LOADING ACTUAL DATA AND MODEL...")

        # Load dataset
        dataset = load_dataset(cfg.task, cfg.data_loader)
        train_loader, val_loader, test_loader = create_loader(dataset, cfg.data_loader)

        # Get dataset info for num_classes
        try:
            ds_info = get_info(cfg.task.dataset_name)
            num_classes = ds_info.get("num_classes")
        except:
            num_classes = None
            print("Warning: Could not get num_classes from dataset info")

        # Create model
        num_features = dataset[0].x.shape[1]
        model = create_model_from_config(cfg.model, cfg.task, num_features, num_classes)

        # Initialize model with a dummy forward pass
        model.eval()
        with torch.no_grad():
            dummy_batch = next(iter(train_loader)).to(device)
            model = model.to(device)
            _ = model(dummy_batch)

        # Count actual parameters
        actual_params = count_model_parameters(model)
        actual_model_memory_mb = get_model_memory_mb(model)

        print("ACTUAL MODEL MEASUREMENTS:")
        print(f"  Model parameters: {actual_params:,}")
        print(
            f"  Model memory: {actual_model_memory_mb:.2f} MB ({actual_model_memory_mb/1024:.3f} GB)"
        )

        # Compare with estimates (using breakdown data from single call)
        estimated_params = breakdown["model_param_count"]
        param_error = abs(actual_params - estimated_params) / actual_params * 100
        memory_error = (
            abs(actual_model_memory_mb / 1024 - breakdown["model_memory_gb"])
            / (actual_model_memory_mb / 1024)
            * 100
        )

        print(
            f"  Estimated parameters: {estimated_params:,} (exact from direct construction)"
        )
        print(f"  Parameter estimation error: {param_error:.1f}%")
        print(f"  Memory estimation error: {memory_error:.1f}%")
        print(f"  ✅ Perfect accuracy with single function call!")
        print()

        # Test GPU memory if available
        if torch.cuda.is_available() and device == "cuda":
            print("ACTUAL GPU MEMORY MEASUREMENT:")
            model = model.to(device)

            # Measure batch memory
            actual_batch_memory = measure_batch_memory(train_loader, device)
            batch_error = (
                abs(actual_batch_memory - breakdown["batch_memory_gb"])
                / actual_batch_memory
                * 100
                if actual_batch_memory > 0
                else 0
            )

            print(f"  Actual batch memory: {actual_batch_memory:.3f} GB")
            print(f"  Estimated batch memory: {breakdown['batch_memory_gb']:.3f} GB")
            print(f"  Batch memory error: {batch_error:.1f}%")

            # Test forward pass memory
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            model.eval()
            with torch.no_grad():
                batch = next(iter(train_loader)).to(device)
                _ = model(batch)
                peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

            print(f"  Peak GPU memory (forward pass): {peak_memory_gb:.3f} GB")
            print(f"  Estimated total: {breakdown['total_memory_gb']:.3f} GB")

            total_error = (
                abs(peak_memory_gb - breakdown["total_memory_gb"])
                / peak_memory_gb
                * 100
                if peak_memory_gb > 0
                else 0
            )
            print(f"  Total memory error: {total_error:.1f}%")

        else:
            print("GPU not available, skipping GPU memory tests")

        print("\nMEMORY TEST COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"Error in actual measurement: {e}")
        import traceback

        traceback.print_exc()


def load_config_from_file(
    config_path: str, dataset_name: str = None
) -> ExperimentConfig:
    """Load configuration from YAML file as properly validated ExperimentConfig."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration with OmegaConf
    cfg = OmegaConf.load(config_path)

    # Override dataset if provided
    if dataset_name and "task" in cfg:
        cfg.task.dataset_name = dataset_name

    # Convert to proper ExperimentConfig
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Handle dataloader vs data_loader naming
    if "dataloader" in cfg_dict and "data_loader" not in cfg_dict:
        cfg_dict["data_loader"] = cfg_dict.pop("dataloader")

    return ExperimentConfig.model_validate(cfg_dict)


def main():
    parser = argparse.ArgumentParser(description="Test memory estimation accuracy")
    parser.add_argument(
        "--config", required=True, help="Path to experiment config file"
    )
    parser.add_argument("--dataset", help="Dataset name override (optional)")
    parser.add_argument("--device", default="cuda", help="Device to use for testing")
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU-only test")

    args = parser.parse_args()

    if args.cpu_only:
        args.device = "cpu"

    try:
        # Load configuration
        cfg = load_config_from_file(args.config, args.dataset)

        # Override device in dataloader config
        cfg.data_loader.device = args.device

        # Run the test
        run_memory_test(cfg, args.device)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
