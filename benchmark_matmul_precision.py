#!/usr/bin/env python
"""
Benchmark script to test PyTorch float32 matmul precision performance.
Compares training times with and without float32 matmul precision enabled.
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from logzero import logger
from omegaconf import OmegaConf

from stgym.config_schema import ExperimentConfig, MLFlowConfig
from stgym.rct.run import run_exp


def load_yaml_config(yaml_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)


def benchmark_matmul_precision(
    exp_cfg: ExperimentConfig,
    mlflow_cfg: MLFlowConfig,
    enable_matmul_precision: bool,
    warmup_runs: int = 1,
) -> dict:
    """Run benchmark with specified matmul precision setting."""

    # Set matmul precision
    if enable_matmul_precision:
        torch.set_float32_matmul_precision("medium")
        precision_setting = "medium"
        logger.info("Float32 matmul precision set to 'medium'")
    else:
        torch.set_float32_matmul_precision("highest")
        precision_setting = "highest"
        logger.info("Float32 matmul precision set to 'highest' (default)")

    # Warmup runs (not timed)
    if warmup_runs > 0:
        logger.info(f"Performing {warmup_runs} warmup run(s)...")
        for i in range(warmup_runs):
            logger.info(f"Warmup run {i+1}/{warmup_runs}")
            run_exp(exp_cfg, mlflow_cfg)

    # Actual timed run
    logger.info("Starting timed benchmark run...")
    start_time = time.perf_counter()

    success = run_exp(exp_cfg, mlflow_cfg)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    results = {
        "success": success,
        "total_time": total_time,
        "precision_setting": precision_setting,
        "enabled": enable_matmul_precision,
    }

    logger.info(f"Benchmark completed in {total_time:.2f} seconds")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch float32 matmul precision performance"
    )
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--enable-float32-matmul",
        action="store_true",
        help="Enable float32 matmul precision optimization (set to 'medium')",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLFlow tracking URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="matmul-precision-benchmark",
        help="MLFlow experiment name (default: matmul-precision-benchmark)",
    )
    parser.add_argument(
        "--no-tracking", action="store_true", help="Disable MLFlow tracking"
    )
    parser.add_argument("--run-name", type=str, help="MLFlow run name (optional)")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before timed run (default: 1)",
    )
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Run benchmark with both precision settings and compare",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config_path}")

    # Load experiment configuration
    logger.info(f"Loading experiment configuration from: {args.config_path}")
    exp_cfg = load_yaml_config(args.config_path)

    # Override device based on CUDA availability
    if torch.cuda.is_available():
        exp_cfg.data_loader.device = "cuda:0"
        exp_cfg.train.devices = 1
        logger.info("CUDA available - using GPU")
    else:
        exp_cfg.data_loader.device = "cpu"
        logger.info("CUDA not available - using CPU")

    # Log configuration
    logger.info("Experiment configuration:")
    logger.info(OmegaConf.to_yaml(exp_cfg.model_dump()))

    if args.compare_both:
        # Run benchmarks with both settings
        results = {}

        for enable_precision in [False, True]:
            setting_name = "optimized" if enable_precision else "default"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark with {setting_name} matmul precision")
            logger.info(f"{'='*60}")

            # Create separate MLFlow runs for each setting
            run_name_suffix = f"-{setting_name}"
            run_name = (
                f"{args.run_name}{run_name_suffix}"
                if args.run_name
                else f"matmul-benchmark{run_name_suffix}"
            )

            mlflow_cfg = MLFlowConfig(
                track=not args.no_tracking,
                tracking_uri=args.mlflow_uri,
                experiment_name=args.experiment_name,
                run_name=run_name,
            )

            result = benchmark_matmul_precision(
                exp_cfg, mlflow_cfg, enable_precision, args.warmup_runs
            )
            results[setting_name] = result

        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK COMPARISON RESULTS")
        logger.info(f"{'='*60}")

        default_time = results["default"]["total_time"]
        optimized_time = results["optimized"]["total_time"]

        logger.info(f"Default precision (highest):  {default_time:.2f} seconds")
        logger.info(f"Optimized precision (medium): {optimized_time:.2f} seconds")

        if default_time > 0:
            speedup = default_time / optimized_time
            improvement_pct = ((default_time - optimized_time) / default_time) * 100

            if speedup > 1:
                logger.info(f"Speedup: {speedup:.2f}x ({improvement_pct:.1f}% faster)")
            else:
                logger.info(
                    f"Slowdown: {1/speedup:.2f}x ({-improvement_pct:.1f}% slower)"
                )

        logger.info(f"{'='*60}")

    else:
        # Run single benchmark
        setting_name = "optimized" if args.enable_float32_matmul else "default"
        run_name = (
            f"{args.run_name}" if args.run_name else f"matmul-benchmark-{setting_name}"
        )

        mlflow_cfg = MLFlowConfig(
            track=not args.no_tracking,
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment_name,
            run_name=run_name,
        )

        result = benchmark_matmul_precision(
            exp_cfg, mlflow_cfg, args.enable_float32_matmul, args.warmup_runs
        )

        if result["success"]:
            logger.info(
                f"Benchmark completed successfully in {result['total_time']:.2f} seconds"
            )
        else:
            logger.error("Benchmark failed!")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
