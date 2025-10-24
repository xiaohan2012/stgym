#!/usr/bin/env python
"""
Script to launch experiments from YAML configuration files.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Launch experiment from YAML configuration"
    )
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLFlow tracking URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="yaml-experiments",
        help="MLFlow experiment name (default: yaml-experiments)",
    )
    parser.add_argument(
        "--no-tracking", action="store_true", help="Disable MLFlow tracking"
    )
    parser.add_argument("--run-name", type=str, help="MLFlow run name (optional)")
    parser.add_argument(
        "--disable-float32-matmul",
        action="store_true",
        help="Disable float32 matmul precision optimization (default: enabled for GPU)",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config_path}")

    # Load experiment configuration
    logger.info(f"Loading experiment configuration from: {args.config_path}")
    exp_cfg = load_yaml_config(args.config_path)

    # Override float32 matmul precision setting if disabled via command line
    if args.disable_float32_matmul:
        exp_cfg.train.enable_float32_matmul_precision = False

    # Override device based on CUDA availability
    if torch.cuda.is_available():
        exp_cfg.data_loader.device = "cuda:0"
        exp_cfg.train.devices = 1
        logger.info("CUDA available - using GPU (auto)")

        # Enable float32 matmul precision optimization for GPU training
        if exp_cfg.train.enable_float32_matmul_precision:
            logger.info("Float32 matmul precision optimization enabled")
        else:
            logger.info("Float32 matmul precision optimization disabled")
    else:
        exp_cfg.data_loader.device = "cpu"
        logger.info("CUDA not available - using CPU (1 device)")

    # Setup MLFlow configuration
    mlflow_cfg = MLFlowConfig(
        track=not args.no_tracking,
        tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    # Log configuration
    logger.info("Experiment configuration:")
    logger.info(OmegaConf.to_yaml(exp_cfg.model_dump()))

    # Run experiment
    logger.info("Starting experiment...")

    success = run_exp(exp_cfg, mlflow_cfg)
    if success:
        logger.info("Experiment completed successfully!")
    else:
        logger.error("Experiment failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
