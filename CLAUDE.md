# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STGym is a machine learning framework for spatial transcriptomics graph neural networks. It provides tools for graph-based classification and node classification tasks on spatial genomics datasets using PyTorch Lightning, MLflow, and Ray for distributed training.

## Core Architecture

- **stgym/**: Main package containing the ML pipeline
  - `train.py`: Core training logic using PyTorch Lightning
  - `model.py`: Graph neural network model definitions
  - `tl_model.py`: PyTorch Lightning module wrapper
  - `data_loader/`: Dataset loaders for various spatial transcriptomics datasets
  - `design_space/`: Design space definition and experiment generation
  - `rct/`: Randomized controlled trial experiment management
  - `pooling/`: Graph pooling layers (DMoN, MinCut)
  - `config_schema.py`: Pydantic configuration schemas

- **Configuration System**: Uses Hydra for configuration management with YAML files in `conf/`
  - `conf/config.yaml`: Main configuration defaults
  - `conf/exp/`: Experiment-specific configurations
  - `conf/design_space/`: Design space definitions for different tasks
  - `conf/resource/`: Resource allocation configurations

### Experiment Launching Scripts

The `./scripts` directory contains shell scripts for launching experiments focused on specific design dimensions and tasks:

Common pattern is `scripts/exp-{design_dimension}/{task_type}.sh`

Examples:

- `scripts/exp-bn/graph-clf.sh`: batch normalization experiment for graph classification task
- `scripts/exp-hpooling/node-clf.sh`: hierarchical pooling experiment for node classification task


## Common Development Commands


### Running Single Experiments

Run single experiment from YAML config:
```bash
python run_experiment_by_yaml.py <config_path> [--mlflow-uri URI] [--experiment-name NAME]
```

Run distributed randomized control test (RCT):
```bash
python run_rct.py +exp=<experiment> design_space=<space> resource=<resource> sample_size=<n>
```

### Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_train.py -v
python -m pytest tests/test_config_schema.py -v
```

Quick CPU test (limited epochs):
```bash
bash scripts/test_on_cpu.sh
```

Quick GPU test (limited epochs):
```bash
bash scripts/test_on_gpu.sh
```


### MLflow Tracking

The project uses MLflow for experiment tracking. MLflow runs are automatically configured when using the experiment scripts. Default local MLflow URI is `http://127.0.0.1:5000`.

## Key Configuration Patterns

- Experiment configs inherit from design spaces and resource configurations
- Use `+exp=<name>` to add experiment-specific overrides
- Resource configs define CPU/GPU allocation for distributed training
- Design spaces define parameter search spaces for hyperparameter optimization

## Dependencies

- PyTorch and PyTorch Geometric for graph neural networks
- PyTorch Lightning for training infrastructure
- MLflow for experiment tracking
- Ray for distributed computing
- Hydra for configuration management
- Pydantic for configuration validation

The project supports both CPU and GPU training with automatic device detection.
