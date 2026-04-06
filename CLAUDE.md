# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STGym is a machine learning framework for spatial transcriptomics graph neural networks. It supports graph-classification and node-classification tasks on spatial genomics datasets using PyTorch Lightning, MLflow, and Ray for distributed training.

## Core Architecture

### Execution Paths

There are two entry points:

1. **Single experiment** (`run_experiment_by_yaml.py`): Loads a flat `ExperimentConfig` YAML and calls `stgym/rct/run.py:run_exp()`. Used for ad-hoc runs and debugging.

2. **Distributed RCT sweep** (`run_rct.py`): Uses Hydra + Ray. Reads a `DesignSpace` YAML that defines parameter search ranges, samples `sample_size` configurations via `stgym/design_space/design_gen.py:generate_experiment()`, and dispatches each as a Ray remote task calling `run_exp()`.

### Config System

Two distinct config schemas live in `stgym/config_schema.py`:

- `ExperimentConfig` — the flat per-run config used by `run_experiment_by_yaml.py` and as the output of design space sampling. Fields: `task`, `data_loader`, `model`, `train`.
- `DesignSpace` (`stgym/design_space/schema.py`) — each field can be a scalar or list; the generator randomly samples one combination per trial. Fields with matching lists can be co-sampled using `zip_`.

Hydra config (`conf/config.yaml`) defaults to `design_space: node_clf`, `resource: cpu-4`, `mlflow: local`. Override with `+exp=<name>` to add experiment-specific params (from `conf/exp/`).

### Data Pipeline

Each dataset class extends `AbstractDataset` (`stgym/data_loader/base.py`), which extends PyG `InMemoryDataset`. The pipeline:

1. `process_data()` reads raw files (CSV, Parquet, HDF5) from `data/<dataset-name>/raw/`
2. Graph construction (KNN or radius) and sparse tensor conversion run as `pre_transform`
3. Processed graphs are cached to `data/<dataset-name>/processed/data_<tag>.pt` where `<tag>` encodes graph construction params (e.g., `knn10`, `radius0.1`)

**Test datasets** use the `-test` suffix (e.g., `brca-test`), are stored in `tests/data/`, and apply a `num_nodes <= 500` pre-filter to keep graphs small.

**K-fold CV** is automatically used for datasets with few samples (defined in `dataset_eval_mode` in `config_schema.py`): `human-intestine`, `spatial-vdj`, `human-pancreas`, `colorectal-cancer`, `gastric-bladder-cancer`, `cellcontrast-breast`. For k-fold, each fold runs as a separate MLflow run tagged with `fold`.

### Model Architecture

`STGymModule` (PyTorch Lightning module) wraps either `STGraphClassifier` or `STNodeClassifier`. Both use:
- `GeneralMultiLayer` (`stgym/layers.py`): stacks MP layers from `MessagePassingConfig` list
- Optional hierarchical pooling (DMoN or MinCut) after any MP layer — only valid for graph classification
- Global pooling (mean/sum/max) + post-MP MLP for graph classification
- Binary classification outputs `dim_out=1`; multi-class outputs `num_classes`

OOM handling: if a Ray worker hits CUDA OOM, it calls `os._exit(1)` to release the GPU slot immediately.

### Experiment Launching Scripts

Shell scripts in `./scripts/` follow the pattern `scripts/exp-{design_dimension}/{task_type}.sh`. Examples:
- `scripts/exp-bn/graph-clf.sh` — batch normalization sweep for graph classification
- `scripts/exp-hpooling/node-clf.sh` — hierarchical pooling sweep for node classification

Ad-hoc single-run configs (for debugging, repro) live in `conf/adhoc/`.

## Common Development Commands

### Environment

```bash
uv sync --group dev       # install all dependencies including dev tools
source .venv/bin/activate # activate venv
```

### Running Single Experiments

```bash
python run_experiment_by_yaml.py <config_path> [--mlflow-uri URI] [--experiment-name NAME] [--no-tracking]
```

### Running RCT Sweeps

```bash
python run_rct.py +exp=<experiment> design_space=<space> resource=<resource> sample_size=<n>
```

### Testing

```bash
pytest tests/ -m 'not slow'   # run fast tests (preferred)
pytest tests/test_train.py -v  # single module
bash scripts/test_on_cpu.sh    # quick CPU smoke test
bash scripts/test_on_gpu.sh    # quick GPU smoke test
```

### MLflow Tracking

Default local MLflow URI is `http://127.0.0.1:5000`. Start the server with the launch script if needed (see `scripts/`).

## Key Conventions

- Dataset names use kebab-case strings (e.g., `mouse-kidney`, `gastric-bladder-cancer`). The canonical list is in `stgym/data_loader/const.py:DatasetName` and `stgym/data_loader/ds_info.py`.
- Raw data is never downloaded automatically — each dataset has a preprocessing script in `scripts/data_preprocessing/`. See `stgym/data_loader/README.md` for per-dataset download and preprocessing instructions.
- Adding a new dataset requires: a loader class in `stgym/data_loader/`, an entry in `get_dataset_class()` (`stgym/data_loader/__init__.py`), and an entry in `ds_info.py` with `num_classes`, `task_type`, and spatial span bounds.
