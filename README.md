# STGym

[![Lint](https://github.com/xiaohan2012/stgym/actions/workflows/lint.yml/badge.svg)](https://github.com/xiaohan2012/stgym/actions/workflows/lint.yml)
[![Test](https://github.com/xiaohan2012/stgym/actions/workflows/test.yml/badge.svg)](https://github.com/xiaohan2012/stgym/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--checked-ty-blue)](https://github.com/astral-sh/ty)

STGym is a platform for designing and evaluating Graph Neural Networks for spatial transcriptoics tasks.

Built with PyTorch Lightning and PyTorch Geometric, it provides scalable tools for graph-based classification and node classification tasks.

## Installation

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stgym
```

2. Install dependencies and create the virtual environment (requires Python 3.12+):
```bash
uv sync --group dev
```

3. Activate the environment:
```bash
source .venv/bin/activate
```

## Quick Start

### Running Single Experiments

Launch an experiment from a YAML configuration:
```bash
python run_experiment_by_yaml.py conf/adhoc/test.yaml --mlflow-uri http://127.0.0.1:5000
```

### Distributed Experiments

Run randomized controlled trials with distributed training:
```bash
python run_rct.py +exp=bn design_space=graph_clf resource=gpu-4 sample_size=64
```

### Testing

Run comprehensive tests:

```bash
pytest tests/ -v
```


Skip slow tests

```bash
pytest tests/ -m "not slow"
```
