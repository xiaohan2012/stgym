# STGym

STGym is a platform for designing and evaluating Graph Neural Networks for spatial transcriptoics tasks.

Built with PyTorch Lightning and PyTorch Geometric, it provides scalable tools for graph-based classification and node classification tasks.

## Installation

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stgym
```

2. Activate the environment:
```bash
pyenv activate stgym
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running Single Experiments

Launch an experiment from a YAML configuration:
```bash
python run_experiment_by_yaml.py conf/exp/bn.yaml --mlflow-uri http://127.0.0.1:5000
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

Quick CPU validation:
```bash
bash scripts/test_on_cpu.sh
```
