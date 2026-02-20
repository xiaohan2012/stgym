# Run Experiment

Runs a single experiment from a YAML configuration file using the STGym framework.

## Usage

```
/run_exp <config_path> [--no-tracking]
```

## Parameters

- `config_path`: Path to the YAML configuration file (required)
- `--no-tracking`: Disable MLflow tracking (optional)

## Examples

```
/run_exp conf/exp/basic_gcn.yaml
/run_exp conf/exp/gat_experiment.yaml --no-tracking
```

## Implementation

```bash
python run_experiment_by_yaml.py "$@"
```
