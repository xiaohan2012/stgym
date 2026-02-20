# Run Experiment

Runs a single experiment from a YAML configuration file using the STGym framework.

## Usage

```
/run_exp <config_path> --no-tracking  # always avoid MLFlow tracking
```

## Parameters

- `config_path`: Path to the YAML configuration file (required)
- `--no-tracking`: Disable MLflow tracking (use it)


## Implementation

```bash
python run_experiment_by_yaml.py "$@" --no-tracking
```
