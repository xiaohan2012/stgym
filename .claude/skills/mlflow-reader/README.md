# MLflow Reader Claude Code Skill

A Claude Code skill for reading and analyzing MLflow tracking data. This skill provides comprehensive capabilities for querying experiments, runs, metrics, and artifacts from MLflow servers.

## Features

- **Experiment Operations**: List, search, and get detailed experiment information
- **Run Analysis**: Query runs with filters, compare multiple runs, export data
- **Status-Based Filtering**: Filter runs by status (FAILED, FINISHED, RUNNING, etc.)
- **Error Analysis**: Extract training error information from failed runs
- **URL Parsing**: Extract tracking server and run information from MLflow URLs
- **Metrics & Artifacts**: Retrieve metric history, list and read local artifact files
- **Data Export**: Format data as JSON, CSV, or human-readable summaries
- **Integration**: Leverages STGym project's MLflow configuration patterns

## Files

- `SKILL.md`: Claude Code skill definition with comprehensive instructions
- `mlflow_client_utils.py`: Reusable MLflow client utilities and helper functions
- `test_skill.py`: Test script to validate skill functionality
- `README.md`: This documentation file

## Usage

### Automatic Activation
The skill automatically activates when:
- User provides MLflow URLs
- User asks about experiment analysis
- User requests run comparisons
- User mentions metrics or artifacts

### Manual Invocation
Use the slash command:
```
/mlflow-reader --tracking-uri http://localhost:5000 --experiment my-experiment
```

### URL Analysis
Provide MLflow URLs like:
```
http://127.0.0.1:5001/#/experiments/885309957287214681/runs/32a2aa5d98444da8b511a80ccd68f2e3
```

The skill will automatically parse and extract data from the run.

## Configuration

### Default Settings
- **Tracking URI**: `http://127.0.0.1:5000` (STGym project default)
- **Max Results**: 1000 runs per query
- **Timeout**: 30 seconds for API operations

### STGym Integration
The skill integrates with:
- `stgym.config_schema.MLFlowConfig` for configuration
- Existing URL parsing patterns from `get_mlflow_params.py`
- Project-specific metric types and experiment metadata

## Testing

Run the test suite:
```bash
cd .claude/skills/mlflow-reader
python test_skill.py
```

Tests cover:
- URL parsing functionality
- MLflow client creation
- Server connectivity (requires running MLflow server)
- Data formatting and output

## Error Handling

The skill gracefully handles:
- Connection failures with clear error messages
- Missing experiments or runs
- Invalid URL formats
- Server timeouts and performance issues

## Dependencies

- `mlflow`: MLflow tracking client
- `pandas`: Data manipulation and export
- Standard Python libraries (json, re, pathlib, urllib)

All dependencies are included in the STGym project requirements.

## Examples

### Query Failed Runs with Error Information
```python
from mlflow_client_utils import create_mlflow_reader, get_failed_runs_with_errors

reader = create_mlflow_reader("http://127.0.0.1:5000")
experiment = reader.get_experiment("my-experiment")
failed_runs = get_failed_runs_with_errors(reader, [experiment.experiment_id])

for run_data in failed_runs:
    print(f"Failed Run: {run_data['run_id']}")
    if run_data['training_error']:
        print(f"Error: {run_data['training_error'][:200]}...")
```

### Filter Runs by Status
```python
reader = create_mlflow_reader()
experiment = reader.get_experiment("my-experiment")

# Get all failed runs
failed_runs = reader.get_failed_runs([experiment.experiment_id])

# Get all successful runs
successful_runs = reader.get_successful_runs([experiment.experiment_id])

# Get runs with specific status
running_runs = reader.get_runs_by_status([experiment.experiment_id], "RUNNING")
```

### Extract Error Information from Specific Run
```python
run_id = "abc123"
error_content = reader.get_training_error(run_id)
if error_content:
    print(f"Training error for run {run_id}:")
    print(error_content)
```

### Compare Multiple Runs
```python
comparison = reader.compare_runs(["run1", "run2", "run3"])
print(comparison)
```

### Export Experiment Data
```python
reader.export_experiment_data("my-experiment", format="csv", output_path=Path("results.csv"))
```

### Analyze Run from URL
```python
url = "http://127.0.0.1:5000/#/experiments/1/runs/abc123"
run_data = reader.get_run_from_url(url)
summary = format_run_summary(run_data)
print(summary)
```

This skill makes MLflow data analysis seamless within Claude Code, supporting both quick queries and detailed experiment investigation.
