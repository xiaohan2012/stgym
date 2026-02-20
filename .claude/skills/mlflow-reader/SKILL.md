---
name: mlflow-reader
description: Read and query MLflow servers, experiments, runs, metrics, and artifacts. Use when user asks about MLflow data, experiment analysis, run comparisons, or provides MLflow URLs.
---

# MLflow Reader Skill

This skill provides comprehensive capabilities for reading and analyzing MLflow tracking data.

## Core Operations

### Experiment Operations
When working with MLflow experiments:

1. **List Experiments**: Use `mlflow.list_experiments()` to show all experiments on a tracking server
2. **Get Experiment**: Use `mlflow.get_experiment()` or `mlflow.get_experiment_by_name()` for specific experiments
3. **Search Experiments**: Filter experiments by name patterns, creation date, or lifecycle stage

### Run Operations
For MLflow runs:

1. **List Runs**: Use `mlflow.search_runs()` to get runs from experiments
   - Support filtering by status, start_time, end_time
   - Order by metrics or parameters
   - Limit results for performance

2. **Status-Based Filtering**: Query runs by their execution status
   - `get_failed_runs()`: Get all failed runs from experiments
   - `get_successful_runs()`: Get all successfully finished runs
   - `get_runs_by_status()`: Filter by any status (FAILED, FINISHED, RUNNING, etc.)

3. **Get Run Details**: Use `mlflow.get_run(run_id)` for complete run information
   - Extract parameters, metrics, tags, and metadata
   - Get artifact locations and local file paths
   - Include duration calculations and comprehensive run info

4. **Error Analysis**: Extract error information from failed runs
   - `get_training_error()`: Read training_error.txt content from local artifacts
   - `read_artifact_file()`: Read any artifact file from local filesystem
   - Automatic error inclusion in run data for failed runs

5. **URL Parsing**: Parse MLflow URLs to extract tracking URI and run ID
   - Pattern: `{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}`
   - Extract components for direct API access

### Metrics and Parameters
For run data analysis:

1. **Parameter Extraction**: Get all run parameters with proper type conversion
2. **Metric Retrieval**: Fetch metrics with history when available
3. **Comparison**: Compare parameters and metrics across multiple runs
4. **Export**: Format data as JSON, CSV, or structured tables

### Artifact Handling
For MLflow artifacts:

1. **List Artifacts**: Show artifact structure for runs
2. **Local File Reading**: Direct access to artifact files on local filesystem
   - Read training error logs from failed runs
   - Access any artifact file by relative path
   - No download required - direct file system access
3. **Model Loading**: Load registered models and their versions

## Configuration Patterns

### Default Settings
Use these defaults based on STGym project configuration:
- **Default Tracking URI**: `http://127.0.0.1:5000` (from project patterns)
- **Timeout**: 30 seconds for API calls
- **Batch Size**: 1000 runs per query (for performance)

### Connection Setup
```python
import mlflow
from mlflow import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()
```

## Usage Patterns

### Automatic Activation
This skill activates automatically when:
- User provides MLflow URLs
- User asks about experiment analysis
- User requests run comparisons
- User mentions metrics or artifacts

### Manual Invocation
Use `/mlflow-reader` with arguments:
- `--tracking-uri`: Override default tracking server
- `--experiment`: Specify experiment name or ID
- `--run-id`: Target specific run
- `--format`: Output format (json, csv, table)

### URL Parsing
When given MLflow URLs like:
`http://127.0.0.1:5001/#/experiments/885309957287214681/runs/32a2aa5d98444da8b511a80ccd68f2e3`

Extract:
- Tracking URI: `http://127.0.0.1:5001`
- Experiment ID: `885309957287214681`
- Run ID: `32a2aa5d98444da8b511a80ccd68f2e3`

## Error Handling

### Connection Issues
- Validate tracking URI accessibility
- Provide clear error messages for connection failures
- Suggest common troubleshooting steps

### Missing Data
- Handle non-existent experiments/runs gracefully
- Show available alternatives when requested data not found
- Provide partial results when some operations fail

### Performance Considerations
- Limit large queries to prevent timeouts
- Use pagination for experiments with many runs
- Cache frequently accessed data during session

## Integration with STGym

### Configuration Schema
Leverage existing `MLFlowConfig` from `stgym.config_schema`:
- `tracking_uri`: Server URL
- `experiment_name`: Target experiment
- `tags`: Additional metadata

### Project Patterns
Follow patterns from:
- `get_mlflow_params.py`: URL parsing and data extraction
- `run_experiment_by_yaml.py`: MLflow client setup
- `stgym/train.py`: Logger integration

### Data Types
Support STGym-specific metrics:
- Training/validation losses
- Model accuracy scores
- System metrics (GPU memory, training time)
- Custom experiment metadata

## Output Formatting

### Structured Data
Format results as:
- **JSON**: For programmatic access
- **Tables**: For human-readable summaries
- **Charts**: For metric comparisons (when visualization requested)

### Metric History
When displaying metrics:
- Show latest value prominently
- Include min/max/average for training metrics
- Highlight significant changes or anomalies

### Run Summaries
For run overviews include:
- Run status and duration
- Key parameters (learning rate, batch size, etc.)
- Final metric values
- Artifact counts and sizes

This skill makes MLflow data analysis seamless within Claude Code, supporting both quick queries and detailed experiment investigation.
