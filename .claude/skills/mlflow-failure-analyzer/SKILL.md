---
name: mlflow-failure-analyzer
description: Comprehensive analysis of failed MLflow experiments with automated error categorization, configuration pattern detection, and artifact retrieval from remote servers. Use this skill whenever users mention MLflow failures, experiment debugging, want to analyze failed runs, need error pattern detection, or provide MLflow URLs with failing experiments.
---

# MLflow Failure Analysis Skill

Automate the analysis of failed MLflow experiments to quickly identify error patterns, configuration issues, and provide actionable debugging insights.

## Core Workflow

When a user requests analysis of MLflow experiment failures, follow this systematic approach:

### Step 1: Input Processing & Validation

**Parse MLflow Information:**
- Extract tracking URI and experiment ID/name from URLs or user input
- Handle common URL patterns: `{tracking_uri}/#/experiments/{exp_id}`
- Validate connectivity to MLflow server
- Prompt for missing information if needed

**Server Configuration Detection:**
- Determine artifact storage location (local vs remote)
- For remote servers, identify SSH hostname and base path
- Default patterns: `cyy2:~/stgym/mlruns/` (from project experience)

### Step 2: Failed Run Discovery

**Use MLflow Reader Skill:**
Explicitly invoke the existing mlflow-reader skill located at `.claude/skills/mlflow-reader/SKILL.md` to:
- Connect to the MLflow tracking server
- Filter runs by `attribute.status = "FAILED"`
- Extract run metadata: IDs, durations, start/end times, parameters
- Categorize failures by timing:
  - **Quick failures** (<10 seconds): Likely configuration/setup errors
  - **Medium failures** (10s-5min): Training initialization issues
  - **Long failures** (>5min): Training convergence or resource problems

### Step 3: Artifact Retrieval

**Remote Artifact Fetching:**
Use the `scripts/fetch_artifacts.py` script to systematically retrieve:

**Target Artifacts:**
- `training_error.txt`: Complete error logs and stack traces
- `experiment_config.yaml`: Full training configurations and parameters

**SCP Command Patterns:**
```bash
# Test connectivity
ssh {hostname} "ls -la {base_path}/{experiment_id}/"

# Batch retrieval
scp {hostname}:{base_path}/{experiment_id}/{run_id}/artifacts/training_error.txt /tmp/analysis/errors/
scp {hostname}:{base_path}/{experiment_id}/{run_id}/artifacts/experiment_config.yaml /tmp/analysis/configs/
```

**Organization:**
- Create timestamped working directory: `/tmp/mlflow_analysis_{timestamp}/`
- Separate subdirectories for errors and configurations
- Handle missing artifacts gracefully (not all runs may have error files)

### Step 4: Error Pattern Analysis

**Automated Categorization:**
Use `scripts/categorize_errors.py` to classify errors into:

**Common Error Types:**
- **CUDA/GPU Errors**: NVML failures, memory allocation issues
- **Validation Metric Errors**: Missing val_loss, k-fold cross-validation issues
- **Training Convergence**: NaN losses, gradient explosion/vanishing
- **Infrastructure Issues**: Timeouts, network connectivity
- **Configuration Errors**: Invalid parameters, missing dependencies

**Analysis Methods:**
- Regex pattern matching for known error signatures
- Stack trace parsing for root cause identification
- Frequency analysis across all failed runs
- Correlation with failure timing and duration

### Step 5: Configuration Pattern Detection

**Parameter Analysis:**
Use `scripts/analyze_configs.py` to identify:

**Common Failure Patterns:**
- **Dataset-specific issues**: Certain datasets causing systematic failures
- **Model architecture problems**: Specific layer types, activation functions
- **Hyperparameter combinations**: Learning rates, batch sizes causing issues
- **Environmental factors**: GPU settings, device assignments
- **K-fold configuration**: Cross-validation setup problems

**Pattern Recognition:**
- Group failed runs by common configuration elements
- Identify parameter ranges associated with failures
- Compare against successful run configurations when available
- Flag unusual or extreme parameter values

### Step 6: Comprehensive Reporting

Generate a structured markdown report using this exact template:

```markdown
# MLflow Experiment Failure Analysis Report

## Executive Summary
- **Total runs analyzed**: {total_runs}
- **Failed runs**: {failed_runs} ({failure_rate}% failure rate)
- **Analysis period**: {start_date} to {end_date}
- **Primary failure cause**: {main_error_type}

## Error Type Breakdown

### 1. {Error_Type_1} ({percentage}% of failures)
- **Description**: {root_cause_description}
- **Frequency**: {count} occurrences
- **Typical duration**: {average_duration}
- **Sample error**:
  ```
  {representative_error_message}
  ```
- **Affected runs**: {run_id_list}

### 2. {Error_Type_2} ({percentage}% of failures)
[Repeat pattern for each error type]

## Configuration Pattern Analysis

### Failed Run Characteristics
- **Common datasets**: {dataset_list}
- **Model architectures**: {architecture_patterns}
- **Hyperparameter ranges**:
  - Learning rates: {lr_range}
  - Batch sizes: {batch_range}
  - Other patterns: {other_patterns}
- **Environmental factors**: {env_factors}

### Correlation Insights
- {config_pattern_1} → {associated_error_type}
- {config_pattern_2} → {associated_error_type}

## Actionable Recommendations

### Priority 1: Critical Fixes
- {specific_recommendation_1}
- {specific_recommendation_2}

### Priority 2: Important Improvements
- {improvement_1}
- {improvement_2}

### Priority 3: Optimizations
- {optimization_1}
- {optimization_2}

## Sample Configurations for Reproduction

### Representative Failed Configuration
```yaml
{sample_failed_config}
```

### Suggested Fixed Configuration
```yaml
{suggested_fix_config}
```

## Technical Details
- **Artifacts retrieved**: {error_count} error logs, {config_count} configurations
- **Analysis server**: {remote_hostname}
- **Processing time**: {analysis_duration}
- **Generated**: {timestamp}
```

## Error Handling & Edge Cases

**Connectivity Issues:**
- Test SSH connectivity before bulk operations
- Provide clear error messages for authentication failures
- Suggest troubleshooting steps for common connection problems

**Missing Artifacts:**
- Handle runs without error files gracefully
- Log retrieval failures for user awareness
- Continue analysis with available data

**Large Experiments:**
- For >100 failed runs, sample representative runs for detailed analysis
- Provide progress indicators for long-running analysis
- Implement batching for memory efficiency

## Integration Points

**MLflow Reader Dependency:**
Always use the existing mlflow-reader skill (`.claude/skills/mlflow-reader/SKILL.md`) for:
- MLflow server connectivity
- Run filtering and metadata extraction
- URL parsing and validation

**Script Coordination:**
The three analysis scripts work together:
1. `fetch_artifacts.py`: Retrieves files from remote servers
2. `categorize_errors.py`: Analyzes error patterns
3. `analyze_configs.py`: Detects configuration patterns

## Performance Considerations

- **Parallel retrieval**: Batch SCP operations for efficiency
- **Local caching**: Store retrieved artifacts temporarily for re-analysis
- **Progress tracking**: Show status for large experiment analysis
- **Memory management**: Stream process large error logs

This skill transforms manual error log analysis into automated insights, enabling rapid identification of systematic experiment failures and accelerating ML research debugging workflows.
