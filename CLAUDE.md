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
- `DesignSpace` (`stgym/design_space/schema.py`) — each field can be a scalar or list; the generator randomly samples one combination per trial.

Hydra config (`conf/config.yaml`) defaults to `design_space: node_clf`, `resource: cpu-4`, `mlflow: local`. Override with `+exp=<name>` to add experiment-specific params (from `conf/exp/`).

### Data Pipeline

Each dataset class extends `AbstractDataset` (`stgym/data_loader/base.py`), which extends PyG `InMemoryDataset`. The pipeline:

1. `process_data()` reads raw files (CSV, Parquet, HDF5) from `data/<dataset-name>/raw/`
2. Graph construction (KNN or radius) and sparse tensor conversion run as `pre_transform`
3. Processed graphs are cached to `data/<dataset-name>/processed/data_<tag>.pt` where `<tag>` encodes graph construction params (e.g., `knn10`, `radius0.1`)

**Test datasets** contain graphs with moderate number of nodes, use the `-test` suffix (e.g., `brca-test`), and are stored in `tests/data/`.

**K-fold CV** is automatically used for datasets with few samples (defined in `dataset_eval_mode` in `config_schema.py`). For k-fold, each fold runs as a separate MLflow run tagged with `fold`.

### Model Architecture

`STGymModule` (PyTorch Lightning module) wraps either `STGraphClassifier` or `STNodeClassifier`. Both use:
- `GeneralMultiLayer` (`stgym/layers.py`): stacks MP layers from `MessagePassingConfig` list
- Optional hierarchical pooling (DMoN or MinCut) after any MP layer — only valid for graph classification
- Global pooling (mean/sum/max) + post-MP MLP for graph classification
- Binary classification outputs `dim_out=1`; multi-class outputs `num_classes`

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

Run a single design dimension:
```bash
python run_rct.py +exp=<experiment> design_space=<space> resource=<resource> sample_size=<n>
```

Run all design dimensions across both task types (the typical full sweep):
```bash
# defaults: RESOURCE=gpu-6, SAMPLE_SIZE=100
RESOURCE=gpu-4 SAMPLE_SIZE=50 ./scripts/design-dimension-sweep-all.sh
```
This sweeps every `conf/exp/*.yaml` dimension for `graph_clf` and `node_clf` (excluding pooling dimensions for node_clf), grouping all runs under a single timestamped MLflow experiment.

Test the sweep setup with a small smoke test:
```bash
bash scripts/test-design-dimension-sweep.sh
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

## Claude Code Skills

Several project-specific skills are available via `/skill-name`:

- **`/mlflow-reader`** — Query MLflow servers, list/filter experiments and runs, extract metrics/parameters, read artifact files. Activates automatically when given MLflow URLs or asked about experiment results.
- **`/mlflow-failure-analyzer`** — Analyze failed MLflow runs: retrieves `training_error.txt` and `experiment_config.yaml` artifacts (including via SCP from cyy2), categorizes errors, and produces a structured debugging report.
- **`/run-on-cyy2`** — Run commands, sync code, and manage long-running jobs on the `cyy2` GPU server. Handles screen session management, `git pull` + launch workflows for single experiments and sweeps, and fetching MLflow artifacts from the remote `mlruns/` store.

## Key Conventions

- Dataset names use kebab-case strings (e.g., `mouse-kidney`). The canonical list is in `stgym/data_loader/const.py:DatasetName` and `stgym/data_loader/ds_info.py`.
- Raw data is never downloaded automatically — one dataset may have a preprocessing script in `scripts/data_preprocessing/`. See `stgym/data_loader/README.md` for per-dataset download and preprocessing instructions.
- Adding a new dataset requires: a loader class in `stgym/data_loader/`, an entry in `get_dataset_class()` (`stgym/data_loader/__init__.py`), and an entry in `ds_info.py` with `num_classes`, `task_type`, and spatial span bounds.

## Coding Style

### Unit Tests

Follow the guidelines in `.claude/skills/write-test/SKILL.md` (invoke via `/write-test`). Key principles:
- Group related tests under one class; use `pytest.parametrize` for variant cases
- Share test data as class properties and `@mock.patch` at the class level
- Share fixtures via `conftest.py`; avoid duplicating mock setup across test methods
- Mock external dependencies (MLflow, network); don't mock pure calculations

### Code Reuse & Cleanness

- Extract shared logic into helpers when used 3+ times
- No dead code or commented-out blocks
- Functions do one thing; keep them under ~50 lines

### Functional Style

- Prefer `pydash` for collection transforms (map, filter, group_by, etc.)
- Prefer pure functions and immutable data where practical
- Use list/dict comprehensions over manual loops for simple transforms

### Preferred Libraries

- `logzero` for logging (not `print()` or stdlib `logging`)
- `pydash` for collection operations
- `pydantic` for config validation

### Linting

`pre-commit` is configured to run before each commit, invoking `ruff` (linting + formatting) and `ty` (type checking). Both are configured in `pyproject.toml`.

### Other

- Type hints on all public function signatures
- `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants
- Fail fast with clear error messages; no silent `except: pass`

## Labeling Scheme

All agents and contributors follow this labeling scheme for GitHub issues and PRs.

### Status (lifecycle, mutually exclusive, prefixed `status/`)

- `status/new` — just created, not yet triaged
- `status/triaged` — PM reviewed, priority/type assigned
- `status/ready` — all blockers resolved, can be picked up
- `status/in-progress` — developer working on it
- `status/needs-review` — PR opened, awaiting reviewer
- `status/done` — merged and closed

### Type (mutually exclusive)

- `bug` — something broken
- `enhancement` — new feature
- `optimization` — performance/efficiency
- `refactor` — code cleanup, no behavior change
- `docs` — documentation
- `infra` — CI/CD, tooling, dev environment

### Priority

- `P0` — critical, blocks everything
- `P1` — high, do this sprint
- `P2` — normal, planned work
- `P3` — low, nice to have

## Agent Communication and Sync

The autonomous dev team (PM, Developer, Reviewer — see `.claude/agents/`) coordinates through a small, explicit set of channels. Keep these conventions in sync across all agents.

### GitHub is the primary bus

- Issues and PRs are the source of truth for work state.
- Labels drive the workflow: PM sets `status/ready` → Developer picks up and sets `status/in-progress` → opens PR and sets `status/needs-review` → Reviewer picks up → merge sets `status/done`.
- Progress updates, questions, handoffs, and review feedback all live as **issue/PR comments**, not in side channels.

### Orchestration: human-triggered via PM (v1)

- A human invokes the PM agent manually to kick off a cycle (triage, prioritization, picking the next ready issue).
- PM triages → Developer picks up ready issues → Reviewer picks up PRs marked `status/needs-review`.
- **No agent-to-agent direct calls in v1.** Agents never invoke each other; they communicate through GitHub state and `ACTIVITY.log`.

### `ACTIVITY.log`

A concise, local activity history that sits side-by-side with GitHub. All agents append to it when starting/finishing a task.

- Location: repo root, **gitignored** (not version controlled).
- Format: `[timestamp] [agent] [issue] [action] [summary]` — timestamp is UTC ISO-8601.
- Use `[N/A]` in the issue field for cross-cutting entries not tied to a single issue (e.g. status reports, environment setup).
- **Shared responsibility** — each agent writes its own entries:
  - **PM** — triage actions, e.g. `[2026-04-10T12:00:00Z] [PM] [#123] [triaged] bug P1 — stale mouse-kidney cache`
  - **Developer** — start/finish + PR events, e.g. `[...] [Developer] [#123] [started] investigate stale cache` / `[...] [Developer] [#123] [pr-opened] https://.../pull/456`
  - **Reviewer** — review outcomes, e.g. `[...] [Reviewer] [PR #456] [approved] LGTM` / `[...] [Reviewer] [PR #456] [changes-requested] 2 blockers`

Each agent's prompt already includes the append-on-start/finish rule; this section is the canonical spec they inherit from.

### What goes where

| Info | Where |
|------|-------|
| Issue status, type, priority | GitHub labels (see Labeling Scheme above) |
| Progress updates, questions, handoffs | GitHub issue comments |
| Code changes | Git branches + PRs |
| Review feedback | GitHub PR review comments |
| Agent activity log | `ACTIVITY.log` (gitignored, repo root) |
| Project conventions | `CLAUDE.md` (this file) |
