#!/bin/bash
# Start MLflow server with SQLite backend (instead of file-based backend)
# See: https://github.com/xiaohan2012/stgym/issues/105

set -euo pipefail

MLFLOW_DIR="${1:-.}"

mlflow server \
  --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
  --default-artifact-root "${MLFLOW_DIR}/mlruns" \
  --host 0.0.0.0 \
  --port 5000
