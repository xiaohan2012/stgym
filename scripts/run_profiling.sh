#!/bin/bash
# Run PyTorch profiler on all 5 profiling configs.
# Traces saved to /tmp/stgym_profile/<dataset_name>.pt.trace.json
# View with: tensorboard --logdir /tmp/stgym_profile

set -e

CONFIGS=(
  conf/profiling/profile-small-knn.yaml
  conf/profiling/profile-large-knn.yaml
  conf/profiling/profile-small-radius.yaml
  conf/profiling/profile-large-radius.yaml
  conf/profiling/profile-pooling.yaml
)

mkdir -p /tmp/stgym_profile

for cfg in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "Profiling: $cfg"
  echo "========================================"
  python run_experiment_by_yaml.py "$cfg" --no-tracking --profile
  echo ""
done

echo "All profiling runs complete."
echo "Traces in /tmp/stgym_profile/"
echo "View with: tensorboard --logdir /tmp/stgym_profile"
