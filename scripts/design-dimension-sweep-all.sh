#!/bin/bash

# All-dimension sweep script - runs all design dimensions with configurable parameters
#
# Usage:
#   Default: ./scripts/design-dimension-sweep-all.sh
#   Custom: RESOURCE=gpu-4 SAMPLE_SIZE=10 ./scripts/design-dimension-sweep-all.sh

# Get all available experiments from conf/exp/*.yaml
EXPERIMENTS_ALL=$(ls conf/exp/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')
DESIGN_SPACES="graph_clf,node_clf"

# Configuration variables with defaults
RESOURCE=${RESOURCE:-gpu-6}
SAMPLE_SIZE=${SAMPLE_SIZE:-500}

# Generate datetime for experiment name
DATETIME=$(date '+%Y%m%d-%H%M%S')
EXPERIMENT_NAME="sweep-all-${DATETIME}"

echo "Running all design dimensions sweep"
echo "Resource: $RESOURCE"
echo "Sample size: $SAMPLE_SIZE"
echo "Experiments: $EXPERIMENTS_ALL"
echo "Design spaces: $DESIGN_SPACES"
echo "Experiment name: $EXPERIMENT_NAME"

time python run_rct.py --multirun \
       +exp=$EXPERIMENTS_ALL \
       design_space=$DESIGN_SPACES \
       resource=$RESOURCE \
       sample_size=$SAMPLE_SIZE \
       ++mlflow.experiment_name=$EXPERIMENT_NAME
