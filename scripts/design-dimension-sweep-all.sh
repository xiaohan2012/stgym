#!/bin/bash

# All-dimension sweep script - runs all design dimensions with configurable parameters
#
# Usage:
#   Default: ./scripts/design-dimension-sweep-all.sh
#   Custom: RESOURCE=gpu-4 SAMPLE_SIZE=10 ./scripts/design-dimension-sweep-all.sh

# Get all available experiments from conf/exp/*.yaml
# oom_verification is excluded: it uses model.dim_inner=512 for OOM testing only,
# not as a sweep dimension, and would conflict with mlp_dim_inner in sweep_status.py
EXPERIMENTS_GRAPH_CLF=$(ls conf/exp/*.yaml | grep -v -E '(oom_verification)' | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')
# node_clf design space has pooling: null, so hpooling and clusters dimensions don't exist there
EXPERIMENTS_NODE_CLF=$(ls conf/exp/*.yaml | grep -v -E '(hpooling|clusters|oom_verification)' | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')

# Configuration variables with defaults
RESOURCE=${RESOURCE:-gpu-6}
SAMPLE_SIZE=${SAMPLE_SIZE:-100}

# Generate datetime for experiment name
DATETIME=$(date '+%Y%m%d-%H%M%S')
EXPERIMENT_NAME="sweep-all-${DATETIME}"

echo "Running all design dimensions sweep"
echo "Resource: $RESOURCE"
echo "Sample size: $SAMPLE_SIZE"
echo "Experiments (graph_clf): $EXPERIMENTS_GRAPH_CLF"
echo "Experiments (node_clf): $EXPERIMENTS_NODE_CLF"
echo "Experiment name: $EXPERIMENT_NAME"

# graph_clf: all experiments (pooling dimensions are valid)
time python run_rct.py --multirun \
       +exp=$EXPERIMENTS_GRAPH_CLF \
       design_space=graph_clf \
       resource=$RESOURCE \
       sample_size=$SAMPLE_SIZE \
       ++mlflow.experiment_name=$EXPERIMENT_NAME

# node_clf: pooling experiments excluded (hpooling/clusters reference model.pooling which is null in node_clf)
time python run_rct.py --multirun \
       +exp=$EXPERIMENTS_NODE_CLF \
       design_space=node_clf \
       resource=$RESOURCE \
       sample_size=$SAMPLE_SIZE \
       ++mlflow.experiment_name=$EXPERIMENT_NAME
