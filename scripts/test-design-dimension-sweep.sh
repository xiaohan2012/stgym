#! /bin/bash

# Multi-run experiment sweep script
#
# Usage:
#   Debug mode (2 experiments): ./scripts/test-design-dimension-sweep.sh
#   Full mode (all experiments): MODE=full ./scripts/test-design-dimension-sweep.sh
#   With GPU: RESOURCE=gpu-6 ./scripts/test-design-dimension-sweep.sh
#   Full mode + GPU: MODE=full RESOURCE=gpu-6 ./scripts/test-design-dimension-sweep.sh
#
# Debug mode: 2 experiments × 2 design spaces = 4 total runs
# Full mode: all experiments × 2 design spaces

# Configuration variables
# remove 'epochs' experiment for efficiency reasons
EXPERIMENTS_FULL_GRAPH_CLF=$(ls conf/exp/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | grep -v '^epochs$' | tr '\n' ',' | sed 's/,$//')
# node_clf design space has pooling: null, so hpooling and clusters dimensions don't exist there
EXPERIMENTS_FULL_NODE_CLF=$(ls conf/exp/*.yaml | grep -v -E '(hpooling|clusters)' | xargs -n1 basename | sed 's/\.yaml$//' | grep -v '^epochs$' | tr '\n' ',' | sed 's/,$//')
EXPERIMENTS_DEBUG="hpooling,bn"

# Set mode (default to debug, can be overridden with MODE environment variable)
MODE=${MODE:-debug}
# Set resource (default to cpu-4, can be overridden with RESOURCE environment variable)
RESOURCE=${RESOURCE:-cpu-4}

echo "Mode: $MODE, Resource: $RESOURCE"

EXPERIMENT_NAME="test-sweep-$(date +%m-%d-%Y--%H:%M:%S)"

if [ "$MODE" = "full" ]; then
    echo "Running in full mode"
    echo "  graph_clf experiments: $EXPERIMENTS_FULL_GRAPH_CLF"
    echo "  node_clf experiments: $EXPERIMENTS_FULL_NODE_CLF"

    # graph_clf: all experiments (pooling dimensions are valid)
    time python run_rct.py --multirun \
           +exp=$EXPERIMENTS_FULL_GRAPH_CLF \
           design_space=graph_clf \
           resource=$RESOURCE \
           sample_size=5 \
           design_space.train.max_epoch=2 \
           ++mlflow.experiment_name=$EXPERIMENT_NAME

    # node_clf: pooling experiments excluded (hpooling/clusters reference model.pooling which is null in node_clf)
    time python run_rct.py --multirun \
           +exp=$EXPERIMENTS_FULL_NODE_CLF \
           design_space=node_clf \
           resource=$RESOURCE \
           sample_size=5 \
           design_space.train.max_epoch=2 \
           ++mlflow.experiment_name=$EXPERIMENT_NAME
else
    # Debug mode: hpooling and bn are both valid for graph_clf; only bn is valid for node_clf
    echo "Running in debug mode: $EXPERIMENTS_DEBUG with graph_clf, then bn with node_clf"

    time python run_rct.py --multirun \
           +exp=$EXPERIMENTS_DEBUG \
           design_space=graph_clf \
           resource=$RESOURCE \
           sample_size=5 \
           design_space.train.max_epoch=2 \
           ++mlflow.experiment_name=$EXPERIMENT_NAME

    time python run_rct.py --multirun \
           +exp=bn \
           design_space=node_clf \
           resource=$RESOURCE \
           sample_size=5 \
           design_space.train.max_epoch=2 \
           ++mlflow.experiment_name=$EXPERIMENT_NAME
fi
