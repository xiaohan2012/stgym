#! /bin/bash

# Multi-run experiment sweep script
#
# Usage:
#   Debug mode (2 experiments): ./scripts/test-design-dimension-sweep.sh
#   Full mode (13 experiments): MODE=full ./scripts/test-design-dimension-sweep.sh
#
# Debug mode: 2 experiments × 2 design spaces = 4 total runs
# Full mode: all experiments × 2 design spaces

# Configuration variables
EXPERIMENTS_FULL=$(ls conf/exp/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')
EXPERIMENTS_DEBUG="hpooling,bn"
DESIGN_SPACES="graph_clf,node_clf"

# Set mode (default to debug, can be overridden with MODE environment variable)
MODE=${MODE:-debug}

echo "Mode: $MODE"
# Select experiment list based on mode
if [ "$MODE" = "full" ]; then
    EXPERIMENTS=$EXPERIMENTS_FULL
    echo "Running in full mode: $EXPERIMENTS with $DESIGN_SPACES"
else
    EXPERIMENTS=$EXPERIMENTS_DEBUG
    echo "Running in debug mode: $EXPERIMENTS with $DESIGN_SPACES"
fi

time python run_rct.py --multirun \
       +exp=$EXPERIMENTS \
       design_space=$DESIGN_SPACES \
       resource=cpu-4 \
       sample_size=1 \
       design_space.train.max_epoch=1 \
       ++mlflow.experiment_name=test
