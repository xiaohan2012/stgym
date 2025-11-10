#! /bin/bash

# Check if num_cpus_per_trial argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_cpus_per_trial>"
    echo "Example: $0 4"
    exit 1
fi

NUM_CPUS_PER_TRIAL=$1

time python run_rct.py \
       +exp=hpooling \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=5 \
       resource.num_cpus_per_trial=$NUM_CPUS_PER_TRIAL \
       design_space.train.max_epoch=2 \
       mlflow.experiment_name=test-dynamic-resource-allocation-cpu-$NUM_CPUS_PER_TRIAL-nov-10
