#! /bin/bash

time python run_rct.py \
       +exp=hpooling \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=5 \
       resource.num_cpus_per_trial=4\
       design_space.train.max_epoch=2 \
       mlflow.experiment_name=test-dynamic-resource-allocation-cpu-4-sleep-2
