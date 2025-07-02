#! /bin/bash

python run_rct.py \
       +exp=bn \
       design_space=clustering \
       resource=cpu-4 \
       sample_size=1 \
       design_space.train.max_epoch=5 \
       mlflow.experiment_name=test
