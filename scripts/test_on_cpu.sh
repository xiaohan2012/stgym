#! /bin/bash

time python run_rct.py \
       +exp=bn \
       design_space=graph_clf \
       resource=cpu-4 \
       sample_size=2 \
       design_space.train.max_epoch=2 \
       mlflow.experiment_name=test
