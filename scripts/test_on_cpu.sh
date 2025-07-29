#! /bin/bash

python run_rct.py \
       +exp=bn \
       design_space=graph_clf \
       resource=cpu-4 \
       sample_size=1 \
       design_space.train.max_epoch=10 \
       mlflow.experiment_name=test
