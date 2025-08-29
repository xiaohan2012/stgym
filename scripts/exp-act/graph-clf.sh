#! /bin/bash

time python run_rct.py \
       +exp=activation \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=graph-clf-activation-aug-29-2025
