#!/bin/bash

time python run_rct.py \
       +exp=bn \
       design_space=node_clf \
       resource=gpu-2 \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-bn-sep-16-2025
