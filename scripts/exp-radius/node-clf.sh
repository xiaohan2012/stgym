#!/bin/bash

time python run_rct.py \
       +exp=radius \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-radius-$(date +%m-%d-%Y)
