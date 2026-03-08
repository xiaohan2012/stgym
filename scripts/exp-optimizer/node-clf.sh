#!/bin/bash

time python run_rct.py \
       +exp=optimizer \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-optimizer-$(date +%m-%d-%Y)
