#!/bin/bash

time python run_rct.py \
       +exp=hpooling \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-hpooling-$(date +%m-%d-%Y)
