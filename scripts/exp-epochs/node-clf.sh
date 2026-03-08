#!/bin/bash

time python run_rct.py \
       +exp=epochs \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-epochs-$(date +%m-%d-%Y)
