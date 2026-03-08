#!/bin/bash

time python run_rct.py \
       +exp=postmp \
       design_space=graph_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=graph-clf-postmp-$(date +%m-%d-%Y)
