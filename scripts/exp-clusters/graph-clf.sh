#!/bin/bash

time python run_rct.py \
       +exp=clusters \
       design_space=graph_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=graph-clf-clusters-$(date +%m-%d-%Y)
