#!/bin/bash

time python run_rct.py \
       +exp=batch_size \
       design_space=graph_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=graph-clf-batch_size-$(date +%m-%d-%Y)
