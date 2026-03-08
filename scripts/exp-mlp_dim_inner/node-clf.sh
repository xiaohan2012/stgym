#!/bin/bash

time python run_rct.py \
       +exp=mlp_dim_inner \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       mlflow.experiment_name=node-clf-mlp_dim_inner-$(date +%m-%d-%Y)
